//! Wave 27 unit tests for `VICTORY_VRAM_CBV` (`vram_cbv()`):
//! - Default `=0` produces unchanged lowering, `host_settings`, and fingerprint.
//! - `=1` lowers table, bindless heap, vector, selected, and root CBVs to DXIL
//!   `rawBufferLoad` on `ByteAddressBuffer` (`R32_TYPELESS` SRV) without AMD
//!   `raw_twins` (`raw_views() == 0`), while preserving `raw_twin` routing when
//!   `raw_views() != 0`.
//! - Covers 4-byte/16-byte and non-4-byte offsets, out-of-bounds zero reads,
//!   `USER_VIEW_BIAS` on root CBVs, untouched root/immediate constants (`b0`),
//!   `dxil.dll` validation, and D3D12 WARP execution parity.

use super::lower;
use super::shader;
use super::{HostShader, translation_fingerprint, vram_cbv, with_vram_cbv};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (offset & 0x1F_FFFF),
    ]
}

/// SMEM `s_buffer_load_dword` / `x4` (`op`: 8 = dword, 10 = dwordx4).
const fn s_buffer_load(op: u32, dst: u8, sbase: u8, soffset: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (dst as u32) << 6 | (sbase as u32 >> 1),
        (soffset as u32) << 25 | (offset & 0x1F_FFFF),
    ]
}

/// VOP1 `v_mov_b32 vdst, s<src>`.
const fn v_mov_sgpr(vdst: u8, ssrc: u8) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | ssrc as u32
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// MUBUF `buffer_load_dwordx4 v[vdst:vdst+3], vaddr, s[srsrc:srsrc+3], 0 idxen`.
const fn buffer_load_dwordx4_idxen(vdst: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x0E << 18 | 1 << 13 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
    ]
}

fn lower_cs(native: &NativeProgram, vram: bool, raw_views: u32) -> lower::Lowered {
    with_vram_cbv(vram, || {
        struct RestoreRaw(Option<u32>);
        impl Drop for RestoreRaw {
            fn drop(&mut self) {
                shader::RAW_VIEWS_HERE.set(self.0);
            }
        }
        let _restore = RestoreRaw(shader::RAW_VIEWS_HERE.replace(Some(raw_views)));
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, native).unwrap();
        lower::lower(&lower::Parts {
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap()
    })
}

fn host_shader(l: lower::Lowered) -> HostShader {
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: Vec::new(),
    }
}

#[cfg(windows)]
fn validate_dxil(dxil: &mut [u8]) {
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(dxil).unwrap();
    }
}

#[cfg(windows)]
static WARP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn vram_cbv_defaults_off_and_isolates_fingerprint_and_host_settings() {
    if std::env::var_os("VICTORY_VRAM_CBV").is_none() {
        assert!(!vram_cbv(), "VICTORY_VRAM_CBV must default to off");
    }
    let (fp_off, hs_off) = with_vram_cbv(false, || {
        assert!(!vram_cbv());
        with_vram_cbv(true, || assert!(vram_cbv()));
        assert!(!vram_cbv(), "nested with_vram_cbv must restore prior value");
        (translation_fingerprint(), super::host_settings())
    });
    let (fp_on, hs_on) = with_vram_cbv(true, || {
        assert!(vram_cbv());
        (translation_fingerprint(), super::host_settings())
    });
    assert_ne!(
        fp_off, fp_on,
        "translation_fingerprint must isolate VICTORY_VRAM_CBV=0 and =1"
    );
    assert_eq!(
        hs_off, hs_on,
        "host_settings must stay unchanged so =0 cache keys stay bit-identical"
    );
}

#[test]
#[cfg(windows)]
fn table_cbv_reads_raw_in_place_without_twins_on_nvidia_heap() {
    // s0..s3: root constants (s0=dynamic aligned off, s1=dynamic unaligned off, s2=passthrough)
    // s4: table byte offset (slot 0 = CBV, slot 1 = Raw UAV)
    let mut words: Vec<u32> = vec![s_mov_b32(5, 0x80)];
    words.extend(s_load_dwordx4(8, 4, 0x00)); // s[8:11] = CBV descriptor (slot 0)
    words.extend(s_load_dwordx4(12, 4, 0x20)); // s[12:15] = UAV descriptor (slot 1)
    words.extend(s_buffer_load(10, 16, 8, 125, 0x10)); // s[16:19] = 16B-aligned dwordx4 @ 0x10
    words.extend(s_buffer_load(8, 20, 8, 0, 0x00)); // s20 = dynamic aligned @ s0
    words.extend(s_buffer_load(8, 21, 8, 125, 0x06)); // s21 = non-4B constant @ 0x06 (-> 0x04)
    words.extend(s_buffer_load(8, 22, 8, 1, 0x00)); // s22 = non-4B dynamic @ s1 (9 -> 8)
    words.extend(s_buffer_load(8, 23, 8, 125, 0x100)); // s23 = out-of-bounds @ 0x100 (-> 0)
    // Store s16..s23 and root constant s2 into UAV slots 0..9.
    let outputs = [16u8, 17, 18, 19, 20, 21, 22, 23, 2];
    for (out, &s) in outputs.iter().enumerate() {
        words.push(v_mov_sgpr(1, s));
        words.extend(buffer_store_dword(1, 12, (4 * out) as u16));
    }
    words.push(S_ENDPGM);

    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![table(ResourceType::Cbv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    // NVIDIA 1M heap: raw_views == 0 (NO raw_twins!).
    let off = lower_cs(&native, false, 0);
    let off_repeat = lower_cs(&native, false, 0);
    assert_eq!(off.dxil, off_repeat.dxil, "=0 lowering must be deterministic");

    let on = lower_cs(&native, true, 0);
    assert_ne!(off.dxil, on.dxil, "VICTORY_VRAM_CBV=1 must emit rawBufferLoad DXIL");
    assert!(on.root_cbvs.is_empty());
    assert!(on.root_srvs.is_empty());

    let data: Vec<u32> = (0..16u32)
        .map(|i| 0x0101_0101u32.wrapping_mul(i + 3) ^ 0x4000_0000)
        .collect();
    let (s0, s1, s2) = (12u32, 9u32, 0xCAFE_BABEu32);
    let want = vec![
        data[4], data[5], data[6], data[7], // @ 0x10..0x1F
        data[3], // @ s0 = 12
        data[1], // @ 0x06 -> 0x04
        data[2], // @ s1 = 9 -> 8
        0,       // @ 0x100 (OOB)
        s2,      // root constant untouched
    ];

    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        eprintln!("no software adapter; WARP dispatch skipped");
        return;
    };
    for (vram, lowered) in [(false, off), (true, on)] {
        let mut cs = host_shader(lowered);
        validate_dxil(&mut cs.dxil);
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let views = if vram {
            // In-place slot 0 holds a raw ByteAddressBuffer SRV (no twins!).
            vec![
                super::host::View::Buffer {
                    data: &data,
                    stride: super::host::Stride::Raw,
                },
                super::host::View::RawUav(16),
            ]
        } else {
            vec![super::host::View::Cbv(&data), super::host::View::RawUav(16)]
        };
        let out = host
            .dispatch(
                &root,
                &pso,
                &super::host::Dispatch {
                    roots: vec![(0, super::host::Root::Constants(vec![s0, s1, s2, 0, 0]))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(&out[0][..want.len()], &want[..], "vram_cbv={vram}: {out:x?}");
    }
}

#[test]
#[cfg(windows)]
fn table_cbv_preserves_raw_twin_routing_when_raw_views_enabled() {
    let mut words: Vec<u32> = vec![s_mov_b32(5, 0x80)];
    words.extend(s_load_dwordx4(8, 4, 0x00));
    words.extend(s_load_dwordx4(12, 4, 0x20));
    words.extend(s_buffer_load(10, 16, 8, 125, 0x10));
    for (out, s) in (16..20u8).enumerate() {
        words.push(v_mov_sgpr(1, s));
        words.extend(buffer_store_dword(1, 12, (4 * out) as u16));
    }
    words.push(S_ENDPGM);

    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![table(ResourceType::Cbv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    const TWINS: u32 = 2;
    let lowered = lower_cs(&native, true, TWINS);
    let mut cs = host_shader(lowered);
    validate_dxil(&mut cs.dxil);

    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    let data: Vec<u32> = (0..16u32).map(|i| 0x1111_0000 + i).collect();
    let poison: Vec<u32> = vec![0xDEAD_BEEF; 64];
    let views = vec![
        super::host::View::Cbv(&poison),
        super::host::View::RawUav(8),
        super::host::View::Buffer {
            data: &data,
            stride: super::host::Stride::Raw,
        },
        super::host::View::RawUav(8),
    ];
    let out = host
        .dispatch(
            &root,
            &pso,
            &super::host::Dispatch {
                roots: vec![(0, super::host::Root::Constants(vec![0; 5]))],
                views: &views,
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    // UAV slot 3 (twin of slot 1) holds the output; must come from twin slot 2, not poison slot 0.
    assert_eq!(&out[1][..4], &data[4..8], "must read raw twin: {out:x?}");
}

#[test]
#[cfg(windows)]
fn vector_cbv_load_lowers_to_raw_and_matches_cbv_output() {
    // s0: row index (copied to v2), s4: table (slot 0 = CBV, slot 1 = Raw UAV)
    let mut words: Vec<u32> = vec![s_mov_b32(5, 0x80), v_mov_sgpr(2, 0)];
    words.extend(s_load_dwordx4(8, 4, 0x00));
    words.extend(s_load_dwordx4(12, 4, 0x20));
    // buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 idxen (16B stride per row)
    words.extend(buffer_load_dwordx4_idxen(4, 2, 8, 0));
    for k in 0..4u8 {
        words.extend(buffer_store_dword(4 + k, 12, u16::from(k) * 4));
    }
    words.push(S_ENDPGM);

    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 16,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![table(ResourceType::Cbv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let off = lower_cs(&native, false, 0);
    let on = lower_cs(&native, true, 0);
    assert_ne!(off.dxil, on.dxil);

    let data: Vec<u32> = (0..16u32).map(|i| 0x2000_0000 + i * 17).collect();
    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        return;
    };
    for (vram, lowered) in [(false, off), (true, on)] {
        let mut cs = host_shader(lowered);
        validate_dxil(&mut cs.dxil);
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let views = if vram {
            vec![
                super::host::View::Buffer {
                    data: &data,
                    stride: super::host::Stride::Raw,
                },
                super::host::View::RawUav(8),
            ]
        } else {
            vec![super::host::View::Cbv(&data), super::host::View::RawUav(8)]
        };
        for (row, want) in [(1u32, &data[4..8]), (100u32, &[0, 0, 0, 0][..])] {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(0, super::host::Root::Constants(vec![row, 0, 0, 0, 0]))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(&out[0][..4], want, "vram_cbv={vram} row={row}: {out:x?}");
        }
    }
}

#[test]
#[cfg(windows)]
fn root_cbv_lowers_to_root_srv_with_user_view_bias() {
    // s0: dynamic offset (RootType::Constants, slot 0)
    // s[2:3]: root CBV 64-bit address (RootType::Descriptor, slot 1)
    // s4: table (slot 2, entry 0 = Raw UAV)
    let mut words: Vec<u32> = vec![
        s_mov_b32(5, 0x80),
        // Build root CBV descriptor in s[8:11] from s[2:3].
        0xBE88_0302, // s_mov_b32 s8, s2
        0x8809_FF03,
        0x0010_0000, // s_or_b32 s9, s3, 0x100000
        0xBE8A_03C1, // s_mov_b32 s10, -1
        0xBE8B_03FF,
        0x1101_4FAC, // s_mov_b32 s11, 0x11014fac
    ];
    words.extend(s_load_dwordx4(12, 4, 0x00)); // s[12:15] = UAV (slot 0)
    words.extend(s_buffer_load(10, 16, 8, 125, 0x10)); // s[16:19] @ 0x10
    words.extend(s_buffer_load(8, 20, 8, 0, 0x00)); // s20 @ s0
    for (out, s) in (16..21u8).enumerate() {
        words.push(v_mov_sgpr(1, s));
        words.extend(buffer_store_dword(1, 12, (4 * out) as u16));
    }
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let off = lower_cs(&native, false, 0);
    assert_eq!(off.root_cbvs, vec![1]);
    assert!(off.root_srvs.is_empty());
    assert!(
        !off.user_data
            .iter()
            .any(|u| u.root == RootType::Other(shader::USER_VIEW_BIAS))
    );

    let on = lower_cs(&native, true, 0);
    assert!(on.root_cbvs.is_empty(), "root CBV must move out of root_cbvs");
    assert_eq!(on.root_srvs, vec![1], "root CBV must move into root_srvs");
    let bias_entry = on
        .user_data
        .iter()
        .find(|u| u.slot == 1 && u.root == RootType::Other(shader::USER_VIEW_BIAS))
        .expect("lowered root CBV must request USER_VIEW_BIAS");
    let bias_index = bias_entry.index as usize;

    let data: Vec<u32> = (0..20u32).map(|i| 0x3300_0000 + i * 13).collect();
    let s0 = 8u32;

    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        return;
    };

    // Verify =0 (root CBV) and =1 (root SRV with bias=0 and bias=4).
    let mut cs_off = host_shader(off);
    validate_dxil(&mut cs_off.dxil);
    let (root_off, pso_off) = host.compute_pipeline(&cs_off).unwrap();
    let mut ud_off = vec![0u32; cs_off.user_data_dwords() as usize];
    ud_off[0] = s0;
    let out_off = host
        .dispatch(
            &root_off,
            &pso_off,
            &super::host::Dispatch {
                roots: vec![
                    (0, super::host::Root::Constants(ud_off)),
                    (1, super::host::Root::View { srv: false, data: data.clone() }),
                ],
                views: &[super::host::View::RawUav(8)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    let want_zero_bias = [data[4], data[5], data[6], data[7], data[2]];
    assert_eq!(&out_off[0][..5], &want_zero_bias[..]);

    let mut cs_on = host_shader(on);
    validate_dxil(&mut cs_on.dxil);
    let (root_on, pso_on) = host.compute_pipeline(&cs_on).unwrap();
    for bias in [0u32, 4u32, 12u32] {
        let mut ud = vec![0u32; cs_on.user_data_dwords() as usize];
        ud[0] = s0;
        ud[bias_index] = bias;
        let out_on = host
            .dispatch(
                &root_on,
                &pso_on,
                &super::host::Dispatch {
                    roots: vec![
                        (0, super::host::Root::Constants(ud)),
                        (1, super::host::Root::View { srv: true, data: data.clone() }),
                    ],
                    views: &[super::host::View::RawUav(8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let shift = (bias / 4) as usize;
        let want = [
            data[4 + shift],
            data[5 + shift],
            data[6 + shift],
            data[7 + shift],
            data[2 + shift],
        ];
        assert_eq!(&out_on[0][..5], &want[..], "root CBV bias={bias}: {out_on:x?}");
    }
}

#[test]
#[cfg(windows)]
fn bindless_resource_heap_cbv_reads_raw_with_slot_strides_intact() {
    // s0: heap byte offset of dynamic CBV (ResourceHeap, slot 0 -> byte offset 0)
    // s2: table byte offset (slot 1 = Raw UAV)
    let mut words: Vec<u32> = vec![
        s_mov_b32(1, 0x80), // s[0:1] = ResourceHeap base + 0
        s_mov_b32(3, 0x80), // s[2:3] = Table base
    ];
    words.extend(s_load_dwordx4(8, 0, 0x00)); // s[8:11] = ResourceHeap slot 0 (no UsedResource)
    words.extend(s_load_dwordx4(12, 2, 0x20)); // s[12:15] = UAV (slot 1)
    words.extend(s_buffer_load(10, 16, 8, 125, 0x10)); // s[16:19] @ 0x10
    for (out, s) in (16..20u8).enumerate() {
        words.push(v_mov_sgpr(1, s));
        words.extend(buffer_store_dword(1, 12, (4 * out) as u16));
    }
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::ResourceHeap,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 2,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let off = lower_cs(&native, false, 0);
    let on = lower_cs(&native, true, 0);
    assert!(off.root_srvs.contains(&shader::SLOT_STRIDES));
    assert!(
        on.root_srvs.contains(&shader::SLOT_STRIDES),
        "SLOT_STRIDES must stay live for ResourceHeap buffer sweep"
    );
    assert_ne!(off.dxil, on.dxil);

    let data: Vec<u32> = (0..16u32).map(|i| 0x5500_0000 + i * 7).collect();
    let strides_zero = vec![0u32; 16]; // stride == 0 marks a CBV slot in SLOT_STRIDES

    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        return;
    };
    for (vram, lowered) in [(false, off), (true, on)] {
        let mut cs = host_shader(lowered);
        validate_dxil(&mut cs.dxil);
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let views = if vram {
            vec![
                super::host::View::Buffer {
                    data: &data,
                    stride: super::host::Stride::Raw,
                },
                super::host::View::RawUav(8),
            ]
        } else {
            vec![super::host::View::Cbv(&data), super::host::View::RawUav(8)]
        };
        let ud = vec![0u32; cs.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &super::host::Dispatch {
                    roots: vec![
                        (0, super::host::Root::Constants(ud)),
                        (
                            1,
                            super::host::Root::View {
                                srv: true,
                                data: strides_zero.clone(),
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(&out[0][..4], &data[4..8], "vram_cbv={vram}: {out:x?}");
    }
}

#[test]
#[cfg(windows)]
fn selected_table_cbv_lowers_to_raw_in_place() {
    // s0: selector (0 => CBV slot 0, nonzero => CBV slot 1)
    // s4: table (slot 0 = CBV A, slot 1 = CBV B, slot 2 = Raw UAV)
    let mut words: Vec<u32> = vec![s_mov_b32(5, 0x80)];
    words.extend(s_load_dwordx4(8, 4, 0x00)); // CBV A (slot 0)
    words.extend(s_load_dwordx4(12, 4, 0x20)); // CBV B (slot 1)
    words.extend(s_load_dwordx4(24, 4, 0x40)); // UAV (slot 2)
    words.push(0xBF06_8000); // s_cmp_eq_u32 s0, 0
    words.push(0x8590_0C08); // s_cselect_b64 s[16:17], s[8:9], s[12:13]
    words.push(0x8592_0E0A); // s_cselect_b64 s[18:19], s[10:11], s[14:15]
    words.extend(s_buffer_load(10, 20, 16, 125, 0x00)); // s[20:23] from selected CBV
    for (out, s) in (20..24u8).enumerate() {
        words.push(v_mov_sgpr(1, s));
        words.extend(buffer_store_dword(1, 24, (4 * out) as u16));
    }
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 16,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let lowered = lower_cs(&native, true, 0);
    let mut cs = host_shader(lowered);
    validate_dxil(&mut cs.dxil);

    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    let cbv_a = vec![0xAAAA_0001u32, 0xAAAA_0002, 0xAAAA_0003, 0xAAAA_0004];
    let cbv_b = vec![0xBBBB_0001u32, 0xBBBB_0002, 0xBBBB_0003, 0xBBBB_0004];
    let views = vec![
        super::host::View::Buffer {
            data: &cbv_a,
            stride: super::host::Stride::Raw,
        },
        super::host::View::Buffer {
            data: &cbv_b,
            stride: super::host::Stride::Raw,
        },
        super::host::View::RawUav(8),
    ];
    for (sel, want) in [(0u32, &cbv_a[..]), (1u32, &cbv_b[..])] {
        let out = host
            .dispatch(
                &root,
                &pso,
                &super::host::Dispatch {
                    roots: vec![(0, super::host::Root::Constants(vec![sel, 0, 0, 0, 0]))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(&out[0][..4], want, "sel={sel}: {out:x?}");
    }
}

#[test]
#[cfg(windows)]
fn wide_and_partially_oob_cbv_loads_return_zeros_past_end() {
    // s4: table (slot 0 = 6-dword CBV = 24 bytes, slot 1 = Raw UAV)
    // s_buffer_load_dwordx8 (op = 11) reads 8 dwords @ 0x00: dwords 0..6 in-bounds, 6..8 OOB -> 0.
    let mut words: Vec<u32> = vec![s_mov_b32(5, 0x80)];
    words.extend(s_load_dwordx4(8, 4, 0x00));
    words.extend(s_load_dwordx4(12, 4, 0x20));
    words.extend(s_buffer_load(11, 16, 8, 125, 0x00)); // s[16:23] = dwordx8 @ 0x00
    for (out, s) in (16..24u8).enumerate() {
        words.push(v_mov_sgpr(1, s));
        words.extend(buffer_store_dword(1, 12, (4 * out) as u16));
    }
    words.push(S_ENDPGM);

    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![table(ResourceType::Cbv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let lowered = lower_cs(&native, true, 0);
    let mut cs = host_shader(lowered);
    validate_dxil(&mut cs.dxil);

    let _guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = super::host::Host::new(true) else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    // 6 dwords = 24 bytes: dwords 0..6 valid, dwords 6..8 past NumElements must read 0.
    let data = vec![0x1010_0001u32, 0x1010_0002, 0x1010_0003, 0x1010_0004, 0x1010_0005, 0x1010_0006];
    let want = [data[0], data[1], data[2], data[3], data[4], data[5], 0, 0];
    let views = vec![
        super::host::View::Buffer {
            data: &data,
            stride: super::host::Stride::Raw,
        },
        super::host::View::RawUav(8),
    ];
    let out = host
        .dispatch(
            &root,
            &pso,
            &super::host::Dispatch {
                roots: vec![(0, super::host::Root::Constants(vec![0; 5]))],
                views: &views,
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap();
    assert_eq!(&out[0][..8], &want[..], "partial OOB dwordx8: {out:x?}");
}
