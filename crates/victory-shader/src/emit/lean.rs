//! Exact Wave32 uniformity and IEEE-754 bit-range class helpers for NVIDIA
//! Blackwell (`host_max_wave == 32`) and general SM 6.6 hosts.
//!
//! - `VICTORY_CLASS_RANGES` (on unless `0`): `v_cmp_class_f32/f16` with a
//!   constant class mask tests the magnitude bit ranges covered by those
//!   classes (zero, denormals, normals, infinity, signalling and quiet NaNs
//!   lie in monotonic order) with sign checks only where required: `±inf`
//!   lowers to two instructions instead of ~20.
//! - `VICTORY_WAVE_UNIFORM` (default `3` on Wave32 hosts, `0` on Wave64 hosts,
//!   override with `0..=3`): strengthens [`lane_uniform`] so console wave64
//!   compute shaders running in two host waves of 32 prove more values alike
//!   across every lane, turning `v_readfirstlane` into a direct host
//!   `WaveReadLaneFirst` instead of a two-barrier LDS round-trip.
//!   - Bit 1: judge each VGPR operand by its own register span rather than
//!     every higher VGPR read (ignoring overwritten old dst values on unguarded
//!     non-`mac`/`dot`/`movrel` instructions).
//!   - Bit 2: propagate uniformity across unguarded read-only SRV/CBV buffer
//!     loads at uniform addresses and unraced `ds_read*` instructions.

use std::collections::HashSet;

use super::bitcode::{Bin, Module, V, pred};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::resource::Site;
use crate::structure::Structured;

/// Whether constant-mask `v_cmp_class_f32`/`f16` lowers to exact bit-range
/// comparisons (`VICTORY_CLASS_RANGES`, on by default unless `"0"`).
pub(super) fn class_ranges_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_CLASS_RANGES.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLASS_RANGES").is_none_or(|v| v != "0"))
}

/// `VICTORY_WAVE_UNIFORM` bitmask (`& 3`): defaults to `3` on Wave32 hosts
/// (`host_max_wave() == 32`, e.g. NVIDIA Blackwell) where every false
/// non-uniform `v_readfirstlane` pays two group barriers + LDS exchange, and
/// `0` otherwise unless explicitly configured.
pub(super) fn wave_uniform() -> u8 {
    #[cfg(test)]
    if let Some(bits) = TEST_WAVE_UNIFORM.get() {
        return bits & 3;
    }
    static OVERRIDE: std::sync::OnceLock<Option<u8>> = std::sync::OnceLock::new();
    let asked = *OVERRIDE.get_or_init(|| {
        std::env::var("VICTORY_WAVE_UNIFORM")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .map(|b| b & 3)
    });
    asked.unwrap_or_else(|| if super::lower::host_max_wave() == 32 { 3 } else { 0 })
}

#[cfg(test)]
thread_local! {
    /// Per-thread test override for [`class_ranges_on`].
    pub(super) static TEST_CLASS_RANGES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Per-thread test override for [`wave_uniform`].
    pub(super) static TEST_WAVE_UNIFORM: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// Whether no LDS write in the program can race with the read at `at`
/// between two waiting `s_barrier`s (`waits` names the blocks whose barriers
/// wait for every thread in the group).
fn lds_unraced(p: &Program, at: (usize, usize), waits: &dyn Fn(usize) -> bool) -> bool {
    let writes = |inst: &Inst| match &inst.fields {
        Fields::Ds { .. } => {
            let n = inst.name();
            !(n.starts_with("ds_read")
                || matches!(
                    n,
                    "ds_swizzle_b32" | "ds_permute_b32" | "ds_bpermute_b32" | "ds_nop"
                ))
        }
        Fields::Buffer { lds, .. } | Fields::Flat { lds, .. } => *lds,
        _ => false,
    };
    let barrier = |b: usize, inst: &Inst| inst.name() == "s_barrier" && waits(b);
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut seen = vec![false; p.blocks.len()];
    let mut back = vec![at];
    while let Some((b, end)) = back.pop() {
        let insts = &p.blocks[b].insts;
        let mut started = false;
        for j in (0..end.min(insts.len())).rev() {
            if barrier(b, &insts[j]) {
                starts.push((b, j + 1));
                started = true;
                break;
            }
            if writes(&insts[j]) {
                return false;
            }
        }
        if started {
            continue;
        }
        if b == 0 || p.blocks[b].preds.is_empty() {
            starts.push((b, 0));
        }
        for &q in &p.blocks[b].preds {
            if !std::mem::replace(&mut seen[q], true) {
                back.push((q, p.blocks[q].insts.len()));
            }
        }
    }
    let mut seen = vec![false; p.blocks.len()];
    let mut ahead = starts;
    while let Some((b, from)) = ahead.pop() {
        let block = &p.blocks[b];
        let mut ended = false;
        for inst in block.insts.iter().skip(from) {
            if barrier(b, inst) {
                ended = true;
                break;
            }
            if writes(inst) {
                return false;
            }
        }
        if ended {
            continue;
        }
        if matches!(block.exit, crate::ir::Exit::Indirect) {
            return false;
        }
        for s in block.exit.successors() {
            if !std::mem::replace(&mut seen[s], true) {
                ahead.push((s, 0));
            }
        }
    }
    true
}

/// VGPR values proven equal in every lane of a console wave, for avoiding
/// cross-half LDS exchanges on Wave32 hosts.
pub(super) fn lane_uniform(
    p: &Program,
    st: &Structured,
    all: &[bool],
    ordered_lds: bool,
    waits: &dyn Fn(usize) -> bool,
    read_only: &dyn Fn(Site) -> bool,
) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let bits = wave_uniform();
    let own_registers = bits & 1 != 0;
    let rich_loads = bits & 2 != 0;
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block
            .preds
            .iter()
            .all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            if let Fields::Ds {
                addr: Some(Operand::Vector { reg, dwords }),
                gds: false,
                ..
            } = inst.fields
                && inst.read(Reg::V(reg)).is_some_and(|a| same.contains(&a))
            {
                let ds_ok = if rich_loads {
                    name.starts_with("ds_read") && (ordered_lds || lds_unraced(p, (b, i), waits))
                } else {
                    ordered_lds && dwords == 1 && name == "ds_read_b32"
                };
                if ds_ok {
                    same.extend(
                        inst.writes
                            .iter()
                            .filter(|w| matches!(w.0, Reg::V(_)))
                            .map(|w| w.1),
                    );
                }
                continue;
            }
            if rich_loads
                && let Fields::Buffer {
                    addr: Operand::Vector { reg, dwords },
                    offen,
                    idxen,
                    lds: false,
                    tfe: false,
                    ..
                } = inst.fields
                && name.contains("_load_")
                && read_only((b, i))
            {
                // without `offen` and `idxen` the address register is not read
                let addressed = !(offen || idxen)
                    || (reg..reg.saturating_add(dwords))
                        .all(|r| inst.read(Reg::V(r)).is_some_and(|a| same.contains(&a)));
                if addressed {
                    same.extend(
                        inst.writes
                            .iter()
                            .filter(|w| matches!(w.0, Reg::V(_)))
                            .map(|w| w.1),
                    );
                }
                continue;
            }
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                _ => continue,
            };
            // `VICTORY_WAVE_UNIFORM` bit 1: an operand is its own registers;
            // of the other VGPRs the instruction reads, the old values of
            // those it writes (unguarded here) do not count.
            let own = own_registers
                && !(name.contains("mac") || name.contains("dot") || name.contains("movrel"));
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, dwords } => {
                    let end = if own {
                        reg.saturating_add(dwords.max(1))
                    } else {
                        u8::MAX
                    };
                    let read: Vec<Value> = inst
                        .reads
                        .iter()
                        .filter(|r| matches!(r.0, Reg::V(x) if x >= reg && (x < end || !own)))
                        .map(|r| r.1)
                        .collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => inst
                    .read(Reg::S(code))
                    .is_none_or(|x| !is_mask(x) || same.contains(&x)),
                _ => true,
            }) && (!own || {
                let in_operand = |x: u8| {
                    ops.iter().any(|op| {
                        matches!(*op, Operand::Vector { reg, dwords } if x >= reg && x < reg.saturating_add(dwords.max(1)))
                    })
                };
                let written = |x: u8| inst.writes.iter().any(|w| w.0 == Reg::V(x));
                inst.reads.iter().all(|r| match r.0 {
                    Reg::V(x) if !in_operand(x) => written(x) || same.contains(&r.1),
                    _ => true,
                })
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

/// A run of magnitude bits `lo..=hi` and the sign constraint (`None`: either).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Run {
    pub lo: u32,
    pub hi: u32,
    pub negative: Option<bool>,
}

/// Magnitudes of each float class in monotonic order: zero, denormals,
/// normals, infinity, signalling NaNs, quiet NaNs; with positive/negative
/// mask bit indices.
fn kinds(f16: bool) -> [(u32, u32, u32, u32); 6] {
    let (inf, quiet, top) = if f16 {
        (0x7C00, 0x7E00, 0x7FFF)
    } else {
        (0x7F80_0000, 0x7FC0_0000, 0x7FFF_FFFF)
    };
    let first_normal = if f16 { 0x0400 } else { 0x0080_0000 };
    [
        (0, 0, 6, 5),
        (1, first_normal - 1, 7, 4),
        (first_normal, inf - 1, 8, 3),
        (inf, inf, 9, 2),
        (inf + 1, quiet - 1, 0, 0),
        (quiet, top, 1, 1),
    ]
}

/// Coalesced magnitude bit ranges selected by a 10-bit RDNA2 class mask.
pub(super) fn class_runs(mask: u32, f16: bool) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (lo, hi, pos, neg) in kinds(f16) {
        let (p, n) = (mask >> pos & 1 != 0, mask >> neg & 1 != 0);
        let negative = match (p, n) {
            (false, false) => continue,
            (true, true) => None,
            (true, false) => Some(false),
            (false, true) => Some(true),
        };
        match runs.last_mut() {
            Some(r) if r.hi + 1 == lo && r.negative == negative => r.hi = hi,
            _ => runs.push(Run { lo, hi, negative }),
        }
    }
    runs
}

/// Bit-exact `v_cmp_class_f32`/`f16` lowering for a constant class mask.
pub(super) fn class_test(m: &mut Module, c: usize, x: V, mask: u32, f16: bool) -> V {
    let runs = class_runs(mask, f16);
    let (top, sign_shift) = if f16 {
        (0x7FFF, 15)
    } else {
        (0x7FFF_FFFF, 31)
    };
    if runs.is_empty() {
        return m.bool(false);
    }
    let k = |m: &mut Module, v: u32| m.i32(v);
    let km = k(m, top);
    let abs = m.bin(c, Bin::And, x, km);
    let mut sign = None;
    let mut result: Option<V> = None;
    for run in runs {
        let range = match (run.lo, run.hi) {
            (0, hi) if hi == top => None,
            (lo, hi) if lo == hi => {
                let v = k(m, lo);
                Some(m.cmp(c, pred::EQ, abs, v))
            }
            (0, hi) => {
                let v = k(m, hi);
                Some(m.cmp(c, pred::IULE, abs, v))
            }
            (lo, hi) if hi == top => {
                let v = k(m, lo);
                Some(m.cmp(c, pred::IUGE, abs, v))
            }
            (lo, hi) => {
                let (l, n) = (k(m, lo), k(m, hi - lo));
                let off = m.bin(c, Bin::Sub, abs, l);
                Some(m.cmp(c, pred::IULE, off, n))
            }
        };
        let signed = run.negative.map(|negative| {
            let s = *sign.get_or_insert_with(|| {
                let sh = k(m, sign_shift);
                let s = m.bin(c, Bin::LShr, x, sh);
                let z = k(m, 0);
                m.cmp(c, pred::NE, s, z)
            });
            if negative {
                s
            } else {
                let t = m.bool(true);
                m.bin(c, Bin::Xor, s, t)
            }
        });
        let term = match (range, signed) {
            (Some(a), Some(b)) => m.bin(c, Bin::And, a, b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => m.bool(true),
        };
        result = Some(match result {
            Some(r) => m.bin(c, Bin::Or, r, term),
            None => term,
        });
    }
    result.unwrap_or_else(|| m.bool(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class_of(x: u32, f16: bool) -> u32 {
        let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
            (10, 15, 0x1F, 0x03FF, 0x0200)
        } else {
            (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
        };
        let e = x >> exp_sh & exp_mask;
        let m = x & mant_mask;
        let neg = x >> sign_sh != 0;
        let t = if m == 0 { 6 } else { 7 };
        let t = if e == 0 { t } else { 8 };
        let t = if e == exp_mask { 9 } else { t };
        let t = if neg { 11 - t } else { t };
        if m != 0 && e == exp_mask {
            u32::from(x & quiet_bit != 0)
        } else {
            t
        }
    }

    fn runs_say(runs: &[Run], x: u32, f16: bool) -> bool {
        let (top, sign_shift) = if f16 {
            (0x7FFF, 15)
        } else {
            (0x7FFF_FFFF, 31)
        };
        let abs = x & top;
        let neg = x >> sign_shift != 0;
        runs.iter()
            .any(|r| abs >= r.lo && abs <= r.hi && r.negative.is_none_or(|n| n == neg))
    }

    #[test]
    fn runs_agree_with_the_general_form_for_every_half_and_mask() {
        for mask in 0..1024u32 {
            let runs = class_runs(mask, true);
            for x in 0..=0xFFFFu32 {
                let want = mask >> class_of(x, true) & 1 != 0;
                assert_eq!(runs_say(&runs, x, true), want, "mask {mask:#x} x {x:#06x}");
            }
            for x in [0x1_0000u32, 0x1_3C00, 0xFFFF_7C00] {
                let want = mask >> class_of(x, true) & 1 != 0;
                assert_eq!(runs_say(&runs, x, true), want, "mask {mask:#x} x {x:#x}");
            }
        }
    }

    #[test]
    fn runs_agree_with_the_general_form_for_floats() {
        let mut xs = Vec::new();
        for edge in [
            0u32,
            1,
            0x7F_FFFF,
            0x80_0000,
            0x3F80_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7F80_0001,
            0x7FBF_FFFF,
            0x7FC0_0000,
            0x7FFF_FFFF,
        ] {
            for d in [0u32, 1, 2] {
                for x in [edge.wrapping_sub(d), edge.wrapping_add(d)] {
                    xs.push(x & 0x7FFF_FFFF);
                    xs.push(x | 0x8000_0000);
                }
            }
        }
        let mut seed = 0x9E37_79B9u32;
        for _ in 0..4096 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            xs.push(seed);
        }
        for mask in 0..1024u32 {
            let runs = class_runs(mask, false);
            for &x in &xs {
                let want = mask >> class_of(x, false) & 1 != 0;
                assert_eq!(runs_say(&runs, x, false), want, "mask {mask:#x} x {x:#010x}");
            }
        }
    }

    #[test]
    fn wave_uniform_defaults_on_for_wave32_hosts() {
        let prev_wave = super::super::lower::TEST_HOST_WAVE.replace(Some(32));
        let prev_uni = TEST_WAVE_UNIFORM.replace(None);
        assert_eq!(wave_uniform(), 3);
        super::super::lower::TEST_HOST_WAVE.set(Some(64));
        assert_eq!(wave_uniform(), 0);
        TEST_WAVE_UNIFORM.set(prev_uni);
        super::super::lower::TEST_HOST_WAVE.set(prev_wave);
    }
}
