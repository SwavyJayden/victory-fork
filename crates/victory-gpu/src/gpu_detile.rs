//! GPU compute texture detiling (`cs_5_1`) for streamed console textures.
//!
//! Streamed 1D/2D/2D-array textures arrive in title memory in GFX10.3 tiled
//! layouts (`SW_256B_*`, `SW_4KB_*`, `SW_64KB_*`). Detiling on the CPU scatters
//! small 8/16/32-byte runs into write-combined `D3D12_HEAP_TYPE_UPLOAD` staging
//! memory. When eligible (`VICTORY_GPU_DETILE`, on unless `0`), this module:
//!
//! 1. Keeps each texture's linear staging buffer in a pooled `DEFAULT`-heap
//!    VRAM buffer (`ALLOW_UNORDERED_ACCESS`, resting in `COMMON`), so
//!    `CopyTextureRegion` recorded on the title's command list reads directly
//!    from VRAM without touching the list's pipeline state, root signature, or
//!    descriptor heaps.
//! 2. At submission refresh time, copies the raw tiled bytes sequentially with
//!    a single `copy_nonoverlapping` into a reusable `UPLOAD` buffer and runs a
//!    `cs_5_1` detile pass on a dedicated compute queue (falling back to a
//!    direct queue if needed) using direct root SRVs and a direct root UAV (zero
//!    descriptor heaps).
//! 3. Transitions the linear VRAM buffer back to `COMMON` before returning so
//!    the host direct queue's `CopyTextureRegion` implicitly promotes it to
//!    `COPY_SOURCE` and decays it back to `COMMON` at submission end.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_formats::texture::{Dimension, Layout, Swizzle};
use windows::Win32::Foundation::{E_FAIL, HANDLE, WAIT_FAILED, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, s};

use crate::host::{Host, transition};

/// Minimum tiled size eligible for GPU detiling; smaller textures stay on the
/// CPU where scatter cost is negligible and dispatch overhead is not worth it.
pub(crate) const MIN_BYTES: usize = 16 << 10;

/// Largest single VRAM linear staging buffer kept in [`VRAM_SPARE`].
const VRAM_SPARE_MAX_SINGLE: u64 = 16 << 20;
/// Maximum idle VRAM bytes kept in [`VRAM_SPARE`].
const VRAM_SPARE_BUDGET: u64 = 64 << 20;
/// Maximum idle VRAM bytes retained after an idle trim period.
const VRAM_SPARE_IDLE_KEEP: u64 = 16 << 20;
/// Largest persistent `UPLOAD` buffer kept for raw tiled inputs.
const RAW_UPLOAD_KEEP_MAX: u64 = 16 << 20;
/// Persistent raw upload size retained after idle trim.
const RAW_UPLOAD_IDLE_KEEP: u64 = 4 << 20;

/// Compute shader detiling one 2D/1D/array subresource from console tiled
/// layout (`t0`) using a 2 KiB coordinate-bit LUT (`t1`: `pat_x[256]`, then
/// `pat_y[256]`) into D3D12 placed linear rows (`u0`).
const DETILE_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=16, b0), SRV(t0), SRV(t1), UAV(u0)"

cbuffer Params : register(b0) {
    uint src_base;
    uint dst_base;
    uint width;
    uint height;
    uint row_pitch;
    uint elem_bytes;
    uint block_log2;
    uint bw_log2;
    uint bh_log2;
    uint bw_mask;
    uint bh_mask;
    uint pitch_blocks;
    uint tail_x;
    uint tail_y;
    uint reserved0;
    uint reserved1;
};

ByteAddressBuffer src : register(t0);
ByteAddressBuffer pat : register(t1);
RWByteAddressBuffer dst : register(u0);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void detile_main(uint3 id : SV_DispatchThreadID) {
    uint y = id.y;
    if (y >= height) return;

    uint ye = y + tail_y;
    uint row_blocks = (ye >> bh_log2) * pitch_blocks;
    uint py = pat.Load(1024u + (ye & bh_mask) * 4u);
    uint row_src_base = src_base + (row_blocks << block_log2) + py;
    uint dst_row = dst_base + y * row_pitch;

    if (elem_bytes == 1u) {
        uint x0 = id.x * 4u;
        if (x0 >= width) return;
        uint packed = 0u;
        [unroll]
        for (uint k = 0u; k < 4u; ++k) {
            uint x = x0 + k;
            if (x < width) {
                uint xe = x + tail_x;
                uint px = pat.Load((xe & bw_mask) * 4u);
                uint a = row_src_base + ((xe >> bw_log2) << block_log2) + px;
                uint b = (src.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xffu;
                packed |= b << (k * 8u);
            }
        }
        dst.Store(dst_row + x0, packed);
    } else if (elem_bytes == 2u) {
        uint x0 = id.x * 2u;
        if (x0 >= width) return;
        uint packed = 0u;
        [unroll]
        for (uint k = 0u; k < 2u; ++k) {
            uint x = x0 + k;
            if (x < width) {
                uint xe = x + tail_x;
                uint px = pat.Load((xe & bw_mask) * 4u);
                uint a = row_src_base + ((xe >> bw_log2) << block_log2) + px;
                uint h = (src.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xffffu;
                packed |= h << (k * 16u);
            }
        }
        dst.Store(dst_row + x0 * 2u, packed);
    } else if (elem_bytes == 4u) {
        uint x = id.x;
        if (x >= width) return;
        uint xe = x + tail_x;
        uint px = pat.Load((xe & bw_mask) * 4u);
        uint a = row_src_base + ((xe >> bw_log2) << block_log2) + px;
        dst.Store(dst_row + x * 4u, src.Load(a));
    } else if (elem_bytes == 8u) {
        uint x = id.x;
        if (x >= width) return;
        uint xe = x + tail_x;
        uint px = pat.Load((xe & bw_mask) * 4u);
        uint a = row_src_base + ((xe >> bw_log2) << block_log2) + px;
        dst.Store2(dst_row + x * 8u, src.Load2(a));
    } else {
        uint x = id.x;
        if (x >= width) return;
        uint xe = x + tail_x;
        uint px = pat.Load((xe & bw_mask) * 4u);
        uint a = row_src_base + ((xe >> bw_log2) << block_log2) + px;
        dst.Store4(dst_row + x * 16u, src.Load4(a));
    }
}
"#;

/// Zero-fills a 4-byte-aligned dword range inside a linear VRAM buffer (used
/// for unheld `on_demand` mips and recycled VRAM buffers whose refresh skipped
/// detiling).
const ZERO_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"

cbuffer ZeroParams : register(b0) {
    uint dst_off;
    uint count_dwords;
    uint num_groups_x;
    uint reserved;
};

RWByteAddressBuffer dst : register(u0);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void zero_main(uint3 id : SV_DispatchThreadID) {
    uint stride = num_groups_x * 64u;
    uint vec_count = count_dwords >> 2u;
    [loop]
    for (uint i = id.x; i < vec_count; i += stride) {
        dst.Store4(dst_off + i * 16u, uint4(0u, 0u, 0u, 0u));
    }
    uint tail_base = vec_count << 2u;
    uint tail = count_dwords - tail_base;
    if (id.x < tail) {
        dst.Store(dst_off + (tail_base + id.x) * 4u, 0u);
    }
}
"#;

/// Counters since the last [`line`] call:
/// 0 = GPU-detiled textures, 1 = raw tiled bytes, 2 = linear bytes,
/// 3 = GPU detile wall microseconds, 4 = CPU-detiled textures,
/// 5 = VRAM-cap / engine fallbacks from eligible textures.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Committed VRAM bytes currently held by live `Staging` entries.
static ACTIVE_VRAM_BYTES: AtomicU64 = AtomicU64::new(0);

/// Pooled idle VRAM linear staging buffers by power-of-two size, and total
/// pooled bytes.
struct VramSpare(ID3D12Resource);
// SAFETY: `ID3D12Resource` is free-threaded; entries are only touched under
// `VRAM_SPARE`'s mutex and only returned after GPU completion.
unsafe impl Send for VramSpare {}

static VRAM_SPARE: Mutex<(BTreeMap<u64, Vec<VramSpare>>, u64)> =
    Mutex::new((BTreeMap::new(), 0));
static VRAM_SPARE_USED_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
static TEST_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether GPU compute texture detiling is enabled (`VICTORY_GPU_DETILE`;
/// on by default, `0` disables).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    match TEST_OVERRIDE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| enabled_from(std::env::var("VICTORY_GPU_DETILE").ok().as_deref()))
}

fn enabled_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Active VRAM staging budget (`VICTORY_GPU_DETILE_VRAM_MB`, else scaled by
/// adapter VRAM headroom: 256 MiB with >12 GiB budget and >=3 GiB free,
/// 128 MiB with >=1.5 GiB free or unknown, 64 MiB under tighter pressure).
fn vram_active_budget(host: &Host) -> u64 {
    static OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();
    if let Some(bytes) = *OVERRIDE.get_or_init(|| {
        std::env::var("VICTORY_GPU_DETILE_VRAM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|mb| mb << 20)
    }) {
        return bytes;
    }
    const GIB: u64 = 1024 * 1024 * 1024;
    match host.video_memory() {
        Some([budget, used, ..]) if budget > 0 => {
            let free = budget.saturating_sub(used);
            if budget > 12 * GIB && free >= 3 * GIB {
                256 << 20
            } else if free >= (3 * GIB) / 2 {
                128 << 20
            } else {
                64 << 20
            }
        }
        _ => 128 << 20,
    }
}

/// Checks whether a console texture layout and raw tiled byte span satisfy
/// GPU-detile preconditions.
pub(crate) fn eligible_layout(layout: &Layout, tiled_len: usize) -> bool {
    if !layout.addressable()
        || layout.desc.swizzle == Swizzle::LINEAR
        || layout.thick
        || layout.desc.samples != 1
        || layout.desc.dimension == Dimension::Tex3d
        || !matches!(layout.desc.element_bytes, 1 | 2 | 4 | 8 | 16)
        || tiled_len < MIN_BYTES
        || tiled_len > (u32::MAX as usize) - 256
    {
        return false;
    }
    let [bw, bh, bd] = layout.block;
    if bd != 1 || !bw.is_power_of_two() || !bh.is_power_of_two() {
        return false;
    }
    let pat = layout.pattern();
    pat[0].len() == bw as usize
        && pat[1].len() == bh as usize
        && pat[0].len() <= 256
        && pat[1].len() <= 256
        && pat[2].first().copied().unwrap_or(0) == 0
}

/// Full eligibility check for a `Staging` entry (layout, footprints, bounds).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eligible(
    layout: &Layout,
    tiled_len: usize,
    total_linear: u64,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    row_sizes: &[u64],
    first: u32,
    mips: u32,
) -> bool {
    if !enabled() || !eligible_layout(layout, tiled_len) {
        return false;
    }
    if total_linear == 0 || total_linear > u64::from(u32::MAX - 512) || footprints.is_empty() {
        return false;
    }
    if first >= mips || layout.mips.len() < mips as usize {
        return false;
    }
    let mips_per_slice = (mips - first) as usize;
    if footprints.len() != rows.len()
        || footprints.len() != row_sizes.len()
        || !footprints.len().is_multiple_of(mips_per_slice)
    {
        return false;
    }
    let slices = (footprints.len() / mips_per_slice) as u32;
    if slices == 0 || slices > layout.slices {
        return false;
    }
    let total_aligned = total_linear.next_multiple_of(4);
    for (i, ((fp, &sub_rows), &row_size)) in footprints
        .iter()
        .zip(rows)
        .zip(row_sizes)
        .enumerate()
    {
        let mip = first + (i % mips_per_slice) as u32;
        let slice = (i / mips_per_slice) as u32;
        let m = &layout.mips[mip as usize];
        let f = fp.Footprint;
        if f.Depth != 1
            || !fp.Offset.is_multiple_of(4)
            || !f.RowPitch.is_multiple_of(4)
            || sub_rows != m.height
            || row_size != u64::from(m.width) * u64::from(layout.desc.element_bytes)
            || u64::from(f.RowPitch) < row_size
        {
            return false;
        }
        let Some(src_base) = u64::from(slice)
            .checked_mul(layout.slice_size)
            .and_then(|s| s.checked_add(m.offset))
        else {
            return false;
        };
        if src_base >= tiled_len as u64 {
            return false;
        }
        let span = subresource_span(fp.Offset, f.RowPitch, sub_rows, row_size, total_linear);
        if fp.Offset.saturating_add((span as u64).next_multiple_of(4)) > total_aligned {
            return false;
        }
    }
    true
}

fn subresource_span(offset: u64, pitch: u32, rows: u32, row_size: u64, total: u64) -> usize {
    if rows == 0 {
        return 0;
    }
    let exact = u64::from(rows - 1) * u64::from(pitch) + row_size;
    exact.min(total.saturating_sub(offset)) as usize
}

/// Records a CPU detile for telemetry.
pub(crate) fn note_cpu_detile() {
    STATS[4].fetch_add(1, Ordering::Relaxed);
}

/// Records an eligible texture falling back to CPU detiling.
pub(crate) fn note_fallback() {
    STATS[5].fetch_add(1, Ordering::Relaxed);
}

/// Takes a `DEFAULT`-heap UAV buffer for linear staging of `total` bytes,
/// returning `(buffer, alloc_bytes, pooled_bucket)`.
/// Returns `None` if the engine failed to initialize, the active VRAM cap
/// would be exceeded, or D3D12 allocation failed.
pub(crate) fn take_vram_buf(host: &Host, total: u64) -> Option<(ID3D12Resource, u64, u64)> {
    if engine(host).is_none() {
        note_fallback();
        return None;
    }
    let min_alloc = total.max(4).next_multiple_of(512);
    let bucket = min_alloc.max(64 << 10).next_power_of_two();
    let (alloc_size, pooled) = if bucket <= VRAM_SPARE_MAX_SINGLE {
        (bucket, bucket)
    } else {
        (min_alloc, 0)
    };
    let cap = vram_active_budget(host);
    if ACTIVE_VRAM_BYTES
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            let next = cur.checked_add(alloc_size)?;
            (next <= cap).then_some(next)
        })
        .is_err()
    {
        note_fallback();
        return None;
    }
    if pooled != 0 {
        VRAM_SPARE_USED_MS.store(crate::host::now_ms(), Ordering::Relaxed);
        let reused = {
            let mut spare = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner());
            let taken = spare.0.get_mut(&pooled).and_then(Vec::pop);
            if taken.is_some() {
                spare.1 -= pooled;
            }
            taken
        };
        if let Some(VramSpare(buffer)) = reused {
            return Some((buffer, alloc_size, pooled));
        }
    }
    let Some(buffer) = host.uav_buffer(alloc_size) else {
        ACTIVE_VRAM_BYTES.fetch_sub(alloc_size, Ordering::Relaxed);
        note_fallback();
        return None;
    };
    crate::host::name(&buffer, || format!("gpu detile linear {alloc_size:#x}"));
    Some((buffer, alloc_size, pooled))
}

/// Returns a VRAM linear staging buffer once its `UploadBuf` drops (after GPU
/// completion / retirement).
pub(crate) fn release_vram_buf(buffer: ID3D12Resource, alloc_size: u64, pooled: u64) {
    ACTIVE_VRAM_BYTES.fetch_sub(alloc_size, Ordering::Relaxed);
    if pooled == 0 {
        return;
    }
    let mut spare = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner());
    if spare.1 + pooled <= VRAM_SPARE_BUDGET {
        spare.1 += pooled;
        spare.0.entry(pooled).or_default().push(VramSpare(buffer));
    }
}

/// Trims idle VRAM spare buffers and oversized raw upload memory (called from
/// `texture::let_go_idle`).
pub(crate) fn trim_idle(idle_ms: u64) {
    let now = crate::host::now_ms();
    let last = VRAM_SPARE_USED_MS.load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) >= idle_ms {
        let mut retired = Vec::new();
        {
            let mut spare = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner());
            let (buckets, total) = &mut *spare;
            if *total > VRAM_SPARE_IDLE_KEEP {
                let sizes: Vec<u64> = buckets.keys().copied().rev().collect();
                for size in sizes {
                    if *total <= VRAM_SPARE_IDLE_KEEP {
                        break;
                    }
                    if let Some(bucket) = buckets.get_mut(&size) {
                        while *total > VRAM_SPARE_IDLE_KEEP {
                            let Some(VramSpare(buf)) = bucket.pop() else {
                                break;
                            };
                            *total -= size;
                            retired.push(buf);
                        }
                        if bucket.is_empty() {
                            buckets.remove(&size);
                        }
                    }
                }
            }
        }
        for buf in retired {
            crate::host::retire(buf);
        }
    }
    if let Some(Some(mutex)) = ENGINE.get()
        && let Ok(mut eng) = mutex.try_lock()
        && eng.last_used_ms != 0
        && now.saturating_sub(eng.last_used_ms) >= idle_ms
        && eng
            .raw_upload
            .as_ref()
            .is_some_and(|r| r.cap > RAW_UPLOAD_IDLE_KEEP)
    {
        eng.raw_upload = None;
    }
}

struct RawUpload {
    buffer: ID3D12Resource,
    data: *mut u8,
    cap: u64,
}

struct LutEntry {
    _buffer: ID3D12Resource,
    gpu: u64,
}

struct DetileEngine {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fence_val: u64,
    event: HANDLE,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    detile_root: ID3D12RootSignature,
    detile_pso: ID3D12PipelineState,
    zero_root: ID3D12RootSignature,
    zero_pso: ID3D12PipelineState,
    luts: HashMap<(u32, u32), LutEntry>,
    raw_upload: Option<RawUpload>,
    last_used_ms: u64,
}

// SAFETY: D3D12 objects are free-threaded; the event handle, command list,
// allocator, LUT cache, and persistent upload mapping are accessed exclusively
// under `ENGINE`'s mutex.
unsafe impl Send for DetileEngine {}

static ENGINE: OnceLock<Option<Mutex<DetileEngine>>> = OnceLock::new();

fn engine(host: &Host) -> Option<&'static Mutex<DetileEngine>> {
    ENGINE
        .get_or_init(|| match DetileEngine::new(host) {
            Ok(eng) => {
                tracing::info!("gpu compute texture detiling ready (VICTORY_GPU_DETILE=0 disables)");
                Some(Mutex::new(eng))
            }
            Err(e) => {
                tracing::warn!(error = %e, "gpu compute texture detiling unavailable; falling back to CPU detiling");
                None
            }
        })
        .as_ref()
}

impl DetileEngine {
    fn new(host: &Host) -> windows::core::Result<Self> {
        let detile_bc = crate::host::compile(DETILE_SOURCE, s!("detile_main"), s!("cs_5_1"))?;
        let zero_bc = crate::host::compile(ZERO_SOURCE, s!("zero_main"), s!("cs_5_1"))?;
        let device = host.device();
        // SAFETY: compiled DXBC blobs with embedded root signatures; plain
        // D3D12 object creation on the live host device.
        unsafe {
            let detile_root: ID3D12RootSignature = device.CreateRootSignature(0, &detile_bc)?;
            let detile_pso: ID3D12PipelineState =
                device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&detile_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: detile_bc.as_ptr().cast(),
                        BytecodeLength: detile_bc.len(),
                    },
                    ..Default::default()
                })?;
            let zero_root: ID3D12RootSignature = device.CreateRootSignature(0, &zero_bc)?;
            let zero_pso: ID3D12PipelineState =
                device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&zero_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: zero_bc.as_ptr().cast(),
                        BytecodeLength: zero_bc.len(),
                    },
                    ..Default::default()
                })?;
            let (list_type, queue) = match device.CreateCommandQueue::<ID3D12CommandQueue>(
                &D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    ..Default::default()
                },
            ) {
                Ok(q) => (D3D12_COMMAND_LIST_TYPE_COMPUTE, q),
                Err(_) => (
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })?,
                ),
            };
            crate::host::name(&queue, || "gpu detile queue".to_owned());
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(list_type)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, list_type, &allocator, None)?;
            list.Close()?;
            Ok(Self {
                queue,
                fence,
                fence_val: 0,
                event,
                allocator,
                list,
                detile_root,
                detile_pso,
                zero_root,
                zero_pso,
                luts: HashMap::new(),
                raw_upload: None,
                last_used_ms: 0,
            })
        }
    }

    fn execute_and_wait(&mut self) -> windows::core::Result<()> {
        // SAFETY: `self.list` is open; queue, fence, and event belong to `self`
        // and are serialized by `ENGINE`'s mutex.
        unsafe {
            self.list.Close()?;
            let cmd: ID3D12CommandList = self.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            self.fence_val += 1;
            self.queue.Signal(&self.fence, self.fence_val)?;
            let completed = self.fence.GetCompletedValue();
            if completed == u64::MAX {
                return Err(windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED));
            }
            if completed < self.fence_val {
                self.fence.SetEventOnCompletion(self.fence_val, self.event)?;
                match WaitForSingleObject(self.event, INFINITE) {
                    WAIT_OBJECT_0 => {}
                    WAIT_FAILED => return Err(windows::core::Error::from_thread()),
                    _ => return Err(windows::core::Error::from_hresult(E_FAIL)),
                }
                let after = self.fence.GetCompletedValue();
                if after == u64::MAX || after < self.fence_val {
                    return Err(windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED));
                }
            }
            Ok(())
        }
    }

    /// Returns the GPU virtual address of the cached 2 KiB coordinate-bit LUT
    /// (`pat_x[256]`, `pat_y[256]`) for `(layout.desc.swizzle, element_log2)`.
    fn pattern_lut(&mut self, host: &Host, layout: &Layout) -> windows::core::Result<u64> {
        let key = (
            layout.desc.swizzle.0,
            layout.desc.element_bytes.trailing_zeros(),
        );
        if let Some(entry) = self.luts.get(&key) {
            return Ok(entry.gpu);
        }
        let pat = layout.pattern();
        let mut table = [0u32; 512];
        table[..pat[0].len()].copy_from_slice(&pat[0]);
        table[256..256 + pat[1].len()].copy_from_slice(&pat[1]);

        let (stage, stage_ptr) = host
            .upload_buffer(2048)
            .ok_or_else(windows::core::Error::empty)?;
        // SAFETY: `stage_ptr` is a valid 2048-byte mapped upload buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(table.as_ptr().cast::<u8>(), stage_ptr, 2048);
        }

        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 2048,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let mut lut_buf: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap buffer creation and one-shot upload on
        // `self.list`, waited to completion before `stage` drops.
        let gpu = unsafe {
            host.device().CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut lut_buf,
            )?;
            let lut_buf = lut_buf.ok_or_else(windows::core::Error::empty)?;
            crate::host::name(&lut_buf, || {
                format!("gpu detile lut sw{} e{}", key.0, key.1)
            });
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list.ResourceBarrier(&[transition(
                &lut_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            self.list.CopyBufferRegion(&lut_buf, 0, &stage, 0, 2048);
            self.list.ResourceBarrier(&[transition(
                &lut_buf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            self.execute_and_wait()?;
            let gpu = lut_buf.GetGPUVirtualAddress();
            self.luts.insert(
                key,
                LutEntry {
                    _buffer: lut_buf,
                    gpu,
                },
            );
            gpu
        };
        drop(stage);
        Ok(gpu)
    }

    /// Copies raw `tiled` bytes sequentially into a mapped `UPLOAD` buffer and
    /// returns `(gpu_va, optional_temp_owner)`.
    fn stage_raw(
        &mut self,
        host: &Host,
        tiled: &[u8],
    ) -> windows::core::Result<(u64, Option<ID3D12Resource>)> {
        let len = tiled.len() as u64;
        let need = len.max(4).next_multiple_of(256);
        let (ptr, gpu, temp) = if need <= RAW_UPLOAD_KEEP_MAX {
            let grow = self.raw_upload.as_ref().is_none_or(|r| r.cap < need);
            if grow {
                let cap = need.next_power_of_two().clamp(1 << 20, RAW_UPLOAD_KEEP_MAX);
                let (buffer, data) = host
                    .upload_buffer(cap)
                    .ok_or_else(windows::core::Error::empty)?;
                crate::host::name(&buffer, || format!("gpu detile raw upload {cap:#x}"));
                self.raw_upload = Some(RawUpload { buffer, data, cap });
            }
            let raw = self.raw_upload.as_ref().expect("raw_upload ensured above");
            // SAFETY: live upload buffer.
            let gpu = unsafe { raw.buffer.GetGPUVirtualAddress() };
            (raw.data, gpu, None)
        } else {
            let (buffer, data) = host
                .upload_buffer(need)
                .ok_or_else(windows::core::Error::empty)?;
            crate::host::name(&buffer, || format!("gpu detile raw temp {need:#x}"));
            // SAFETY: live upload buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            (data, gpu, Some(buffer))
        };
        // SAFETY: `ptr` holds at least `need >= tiled.len().next_multiple_of(4)`
        // mapped bytes, exclusively owned by `ENGINE`'s mutex and idle on GPU.
        unsafe {
            std::ptr::copy_nonoverlapping(tiled.as_ptr(), ptr, tiled.len());
            let rem = tiled.len() & 3;
            if rem != 0 {
                std::ptr::write_bytes(ptr.add(tiled.len()), 0, 4 - rem);
            }
        }
        Ok((gpu, temp))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_detile(
        &mut self,
        host: &Host,
        dst_buffer: &ID3D12Resource,
        dst_base: u64,
        dst_total: u64,
        layout: &Layout,
        tiled: &[u8],
        footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
        rows: &[u32],
        row_sizes: &[u64],
        first: u32,
        mips: u32,
        held: u32,
    ) -> windows::core::Result<()> {
        let mips_per_slice = mips - first;
        let mut has_held = false;
        let mut has_unheld = false;
        for i in 0..footprints.len() {
            let mip = first + (i as u32 % mips_per_slice);
            if mip < 32 && held & (1 << mip) == 0 {
                has_unheld = true;
            } else {
                has_held = true;
            }
        }

        let (pat_gpu, raw_gpu, _temp_raw) = if has_held {
            let pat_gpu = self.pattern_lut(host, layout)?;
            let (raw_gpu, temp) = self.stage_raw(host, tiled)?;
            (pat_gpu, raw_gpu, temp)
        } else {
            (0, 0, None)
        };

        let elem_bytes = layout.desc.element_bytes;
        let block_log2 = layout.block_log2();
        let [bw, bh, _] = layout.block;
        let bw_log2 = bw.trailing_zeros();
        let bh_log2 = bh.trailing_zeros();
        let bw_mask = bw - 1;
        let bh_mask = bh - 1;

        // SAFETY: `self.allocator` and `self.list` are idle; `dst_buffer`
        // starts and ends in `COMMON`; all GPU VAs and dispatches are verified
        // in-bounds by `eligible()`.
        unsafe {
            let dst_gpu = dst_buffer.GetGPUVirtualAddress() + dst_base;
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list.ResourceBarrier(&[transition(
                dst_buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);

            if has_held {
                self.list.SetComputeRootSignature(&self.detile_root);
                self.list.SetPipelineState(&self.detile_pso);
                self.list.SetComputeRootShaderResourceView(1, raw_gpu);
                self.list.SetComputeRootShaderResourceView(2, pat_gpu);
                self.list.SetComputeRootUnorderedAccessView(3, dst_gpu);

                for (i, fp) in footprints.iter().enumerate() {
                    let mip = first + (i as u32 % mips_per_slice);
                    if mip < 32 && held & (1 << mip) == 0 {
                        continue;
                    }
                    let slice = i as u32 / mips_per_slice;
                    let m = &layout.mips[mip as usize];
                    let src_base = (u64::from(slice) * layout.slice_size + m.offset) as u32;
                    let dst_off = fp.Offset as u32;
                    let pitch_blocks = m.pitch / bw;
                    let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));
                    let constants: [u32; 16] = [
                        src_base,
                        dst_off,
                        m.width,
                        m.height,
                        fp.Footprint.RowPitch,
                        elem_bytes,
                        block_log2,
                        bw_log2,
                        bh_log2,
                        bw_mask,
                        bh_mask,
                        pitch_blocks,
                        tail_x,
                        tail_y,
                        0,
                        0,
                    ];
                    self.list.SetComputeRoot32BitConstants(
                        0,
                        16,
                        constants.as_ptr().cast(),
                        0,
                    );
                    let cols = match elem_bytes {
                        1 => m.width.div_ceil(4),
                        2 => m.width.div_ceil(2),
                        _ => m.width,
                    };
                    self.list
                        .Dispatch(cols.div_ceil(8), m.height.div_ceil(8), 1);
                }
            }

            if has_unheld {
                self.list.SetComputeRootSignature(&self.zero_root);
                self.list.SetPipelineState(&self.zero_pso);
                self.list.SetComputeRootUnorderedAccessView(1, dst_gpu);
                for (i, ((fp, &sub_rows), &row_size)) in footprints
                    .iter()
                    .zip(rows)
                    .zip(row_sizes)
                    .enumerate()
                {
                    let mip = first + (i as u32 % mips_per_slice);
                    if mip >= 32 || held & (1 << mip) != 0 {
                        continue;
                    }
                    let span = subresource_span(
                        fp.Offset,
                        fp.Footprint.RowPitch,
                        sub_rows,
                        row_size,
                        dst_total,
                    );
                    let dwords = (span as u32).div_ceil(4);
                    if dwords == 0 {
                        continue;
                    }
                    let groups = (dwords >> 2).max(1).div_ceil(64).min(65535);
                    let constants: [u32; 4] = [fp.Offset as u32, dwords, groups, 0];
                    self.list
                        .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                    self.list.Dispatch(groups, 1, 1);
                }
            }

            self.list.ResourceBarrier(&[transition(
                dst_buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
            self.execute_and_wait()?;
        }
        self.last_used_ms = crate::host::now_ms();
        Ok(())
    }

    fn run_zero(
        &mut self,
        dst_buffer: &ID3D12Resource,
        dst_base: u64,
        dst_total: u64,
    ) -> windows::core::Result<()> {
        let dwords = (dst_total as u32).div_ceil(4);
        if dwords == 0 {
            return Ok(());
        }
        let groups = (dwords >> 2).max(1).div_ceil(64).min(65535);
        let constants: [u32; 4] = [0, dwords, groups, 0];
        // SAFETY: `self.allocator` and `self.list` are idle; `dst_buffer`
        // starts and ends in `COMMON`.
        unsafe {
            let dst_gpu = dst_buffer.GetGPUVirtualAddress() + dst_base;
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list.ResourceBarrier(&[transition(
                dst_buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            self.list.SetComputeRootSignature(&self.zero_root);
            self.list.SetPipelineState(&self.zero_pso);
            self.list
                .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            self.list.SetComputeRootUnorderedAccessView(1, dst_gpu);
            self.list.Dispatch(groups, 1, 1);
            self.list.ResourceBarrier(&[transition(
                dst_buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
            self.execute_and_wait()?;
        }
        self.last_used_ms = crate::host::now_ms();
        Ok(())
    }
}

/// GPU-detiles `tiled` into `dst_buffer` (a `DEFAULT` UAV buffer resting in
/// `COMMON`, returned to `COMMON` before returning). Returns `true` on success.
#[allow(clippy::too_many_arguments)]
pub(crate) fn detile(
    host: &Host,
    dst_buffer: &ID3D12Resource,
    dst_base: u64,
    dst_total: u64,
    layout: &Layout,
    tiled: &[u8],
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    row_sizes: &[u64],
    first: u32,
    mips: u32,
    held: u32,
) -> bool {
    let Some(mutex) = engine(host) else {
        return false;
    };
    let started = std::time::Instant::now();
    let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
    match eng.run_detile(
        host,
        dst_buffer,
        dst_base,
        dst_total,
        layout,
        tiled,
        footprints,
        rows,
        row_sizes,
        first,
        mips,
        held,
    ) {
        Ok(()) => {
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(tiled.len() as u64, Ordering::Relaxed);
            STATS[2].fetch_add(dst_total, Ordering::Relaxed);
            STATS[3].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            true
        }
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(error = %e, "gpu compute detile dispatch failed");
            });
            false
        }
    }
}

/// Zero-fills `dst_total` bytes of a recycled VRAM linear buffer on the GPU.
pub(crate) fn zero_buffer(
    host: &Host,
    dst_buffer: &ID3D12Resource,
    dst_base: u64,
    dst_total: u64,
) -> bool {
    let Some(mutex) = engine(host) else {
        return false;
    };
    let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
    match eng.run_zero(dst_buffer, dst_base, dst_total) {
        Ok(()) => true,
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(error = %e, "gpu compute detile zero-fill failed");
            });
            false
        }
    }
}

/// Telemetry summary line per frame for merge hookup in `host.rs`.
#[allow(dead_code)]
pub(crate) fn line(frames: f64) {
    let frames = frames.max(1.0);
    let gpu_tex = STATS[0].swap(0, Ordering::Relaxed);
    let gpu_raw_bytes = STATS[1].swap(0, Ordering::Relaxed);
    let gpu_lin_bytes = STATS[2].swap(0, Ordering::Relaxed);
    let gpu_us = STATS[3].swap(0, Ordering::Relaxed);
    let cpu_tex = STATS[4].swap(0, Ordering::Relaxed);
    let fallback = STATS[5].swap(0, Ordering::Relaxed);
    if gpu_tex == 0 && cpu_tex == 0 && fallback == 0 {
        return;
    }
    let active_mb = ACTIVE_VRAM_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    let spare_mb = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner()).1 as f64 / (1024.0 * 1024.0);
    tracing::info!(
        "gpu detile per frame (VICTORY_GPU_DETILE=0 turns it off): {:.1} textures ({:.1} MB tiled -> {:.1} MB linear) \
         in {:.2} ms; {:.1} CPU-detiled ({:.1} VRAM-cap/init fallbacks); VRAM staging {:.1} MB active, {:.1} MB spare",
        gpu_tex as f64 / frames,
        gpu_raw_bytes as f64 / (1024.0 * 1024.0) / frames,
        gpu_lin_bytes as f64 / (1024.0 * 1024.0) / frames,
        gpu_us as f64 / 1000.0 / frames,
        cpu_tex as f64 / frames,
        fallback as f64 / frames,
        active_mb,
        spare_mb,
    );
}

#[cfg(test)]
pub(crate) struct OverrideGuard(u8);

#[cfg(test)]
impl OverrideGuard {
    pub(crate) fn set(on: bool) -> Self {
        let prev = TEST_OVERRIDE.swap(if on { 2 } else { 1 }, Ordering::SeqCst);
        Self(prev)
    }
}

#[cfg(test)]
impl Drop for OverrideGuard {
    fn drop(&mut self) {
        TEST_OVERRIDE.store(self.0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::Desc;
    use windows::Win32::Graphics::Dxgi::Common::*;

    fn dxgi_for_element(elem_bytes: u32, block: u32) -> DXGI_FORMAT {
        match (elem_bytes, block) {
            (1, 1) => DXGI_FORMAT_R8_UNORM,
            (2, 1) => DXGI_FORMAT_R16_UNORM,
            (4, 1) => DXGI_FORMAT_R8G8B8A8_UNORM,
            (8, 1) => DXGI_FORMAT_R16G16B16A16_UNORM,
            (16, 1) => DXGI_FORMAT_R32G32B32A32_UINT,
            (8, 4) => DXGI_FORMAT_BC1_UNORM,
            (16, 4) => DXGI_FORMAT_BC7_UNORM,
            _ => unreachable!(),
        }
    }

    fn readback_buffer(host: &Host, bytes: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes.max(4),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            host.device()
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut res,
                )
                .unwrap();
        }
        res.unwrap()
    }

    fn fetch_vram(host: &Host, vram: &ID3D12Resource, bytes: usize) -> Vec<u8> {
        let rb = readback_buffer(host, bytes as u64);
        let list = host.new_list().unwrap();
        // SAFETY: `vram` rests in `COMMON` (implicitly promotes to `COPY_SOURCE`),
        // `rb` is in `COPY_DEST`.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, vram, 0, bytes as u64);
        }
        list.close();
        host.execute(&[&list]);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: readback completed via `host.execute`.
        unsafe {
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let out = std::slice::from_raw_parts(ptr.cast::<u8>(), bytes).to_vec();
            rb.Unmap(0, None);
            out
        }
    }

    fn assert_no_debug_errors(host: &Host, context: &str) {
        if let Some(err) = host.log_messages() {
            panic!("D3D12 debug layer error during {context}: {err}");
        }
    }

    fn verify_case(
        host: &Host,
        desc: Desc,
        first: u32,
        held: u32,
        label: &str,
    ) {
        let layout = Layout::new(&desc).unwrap();
        let raw_len = layout.bytes_from(first) as usize;
        // Deterministic pseudo-random tiled bytes so every byte position is unique.
        let tiled: Vec<u8> = (0..raw_len)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add((desc.swizzle.0 as u64) << 17)
                    .wrapping_add(u64::from(desc.element_bytes) << 9);
                ((x >> 24) ^ (x >> 40) ^ (i as u64)) as u8
            })
            .collect();

        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: if desc.dimension == Dimension::Tex1d {
                D3D12_RESOURCE_DIMENSION_TEXTURE1D
            } else {
                D3D12_RESOURCE_DIMENSION_TEXTURE2D
            },
            Width: u64::from(desc.width),
            Height: desc.height,
            DepthOrArraySize: desc.depth as u16,
            MipLevels: desc.mips as u16,
            Format: dxgi_for_element(desc.element_bytes, desc.block),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let count = desc.mips * desc.depth - first;
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
        let mut rows = vec![0u32; count as usize];
        let mut row_sizes = vec![0u64; count as usize];
        let mut total = 0u64;
        // SAFETY: plain footprint query into our slices.
        unsafe {
            host.device().GetCopyableFootprints(
                &tex_desc,
                first,
                count,
                0,
                Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                Some(row_sizes.as_mut_ptr()),
                Some(&mut total),
            );
        }

        // Reference CPU detile into a zeroed linear buffer of `total` bytes.
        let total_aligned = total.next_multiple_of(4) as usize;
        let mut cpu_linear = vec![0u8; total_aligned];
        let mips_per_slice = desc.mips - first;
        for (i, ((fp, &sub_rows), &row_size)) in footprints
            .iter()
            .zip(&rows)
            .zip(&row_sizes)
            .enumerate()
        {
            let mip = first + (i as u32 % mips_per_slice);
            let slice = i as u32 / mips_per_slice;
            let pitch = fp.Footprint.RowPitch as usize;
            let span = subresource_span(fp.Offset, fp.Footprint.RowPitch, sub_rows, row_size, total);
            let dst = &mut cpu_linear[fp.Offset as usize..fp.Offset as usize + span];
            if mip < 32 && held & (1 << mip) == 0 {
                dst.fill(0);
            } else {
                layout
                    .detile(&tiled, mip, slice, dst, pitch, pitch * sub_rows as usize)
                    .unwrap();
            }
        }

        // Test GPU detile (directly through `detile()`, even if raw_len < MIN_BYTES
        // so small odd dimensions like 13x7 are bit-exact verified too!).
        // Zero-initialize `vram` first because WARP can recycle unzeroed
        // default-heap pages in unused row-pitch padding bytes.
        let vram = host.uav_buffer(total_aligned.max(512) as u64).unwrap();
        assert!(zero_buffer(host, &vram, 0, total_aligned as u64));
        let ok = detile(
            host,
            &vram,
            0,
            total,
            &layout,
            &tiled,
            &footprints,
            &rows,
            &row_sizes,
            first,
            desc.mips,
            held,
        );
        assert!(ok, "gpu_detile::detile failed for {label}");
        assert_no_debug_errors(host, label);

        let gpu_linear = fetch_vram(host, &vram, total_aligned);
        assert_no_debug_errors(host, label);

        // Compare every active byte of every subresource AND verify unheld mips are 0.
        for (i, ((fp, &sub_rows), &row_size)) in footprints
            .iter()
            .zip(&rows)
            .zip(&row_sizes)
            .enumerate()
        {
            let mip = first + (i as u32 % mips_per_slice);
            let slice = i as u32 / mips_per_slice;
            let pitch = fp.Footprint.RowPitch as usize;
            let base = fp.Offset as usize;
            let active = row_size as usize;
            for y in 0..sub_rows as usize {
                let row_start = base + y * pitch;
                let gpu_row = &gpu_linear[row_start..row_start + active];
                let cpu_row = &cpu_linear[row_start..row_start + active];
                if gpu_row != cpu_row {
                    let pos = gpu_row
                        .iter()
                        .zip(cpu_row)
                        .position(|(a, b)| a != b)
                        .unwrap();
                    panic!(
                        "mismatch in {label} slice={slice} mip={mip} y={y} byte={pos}: \
                         gpu={:#04x} cpu={:#04x}",
                        gpu_row[pos], cpu_row[pos]
                    );
                }
            }
        }
        // Full-buffer equality (zeroed VRAM buffer + 0-padded trailing words).
        if gpu_linear != cpu_linear {
            let pos = gpu_linear
                .iter()
                .zip(&cpu_linear)
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "full-buffer mismatch in {label} at byte {pos}: gpu={:#04x} cpu={:#04x}",
                gpu_linear[pos], cpu_linear[pos]
            );
        }
    }

    #[test]
    fn toggle_defaults_on_and_zero_disables() {
        assert!(enabled_from(None));
        assert!(enabled_from(Some("1")));
        assert!(enabled_from(Some("yes")));
        assert!(!enabled_from(Some("0")));
        assert!(!enabled_from(Some(" 0 ")));
    }

    #[test]
    fn eligibility_filters_linear_3d_msaa_and_tiny_textures() {
        let base = Desc {
            dimension: Dimension::Tex2d,
            width: 128,
            height: 128,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        };
        let ok = Layout::new(&base).unwrap();
        assert!(eligible_layout(&ok, ok.size as usize));

        // Linear rejected.
        let lin = Layout::new(&Desc { swizzle: Swizzle::LINEAR, ..base }).unwrap();
        assert!(!eligible_layout(&lin, lin.size as usize));

        // XOR (non-addressable) rejected.
        let xor = Layout::new(&Desc { swizzle: Swizzle::SW_64KB_D_X, ..base }).unwrap();
        assert!(!eligible_layout(&xor, xor.size as usize));

        // 3D rejected.
        let tex3d = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 8,
            swizzle: Swizzle::SW_4KB_S,
            ..base
        })
        .unwrap();
        assert!(!eligible_layout(&tex3d, tex3d.size as usize));

        // MSAA rejected.
        let msaa = Layout::new(&Desc { samples: 4, ..base }).unwrap();
        assert!(!eligible_layout(&msaa, msaa.size as usize));

        // Tiny (< 16 KiB) rejected.
        let tiny = Layout::new(&Desc {
            width: 16,
            height: 16,
            swizzle: Swizzle::SW_256B_S,
            ..base
        })
        .unwrap();
        assert!((tiny.size as usize) < MIN_BYTES);
        assert!(!eligible_layout(&tiny, tiny.size as usize));
    }

    /// 100% byte-for-byte verification across all 5 element sizes (1, 2, 4, 8,
    /// 16 bytes) and all 6 addressable non-linear swizzle modes (`SW_256B_S/D`,
    /// `SW_4KB_S/D`, `SW_64KB_S/D`), with full mip chains including mip tails.
    #[test]
    fn bit_exact_all_element_sizes_and_swizzles_with_mip_tails() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let swizzles = [
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        // (element_bytes, block, width, height, mips)
        let formats = [
            (1u32, 1u32, 131u32, 79u32, 8u32),  // R8, odd dims + tail
            (2, 1, 97, 63, 7),                  // R16, odd dims + tail
            (4, 1, 128, 96, 8),                 // RGBA8 + tail
            (8, 4, 256, 192, 9),                // BC1 (64x48 blocks) + tail
            (16, 4, 256, 256, 9),               // BC7 (64x64 blocks) + tail
        ];
        for swizzle in swizzles {
            for (elem_bytes, block, width, height, mips) in formats {
                let desc = Desc {
                    dimension: Dimension::Tex2d,
                    width,
                    height,
                    depth: 1,
                    mips,
                    element_bytes: elem_bytes,
                    block,
                    samples: 1,
                    swizzle,
                };
                let label = format!("{swizzle:?} e{elem_bytes} b{block} {width}x{height} m{mips}");
                verify_case(host, desc, 0, u32::MAX, &label);
            }
        }
    }

    /// Verifies NPOT dimensions (`300x173`, `13x7`, `513x257`), multi-slice 2D
    /// arrays (`depth=3`), `bytes_from(first)` partial mip chains, and
    /// `on_demand` partial `held` masks + GPU zero-fill on recycled buffers.
    #[test]
    fn bit_exact_npot_arrays_partial_chains_and_on_demand_held() {
        let Some(host) = crate::host::host() else {
            return;
        };
        // NPOT dimensions across multiple formats & swizzles.
        let npot_cases = [
            (300u32, 173u32, 9u32, 4u32, 1u32, Swizzle::SW_64KB_D),
            (300, 173, 9, 1, 1, Swizzle::SW_4KB_S),
            (300, 173, 9, 2, 1, Swizzle::SW_4KB_D),
            (300, 173, 9, 16, 4, Swizzle::SW_64KB_S),
            (13, 7, 4, 1, 1, Swizzle::SW_4KB_D),
            (13, 7, 4, 2, 1, Swizzle::SW_64KB_S),
            (13, 7, 4, 4, 1, Swizzle::SW_256B_D),
            (513, 257, 10, 4, 1, Swizzle::SW_64KB_D),
            (513, 257, 10, 8, 4, Swizzle::SW_4KB_S),
        ];
        for (w, h, mips, elem, block, swizzle) in npot_cases {
            let desc = Desc {
                dimension: Dimension::Tex2d,
                width: w,
                height: h,
                depth: 1,
                mips,
                element_bytes: elem,
                block,
                samples: 1,
                swizzle,
            };
            verify_case(
                host,
                desc,
                0,
                u32::MAX,
                &format!("npot {w}x{h} e{elem} b{block} {swizzle:?}"),
            );
        }

        // Multi-slice 2D array (depth = 3), full mip chain with mip tail.
        for (elem, block, swizzle) in [
            (1u32, 1u32, Swizzle::SW_4KB_S),
            (4, 1, Swizzle::SW_64KB_D),
            (16, 4, Swizzle::SW_64KB_S),
        ] {
            let desc = Desc {
                dimension: Dimension::Tex2d,
                width: 132,
                height: 76,
                depth: 3,
                mips: 7,
                element_bytes: elem,
                block,
                samples: 1,
                swizzle,
            };
            verify_case(
                host,
                desc,
                0,
                u32::MAX,
                &format!("array3 e{elem} b{block} {swizzle:?}"),
            );
        }

        // `bytes_from(first)` partial mip chains (first = 1, 2, 3, and inside mip tail).
        let chain = Desc {
            dimension: Dimension::Tex2d,
            width: 512,
            height: 512,
            depth: 1,
            mips: 10,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_4KB_S,
        };
        let chain_layout = Layout::new(&chain).unwrap();
        assert!(chain_layout.first_tail_mip < chain.mips);
        for first in [1u32, 2, 3, chain_layout.first_tail_mip, chain.mips - 1] {
            verify_case(host, chain, first, u32::MAX, &format!("bytes_from({first})"));
        }

        // `on_demand` held mask: only mips 2..10 held (mips 0 and 1 must be zeroed!).
        let held_mask = !0b11u32;
        verify_case(host, chain, 0, held_mask, "on_demand_held_mask");

        // Zero-fill on a non-zero VRAM buffer: detile into it first, then zero
        // unheld mips and finally zero the whole buffer.
        let vram = host.uav_buffer(64 << 10).unwrap();
        // Fill with non-zero pattern via detile, then zero_buffer.
        let small_desc = Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_4KB_D,
        };
        let small_layout = Layout::new(&small_desc).unwrap();
        let ones = vec![0xA5u8; small_layout.size as usize];
        let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Width: 64,
                Height: 64,
                Depth: 1,
                RowPitch: 256,
            },
        };
        assert!(detile(
            host,
            &vram,
            0,
            16384,
            &small_layout,
            &ones,
            &[fp],
            &[64],
            &[256],
            0,
            1,
            u32::MAX,
        ));
        let filled = fetch_vram(host, &vram, 16384);
        assert!(filled.iter().all(|&b| b == 0xA5));

        assert!(zero_buffer(host, &vram, 0, 16384));
        assert_no_debug_errors(host, "zero_buffer");
        let cleared = fetch_vram(host, &vram, 16384);
        assert!(cleared.iter().all(|&b| b == 0), "zero_buffer must clear all bytes");
    }
}
