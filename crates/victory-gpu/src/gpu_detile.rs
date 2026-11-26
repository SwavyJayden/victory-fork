//! GPU compute texture detiling (`cs_5_1`) for streamed console textures.
//!
//! Streamed 1D/2D/2D-array textures arrive in title memory in GFX10.3 tiled
//! layouts (`SW_256B_*`, `SW_4KB_*`, `SW_64KB_*`). Detiling on the CPU scatters
//! small 8/16/32-byte runs into write-combined `D3D12_HEAP_TYPE_UPLOAD` staging
//! memory. When eligible (`VICTORY_GPU_DETILE`, on unless `0`), this module:
//!
//! 1. Keeps each texture's linear staging buffer in a pooled `DEFAULT`-heap
//!    VRAM buffer (`ALLOW_UNORDERED_ACCESS`, resting in `COMMON`), so
//!    `CopyTextureRegion` recorded on the title's command list or copy queue
//!    reads directly from VRAM without touching pipeline state or descriptor
//!    heaps.
//! 2. At submission/copy-queue refresh time, copies the raw tiled bytes
//!    sequentially with a single `copy_nonoverlapping` into a 256-byte-aligned
//!    slice of a pooled `UPLOAD` ring slab (outside the compute-list mutex) and
//!    records `cs_5_1` detile dispatches onto a dedicated compute queue
//!    (falling back to a direct queue if needed) using direct root SRVs and a
//!    direct root UAV (zero descriptor heaps).
//! 3. Supports isolated concurrent async batches:
//!    - [`BatchGuard::begin`] opens the submission batch (`Host::submit` and its
//!      `wide_each` worker threads) and flushes via `direct_queue.Wait`.
//!    - [`BatchGuard::begin_thread`] opens a thread-private batch (for
//!      `texture::ahead` copy-queue workers) and flushes via `copy_queue.Wait`.
//!    - Outside any batch, calls fall back cleanly to synchronous submit+wait.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_formats::texture::{Dimension, Layout, Swizzle};
use windows::Win32::Foundation::{E_FAIL, HANDLE, WAIT_FAILED, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, s};

use crate::host::{Host, transition};

/// Minimum tiled size eligible for GPU detiling; in async batched mode
/// per-texture dispatch overhead is sub-microsecond (zero CPU fence waits),
/// so textures from 4 KiB upward stay on the GPU.
pub(crate) const MIN_BYTES: usize = 4 << 10;

/// Largest single VRAM linear staging buffer kept in [`VRAM_SPARE`].
const VRAM_SPARE_MAX_SINGLE: u64 = 16 << 20;
/// Maximum idle VRAM bytes kept in [`VRAM_SPARE`].
const VRAM_SPARE_BUDGET: u64 = 128 << 20;
/// Maximum idle VRAM bytes retained after an idle trim period.
const VRAM_SPARE_IDLE_KEEP: u64 = 32 << 20;
/// Standard size of one raw tiled `UPLOAD` slab in the async ring pool.
const RAW_SLAB_SIZE: u64 = 16 << 20;
/// Pooled raw `UPLOAD` ring bytes retained after idle trim.
const RAW_UPLOAD_IDLE_KEEP: u64 = 16 << 20;
/// Maximum raw tiled bytes queued into one open compute command list before
/// flushing mid-batch to bound transient upload memory.
const BATCH_MAX_RAW_BYTES: u64 = 256 << 20;
/// Maximum detile/zero operations queued into one open compute command list
/// before flushing mid-batch.
const BATCH_MAX_OPS: usize = 2048;

/// Batch ID reserved for `Host::submit` (shared across its `wide_each` workers).
const SUBMIT_BATCH_ID: u64 = 1;

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
/// 0 = GPU-detiled textures,
/// 1 = raw tiled bytes,
/// 2 = linear bytes,
/// 3 = GPU detile CPU wall microseconds (copy + record + flush),
/// 4 = CPU-detiled textures,
/// 5 = VRAM-cap / engine fallbacks from eligible textures,
/// 6 = async batched compute flushes (zero CPU fence waits),
/// 7 = immediate synchronous compute fence waits (outside batch).
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Committed VRAM bytes currently held by live `Staging` entries.
static ACTIVE_VRAM_BYTES: AtomicU64 = AtomicU64::new(0);

/// Whether the submission-wide compute detile batch (`SUBMIT_BATCH_ID`) is open.
static SUBMIT_BATCH_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Monotonic generator for thread-private batch IDs (`>= 2`).
static NEXT_BATCH_ID: AtomicU64 = AtomicU64::new(2);

thread_local! {
    /// Non-zero while this thread owns an isolated compute batch (e.g. an
    /// `uploads ahead` copy-queue worker). Takes precedence over
    /// [`SUBMIT_BATCH_ACTIVE`] so background workers never mix work into
    /// `Host::submit`'s open batch.
    static THREAD_BATCH_ID: Cell<u64> = const { Cell::new(0) };
}

fn active_batch_id() -> u64 {
    let own = THREAD_BATCH_ID.get();
    if own != 0 {
        own
    } else if SUBMIT_BATCH_ACTIVE.load(Ordering::Acquire) {
        SUBMIT_BATCH_ID
    } else {
        0
    }
}

/// Highest compute fence value that any currently recorded or submitted GPU
/// compute detile/zero work will signal. `release_vram_buf` stamps recycled
/// VRAM buffers with this value so `take_vram_buf` never hands out a buffer
/// while an in-flight compute batch is still writing to it.
static COMPUTE_SAFE_FENCE: AtomicU64 = AtomicU64::new(0);

struct ComputeFenceRef(ID3D12Fence);
// SAFETY: `ID3D12Fence` is free-threaded; `GetCompletedValue` is lock-free.
unsafe impl Send for ComputeFenceRef {}
unsafe impl Sync for ComputeFenceRef {}

static COMPUTE_FENCE: OnceLock<ComputeFenceRef> = OnceLock::new();

fn compute_completed_fence() -> u64 {
    let Some(f) = COMPUTE_FENCE.get() else {
        return 0;
    };
    // SAFETY: plain lock-free fence query on a live fence.
    let v = unsafe { f.0.GetCompletedValue() };
    if v == u64::MAX { 0 } else { v }
}

/// Pooled idle VRAM linear staging buffer tagged with the compute fence value
/// after which it is safe to reuse.
struct VramSpare {
    buffer: ID3D12Resource,
    until_fence: u64,
}
// SAFETY: `ID3D12Resource` is free-threaded; entries are only touched under
// `VRAM_SPARE`'s mutex and only reused once `compute_fence >= until_fence`.
unsafe impl Send for VramSpare {}

static VRAM_SPARE: Mutex<(BTreeMap<u64, VecDeque<VramSpare>>, u64)> =
    Mutex::new((BTreeMap::new(), 0));
static VRAM_SPARE_USED_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
static TEST_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Maximum pooled raw `UPLOAD` ring bytes retained between bursts, scaled to
/// the host's RAM tier (64 MiB on <=32 GiB, 96 MiB on 48 GiB, 128 MiB on >=64 GiB).
fn raw_upload_keep_max() -> u64 {
    match crate::residency::host_ram_tier() {
        crate::residency::HostRamTier::Tight => 64 << 20,
        crate::residency::HostRamTier::Moderate => 96 << 20,
        crate::residency::HostRamTier::Plentiful => 128 << 20,
    }
}

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
/// adapter VRAM headroom: 1024 MiB with >12 GiB budget and >=2 GiB free,
/// 512 MiB with >=1.5 GiB free, 128 MiB otherwise).
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
    vram_active_budget_from(host.video_memory())
}

fn vram_active_budget_from(mem: Option<[u64; 4]>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    match mem {
        Some([budget, used, ..]) if budget > 0 => {
            let free = budget.saturating_sub(used);
            if budget > 12 * GIB && free >= 2 * GIB {
                1024 << 20
            } else if free >= (3 * GIB) / 2 {
                512 << 20
            } else {
                128 << 20
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
    let required_tiled = layout.bytes_from(first);
    if required_tiled == 0 || (tiled_len as u64) < required_tiled {
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

/// Active VRAM bytes currently held by GPU-detiled `Staging` entries.
pub(crate) fn active_vram_bytes() -> u64 {
    ACTIVE_VRAM_BYTES.load(Ordering::Relaxed)
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
        let completed = compute_completed_fence();
        let reused = {
            let mut spare = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner());
            let taken = spare.0.get_mut(&pooled).and_then(|q| {
                // Only reuse once the compute detile fence completed AND no
                // Direct/Copy command list or `in_flight` batch still holds a
                // COM reference (`references == 1` => sole owner in pool).
                let idx = q.iter().position(|s| {
                    s.until_fence <= completed && crate::host::references(&s.buffer) == 1
                })?;
                q.remove(idx)
            });
            if taken.is_some() {
                spare.1 -= pooled;
            }
            taken
        };
        if let Some(VramSpare { buffer, .. }) = reused {
            return Some((buffer, alloc_size, pooled));
        }
    }
    let buffer = match host.uav_buffer(alloc_size) {
        Some(buf) => buf,
        None => {
            evict_completed_spares(alloc_size);
            match host.uav_buffer(alloc_size) {
                Some(buf) => buf,
                None => {
                    ACTIVE_VRAM_BYTES.fetch_sub(alloc_size, Ordering::Relaxed);
                    note_fallback();
                    return None;
                }
            }
        }
    };
    crate::host::name(&buffer, || format!("gpu detile linear {alloc_size:#x}"));
    Some((buffer, alloc_size, pooled))
}

/// Evicts completed idle buffers from [`VRAM_SPARE`] (largest buckets first)
/// until at least `target_bytes` have been released back to the host.
fn evict_completed_spares(target_bytes: u64) {
    let completed = compute_completed_fence();
    let mut freed = 0u64;
    let mut retired = Vec::new();
    {
        let mut spare = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let (buckets, total) = &mut *spare;
        let sizes: Vec<u64> = buckets.keys().copied().rev().collect();
        for size in sizes {
            if freed >= target_bytes {
                break;
            }
            if let Some(bucket) = buckets.get_mut(&size) {
                while freed < target_bytes {
                    let Some(idx) = bucket.iter().position(|s| {
                        s.until_fence <= completed && crate::host::references(&s.buffer) == 1
                    }) else {
                        break;
                    };
                    if let Some(VramSpare { buffer, .. }) = bucket.remove(idx) {
                        *total -= size;
                        freed += size;
                        retired.push(buffer);
                    }
                }
                if bucket.is_empty() {
                    buckets.remove(&size);
                }
            }
        }
    }
    drop(retired);
}

/// Returns a VRAM linear staging buffer once its `Staging` drops (after all
/// direct/copy lists naming it have completed).
pub(crate) fn release_vram_buf(buffer: ID3D12Resource, alloc_size: u64, pooled: u64) {
    ACTIVE_VRAM_BYTES.fetch_sub(alloc_size, Ordering::Relaxed);
    if pooled == 0 {
        crate::host::retire(buffer);
        return;
    }
    let until_fence = COMPUTE_SAFE_FENCE.load(Ordering::Acquire);
    let mut spare = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner());
    if spare.1 + pooled <= VRAM_SPARE_BUDGET {
        spare.1 += pooled;
        spare
            .0
            .entry(pooled)
            .or_default()
            .push_back(VramSpare { buffer, until_fence });
    } else {
        drop(spare);
        crate::host::retire(buffer);
    }
}

/// Trims idle VRAM spare buffers and oversized raw upload ring slabs (called
/// from `texture::let_go_idle`).
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
                            let Some(VramSpare { buffer, .. }) = bucket.pop_front() else {
                                break;
                            };
                            *total -= size;
                            retired.push(buffer);
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
    trim_raw_pool_idle(idle_ms, now);
}

static NEXT_SLAB_ID: AtomicU64 = AtomicU64::new(1);
static RAW_POOL_BYTES: AtomicU64 = AtomicU64::new(0);

/// One persistently mapped `UPLOAD` slab for raw tiled texture inputs.
struct RawSlab {
    id: u64,
    _buffer: ID3D12Resource,
    data: *mut u8,
    gpu_base: u64,
    cap: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded; disjoint bump-allocated slices of
// the mapped `UPLOAD` buffer are written without overlap while an `Arc<RawSlab>`
// is held, and recycled only after `Arc::strong_count == 1`.
unsafe impl Send for RawSlab {}
unsafe impl Sync for RawSlab {}

struct RawSlice {
    slab: Arc<RawSlab>,
    ptr: *mut u8,
    gpu_va: u64,
    aligned_len: u64,
}

struct ActiveSlab {
    slab: Arc<RawSlab>,
    offset: u64,
}

struct RawUploadPool {
    active: Option<ActiveSlab>,
    free: Vec<Arc<RawSlab>>,
    retiring: Vec<Arc<RawSlab>>,
    pooled_bytes: u64,
    last_used_ms: u64,
}

impl RawUploadPool {
    const fn new() -> Self {
        Self {
            active: None,
            free: Vec::new(),
            retiring: Vec::new(),
            pooled_bytes: 0,
            last_used_ms: 0,
        }
    }

    fn recycle_ready(&mut self) {
        let keep_max = raw_upload_keep_max();
        let mut i = 0;
        while i < self.retiring.len() {
            if Arc::strong_count(&self.retiring[i]) == 1 {
                let slab = self.retiring.swap_remove(i);
                if self.pooled_bytes > keep_max {
                    self.pooled_bytes -= slab.cap;
                    drop(slab);
                } else {
                    self.free.push(slab);
                }
            } else {
                i += 1;
            }
        }
        RAW_POOL_BYTES.store(self.pooled_bytes, Ordering::Relaxed);
    }

    fn try_carve(&mut self, aligned: u64) -> Option<RawSlice> {
        if let Some(act) = &mut self.active {
            if act.offset > 0 && Arc::strong_count(&act.slab) == 1 {
                act.offset = 0;
            }
            if act.offset + aligned <= act.slab.cap {
                let off = act.offset;
                act.offset += aligned;
                return Some(RawSlice {
                    slab: Arc::clone(&act.slab),
                    // SAFETY: `off + aligned <= act.slab.cap`, within mapped upload memory.
                    ptr: unsafe { act.slab.data.add(off as usize) },
                    gpu_va: act.slab.gpu_base + off,
                    aligned_len: aligned,
                });
            }
            let old = self.active.take().expect("checked Some");
            if Arc::strong_count(&old.slab) == 1 {
                let slab = old.slab;
                let ptr = slab.data;
                let gpu_va = slab.gpu_base;
                let slice = RawSlice {
                    slab: Arc::clone(&slab),
                    ptr,
                    gpu_va,
                    aligned_len: aligned,
                };
                self.active = Some(ActiveSlab {
                    slab,
                    offset: aligned,
                });
                return Some(slice);
            }
            self.retiring.push(old.slab);
        }
        if let Some(slab) = self.free.pop() {
            let ptr = slab.data;
            let gpu_va = slab.gpu_base;
            let slice = RawSlice {
                slab: Arc::clone(&slab),
                ptr,
                gpu_va,
                aligned_len: aligned,
            };
            self.active = Some(ActiveSlab {
                slab,
                offset: aligned,
            });
            return Some(slice);
        }
        None
    }
}

static RAW_POOL: Mutex<RawUploadPool> = Mutex::new(RawUploadPool::new());

fn reclaim_engine_completed() {
    if let Some(Some(mutex)) = ENGINE.get() {
        let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
        eng.reclaim_completed();
    }
}

fn trim_raw_pool_idle(idle_ms: u64, now: u64) {
    if let Some(Some(mutex)) = ENGINE.get()
        && let Ok(mut eng) = mutex.try_lock()
    {
        eng.reclaim_completed();
    }
    let mut pool = RAW_POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.recycle_ready();
    if pool.last_used_ms == 0 || now.saturating_sub(pool.last_used_ms) < idle_ms {
        return;
    }
    while pool.pooled_bytes > RAW_UPLOAD_IDLE_KEEP {
        if let Some(slab) = pool.free.pop() {
            pool.pooled_bytes -= slab.cap;
            drop(slab);
        } else if pool
            .active
            .as_ref()
            .is_some_and(|a| Arc::strong_count(&a.slab) == 1)
        {
            let act = pool.active.take().expect("checked Some");
            pool.pooled_bytes -= act.slab.cap;
            drop(act);
        } else {
            break;
        }
    }
    RAW_POOL_BYTES.store(pool.pooled_bytes, Ordering::Relaxed);
}

fn alloc_dedicated_raw(host: &Host, aligned: u64) -> Option<RawSlice> {
    let (buffer, data) = host.upload_buffer(aligned)?;
    let id = NEXT_SLAB_ID.fetch_add(1, Ordering::Relaxed);
    crate::host::name(&buffer, || format!("gpu detile raw temp {aligned:#x}"));
    // SAFETY: live upload buffer.
    let gpu_base = unsafe { buffer.GetGPUVirtualAddress() };
    let slab = Arc::new(RawSlab {
        id,
        _buffer: buffer,
        data,
        gpu_base,
        cap: aligned,
    });
    Some(RawSlice {
        slab,
        ptr: data,
        gpu_va: gpu_base,
        aligned_len: aligned,
    })
}

/// Suballocates a 256-byte-aligned raw tiled `UPLOAD` slice from [`RAW_POOL`]
/// (or a dedicated one-shot buffer if larger than [`RAW_SLAB_SIZE`]).
fn alloc_raw_slice(host: &Host, len: usize) -> Option<RawSlice> {
    let aligned = (len as u64).max(4).next_multiple_of(256);
    if aligned > RAW_SLAB_SIZE {
        return alloc_dedicated_raw(host, aligned);
    }
    let mut tried_reclaim = false;
    loop {
        let should_reclaim = {
            let mut pool = RAW_POOL.lock().unwrap_or_else(|e| e.into_inner());
            pool.last_used_ms = crate::host::now_ms();
            pool.recycle_ready();
            if let Some(slice) = pool.try_carve(aligned) {
                return Some(slice);
            }
            !tried_reclaim && !pool.retiring.is_empty()
        };
        if should_reclaim {
            reclaim_engine_completed();
            tried_reclaim = true;
            continue;
        }
        break;
    }

    let Some((buffer, data)) = host.upload_buffer(RAW_SLAB_SIZE) else {
        return alloc_dedicated_raw(host, aligned);
    };
    let id = NEXT_SLAB_ID.fetch_add(1, Ordering::Relaxed);
    crate::host::name(&buffer, || format!("gpu detile raw slab {id}"));
    // SAFETY: live upload buffer.
    let gpu_base = unsafe { buffer.GetGPUVirtualAddress() };
    let slab = Arc::new(RawSlab {
        id,
        _buffer: buffer,
        data,
        gpu_base,
        cap: RAW_SLAB_SIZE,
    });

    let keep_max = raw_upload_keep_max();
    let mut pool = RAW_POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.last_used_ms = crate::host::now_ms();
    pool.recycle_ready();
    if let Some(slice) = pool.try_carve(aligned) {
        if pool.pooled_bytes + RAW_SLAB_SIZE <= keep_max {
            pool.pooled_bytes += RAW_SLAB_SIZE;
            pool.free.push(slab);
            RAW_POOL_BYTES.store(pool.pooled_bytes, Ordering::Relaxed);
        }
        return Some(slice);
    }
    if let Some(old) = pool.active.take() {
        if Arc::strong_count(&old.slab) == 1 {
            pool.free.push(old.slab);
        } else {
            pool.retiring.push(old.slab);
        }
    }
    pool.pooled_bytes += RAW_SLAB_SIZE;
    RAW_POOL_BYTES.store(pool.pooled_bytes, Ordering::Relaxed);
    let ptr = slab.data;
    let gpu_va = slab.gpu_base;
    let slice = RawSlice {
        slab: Arc::clone(&slab),
        ptr,
        gpu_va,
        aligned_len: aligned,
    };
    pool.active = Some(ActiveSlab {
        slab,
        offset: aligned,
    });
    Some(slice)
}

fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

struct LutEntry {
    _buffer: ID3D12Resource,
    gpu: u64,
}

struct ComputeCmdList {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

struct OpenBatch {
    cmd: ComputeCmdList,
    touched_dsts: HashMap<u64, ID3D12Resource>,
    used_slabs: HashMap<u64, Arc<RawSlab>>,
    keep_resources: Vec<ID3D12Resource>,
    raw_bytes: u64,
    ops: usize,
    bound_pso: u8,
    bound_lut: u64,
}

impl OpenBatch {
    fn new(cmd: ComputeCmdList) -> Self {
        Self {
            cmd,
            touched_dsts: HashMap::new(),
            used_slabs: HashMap::new(),
            keep_resources: Vec::new(),
            raw_bytes: 0,
            ops: 0,
            bound_pso: 0,
            bound_lut: 0,
        }
    }

    fn prepare_dst(&mut self, dst_buffer: &ID3D12Resource) -> u64 {
        // SAFETY: live D3D12 buffer; `self.cmd.list` is open.
        unsafe {
            let res_va = dst_buffer.GetGPUVirtualAddress();
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.touched_dsts.entry(res_va)
            {
                entry.insert(dst_buffer.clone());
                self.cmd.list.ResourceBarrier(&[transition(
                    dst_buffer,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            } else {
                self.cmd.list.ResourceBarrier(&[uav_barrier(dst_buffer)]);
            }
            res_va
        }
    }
}

#[derive(Default)]
struct BatchSlot {
    open: Option<OpenBatch>,
    wait_fence: u64,
}

struct InFlightBatch {
    until_fence: u64,
    cmd: ComputeCmdList,
    _used_slabs: Vec<Arc<RawSlab>>,
    _keep_resources: Vec<ID3D12Resource>,
}

struct DetileEngine {
    device: ID3D12Device,
    list_type: D3D12_COMMAND_LIST_TYPE,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fence_val: u64,
    event: HANDLE,
    free_lists: Vec<ComputeCmdList>,
    in_flight: VecDeque<InFlightBatch>,
    /// Concurrent isolated batch slots keyed by batch ID (`1` = `Host::submit`,
    /// `>= 2` = thread-private batches such as `texture::ahead` copy workers).
    batches: HashMap<u64, BatchSlot>,
    detile_root: ID3D12RootSignature,
    detile_pso: ID3D12PipelineState,
    zero_root: ID3D12RootSignature,
    zero_pso: ID3D12PipelineState,
    luts: HashMap<(u32, u32), LutEntry>,
    last_used_ms: u64,
}

// SAFETY: D3D12 objects are free-threaded; the event handle, command lists,
// allocators, batch state, and LUT cache are accessed exclusively under
// `ENGINE`'s mutex.
unsafe impl Send for DetileEngine {}

static ENGINE: OnceLock<Option<Mutex<DetileEngine>>> = OnceLock::new();

fn engine(host: &Host) -> Option<&'static Mutex<DetileEngine>> {
    ENGINE
        .get_or_init(|| match DetileEngine::new(host) {
            Ok(eng) => {
                let _ = COMPUTE_FENCE.set(ComputeFenceRef(eng.fence.clone()));
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

#[allow(clippy::too_many_arguments)]
fn record_detile_op(
    detile_root: &ID3D12RootSignature,
    detile_pso: &ID3D12PipelineState,
    zero_root: &ID3D12RootSignature,
    zero_pso: &ID3D12PipelineState,
    pat_gpu: u64,
    batch: &mut OpenBatch,
    dst_buffer: &ID3D12Resource,
    dst_base: u64,
    dst_total: u64,
    layout: &Layout,
    raw_slice: Option<RawSlice>,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    row_sizes: &[u64],
    first: u32,
    mips: u32,
    held: u32,
    has_held: bool,
    has_unheld: bool,
) -> windows::core::Result<()> {

    let mips_per_slice = mips - first;
    let elem_bytes = layout.desc.element_bytes;
    let block_log2 = layout.block_log2();
    let [bw, bh, _] = layout.block;
    let bw_log2 = bw.trailing_zeros();
    let bh_log2 = bh.trailing_zeros();
    let bw_mask = bw - 1;
    let bh_mask = bh - 1;

    let dst_res_va = batch.prepare_dst(dst_buffer);
    let dst_gpu = dst_res_va + dst_base;

    let mut raw_aligned = 0u64;
    // SAFETY: `batch.cmd.list` is open; `dst_buffer` is in `UNORDERED_ACCESS`;
    // all GPU VAs and dispatches are verified in-bounds by `eligible()`.
    unsafe {
        if has_held {
            let slice = raw_slice.expect("raw slice allocated when has_held");
            let raw_gpu = slice.gpu_va;
            raw_aligned = slice.aligned_len;
            batch
                .used_slabs
                .entry(slice.slab.id)
                .or_insert(slice.slab);

            if batch.bound_pso != 1 {
                batch.cmd.list.SetComputeRootSignature(detile_root);
                batch.cmd.list.SetPipelineState(detile_pso);
                batch.bound_pso = 1;
                batch.bound_lut = 0;
            }
            batch.cmd.list.SetComputeRootShaderResourceView(1, raw_gpu);
            if batch.bound_lut != pat_gpu {
                batch.cmd.list.SetComputeRootShaderResourceView(2, pat_gpu);
                batch.bound_lut = pat_gpu;
            }
            batch.cmd.list.SetComputeRootUnorderedAccessView(3, dst_gpu);

            for (i, fp) in footprints.iter().enumerate() {
                let mip = first + (i as u32 % mips_per_slice);
                if mip < 32 && held & (1 << mip) == 0 {
                    continue;
                }
                let slice_idx = i as u32 / mips_per_slice;
                let m = &layout.mips[mip as usize];
                let src_base = (u64::from(slice_idx) * layout.slice_size + m.offset) as u32;
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
                batch.cmd.list.SetComputeRoot32BitConstants(
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
                batch
                    .cmd
                    .list
                    .Dispatch(cols.div_ceil(8), m.height.div_ceil(8), 1);
            }
        }

        if has_unheld {
            if batch.bound_pso != 2 {
                batch.cmd.list.SetComputeRootSignature(zero_root);
                batch.cmd.list.SetPipelineState(zero_pso);
                batch.bound_pso = 2;
                batch.bound_lut = 0;
            }
            batch.cmd.list.SetComputeRootUnorderedAccessView(1, dst_gpu);
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
                batch
                    .cmd
                    .list
                    .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                batch.cmd.list.Dispatch(groups, 1, 1);
            }
        }
    }

    batch.ops += 1;
    batch.raw_bytes += raw_aligned;
    Ok(())
}

fn record_zero_op(
    zero_root: &ID3D12RootSignature,
    zero_pso: &ID3D12PipelineState,
    batch: &mut OpenBatch,
    dst_buffer: &ID3D12Resource,
    dst_base: u64,
    dst_total: u64,
) {
    let dwords = (dst_total as u32).div_ceil(4);
    if dwords == 0 {
        return;
    }
    let groups = (dwords >> 2).max(1).div_ceil(64).min(65535);
    let constants: [u32; 4] = [0, dwords, groups, 0];
    let dst_res_va = batch.prepare_dst(dst_buffer);
    let dst_gpu = dst_res_va + dst_base;
    // SAFETY: `batch.cmd.list` is open and `dst_buffer` is in `UNORDERED_ACCESS`.
    unsafe {
        if batch.bound_pso != 2 {
            batch.cmd.list.SetComputeRootSignature(zero_root);
            batch.cmd.list.SetPipelineState(zero_pso);
            batch.bound_pso = 2;
            batch.bound_lut = 0;
        }
        batch
            .cmd
            .list
            .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        batch.cmd.list.SetComputeRootUnorderedAccessView(1, dst_gpu);
        batch.cmd.list.Dispatch(groups, 1, 1);
    }
    batch.ops += 1;
}

impl DetileEngine {
    fn new(host: &Host) -> windows::core::Result<Self> {
        let detile_bc = crate::host::compile(DETILE_SOURCE, s!("detile_main"), s!("cs_5_1"))?;
        let zero_bc = crate::host::compile(ZERO_SOURCE, s!("zero_main"), s!("cs_5_1"))?;
        let device = host.device.clone();
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
            let mut free_lists = Vec::with_capacity(4);
            for _ in 0..3 {
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(list_type)?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, list_type, &allocator, None)?;
                list.Close()?;
                free_lists.push(ComputeCmdList { allocator, list });
            }
            Ok(Self {
                device,
                list_type,
                queue,
                fence,
                fence_val: 0,
                event,
                free_lists,
                in_flight: VecDeque::new(),
                batches: HashMap::new(),
                detile_root,
                detile_pso,
                zero_root,
                zero_pso,
                luts: HashMap::new(),
                last_used_ms: 0,
            })
        }
    }

    fn reclaim_completed(&mut self) {
        // SAFETY: plain fence query on `self.fence`.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            return;
        }
        while let Some(front) = self.in_flight.front() {
            if front.until_fence > completed {
                break;
            }
            let done = self.in_flight.pop_front().expect("checked front");
            self.free_lists.push(done.cmd);
        }
    }

    fn take_cmd_list(&mut self) -> windows::core::Result<ComputeCmdList> {
        self.reclaim_completed();
        // SAFETY: idle allocator/list reset or fresh creation on `self.device`.
        unsafe {
            if let Some(cmd) = self.free_lists.pop() {
                cmd.allocator.Reset()?;
                cmd.list.Reset(&cmd.allocator, None)?;
                return Ok(cmd);
            }
            let allocator: ID3D12CommandAllocator =
                self.device.CreateCommandAllocator(self.list_type)?;
            let list: ID3D12GraphicsCommandList =
                self.device
                    .CreateCommandList(0, self.list_type, &allocator, None)?;
            Ok(ComputeCmdList { allocator, list })
        }
    }

    fn wait_fence(&self, target: u64) -> windows::core::Result<()> {
        // SAFETY: `self.fence` and `self.event` belong to `self` and are
        // serialized by `ENGINE`'s mutex.
        unsafe {
            let completed = self.fence.GetCompletedValue();
            if completed == u64::MAX {
                return Err(windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED));
            }
            if completed < target {
                self.fence.SetEventOnCompletion(target, self.event)?;
                match WaitForSingleObject(self.event, INFINITE) {
                    WAIT_OBJECT_0 => {}
                    WAIT_FAILED => return Err(windows::core::Error::from_thread()),
                    _ => return Err(windows::core::Error::from_hresult(E_FAIL)),
                }
                let after = self.fence.GetCompletedValue();
                if after == u64::MAX || after < target {
                    return Err(windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED));
                }
            }
            Ok(())
        }
    }

    fn submit_batch_inner(
        &mut self,
        batch: OpenBatch,
        is_batched: bool,
    ) -> windows::core::Result<u64> {
        if batch.ops == 0 {
            // SAFETY: open empty list returned to `free_lists` in closed state.
            unsafe {
                batch.cmd.list.Close()?;
            }
            self.free_lists.push(batch.cmd);
            return Ok(0);
        }
        // SAFETY: `batch.cmd.list` is open; every resource in `touched_dsts`
        // is live and currently in `UNORDERED_ACCESS`.
        unsafe {
            if !batch.touched_dsts.is_empty() {
                let barriers: Vec<D3D12_RESOURCE_BARRIER> = batch
                    .touched_dsts
                    .values()
                    .map(|res| {
                        transition(
                            res,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COMMON,
                        )
                    })
                    .collect();
                batch.cmd.list.ResourceBarrier(&barriers);
            }
            batch.cmd.list.Close()?;
            let cmd_list: ID3D12CommandList = batch.cmd.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(cmd_list)]);
            self.fence_val += 1;
            self.queue.Signal(&self.fence, self.fence_val)?;
        }
        let val = self.fence_val;
        COMPUTE_SAFE_FENCE.fetch_max(val, Ordering::AcqRel);

        let mut keep = batch.keep_resources;
        keep.extend(batch.touched_dsts.into_values());
        let used_slabs: Vec<Arc<RawSlab>> = batch.used_slabs.into_values().collect();
        self.in_flight.push_back(InFlightBatch {
            until_fence: val,
            cmd: batch.cmd,
            _used_slabs: used_slabs,
            _keep_resources: keep,
        });
        self.reclaim_completed();
        self.last_used_ms = crate::host::now_ms();
        if is_batched {
            STATS[6].fetch_add(1, Ordering::Relaxed);
        }
        Ok(val)
    }

    /// Defensive hazard prevention: if any *other* open batch has already
    /// transitioned `dst_res_va` to `UNORDERED_ACCESS`, flush that other batch
    /// first so `dst_res_va` returns to `COMMON` before `target_batch_id`
    /// touches it.
    fn flush_conflicting_batches(
        &mut self,
        target_batch_id: u64,
        dst_res_va: u64,
    ) -> windows::core::Result<()> {
        let conflicts: Vec<u64> = self
            .batches
            .iter()
            .filter_map(|(&id, slot)| {
                (id != target_batch_id
                    && slot
                        .open
                        .as_ref()
                        .is_some_and(|b| b.touched_dsts.contains_key(&dst_res_va)))
                .then_some(id)
            })
            .collect();
        for id in conflicts {
            if let Some(slot) = self.batches.get_mut(&id)
                && let Some(open) = slot.open.take()
            {
                let val = self.submit_batch_inner(open, true)?;
                if val != 0
                    && let Some(slot) = self.batches.get_mut(&id)
                {
                    slot.wait_fence = slot.wait_fence.max(val);
                }
            }
        }
        Ok(())
    }

    fn flush_batch_to_queue(
        &mut self,
        batch_id: u64,
        target_queue: &ID3D12CommandQueue,
    ) -> windows::core::Result<bool> {
        let Some(slot) = self.batches.remove(&batch_id) else {
            self.reclaim_completed();
            return Ok(false);
        };
        let mut wait_val = slot.wait_fence;
        if let Some(batch) = slot.open {
            let val = self.submit_batch_inner(batch, true)?;
            if val != 0 {
                wait_val = wait_val.max(val);
            }
        }
        if wait_val == 0 {
            self.reclaim_completed();
            return Ok(false);
        }
        // SAFETY: `wait_val` was signalled on `self.queue`; `target_queue`
        // waits on the GPU hardware fence with zero CPU stall.
        unsafe {
            target_queue.Wait(&self.fence, wait_val)?;
        }
        Ok(true)
    }

    fn finish_and_wait_batch(&mut self, batch_id: u64) -> windows::core::Result<()> {
        let Some(slot) = self.batches.remove(&batch_id) else {
            self.reclaim_completed();
            return Ok(());
        };
        let mut wait_val = slot.wait_fence;
        if let Some(batch) = slot.open {
            let val = self.submit_batch_inner(batch, true)?;
            if val != 0 {
                wait_val = wait_val.max(val);
            }
        }
        if wait_val != 0 {
            self.wait_fence(wait_val)?;
        }
        self.reclaim_completed();
        Ok(())
    }

    /// Ensures the 2 KiB swizzle pattern LUT for `layout` is resident on the
    /// GPU before ANY open batch submits on `self.queue`. Uploading on an
    /// immediate one-shot compute command list submitted to `self.queue`
    /// guarantees FIFO ordering for every concurrent batch (`Host::submit`
    /// and `texture::ahead` copy-queue workers) with zero CPU stalls.
    fn ensure_pattern_lut(
        &mut self,
        host: &Host,
        layout: &Layout,
    ) -> windows::core::Result<u64> {
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
        let cmd = self.take_cmd_list()?;
        let mut lut_buf: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap buffer creation and immediate one-shot
        // upload submitted on `self.queue` ahead of every open detile batch.
        let (lut_buf, gpu) = unsafe {
            host.device.CreateCommittedResource(
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
            cmd.list.ResourceBarrier(&[transition(
                &lut_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            cmd.list.CopyBufferRegion(&lut_buf, 0, &stage, 0, 2048);
            cmd.list.ResourceBarrier(&[transition(
                &lut_buf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            cmd.list.Close()?;
            let cmd_list: ID3D12CommandList = cmd.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(cmd_list)]);
            self.fence_val += 1;
            self.queue.Signal(&self.fence, self.fence_val)?;
            let gpu = lut_buf.GetGPUVirtualAddress();
            (lut_buf, gpu)
        };
        let val = self.fence_val;
        COMPUTE_SAFE_FENCE.fetch_max(val, Ordering::AcqRel);
        self.in_flight.push_back(InFlightBatch {
            until_fence: val,
            cmd,
            _used_slabs: Vec::new(),
            _keep_resources: vec![stage, lut_buf.clone()],
        });
        self.luts.insert(
            key,
            LutEntry {
                _buffer: lut_buf,
                gpu,
            },
        );
        Ok(gpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_detile(
        &mut self,
        host: &Host,
        dst_buffer: &ID3D12Resource,
        dst_base: u64,
        dst_total: u64,
        layout: &Layout,
        raw_slice: Option<RawSlice>,
        footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
        rows: &[u32],
        row_sizes: &[u64],
        first: u32,
        mips: u32,
        held: u32,
        has_held: bool,
        has_unheld: bool,
    ) -> windows::core::Result<()> {
        self.reclaim_completed();
        let pat_gpu = if has_held {
            self.ensure_pattern_lut(host, layout)?
        } else {
            0
        };
        let batch_id = active_batch_id();
        // SAFETY: live D3D12 buffer.
        let dst_res_va = unsafe { dst_buffer.GetGPUVirtualAddress() };
        self.flush_conflicting_batches(batch_id, dst_res_va)?;

        if batch_id != 0 {
            if self.batches.get(&batch_id).is_none_or(|s| s.open.is_none()) {
                let cmd = self.take_cmd_list()?;
                self.batches.entry(batch_id).or_default().open = Some(OpenBatch::new(cmd));
            }
            let slot = self.batches.get_mut(&batch_id).expect("slot ensured above");
            let batch = slot.open.as_mut().expect("open_batch ensured above");
            record_detile_op(
                &self.detile_root,
                &self.detile_pso,
                &self.zero_root,
                &self.zero_pso,
                pat_gpu,
                batch,
                dst_buffer,
                dst_base,
                dst_total,
                layout,
                raw_slice,
                footprints,
                rows,
                row_sizes,
                first,
                mips,
                held,
                has_held,
                has_unheld,
            )?;
            COMPUTE_SAFE_FENCE.fetch_max(self.fence_val + 1, Ordering::AcqRel);
            let overflow = batch.raw_bytes >= BATCH_MAX_RAW_BYTES || batch.ops >= BATCH_MAX_OPS;
            if overflow {
                let full = self
                    .batches
                    .get_mut(&batch_id)
                    .and_then(|s| s.open.take())
                    .expect("checked Some");
                let val = self.submit_batch_inner(full, true)?;
                if val != 0
                    && let Some(slot) = self.batches.get_mut(&batch_id)
                {
                    slot.wait_fence = slot.wait_fence.max(val);
                }
            }
            return Ok(());
        }

        // Immediate synchronous fallback outside any active batch.
        let mut batch = OpenBatch::new(self.take_cmd_list()?);
        record_detile_op(
            &self.detile_root,
            &self.detile_pso,
            &self.zero_root,
            &self.zero_pso,
            pat_gpu,
            &mut batch,
            dst_buffer,
            dst_base,
            dst_total,
            layout,
            raw_slice,
            footprints,
            rows,
            row_sizes,
            first,
            mips,
            held,
            has_held,
            has_unheld,
        )?;
        let val = self.submit_batch_inner(batch, false)?;
        if val != 0 {
            self.wait_fence(val)?;
            STATS[7].fetch_add(1, Ordering::Relaxed);
            self.reclaim_completed();
        }
        Ok(())
    }

    fn run_zero(
        &mut self,
        dst_buffer: &ID3D12Resource,
        dst_base: u64,
        dst_total: u64,
    ) -> windows::core::Result<()> {
        if (dst_total as u32).div_ceil(4) == 0 {
            return Ok(());
        }
        self.reclaim_completed();
        let batch_id = active_batch_id();
        // SAFETY: live D3D12 buffer.
        let dst_res_va = unsafe { dst_buffer.GetGPUVirtualAddress() };
        self.flush_conflicting_batches(batch_id, dst_res_va)?;

        if batch_id != 0 {
            if self.batches.get(&batch_id).is_none_or(|s| s.open.is_none()) {
                let cmd = self.take_cmd_list()?;
                self.batches.entry(batch_id).or_default().open = Some(OpenBatch::new(cmd));
            }
            let slot = self.batches.get_mut(&batch_id).expect("slot ensured above");
            let batch = slot.open.as_mut().expect("open_batch ensured above");
            record_zero_op(
                &self.zero_root,
                &self.zero_pso,
                batch,
                dst_buffer,
                dst_base,
                dst_total,
            );
            COMPUTE_SAFE_FENCE.fetch_max(self.fence_val + 1, Ordering::AcqRel);
            let overflow = batch.ops >= BATCH_MAX_OPS;
            if overflow {
                let full = self
                    .batches
                    .get_mut(&batch_id)
                    .and_then(|s| s.open.take())
                    .expect("checked Some");
                let val = self.submit_batch_inner(full, true)?;
                if val != 0
                    && let Some(slot) = self.batches.get_mut(&batch_id)
                {
                    slot.wait_fence = slot.wait_fence.max(val);
                }
            }
            return Ok(());
        }

        let mut batch = OpenBatch::new(self.take_cmd_list()?);
        record_zero_op(
            &self.zero_root,
            &self.zero_pso,
            &mut batch,
            dst_buffer,
            dst_base,
            dst_total,
        );
        let val = self.submit_batch_inner(batch, false)?;
        if val != 0 {
            self.wait_fence(val)?;
            STATS[7].fetch_add(1, Ordering::Relaxed);
            self.reclaim_completed();
        }
        Ok(())
    }
}

fn begin_submit_batch() {
    if let Some(Some(mutex)) = ENGINE.get() {
        let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
        if eng.batches.contains_key(&SUBMIT_BATCH_ID) {
            let _ = eng.finish_and_wait_batch(SUBMIT_BATCH_ID);
        }
        SUBMIT_BATCH_ACTIVE.store(true, Ordering::Release);
    } else {
        SUBMIT_BATCH_ACTIVE.store(true, Ordering::Release);
    }
}

fn flush_batch_id(batch_id: u64, target_queue: &ID3D12CommandQueue) {
    if let Some(Some(mutex)) = ENGINE.get() {
        let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
        if batch_id == SUBMIT_BATCH_ID {
            SUBMIT_BATCH_ACTIVE.store(false, Ordering::Release);
        }
        let started = std::time::Instant::now();
        match eng.flush_batch_to_queue(batch_id, target_queue) {
            Ok(waited) => {
                if waited {
                    STATS[3].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(error = %e, "gpu compute detile batch flush failed");
                });
            }
        }
    } else if batch_id == SUBMIT_BATCH_ID {
        SUBMIT_BATCH_ACTIVE.store(false, Ordering::Release);
    }
}

fn cancel_batch_id(batch_id: u64) {
    if let Some(Some(mutex)) = ENGINE.get() {
        let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
        if batch_id == SUBMIT_BATCH_ID {
            SUBMIT_BATCH_ACTIVE.store(false, Ordering::Release);
        }
        if let Err(e) = eng.finish_and_wait_batch(batch_id) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(error = %e, "gpu compute detile cancel_batch failed");
            });
        }
    } else if batch_id == SUBMIT_BATCH_ID {
        SUBMIT_BATCH_ACTIVE.store(false, Ordering::Release);
    }
}

/// RAII scope guard for an async batched GPU compute detile window.
///
/// - [`BatchGuard::begin`] opens the submission-wide batch for `Host::submit`
///   and its `wide_each` workers.
/// - [`BatchGuard::begin_thread`] opens an isolated thread-private batch for a
///   background copy-queue worker (`texture::ahead`).
///
/// Dropping without calling [`BatchGuard::flush`] drains the batch cleanly so
/// an open batch can never leak on early return or panic.
pub(crate) struct BatchGuard {
    batch_id: u64,
    prev_thread_id: Option<u64>,
    active: bool,
}

impl BatchGuard {
    pub(crate) fn begin() -> Self {
        begin_submit_batch();
        Self {
            batch_id: SUBMIT_BATCH_ID,
            prev_thread_id: None,
            active: true,
        }
    }

    pub(crate) fn begin_thread() -> Self {
        let id = NEXT_BATCH_ID.fetch_add(1, Ordering::Relaxed);
        let prev = THREAD_BATCH_ID.replace(id);
        Self {
            batch_id: id,
            prev_thread_id: Some(prev),
            active: true,
        }
    }

    pub(crate) fn flush(mut self, target_queue: &ID3D12CommandQueue) {
        self.active = false;
        if let Some(prev) = self.prev_thread_id.take() {
            THREAD_BATCH_ID.set(prev);
        }
        flush_batch_id(self.batch_id, target_queue);
    }
}

impl Drop for BatchGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev_thread_id.take() {
            THREAD_BATCH_ID.set(prev);
        }
        if self.active {
            cancel_batch_id(self.batch_id);
        }
    }
}

/// GPU-detiles `tiled` into `dst_buffer` (a `DEFAULT` UAV buffer resting in
/// `COMMON`, returned to `COMMON` when the batch flushes or before returning
/// in immediate mode). Returns `true` on success.
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
    if !has_held && !has_unheld {
        return true;
    }

    let started = std::time::Instant::now();
    // Phase 1 & 2: suballocate a raw `UPLOAD` slice and copy `tiled` bytes
    // sequentially OUTSIDE `ENGINE`'s mutex so worker threads run in parallel.
    let raw_slice = if has_held {
        let Some(slice) = alloc_raw_slice(host, tiled.len()) else {
            return false;
        };
        // SAFETY: `slice.ptr` points to a uniquely carved 256-byte-aligned
        // range of at least `tiled.len().next_multiple_of(4)` mapped bytes kept
        // alive by `slice.slab`.
        unsafe {
            std::ptr::copy_nonoverlapping(tiled.as_ptr(), slice.ptr, tiled.len());
            let rem = tiled.len() & 3;
            if rem != 0 {
                std::ptr::write_bytes(slice.ptr.add(tiled.len()), 0, 4 - rem);
            }
        }
        Some(slice)
    } else {
        None
    };

    // Phase 3: append compute dispatches onto the open batch (zero CPU wait)
    // or execute synchronously if no batch is active.
    let mut eng = mutex.lock().unwrap_or_else(|e| e.into_inner());
    match eng.run_detile(
        host,
        dst_buffer,
        dst_base,
        dst_total,
        layout,
        raw_slice,
        footprints,
        rows,
        row_sizes,
        first,
        mips,
        held,
        has_held,
        has_unheld,
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

/// Telemetry summary line per frame for `host.rs`.
pub(crate) fn line(frames: f64) {
    let frames = frames.max(1.0);
    let gpu_tex = STATS[0].swap(0, Ordering::Relaxed);
    let gpu_raw_bytes = STATS[1].swap(0, Ordering::Relaxed);
    let gpu_lin_bytes = STATS[2].swap(0, Ordering::Relaxed);
    let gpu_us = STATS[3].swap(0, Ordering::Relaxed);
    let cpu_tex = STATS[4].swap(0, Ordering::Relaxed);
    let fallback = STATS[5].swap(0, Ordering::Relaxed);
    let flushes = STATS[6].swap(0, Ordering::Relaxed);
    let sync_waits = STATS[7].swap(0, Ordering::Relaxed);
    if gpu_tex == 0 && cpu_tex == 0 && fallback == 0 {
        return;
    }
    let active_mb = ACTIVE_VRAM_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    let spare_mb = VRAM_SPARE.lock().unwrap_or_else(|e| e.into_inner()).1 as f64 / (1024.0 * 1024.0);
    let raw_ring_mb = RAW_POOL_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    tracing::info!(
        "gpu detile per frame (VICTORY_GPU_DETILE=0 turns it off): {:.1} textures ({:.1} MB tiled -> {:.1} MB linear) \
         in {:.2} ms ({:.1} flushes, {:.1} sync waits); {:.1} CPU-detiled ({:.1} VRAM-cap/init fallbacks); \
         VRAM staging {:.1} MB active, {:.1} MB spare; raw upload ring {:.1} MB",
        gpu_tex as f64 / frames,
        gpu_raw_bytes as f64 / (1024.0 * 1024.0) / frames,
        gpu_lin_bytes as f64 / (1024.0 * 1024.0) / frames,
        gpu_us as f64 / 1000.0 / frames,
        flushes as f64 / frames,
        sync_waits as f64 / frames,
        cpu_tex as f64 / frames,
        fallback as f64 / frames,
        active_mb,
        spare_mb,
        raw_ring_mb,
    );
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
            host.device
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
        // SAFETY: `vram` rests in `COMMON`, `rb` is in `COPY_DEST`.
        unsafe {
            list.list.CopyBufferRegion(&rb, 0, vram, 0, bytes as u64);
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

    struct PreparedCase {
        label: String,
        layout: Layout,
        tiled: Vec<u8>,
        footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
        rows: Vec<u32>,
        row_sizes: Vec<u64>,
        total: u64,
        total_aligned: usize,
        first: u32,
        mips: u32,
        held: u32,
        cpu_linear: Vec<u8>,
        vram: ID3D12Resource,
    }

    fn prepare_case(
        host: &Host,
        desc: Desc,
        first: u32,
        held: u32,
        seed: u64,
        label: String,
    ) -> PreparedCase {
        let layout = Layout::new(&desc).unwrap();
        let raw_len = layout.bytes_from(first) as usize;
        let tiled: Vec<u8> = (0..raw_len)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(seed.wrapping_mul(0xbf58_476d_1ce4_e5b9))
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
            host.device.GetCopyableFootprints(
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

        let vram = host.uav_buffer(total_aligned.max(512) as u64).unwrap();
        PreparedCase {
            label,
            layout,
            tiled,
            footprints,
            rows,
            row_sizes,
            total,
            total_aligned,
            first,
            mips: desc.mips,
            held,
            cpu_linear,
            vram,
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
        let lin = Layout::new(&Desc { swizzle: Swizzle::LINEAR, ..base }).unwrap();
        assert!(!eligible_layout(&lin, lin.size as usize));
        let xor = Layout::new(&Desc { swizzle: Swizzle::SW_64KB_D_X, ..base }).unwrap();
        assert!(!eligible_layout(&xor, xor.size as usize));
        let tex3d = Layout::new(&Desc {
            dimension: Dimension::Tex3d,
            depth: 8,
            swizzle: Swizzle::SW_4KB_S,
            ..base
        })
        .unwrap();
        assert!(!eligible_layout(&tex3d, tex3d.size as usize));
        let msaa = Layout::new(&Desc { samples: 4, ..base }).unwrap();
        assert!(!eligible_layout(&msaa, msaa.size as usize));
        let tiny = Layout::new(&Desc {
            width: 16,
            height: 16,
            swizzle: Swizzle::SW_256B_S,
            ..base
        })
        .unwrap();
        assert!(!eligible_layout(&tiny, tiny.size as usize));
    }

    #[test]
    fn vram_active_budget_scales_with_adapter_headroom() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(vram_active_budget_from(Some([16 * GIB, 11 * GIB, 0, 0])), 1024 << 20);
        assert_eq!(vram_active_budget_from(Some([12 * GIB, 4 * GIB, 0, 0])), 512 << 20);
        assert_eq!(vram_active_budget_from(Some([16 * GIB, 15 * GIB, 0, 0])), 128 << 20);
        assert_eq!(vram_active_budget_from(None), 128 << 20);
    }

    #[test]
    fn bit_exact_all_element_sizes_and_swizzles_with_mip_tails() {
        let Some(host) = crate::host::host() else { return };
        let swizzles = [
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let formats = [
            (1u32, 1u32, 131u32, 79u32, 8u32),
            (2, 1, 97, 63, 7),
            (4, 1, 128, 96, 8),
            (8, 4, 256, 192, 9),
            (16, 4, 256, 256, 9),
        ];
        for swizzle in swizzles {
            for (elem_bytes, block, width, height, mips) in formats {
                let c = prepare_case(
                    host,
                    Desc {
                        dimension: Dimension::Tex2d,
                        width,
                        height,
                        depth: 1,
                        mips,
                        element_bytes: elem_bytes,
                        block,
                        samples: 1,
                        swizzle,
                    },
                    0,
                    u32::MAX,
                    1,
                    format!("{swizzle:?} e{elem_bytes} b{block} {width}x{height} m{mips}"),
                );
                assert!(zero_buffer(host, &c.vram, 0, c.total_aligned as u64));
                assert!(detile(
                    host,
                    &c.vram,
                    0,
                    c.total,
                    &c.layout,
                    &c.tiled,
                    &c.footprints,
                    &c.rows,
                    &c.row_sizes,
                    c.first,
                    c.mips,
                    c.held,
                ));
                let gpu = fetch_vram(host, &c.vram, c.total_aligned);
                assert_no_debug_errors(host, &c.label);
                assert_eq!(gpu, c.cpu_linear, "mismatch in {}", c.label);
            }
        }
    }

    #[test]
    fn batched_async_and_isolated_thread_batches_match_cpu_bit_exact() {
        let Some(host) = crate::host::host() else { return };

        let submit_case = prepare_case(
            host,
            Desc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 256,
                depth: 1,
                mips: 9,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
            0,
            !0b11u32,
            10,
            "submit_bc7_on_demand".to_owned(),
        );
        let thread_case = prepare_case(
            host,
            Desc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 192,
                depth: 1,
                mips: 9,
                element_bytes: 8,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            },
            0,
            u32::MAX,
            20,
            "thread_bc1_copy_queue".to_owned(),
        );

        let sync_before = STATS[7].load(Ordering::Relaxed);
        // Keep `Host::submit`'s batch open while a concurrent `uploads ahead`
        // thread runs and flushes its own isolated batch onto a COPY queue!
        let submit_batch = BatchGuard::begin();
        assert!(zero_buffer(host, &submit_case.vram, 0, submit_case.total_aligned as u64));
        assert!(detile(
            host,
            &submit_case.vram,
            0,
            submit_case.total,
            &submit_case.layout,
            &submit_case.tiled,
            &submit_case.footprints,
            &submit_case.rows,
            &submit_case.row_sizes,
            submit_case.first,
            submit_case.mips,
            submit_case.held,
        ));

        let thread_gpu = std::thread::scope(|s| {
            let tc = &thread_case;
            s.spawn(move || {
                // SAFETY: plain D3D12 copy queue, allocator, list, and fence.
                let (copy_q, alloc, list, fence): (
                    ID3D12CommandQueue,
                    ID3D12CommandAllocator,
                    ID3D12GraphicsCommandList,
                    ID3D12Fence,
                ) = unsafe {
                    let q = host
                        .device
                        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                            Type: D3D12_COMMAND_LIST_TYPE_COPY,
                            ..Default::default()
                        })
                        .unwrap();
                    let a = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).unwrap();
                    let l = host
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &a, None)
                        .unwrap();
                    let f = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                    (q, a, l, f)
                };
                let tb = BatchGuard::begin_thread();
                assert!(zero_buffer(host, &tc.vram, 0, tc.total_aligned as u64));
                assert!(detile(
                    host,
                    &tc.vram,
                    0,
                    tc.total,
                    &tc.layout,
                    &tc.tiled,
                    &tc.footprints,
                    &tc.rows,
                    &tc.row_sizes,
                    tc.first,
                    tc.mips,
                    tc.held,
                ));
                // Flush ONLY the thread-private batch to the copy queue while
                // `submit_batch` is still open!
                tb.flush(&copy_q);
                let rb = readback_buffer(host, tc.total_aligned as u64);
                // SAFETY: `tc.vram` transitioned back to `COMMON` before signal,
                // and `copy_q` waits on compute fence before running.
                unsafe {
                    list.CopyBufferRegion(&rb, 0, &tc.vram, 0, tc.total_aligned as u64);
                    list.Close().unwrap();
                    copy_q.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                    copy_q.Signal(&fence, 1).unwrap();
                    fence
                        .SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default())
                        .unwrap();
                    let _ = alloc;
                    let mut ptr = std::ptr::null_mut();
                    rb.Map(0, None, Some(&mut ptr)).unwrap();
                    let out = std::slice::from_raw_parts(ptr.cast::<u8>(), tc.total_aligned).to_vec();
                    rb.Unmap(0, None);
                    out
                }
            })
            .join()
            .unwrap()
        });
        assert_eq!(thread_gpu, thread_case.cpu_linear, "thread copy-queue batch mismatch");
        assert_eq!(STATS[7].load(Ordering::Relaxed), sync_before, "zero CPU fence waits expected");

        // Now flush `submit_batch` to the direct queue and verify!
        submit_batch.flush(&host.queue);
        let submit_gpu = fetch_vram(host, &submit_case.vram, submit_case.total_aligned);
        assert_no_debug_errors(host, "isolated_submit_and_thread_batches");
        assert_eq!(submit_gpu, submit_case.cpu_linear, "submit direct-queue batch mismatch");
    }
}
