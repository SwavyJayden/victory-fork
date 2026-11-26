//! Hints the HLE layer (`victory-hle`) forwards to the GPU runtime (`victory-gpu`):
//!
//! 1. **Xbox GPU read-only memory ranges** (`PAGE_GRAPHICS_READONLY` /
//!    `PAGE_GRAPHICS_EXECUTE_READ` from `XMemVirtualAlloc`): ranges the console
//!    GPU is guaranteed never to write, letting `shadow` skip GPU-write
//!    declarations, `Card::before` snapshots, GPU compares, and copy-back.
//! 2. **DirectStorage streaming backlog counters**: live count of queued and
//!    in-flight DirectStorage reads plus total completed bytes, letting loading
//!    screens and world-entry pacing drain textures into VRAM before 3D scenes
//!    begin.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{OnceLock, RwLock};

/// Sorted, coalesced, non-overlapping half-open `[start, end)` ranges marked
/// GPU-read-only by the title via Xbox memory protection flags.
static READONLY_RANGES: RwLock<Vec<(u64, u64)>> = RwLock::new(Vec::new());
/// Fast-path mirror of `!READONLY_RANGES.is_empty()` so queries avoid locking
/// when no read-only ranges have been registered.
static HAS_READONLY: AtomicBool = AtomicBool::new(false);

/// Queued + in-flight DirectStorage read requests across all lanes.
static DSTORAGE_PENDING: AtomicUsize = AtomicUsize::new(0);
/// Cumulative completed DirectStorage read operations.
static DSTORAGE_COMPLETED_OPS: AtomicU64 = AtomicU64::new(0);
/// Cumulative completed DirectStorage destination bytes.
static DSTORAGE_COMPLETED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Whether Xbox GPU read-only memory hints (`PAGE_GRAPHICS_READONLY` /
/// `PAGE_GRAPHICS_EXECUTE_READ`) skip GPU-write tracking in `shadow`.
/// Enabled by default; set `VICTORY_XMEM_READONLY=0` to disable.
pub fn gpu_readonly_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_XMEM_READONLY").is_none_or(|v| v != "0"))
}

/// Records or clears a GPU-read-only half-open address range `[base, base + len)`.
///
/// When `readonly` is `true`, the interval is inserted and merged with any
/// overlapping or adjacent intervals. When `readonly` is `false`, any portion
/// overlapping `[base, base + len)` is carved out (splitting an enclosing
/// interval if needed), matching `MEM_REPLACE_PLACEHOLDER` and release semantics.
pub fn set_gpu_memory_readonly(base: u64, len: u64, readonly: bool) {
    if len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    if end <= base {
        return;
    }
    let mut ranges = READONLY_RANGES.write().unwrap_or_else(|e| e.into_inner());
    if readonly {
        let mut new_start = base;
        let mut new_end = end;
        let mut out = Vec::with_capacity(ranges.len() + 1);
        let mut inserted = false;
        for &(s, e) in ranges.iter() {
            if e < new_start {
                out.push((s, e));
            } else if new_end < s {
                if !inserted {
                    out.push((new_start, new_end));
                    inserted = true;
                }
                out.push((s, e));
            } else {
                new_start = new_start.min(s);
                new_end = new_end.max(e);
            }
        }
        if !inserted {
            out.push((new_start, new_end));
        }
        *ranges = out;
    } else if !ranges.is_empty() {
        let mut out = Vec::with_capacity(ranges.len() + 1);
        for &(s, e) in ranges.iter() {
            if e <= base || s >= end {
                out.push((s, e));
            } else {
                if s < base {
                    out.push((s, base));
                }
                if end < e {
                    out.push((end, e));
                }
            }
        }
        *ranges = out;
    }
    HAS_READONLY.store(!ranges.is_empty(), Ordering::Release);
}

/// Returns `true` when `VICTORY_XMEM_READONLY` is active and the entire range
/// `[addr, addr + len.max(1))` lies inside a registered GPU-read-only range.
pub fn gpu_readonly_contains(addr: u64, len: u64) -> bool {
    if !HAS_READONLY.load(Ordering::Acquire) || !gpu_readonly_enabled() {
        return false;
    }
    let end = addr.saturating_add(len.max(1));
    let ranges = READONLY_RANGES.read().unwrap_or_else(|e| e.into_inner());
    let idx = ranges.partition_point(|&(start, _)| start <= addr);
    if idx == 0 {
        return false;
    }
    let (start, range_end) = ranges[idx - 1];
    addr >= start && end <= range_end
}

/// Returns `(interval_count, total_bytes)` of currently registered GPU-read-only ranges.
pub fn gpu_readonly_stats() -> (usize, u64) {
    if !HAS_READONLY.load(Ordering::Acquire) {
        return (0, 0);
    }
    let ranges = READONLY_RANGES.read().unwrap_or_else(|e| e.into_inner());
    let bytes = ranges.iter().map(|&(s, e)| e.saturating_sub(s)).sum();
    (ranges.len(), bytes)
}

/// Removes all registered GPU-read-only ranges (for tests).
pub fn clear_gpu_readonly_ranges() {
    let mut ranges = READONLY_RANGES.write().unwrap_or_else(|e| e.into_inner());
    ranges.clear();
    HAS_READONLY.store(false, Ordering::Release);
}

/// Increments the count of queued/in-flight DirectStorage read requests.
pub fn dstorage_add_pending(count: usize) {
    if count > 0 {
        DSTORAGE_PENDING.fetch_add(count, Ordering::Relaxed);
    }
}

/// Decrements the count of queued/in-flight DirectStorage read requests without
/// underflow.
pub fn dstorage_sub_pending(count: usize) {
    if count > 0 {
        let _ = DSTORAGE_PENDING.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_sub(count))
        });
    }
}

/// Returns the current number of queued/in-flight DirectStorage read requests.
pub fn dstorage_pending() -> usize {
    DSTORAGE_PENDING.load(Ordering::Relaxed)
}

/// Records completion of a DirectStorage read request of `bytes` destination bytes.
pub fn dstorage_note_completed(bytes: u64) {
    DSTORAGE_COMPLETED_OPS.fetch_add(1, Ordering::Relaxed);
    DSTORAGE_COMPLETED_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Returns `(completed_ops, completed_bytes)` since process start.
pub fn dstorage_completed() -> (u64, u64) {
    (
        DSTORAGE_COMPLETED_OPS.load(Ordering::Relaxed),
        DSTORAGE_COMPLETED_BYTES.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readonly_coalesce_split_and_query() {
        clear_gpu_readonly_ranges();
        assert!(!gpu_readonly_contains(0x1000, 0x100));

        // Insert and coalesce adjacent/overlapping ranges.
        set_gpu_memory_readonly(0x1000_0000, 0x1000_0000, true);
        set_gpu_memory_readonly(0x2000_0000, 0x1000_0000, true);
        assert_eq!(gpu_readonly_stats(), (1, 0x2000_0000));
        assert!(gpu_readonly_contains(0x1000_0000, 0x2000_0000));
        assert!(!gpu_readonly_contains(0x1000_0000, 0x2000_0001));

        // Carve out a middle RW sub-pool (MEM_REPLACE_PLACEHOLDER).
        set_gpu_memory_readonly(0x1800_0000, 0x0400_0000, false);
        assert_eq!(gpu_readonly_stats(), (2, 0x1C00_0000));
        assert!(gpu_readonly_contains(0x1000_0000, 0x0800_0000));
        assert!(!gpu_readonly_contains(0x17FF_F000, 0x2000));
        assert!(!gpu_readonly_contains(0x1800_0000, 0x1000));
        assert!(gpu_readonly_contains(0x1C00_0000, 0x1400_0000));

        // Free everything.
        set_gpu_memory_readonly(0x1000_0000, 0x2000_0000, false);
        assert_eq!(gpu_readonly_stats(), (0, 0));
        assert!(!gpu_readonly_contains(0x1000_0000, 0x1000));
    }

    #[test]
    fn dstorage_pending_saturates_at_zero() {
        let base = dstorage_pending();
        dstorage_add_pending(5);
        assert!(dstorage_pending() >= base + 5);
        dstorage_sub_pending(5);
        dstorage_sub_pending(usize::MAX / 2);
        assert_eq!(dstorage_pending(), 0);
    }
}
