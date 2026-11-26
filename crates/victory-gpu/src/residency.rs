//! Host-memory admission for speculative texture work and reusable upload storage.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

thread_local! {
    static STAGING_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn note_staging_deferred() {
    STAGING_DEFERRED.set(true);
}

pub(crate) fn take_staging_deferred() -> bool {
    STAGING_DEFERRED.replace(false)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bake: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool.
    commit >= if bake { 256 * MIB } else { 1024 * MIB }
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_upload_reserve(sample_commit(now), bake)
}

/// Whether the staging allocator may commit a brand-new OS upload buffer
/// (reusing existing free ranges in already-committed pool blocks stays open
/// whenever `admit_texture_pass` is true).
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_growth_reserve(sample_commit(now), bake)
}

/// Host physical-memory tier used to size upload pools, staging caps, and
/// retirement lag without starving 32–48 GiB hosts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostRamTier {
    /// <= 32 GiB physical RAM.
    Tight,
    /// 33–52 GiB physical RAM (e.g. 48 GiB DDR4 desktop).
    Moderate,
    /// > 52 GiB physical RAM (e.g. 64 GiB+ workstation).
    Plentiful,
}

fn ram_tier_for(total_phys: u64) -> HostRamTier {
    if total_phys <= 32 * GIB {
        HostRamTier::Tight
    } else if total_phys <= 52 * GIB {
        HostRamTier::Moderate
    } else {
        HostRamTier::Plentiful
    }
}

pub(crate) fn host_ram_tier() -> HostRamTier {
    static TIER: OnceLock<HostRamTier> = OnceLock::new();
    *TIER.get_or_init(|| memory_status().map_or(HostRamTier::Plentiful, |s| ram_tier_for(s.ullTotalPhys)))
}

pub(crate) fn limited_host_ram() -> bool {
    host_ram_tier() == HostRamTier::Tight
}

pub(crate) fn moderate_or_limited_ram() -> bool {
    host_ram_tier() != HostRamTier::Plentiful
}

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was {
        physical < 5 * GIB || commit < 6 * GIB
    } else {
        physical <= 3 * GIB || commit <= 4 * GIB
    }
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    let tier = if limited {
        HostRamTier::Tight
    } else if host_ram_tier() == HostRamTier::Moderate {
        HostRamTier::Moderate
    } else {
        HostRamTier::Plentiful
    };
    prefetch_limit_for_tier(bytes, pressure, critical, tier, staging)
}

pub(crate) fn prefetch_limit_for_tier(
    bytes: u64,
    pressure: bool,
    critical: bool,
    tier: HostRamTier,
    staging: u64,
) -> u64 {
    let staging_cap = match tier {
        HostRamTier::Tight => GIB,
        HostRamTier::Moderate => 1536 * MIB,
        HostRamTier::Plentiful => u64::MAX,
    };
    if critical {
        bytes.min(16 << 20)
    } else if pressure || staging >= staging_cap {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve(commit, false), growth);
        }
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure(false, 12 * GIB, 4 * GIB));
        assert!(next_pressure(false, 3 * GIB, 16 * GIB));
        assert!(next_pressure(true, 4 * GIB, 7 * GIB));
        assert!(next_pressure(true, 6 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit_for_tier(512 << 20, false, false, HostRamTier::Tight, GIB), 64 << 20);
        assert_eq!(prefetch_limit_for_tier(512 << 20, false, false, HostRamTier::Tight, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit_for_tier(512 << 20, false, false, HostRamTier::Moderate, 1536 * MIB), 64 << 20);
        assert_eq!(prefetch_limit_for_tier(512 << 20, false, false, HostRamTier::Moderate, 1536 * MIB - 1), 512 << 20);
        assert_eq!(prefetch_limit_for_tier(512 << 20, false, false, HostRamTier::Plentiful, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit_for_tier(0, true, true, HostRamTier::Plentiful, 0), 0);
        assert_eq!(prefetch_limit_for_tier(8 << 20, true, true, HostRamTier::Plentiful, 0), 8 << 20);
        assert_eq!(prefetch_limit_for_tier(512 << 20, true, false, HostRamTier::Plentiful, 0), 64 << 20);
        assert_eq!(prefetch_limit_for_tier(512 << 20, true, true, HostRamTier::Plentiful, 0), 16 << 20);
        assert_eq!(prefetch_limit_for_tier(192 << 20, false, false, HostRamTier::Plentiful, 0), 192 << 20);
        assert_eq!(ram_tier_for(32 * GIB), HostRamTier::Tight);
        assert_eq!(ram_tier_for(48 * GIB), HostRamTier::Moderate);
        assert_eq!(ram_tier_for(64 * GIB), HostRamTier::Plentiful);
    }
}
