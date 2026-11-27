//! Zero-fault working-set `GpuOnly` readback residency (`VICTORY_GPU_ONLY_BACK`,
//! on by default; `=0` disables completely for A/B comparison).
//!
//! Empirical 49-minute GTA VI telemetry showed 99.81% of GPU->CPU landed pages
//! are never read by the title CPU, yet `CARD_COMPARE`, `relay_card_back`,
//! `land`, and `stale_under_flights` scan and copy back tens of megabytes per
//! frame.
//!
//! This module tracks per-64 KiB piece residency on carded `Shadow`s (`>= 64 KiB`)
//! strictly using zero-fault Windows working-set trimming (`VirtualUnlock` on
//! unlocked pages) and page-table inspection (`K32QueryWorkingSetEx`) — never
//! `PAGE_NOACCESS` traps — gated by five airtight rules:
//! 1. Size & carded (`shadow.carded() && shadow.len >= 64 KiB`, whole 64 KiB block).
//! 2. Disjoint / non-aliased (`inplace::pages_of(block).len() <= 1` and no other
//!    live `Shadow` overlapping `[block, block + 64 KiB)`).
//! 3. CPU-write-quiet and step-quiet for `>= 30` consecutive frames.
//! 4. Working-set verified (`K32QueryWorkingSetEx` sees 0 of 16 4 KiB pages in
//!    working set across `>= 8` landed probes or `>= 6` unflagged probes).
//! 5. Instant lossless demotion + permanent session pin on any CPU write, CPU
//!    step (`Copy`/`Placed`/`Fill`/`Texture`/`Plane`/`Host`/`Run`), alias/new
//!    overlapping shadow, or background watchdog working-set touch.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Size of one residency-tracked piece (matches `shadow::PIECE` / `watch::BLOCK`).
pub(crate) const PIECE: usize = 64 << 10;
/// Size of one Windows 4 KiB page.
const KERNEL_PAGE: usize = 4096;
/// Number of 4 KiB pages per 64 KiB piece.
const PAGES_PER_PIECE: usize = PIECE / KERNEL_PAGE;

/// Consecutive frames without CPU writes or overlapping CPU steps required
/// before a piece may promote to `GpuOnly`.
pub(crate) const QUIET_FRAMES: u64 = 30;
/// Consecutive untouched working-set probes required on landed pieces.
pub(crate) const LANDED_CLEAN_PROBES: u16 = 8;
/// Consecutive untouched working-set probes required on declared-but-unflagged pieces.
pub(crate) const UNFLAGGED_CLEAN_PROBES: u16 = 6;
/// Minimum frames between trim and working-set probe for unflagged pieces.
pub(crate) const UNFLAGGED_MIN_PROBE_GAP: u64 = 2;
/// Maximum active `GpuOnly` pieces audited per frame by the rotating watchdog.
pub(crate) const WATCHDOG_BUDGET_PER_FRAME: usize = 256;
/// Minimum shadow byte length eligible for unflagged-only promotion (`256 KiB`).
/// Small `< 256 KiB` buffers (e.g. occlusion/streaming feedback buffers that may
/// sit unchanged across a cutscene) still promote via active landed probes (`>= 64 KiB`),
/// but are kept out of unflagged-only promotion.
pub(crate) const UNFLAGGED_MIN_SHADOW_LEN: usize = 256 << 10;
/// Maximum unflagged candidate piece actions (trim or probe) per frame.
pub(crate) const UNFLAGGED_BUDGET_PER_FRAME: usize = 64;
/// Number of pre-promotion working-set touches required before permanently
/// pinning a piece for the session. Strikes 1 and 2 enter exponential cooldown
/// backoff (`TOUCH_BACKOFF_FRAMES`) so one-time boot/load CPU touches never
/// condemn a GPU-only buffer for the rest of the session.
pub(crate) const TOUCH_STRIKE_LIMIT: u8 = 3;
/// Cooldown frames after pre-promotion working-set touch strike 1 and strike 2.
pub(crate) const TOUCH_BACKOFF_FRAMES: [u64; 2] = [180, 600];

/// Whether `VICTORY_GPU_ONLY_BACK` is enabled (default ON = `1`; `0` disables).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_GPU_ONLY_BACK").is_none_or(|v| v != "0")
            && std::env::var_os("VICTORY_UPCHECK").is_none_or(|v| v == "0")
    })
}

/// Reason a `GpuOnly` piece was demoted back to normal (and pinned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DemoteReason {
    CpuWrite,
    Step,
    Alias,
    CpuRead,
}

/// Permanent session pin cause recorded on `PieceState::Pinned`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinReason {
    Touch,
    CpuWrite,
    Step,
    Alias,
    CpuRead,
}

impl From<DemoteReason> for PinReason {
    fn from(reason: DemoteReason) -> Self {
        match reason {
            DemoteReason::CpuWrite => Self::CpuWrite,
            DemoteReason::Step => Self::Step,
            DemoteReason::Alias => Self::Alias,
            DemoteReason::CpuRead => Self::CpuRead,
        }
    }
}

/// Classification of why a landed carded piece did (or did not) promote to `GpuOnly`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum LandBlockReason {
    SmallBuf = 0,
    PartialPiece = 1,
    Unwatched = 2,
    NotCopied = 3,
    PinTouch = 4,
    PinWrite = 5,
    PinStep = 6,
    PinAlias = 7,
    PinRead = 8,
    Backoff = 9,
    Unquiet = 10,
    Probing = 11,
    Alias = 12,
    DirtyOrLand = 13,
    Promoted = 14,
}

pub(crate) const LAND_REASON_COUNT: usize = 15;

impl LandBlockReason {
    const fn label(self) -> &'static str {
        match self {
            Self::SmallBuf => "small",
            Self::PartialPiece => "partial",
            Self::Unwatched => "unwatched",
            Self::NotCopied => "not_copied",
            Self::PinTouch => "pin_touch",
            Self::PinWrite => "pin_write",
            Self::PinStep => "pin_step",
            Self::PinAlias => "pin_alias",
            Self::PinRead => "pin_read",
            Self::Backoff => "backoff",
            Self::Unquiet => "unquiet",
            Self::Probing => "probing",
            Self::Alias => "alias",
            Self::DirtyOrLand => "dirty",
            Self::Promoted => "promoted",
        }
    }

    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::SmallBuf,
            1 => Self::PartialPiece,
            2 => Self::Unwatched,
            3 => Self::NotCopied,
            4 => Self::PinTouch,
            5 => Self::PinWrite,
            6 => Self::PinStep,
            7 => Self::PinAlias,
            8 => Self::PinRead,
            9 => Self::Backoff,
            10 => Self::Unquiet,
            11 => Self::Probing,
            12 => Self::Alias,
            13 => Self::DirtyOrLand,
            _ => Self::Promoted,
        }
    }
}

impl From<PinReason> for LandBlockReason {
    fn from(reason: PinReason) -> Self {
        match reason {
            PinReason::Touch => Self::PinTouch,
            PinReason::CpuWrite => Self::PinWrite,
            PinReason::Step => Self::PinStep,
            PinReason::Alias => Self::PinAlias,
            PinReason::CpuRead => Self::PinRead,
        }
    }
}

/// Live gauge of active `GpuOnly` pieces across all shadows.
static ACTIVE_GPU_ONLY_PIECES: AtomicU64 = AtomicU64::new(0);
/// Per-interval counters reset once per second in [`log_line`]:
/// 0: compare_pieces_skipped
/// 1: pending_ranges_skipped
/// 2: promoted
/// 3: demoted_write
/// 4: demoted_step
/// 5: demoted_alias
/// 6: demoted_read
/// 7: pinned_by_touch
/// 8: touch_backoff
static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
/// Per-interval landing blocker counters reset once per second in [`log_line`].
static LAND_REASONS: [AtomicU64; LAND_REASON_COUNT] = [const { AtomicU64::new(0) }; LAND_REASON_COUNT];

/// Compact per-shadow landing summary for periodic telemetry (bounded to top 64
/// landing shadows per 1-second interval, updated once per shadow landing call).
struct LandShadowSample {
    id: u64,
    start: u64,
    len: u64,
    watched: u8,
    pieces: u64,
    reasons: [u64; LAND_REASON_COUNT],
}

static LAND_SHADOWS: Mutex<Vec<LandShadowSample>> = Mutex::new(Vec::new());

/// Records one shadow's landing reason histogram for the periodic breakdown log.
/// Called at most once per shadow per landing batch on background landing threads.
pub(crate) fn record_land_summary(
    shadow_id: u64,
    start: u64,
    len: u64,
    watched: u8,
    counts: &[u64; LAND_REASON_COUNT],
) {
    let mut total = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 {
            LAND_REASONS[i].fetch_add(c, Ordering::Relaxed);
            total += c;
        }
    }
    if total == 0 {
        return;
    }
    let mut samples = LAND_SHADOWS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = samples.iter_mut().find(|e| e.id == shadow_id) {
        entry.watched = watched;
        entry.pieces += total;
        for (dst, &src) in entry.reasons.iter_mut().zip(counts.iter()) {
            *dst += src;
        }
    } else if samples.len() < 64 {
        samples.push(LandShadowSample {
            id: shadow_id,
            start,
            len,
            watched,
            pieces: total,
            reasons: *counts,
        });
    } else if let Some(min_entry) = samples.iter_mut().min_by_key(|e| e.pieces)
        && total > min_entry.pieces
    {
        *min_entry = LandShadowSample {
            id: shadow_id,
            start,
            len,
            watched,
            pieces: total,
            reasons: *counts,
        };
    }
}

/// Convenience helper for adding one piece count to a local histogram array.
#[inline]
pub(crate) fn bump_land_reason(counts: &mut [u64; LAND_REASON_COUNT], reason: LandBlockReason, n: u64) {
    counts[reason as usize] += n;
}

/// Compact registry of only the shader-written shadows that have entered
/// `gpu_only` tracking (~10–40 shadows instead of all ~3,600 carded shadows).
static TRACKED_SHADOWS: Mutex<Vec<Weak<crate::shadow::Shadow>>> = Mutex::new(Vec::new());
static TRACKED_SHADOW_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Registers `shadow` in [`TRACKED_SHADOWS`] once it first enters `gpu_only`
/// probing/trimming so `gpu_only_pre_submit` and the background watchdog never
/// scan thousands of read-only static carded buffers.
pub(crate) fn register_tracked_shadow(shadow: &Arc<crate::shadow::Shadow>) {
    let mut list = TRACKED_SHADOWS.lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|w| w.strong_count() > 0);
    if !list.iter().any(|w| w.as_ptr() == Arc::as_ptr(shadow)) {
        list.push(Arc::downgrade(shadow));
    }
    TRACKED_SHADOW_COUNT.store(list.len(), Ordering::Release);
}

#[inline]
pub(crate) fn has_tracked_shadows() -> bool {
    TRACKED_SHADOW_COUNT.load(Ordering::Acquire) != 0
}

/// Upgrades all currently live tracked shadows (~10–40), pruning dead entries.
pub(crate) fn tracked_shadows() -> Vec<Arc<crate::shadow::Shadow>> {
    if !has_tracked_shadows() {
        return Vec::new();
    }
    let mut list = TRACKED_SHADOWS.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::with_capacity(list.len());
    list.retain(|w| match w.upgrade() {
        Some(s) => {
            out.push(s);
            true
        }
        None => false,
    });
    TRACKED_SHADOW_COUNT.store(list.len(), Ordering::Release);
    out
}

/// Number of currently active `GpuOnly` pieces across all live shadows.
#[inline]
pub(crate) fn active_pieces() -> u64 {
    ACTIVE_GPU_ONLY_PIECES.load(Ordering::Acquire)
}

#[inline]
pub(crate) fn record_compare_pieces_skipped(n: u64) {
    if n > 0 {
        STATS[0].fetch_add(n, Ordering::Relaxed);
    }
}

#[inline]
pub(crate) fn record_pending_range_skipped() {
    STATS[1].fetch_add(1, Ordering::Relaxed);
}

fn record_demote(reason: DemoteReason, count: u64) {
    if count == 0 {
        return;
    }
    let idx = match reason {
        DemoteReason::CpuWrite => 3,
        DemoteReason::Step => 4,
        DemoteReason::Alias => 5,
        DemoteReason::CpuRead => 6,
    };
    STATS[idx].fetch_add(count, Ordering::Relaxed);
}

/// Logs `GpuOnly` residency stats once per second alongside card pass stats.
pub(crate) fn log_line(frames: f64) {
    if !enabled() {
        return;
    }
    let active = ACTIVE_GPU_ONLY_PIECES.load(Ordering::Relaxed);
    let tracked = TRACKED_SHADOW_COUNT.load(Ordering::Relaxed);
    let [compare_skipped, pending_skipped, promoted, d_write, d_step, d_alias, d_read, pin_touch, touch_backoff] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| STATS[i].swap(0, Ordering::Relaxed));
    let reasons: [u64; LAND_REASON_COUNT] =
        std::array::from_fn(|i| LAND_REASONS[i].swap(0, Ordering::Relaxed));
    let land_total: u64 = reasons.iter().sum();
    let mut top_shadows = {
        let mut guard = LAND_SHADOWS.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *guard)
    };
    if active == 0
        && compare_skipped == 0
        && pending_skipped == 0
        && promoted == 0
        && d_write == 0
        && d_step == 0
        && d_alias == 0
        && d_read == 0
        && pin_touch == 0
        && touch_backoff == 0
        && land_total == 0
    {
        return;
    }
    tracing::info!(
        "gpu-only readback residency (VICTORY_GPU_ONLY_BACK=1): {active} active pieces ({:.1} MiB, {tracked} shadows), \
         skipped/frame: {:.0} compare pieces, {:.0} pending ranges; \
         promoted +{promoted}, pin_touch {pin_touch}, backoff {touch_backoff}, demoted -{} (write {d_write}, step {d_step}, alias {d_alias}, read {d_read})",
        (active as f64 * PIECE as f64) / (1 << 20) as f64,
        compare_skipped as f64 / frames,
        pending_skipped as f64 / frames,
        d_write + d_step + d_alias + d_read,
    );
    if land_total > 0 {
        top_shadows.sort_unstable_by_key(|a| std::cmp::Reverse(a.pieces));
        let mut top_str = String::new();
        for (idx, s) in top_shadows.iter().take(5).enumerate() {
            if idx > 0 {
                top_str.push_str(", ");
            }
            let dom_idx = s
                .reasons
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map_or(0, |(i, _)| i);
            let dom = LandBlockReason::from_index(dom_idx).label();
            let size_str = if s.len >= 1 << 20 {
                format!("{:.1}M", s.len as f64 / (1 << 20) as f64)
            } else if s.len >= 1 << 10 {
                format!("{}K", s.len >> 10)
            } else {
                format!("{}B", s.len)
            };
            top_str.push_str(&format!(
                "{:#x}({},w={}, {:.0}/f,{})",
                s.start,
                size_str,
                s.watched,
                s.pieces as f64 / frames,
                dom
            ));
        }
        let r = |b: LandBlockReason| reasons[b as usize] as f64 / frames;
        tracing::info!(
            "gpu-only landing breakdown/frame ({:.0} pcs): small={:.0} partial={:.0} unwatched={:.0} not_copied={:.0} \
             pin(touch={:.0} write={:.0} step={:.0} alias={:.0} read={:.0}) backoff={:.0} unquiet={:.0} probing={:.0} \
             alias={:.0} dirty={:.0} promoted={:.0} | top: {top_str}",
            land_total as f64 / frames,
            r(LandBlockReason::SmallBuf),
            r(LandBlockReason::PartialPiece),
            r(LandBlockReason::Unwatched),
            r(LandBlockReason::NotCopied),
            r(LandBlockReason::PinTouch),
            r(LandBlockReason::PinWrite),
            r(LandBlockReason::PinStep),
            r(LandBlockReason::PinAlias),
            r(LandBlockReason::PinRead),
            r(LandBlockReason::Backoff),
            r(LandBlockReason::Unquiet),
            r(LandBlockReason::Probing),
            r(LandBlockReason::Alias),
            r(LandBlockReason::DirtyOrLand),
            r(LandBlockReason::Promoted),
        );
    }
}

/// Per-frame watchdog gate: returns `true` at most once per frame number.
pub(crate) fn claim_watchdog_frame(frame: u64) -> bool {
    static LAST_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    let prev = LAST_FRAME.load(Ordering::Relaxed);
    if prev == frame {
        return false;
    }
    LAST_FRAME.compare_exchange(prev, frame, Ordering::AcqRel, Ordering::Relaxed).is_ok()
}

/// Rotating cursor for the background working-set watchdog.
static WATCHDOG_CURSOR: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn watchdog_cursor() -> usize {
    WATCHDOG_CURSOR.load(Ordering::Relaxed)
}

pub(crate) fn set_watchdog_cursor(next: usize) {
    WATCHDOG_CURSOR.store(next, Ordering::Relaxed);
}

/// Per-frame unflagged candidate action budget (`<= UNFLAGGED_BUDGET_PER_FRAME`).
static UNFLAGGED_BUDGET_STATE: Mutex<(u64, usize)> = Mutex::new((u64::MAX, 0));

/// Reserves up to `want` unflagged probe/trim actions for `frame`.
pub(crate) fn take_unflagged_budget(frame: u64, want: usize) -> usize {
    if want == 0 {
        return 0;
    }
    let mut state = UNFLAGGED_BUDGET_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if state.0 != frame {
        *state = (frame, UNFLAGGED_BUDGET_PER_FRAME);
    }
    let taken = state.1.min(want);
    state.1 -= taken;
    taken
}

/// Returns unused unflagged budget for `frame`.
pub(crate) fn refund_unflagged_budget(frame: u64, unused: usize) {
    if unused == 0 {
        return;
    }
    let mut state = UNFLAGGED_BUDGET_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if state.0 == frame {
        state.1 = (state.1 + unused).min(UNFLAGGED_BUDGET_PER_FRAME);
    }
}

/// Queued demotion generated asynchronously from non-submitting threads
/// (e.g. overlapping shadow creation in `lookup`/`loose`/`exact`/`by_address`
/// or `unregister`). Drained at the start of `Host::submit` before any step or
/// refresh reads title memory.
pub(crate) struct PendingDemotion {
    pub(crate) shadow: Arc<crate::shadow::Shadow>,
    pub(crate) pieces: Vec<usize>,
    pub(crate) reason: DemoteReason,
    pub(crate) already_cleared: bool,
}

static PENDING_DEMOTIONS: Mutex<Vec<PendingDemotion>> = Mutex::new(Vec::new());
static PENDING_DEMOTION_COUNT: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn queue_demotion(
    shadow: Arc<crate::shadow::Shadow>,
    pieces: Vec<usize>,
    reason: DemoteReason,
    already_cleared: bool,
) {
    if pieces.is_empty() {
        return;
    }
    let mut q = PENDING_DEMOTIONS.lock().unwrap_or_else(|e| e.into_inner());
    q.push(PendingDemotion {
        shadow,
        pieces,
        reason,
        already_cleared,
    });
    PENDING_DEMOTION_COUNT.store(q.len(), Ordering::Release);
}

pub(crate) fn take_pending_demotions() -> Vec<PendingDemotion> {
    if PENDING_DEMOTION_COUNT.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let mut q = PENDING_DEMOTIONS.lock().unwrap_or_else(|e| e.into_inner());
    PENDING_DEMOTION_COUNT.store(0, Ordering::Release);
    std::mem::take(&mut *q)
}

/// Per-64 KiB piece residency state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PieceState {
    Normal {
        clean_probes: u16,
        last_cpu_write_frame: u64,
        last_step_frame: u64,
        trimmed_at_frame: u64,
        trimmed: bool,
        unflagged_probe: bool,
        touch_strikes: u8,
        backoff_until_frame: u64,
    },
    GpuOnly {
        promoted_frame: u64,
    },
    /// Permanently pinned for this session (repeated working-set touches or
    /// post-promotion CPU read/write/step/alias/remap) so it never thrashes.
    Pinned(PinReason),
}

/// Action decided before a landing or `take_landed` reads/writes title memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BeforeReadAction {
    /// Piece is not trimmed (or not `Normal`).
    Skip,
    /// Piece was trimmed earlier on the same frame (`trimmed_at_frame >= frame`);
    /// `trimmed` was cleared to `false` so our upcoming read/write cannot leave
    /// resident pages under `trimmed = true` and trigger a false pin next frame.
    ClearedSameFrame,
    /// Piece was trimmed on an earlier frame (`trimmed_at_frame < frame`);
    /// caller should query `K32QueryWorkingSetEx` and call `apply_probe_result`.
    Probe,
}

/// Action decided for an unflagged declared piece on a given frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnflaggedAction {
    Skip,
    NeedTrim,
    NeedQuery,
}

/// Outcome of applying a working-set probe result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProbeOutcome {
    Untouched(u16),
    /// Pre-promotion touch strike `< TOUCH_STRIKE_LIMIT`: entered cooldown backoff.
    BackedOff(u8),
    /// Reached `TOUCH_STRIKE_LIMIT` touches: permanently pinned for the session.
    Pinned,
    Skipped,
}

/// Single-lock post-landing decision for a whole 64 KiB piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AfterLandDecision {
    /// Piece satisfies `clean_probes` + `QUIET_FRAMES` + not in backoff.
    /// `should_trim` indicates whether to soft-trim if external promotion gates
    /// (e.g. private landing echo or dirty bytes) defer promotion on this frame.
    ReadyToPromote { should_trim: bool },
    /// Normal piece not yet ready to promote that should be soft-trimmed now.
    NeedTrim(LandBlockReason),
    /// Piece should not be trimmed on this frame (already trimmed, in backoff,
    /// permanently pinned, or already `GpuOnly`).
    Hold(LandBlockReason),
}

/// Compact per-shadow 64 KiB piece residency tracker.
///
/// Hot-path queries ([`Self::any_gpu_only`], [`Self::is_gpu_only`],
/// [`Self::non_gpu_only_ranges`]) are completely lock-free using `active_count`
/// and `gpu_only_bits`.
pub(crate) struct Tracker {
    start: usize,
    len: usize,
    piece_count: usize,
    /// Whether this shadow has been registered in [`TRACKED_SHADOWS`].
    registered: AtomicBool,
    /// Number of pieces currently in `PieceState::GpuOnly`.
    active_count: AtomicU32,
    /// Lock-free bitmap of `GpuOnly` pieces (1 bit per 64 KiB piece).
    gpu_only_bits: Box<[AtomicU64]>,
    /// Per-piece state machine, lazily populated only for eligible shadows.
    states: Mutex<Vec<PieceState>>,
}

impl Drop for Tracker {
    fn drop(&mut self) {
        let active = *self.active_count.get_mut();
        if active > 0 {
            ACTIVE_GPU_ONLY_PIECES.fetch_sub(u64::from(active), Ordering::Relaxed);
        }
    }
}

impl Tracker {
    /// Constructs an empty (ineligible / zero-allocation) tracker.
    pub(crate) fn empty() -> Self {
        Self {
            start: 0,
            len: 0,
            piece_count: 0,
            registered: AtomicBool::new(false),
            active_count: AtomicU32::new(0),
            gpu_only_bits: Box::new([]),
            states: Mutex::new(Vec::new()),
        }
    }

    /// Constructs a tracker for a carded shadow `[start, start + len)`.
    /// Shadows under `64 KiB` (or when `VICTORY_GPU_ONLY_BACK=0`) return an
    /// empty zero-allocation tracker (Rule 1).
    pub(crate) fn new(start: usize, len: usize) -> Self {
        if !enabled() || len < PIECE {
            return Self::empty();
        }
        Self::with_frame(start, len, crate::frame::current())
    }

    /// Constructs a tracker with an explicit creation frame (deterministic for
    /// unit tests and explicit initialization).
    pub(crate) fn with_frame(start: usize, len: usize, created_frame: u64) -> Self {
        if len < PIECE {
            return Self::empty();
        }
        let first_block = start / PIECE;
        let last_block = (start + len - 1) / PIECE;
        let piece_count = last_block - first_block + 1;
        let words = piece_count.div_ceil(64);
        let gpu_only_bits: Vec<AtomicU64> = (0..words).map(|_| AtomicU64::new(0)).collect();
        let initial = PieceState::Normal {
            clean_probes: 0,
            last_cpu_write_frame: created_frame,
            last_step_frame: created_frame,
            trimmed_at_frame: 0,
            trimmed: false,
            unflagged_probe: false,
            touch_strikes: 0,
            backoff_until_frame: 0,
        };
        Self {
            start,
            len,
            piece_count,
            registered: AtomicBool::new(false),
            active_count: AtomicU32::new(0),
            gpu_only_bits: gpu_only_bits.into_boxed_slice(),
            states: Mutex::new(vec![initial; piece_count]),
        }
    }

    /// Whether this shadow has already been registered in [`TRACKED_SHADOWS`].
    #[inline]
    pub(crate) fn is_registered(&self) -> bool {
        self.registered.load(Ordering::Acquire)
    }

    /// Marks this tracker as registered in [`TRACKED_SHADOWS`], returning `true`
    /// only on the first call.
    #[inline]
    pub(crate) fn mark_registered(&self) -> bool {
        !self.registered.load(Ordering::Acquire) && !self.registered.swap(true, Ordering::AcqRel)
    }

    /// Whether this shadow is eligible by size (Rule 1: `>= 64 KiB` and enabled).
    #[inline]
    pub(crate) fn eligible(&self) -> bool {
        self.piece_count > 0
    }

    #[inline]
    pub(crate) fn piece_count(&self) -> usize {
        self.piece_count
    }

    /// Lock-free check whether any piece on this shadow is currently `GpuOnly`.
    #[inline]
    pub(crate) fn any_gpu_only(&self) -> bool {
        self.active_count.load(Ordering::Acquire) != 0
    }

    /// Lock-free check whether `piece` is currently `GpuOnly`.
    #[inline]
    pub(crate) fn is_gpu_only(&self, piece: usize) -> bool {
        if !self.any_gpu_only() || piece >= self.piece_count {
            return false;
        }
        (self.gpu_only_bits[piece / 64].load(Ordering::Acquire) >> (piece % 64)) & 1 != 0
    }

    /// Piece index range covering title addresses `[from_addr, to_addr)`, clipped
    /// to this shadow.
    fn piece_range(&self, from_addr: usize, to_addr: usize) -> std::ops::Range<usize> {
        if self.piece_count == 0 {
            return 0..0;
        }
        let end = self.start + self.len;
        let a = from_addr.max(self.start);
        let b = to_addr.min(end);
        if a >= b {
            return 0..0;
        }
        let first_block = self.start / PIECE;
        let lo = a / PIECE - first_block;
        let hi = ((b - 1) / PIECE - first_block + 1).min(self.piece_count);
        lo..hi
    }

    /// Title address range `[lo, hi)` covered by `piece` inside this shadow.
    fn piece_title_bounds(&self, piece: usize) -> (usize, usize) {
        let block = (self.start / PIECE + piece) * PIECE;
        let lo = block.max(self.start);
        let hi = (block + PIECE).min(self.start + self.len);
        (lo, hi)
    }

    /// Returns `Some(block_addr)` iff `piece` covers the entire 64 KiB title
    /// block `[block_addr, block_addr + PIECE)` wholly inside this shadow.
    /// Partial boundary pieces never promote so we never trim or demote memory
    /// outside `[self.start, self.start + self.len)`.
    #[inline]
    pub(crate) fn piece_block_if_whole(&self, piece: usize) -> Option<usize> {
        if piece >= self.piece_count {
            return None;
        }
        let block = (self.start / PIECE + piece) * PIECE;
        (block >= self.start && block + PIECE <= self.start + self.len).then_some(block)
    }

    /// Lock-free split of title address range `[from_addr, to_addr)` into
    /// contiguous subranges excluding all `GpuOnly` pieces, plus the count of
    /// `GpuOnly` pieces excluded.
    pub(crate) fn non_gpu_only_ranges(&self, from_addr: usize, to_addr: usize) -> (Vec<(usize, usize)>, usize) {
        let a = from_addr.max(self.start);
        let b = to_addr.min(self.start + self.len);
        if a >= b {
            return (Vec::new(), 0);
        }
        if !self.any_gpu_only() {
            return (vec![(a, b)], 0);
        }
        let pieces = self.piece_range(a, b);
        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut skipped = 0usize;
        for piece in pieces {
            if self.is_gpu_only(piece) {
                skipped += 1;
                continue;
            }
            let (p_lo, p_hi) = self.piece_title_bounds(piece);
            let lo = p_lo.max(a);
            let hi = p_hi.min(b);
            if lo >= hi {
                continue;
            }
            match out.last_mut() {
                Some(last) if last.1 == lo => last.1 = hi,
                _ => out.push((lo, hi)),
            }
        }
        (out, skipped)
    }

    /// Lock-free list of all `GpuOnly` piece indices overlapping `[from_addr, to_addr)`.
    pub(crate) fn gpu_only_pieces_in(&self, from_addr: usize, to_addr: usize) -> Vec<usize> {
        if !self.any_gpu_only() {
            return Vec::new();
        }
        let range = self.piece_range(from_addr, to_addr);
        if range.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let first_word = range.start / 64;
        let last_word = (range.end - 1) / 64;
        for w in first_word..=last_word {
            let mut bits = self.gpu_only_bits[w].load(Ordering::Acquire);
            if bits == 0 {
                continue;
            }
            let base = w * 64;
            if w == first_word {
                let shift = range.start - base;
                bits &= !0u64 << shift;
            }
            if w == last_word {
                let keep = range.end - base;
                if keep < 64 {
                    bits &= (1u64 << keep) - 1;
                }
            }
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                out.push(base + tz);
                bits &= bits - 1;
            }
        }
        out
    }

    /// Lock-free list of every active `GpuOnly` piece index on this shadow.
    pub(crate) fn all_gpu_only_pieces(&self) -> Vec<usize> {
        if !self.any_gpu_only() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (w, word) in self.gpu_only_bits.iter().enumerate() {
            let mut bits = word.load(Ordering::Acquire);
            let base = w * 64;
            while bits != 0 {
                let tz = bits.trailing_zeros() as usize;
                let piece = base + tz;
                if piece < self.piece_count {
                    out.push(piece);
                }
                bits &= bits - 1;
            }
        }
        out
    }

    /// Records a CPU write to title addresses `[from_addr, to_addr)` on `frame`
    /// (Rule 3: resets `clean_probes` and updates `last_cpu_write_frame`).
    pub(crate) fn note_cpu_write(&self, from_addr: usize, to_addr: usize, frame: u64) {
        let range = self.piece_range(from_addr, to_addr);
        if range.is_empty() {
            return;
        }
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        for piece in range {
            if let Some(PieceState::Normal {
                clean_probes,
                last_cpu_write_frame,
                trimmed,
                ..
            }) = states.get_mut(piece)
            {
                *clean_probes = 0;
                *last_cpu_write_frame = (*last_cpu_write_frame).max(frame);
                *trimmed = false;
            }
        }
    }

    /// Batched single-lock CPU write notification for shadow-relative `(lo, hi)`
    /// offset runs (avoids locking `states` tens of thousands of times per frame
    /// when `merge_under_writes` or `take_landed` produces many dword runs).
    pub(crate) fn note_cpu_write_offsets(&self, runs: &[(usize, usize)], frame: u64) {
        if self.piece_count == 0 || runs.is_empty() {
            return;
        }
        let first_block = self.start / PIECE;
        let end = self.start + self.len;
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let mut last_piece = usize::MAX;
        for &(lo_off, hi_off) in runs {
            if lo_off >= hi_off {
                continue;
            }
            let a = self.start.saturating_add(lo_off).max(self.start);
            let b = self.start.saturating_add(hi_off).min(end);
            if a >= b {
                continue;
            }
            let lo = a / PIECE - first_block;
            let hi = ((b - 1) / PIECE - first_block + 1).min(self.piece_count);
            if lo + 1 == hi && lo == last_piece {
                continue;
            }
            for piece in lo..hi {
                if piece == last_piece {
                    continue;
                }
                if let Some(PieceState::Normal {
                    clean_probes,
                    last_cpu_write_frame,
                    trimmed,
                    ..
                }) = states.get_mut(piece)
                {
                    *clean_probes = 0;
                    *last_cpu_write_frame = (*last_cpu_write_frame).max(frame);
                    *trimmed = false;
                }
                last_piece = piece;
            }
        }
    }

    /// Batched single-lock CPU write notification for an explicit list of piece indices.
    pub(crate) fn note_cpu_write_pieces(&self, pieces: &[usize], frame: u64) {
        if self.piece_count == 0 || pieces.is_empty() {
            return;
        }
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        for &piece in pieces {
            if let Some(PieceState::Normal {
                clean_probes,
                last_cpu_write_frame,
                trimmed,
                ..
            }) = states.get_mut(piece)
            {
                *clean_probes = 0;
                *last_cpu_write_frame = (*last_cpu_write_frame).max(frame);
                *trimmed = false;
            }
        }
    }

    /// Atomically records sorted overlapping CPU step ranges (`Copy`/`Placed`/
    /// `Fill`/`Texture`/`Plane`/`Host`) in a SINGLE `states` lock acquisition
    /// and returns any overlapping `GpuOnly` pieces (deduplicated).
    pub(crate) fn note_steps_and_collect_gpu_only(&self, ranges: &[(usize, usize)], frame: u64) -> Vec<usize> {
        if self.piece_count == 0 || ranges.is_empty() {
            return Vec::new();
        }
        let mut to_demote = Vec::new();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let mut last_piece = usize::MAX;
        for &(from_addr, to_addr) in ranges {
            let range = self.piece_range(from_addr, to_addr);
            if range.is_empty() {
                continue;
            }
            let start = if last_piece != usize::MAX && range.start <= last_piece {
                last_piece + 1
            } else {
                range.start
            };
            for piece in start..range.end {
                match states.get_mut(piece) {
                    Some(PieceState::Normal {
                        clean_probes,
                        last_step_frame,
                        trimmed,
                        ..
                    }) => {
                        *clean_probes = 0;
                        *last_step_frame = (*last_step_frame).max(frame);
                        *trimmed = false;
                    }
                    Some(PieceState::GpuOnly { .. }) => {
                        to_demote.push(piece);
                    }
                    _ => {}
                }
                last_piece = piece;
            }
        }
        to_demote
    }

    /// Atomically pins all pieces overlapping `[from_addr, to_addr)` (used when
    /// a new overlapping shadow is created or memory is remapped) and returns
    /// any pieces that were `GpuOnly` and therefore need lossless readback.
    pub(crate) fn pin_range_and_demote(&self, from_addr: usize, to_addr: usize, reason: DemoteReason) -> Vec<usize> {
        let range = self.piece_range(from_addr, to_addr);
        if range.is_empty() {
            return Vec::new();
        }
        let pin_reason = PinReason::from(reason);
        let mut demoted = Vec::new();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        for piece in range {
            let Some(st) = states.get_mut(piece) else {
                continue;
            };
            if matches!(*st, PieceState::GpuOnly { .. }) {
                self.clear_gpu_only_bit(piece);
                demoted.push(piece);
                *st = PieceState::Pinned(pin_reason);
            } else if !matches!(*st, PieceState::Pinned(_)) {
                *st = PieceState::Pinned(pin_reason);
            }
        }
        drop(states);
        if !demoted.is_empty() {
            let n = demoted.len() as u64;
            self.active_count.fetch_sub(n as u32, Ordering::AcqRel);
            ACTIVE_GPU_ONLY_PIECES.fetch_sub(n, Ordering::Relaxed);
            record_demote(reason, n);
        }
        demoted
    }

    /// Transitions each `GpuOnly` piece in `pieces` to `Pinned` (clearing its
    /// lock-free bitmap bit and updating global counts) and returns the subset
    /// of `pieces` that actually transitioned from `GpuOnly`.
    pub(crate) fn demote_and_pin(&self, pieces: &[usize], reason: DemoteReason) -> Vec<usize> {
        if self.piece_count == 0 || pieces.is_empty() {
            return Vec::new();
        }
        let pin_reason = PinReason::from(reason);
        let mut demoted = Vec::new();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        for &piece in pieces {
            let Some(st) = states.get_mut(piece) else {
                continue;
            };
            if matches!(*st, PieceState::GpuOnly { .. }) {
                self.clear_gpu_only_bit(piece);
                *st = PieceState::Pinned(pin_reason);
                demoted.push(piece);
            } else if !matches!(*st, PieceState::Pinned(_)) {
                *st = PieceState::Pinned(pin_reason);
            }
        }
        drop(states);
        if !demoted.is_empty() {
            let n = demoted.len() as u64;
            self.active_count.fetch_sub(n as u32, Ordering::AcqRel);
            ACTIVE_GPU_ONLY_PIECES.fetch_sub(n, Ordering::Relaxed);
            record_demote(reason, n);
        }
        demoted
    }

    /// Decides what working-set action to take before a landing or `take_landed`
    /// reads/writes title memory for `piece` at `frame`.
    ///
    /// - If trimmed on an earlier frame (`trimmed_at_frame < frame`), returns
    ///   [`BeforeReadAction::Probe`].
    /// - If trimmed earlier on the same frame (`trimmed_at_frame >= frame`),
    ///   clears `trimmed = false` immediately and returns
    ///   [`BeforeReadAction::ClearedSameFrame`] so our own memory access does
    ///   not leave resident pages under `trimmed = true` and trigger a false
    ///   working-set pin on the next frame.
    pub(crate) fn before_read_action(&self, piece: usize, frame: u64) -> BeforeReadAction {
        if piece >= self.piece_count {
            return BeforeReadAction::Skip;
        }
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.get_mut(piece) {
            Some(PieceState::Normal {
                trimmed: true,
                trimmed_at_frame,
                ..
            }) if *trimmed_at_frame < frame => BeforeReadAction::Probe,
            Some(PieceState::Normal {
                trimmed,
                trimmed_at_frame,
                ..
            }) if *trimmed && *trimmed_at_frame >= frame => {
                *trimmed = false;
                BeforeReadAction::ClearedSameFrame
            }
            _ => BeforeReadAction::Skip,
        }
    }

    /// Decides what unflagged probe action (if any) to take for `piece` at `frame`.
    pub(crate) fn unflagged_action(&self, piece: usize, frame: u64) -> UnflaggedAction {
        if self.len < UNFLAGGED_MIN_SHADOW_LEN || self.piece_block_if_whole(piece).is_none() {
            return UnflaggedAction::Skip;
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.get(piece) {
            Some(PieceState::Normal {
                trimmed: false,
                last_cpu_write_frame,
                last_step_frame,
                backoff_until_frame,
                ..
            }) => {
                // Start probing once the piece has been quiet for a few frames
                // and is outside any pre-promotion working-set touch cooldown.
                if frame >= *backoff_until_frame
                    && frame.saturating_sub(*last_cpu_write_frame) >= 8
                    && frame.saturating_sub(*last_step_frame) >= 8
                {
                    UnflaggedAction::NeedTrim
                } else {
                    UnflaggedAction::Skip
                }
            }
            Some(PieceState::Normal {
                trimmed: true,
                trimmed_at_frame,
                backoff_until_frame,
                ..
            }) if frame >= *backoff_until_frame
                && frame >= trimmed_at_frame.saturating_add(UNFLAGGED_MIN_PROBE_GAP) =>
            {
                UnflaggedAction::NeedQuery
            }
            _ => UnflaggedAction::Skip,
        }
    }

    /// Applies the result of a `K32QueryWorkingSetEx` probe for `piece`.
    /// - `touched == None`: sample discarded safely (remapping busy); clears `trimmed`.
    /// - `touched == Some(0)`: increments `clean_probes`, updates `trimmed_at_frame`
    ///   if keeping trimmed (`!clear_trimmed`), else clears `trimmed`.
    /// - `touched == Some(n > 0)`: resets `clean_probes = 0`, clears `trimmed`,
    ///   and increments `touch_strikes`. Strikes 1..`TOUCH_STRIKE_LIMIT` enter
    ///   exponential cooldown backoff (`TOUCH_BACKOFF_FRAMES`, incrementing
    ///   `STATS[8]`); strike `TOUCH_STRIKE_LIMIT` permanently transitions to
    ///   `PieceState::Pinned(PinReason::Touch)` and increments `STATS[7]`.
    pub(crate) fn apply_probe_result(
        &self,
        piece: usize,
        frame: u64,
        touched: Option<u32>,
        clear_trimmed: bool,
    ) -> ProbeOutcome {
        if piece >= self.piece_count {
            return ProbeOutcome::Skipped;
        }
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let Some(st) = states.get_mut(piece) else {
            return ProbeOutcome::Skipped;
        };
        let PieceState::Normal {
            clean_probes,
            trimmed,
            trimmed_at_frame,
            touch_strikes,
            backoff_until_frame,
            ..
        } = st
        else {
            return ProbeOutcome::Skipped;
        };
        if !*trimmed {
            return ProbeOutcome::Skipped;
        }
        let Some(touched) = touched else {
            *trimmed = false;
            return ProbeOutcome::Skipped;
        };
        if touched == 0 {
            *clean_probes = clean_probes.saturating_add(1);
            if clear_trimmed {
                *trimmed = false;
            } else {
                *trimmed_at_frame = frame;
            }
            ProbeOutcome::Untouched(*clean_probes)
        } else {
            *clean_probes = 0;
            *trimmed = false;
            *touch_strikes = touch_strikes.saturating_add(1);
            let strikes = *touch_strikes;
            if strikes >= TOUCH_STRIKE_LIMIT {
                *st = PieceState::Pinned(PinReason::Touch);
                drop(states);
                STATS[7].fetch_add(1, Ordering::Relaxed);
                ProbeOutcome::Pinned
            } else {
                let wait = TOUCH_BACKOFF_FRAMES
                    .get((strikes - 1) as usize)
                    .copied()
                    .unwrap_or(600);
                *backoff_until_frame = frame.saturating_add(wait);
                drop(states);
                STATS[8].fetch_add(1, Ordering::Relaxed);
                ProbeOutcome::BackedOff(strikes)
            }
        }
    }

    /// Single-lock post-landing decision combining promotion readiness, trim
    /// eligibility, and exact landing-blocker classification for `piece`.
    pub(crate) fn after_land_decision(&self, piece: usize, frame: u64, force_retrim: bool) -> AfterLandDecision {
        if self.piece_block_if_whole(piece).is_none() {
            return AfterLandDecision::Hold(LandBlockReason::PartialPiece);
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.get(piece) {
            Some(PieceState::Normal {
                clean_probes,
                last_cpu_write_frame,
                last_step_frame,
                trimmed,
                trimmed_at_frame,
                backoff_until_frame,
                ..
            }) => {
                if frame < *backoff_until_frame {
                    return AfterLandDecision::Hold(LandBlockReason::Backoff);
                }
                let should_trim = force_retrim || !*trimmed || *trimmed_at_frame < frame;
                let quiet = frame.saturating_sub(*last_cpu_write_frame) >= QUIET_FRAMES
                    && frame.saturating_sub(*last_step_frame) >= QUIET_FRAMES;
                if quiet && *clean_probes >= LANDED_CLEAN_PROBES {
                    AfterLandDecision::ReadyToPromote { should_trim }
                } else {
                    let reason = if !quiet {
                        LandBlockReason::Unquiet
                    } else {
                        LandBlockReason::Probing
                    };
                    if should_trim {
                        AfterLandDecision::NeedTrim(reason)
                    } else {
                        AfterLandDecision::Hold(reason)
                    }
                }
            }
            Some(PieceState::Pinned(reason)) => AfterLandDecision::Hold(LandBlockReason::from(*reason)),
            Some(PieceState::GpuOnly { .. }) => AfterLandDecision::Hold(LandBlockReason::Promoted),
            None => AfterLandDecision::Hold(LandBlockReason::PartialPiece),
        }
    }

    /// Whether `piece` meets Rule 3 & Rule 4 counts for promotion at `frame`
    /// (caller must also verify Rule 1, Rule 2, `copied`, and working-set trim).
    pub(crate) fn ready_to_promote(&self, piece: usize, frame: u64, unflagged: bool) -> bool {
        if (unflagged && self.len < UNFLAGGED_MIN_SHADOW_LEN) || self.piece_block_if_whole(piece).is_none() {
            return false;
        }
        let needed = if unflagged {
            UNFLAGGED_CLEAN_PROBES
        } else {
            LANDED_CLEAN_PROBES
        };
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        matches!(
            states.get(piece),
            Some(PieceState::Normal {
                clean_probes,
                last_cpu_write_frame,
                last_step_frame,
                backoff_until_frame,
                ..
            }) if frame >= *backoff_until_frame
                && *clean_probes >= needed
                && frame.saturating_sub(*last_cpu_write_frame) >= QUIET_FRAMES
                && frame.saturating_sub(*last_step_frame) >= QUIET_FRAMES
        )
    }

    /// Whether `piece` is `Normal`, not in touch backoff, and should be
    /// soft-trimmed via `VirtualUnlock`. Passing `force_retrim: true` (e.g.,
    /// after a landing writes title memory directly) forces re-trimming even if
    /// `piece` was already trimmed earlier on the same `frame`.
    pub(crate) fn needs_trim(&self, piece: usize, frame: u64, force_retrim: bool) -> bool {
        if self.piece_block_if_whole(piece).is_none() {
            return false;
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        matches!(
            states.get(piece),
            Some(PieceState::Normal {
                trimmed,
                trimmed_at_frame,
                backoff_until_frame,
                ..
            }) if frame >= *backoff_until_frame
                && (force_retrim || !*trimmed || *trimmed_at_frame < frame)
        )
    }

    /// Notes that `piece` was successfully soft-trimmed at `frame`.
    pub(crate) fn mark_trimmed(&self, piece: usize, frame: u64, unflagged: bool) {
        if piece >= self.piece_count {
            return;
        }
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(PieceState::Normal {
            trimmed,
            trimmed_at_frame,
            unflagged_probe,
            ..
        }) = states.get_mut(piece)
        {
            *trimmed = true;
            *trimmed_at_frame = frame;
            *unflagged_probe = unflagged;
        }
    }

    /// Final linearizable promotion commit: re-verifies `Normal` state and
    /// thresholds under `states` lock so a concurrent `note_cpu_write`,
    /// `note_steps_and_collect_gpu_only`, or `pin_range_and_demote` can never
    /// race with promotion.
    pub(crate) fn try_promote(&self, piece: usize, frame: u64, unflagged: bool) -> bool {
        if (unflagged && self.len < UNFLAGGED_MIN_SHADOW_LEN) || self.piece_block_if_whole(piece).is_none() {
            return false;
        }
        let needed = if unflagged {
            UNFLAGGED_CLEAN_PROBES
        } else {
            LANDED_CLEAN_PROBES
        };
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let Some(st) = states.get_mut(piece) else {
            return false;
        };
        let ok = matches!(
            *st,
            PieceState::Normal {
                clean_probes,
                last_cpu_write_frame,
                last_step_frame,
                backoff_until_frame,
                ..
            } if frame >= backoff_until_frame
                && clean_probes >= needed
                && frame.saturating_sub(last_cpu_write_frame) >= QUIET_FRAMES
                && frame.saturating_sub(last_step_frame) >= QUIET_FRAMES
        );
        if !ok {
            return false;
        }
        *st = PieceState::GpuOnly { promoted_frame: frame };
        self.set_gpu_only_bit(piece);
        drop(states);
        self.active_count.fetch_add(1, Ordering::AcqRel);
        ACTIVE_GPU_ONLY_PIECES.fetch_add(1, Ordering::Relaxed);
        STATS[2].fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Collects up to `limit` active `GpuOnly` `(piece, block_addr)` entries
    /// starting at `start_piece` (lock-free bitmap scan) for the rotating
    /// working-set watchdog.
    pub(crate) fn collect_watchdog_candidates(&self, start_piece: usize, limit: usize) -> Vec<(usize, usize)> {
        if !self.any_gpu_only() || limit == 0 || self.piece_count == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        let start = start_piece % self.piece_count;
        let mut scan_range = |from: usize, to: usize| {
            if from >= to || out.len() >= limit {
                return;
            }
            let first_word = from / 64;
            let last_word = (to - 1) / 64;
            for w in first_word..=last_word {
                let mut bits = self.gpu_only_bits[w].load(Ordering::Acquire);
                if bits == 0 {
                    continue;
                }
                let base = w * 64;
                if w == first_word {
                    bits &= !0u64 << (from - base);
                }
                if w == last_word {
                    let keep = to - base;
                    if keep < 64 {
                        bits &= (1u64 << keep) - 1;
                    }
                }
                while bits != 0 && out.len() < limit {
                    let tz = bits.trailing_zeros() as usize;
                    let piece = base + tz;
                    if let Some(block) = self.piece_block_if_whole(piece) {
                        out.push((piece, block));
                    }
                    bits &= bits - 1;
                }
                if out.len() >= limit {
                    break;
                }
            }
        };
        scan_range(start, self.piece_count);
        if start > 0 {
            scan_range(0, start);
        }
        out
    }

    #[inline]
    fn set_gpu_only_bit(&self, piece: usize) {
        self.gpu_only_bits[piece / 64].fetch_or(1u64 << (piece % 64), Ordering::Release);
    }

    #[inline]
    fn clear_gpu_only_bit(&self, piece: usize) {
        self.gpu_only_bits[piece / 64].fetch_and(!(1u64 << (piece % 64)), Ordering::Release);
    }

    #[cfg(test)]
    fn state_of(&self, piece: usize) -> Option<PieceState> {
        self.states.lock().unwrap_or_else(|e| e.into_inner()).get(piece).copied()
    }
}

/// Result of lossless byte-level merge on demotion of a `GpuOnly` piece.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DemoteMerge {
    /// Final merged bytes for both `shadow.data` and title memory.
    pub(crate) merged: Vec<u8>,
    /// True if `merged` differs from `title` (GPU bytes must be written back to
    /// title memory via `put_changed`).
    pub(crate) title_changed: bool,
    /// Offset ranges within the piece where `merged != gpu` (unexpected CPU
    /// writes while `GpuOnly`, which must be uploaded to `shadow.buffer` via
    /// `card_changed_runs` so the GPU sees CPU's bytes!).
    pub(crate) cpu_dirty_runs: Vec<(usize, usize)>,
}

/// Pure lossless byte-merge for demoting a `GpuOnly` piece:
/// - `gpu`: latest bytes read back synchronously from `shadow.buffer`.
/// - `old_data`: pre-`GpuOnly` snapshot still held in `shadow.data`.
/// - `title`: current bytes read from title memory.
///
/// For each byte `i`:
/// - If `title[i] != old_data[i]` (CPU wrote byte `i` while `GpuOnly`): keep
///   `title[i]` (and record in `cpu_dirty_runs` if `title[i] != gpu[i]`).
/// - Else (CPU did not modify byte `i`): take `gpu[i]` (and set `title_changed`
///   if `gpu[i] != title[i]`).
pub(crate) fn merge_demoted_piece(gpu: &[u8], old_data: &[u8], title: &[u8]) -> DemoteMerge {
    let n = gpu.len().min(old_data.len()).min(title.len());
    if title[..n] == old_data[..n] {
        let title_changed = gpu[..n] != title[..n];
        return DemoteMerge {
            merged: gpu[..n].to_vec(),
            title_changed,
            cpu_dirty_runs: Vec::new(),
        };
    }
    let mut merged = Vec::with_capacity(n);
    let mut title_changed = false;
    let mut cpu_dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut dirty_start: Option<usize> = None;
    for i in 0..n {
        let cpu_wrote = title[i] != old_data[i];
        let b = if cpu_wrote { title[i] } else { gpu[i] };
        merged.push(b);
        if b != title[i] {
            title_changed = true;
        }
        if b != gpu[i] {
            dirty_start.get_or_insert(i);
        } else if let Some(s) = dirty_start.take() {
            cpu_dirty_runs.push((s, i));
        }
    }
    if let Some(s) = dirty_start {
        cpu_dirty_runs.push((s, n));
    }
    DemoteMerge {
        merged,
        title_changed,
        cpu_dirty_runs,
    }
}

/// Zero-fault working-set query on all 16 4 KiB pages of `[block, block + 64 KiB)`.
///
/// Returns `Some((touched_pages, 16))` on success, or `None` if title memory
/// remapping is currently active or Windows did not answer. Never faults and
/// never touches any page.
pub(crate) fn query_working_set_block(block: usize) -> Option<(u32, u32)> {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let _still = crate::watch::title_memory_reading_now()?;
    // SAFETY: plain zeroed Win32 struct array.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; PAGES_PER_PIECE] = unsafe { std::mem::zeroed() };
    for (i, entry) in info.iter_mut().enumerate() {
        entry.VirtualAddress = (block + i * KERNEL_PAGE) as *mut _;
    }
    // SAFETY: `PAGES_PER_PIECE` valid entries of exact byte size for our own
    // process. `K32QueryWorkingSetEx` inspects page table state without
    // touching or faulting the queried pages.
    let ok = unsafe {
        K32QueryWorkingSetEx(
            GetCurrentProcess(),
            info.as_mut_ptr().cast(),
            (PAGES_PER_PIECE * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
        )
    } != 0;
    if !ok {
        return None;
    }
    let mut touched = 0u32;
    for entry in &info {
        // Bit 0 (`Valid`): page is currently resident in the process working set.
        // SAFETY: plain union field filled in by `K32QueryWorkingSetEx`.
        if unsafe { entry.VirtualAttributes.Flags } & 1 != 0 {
            touched += 1;
        }
    }
    Some((touched, PAGES_PER_PIECE as u32))
}

/// Soft-trims all 16 4 KiB pages of `[block, block + 64 KiB)` out of the
/// process working set via `VirtualUnlock` on unlocked pages (zero faults,
/// zero protection changes, RAM data intact). Returns `false` with nothing
/// done while the runtime is actively remapping title memory.
pub(crate) fn trim_block(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::VirtualUnlock;

    let Some(_still) = crate::watch::title_memory_reading_now() else {
        return false;
    };
    // SAFETY: `VirtualUnlock` on unlocked pages is Windows' documented way to
    // soft-trim pages from the working set without changing protections or
    // discarding contents; an unmapped address fails cleanly and does nothing.
    unsafe { VirtualUnlock(block as *const _, PIECE) };
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn promote_piece(tracker: &Tracker, piece: usize, frame: u64, unflagged: bool) -> bool {
        let probes = if unflagged {
            UNFLAGGED_CLEAN_PROBES
        } else {
            LANDED_CLEAN_PROBES
        };
        for i in 0..probes {
            let f = u64::from(i) * 2 + 1;
            tracker.mark_trimmed(piece, f, unflagged);
            assert_eq!(
                tracker.apply_probe_result(piece, f + 2, Some(0), !unflagged),
                ProbeOutcome::Untouched(i + 1)
            );
        }
        tracker.try_promote(piece, frame, unflagged)
    }

    #[test]
    fn non_gpu_only_ranges_splits_all_cases() {
        let base = 0x1000_0000usize;
        let tracker = Tracker::with_frame(base, 5 * PIECE, 0);

        // None GpuOnly: whole range returned untouched, 0 skipped.
        assert_eq!(tracker.non_gpu_only_ranges(base, base + 5 * PIECE), (vec![(base, base + 5 * PIECE)], 0));

        // Promote middle piece (piece 2) and right piece (piece 4).
        assert!(promote_piece(&tracker, 2, QUIET_FRAMES, false));
        assert!(promote_piece(&tracker, 4, QUIET_FRAMES, false));
        assert!(tracker.any_gpu_only());
        assert!(tracker.is_gpu_only(2));
        assert!(tracker.is_gpu_only(4));
        assert!(!tracker.is_gpu_only(1));

        // Middle & right excluded.
        assert_eq!(
            tracker.non_gpu_only_ranges(base, base + 5 * PIECE),
            (
                vec![(base, base + 2 * PIECE), (base + 3 * PIECE, base + 4 * PIECE)],
                2
            )
        );

        // Unaligned offsets crossing GpuOnly piece 2.
        assert_eq!(
            tracker.non_gpu_only_ranges(base + PIECE + 123, base + 3 * PIECE + 456),
            (
                vec![
                    (base + PIECE + 123, base + 2 * PIECE),
                    (base + 3 * PIECE, base + 3 * PIECE + 456)
                ],
                1
            )
        );

        // Range strictly inside a single GpuOnly piece -> empty ranges, 1 skipped.
        assert_eq!(
            tracker.non_gpu_only_ranges(base + 2 * PIECE + 64, base + 2 * PIECE + 512),
            (vec![], 1)
        );

        // Left excluded (promote piece 0 as well).
        assert!(promote_piece(&tracker, 0, QUIET_FRAMES, false));
        assert_eq!(
            tracker.non_gpu_only_ranges(base, base + 2 * PIECE),
            (vec![(base + PIECE, base + 2 * PIECE)], 1)
        );

        // All GpuOnly (promote remaining pieces 1 and 3).
        assert!(promote_piece(&tracker, 1, QUIET_FRAMES, false));
        assert!(promote_piece(&tracker, 3, QUIET_FRAMES, false));
        assert_eq!(tracker.non_gpu_only_ranges(base, base + 5 * PIECE), (vec![], 5));

        // Demoting piece 2 restores it cleanly.
        assert_eq!(tracker.demote_and_pin(&[2], DemoteReason::Step), vec![2]);
        assert!(!tracker.is_gpu_only(2));
        assert_eq!(
            tracker.non_gpu_only_ranges(base, base + 5 * PIECE),
            (vec![(base + 2 * PIECE, base + 3 * PIECE)], 4)
        );
    }

    #[test]
    fn promotion_gate_enforces_size_quiet_and_pin_rules() {
        // Rule 1: < 64 KiB is never eligible.
        let small = Tracker::with_frame(0x2000_0000, PIECE - 1, 0);
        assert!(!small.eligible());
        assert!(!small.try_promote(0, 1000, false));

        // Partial boundary piece on unaligned start cannot promote, interior whole piece can.
        let unaligned = Tracker::with_frame(0x2000_0000 + 4096, 2 * PIECE, 0);
        assert_eq!(unaligned.piece_block_if_whole(0), None);
        assert_eq!(unaligned.piece_block_if_whole(1), Some(0x2001_0000));
        assert_eq!(unaligned.piece_block_if_whole(2), None);
        assert!(!promote_piece(&unaligned, 0, 100, false));
        assert!(promote_piece(&unaligned, 1, 100, false));

        // Rule 3: CPU write within QUIET_FRAMES blocks promotion even with LANDED_CLEAN_PROBES clean probes.
        let tracker = Tracker::with_frame(0x3000_0000, 3 * PIECE, 10);
        // Too early (QUIET_FRAMES - 1 frames since creation at frame 10).
        assert!(!promote_piece(&tracker, 0, 10 + QUIET_FRAMES - 1, false));
        assert!(!tracker.is_gpu_only(0));
        // At frame 10 + QUIET_FRAMES: promotes cleanly!
        assert!(tracker.try_promote(0, 10 + QUIET_FRAMES, false));
        assert!(tracker.is_gpu_only(0));

        // CPU write at frame 40 resets clean_probes and delays quiet window to frame 40 + QUIET_FRAMES.
        for i in 0..LANDED_CLEAN_PROBES {
            tracker.mark_trimmed(1, u64::from(i) + 11, false);
            tracker.apply_probe_result(1, u64::from(i) + 12, Some(0), true);
        }
        tracker.note_cpu_write(0x3001_0000, 0x3001_0010, 40);
        assert!(!tracker.try_promote(1, 40 + QUIET_FRAMES - 1, false));
        assert!(!tracker.try_promote(1, 40 + QUIET_FRAMES, false)); // clean_probes was reset to 0!
        assert!(promote_piece(&tracker, 1, 40 + QUIET_FRAMES, false));
        assert!(tracker.is_gpu_only(1));

        // Rule 4: pre-promotion working-set touches enter bounded exponential
        // backoff (strikes 1 & 2) and permanently pin on strike 3 (`TOUCH_STRIKE_LIMIT`).
        tracker.mark_trimmed(2, 20, false);
        assert_eq!(
            tracker.apply_probe_result(2, 21, Some(1), true),
            ProbeOutcome::BackedOff(1)
        );
        assert!(!tracker.needs_trim(2, 21 + TOUCH_BACKOFF_FRAMES[0] - 1, true));
        assert_eq!(
            tracker.after_land_decision(2, 21 + TOUCH_BACKOFF_FRAMES[0] - 1, false),
            AfterLandDecision::Hold(LandBlockReason::Backoff)
        );
        let f2 = 21 + TOUCH_BACKOFF_FRAMES[0];
        assert!(tracker.needs_trim(2, f2, false));
        tracker.mark_trimmed(2, f2, false);
        assert_eq!(
            tracker.apply_probe_result(2, f2 + 1, Some(2), true),
            ProbeOutcome::BackedOff(2)
        );
        let f3 = f2 + 1 + TOUCH_BACKOFF_FRAMES[1];
        assert!(!tracker.needs_trim(2, f3 - 1, true));
        assert!(tracker.needs_trim(2, f3, false));
        tracker.mark_trimmed(2, f3, false);
        assert_eq!(
            tracker.apply_probe_result(2, f3 + 1, Some(1), true),
            ProbeOutcome::Pinned
        );
        assert_eq!(tracker.state_of(2), Some(PieceState::Pinned(PinReason::Touch)));
        assert!(!tracker.try_promote(2, f3 + 10_000, false));
        assert!(!tracker.try_promote(2, f3 + 10_000, true));
    }

    #[test]
    fn same_frame_before_read_clears_trimmed_without_pinning() {
        let tracker = Tracker::with_frame(0x4000_0000, 2 * PIECE, 0);

        // Untrimmed piece returns Skip.
        assert_eq!(tracker.before_read_action(0, 10), BeforeReadAction::Skip);

        // Trim at frame 10; same-frame read at frame 10 clears `trimmed` without pinning.
        tracker.mark_trimmed(0, 10, false);
        assert_eq!(tracker.before_read_action(0, 10), BeforeReadAction::ClearedSameFrame);
        assert!(matches!(
            tracker.state_of(0),
            Some(PieceState::Normal {
                trimmed: false,
                clean_probes: 0,
                ..
            })
        ));
        // Because `trimmed` was cleared before our own read touched title memory,
        // frame 11 sees `Skip` instead of probing our own resident pages and false-pinning.
        assert_eq!(tracker.before_read_action(0, 11), BeforeReadAction::Skip);
        assert!(!matches!(tracker.state_of(0), Some(PieceState::Pinned(_))));

        // Trim at frame 11; next-frame read at frame 12 requests a working-set probe.
        tracker.mark_trimmed(0, 11, false);
        assert_eq!(tracker.before_read_action(0, 12), BeforeReadAction::Probe);
        assert_eq!(
            tracker.apply_probe_result(0, 12, Some(0), true),
            ProbeOutcome::Untouched(1)
        );
    }

    #[test]
    fn force_retrim_allows_same_frame_retrim_on_normal_only() {
        let tracker = Tracker::with_frame(0x5000_0000, 2 * PIECE, 0);

        // Initially untrimmed -> both false and true request trim.
        assert!(tracker.needs_trim(0, 10, false));
        assert!(tracker.needs_trim(0, 10, true));

        // After trimming on frame 10: without force_retrim same frame is skipped,
        // while `force_retrim: true` (e.g. second landing wrote title memory) re-trims!
        tracker.mark_trimmed(0, 10, false);
        assert!(!tracker.needs_trim(0, 10, false));
        assert!(tracker.needs_trim(0, 10, true));
        assert!(tracker.needs_trim(0, 11, false));

        // Backoff suppresses even force_retrim until cooldown elapses.
        assert_eq!(tracker.apply_probe_result(0, 11, Some(1), true), ProbeOutcome::BackedOff(1));
        assert!(!tracker.needs_trim(0, 11, true));

        // Pinned and GpuOnly pieces never trim, even with force_retrim: true.
        let _ = tracker.demote_and_pin(&[0], DemoteReason::CpuRead);
        assert_eq!(tracker.state_of(0), Some(PieceState::Pinned(PinReason::CpuRead)));
        assert!(!tracker.needs_trim(0, 1000, true));

        assert!(promote_piece(&tracker, 1, QUIET_FRAMES, false));
        assert!(tracker.is_gpu_only(1));
        assert!(!tracker.needs_trim(1, QUIET_FRAMES, true));
    }

    #[test]
    fn pin_touch_stat_increments_on_working_set_touch_and_mark_registered_is_once() {
        let tracker = Tracker::with_frame(0x6000_0000, 2 * PIECE, 0);
        assert!(!tracker.is_registered());
        assert!(tracker.mark_registered());
        assert!(tracker.is_registered());
        assert!(!tracker.mark_registered());

        let before_pin = STATS[7].load(Ordering::Relaxed);
        let before_backoff = STATS[8].load(Ordering::Relaxed);
        // Clean probe and discarded (None) probe do NOT increment pin_touch or backoff.
        tracker.mark_trimmed(0, 5, false);
        assert_eq!(tracker.apply_probe_result(0, 6, Some(0), true), ProbeOutcome::Untouched(1));
        tracker.mark_trimmed(0, 6, false);
        assert_eq!(tracker.apply_probe_result(0, 7, None, true), ProbeOutcome::Skipped);
        // Strikes 1 & 2 increment touch_backoff (`STATS[8]`), strike 3 pins and increments `STATS[7]`.
        let mut f = 7u64;
        for strike in 1..TOUCH_STRIKE_LIMIT {
            tracker.mark_trimmed(0, f, false);
            assert_eq!(tracker.apply_probe_result(0, f + 1, Some(4), true), ProbeOutcome::BackedOff(strike));
            f = f + 1 + TOUCH_BACKOFF_FRAMES[(strike - 1) as usize];
        }
        assert!(STATS[8].load(Ordering::Relaxed) >= before_backoff + u64::from(TOUCH_STRIKE_LIMIT - 1));
        tracker.mark_trimmed(0, f, false);
        assert_eq!(tracker.apply_probe_result(0, f + 1, Some(4), true), ProbeOutcome::Pinned);
        let after_pin = STATS[7].load(Ordering::Relaxed);
        assert!(after_pin > before_pin, "STATS[7] (pin_touch) must increment on strike 3 ({before_pin} -> {after_pin})");

        // Transient single touch during boot/load recovers and promotes cleanly after backoff!
        tracker.mark_trimmed(1, 10, false);
        assert_eq!(tracker.apply_probe_result(1, 11, Some(16), true), ProbeOutcome::BackedOff(1));
        let recover_start = 11 + TOUCH_BACKOFF_FRAMES[0];
        for i in 0..LANDED_CLEAN_PROBES {
            let pf = recover_start + u64::from(i) * 2;
            tracker.mark_trimmed(1, pf, false);
            assert_eq!(
                tracker.apply_probe_result(1, pf + 1, Some(0), true),
                ProbeOutcome::Untouched(i + 1)
            );
        }
        let promote_frame = recover_start + u64::from(LANDED_CLEAN_PROBES) * 2 + QUIET_FRAMES;
        assert!(tracker.try_promote(1, promote_frame, false));
        assert!(tracker.is_gpu_only(1));
    }

    #[test]
    fn batched_cpu_write_offsets_and_steps_deduplicate_pieces_in_one_lock() {
        let base = 0x8000_0000usize;
        let tracker = Tracker::with_frame(base, 4 * PIECE, 0);
        assert!(promote_piece(&tracker, 2, QUIET_FRAMES, false));
        assert!(tracker.is_gpu_only(2));

        // Many tiny runs inside pieces 0 and 1 reset clean_probes & trimmed once.
        tracker.mark_trimmed(0, 10, false);
        tracker.mark_trimmed(1, 10, false);
        let runs: Vec<(usize, usize)> = (0..1000)
            .map(|i| (i * 16, i * 16 + 4))
            .chain((0..1000).map(|i| (PIECE + i * 16, PIECE + i * 16 + 4)))
            .collect();
        tracker.note_cpu_write_offsets(&runs, 50);
        assert!(matches!(
            tracker.state_of(0),
            Some(PieceState::Normal {
                trimmed: false,
                clean_probes: 0,
                last_cpu_write_frame: 50,
                ..
            })
        ));
        assert!(matches!(
            tracker.state_of(1),
            Some(PieceState::Normal {
                trimmed: false,
                clean_probes: 0,
                last_cpu_write_frame: 50,
                ..
            })
        ));
        // Piece 3 untouched.
        assert!(matches!(
            tracker.state_of(3),
            Some(PieceState::Normal {
                last_cpu_write_frame: 0,
                ..
            })
        ));

        // Sorted overlapping step ranges touching piece 1 (Normal) and piece 2 (GpuOnly).
        let step_ranges = [
            (base + PIECE + 100, base + PIECE + 200),
            (base + PIECE + 300, base + 2 * PIECE + 64),
            (base + 2 * PIECE + 128, base + 2 * PIECE + 512),
        ];
        let hit = tracker.note_steps_and_collect_gpu_only(&step_ranges, 60);
        assert_eq!(hit, vec![2], "GpuOnly piece 2 must be reported exactly once");
        assert!(matches!(
            tracker.state_of(1),
            Some(PieceState::Normal {
                last_step_frame: 60,
                ..
            })
        ));
    }

    #[test]
    fn unflagged_promotion_skips_small_buffers_below_256k() {
        // 192 KiB (< UNFLAGGED_MIN_SHADOW_LEN = 256 KiB): unflagged probing and
        // unflagged promotion are skipped, while landed promotion still works.
        let small = Tracker::with_frame(0x7000_0000, 3 * PIECE, 0);
        assert_eq!(small.unflagged_action(0, 100), UnflaggedAction::Skip);
        assert!(!promote_piece(&small, 0, QUIET_FRAMES, true));
        assert!(!small.ready_to_promote(0, QUIET_FRAMES, true));
        assert!(!small.is_gpu_only(0));
        assert!(promote_piece(&small, 1, QUIET_FRAMES, false));
        assert!(small.is_gpu_only(1));

        // 256 KiB (== UNFLAGGED_MIN_SHADOW_LEN): unflagged probing & promotion work.
        let large = Tracker::with_frame(0x7100_0000, UNFLAGGED_MIN_SHADOW_LEN, 0);
        assert_eq!(large.unflagged_action(0, 10), UnflaggedAction::NeedTrim);
        assert!(promote_piece(&large, 0, QUIET_FRAMES, true));
        assert!(large.is_gpu_only(0));
    }

    #[test]
    fn lossless_demotion_merge_preserves_cpu_and_gpu_writes() {
        let old_data: Vec<u8> = (0..128u8).collect();

        // Case 1: no CPU writes while GpuOnly (title == old_data) -> take all GPU bytes.
        let mut gpu = old_data.clone();
        gpu[10] = 0xAA;
        gpu[50] = 0xBB;
        let res = merge_demoted_piece(&gpu, &old_data, &old_data);
        assert_eq!(res.merged, gpu);
        assert!(res.title_changed);
        assert!(res.cpu_dirty_runs.is_empty());

        // Case 1b: neither GPU nor CPU changed anything.
        let res_noop = merge_demoted_piece(&old_data, &old_data, &old_data);
        assert_eq!(res_noop.merged, old_data);
        assert!(!res_noop.title_changed);
        assert!(res_noop.cpu_dirty_runs.is_empty());

        // Case 2: partial CPU write while GpuOnly -> preserve BOTH CPU writes and GPU writes!
        let mut title = old_data.clone();
        title[4] = 0x11;
        title[5] = 0x22;
        title[50] = 0xCC; // overlapping byte where both wrote -> CPU write wins!
        title[100] = 0xDD;
        let res_both = merge_demoted_piece(&gpu, &old_data, &title);
        assert!(res_both.title_changed, "GPU byte at index 10 must be written back to title");
        assert_eq!(res_both.merged[4], 0x11);
        assert_eq!(res_both.merged[5], 0x22);
        assert_eq!(res_both.merged[10], 0xAA, "GPU write at untouched offset preserved");
        assert_eq!(res_both.merged[50], 0xCC, "CPU write preserved");
        assert_eq!(res_both.merged[100], 0xDD);
        assert_eq!(res_both.cpu_dirty_runs, vec![(4, 6), (50, 51), (100, 101)]);
    }

    #[test]
    fn virtual_unlock_and_query_working_set_ex_on_real_64k_block() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        // SAFETY: private 64 KiB allocation owned solely by this test.
        let ptr = unsafe { VirtualAlloc(std::ptr::null(), PIECE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(ptr, 0);
        assert_eq!(ptr % PIECE, 0);

        // Touch every 4 KiB page with distinct pattern so all 16 pages enter the working set.
        for page in 0..PAGES_PER_PIECE {
            // SAFETY: within our committed 64 KiB allocation.
            unsafe { ((ptr + page * KERNEL_PAGE) as *mut u8).write_volatile((page as u8).wrapping_add(0x41)) };
        }
        let (touched_before, total) = query_working_set_block(ptr).expect("K32QueryWorkingSetEx succeeded");
        assert_eq!(total, PAGES_PER_PIECE as u32);
        assert!(touched_before > 0, "touched pages must report in working set");

        // Soft-trim via VirtualUnlock (zero faults, zero protection change).
        assert!(trim_block(ptr));
        let (touched_after_trim, _) = query_working_set_block(ptr).expect("K32QueryWorkingSetEx after trim");
        assert_eq!(touched_after_trim, 0, "trimmed untouched 64 KiB block must report 0 pages in working set");

        // Reading back through the pointer succeeds with zero crash, preserves
        // exact RAM contents, and brings the touched page back into working set.
        // SAFETY: within our committed 64 KiB allocation.
        let first = unsafe { (ptr as *const u8).read_volatile() };
        assert_eq!(first, 0x41);
        let (touched_after_read, _) = query_working_set_block(ptr).expect("K32QueryWorkingSetEx after read");
        assert!(touched_after_read >= 1, "reading trimmed page must bring it back into working set");

        // SAFETY: release our test allocation.
        unsafe { VirtualFree(ptr as *mut _, 0, MEM_RELEASE) };
    }
}
