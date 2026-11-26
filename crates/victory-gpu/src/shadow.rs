//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    /// Title memory's pages of 4 KiB (from the one `start` lies in on), a bit
    /// each, that what shaders wrote was put into since a refresh last looked
    /// at them ([`Shadow::take_landed`]); empty until then.
    landings: Mutex<Vec<u64>>,
    /// Whether any is.
    landing: std::sync::atomic::AtomicBool,
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// With [`gpu_upload_copies`]: `buffer` is in the card's memory, mapped
    /// for the CPU (a GPU upload heap), not a shared buffer in the PC's;
    /// its bytes.
    bar: Option<u64>,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// Synchronizes access to the CPU side (`staging` / `self.data`):
    /// `merge_under_writes` takes `.write()` (exclusive), while parallel
    /// disjoint `card_merge_relay` workers take `.read()` (shared).
    side: std::sync::RwLock<()>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
    /// The copy itself, for [`CHANGED`]: set once it is shared.
    me: std::sync::OnceLock<std::sync::Weak<Shadow>>,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (1) and `VICTORY_CARD_BUFFERS_MB` (1024).
/// The fast path measures with both: every buffer of 1 MiB or more is kept
/// on the card, so what its shaders write comes back as changed pieces
/// instead of a whole copy. Run 69's gameplay has ten title buffers of
/// 16 MiB or more, 494 MB, and 1,650 MB of the card's budget free.
/// Default `VICTORY_CARD_BUFFERS_MB` budget scaled to the host GPU's local
/// VRAM budget: 1024 MiB on <= 12 GiB cards, 2048 MiB on 12–20 GiB cards
/// (e.g. RTX 5070 Ti 16 GB), and 3072 MiB on >= 20 GiB cards.
fn default_card_buffers_mb(vram_budget: Option<u64>) -> u64 {
    const GIB: u64 = 1 << 30;
    match vram_budget {
        Some(budget) if budget >= 20 * GIB => 3072,
        Some(budget) if budget > 12 * GIB => 2048,
        _ => 1024,
    }
}

fn parse_card_limits(
    vram_budget: Option<u64>,
    min_mb: Option<&str>,
    min_kb: Option<&str>,
    max_mb: Option<&str>,
) -> (u64, u64) {
    let parse = |val: Option<&str>, default: u64| {
        val.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
    };
    let mut min_bytes = parse(min_mb, 1).max(1).saturating_mul(1 << 20);
    let max_bytes = parse(max_mb, default_card_buffers_mb(vram_budget)).saturating_mul(1 << 20);
    // `VICTORY_CARD_BUFFERS_MIN_KB` (after fxceless c36971b2) says the
    // least size in KiB, so that it can go under 1 MiB: 64 is every title
    // buffer, clamped to a 64 KiB (`PIECE`) floor so GPU 64 KiB card-compare
    // pieces align cleanly. Copies under 1 MiB are nearly all there are and
    // few bytes; in the PC's memory every load, store and atomic a shader
    // does on them crosses the bus.
    if let Some(kib) = min_kb.and_then(|v| v.trim().parse::<u64>().ok())
        && kib > 0
    {
        min_bytes = kib.saturating_mul(1 << 10).max(PIECE as u64);
    }
    (min_bytes, max_bytes)
}

fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let vram_budget = host().and_then(|h| h.video_memory()).map(|m| m[0]);
        let (min_bytes, max_bytes) = parse_card_limits(
            vram_budget,
            std::env::var("VICTORY_CARD_BUFFERS_MIN_MB").ok().as_deref(),
            std::env::var("VICTORY_CARD_BUFFERS_MIN_KB").ok().as_deref(),
            std::env::var("VICTORY_CARD_BUFFERS_MB").ok().as_deref(),
        );
        tracing::info!(
            vram_budget_mb = vram_budget.map(|b| b >> 20),
            card_buffers_min_kb = min_bytes >> 10,
            card_buffers_max_mb = max_bytes >> 20,
            "card buffer limits"
        );
        (min_bytes, max_bytes)
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Kept card buffer room for a configured max `limit`, `used` bytes already on
/// the card, and `refused` bytes turned down: capped at 1 GiB (`SEEN`) so
/// unused buffer reservation never starves texture streaming.
fn card_room_kept_for(limit: u64, used: u64, refused: u64) -> u64 {
    // Not more than such copies came to in any session, whatever they may
    // come to: on a card of 24 GB a quarter of its budget was kept for them,
    // 5 GB that never came (941 MB in 97 copies after fifteen minutes), and
    // no texture was made any more with 5 GB of the card free (session
    // play-1125-0127: 1,453 passes refused, the ground and people black).
    const SEEN: u64 = 1 << 30;
    limit.min(SEEN).saturating_sub(used.saturating_add(refused))
}

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_room_kept_for(
        card_limits().1,
        CARD_BYTES[0].load(Ordering::Relaxed),
        CARD_REFUSED[1].load(Ordering::Relaxed),
    )
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// The copies in the card's memory whose CPU side changed since
/// [`card_dirty`] last took their ranges: each is put here when its list of
/// changed offsets goes from empty to not empty (`Shadow::card_changed` /
/// `Shadow::card_changed_runs`).
static CHANGED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

fn parse_changed_list_mode(raw: Option<&str>) -> u8 {
    match raw.map(str::trim) {
        Some("1") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

fn changed_list_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = parse_changed_list_mode(std::env::var("VICTORY_CARD_CHANGED_LIST").ok().as_deref());
        if mode == 1 {
            tracing::info!(
                "changed card copies are taken from a dirty list instead of walking every card copy (VICTORY_CARD_CHANGED_LIST=1)"
            );
        }
        mode
    })
}

/// Whether [`card_dirty`] takes the changed copies from [`CHANGED`]
/// (`VICTORY_CARD_CHANGED_LIST=1`; `alt`: twenty seconds so, twenty the walk
/// over every copy, in turn, each change said in the log). Off by default.
fn changed_list() -> bool {
    match changed_list_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the changed copies in the card's memory are taken from a list from now (VICTORY_CARD_CHANGED_LIST=alt): {}",
                    if on { "ON" } else { "off, every copy is walked as usual" }
                );
            }
            on
        }
    }
}

/// Whether [`CHANGED`] is kept at all (`VICTORY_CARD_CHANGED_LIST=1|alt`).
fn changed_kept() -> bool {
    changed_list_mode() != 0
}

/// Whether per-copy CPU->GPU upload and GPU->CPU copy-back profiling is
/// enabled (`VICTORY_COPY_PROFILE=1`; off by default, single `OnceLock<bool>`
/// branch when unset).
pub(crate) fn copy_profile_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_COPY_PROFILE").is_some_and(|v| v == "1");
        if on {
            tracing::info!("per-copy upload/copy-back top-10 profiler enabled (VICTORY_COPY_PROFILE=1)");
        }
        on
    })
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct CopyProfileEntry {
    upload_bytes: u64,
    upload_count: u64,
    back_bytes: u64,
    back_count: u64,
}

static COPY_PROFILE: Mutex<Option<std::collections::HashMap<(u64, u64), CopyProfileEntry>>> = Mutex::new(None);

fn record_copy_profile_upload(start: u64, len: u64, bytes: u64) {
    if !copy_profile_on() || bytes == 0 {
        return;
    }
    let mut guard = COPY_PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let entry = guard.get_or_insert_with(Default::default).entry((start, len)).or_default();
    entry.upload_bytes = entry.upload_bytes.saturating_add(bytes);
    entry.upload_count = entry.upload_count.saturating_add(1);
}

fn record_copy_profile_back(start: u64, len: u64, bytes: u64) {
    if !copy_profile_on() || bytes == 0 {
        return;
    }
    let mut guard = COPY_PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let entry = guard.get_or_insert_with(Default::default).entry((start, len)).or_default();
    entry.back_bytes = entry.back_bytes.saturating_add(bytes);
    entry.back_count = entry.back_count.saturating_add(1);
}

fn format_copy_profile(entries: &std::collections::HashMap<(u64, u64), CopyProfileEntry>) -> Option<String> {
    if entries.is_empty() {
        return None;
    }
    let mut uploads: Vec<((u64, u64), u64, u64)> = entries
        .iter()
        .filter(|(_, e)| e.upload_bytes > 0)
        .map(|(&k, e)| (k, e.upload_bytes, e.upload_count))
        .collect();
    uploads.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut backs: Vec<((u64, u64), u64, u64)> = entries
        .iter()
        .filter(|(_, e)| e.back_bytes > 0)
        .map(|(&k, e)| (k, e.back_bytes, e.back_count))
        .collect();
    backs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    if uploads.is_empty() && backs.is_empty() {
        return None;
    }
    let fmt = |list: Vec<((u64, u64), u64, u64)>| -> String {
        list.into_iter()
            .take(10)
            .map(|((addr, size), bytes, count)| {
                format!("{addr:#x}+{size:#x}:{:.2}MB/{count}", bytes as f64 / (1 << 20) as f64)
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    Some(format!(
        "copy-profile: top-upload=[{}] top-back=[{}]",
        fmt(uploads),
        fmt(backs)
    ))
}

/// Logs the top 10 CPU->GPU upload and GPU->CPU copy-back offenders accumulated
/// over the last ~5-second reporting interval (`VICTORY_COPY_PROFILE=1`), and
/// resets the interval accumulators.
pub(crate) fn copy_profile_line() {
    if !copy_profile_on() {
        return;
    }
    let taken = {
        let mut guard = COPY_PROFILE.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_mut().map(std::mem::take)
    };
    if let Some(entries) = taken
        && let Some(line) = format_copy_profile(&entries)
    {
        tracing::info!("{line}");
    }
}

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = {
        let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
        if changed_kept() {
            // (The walk over every copy drops the ones that are gone.)
            carded.retain(|w| w.strong_count() > 0);
        }
        carded.iter().filter(|w| w.strong_count() > 0).count() as u64
    };
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// Since the last frame line: GPU-write declarations skipped because their
/// range lies wholly inside an Xbox `PAGE_GRAPHICS_READONLY` /
/// `PAGE_GRAPHICS_EXECUTE_READ` pool (`[count, bytes]`).
static READONLY_SKIPPED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// [`READONLY_SKIPPED`] (`[count, bytes]`), taken and reset per reporting
/// period (matching [`card_stats`]).
pub(crate) fn readonly_stats() -> [u64; 2] {
    [
        READONLY_SKIPPED[0].swap(0, Ordering::Relaxed),
        READONLY_SKIPPED[1].swap(0, Ordering::Relaxed),
    ]
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

fn parse_gpu_upload_copies(level: Option<&str>, mb: Option<&str>) -> (u8, u64) {
    let level = level.and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(2);
    let most = mb.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(2048).saturating_mul(1 << 20);
    (level, most)
}

/// Whether ReBAR GPU upload heap copies are enabled for a copy of this kind
/// (`window` outside buffers at level >= 1; title buffers/views at level >= 2)
/// with a non-zero byte budget.
fn rebar_enabled_for(level: u8, most: u64, window: bool) -> bool {
    level >= if window { 1 } else { 2 } && most > 0
}

/// Whether host copies with no card side are made in the card's memory,
/// mapped for the CPU (a GPU upload heap: resizable BAR), instead of a
/// shared buffer in the PC's memory: `VICTORY_GPU_UPLOAD_COPIES` `1`
/// windows outside buffers, `2` also title buffers and views' copies of
/// their own bytes; 0 (unset) none. At most `VICTORY_GPU_UPLOAD_COPIES_MB`
/// (2048 unless set) of them, and only while the card is under eight
/// tenths of its budget, as windows' card sides are. Read when a copy is
/// made, so it takes a fresh start; where the card has no such heap, nothing
/// changes.
///
/// Such a copy is one buffer, as the shared buffer was: the CPU writes
/// title memory into it at each refresh, through a write-combined mapping
/// (64 MiB in 3 ms, measured on this PC), and shaders read it in the
/// card's memory, not through the bus. No card side, no sends, no hashes:
/// nothing of `Card` is involved.
///
/// What it costs: the copies' bytes come out of the card's budget (the
/// texture queue's room); and reading such a copy on the CPU crosses the
/// bus uncached, some 40 MB/s, so a copy whose ranges shaders write (its
/// snapshot and compare read it) is slow: the memory line counts the bytes
/// so declared, and the first is said.
pub(crate) fn gpu_upload_copies() -> (u8, u64) {
    static ON: std::sync::OnceLock<(u8, u64)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_gpu_upload_copies(
            std::env::var("VICTORY_GPU_UPLOAD_COPIES").ok().as_deref(),
            std::env::var("VICTORY_GPU_UPLOAD_COPIES_MB").ok().as_deref(),
        )
    })
}

/// [`gpu_upload_copies`]' copies alive and their bytes; and since the start,
/// bytes of ranges shaders were declared to write in them.
static BAR_COPIES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// For the memory line: [`BAR_COPIES`].
pub(crate) fn bar_copies() -> [u64; 3] {
    BAR_COPIES.each_ref().map(|b| b.load(Ordering::Relaxed))
}

/// Attempts to allocate a host copy of `size` bytes in a GPU upload heap
/// (ReBAR / GDDR) when [`gpu_upload_copies`] enables it for this kind of copy
/// (`window` vs title buffer/view), VRAM stays under 80% of budget, and the
/// `VICTORY_GPU_UPLOAD_COPIES_MB` budget has room.
fn try_gpu_upload_buffer(size: u64, window: bool) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let (level, most) = gpu_upload_copies();
    if !rebar_enabled_for(level, most, window) {
        return None;
    }
    let host = host()?;
    if !host.video_memory().is_none_or(|[budget, used, ..]| used.saturating_add(size) <= budget / 10 * 8) {
        return None;
    }
    if BAR_COPIES[1].fetch_add(size, Ordering::Relaxed).saturating_add(size) <= most
        && let Some((buffer, data)) = host.card_shared_buffer(size)
    {
        BAR_COPIES[0].fetch_add(1, Ordering::Relaxed);
        return Some((buffer, data, size));
    }
    BAR_COPIES[1].fetch_sub(size, Ordering::Relaxed);
    None
}

/// The buffer of a host copy of `size` bytes that has no card side (a
/// `window` or not): in a GPU upload heap with [`gpu_upload_copies`], else
/// a shared buffer in the PC's memory; and with the first, its bytes.
fn pc_copy_buffer(size: u64, window: bool) -> Option<(ID3D12Resource, *mut u8, Option<u64>)> {
    if let Some((buffer, data, bar)) = try_gpu_upload_buffer(size, window) {
        return Some((buffer, data, Some(bar)));
    }
    let (buffer, data) = host()?.shared_buffer(size)?;
    Some((buffer, data, None))
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`): textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    if changed_kept() {
        // Taken either way: the walk takes every changed copy's ranges, so
        // what the list holds then names copies with nothing left to take.
        let changed = std::mem::take(&mut *CHANGED.lock().unwrap_or_else(|e| e.into_inner()));
        if changed_list() {
            return take_changed_from(changed, sent_as);
        }
    }
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow.card.as_ref().map(|c| take_dirty(c, fetched, sent_as)).unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// Drains dirty ranges for the card copies named in `changed` ([`card_dirty`]).
fn take_changed_from(changed: Vec<std::sync::Weak<Shadow>>, sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    for weak in changed {
        let Some(shadow) = weak.upgrade() else {
            continue;
        };
        let ranges = shadow.card.as_ref().map(|c| take_dirty(c, fetched, sent_as)).unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
    }
    out
}

/// Takes a copy's changed offsets for the send numbered `sent_as`
/// ([`card_dirty`]); with [`relay`] they are noted as sent, and the notes of
/// sends that every fetch still to be merged came after (`fetched`) go.
fn take_dirty(c: &Card, fetched: u64, sent_as: u64) -> Vec<(usize, usize)> {
    let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
    let ranges = std::mem::take(&mut *dirty);
    if relay() {
        let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
        sent.retain(|s| s.2 > fetched);
        sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
    }
    ranges
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// The offsets [`Shadow::card_hashed`] gives, for a hashed byte length of
/// `size` bytes of title memory from `start` on.
#[cfg_attr(not(test), allow(dead_code))]
fn card_hashed_of(start: usize, size: usize, piece: usize) -> (usize, usize) {
    let (lo, hi) = piece_at(start, size, piece);
    (lo & !15, hi.next_multiple_of(16))
}

/// The bytes of a copy covered by the card's hashes: with `own` enabled, the
/// title buffer's own length rounded up to 16 bytes (capped at `card_size`);
/// otherwise the whole `card_size`.
fn card_hash_size_for(len: u64, card_size: u64, own: bool) -> u64 {
    if own {
        len.next_multiple_of(16).min(card_size)
    } else {
        card_size
    }
}

#[cfg(test)]
mod card_tests {
    use super::{
        CardSpan, PIECE, card_hash_size_for, card_hashed_of, card_piece_of, card_room_kept_for, card_span_of, joined,
        joined_piece_runs, parse_card_limits,
    };

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// The fetch inside flights (`CARD_FETCH` in host.rs) copies each piece
    /// from `inflight_piece_from` on: what the merge takes of the piece lies
    /// in that slot.
    #[test]
    fn a_piece_fetched_in_flight_fits_its_slot() {
        let slot = crate::host::INFLIGHT_SLOT as usize;
        for start in [0usize, 4096, 7, PIECE - 16, PIECE - 3, 3 * PIECE + 0x1f09, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    let at = crate::host::inflight_piece_from(&span, i);
                    assert_eq!(at % 16, 0);
                    assert!(lo >= at && hi - at <= slot, "{start:#x} {from} {to} piece {i}: {lo}..{hi} from {at}");
                }
            }
        }
    }

    /// The slots are a power of two, one per 64 KiB asked, within the
    /// shader's 32-bit offsets.
    #[test]
    fn inflight_slots_are_a_power_of_two() {
        use crate::host::{INFLIGHT_SLOT, inflight_slots};
        assert_eq!(inflight_slots(64 << 20), 1024);
        assert_eq!(inflight_slots(100 << 20), 1024);
        assert_eq!(inflight_slots(1), 1);
        assert_eq!(inflight_slots(1 << 20), 16);
        let most = inflight_slots(u64::MAX);
        assert!(most.is_power_of_two() && u64::from(most) * INFLIGHT_SLOT <= 1 << 32);
    }

    /// Tickets drawn and landed in order, the mark raised as the landing
    /// raises it (every ticket a landed flight drew), through the counter's
    /// wrap with the slots the default gives: no two tickets in the air
    /// that got a slot share one, slots are still given after the wrap, and
    /// the mark follows the counter.
    #[test]
    fn inflight_tickets_never_share_a_slot_across_the_wrap() {
        use crate::host::{inflight_gets_slot, inflight_raise, inflight_slots};
        let slots = inflight_slots(64 << 20);
        let mut landed = u32::MAX - 3000;
        let mut next = landed;
        // Flights of 1 to 700 flagged pieces, up to three in the air:
        // (tickets drawn, those that got a slot).
        let mut air: std::collections::VecDeque<(Vec<u32>, Vec<u32>)> = Default::default();
        let mut after_wrap = 0;
        for flight in 0..40u32 {
            let drawn: Vec<u32> = (0..1 + flight * 37 % 700).map(|i| next.wrapping_add(i)).collect();
            next = next.wrapping_add(drawn.len() as u32);
            let slotted: Vec<u32> = drawn.iter().copied().filter(|&k| inflight_gets_slot(k, landed, slots)).collect();
            after_wrap += slotted.iter().filter(|&&k| k < 1 << 20).count();
            air.push_back((drawn, slotted));
            let mut used: Vec<u32> = air.iter().flat_map(|a| a.1.iter()).map(|k| k % slots).collect();
            let n = used.len();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), n, "flight {flight}: two tickets in the air share a slot");
            if air.len() == 3
                && let Some((drawn, _)) = air.pop_front()
            {
                for k in drawn {
                    landed = inflight_raise(landed, k);
                }
            }
        }
        while let Some((drawn, _)) = air.pop_front() {
            for k in drawn {
                landed = inflight_raise(landed, k);
            }
        }
        assert!(next < 1 << 20, "the counter wrapped");
        assert!(after_wrap > 0, "slots after the wrap");
        assert_eq!(landed, next);
        // Raised in wrapping order, never back.
        assert_eq!(inflight_raise(u32::MAX - 1, u32::MAX), 0);
        assert_eq!(inflight_raise(5, 2), 5);
        assert_eq!(inflight_raise(u32::MAX - 1, 3), 4);
        assert!(!inflight_gets_slot(3, 4, 8));
    }

    /// Flights 5, 6, 7 in the air with the fetch inside flights: 5 lands
    /// the old way with the queue's count at 7. The title's byte that went
    /// up in send 7 must still be told apart when 6 lands from its slot
    /// (its merge leaves out sends after 6), so send 7's note must outlive
    /// the mark (`card_dirty` keeps notes above it, `card_merge_relay`
    /// leaves out those above its "fetched after").
    #[test]
    fn a_fetch_the_old_way_keeps_the_notes_flights_in_the_air_need() {
        use crate::host::fallback_fetched_mark;
        let notes = [5u64, 6, 7];
        let mark = fallback_fetched_mark(true, 7, Some(5));
        let kept: Vec<u64> = notes.iter().copied().filter(|&s| s > mark).collect();
        assert!(kept.iter().any(|&s| s > 6), "send 7's note was dropped before flight 6 merged");
        assert_eq!(mark, 5);
        // Without the switch every later merge fetches after 7: as before.
        assert_eq!(fallback_fetched_mark(false, 7, Some(5)), 7);
        assert_eq!(fallback_fetched_mark(true, 7, None), 0);
    }

    /// What is fetched of a piece to hash it is what the shader hashes.
    #[test]
    fn a_piece_is_hashed_here_as_on_the_card() {
        for start in [0usize, 0x10_0000, 0xd_5edb_c780, 0x12_3456_7008, 0xfff8] {
            for size in [288usize, 304, 4096, 1usize << 16, 5 << 16, 0x20_0000] {
                let span = card_span_of(start, 0, size);
                for i in 0..span.pieces {
                    // As `CARD_COMPARE` has it.
                    let (piece, skew, size) = (span.first + i, span.skew, size as u32);
                    let lo = (piece * 65536).saturating_sub(skew);
                    let hi = ((piece + 1) * 65536 - skew).min(size);
                    let shader = (lo & !15, (hi + 15) & !15);
                    let here = card_hashed_of(start, size as usize, piece as usize);
                    assert_eq!((here.0 as u32, here.1 as u32), shader, "start {start:#x} size {size:#x} piece {piece}");
                }
            }
        }
    }

    #[test]
    fn card_hash_size_rounds_up_to_16_and_caps_at_card_size() {
        assert_eq!(card_hash_size_for(288, 64 << 10, true), 288);
        assert_eq!(card_hash_size_for(290, 64 << 10, true), 304);
        assert_eq!(card_hash_size_for(1, 64 << 10, true), 16);
        assert_eq!(card_hash_size_for((64 << 10) - 3, 64 << 10, true), 64 << 10);
        assert_eq!(card_hash_size_for(288, 64 << 10, false), 64 << 10);
        assert_eq!(card_hash_size_for(288, 0, true), 0);
    }

    #[test]
    fn card_limits_supports_min_kb_and_defaults() {
        const GIB: u64 = 1 << 30;
        assert_eq!(parse_card_limits(Some(8 * GIB), None, None, None), (1 << 20, 1024 << 20));
        assert_eq!(parse_card_limits(Some(16 * GIB), None, None, None), (1 << 20, 2048 << 20));
        assert_eq!(parse_card_limits(Some(24 * GIB), None, None, None), (1 << 20, 3072 << 20));
        assert_eq!(parse_card_limits(Some(16 * GIB), None, Some("64"), None), (64 << 10, 2048 << 20));
        assert_eq!(parse_card_limits(Some(16 * GIB), Some("4"), Some("128"), None), (128 << 10, 2048 << 20));
        assert_eq!(parse_card_limits(Some(16 * GIB), None, Some("256"), None), (256 << 10, 2048 << 20));
        // Positive `VICTORY_CARD_BUFFERS_MIN_KB` below 64 KiB clamps to the 64 KiB piece floor.
        for small in ["1", "16", "32", "63", " 48 "] {
            assert_eq!(
                parse_card_limits(Some(16 * GIB), Some("4"), Some(small), None),
                (64 << 10, 2048 << 20),
                "min_kb={small}"
            );
        }
        // `VICTORY_CARD_BUFFERS_MIN_MB` preserved when `MIN_KB` is unset, zero, or invalid.
        assert_eq!(parse_card_limits(Some(16 * GIB), Some("4"), None, None), (4 << 20, 2048 << 20));
        assert_eq!(parse_card_limits(Some(16 * GIB), Some("4"), Some("0"), None), (4 << 20, 2048 << 20));
        assert_eq!(parse_card_limits(Some(16 * GIB), None, Some("bad"), Some("512")), (1 << 20, 512 << 20));
    }

    #[test]
    fn card_room_kept_caps_reservation_at_one_gib() {
        const GIB: u64 = 1 << 30;
        assert_eq!(card_room_kept_for(2 * GIB, 0, 0), GIB);
        assert_eq!(card_room_kept_for(3 * GIB, 400 << 20, 100 << 20), GIB - (500 << 20));
        assert_eq!(card_room_kept_for(3 * GIB, 900 << 20, 200 << 20), 0);
        assert_eq!(card_room_kept_for(512 << 20, 128 << 20, 0), 384 << 20);
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock (on by
/// default, as `VICTORY_RELAY=2` measured; `VICTORY_RELAY=0` turns it off;
/// not with kept snapshots, [`keeps`]). `crate::host::relay` says what it
/// does; here it changes what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(BTreeMap<(usize, usize), (u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it. Index exact ranges; overlapping ranges remain
    /// independent, as before.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        use std::collections::btree_map::Entry;
        match self.0.entry(range) {
            Entry::Occupied(mut entry) => {
                let (count, last) = entry.get_mut();
                if *last != number {
                    *count += 1;
                    *last = number;
                }
            }
            Entry::Vacant(entry) => {
                entry.insert((1, number));
            }
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds, in input order.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        use std::collections::btree_map::Entry;
        let mut free = Vec::new();
        for range in ranges {
            match self.0.entry(*range) {
                Entry::Occupied(mut entry) => {
                    let count = &mut entry.get_mut().0;
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        entry.remove();
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                Entry::Vacant(_) => free.push(*range),
            }
        }
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let len = now.len().min(held.len()).min(title.len());
    let (now, held, title) = (&now[..len], &mut held[..len], &mut title[..len]);
    let mut differs = 0u8;
    let (now_c, now_t) = now.as_chunks::<32>();
    let (held_c, held_t) = held.as_chunks_mut::<32>();
    let (title_c, title_t) = title.as_chunks_mut::<32>();
    for ((nc, hc), tc) in now_c.iter().zip(held_c.iter_mut()).zip(title_c.iter_mut()) {
        if nc == hc {
            continue;
        }
        for i in 0..32 {
            let wrote = if nc[i] != hc[i] { 0xffu8 } else { 0 };
            let merged = (tc[i] & !wrote) | (nc[i] & wrote);
            differs |= merged ^ tc[i];
            tc[i] = merged;
            hc[i] = (hc[i] & !wrote) | (nc[i] & wrote);
        }
    }
    for ((n, h), t) in now_t.iter().zip(held_t.iter_mut()).zip(title_t.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs of 4-byte words of title memory (cut on its addresses;
/// `read` and `merged` are of the memory at `address`) that hold a byte in
/// which `merged` differs from `read`, as (offset, bytes) in order to
/// `put`. A shader stores words, so these are what the card wrote; every
/// other byte of `merged` is the title's as it was read.
fn word_runs(address: usize, read: &[u8], merged: &[u8], mut put: impl FnMut(usize, usize)) {
    let n = read.len().min(merged.len());
    let mut i = 0;
    while i < n {
        // Equal bytes are passed over, 32 at a time while they last.
        match first_difference(&read[i..n], &merged[i..n]) {
            Some(d) => i += d,
            None => break,
        }
        // From the word that holds this byte (or the stretch's first
        // byte) to the end of the last word in a row that holds one.
        let first = i - ((address + i) & 3).min(i);
        let mut end = first;
        while end < n {
            let word = (((address + end) | 3) + 1 - address).min(n);
            if read[end..word] == merged[end..word] {
                break;
            }
            end = word;
        }
        put(first, end - first);
        i = end;
    }
}

/// Whether the merges that write a stretch of title memory back
/// ([`Shadow::relay_merge`], [`Shadow::card_merge`],
/// [`Shadow::card_merge_relay`]) look, right before that write, for title
/// bytes it is about to put back to older values (`VICTORY_MERGE_RACE=1`;
/// off by default). Logging only: [`merge_race`].
fn merge_race_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_RACE").is_some_and(|v| v == "1"))
}

/// [`merge_race`]'s sums. Of frame `frame`, and in `total` since the
/// start: stretches written back, their bytes, title bytes put back to
/// older values, bytes the title and the card both changed, stretches with
/// either, the runs a write of the card's words alone makes
/// ([`word_runs`]) and their bytes. `said`: the frame's first stretches
/// with either. `quiet`: frames since the last line.
struct MergeRace {
    frame: u64,
    now: [u64; 7],
    said: Vec<String>,
    total: [u64; 7],
    quiet: u64,
}

static MERGE_RACE: Mutex<MergeRace> =
    Mutex::new(MergeRace { frame: 0, now: [0; 7], said: Vec::new(), total: [0; 7], quiet: 0 });

/// With [`merge_race_on`], right before a merge writes `merged` over title
/// memory at `address` (a stretch of the copy of the title buffer `copy`:
/// start and bytes): title memory is read once more and compared with
/// `read`, what the merge read of it before. A byte that changed in
/// between and that the write-back stores with its value as first read
/// (the card did not change it) is a title byte put back to an older
/// value; one the write-back stores with the card's byte is one both
/// changed. Only bytes the write-back stores are looked at. Summed per
/// frame; a line for every frame that has any, and one every 100 frames
/// whatever it holds, so that "none" is said, not assumed.
///
/// A lower bound: a title write that lands after this read (while the
/// write-back itself runs) is not seen, nor one that wrote a byte and then
/// its old value again. Costs one more read and two compares of every
/// stretch written back, on the thread that merges.
fn merge_race(read: Option<&[u8]>, copy: (u64, u64), address: usize, merged: &[u8], which: &'static str) {
    if !merge_race_on() {
        return;
    }
    let Some(read) = read else {
        return;
    };
    let n = read.len().min(merged.len());
    let (read, merged) = (&read[..n], &merged[..n]);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    word_runs(address, read, merged, |off, len| runs.push((off, len)));
    let whole = [(0, n)];
    let looked: &[(usize, usize)] = &whole;
    let mut again = vec![0u8; n];
    let (mut put_back, mut both, mut first) = (0u64, 0u64, None);
    if crate::host::read_into(address as *const u8, &mut again) {
        for &(off, len) in looked {
            for i in off..off + len {
                if again[i] == read[i] || again[i] == merged[i] {
                    continue;
                }
                if merged[i] == read[i] {
                    put_back += 1;
                    first.get_or_insert(i);
                } else {
                    both += 1;
                }
            }
        }
    }
    let frame = crate::frame::current();
    let mut race = MERGE_RACE.lock().unwrap_or_else(|e| e.into_inner());
    if race.frame != frame {
        race.quiet += 1;
        if race.now[4] > 0 || race.quiet >= 100 {
            tracing::info!(
                frame = race.frame,
                stretches = race.now[0],
                stretch_bytes = race.now[1],
                put_back = race.now[2],
                both_changed = race.now[3],
                stretches_hit = race.now[4],
                word_runs = race.now[5],
                word_run_bytes = race.now[6],
                stretches_since_start = race.total[0],
                stretch_bytes_since_start = race.total[1],
                put_back_since_start = race.total[2],
                both_changed_since_start = race.total[3],
                stretches_hit_since_start = race.total[4],
                word_runs_since_start = race.total[5],
                word_run_bytes_since_start = race.total[6],
                first = ?race.said,
                "merge race: title bytes that changed between a merge's read of title memory and its write-back of what shaders wrote, in this frame (VICTORY_MERGE_RACE=1)"
            );
            race.quiet = 0;
        }
        race.frame = frame;
        race.now = [0; 7];
        race.said.clear();
    }
    let hit = u64::from(put_back + both > 0);
    let run_bytes: u64 = runs.iter().map(|r| r.1 as u64).sum();
    let add = [1, n as u64, put_back, both, hit, runs.len() as u64, run_bytes];
    for (i, v) in add.iter().enumerate() {
        race.now[i] += v;
        race.total[i] += v;
    }
    if hit > 0 && race.said.len() < 8 {
        race.said.push(format!(
            "{which}: copy {:#x}+{:#x} stretch {address:#x}+{n:#x}: {put_back} put back (first at {:#x}), {both} both changed",
            copy.0,
            copy.1,
            address + first.unwrap_or(0)
        ));
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// Compare indexed bookkeeping with the previous vector algorithm,
    /// including duplicate declarations, overlapping and unknown ranges,
    /// and completions that arrive in a different order.
    #[test]
    fn indexed_flights_match_vector_reference() {
        let mut seed = 0x6a09_e667u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..32 {
            let mut indexed = Flights::default();
            let mut reference: Vec<((usize, usize), u32, u64)> = Vec::new();
            for _ in 0..2048 {
                let start = (next() % 128) as usize * 64;
                let range = (start, start + (next() % 8 + 1) as usize * 64);
                let number = u64::from(next() % 16);
                if next() >> 31 == 0 {
                    indexed.declare(range, number);
                    match reference.iter_mut().find(|f| f.0 == range) {
                        Some(f) if f.2 != number => {
                            f.1 += 1;
                            f.2 = number;
                        }
                        Some(_) => {}
                        None => reference.push((range, 1, number)),
                    }
                } else {
                    let ranges = [range, (start + 32, range.1 + 32)];
                    let mut free = Vec::new();
                    for r in ranges {
                        match reference.iter().position(|f| f.0 == r) {
                            Some(at) => {
                                reference[at].1 = reference[at].1.saturating_sub(1);
                                if reference[at].1 == 0 {
                                    reference.swap_remove(at);
                                    free.push(r);
                                }
                            }
                            None => free.push(r),
                        }
                    }
                    assert_eq!(indexed.done(&ranges), free);
                }
                assert_eq!(indexed.0.len(), reference.len());
                for &(r, count, last) in &reference {
                    assert_eq!(indexed.0.get(&r), Some(&(count, last)));
                }
            }
        }
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (on by default; `VICTORY_WIDE_BACK=0` turns it off).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

struct Waiter {
    left: std::sync::atomic::AtomicUsize,
    done: Mutex<bool>,
    ready: std::sync::Condvar,
}

impl Waiter {
    fn finish(&self) {
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
            *done = true;
            self.ready.notify_all();
        }
    }
}

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<(Job, Arc<Waiter>)>>,
    more: std::sync::Condvar,
}

thread_local! {
    /// True while this thread is already executing inside [`wide`] (either as
    /// a pool worker or the submitting/joining thread), so nested parallel
    /// helpers run inline instead of re-entering the shared worker queue.
    static IN_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Returns whether the current thread is already inside [`wide`].
pub(crate) fn in_wide() -> bool {
    IN_WIDE.get()
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism().map_or(4, |n| n.get() / 3).clamp(2, 6);
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                let workers = WORKERS.get().expect("made above");
                loop {
                    let (job, waiter) = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(item) = jobs.pop_front() {
                                break item;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let prev = IN_WIDE.replace(true);
                    job();
                    IN_WIDE.set(prev);
                    waiter.finish();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 || IN_WIDE.get() {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let waiter = Arc::new(Waiter {
        left: std::sync::atomic::AtomicUsize::new(jobs.len()),
        done: Mutex::new(false),
        ready: std::sync::Condvar::new(),
    });
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            queue.push_back((job, waiter.clone()));
        }
    }
    workers.more.notify_all();
    let prev = IN_WIDE.replace(true);
    loop {
        let item = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match item {
            Some((job, w)) => {
                job();
                w.finish();
            }
            None => break,
        }
    }
    IN_WIDE.set(prev);
    if waiter.left.load(Ordering::Acquire) > 0 {
        let mut done = waiter.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = waiter.ready.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn. Where both changed a byte, the title's stays.
    // Process 32 bytes at a time, skipping unchanged chunks and extracting
    // modified runs inline via a 32-bit bitmask without allocating `mask`.
    let _ = mask;
    let mut open: Option<usize> = None;
    let mut any_diff = false;
    let (title_c, title_t) = title.as_chunks_mut::<32>();
    let (host_c, host_t) = host.as_chunks_mut::<32>();
    let (old_c, old_t) = old.as_chunks::<32>();
    for (ci, ((tc, hc), oc)) in title_c.iter_mut().zip(host_c.iter_mut()).zip(old_c.iter()).enumerate() {
        let base = ci * 32;
        if hc == oc {
            if hc != tc {
                hc.copy_from_slice(tc);
            }
            if let Some(start) = open.take() {
                runs.push((start, base - start));
            }
            continue;
        }
        let mut bits = 0u32;
        for i in 0..32 {
            let gpu = (hc[i] != oc[i]) & (tc[i] == oc[i]);
            let merged = if gpu { hc[i] } else { tc[i] };
            if merged != tc[i] {
                bits |= 1u32 << i;
            }
            tc[i] = merged;
            hc[i] = merged;
        }
        if bits == 0 {
            if let Some(start) = open.take() {
                runs.push((start, base - start));
            }
        } else {
            any_diff = true;
            if bits == u32::MAX {
                open.get_or_insert(base);
            } else {
                let mut b = bits;
                let mut pos = 0usize;
                while b != 0 {
                    let tz = b.trailing_zeros() as usize;
                    if tz > 0 {
                        if let Some(start) = open.take() {
                            runs.push((start, base + pos - start));
                        }
                        pos += tz;
                        b >>= tz;
                    }
                    let ones = (!b).trailing_zeros() as usize;
                    open.get_or_insert(base + pos);
                    pos += ones;
                    if ones == 32 {
                        b = 0;
                    } else {
                        b >>= ones;
                    }
                }
                if pos < 32
                    && let Some(start) = open.take()
                {
                    runs.push((start, base + pos - start));
                }
            }
        }
    }
    let tail_base = title_c.len() * 32;
    for (i, ((t, h), o)) in title_t.iter_mut().zip(host_t.iter_mut()).zip(old_t.iter()).enumerate() {
        let gpu = (*h != *o) & (*t == *o);
        let merged = if gpu { *h } else { *t };
        let changed = merged != *t;
        *t = merged;
        *h = merged;
        if changed {
            any_diff = true;
            open.get_or_insert(tail_base + i);
        } else if let Some(start) = open.take() {
            runs.push((start, tail_base + i - start));
        }
    }
    if let Some(start) = open {
        runs.push((start, n - start));
    }
    if !any_diff {
        return None;
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Why [`Shadow::merge_under_writes`] gave up (the flights are then
/// settled whole, 64 to 94 ms each in run 282), since they were last
/// taken: the copy's memory is not watched; the write tracking had no
/// answer for the range; more changed runs than `VICTORY_MERGE_RUNS`
/// leaves; a copy in PC memory without its snapshot.
static MERGE_GAVE_UP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// [`MERGE_GAVE_UP`], taken.
#[allow(dead_code)]
pub(crate) fn merge_gave_up() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| MERGE_GAVE_UP[i].swap(0, Ordering::Relaxed))
}

/// With the frame line: why [`Shadow::merge_under_writes`] gave up since the
/// last line (causing flights to settle whole).
#[allow(dead_code)]
pub(crate) fn merge_giveup_line(frames: f64) {
    let gave_up = merge_gave_up();
    let total: u64 = gave_up.iter().sum();
    if total == 0 {
        return;
    }
    tracing::info!(
        "in these frames the merge under flights gave up ({total} times, {:.1}/frame; flights settled whole): \
         {} because the copy's memory is not watched, {} because the write tracking had no answer, \
         {} for more changed runs than VICTORY_MERGE_RUNS leaves, {} for a copy in PC memory without its snapshot",
        total as f64 / frames,
        gave_up[0],
        gave_up[1],
        gave_up[2],
        gave_up[3]
    );
}

/// Since the last frame-rate line, where [`Shadow::merge_under_writes`]'s
/// time went, in ns: 0 asking which pages were written, 1 finding the
/// stale ones, 2 waiting for the locks, 3 the pages compared (the wall
/// time), 4 noting the runs; summed over threads: 5 reading title memory,
/// 6 comparing and putting in. Then 7 (unused), 8 stale pages, 9 calls.
pub(crate) static UNDER_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

fn under_ns(i: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    UNDER_NS[i].fetch_add((now - since).as_nanos() as u64, Ordering::Relaxed);
    now
}

/// [`UNDER_NS`], taken.
pub(crate) fn under_stats() -> [u64; 10] {
    std::array::from_fn(|i| UNDER_NS[i].swap(0, Ordering::Relaxed))
}

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// Adds the time since it was made to [`UNDER_NS`]`[6]` when dropped.
struct Compared(std::time::Instant);

impl Drop for Compared {
    fn drop(&mut self) {
        under_ns(6, self.0);
    }
}

/// `runs` (in order) with those a gap of `gap` bytes or fewer apart made
/// one run, the gap's bytes included.
fn joined_over(runs: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if gap == 0 || runs.len() < 2 {
        return runs;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at - (last.0 + last.1) <= gap => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

/// The gap, in bytes, over which the title bytes put into a copy in the
/// card's memory under flights' shader writes are joined into one run
/// (`VICTORY_MERGE_GAP`, 0 to 64; 0 by default: only the bytes the title
/// changed go up, as before).
///
/// Run 219: the title rewrites some hundred pages of its big buffer a
/// frame while flights that may write that buffer are in the air, and
/// byte by byte the new content differs from the old in 38,000 runs a
/// frame (equal bytes in between, mostly zeros), far over the 512 runs a
/// submission may send, so the flights were settled twice a frame, 50 ms
/// of the submitting thread. Joined over small gaps the same pages are a
/// few runs.
///
/// What it gives up: a gap byte is equal in title memory and on the CPU
/// side of the copy, and goes up with its run. If a flight in the air
/// wrote exactly that byte on the card (a shader store within `gap` bytes
/// between two bytes the title changed in the same frame), the store is
/// overwritten with the old value, and no compare finds it. Exact (0) is
/// the default for that reason.
fn merge_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_MERGE_GAP").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(0, |n| n.min(64)))
}

/// Whether a refresh of a copy in the card's memory stores and sends on
/// only the bytes that differ from the copy's CPU side
/// (`VICTORY_COPY_CHANGED=1`; `alt`: twenty seconds with, twenty without,
/// in turn, each change said in the log). Off by default.
///
/// Why: the write watch and the guards say which pages the title wrote
/// to, 4 KiB or 64 KiB at a time, not which bytes changed. A refresh
/// copied every such page into the copy's CPU side and noted all of it
/// for the card: about 78 MB a frame crossed to the card and their 64 KiB
/// pieces were hashed again there (the layer's "sync" lists, 13 to 20 ms
/// of the card a frame), for pages of which the title had changed a few
/// bytes, or had written the same bytes again. The CPU side holds what
/// the card's side holds outside the ranges shaders are writing (which a
/// refresh leaves alone), so bytes that are equal there need not go.
fn copy_changed() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_CHANGED").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "copies in the card's memory are refreshed by what differs from now (VICTORY_COPY_CHANGED=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Since the start, with [`copy_changed`]: bytes of title memory read,
/// bytes that went on, their stretches, and the refreshes.
static COPY_CHANGED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The kernel's page size for landing-echo comparison.
const KERNEL_PAGE: usize = 4096;

/// Whether a refresh checks if a written 4 KiB page only holds what a shader
/// landing wrote back to title memory, skipping redundant re-upload to the GPU
/// (`VICTORY_LAND_ECHO=0`: always copy and upload as before).
fn land_echo() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_ECHO").is_none_or(|v| v != "0"))
}

/// Since the start: bytes skipped as landing echo alone, and bytes where the
/// title also wrote beside the landing.
static ECHO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The chunk a refresh by what differs compares at a time
/// ([`changed_chunks`]): whole chunks go on, so that the compare is the
/// machine's compare of two blocks and a refresh makes a few stretches,
/// not forty (run 254: found byte by byte, the stretches cost the
/// submitting thread more than the card gained). A chunk holds at most
/// this many equal bytes beside a changed one: four times the gap the
/// merges under flights join by, which is why this form is not used for
/// ranges a flight's shaders are writing (they never come here: a refresh
/// leaves them alone, and `Shadow::copy_stale` is given whole pages).
const COPY_CHANGED_CHUNK: usize = 256;

/// Default maximum gap (bytes) between adjacent changed chunks in
/// [`Shadow::copy_changed`] that gets bridged into a single `CopyBufferRegion`
/// range (`VICTORY_COPY_CHANGED_GAP`, default 4 KiB). Live Wave 15 telemetry
/// showed unbridged 256 B chunks shattering 15 MB/frame into 39,864 separate
/// `CopyBufferRegion` commands (~80,000 D3D12 copy calls/frame across both
/// queues, inflating `card_before` from 9.8 ms to 55-69 ms/frame). Bridging
/// gaps <= 4 KiB inside a single non-pending refresh span preserves the byte
/// savings while collapsing D3D12 command count by ~50x.
const DEFAULT_COPY_CHANGED_GAP: usize = 4096;

fn parse_copy_changed_gap(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_COPY_CHANGED_GAP)
}

fn copy_changed_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| parse_copy_changed_gap(std::env::var("VICTORY_COPY_CHANGED_GAP").ok().as_deref()))
}

/// Bridges sorted, non-overlapping `(offset, len)` stretches whose separating
/// gap is at most `max_gap` bytes into single contiguous stretches.
fn coalesce_gap(stretches: &mut Vec<(usize, usize)>, max_gap: usize) {
    if stretches.len() < 2 || max_gap == 0 {
        return;
    }
    let mut write = 0usize;
    for read in 1..stretches.len() {
        let (next_at, next_len) = stretches[read];
        let prev = &mut stretches[write];
        let prev_end = prev.0 + prev.1;
        if next_at.saturating_sub(prev_end) <= max_gap {
            prev.1 = (next_at + next_len) - prev.0;
        } else {
            write += 1;
            stretches[write] = (next_at, next_len);
        }
    }
    stretches.truncate(write + 1);
}

/// The stretches of `now` that differ from `was` (of one length), in
/// whole chunks of `chunk` bytes (the last one shorter), as (offset,
/// bytes) in order, chunks side by side made one stretch. Uses
/// [`first_difference`] (32-byte SIMD chunks) to skip unchanged spans
/// without invoking per-chunk comparisons over equal memory.
fn changed_chunks(now: &[u8], was: &[u8], chunk: usize) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    if chunk == 0 || n == 0 {
        return Vec::new();
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < n {
        let Some(diff) = first_difference(&now[at..n], &was[at..n]) else {
            break;
        };
        let hit = at + (diff / chunk) * chunk;
        let len = chunk.min(n - hit);
        match out.last_mut() {
            Some(last) if last.0 + last.1 == hit => last.1 += len,
            _ => out.push((hit, len)),
        }
        at = hit + len;
    }
    out
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is. Uses AVX2 (64/32 bytes per step) when available.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and both slices have length >= `n`.
        return unsafe { changed_runs_avx2(&now[..n], &was[..n]) };
    }
    changed_runs_scalar(&now[..n], &was[..n])
}

/// Scalar/SWAR fallback for [`changed_runs`].
fn changed_runs_scalar(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        let Some(d) = first_difference(&now[i..n], &was[i..n]) else {
            break;
        };
        i += d;
        let first = i;
        while i + 8 <= n {
            let a = u64::from_ne_bytes(now[i..i + 8].try_into().unwrap());
            let b = u64::from_ne_bytes(was[i..i + 8].try_into().unwrap());
            let diff = a ^ b;
            // SWAR zero-byte test: stops if any byte of `diff` is zero (equal byte).
            if diff.wrapping_sub(0x0101_0101_0101_0101) & !diff & 0x8080_8080_8080_8080 != 0 {
                break;
            }
            i += 8;
        }
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// # Safety
/// Caller must verify AVX2 support and `now.len() == was.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn changed_runs_avx2(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    let n = now.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, m)) if *a + *m == from => *m += len,
        _ => runs.push((from, len)),
    };
    let mut i = 0;
    while i + 64 <= n {
        // SAFETY: `i + 64 <= n` for both slices.
        let (m0, m1) = unsafe {
            let a0 = _mm256_loadu_si256(now.as_ptr().add(i).cast());
            let b0 = _mm256_loadu_si256(was.as_ptr().add(i).cast());
            let a1 = _mm256_loadu_si256(now.as_ptr().add(i + 32).cast());
            let b1 = _mm256_loadu_si256(was.as_ptr().add(i + 32).cast());
            let eq0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a0, b0)) as u32;
            let eq1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a1, b1)) as u32;
            (!eq0, !eq1)
        };
        let mut differ = u64::from(m0) | (u64::from(m1) << 32);
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + first as usize, count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 64;
    }
    if i + 32 <= n {
        // SAFETY: `i + 32 <= n` for both slices.
        let mut differ = unsafe {
            let a0 = _mm256_loadu_si256(now.as_ptr().add(i).cast());
            let b0 = _mm256_loadu_si256(was.as_ptr().add(i).cast());
            !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a0, b0)) as u32)
        };
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + first as usize, count as usize);
            if first + count >= 32 {
                break;
            }
            differ &= u32::MAX << (first + count);
        }
        i += 32;
    }
    while i < n {
        if now[i] != was[i] {
            let first = i;
            i += 1;
            while i < n && now[i] != was[i] {
                i += 1;
            }
            push(first, i - first);
        } else {
            i += 1;
        }
    }
    runs
}

/// Whether [`Shadow::merge_under_writes`] puts the title's bytes into the
/// CPU's side of a copy in the card's memory as it compares, a 32-byte block
/// at a time whenever any 4-byte word differs (`VICTORY_UNDER_FUSED=0`: byte
/// runs are found first and copied one by one afterwards, as before).
///
/// The same bytes end up in `held` (equal words in a 32-byte block are
/// unchanged), and comparing by 4-byte words collapses consecutive updated
/// dwords/floats with matching upper bytes into one contiguous run instead of
/// exploding `VICTORY_MERGE_RUNS`.
fn under_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UNDER_FUSED").is_none_or(|v| v != "0"))
}

#[inline]
fn push_merged_run(runs: &mut Vec<(usize, usize)>, from: usize, len: usize, gap: usize) {
    match runs.last_mut() {
        Some((a, m)) if from <= *a + *m + gap => *m = from + len - *a,
        _ => runs.push((from, len)),
    }
}

/// Finds differing 4-byte word runs (coalesced across gaps <= `gap`) and, when
/// `STORE` is true, puts `title`'s bytes into `held` in one pass, stopping
/// cleanly BEFORE modifying `held` if the next block would push `runs.len()`
/// past `limit`.
///
/// Returns `(runs, exceeded)`:
/// - If `exceeded == false`: `runs.len() <= limit` (and `held == title` when `STORE`).
/// - If `exceeded == true`: `runs.len() <= limit`; when `STORE`, every byte
///   inside `runs` holds `title`'s bytes in `held` and no byte outside `runs`
///   was modified; when `!STORE`, `held` is untouched.
///
/// Caller ensures `title.len() == held.len() && title.len() % 4 == 0`.
fn word_runs_put_limited<const STORE: bool>(
    title: &[u8],
    held: &mut [u8],
    gap: usize,
    limit: usize,
) -> (Vec<(usize, usize)>, bool) {
    #[cfg(target_arch = "x86_64")]
    if title.len().is_multiple_of(4) && title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected and both slices have equal 4-byte-multiple lengths.
        return unsafe { word_runs_put_avx2_limited::<STORE>(title, held, gap, limit) };
    }
    word_runs_put_scalar_limited::<STORE>(title, held, gap, limit)
}

fn word_runs_put_scalar_limited<const STORE: bool>(
    title: &[u8],
    held: &mut [u8],
    gap: usize,
    limit: usize,
) -> (Vec<(usize, usize)>, bool) {
    let n = title.len().min(held.len()) & !3;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if title[i..i + 4] != held[i..i + 4] {
            let merges = runs.last().is_some_and(|&(a, m)| i <= a + m + gap);
            if !merges && runs.len() >= limit {
                return (runs, true);
            }
            if STORE {
                held[i..i + 4].copy_from_slice(&title[i..i + 4]);
            }
            push_merged_run(&mut runs, i, 4, gap);
        }
        i += 4;
    }
    (runs, false)
}

/// # Safety
/// AVX2, and two slices of equal 4-byte-multiple length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_put_avx2_limited<const STORE: bool>(
    title: &[u8],
    held: &mut [u8],
    gap: usize,
    limit: usize,
) -> (Vec<(usize, usize)>, bool) {
    use std::arch::x86_64::*;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let n = title.len();
    let mut i = 0;
    // Fast 256-byte (64-dword) loop while at most 32 new runs cannot exceed `limit`.
    while i + 256 <= n && runs.len() + 32 <= limit {
        let mut differ = 0u64;
        for k in 0..8 {
            let at = i + 32 * k;
            // SAFETY: inside both slices (`i + 256 <= n`).
            unsafe {
                let now = _mm256_loadu_si256(title.as_ptr().add(at).cast());
                let old = _mm256_loadu_si256(held.as_ptr().add(at).cast());
                let block = !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(now, old))) as u8;
                if STORE && block != 0 {
                    _mm256_storeu_si256(held.as_mut_ptr().add(at).cast(), now);
                }
                differ |= u64::from(block) << (8 * k);
            }
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push_merged_run(&mut runs, i + 4 * first as usize, 4 * count as usize, gap);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    // 32-byte (8-dword) loop with pre-store run-budget check so `held` is never
    // written unless every differing word in the block fits inside `limit`.
    while i + 32 <= n {
        // SAFETY: inside both slices (`i + 32 <= n`).
        let (now, block) = unsafe {
            let now = _mm256_loadu_si256(title.as_ptr().add(i).cast());
            let old = _mm256_loadu_si256(held.as_ptr().add(i).cast());
            let block = !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(now, old))) as u8;
            (now, block)
        };
        if block != 0 {
            let mut added = 0usize;
            let mut last_end = runs.last().map(|&(a, m)| a + m);
            let mut m = block;
            while m != 0 {
                let first = m.trailing_zeros();
                let count = (m >> first).trailing_ones();
                let from = i + 4 * first as usize;
                let end = from + 4 * count as usize;
                if !last_end.is_some_and(|le| from <= le + gap) {
                    added += 1;
                }
                last_end = Some(end);
                if first + count >= 8 {
                    break;
                }
                m &= u8::MAX << (first + count);
            }
            if runs.len() + added > limit {
                return (runs, true);
            }
            if STORE {
                // SAFETY: inside `held` (`i + 32 <= n`) and within `limit`.
                unsafe { _mm256_storeu_si256(held.as_mut_ptr().add(i).cast(), now) };
            }
            let mut m = block;
            while m != 0 {
                let first = m.trailing_zeros();
                let count = (m >> first).trailing_ones();
                push_merged_run(&mut runs, i + 4 * first as usize, 4 * count as usize, gap);
                if first + count >= 8 {
                    break;
                }
                m &= u8::MAX << (first + count);
            }
        }
        i += 32;
    }
    while i + 4 <= n {
        if title[i..i + 4] != held[i..i + 4] {
            let merges = runs.last().is_some_and(|&(a, m)| i <= a + m + gap);
            if !merges && runs.len() >= limit {
                return (runs, true);
            }
            if STORE {
                held[i..i + 4].copy_from_slice(&title[i..i + 4]);
            }
            push_merged_run(&mut runs, i, 4, gap);
        }
        i += 4;
    }
    (runs, false)
}

#[cfg(test)]
mod merge_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{
        PAGE, PIECE, back_piece, changed_chunks, changed_runs, coalesce_gap, in_wide,
        parse_copy_changed_gap, wide_run, word_runs_put_limited, word_runs_put_scalar_limited,
    };

    #[test]
    fn changed_chunks_hold_every_changed_byte() {
        let was: Vec<u8> = (0..2000u32).map(|i| (i * 31 + 7) as u8).collect();
        for changes in [vec![], vec![0usize], vec![1999], vec![255, 256], vec![5, 300, 301, 1024, 1990]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x5A;
            }
            for chunk in [1usize, 64, 256, 4096] {
                let stretches = changed_chunks(&now, &was, chunk);
                let mut put = was.clone();
                for &(at, n) in &stretches {
                    assert!(at % chunk == 0 && n > 0 && at + n <= now.len());
                    put[at..at + n].copy_from_slice(&now[at..at + n]);
                }
                assert_eq!(put, now, "{changes:?} chunk {chunk}");
                // No stretch without a changed byte in each of its chunks, and none touching the next.
                assert!(stretches.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0));
                let chunks: usize = stretches.iter().map(|s| s.1.div_ceil(chunk)).sum();
                let mut changed: Vec<usize> = changes.iter().map(|c| c / chunk).collect();
                changed.dedup();
                assert_eq!(chunks, changed.len(), "{changes:?} chunk {chunk}");
            }
        }
        assert_eq!(changed_chunks(&[1, 2, 3], &[1, 2, 4], 256), [(0, 3)], "a last chunk shorter than the others");
    }

    #[test]
    fn coalesce_gap_bridges_close_stretches_and_preserves_far_ones() {
        let mut s = vec![(0, 256), (512, 256), (4864, 256), (10_000, 256), (14_352, 256)];
        // Gaps: 512-256=256 (<=4096 -> merge), 4864-768=4096 (<=4096 -> merge),
        // 10000-5120=4880 (>4096 -> separate), 14352-10256=4096 (<=4096 -> merge).
        coalesce_gap(&mut s, 4096);
        assert_eq!(s, [(0, 5120), (10_000, 4608)]);

        let mut untouched = vec![(0, 256), (512, 256)];
        coalesce_gap(&mut untouched, 0);
        assert_eq!(untouched, [(0, 256), (512, 256)]);

        assert_eq!(parse_copy_changed_gap(None), 4096);
        assert_eq!(parse_copy_changed_gap(Some("")), 4096);
        assert_eq!(parse_copy_changed_gap(Some("0")), 0);
        assert_eq!(parse_copy_changed_gap(Some(" 8192 ")), 8192);
    }

    #[test]
    fn nested_wide_run_executes_inline_without_deadlock() {
        let sum = Arc::new(AtomicU64::new(0));
        let outer: Vec<Box<dyn FnOnce() + Send>> = (0..4)
            .map(|i| {
                let sum = sum.clone();
                Box::new(move || {
                    assert!(in_wide());
                    let inner: Vec<Box<dyn FnOnce() + Send>> = (0..4)
                        .map(|j| {
                            let sum = sum.clone();
                            Box::new(move || {
                                sum.fetch_add((i * 10 + j + 1) as u64, Ordering::Relaxed);
                            }) as Box<dyn FnOnce() + Send>
                        })
                        .collect();
                    wide_run(inner);
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        wide_run(outer);
        let expected: u64 = (0..4).flat_map(|i| (0..4).map(move |j| (i * 10 + j + 1) as u64)).sum();
        assert_eq!(sum.load(Ordering::Relaxed), expected);
    }

    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }

    #[test]
    fn changed_chunks_wide_matches_naive_on_unaligned_and_edge_bytes() {
        let mut seed = 0x51a7_9911u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        let naive = |now: &[u8], was: &[u8], chunk: usize| -> Vec<(usize, usize)> {
            let n = now.len().min(was.len());
            if chunk == 0 || n == 0 {
                return Vec::new();
            }
            let mut out: Vec<(usize, usize)> = Vec::new();
            let mut at = 0;
            while at < n {
                let len = chunk.min(n - at);
                if now[at..at + len] != was[at..at + len] {
                    match out.last_mut() {
                        Some(last) if last.0 + last.1 == at => last.1 += len,
                        _ => out.push((at, len)),
                    }
                }
                at += len;
            }
            out
        };
        assert!(changed_chunks(&[1, 2], &[3, 4], 0).is_empty());
        assert!(changed_chunks(&[], &[], 256).is_empty());
        for len in [0usize, 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 513, 1025] {
            let was: Vec<u8> = (0..len).map(|i| (i * 19 + 3) as u8).collect();
            for edge in [0usize, 31, 32, 33, 63, 64, 255, 256, len.saturating_sub(1)] {
                if edge >= len {
                    continue;
                }
                let mut now = was.clone();
                now[edge] ^= 0xff;
                for chunk in [1usize, 32, 64, 256] {
                    assert_eq!(changed_chunks(&now, &was, chunk), naive(&now, &was, chunk), "len {len} edge {edge} chunk {chunk}");
                }
            }
        }
        // Multiple disjoint & adjacent runs across random unaligned lengths.
        for round in 0..100 {
            let len = next() as usize % 1500;
            let was: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let mut now = was.clone();
            for _ in 0..next() % 16 {
                if len == 0 {
                    break;
                }
                let at = next() as usize % len;
                now[at] ^= 0x3c;
            }
            for chunk in [1usize, 17, 32, 64, 256] {
                assert_eq!(changed_chunks(&now, &was, chunk), naive(&now, &was, chunk), "round {round} len {len} chunk {chunk}");
            }
        }
    }

    #[test]
    fn changed_runs_avx2_matches_scalar() {
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        // Explicit boundary lengths and edge bytes around 32- and 64-byte SIMD boundaries.
        for len in [0usize, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97] {
            let was: Vec<u8> = (0..len).map(|i| (i * 11 + 5) as u8).collect();
            let mut now = was.clone();
            for edge in [0usize, 31, 32, 63, 64, len.saturating_sub(1)] {
                if edge < len {
                    now[edge] ^= 0x81;
                }
            }
            assert_eq!(super::changed_runs(&now, &was), super::changed_runs_scalar(&now, &was), "edges len {len}");
        }
        for round in 0..200 {
            let n = next() as usize % 600;
            let was: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut now = was.clone();
            for _ in 0..next() % 120 {
                if n == 0 {
                    break;
                }
                let at = next() as usize % n;
                let len = (next() as usize % 36).min(n - at);
                for b in &mut now[at..at + len] {
                    *b = next() as u8;
                }
            }
            assert_eq!(
                super::changed_runs(&now, &was),
                super::changed_runs_scalar(&now, &was),
                "round {round} len {n}"
            );
        }
    }

    #[test]
    fn word_runs_put_finds_the_word_runs_and_puts_the_title_in() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..300 {
            let n = [PAGE, 4096, 256, 260, 4, 0, 1000][round % 7];
            let held: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut title = held.clone();
            // Sparse words, dense stretches, single bytes of words.
            let changes = (next() % 600) as usize;
            for _ in 0..changes.min(n) {
                let at = (next() as usize) % n.max(1);
                let len = [1, 3, 4, 40, 300][(next() % 5) as usize].min(n - at);
                for b in &mut title[at..at + len] {
                    *b = b.wrapping_add(1 + (next() % 255) as u8);
                }
            }
            let gap = [0usize, 4, 16][round % 3];
            let mut want_held = held.clone();
            let (want, want_over) = word_runs_put_scalar_limited::<true>(&title, &mut want_held, gap, usize::MAX);
            assert!(!want_over);
            let mut merged = held.clone();
            let (runs, over) = word_runs_put_limited::<true>(&title, &mut merged, gap, usize::MAX);
            assert!(!over, "round {round}");
            assert_eq!(runs, want, "round {round}");
            assert_eq!(merged, title, "round {round}");
            // Non-storing mode finds the exact same runs without touching `held`.
            let mut untouched = held.clone();
            let (ro_runs, ro_over) = word_runs_put_limited::<false>(&title, &mut untouched, gap, usize::MAX);
            assert!(!ro_over, "round {round}");
            assert_eq!(ro_runs, want, "round {round}");
            assert_eq!(untouched, held, "round {round}");
            // Limit enforcement: when capped below `want.len()`, returns `over=true`,
            // stays within `limit`, updates every byte inside `runs`, and never
            // modifies any byte outside `runs`.
            if want.len() > 2 {
                let limit = want.len() / 2;
                let mut capped = held.clone();
                let (partial, capped_over) = word_runs_put_limited::<true>(&title, &mut capped, gap, limit);
                assert!(capped_over, "round {round}");
                assert!(partial.len() <= limit, "round {round}");
                let mut expected = held.clone();
                for &(r, m) in &partial {
                    expected[r..r + m].copy_from_slice(&title[r..r + m]);
                }
                assert_eq!(capped, expected, "round {round}: bytes outside returned runs must remain untouched");
                let mut ro_capped = held.clone();
                let (_, ro_capped_over) = word_runs_put_limited::<false>(&title, &mut ro_capped, gap, limit);
                assert!(ro_capped_over, "round {round}");
                assert_eq!(ro_capped, held, "round {round}: STORE=false must never modify held");
            }
        }
        // Consecutive changed floats with equal upper bytes collapse into ONE word run.
        let held = vec![0x3fu8; 400];
        let mut title = held.clone();
        for w in 0..100 {
            title[w * 4] ^= 0x01;
        }
        assert_eq!(changed_runs(&title, &held).len(), 100);
        let mut merged = held.clone();
        let (runs, over) = word_runs_put_limited::<true>(&title, &mut merged, 0, 512);
        assert!(!over);
        assert_eq!(runs, [(0, 400)]);
        assert_eq!(merged, title);
    }

    #[test]
    fn default_card_buffers_mb_scales_with_vram_budget() {
        use super::default_card_buffers_mb;
        const GIB: u64 = 1 << 30;
        assert_eq!(default_card_buffers_mb(None), 1024);
        assert_eq!(default_card_buffers_mb(Some(8 * GIB)), 1024);
        assert_eq!(default_card_buffers_mb(Some(12 * GIB)), 1024);
        assert_eq!(default_card_buffers_mb(Some(15 * GIB)), 2048);
        assert_eq!(default_card_buffers_mb(Some(16 * GIB)), 2048);
        assert_eq!(default_card_buffers_mb(Some(20 * GIB)), 3072);
        assert_eq!(default_card_buffers_mb(Some(24 * GIB)), 3072);
    }

    #[test]
    fn fast_merge_piece_and_back_piece_match_scalar_reference() {
        use super::{back_piece, merge_piece};
        let mut seed = 0xfeed_face_cafe_beefu64;
        let mut next = || -> u8 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 56) as u8
        };
        for len in [1usize, 31, 32, 33, 64, 97, 256, 515] {
            let old: Vec<u8> = (0..len).map(|_| next()).collect();
            let mut now = old.clone();
            let mut title = old.clone();
            for i in 0..len {
                let r = next();
                if r % 5 == 0 {
                    now[i] ^= 0x55;
                }
                if r % 7 == 0 {
                    title[i] ^= 0xaa;
                }
            }
            // Verify `merge_piece` against scalar reference.
            let mut held_fast = old.clone();
            let mut title_fast = title.clone();
            let changed_fast = merge_piece(&now, &mut held_fast, &mut title_fast);
            let mut held_ref = old.clone();
            let mut title_ref = title.clone();
            let mut changed_ref = false;
            for i in 0..len {
                if now[i] != held_ref[i] {
                    if title_ref[i] != now[i] {
                        changed_ref = true;
                    }
                    title_ref[i] = now[i];
                    held_ref[i] = now[i];
                }
            }
            assert_eq!(changed_fast, changed_ref, "len {len}");
            assert_eq!(held_fast, held_ref, "len {len}");
            assert_eq!(title_fast, title_ref, "len {len}");

            // Verify `back_piece` runs and output against scalar reference.
            let mut host_fast = now.clone();
            let (mut scratch, mut mask, mut runs_fast) = (Vec::new(), Vec::new(), Vec::new());
            let span = back_piece(&mut host_fast, &old, &mut scratch, &mut mask, &mut runs_fast, |first, into| {
                into.copy_from_slice(&title[first..first + into.len()]);
                true
            });
            let first = (0..len).find(|&i| now[i] != old[i]);
            let last = (0..len).rfind(|&i| now[i] != old[i]);
            match (first, last) {
                (Some(f), Some(l)) => {
                    let n = l + 1 - f;
                    let mut exp_title = title[f..=l].to_vec();
                    let mut exp_host = now.clone();
                    let mut exp_mask = vec![0u8; n];
                    for k in 0..n {
                        let i = f + k;
                        let gpu = now[i] != old[i] && title[i] == old[i];
                        let merged = if gpu { now[i] } else { title[i] };
                        exp_mask[k] = merged ^ title[i];
                        exp_title[k] = merged;
                        exp_host[i] = merged;
                    }
                    let mut exp_runs = Vec::new();
                    let mut j = 0;
                    while j < n {
                        while j < n && exp_mask[j] == 0 {
                            j += 1;
                        }
                        let s = j;
                        while j < n && exp_mask[j] != 0 {
                            j += 1;
                        }
                        if j > s {
                            exp_runs.push((s, j - s));
                        }
                    }
                    if exp_runs.is_empty() {
                        assert_eq!(span, None, "len {len}");
                    } else {
                        assert_eq!(span, Some((f, n)), "len {len}");
                        assert_eq!(scratch, exp_title, "len {len}");
                        assert_eq!(host_fast, exp_host, "len {len}");
                        assert_eq!(runs_fast, exp_runs, "len {len}");
                    }
                }
                _ => assert_eq!(span, None, "len {len}"),
            }
        }
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if let Some(bytes) = self.bar {
            BAR_COPIES[0].fetch_sub(1, Ordering::Relaxed);
            BAR_COPIES[1].fetch_sub(bytes, Ordering::Relaxed);
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

/// `runs` (from, to, whether the copy held it before), cut in those of pages
/// held before (`pages`: page number and asking, in order) to the parts of
/// each page that were written since (`parts`: a bit for each 4 KiB).
fn by_parts(runs: Vec<(usize, usize, bool)>, pages: &[(usize, u64)], parts: &[u16]) -> Vec<(usize, usize, bool)> {
    const PART: usize = PAGE / 16;
    if parts.iter().all(|&p| p == u16::MAX) {
        return runs;
    }
    let mut out: Vec<(usize, usize, bool)> = Vec::with_capacity(runs.len() + pages.len());
    let mut pages = pages.iter().zip(parts).peekable();
    for (from, to, known) in runs {
        if !known {
            out.push((from, to, known));
            continue;
        }
        while let Some((&(page, _), &parts)) = pages.next_if(|((page, _), _)| page * PAGE < to) {
            for k in (0..16).filter(|k| parts >> k & 1 != 0) {
                let (a, b) = ((page * PAGE + k * PART).max(from), (page * PAGE + (k + 1) * PART).min(to));
                if a >= b {
                    continue;
                }
                match out.last_mut() {
                    Some((_, end, true)) if *end == a => *end = b,
                    _ => out.push((a, b, true)),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod parts_tests {
    use super::{PAGE, by_parts};

    #[test]
    fn runs_are_cut_to_the_parts_written() {
        const P: usize = PAGE / 16;
        let at = |page: usize, part: usize| page * PAGE + part * P;
        // Pages 10 and 11 again (from the middle of 10's part 1 on), 12
        // never taken, 14 again.
        let runs = vec![
            (at(10, 1) + 100, at(12, 0), true),
            (at(12, 0), at(13, 0), false),
            (at(14, 0), at(14, 9) + 7, true),
        ];
        let pages = [(10, 5), (11, 5), (14, 6)];
        assert_eq!(by_parts(runs.clone(), &pages, &[u16::MAX; 3]), runs);
        assert_eq!(
            by_parts(runs, &pages, &[0b1000_0000_0000_0011, 0b11, 0b10_0000_0000]),
            [
                (at(10, 1) + 100, at(10, 2), true),
                (at(10, 15), at(11, 2), true),
                (at(12, 0), at(13, 0), false),
                (at(14, 9), at(14, 9) + 7, true),
            ]
        );
    }
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.iter().filter(|&&at| at != 0).count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::effective_ram_idle_ms();
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        let removed: Vec<Arc<Shadow>> = idle.into_iter().filter_map(|key| map.remove(&key)).collect();
        if !removed.is_empty() {
            tables_changed();
        }
        removed
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    #[cfg(test)]
    pub(crate) fn byte_at(&self, offset: usize) -> u8 {
        assert!(offset < self.len as usize);
        // SAFETY: `offset` is bounds-checked inside `self.data` of length `self.len`.
        unsafe { *self.data.add(offset) }
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            landings: Mutex::default(),
            landing: std::sync::atomic::AtomicBool::new(false),
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            place: None,
            bar: None,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        // Xbox `PAGE_GRAPHICS_READONLY` / `PAGE_GRAPHICS_EXECUTE_READ` pools
        // are guaranteed by hardware memory protection never to be written by
        // the GPU (CPU->GPU uploads via `refresh`/`update`/`card_before`
        // continue normally).
        if crate::title_hints::gpu_readonly_contains(address, len) {
            READONLY_SKIPPED[0].fetch_add(1, Ordering::Relaxed);
            READONLY_SKIPPED[1].fetch_add(len, Ordering::Relaxed);
            return;
        }
        list.gather_wrote(self.id);
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        if self.bar.is_some() {
            // Their snapshot and compare read the copy back through the
            // bus, uncached (`gpu_upload_copies`): counted, said once.
            BAR_COPIES[2].fetch_add(len, Ordering::Relaxed);
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::warn!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    window = self.window,
                    "shaders write a host copy in a GPU upload heap (VICTORY_GPU_UPLOAD_COPIES): its compare reads it back \
                     through the bus at some 40 MB/s"
                );
            });
        }
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame). Returns `false` if GPU-write tracking was skipped (in-place
    /// copy or Xbox GPU-read-only pool).
    pub(crate) fn writing(&self, address: usize, len: usize) -> bool {
        if self.place.is_some() {
            return false;
        }
        let (from, to) = self.inside(address, len);
        let (check_addr, check_len) = if from < to {
            (self.start + from as u64, (to - from) as u64)
        } else {
            (address as u64, len as u64)
        };
        if crate::title_hints::gpu_readonly_contains(check_addr, check_len) {
            READONLY_SKIPPED[0].fetch_add(1, Ordering::Relaxed);
            READONLY_SKIPPED[1].fetch_add(check_len, Ordering::Relaxed);
            return false;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return true;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            return true;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![(from, to)];
        let mut rest = Vec::new();
        for &(a, e) in pending.iter() {
            if parts.is_empty() {
                break;
            }
            let (a, e) = self.inside(a, e - a);
            rest.clear();
            for &(from, to) in &parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            std::mem::swap(&mut parts, &mut rest);
        }
        pending.push(range);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        if taken == 0 {
            return true;
        }
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return true;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return true;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
        true
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                let at = at + first;
                crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                crate::watch::writing(start + at, n);
                // Only the GPU's bytes (upstream #2009): the rest of the
                // stretch is the title's as read, and a title write landing
                // there since stays.
                for &(off, len) in &runs {
                    crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                    written += len;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        record_copy_profile_back(self.start, self.len, written as u64);
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Who touched the page since its last store is looked at before our
        // own read of it (`VICTORY_BACK_READS`): counted only.
        let _reads_as = crate::backreads::reading_as(crate::backreads::MERGE, self.id);
        crate::backreads::before(start + at);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = merge_race_on().then(|| title.clone());
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        if self.put_back_own(start + at, &title[..]) {
            crate::backreads::stored(self.id, (self.start, self.len), start + at, n, true);
            record_copy_profile_back(self.start, self.len, n as u64);
            return n;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "pc relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        crate::backreads::stored(self.id, (self.start, self.len), start + at, n, false);
        record_copy_profile_back(self.start, self.len, n as u64);
        n
    }

    /// With `crate::inplace::back_own` (`VICTORY_BACK_OWN_VIEW=1`): stores
    /// what a merge made of title memory `[at, at + bytes.len())` (inside
    /// one 64 KiB page, as every merged stretch is) through a view of our
    /// own, tells the journal, and notes the page current for this copy if
    /// it was current before: the copy holds the same bytes (the merge put
    /// them into both), so the next refresh need not take the page again.
    /// The guards are not touched: a title write to the page still faults
    /// and is reported at a later asking, which every copy then takes,
    /// this one too. False with nothing done if the switch is off or the
    /// memory cannot be written that way (the caller writes as before).
    fn put_back_own(&self, at: usize, bytes: &[u8]) -> bool {
        if !crate::inplace::back_own() || bytes.is_empty() {
            return false;
        }
        // Single 64 KiB guest-page fast path (always taken by `card_merge_relay`
        // and `relay_merge` whose pieces are cut at 64 KiB title boundaries).
        if at / PAGE == (at + bytes.len() - 1) / PAGE {
            return self.put_back_own_page(at, bytes);
        }
        // Cross-page stretch: split strictly at 64 KiB guest-page boundaries so
        // every section-backed sub-page writes cleanly without dirtying guards.
        let mut off = 0usize;
        let mut any_own = false;
        while off < bytes.len() {
            let cur = at + off;
            let take = (PAGE - (cur % PAGE)).min(bytes.len() - off);
            let sub = &bytes[off..off + take];
            if self.put_back_own_page(cur, sub) {
                any_own = true;
            } else if !any_own {
                return false;
            } else {
                crate::watch::writing(cur, take);
                crate::host::write_mapped(cur as *mut u8, sub.as_ptr(), take);
            }
            off += take;
        }
        any_own
    }

    fn put_back_own_page(&self, at: usize, bytes: &[u8]) -> bool {
        /// Since the start: stretches and bytes stored this way, pages
        /// noted current for their copy, stretches written the old way;
        /// and with `VICTORY_BACK_OWN_CHECK=1`: stores the title's own
        /// address did not show afterwards, pages noted current whose
        /// CPU side in the copy differs from title memory, and those of
        /// them whose guard was clean (a hole in the rule: a title write
        /// on its way leaves the guard open).
        static OWN_WRITES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
        static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let check = *CHECK.get_or_init(|| std::env::var_os("VICTORY_BACK_OWN_CHECK").is_some_and(|v| v == "1"));
        let Some(shown) = crate::inplace::write_own(at, bytes) else {
            OWN_WRITES[3].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        if check {
            // The title's own address must show what went in through ours.
            let mut back = vec![0u8; bytes.len()];
            if crate::host::read_into(at as *const u8, &mut back) && back != bytes {
                OWN_WRITES[4].fetch_add(1, Ordering::Relaxed);
            }
        }
        let page = at / PAGE;
        let parts = if bytes.is_empty() || (at + bytes.len() - 1) / PAGE != page {
            u16::MAX
        } else {
            let first = (at % PAGE) / KERNEL_PAGE;
            let last = ((at + bytes.len() - 1) % PAGE) / KERNEL_PAGE;
            (((1u32 << (last - first + 1)) - 1) << first) as u16
        };
        let others: Vec<usize> = shown.iter().map(|&a| a / PAGE).filter(|&p| p != page).collect();
        let (before, now) = crate::watch::wrote_own(page, parts, &others);
        // A placed texture there changed with no descriptor written (what
        // `crate::watch::writing` tells the texture table on the old way).
        crate::descriptor::host::remapped(at, bytes.len());
        let base = self.start as usize / PAGE;
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(taken) = page.checked_sub(base).and_then(|i| copied.get_mut(i))
                && *taken != 0
                && *taken >= before
            {
                *taken = (*taken).max(now);
                OWN_WRITES[2].fetch_add(1, Ordering::Relaxed);
                // A copy in the card's memory only: its CPU side is never
                // written by the card, so it must equal title memory now.
                if check && self.card.is_some() {
                    let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                    let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
                    if lo < hi {
                        let mut title = vec![0u8; hi - lo];
                        // SAFETY: inside the copy's mapping; read only.
                        let held = unsafe { std::slice::from_raw_parts(self.data.add(lo - start).cast_const(), hi - lo) };
                        if crate::host::read_into(lo as *const u8, &mut title) && title != held {
                            OWN_WRITES[5].fetch_add(1, Ordering::Relaxed);
                            if crate::watch::clean(page) {
                                OWN_WRITES[6].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
        OWN_WRITES[1].fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let n = OWN_WRITES[0].fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(16384) {
            tracing::info!(
                "what shaders wrote, stored through views of our own (VICTORY_BACK_OWN_VIEW=1), since the start: {n} stretches {} MB, {} pages noted current for the copy that wrote them (not copied in again), {} stretches written the old way; with VICTORY_BACK_OWN_CHECK=1: {} stores the title's address did not show, {} pages noted current whose CPU side differs from title memory, {} of those with the guard clean",
                OWN_WRITES[1].load(Ordering::Relaxed) >> 20,
                OWN_WRITES[2].load(Ordering::Relaxed),
                OWN_WRITES[3].load(Ordering::Relaxed),
                OWN_WRITES[4].load(Ordering::Relaxed),
                OWN_WRITES[5].load(Ordering::Relaxed),
                OWN_WRITES[6].load(Ordering::Relaxed)
            );
        }
        true
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => !stale_pages_in(base, &copied, a, e, &written).is_empty(),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        let _reads_as = crate::backreads::reading_as(crate::backreads::UNDER, self.id);
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            MERGE_GAVE_UP[0].fetch_add(1, Ordering::Relaxed);
            return true;
        }
        let busy = joined(busy);
        let base = start / PAGE;
        let gap = merge_gap();
        let fused_on = under_fused();
        let is_relay = relay();
        thread_local! {
            static MERGE_TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let mut updated_pages: Vec<usize> = Vec::new();
        UNDER_NS[9].fetch_add(1, Ordering::Relaxed);
        for (a, e) in busy {
            let t = std::time::Instant::now();
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards.
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                MERGE_GAVE_UP[1].fetch_add(1, Ordering::Relaxed);
                return true;
            };
            let t = under_ns(0, t);
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            UNDER_NS[8].fetch_add(stale.len() as u64, Ordering::Relaxed);
            let t = under_ns(1, t);
            if stale.is_empty() {
                continue;
            }
            let side = self.card.as_ref().map(|c| c.side.write().unwrap_or_else(|e| e.into_inner()));
            let mut before = self.card.is_none().then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
            if let Some(b) = &before
                && b.is_none()
            {
                MERGE_GAVE_UP[3].fetch_add(1, Ordering::Relaxed);
                return true;
            }
            let t = under_ns(2, t);
            updated_pages.clear();
            let mut changed: Vec<(usize, usize)> = Vec::new();
            let (mut runs_in, mut bytes_in) = (0u64, 0u64);
            const WIDE_MERGE_UNDER_MIN_PAGES: usize = 16;
            let exceeded = if self.card.is_some() && stale.len() >= WIDE_MERGE_UNDER_MIN_PAGES && !in_wide() {
                let chunks = (stale.len() / 8).clamp(2, 5);
                let step = stale.len().div_ceil(chunks);
                let budget = std::sync::atomic::AtomicUsize::new(*runs_left);
                let gave_up = std::sync::atomic::AtomicBool::new(false);
                struct ChunkOut {
                    changed: Vec<(usize, usize)>,
                    updated: Vec<usize>,
                    runs_in: u64,
                    bytes_in: u64,
                }
                let slices: Vec<&[usize]> = stale.chunks(step).collect();
                let outs: Vec<Mutex<Option<ChunkOut>>> = (0..slices.len()).map(|_| Mutex::new(None)).collect();
                let self_addr = self as *const Self as usize;
                let budget_addr = &budget as *const std::sync::atomic::AtomicUsize as usize;
                let gave_up_addr = &gave_up as *const std::sync::atomic::AtomicBool as usize;
                let outs_addr = outs.as_ptr() as usize;
                let jobs: Vec<Job> = slices
                    .into_iter()
                    .enumerate()
                    .map(|(idx, pages_slice)| {
                        let pages_ptr = pages_slice.as_ptr() as usize;
                        let pages_len = pages_slice.len();
                        Box::new(move || {
                            // SAFETY: `wide_run` joins all workers before `self`, `stale`,
                            // `budget`, `gave_up`, `outs`, and `side` leave scope. Pages in
                            // `stale` are unique and sorted, so each worker writes strictly
                            // disjoint `[at, at + n)` slices of `self.data`.
                            let this = unsafe { &*(self_addr as *const Self) };
                            let pages = unsafe { std::slice::from_raw_parts(pages_ptr as *const usize, pages_len) };
                            let budget = unsafe { &*(budget_addr as *const std::sync::atomic::AtomicUsize) };
                            let gave_up = unsafe { &*(gave_up_addr as *const std::sync::atomic::AtomicBool) };
                            let slot = unsafe { &*(outs_addr as *const Mutex<Option<ChunkOut>>).add(idx) };
                            let mut local_changed = Vec::new();
                            let mut local_updated = Vec::with_capacity(pages.len());
                            let (mut local_runs, mut local_bytes) = (0u64, 0u64);
                            MERGE_TITLE.with(|buf| {
                                let mut buf = buf.borrow_mut();
                                if buf.len() < PAGE {
                                    buf.resize(PAGE, 0);
                                }
                                for &page in pages {
                                    if gave_up.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                                    if lo >= hi {
                                        continue;
                                    }
                                    let n = hi - lo;
                                    let at = lo - start;
                                    // SAFETY: inside `self.data`, disjoint per unique `page`, under `side.write()`.
                                    let held_ptr = unsafe { this.data.add(at).cast_const() };
                                    if let Some(diff) = crate::plaincopy::first_diff(lo as *const u8, held_ptr, n)
                                        && diff >= n
                                    {
                                        MERGED[0].fetch_add(1, Ordering::Relaxed);
                                        if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                                            local_updated.push(page);
                                        }
                                        continue;
                                    }
                                    let title = &mut buf[..n];
                                    let tp = std::time::Instant::now();
                                    if !crate::host::read_into(lo as *const u8, title) {
                                        continue;
                                    }
                                    let tp = under_ns(5, tp);
                                    let _compared = Compared(tp);
                                    MERGED[0].fetch_add(1, Ordering::Relaxed);
                                    // SAFETY: disjoint `[at, at + n)` inside `self.data`, under `side.write()`.
                                    let held = unsafe { std::slice::from_raw_parts_mut(this.data.add(at), n) };
                                    let avail = budget.load(Ordering::Relaxed);
                                    let word_ok = fused_on && at.is_multiple_of(4) && lo.is_multiple_of(4) && n.is_multiple_of(4);
                                    let consume = |used: usize, over: bool| -> bool {
                                        let mut prev = 0usize;
                                        let _ = budget.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                                            prev = cur;
                                            Some(cur.saturating_sub(used))
                                        });
                                        over || used > prev
                                    };
                                    if word_ok && (is_relay || n / 8 < avail) {
                                        let (runs, over) = word_runs_put_limited::<true>(title, held, gap, avail);
                                        let hit = consume(runs.len(), over);
                                        local_runs += runs.len() as u64;
                                        local_bytes += runs.iter().map(|r| r.1 as u64).sum::<u64>();
                                        local_changed.extend(runs.into_iter().map(|(r, m)| (at + r, at + r + m)));
                                        if hit {
                                            gave_up.store(true, Ordering::Relaxed);
                                            break;
                                        }
                                    } else if word_ok {
                                        let (runs, over) = word_runs_put_limited::<false>(title, held, gap, avail);
                                        if over || consume(runs.len(), false) {
                                            gave_up.store(true, Ordering::Relaxed);
                                            break;
                                        }
                                        local_runs += runs.len() as u64;
                                        for (r, m) in runs {
                                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                                            local_changed.push((at + r, at + r + m));
                                            local_bytes += m as u64;
                                        }
                                    } else {
                                        let runs = joined_over(changed_runs(title, held), gap);
                                        if runs.len() > avail || consume(runs.len(), false) {
                                            gave_up.store(true, Ordering::Relaxed);
                                            break;
                                        }
                                        local_runs += runs.len() as u64;
                                        for (r, m) in runs {
                                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                                            local_changed.push((at + r, at + r + m));
                                            local_bytes += m as u64;
                                        }
                                    }
                                    if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                                        local_updated.push(page);
                                    }
                                }
                            });
                            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(ChunkOut {
                                changed: local_changed,
                                updated: local_updated,
                                runs_in: local_runs,
                                bytes_in: local_bytes,
                            });
                        }) as Job
                    })
                    .collect();
                wide_run(jobs);
                for slot in outs {
                    if let Some(chunk) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                        changed.extend(chunk.changed);
                        updated_pages.extend(chunk.updated);
                        runs_in += chunk.runs_in;
                        bytes_in += chunk.bytes_in;
                    }
                }
                *runs_left = budget.load(Ordering::Relaxed);
                let hit = gave_up.load(Ordering::Relaxed);
                if hit {
                    MERGE_GAVE_UP[2].fetch_add(1, Ordering::Relaxed);
                }
                hit
            } else {
                MERGE_TITLE.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    if buf.len() < PAGE {
                        buf.resize(PAGE, 0);
                    }
                    for page in stale {
                        let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                        if lo >= hi {
                            continue;
                        }
                        let n = hi - lo;
                        let at = lo - start;
                        // Zero-write AVX2 fast path for card copies: skip `read_into`
                        // and word-run diffing completely when the stale page is
                        // byte-identical to `self.data` (e.g. copy-back echo).
                        if self.card.is_some() {
                            // SAFETY: inside `self.data`, protected exclusively by `side`.
                            let held_ptr = unsafe { self.data.add(at).cast_const() };
                            if let Some(diff) = crate::plaincopy::first_diff(lo as *const u8, held_ptr, n)
                                && diff >= n
                            {
                                MERGED[0].fetch_add(1, Ordering::Relaxed);
                                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                                    updated_pages.push(page);
                                }
                                continue;
                            }
                        }
                        let title = &mut buf[..n];
                        let tp = std::time::Instant::now();
                        // A page that is not there is left as it is, and looked at
                        // again next time.
                        if !crate::host::read_into(lo as *const u8, title) {
                            continue;
                        }
                        let tp = under_ns(5, tp);
                        let _compared = Compared(tp);
                        MERGED[0].fetch_add(1, Ordering::Relaxed);
                        if self.card.is_some() {
                            // SAFETY: inside the mapping of the copy's CPU side,
                            // protected exclusively by `side` (`card.side.write()`).
                            let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                            let word_ok = fused_on && at.is_multiple_of(4) && lo.is_multiple_of(4) && n.is_multiple_of(4);
                            if word_ok && (is_relay || n / 8 < *runs_left) {
                                let (runs, over) = word_runs_put_limited::<true>(title, held, gap, *runs_left);
                                *runs_left -= runs.len();
                                runs_in += runs.len() as u64;
                                bytes_in += runs.iter().map(|r| r.1 as u64).sum::<u64>();
                                changed.extend(runs.into_iter().map(|(r, m)| (at + r, at + r + m)));
                                if over {
                                    MERGE_GAVE_UP[2].fetch_add(1, Ordering::Relaxed);
                                    return true;
                                }
                            } else if word_ok {
                                let (runs, over) = word_runs_put_limited::<false>(title, held, gap, *runs_left);
                                if over {
                                    MERGE_GAVE_UP[2].fetch_add(1, Ordering::Relaxed);
                                    return true;
                                }
                                *runs_left -= runs.len();
                                runs_in += runs.len() as u64;
                                for (r, m) in runs {
                                    held[r..r + m].copy_from_slice(&title[r..r + m]);
                                    changed.push((at + r, at + r + m));
                                    bytes_in += m as u64;
                                }
                            } else {
                                let runs = joined_over(changed_runs(title, held), gap);
                                if runs.len() > *runs_left {
                                    MERGE_GAVE_UP[2].fetch_add(1, Ordering::Relaxed);
                                    return true;
                                }
                                *runs_left -= runs.len();
                                runs_in += runs.len() as u64;
                                for (r, m) in runs {
                                    held[r..r + m].copy_from_slice(&title[r..r + m]);
                                    changed.push((at + r, at + r + m));
                                    bytes_in += m as u64;
                                }
                            }
                        } else if let Some(snapshot) = before.as_mut().and_then(|b| b.as_mut()) {
                            // SAFETY: the snapshot's allocation holds the copy's
                            // bytes at the copy's offsets (`writing`), and these
                            // are in a declared range.
                            let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                            for (r, m) in changed_runs(title, held) {
                                // SAFETY: inside the copy's mapping. The card may be
                                // writing the copy beside these bytes; they are
                                // written here and nothing around them.
                                unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                                held[r..r + m].copy_from_slice(&title[r..r + m]);
                                runs_in += 1;
                                bytes_in += m as u64;
                            }
                        }
                        // Taken, if the range holds all of the page that is in the
                        // copy; else the rest of it is the refresh's, which notes
                        // only whole pages, and the page is looked at again.
                        if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                            updated_pages.push(page);
                        }
                    }
                    false
                })
            };
            let t = under_ns(3, t);
            MERGED[1].fetch_add(runs_in, Ordering::Relaxed);
            MERGED[2].fetch_add(bytes_in, Ordering::Relaxed);
            record_copy_profile_upload(self.start, self.len, bytes_in);
            // Flush all modified runs into `card.dirty` in one call BEFORE
            // releasing `side` (even on `exceeded`), so `card_merge_relay`
            // skips every run whose bytes are already in `self.data`.
            self.card_changed_runs(changed);
            drop(before);
            drop(side);
            if !updated_pages.is_empty() {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for &page in &updated_pages {
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
                updated_pages.clear();
            }
            under_ns(4, t);
            if exceeded {
                return true;
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        let copied = if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            bytes
        } else {
            self.update_free(from, to)
        };
        record_copy_profile_upload(self.start, self.len, copied as u64);
        copied
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            // The light refresh (`VICTORY_LIGHT_REFRESH`): nothing of the
            // range was written that the copy does not hold, and the
            // journal says so with nothing asked.
            if crate::watch::light() && self.nothing_new(from, to) {
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from) as u64);
                }
                return 0;
            }
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
    }

    /// Whether [`Shadow::update_free`] of `[from, to)` would copy nothing,
    /// known without asking the kernel or the guards
    /// (`crate::watch::nothing_new`): the memory is watched, every page of
    /// the range was taken, and none was noted written after it was.
    /// `false` if the refresh would have to ask.
    fn nothing_new(&self, from: usize, to: usize) -> bool {
        if self.watched.load(Ordering::Relaxed) != 1 {
            return false;
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        crate::watch::nothing_new(from, to, self.start as usize / PAGE, &copied)
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), cut to dirty 4 KiB parts where the kernel or a landing named
    /// them (`VICTORY_PAGE_PARTS`), whole pages where never taken before.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let (runs, again_at) = {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if written.is_empty() && copied[first_page - base..=last - base].iter().all(|&c| c != 0) {
                return 0;
            }
            let mut writes = written.iter().peekable();
            let mut runs: Vec<(usize, usize, bool)> = Vec::new();
            let mut run: Option<(usize, bool)> = None;
            // Pages taken before and written since: in the order of their runs.
            let mut again_at: Vec<(usize, u64)> = Vec::new();
            for page in first_page..=last {
                let write = match writes.peek() {
                    Some(&&(p, w)) if p == page => {
                        writes.next();
                        w
                    }
                    _ => 0,
                };
                let at = &mut copied[page - base];
                let taken = *at;
                if taken == 0 || write > taken {
                    // A page only partly in the range is copied in part: it
                    // stays stale for the rest.
                    if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                        *at = now;
                    }
                    // Cut where pages taken before meet pages never taken.
                    let again = taken != 0;
                    if again {
                        again_at.push((page, taken));
                    }
                    if let Some((first, known)) = run
                        && known != again
                    {
                        runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                        run = None;
                    }
                    run.get_or_insert((page, again));
                } else if let Some((first, known)) = run.take() {
                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                }
            }
            // Cut to the range: next to it shaders may be writing the copy.
            if let Some((first, known)) = run {
                runs.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to), known));
            }
            (runs, again_at)
        };
        let runs = if again_at.is_empty() {
            runs
        } else {
            by_parts(runs, &again_at, &crate::watch::parts_written(&again_at))
        };
        let mut bytes = 0;
        for (a, b, _) in runs {
            bytes += self.copy(a, b);
        }
        bytes
    }

    /// What shaders wrote in `[from, to)` (offsets) was put into title memory.
    fn note_landing(&self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let first = self.start as usize / KERNEL_PAGE;
        let pages = (self.start as usize + from) / KERNEL_PAGE - first
            ..(self.start as usize + to).div_ceil(KERNEL_PAGE) - first;
        let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
        if bits.is_empty() {
            let all = (self.start + self.len).div_ceil(KERNEL_PAGE as u64) as usize - first;
            bits.resize(all.div_ceil(64), 0);
        }
        for page in pages {
            if let Some(word) = bits.get_mut(page / 64) {
                *word |= 1 << (page % 64);
            }
        }
        self.landing.store(true, Ordering::Release);
    }

    /// [`Shadow::copy`] of `[from, to)` (inside the buffer): where a 4 KiB
    /// page was written by a GPU landing, compare against the CPU side and
    /// skip re-uploading identical pages ([`land_echo`]).
    fn take_landed(&self, from: usize, to: usize) -> usize {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        let first = start / KERNEL_PAGE;
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        {
            let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
            let mut at = from;
            while at < to {
                let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                let page = at / KERNEL_PAGE - first;
                let noted = bits
                    .get(page / 64)
                    .is_some_and(|word| (word >> (page % 64)) & 1 != 0);
                // Only clear once a full 4 KiB page has been checked.
                if noted && next - at == KERNEL_PAGE {
                    bits[page / 64] &= !(1 << (page % 64));
                }
                match runs.last_mut() {
                    Some(last) if last.2 == noted && next - last.0 <= PAGE => last.1 = next,
                    _ => runs.push((at, next, noted)),
                }
                at = next;
            }
        }
        TITLE.with_borrow_mut(|title| {
            for (from, to, noted) in runs {
                if !noted {
                    // SAFETY: inside the mapping (`from` and `to` are within the buffer).
                    let at = unsafe { self.data.add(from - start) };
                    crate::host::copy_mapped(at, from as *const u8, to - from, false);
                    self.changed(from - start, to - start);
                    self.card_changed(from - start, to - start);
                    continue;
                }
                title.resize(to - from, 0);
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, to - from, false);
                let mut at = from;
                while at < to {
                    let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                    let slice = &title[at - from..next - from];
                    // SAFETY: inside the CPU-side mapping; no shader range holds it.
                    let held = unsafe {
                        std::slice::from_raw_parts_mut(self.data.add(at - start), next - at)
                    };
                    if slice == held {
                        ECHO[0].fetch_add((next - at) as u64, Ordering::Relaxed);
                    } else {
                        ECHO[1].fetch_add((next - at) as u64, Ordering::Relaxed);
                        held.copy_from_slice(slice);
                        self.changed(at - start, next - start);
                        self.card_changed(at - start, next - start);
                    }
                    at = next;
                }
            }
        });
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let _reads_as = crate::backreads::reading_as(crate::backreads::REFRESH, self.id);
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if self.card.is_some() && from < to {
            if copy_changed()
                && let Some(sent) = self.copy_changed(from, to)
            {
                return sent;
            }
            if self.landing.load(Ordering::Acquire) && land_echo() {
                return self.take_landed(from, to);
            }
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// Compares and updates one contiguous slice `[from, from + n)` (inside
    /// the buffer) into `self.data`, returning `Some((card_runs, sent_bytes))`
    /// where `card_runs` are buffer-relative `(start_off, end_off)` spans that
    /// changed, or `None` if title memory could not be read whole.
    fn copy_changed_slice(&self, from: usize, n: usize, gap: usize) -> Option<(Vec<(usize, usize)>, usize)> {
        thread_local! {
            static NOW: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        let base_off = from - start;
        // SAFETY: `[from, from + n)` lies inside the buffer; callers only pass
        // disjoint slices of a non-pending refresh span.
        let held_all = unsafe { std::slice::from_raw_parts(self.data.add(base_off).cast_const(), n) };
        // Zero-write AVX2 fast path: ~81% of copy-changed refreshes read back
        // identical bytes (e.g. MEM_PRIVATE copy-back echo). Skip copying into
        // scratch `NOW` completely when unchanged, or skip any identical prefix.
        let prefix = match crate::plaincopy::first_diff(from as *const u8, held_all.as_ptr(), n) {
            Some(diff) if diff >= n => return Some((Vec::new(), 0)),
            Some(diff) => (diff / COPY_CHANGED_CHUNK) * COPY_CHANGED_CHUNK,
            None => 0,
        };
        let tail_len = n - prefix;
        NOW.with(|now| {
            let mut now = now.borrow_mut();
            if now.len() < tail_len {
                now.resize(tail_len, 0);
            }
            if !crate::host::read_into((from + prefix) as *const u8, &mut now[..tail_len]) {
                return None;
            }
            let held = &held_all[prefix..n];
            let mut stretches = changed_chunks(&now[..tail_len], held, COPY_CHANGED_CHUNK);
            coalesce_gap(&mut stretches, gap);
            let mut sent = 0usize;
            let mut runs = Vec::with_capacity(stretches.len());
            for &(at, len) in &stretches {
                let off = base_off + prefix + at;
                // SAFETY: `off + len <= self.len` and disjoint across parallel slices.
                unsafe { std::ptr::copy_nonoverlapping(now.as_ptr().add(at), self.data.add(off), len) };
                runs.push((off, off + len));
                sent += len;
            }
            Some((runs, sent))
        })
    }

    /// [`Shadow::copy`] for a copy in the card's memory, with
    /// [`copy_changed`]: title memory `[from, to)` (inside the buffer) is
    /// compared against the copy's CPU side and only coalesced changed
    /// stretches are stored and noted for the card. Returns `Some(sent)` on
    /// success, or `None` with nothing done if title memory could not be read
    /// whole (the caller copies as before).
    fn copy_changed(&self, from: usize, to: usize) -> Option<usize> {
        let n = to - from;
        let gap = copy_changed_gap();
        const WIDE_MIN_BYTES: usize = 512 << 10;
        let (all_runs, sent) = if n >= WIDE_MIN_BYTES && !in_wide() {
            let chunks = (n / (256 << 10)).clamp(2, 5);
            let step = n.div_ceil(chunks).next_multiple_of(PAGE);
            let mut ranges = Vec::with_capacity(chunks);
            let mut at = from;
            while at < to {
                let next = (at + step).min(to);
                ranges.push((at, next - at));
                at = next;
            }
            if ranges.len() >= 2 {
                let failed = std::sync::atomic::AtomicBool::new(false);
                let out: Vec<Mutex<Option<(Vec<(usize, usize)>, usize)>>> =
                    (0..ranges.len()).map(|_| Mutex::new(None)).collect();
                let self_addr = self as *const Self as usize;
                let failed_addr = &failed as *const std::sync::atomic::AtomicBool as usize;
                let out_addr = out.as_ptr() as usize;
                let jobs: Vec<Job> = ranges
                    .iter()
                    .enumerate()
                    .map(|(i, &(chunk_from, chunk_len))| {
                        Box::new(move || {
                            // SAFETY: `wide_run` joins all workers before
                            // `self`, `failed`, and `out` go out of scope, and
                            // each chunk operates on a strictly disjoint slice.
                            let this = unsafe { &*(self_addr as *const Self) };
                            let failed = unsafe { &*(failed_addr as *const std::sync::atomic::AtomicBool) };
                            let slot = unsafe { &*(out_addr as *const Mutex<Option<(Vec<(usize, usize)>, usize)>>).add(i) };
                            match this.copy_changed_slice(chunk_from, chunk_len, gap) {
                                Some(res) => *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(res),
                                None => failed.store(true, Ordering::Relaxed),
                            }
                        }) as Job
                    })
                    .collect();
                wide(jobs);
                if failed.load(Ordering::Relaxed) {
                    return None;
                }
                let mut merged_runs = Vec::new();
                let mut total_sent = 0usize;
                for slot in out {
                    if let Some((runs, s)) = slot.into_inner().unwrap_or_else(|e| e.into_inner()) {
                        merged_runs.extend(runs);
                        total_sent += s;
                    }
                }
                (merged_runs, total_sent)
            } else {
                self.copy_changed_slice(from, n, gap)?
            }
        } else {
            self.copy_changed_slice(from, n, gap)?
        };
        for &(a, b) in &all_runs {
            self.changed(a, b);
        }
        let stretch_count = all_runs.len() as u64;
        self.card_changed_runs(all_runs);
        COPY_CHANGED[0].fetch_add(n as u64, Ordering::Relaxed);
        COPY_CHANGED[1].fetch_add(sent as u64, Ordering::Relaxed);
        COPY_CHANGED[2].fetch_add(stretch_count, Ordering::Relaxed);
        let calls = COPY_CHANGED[3].fetch_add(1, Ordering::Relaxed) + 1;
        if calls.is_multiple_of(65536) {
            tracing::info!(
                "copies in the card's memory, refreshed by what differs (VICTORY_COPY_CHANGED=1), since the start: {calls} refreshes read {} MB of title memory the title had written to; {} MB of it differed from the copy and went on to the card, in {} stretches",
                COPY_CHANGED[0].load(Ordering::Relaxed) >> 20,
                COPY_CHANGED[1].load(Ordering::Relaxed) >> 20,
                COPY_CHANGED[2].load(Ordering::Relaxed)
            );
        }
        Some(sent)
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Guest virtual start address of the title buffer this shadow mirrors.
    pub(crate) fn title_start(&self) -> usize {
        self.start as usize
    }

    /// Persistent CPU mapping of this shadow's CPU-side buffer (`card.staging`
    /// when carded), valid for `[0, self.len)` bytes.
    pub(crate) fn cpu_data(&self) -> *const u8 {
        self.data.cast_const()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let first = dirty.is_empty();
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
        drop(dirty);
        // From none to some: [`card_dirty`] finds the copy in the list.
        if first
            && changed_kept()
            && let Some(me) = card.me.get()
        {
            CHANGED.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
        }
    }

    /// Notes many changed offset runs `[from, to)` on the CPU side in one lock,
    /// reserving room upfront.
    fn card_changed_runs(&self, runs: impl IntoIterator<Item = (usize, usize)>) {
        let Some(card) = &self.card else {
            return;
        };
        let runs = runs.into_iter();
        let most = runs.size_hint().0;
        let mut runs = runs.filter(|r| r.0 < r.1).peekable();
        if runs.peek().is_none() {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let first = dirty.is_empty();
        dirty.reserve(most);
        for (from, to) in runs {
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
                _ => dirty.push((from, to)),
            }
        }
        drop(dirty);
        if first
            && changed_kept()
            && let Some(me) = card.me.get()
        {
            CHANGED.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
        }
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        // A shadow wholly inside an Xbox GPU-read-only pool is never written by
        // shaders, so skip allocating its "before" hash twin in VRAM while
        // keeping its main card copy for fast GPU reads and CPU->GPU uploads.
        if crate::title_hints::gpu_readonly_contains(self.start, self.len) {
            return false;
        }
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The offsets the compare on the card hashes of a piece: what lies of
    /// it in the card's hashed bytes, widened to 16 bytes.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn card_hashed(&self, piece: u32) -> (usize, usize) {
        let size = self.card_hash_size() as usize;
        card_hashed_of(self.start as usize, size, piece as usize)
    }

    /// The bytes of the copy the card's hashes cover: the title buffer's own,
    /// rounded up to the hash's 16-byte step. The card's buffer is the title
    /// buffer rounded up to 64 KiB, of which a title buffer of 288 bytes can
    /// give nothing back but those (after fxceless c36971b2).
    /// `VICTORY_CARD_HASH_LEN=0`: the whole buffer.
    pub(crate) fn card_hash_size(&self) -> u64 {
        static OWN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let own = *OWN.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_LEN").is_none_or(|v| v != "0"));
        let size = self.card.as_ref().map_or(self.len, |c| c.size);
        card_hash_size_for(self.len, size, own)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            let as_read = merge_race_on().then(|| title.clone());
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's. Skip equal 32-byte chunks.
            let mut differs = 0u8;
            let (title_c, title_t) = title.as_chunks_mut::<32>();
            let (gpu_c, gpu_t) = gpu.as_chunks::<32>();
            let (old_c, old_t) = old.as_chunks_mut::<32>();
            for ((tc, gc), oc) in title_c.iter_mut().zip(gpu_c.iter()).zip(old_c.iter_mut()) {
                if gc == oc {
                    if oc != tc {
                        oc.copy_from_slice(tc);
                    }
                    continue;
                }
                for i in 0..32 {
                    let wrote = if gc[i] != oc[i] { 0xffu8 } else { 0 };
                    let merged = (tc[i] & !wrote) | (gc[i] & wrote);
                    differs |= merged ^ tc[i];
                    tc[i] = merged;
                    oc[i] = merged;
                }
            }
            for ((t, g), o) in title_t.iter_mut().zip(gpu_t.iter()).zip(old_t.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                crate::host::shaders_write(start + at, &title[..]);
                crate::watch::writing(start + at, n);
                merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card");
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                self.note_landing(at, at + n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        record_copy_profile_back(self.start, self.len, written as u64);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = card.side.read().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; concurrent
        // `card_merge_relay` workers operate on strictly disjoint spans, while
        // `merge_under_writes` takes `card.side.write()`.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        // Who touched the page since its last store is looked at before our
        // own read of it (`VICTORY_BACK_READS`): counted only.
        let _reads_as = crate::backreads::reading_as(crate::backreads::MERGE, self.id);
        crate::backreads::before(start + at);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = (mapped && merge_race_on()).then(|| title.clone());
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        if self.put_back_own(start + at, &title[..]) {
            crate::backreads::stored(self.id, (self.start, self.len), start + at, n, true);
            record_copy_profile_back(self.start, self.len, n as u64);
            self.note_landing(at, at + n);
            return n;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        crate::backreads::stored(self.id, (self.start, self.len), start + at, n, false);
        record_copy_profile_back(self.start, self.len, n as u64);
        self.note_landing(at, at + n);
        n
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
///
/// Read by every view a draw names, on all recording threads at once, and
/// written when a buffer comes, goes or gets its copy: readers share it.
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

/// Write guard for [`BUFFERS`] that bumps [`TABLES_CHANGED`] after all
/// mutations complete and before releasing the write lock, guaranteeing no
/// thread can observe the new generation before the mutated map is visible.
struct BuffersWriteGuard(std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>>);

impl std::ops::Deref for BuffersWriteGuard {
    type Target = BTreeMap<(u64, usize), Entry>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for BuffersWriteGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for BuffersWriteGuard {
    fn drop(&mut self) {
        tables_changed();
    }
}

fn buffers() -> BuffersWriteGuard {
    let map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
    // What threads remember of it ([`FOUND`]) is of before.
    BUFFERS_CHANGED.fetch_add(1, Ordering::SeqCst);
    BuffersWriteGuard(map)
}

/// How often [`BUFFERS`] was taken to be changed.
static BUFFERS_CHANGED: AtomicU64 = AtomicU64::new(1);

/// Answers a thread remembers.
const FOUND_MOST: usize = 1024;

/// An ask (address, bytes), the count of changes it was answered at, and the
/// answer.
type Found = (u64, u64, u64, std::sync::Weak<Shadow>);

thread_local! {
    /// The last answers of [`lookup`] out of [`BUFFERS`], by a hash of the
    /// ask: the same buffers are asked about draw after draw, from five
    /// threads, each time through the map's lock and a dozen of its nodes
    /// (7 % of the recording threads' time, profile waits-1).
    static FOUND: std::cell::RefCell<Vec<Found>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether [`FOUND`] is kept (`VICTORY_QUICK_FIND=0`: not).
fn quick_find() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUICK_FIND").is_none_or(|v| v != "0"))
}

fn found_at(address: u64, len: u64) -> usize {
    let h = (address ^ len.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 54) as usize % FOUND_MOST
}

/// What this thread remembers of an ask, if nothing changed since.
fn found(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let changed = BUFFERS_CHANGED.load(Ordering::SeqCst);
    let shadow = FOUND.with(|f| {
        let f = f.borrow();
        let (a, n, at, shadow) = f.get(found_at(address, len))?;
        ((*a, *n, *at) == (address, len, changed)).then(|| shadow.upgrade()).flatten()
    })?;
    // Whoever takes the map to let a copy go that nobody holds finds it held
    // now, or had counted before: then it is not handed out.
    std::sync::atomic::fence(Ordering::SeqCst);
    (BUFFERS_CHANGED.load(Ordering::SeqCst) == changed).then_some(shadow)
}

/// Remembers an answer given while `changed` was the count.
fn remember(address: u64, len: u64, changed: u64, shadow: &Arc<Shadow>) {
    FOUND.with(|f| {
        let mut f = f.borrow_mut();
        if f.is_empty() {
            f.resize_with(FOUND_MOST, || (0, 0, 0, std::sync::Weak::new()));
        }
        f[found_at(address, len)] = (address, len, changed, Arc::downgrade(shadow));
    });
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers_read()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        let before = gone.len();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
        if gone.len() > before {
            tables_changed();
        }
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers_read().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// Why [`sole_buffer_copy`] found no sole buffer copy for a range: either
/// no registered title buffer covers it (`NoBuffer`), or another shadow
/// copy (`BUFFERS` or `LOOSE`) overlaps `[address, address + len)` (`OtherCopy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotSole {
    NoBuffer,
    OtherCopy,
}

/// For bytes the host would otherwise copy out of title memory on the CPU
/// (a constant view of a descriptor table, `crate::draw`): the copy of the
/// innermost title buffer holding `[address, address + len)`, as [`find`]
/// gives it, if no other copy holds any of those bytes (no other title
/// buffer's copy over them, and no loose window outside buffers).
pub(crate) fn sole_buffer_copy(address: u64, len: u64) -> Result<Arc<Shadow>, NotSole> {
    let end = address.checked_add(len.max(1)).ok_or(NotSole::NoBuffer)?;
    {
        let map = buffers_read();
        let key = map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end)
            .map(|(k, _)| *k)
            .ok_or(NotSole::NoBuffer)?;
        if map
            .range(..(end, 0))
            .rev()
            .take(256)
            .any(|(k, e)| *k != key && e.end > address && e.shadow.is_some())
        {
            return Err(NotSole::OtherCopy);
        }
    }
    {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        if map
            .range((address.saturating_sub(64 << 20), 0)..(end, 0))
            .any(|(k, _)| k.0 + k.1 > address)
        {
            return Err(NotSole::OtherCopy);
        }
    }
    find(address, len).ok_or(NotSole::NoBuffer)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

/// Counts the changes of the tables [`lookup`] reads (`BUFFERS` and `LOOSE`)
/// that can change what an ask finds: a buffer registered or released, a copy
/// inserted or taken out of a table, a window made or let go. A thread's
/// remembered answers ([`LOOKUP_FOUND`]) are valid for one count.
static TABLES_CHANGED: AtomicU64 = AtomicU64::new(0);

fn tables_changed() {
    TABLES_CHANGED.fetch_add(1, Ordering::Release);
}

/// Since the last frame line: asks answered from a thread's own lookup cache,
/// and asks that went to the tables (`VICTORY_LOOKUP_CACHE`).
pub(crate) static LOOKUPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// With the frame line: asks for a buffer's host copy answered from the asking
/// thread's lookup cache vs. the tables (`VICTORY_LOOKUP_CACHE`).
#[allow(dead_code)]
pub(crate) fn lookup_cache_line(frames: f64) {
    let hits = LOOKUPS[0].swap(0, Ordering::Relaxed);
    let misses = LOOKUPS[1].swap(0, Ordering::Relaxed);
    if hits == 0 && misses == 0 && lookup_cache_mode() == 0 {
        return;
    }
    tracing::info!(
        "asks for a buffer's host copy per frame: {:.0} answered from the asking thread's memory of them, {:.0} from the tables (VICTORY_LOOKUP_CACHE)",
        hits as f64 / frames,
        misses as f64 / frames
    );
}

/// Parses `VICTORY_LOOKUP_CACHE`: `0` off, `1` on, `2` alternating (`alt`).
fn parse_lookup_cache_mode(val: Option<&str>) -> u8 {
    match val.map(str::trim) {
        Some("1") => 1,
        Some("alt") => 2,
        _ => 0,
    }
}

fn lookup_cache_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_lookup_cache_mode(std::env::var("VICTORY_LOOKUP_CACHE").ok().as_deref()))
}

/// Whether a thread remembers what its asks found across both `BUFFERS` and
/// `LOOSE` (upstream PR #1237; `VICTORY_LOOKUP_CACHE=1|alt`).
///
/// Off by default (`0`); `1` enables the full lock-free generation-validated
/// thread-local cache; `alt` alternates twenty seconds on / twenty seconds off
/// with a log line on each toggle.
fn lookup_cache() -> bool {
    match lookup_cache_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "a thread remembers what its asks for a buffer's host copy found from now (VICTORY_LOOKUP_CACHE=alt): {}",
                    if on { "ON" } else { "off, every ask goes to the tables as usual" }
                );
            }
            on
        }
    }
}

/// The most asks a thread remembers in [`LOOKUP_FOUND`] before it clears and
/// starts again.
const LOOKUP_CACHE_MOST: usize = 8192;

type LookupFound = std::collections::HashMap<(u64, u64, bool), std::sync::Weak<Shadow>>;

thread_local! {
    /// What this thread's asks found across `BUFFERS` and `LOOSE`, tagged with
    /// the [`TABLES_CHANGED`] generation they came from.
    static LOOKUP_FOUND: std::cell::RefCell<(u64, LookupFound)> = std::cell::RefCell::new((u64::MAX, LookupFound::new()));
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    if lookup_cache() {
        return lookup_cached(address, len, any);
    }
    lookup_tables(address, len, any, lookup_cache_mode() == 0 && quick_find())
}

fn lookup_cached(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let changes = TABLES_CHANGED.load(Ordering::Acquire);
    let found = LOOKUP_FOUND.with(|f| {
        let f = &mut *f.borrow_mut();
        if f.0 != changes {
            f.1.clear();
            f.0 = changes;
        }
        f.1.get(&(address, len, any)).and_then(std::sync::Weak::upgrade)
    });
    if let Some(shadow) = found {
        LOOKUPS[0].fetch_add(1, Ordering::Relaxed);
        if shadow.window {
            // As `loose` notes an ask of a window it hands out (read-first to
            // avoid cache-line bouncing across recording threads).
            let now = crate::host::moment();
            if shadow.used.load(Ordering::Relaxed) != now {
                shadow.used.store(now, Ordering::Relaxed);
            }
            if shadow.asked[0].load(Ordering::Relaxed) > address {
                shadow.asked[0].fetch_min(address, Ordering::Relaxed);
            }
            let end = address.saturating_add(len.max(1));
            if shadow.asked[1].load(Ordering::Relaxed) < end {
                shadow.asked[1].fetch_max(end, Ordering::Relaxed);
            }
            if len <= 1 {
                shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
            }
        }
        return Some(shadow);
    }
    LOOKUPS[1].fetch_add(1, Ordering::Relaxed);
    let shadow = lookup_tables(address, len, any, false)?;
    LOOKUP_FOUND.with(|f| {
        let f = &mut *f.borrow_mut();
        // Tagged with the count read before the tables: any mutation since
        // clears it on the next ask.
        if f.0 == changes {
            if f.1.len() >= LOOKUP_CACHE_MOST {
                f.1.clear();
            }
            f.1.insert((address, len, any), Arc::downgrade(&shadow));
        }
    });
    Some(shadow)
}

fn lookup_tables(address: u64, len: u64, any: bool, quick: bool) -> Option<Arc<Shadow>> {
    if quick && let Some(shadow) = found(address, len) {
        return Some(shadow);
    }
    let end = address.checked_add(len.max(1))?;
    {
        let map = buffers_read();
        match map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end) {
            // The copy is made already (nearly always).
            Some((_, Entry { shadow: Some(shadow), .. })) => {
                if quick {
                    // (Nobody changes the map while this lock is held.)
                    remember(address, len, BUFFERS_CHANGED.load(Ordering::SeqCst), shadow);
                }
                return Some(shadow.clone());
            }
            Some(_) => {}
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
        }
    }
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the CPU's side is made beside it in PC memory.
        // Otherwise, when ReBAR GPU upload copies are enabled for title
        // buffers, allocate in full-speed GDDR before falling back to in-place
        // system RAM (`crate::inplace::place_for`, only for `!big`) and finally
        // a normal PC shared buffer.
        let big = cards() && size >= card_limits().0;
        let (buffer, data, card, bar) = match card_for(size) {
            Some(on_card) => {
                let Some((staging, data)) = host().and_then(|h| h.shared_buffer(size)) else {
                    CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
                    return None;
                };
                crate::host::name(&staging, || format!("title buffer {:#x}+{len:#x}", key.0));
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    side: std::sync::RwLock::new(()),
                    size,
                    me: std::sync::OnceLock::new(),
                };
                (on_card, data, Some(card), None)
            }
            None => {
                if let Some((buffer, data, bar)) = try_gpu_upload_buffer(size, false) {
                    crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x} (GPU upload heap)", key.0));
                    (buffer, data, None, Some(bar))
                } else if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
                    entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
                    return entry.shadow.clone();
                } else {
                    let (buffer, data) = host()?.shared_buffer(size)?;
                    crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
                    (buffer, data, None, None)
                }
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        shadow.bar = bar;
        let shadow = Arc::new(shadow);
        if let Some(card) = &shadow.card {
            let _ = card.me.set(Arc::downgrade(&shadow));
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: std::sync::RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = std::sync::RwLock::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| {
        // (Read first: the lines are shared by all recording threads.)
        let now = crate::host::moment();
        if shadow.used.load(Ordering::Relaxed) != now {
            shadow.used.store(now, Ordering::Relaxed);
        }
        if shadow.asked[0].load(Ordering::Relaxed) > address {
            shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        }
        if shadow.asked[1].load(Ordering::Relaxed) < end {
            shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        }
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    // The window is there already (nearly always): readers share the lock.
    if !(unsized_ask && tails()) && !(any && fine() && !unsized_ask) {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(shadow) = map.get(&named) {
            note(shadow);
            return Some(shadow.clone());
        }
    }
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    let (start, len) = if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    };
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    // When ReBAR GPU upload copies are enabled, allocate hot windows in
    // full-speed GDDR before falling back to in-place system RAM, card-sided
    // windows, or normal PC shared memory. When `VICTORY_GPU_UPLOAD_COPIES=0`,
    // `try_gpu_upload_buffer` returns `None` immediately, preserving exact
    // existing priority.
    let (buffer, data, card, bar) = if let Some((buffer, data, bar)) = try_gpu_upload_buffer(len, true) {
        crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} (GPU upload heap)"));
        (buffer, data, None, Some(bar))
    } else if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        tables_changed();
        return Some(shadow);
    } else {
        // With room for windows on the card (`window_card_most`): shaders get a
        // buffer there, and the CPU's side is made beside it in PC memory.
        match window_card_for(len) {
            Some(on_card) => {
                let Some((staging, data)) = host().and_then(|h| h.shared_buffer(len)) else {
                    WINDOW_CARD_BYTES.fetch_sub(len, Ordering::Relaxed);
                    return None;
                };
                crate::host::name(&staging, || format!("title memory {start:#x}+{len:#x}"));
                crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
                let card = Card {
                    staging,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    side: std::sync::RwLock::new(()),
                    size: len,
                    me: std::sync::OnceLock::new(),
                };
                (on_card, data, Some(card), None)
            }
            None => {
                let (buffer, data) = host()?.shared_buffer(len)?;
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
                (buffer, data, None, None)
            }
        }
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    shadow.bar = bar;
    let shadow = Arc::new(shadow);
    if let Some(card) = &shadow.card {
        let _ = card.me.set(Arc::downgrade(&shadow));
        CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
    }
    note(&shadow);
    map.insert((start, len), shadow.clone());
    tables_changed();
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data, bar) = pc_copy_buffer(len.next_multiple_of(256), false)?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, address, len);
    shadow.bar = bar;
    let shadow = Arc::new(shadow);
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_OWN_COPY=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }

    #[test]
    fn gpu_upload_copies_parsing_defaults_and_clamps() {
        use super::{parse_gpu_upload_copies, rebar_enabled_for};
        assert_eq!(parse_gpu_upload_copies(None, None), (0, 2048 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("1"), None), (1, 2048 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("2"), Some("1024")), (2, 1024 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("9"), Some(" 512 ")), (2, 512 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("bad"), Some("bad")), (0, 2048 << 20));
        // Large ReBAR budgets accepted without artificial caps.
        assert_eq!(parse_gpu_upload_copies(Some("1"), Some("2560")), (1, 2560 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("2"), Some("3072")), (2, 3072 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("2"), Some("4096")), (2, 4096 << 20));
        assert_eq!(parse_gpu_upload_copies(Some("2"), Some("0")), (2, 0));
        // ReBAR eligibility matches level and non-zero budget for windows vs buffers.
        assert!(!rebar_enabled_for(0, 2048 << 20, true));
        assert!(!rebar_enabled_for(0, 2048 << 20, false));
        assert!(rebar_enabled_for(1, 2048 << 20, true));
        assert!(!rebar_enabled_for(1, 2048 << 20, false));
        assert!(rebar_enabled_for(2, 4096 << 20, true));
        assert!(rebar_enabled_for(2, 4096 << 20, false));
        assert!(!rebar_enabled_for(2, 0, true));
        assert!(!rebar_enabled_for(2, 0, false));
    }
}

#[cfg(test)]
mod card_merge_tests {
    use super::{Card, Shadow};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    /// A plain buffer on the software adapter: what a copy's `buffer` and
    /// its card's `staging` are; the merge never touches them.
    fn warp_buffer() -> Option<ID3D12Resource> {
        // SAFETY: plain DXGI and D3D12 creation calls on the software adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).ok()?;
            let device = device?;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 10,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
            buffer
        }
    }

    /// The mixed landing the fetch inside flights must survive, on a real
    /// copy in the card's memory and its merge (`Shadow::card_merge_relay`):
    /// flights 5, 6, 7 in the air; 5 lands the old way with the queue's
    /// count at 7; the title wrote a byte that went up in send 7 into a
    /// piece flight 6's shaders write; 6 then lands from its own slot, which
    /// holds that byte as it was before send 7. The notes of sends are
    /// dropped as `card_dirty` drops them, below the mark the old-way fetch
    /// sets (`crate::host::fallback_fetched_mark`): with the mark at the
    /// landed plans' send the title's byte survives and the shader's write
    /// lands; with the mark at the fetch's count (as before the fix) the
    /// title's byte is put back to the card's older one.
    #[test]
    fn a_title_byte_sent_after_a_flight_survives_its_slot() {
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let (title_byte, shader_byte) = (100usize, 200usize);
        let content: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        let land = |mark: u64| -> Vec<u8> {
            let mut title = content.clone();
            let mut side = content.clone();
            // The title's byte, refreshed into the CPU's side and sent up in send 7.
            title[title_byte] ^= 0x5a;
            side[title_byte] = title[title_byte];
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: staging.clone(),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                side: std::sync::RwLock::new(()),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
                me: std::sync::OnceLock::new(),
            });
            // As `card_dirty` with the mark set by the old-way fetch.
            if let Some(card) = &shadow.card {
                card.sent.lock().unwrap().retain(|s| s.2 > mark);
            }
            // Flight 6's slot: the card's bytes behind its lists, before send 7.
            let mut slot = content.clone();
            slot[shader_byte] ^= 0xc3;
            let written = shadow.card_merge_relay(0, &slot, 6, &mut Vec::new());
            assert!(written > 0);
            drop(shadow);
            title
        };
        let fixed = land(crate::host::fallback_fetched_mark(true, 7, Some(5)));
        assert_eq!(fixed[title_byte], content[title_byte] ^ 0x5a, "the title's byte was lost");
        assert_eq!(fixed[shader_byte], content[shader_byte] ^ 0xc3, "the shader's write did not land");
        // The mark at the fetch's count: the note of send 7 is gone and the
        // older card byte is taken for a shader's write.
        let before = land(7);
        assert_eq!(before[title_byte], content[title_byte]);
    }

    static BUFFERS_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn sole_buffer_copy_rejects_overlapping_buffers_and_loose_windows() {
        let _serial = BUFFERS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(buffer) = warp_buffer() else {
            return;
        };
        let outer = 0x7FF0_1000_0000u64;
        let inner = 0x7FF0_1001_0000u64;
        let loose_addr = 0x7FF0_2000_0000u64;
        let mut backing = vec![0u8; 64 << 10];
        let shadow_outer = std::sync::Arc::new(Shadow::new(
            buffer.clone(),
            backing.as_mut_ptr(),
            0,
            outer,
            256 << 10,
        ));
        let shadow_inner = std::sync::Arc::new(Shadow::new(
            buffer,
            backing.as_mut_ptr(),
            0,
            inner,
            16 << 10,
        ));
        {
            let mut map = super::buffers();
            map.insert(
                (outer, 0x9001),
                super::Entry {
                    end: outer + (256 << 10),
                    shadow: Some(shadow_outer),
                },
            );
        }
        assert!(super::sole_buffer_copy(outer, 256).is_ok());

        // Overlapping registered buffer shadow must return Err(NotSole::OtherCopy).
        {
            let mut map = super::buffers();
            map.insert(
                (inner, 0x9002),
                super::Entry {
                    end: inner + (16 << 10),
                    shadow: Some(shadow_inner.clone()),
                },
            );
        }
        assert_eq!(
            super::sole_buffer_copy(outer, 128 << 10).err().unwrap(),
            super::NotSole::OtherCopy
        );
        {
            let mut map = super::buffers();
            map.remove(&(inner, 0x9002));
        }
        // Loose window overlapping the range must also be rejected.
        {
            let mut loose = super::LOOSE.write().unwrap();
            loose.insert((outer + 0x2_0000, 0x1_0000), shadow_inner);
            super::tables_changed();
        }
        assert_eq!(
            super::sole_buffer_copy(outer + 0x2_0000, 256).err().unwrap(),
            super::NotSole::OtherCopy
        );
        assert_eq!(
            super::sole_buffer_copy(loose_addr, 256).err().unwrap(),
            super::NotSole::NoBuffer
        );
        {
            super::LOOSE.write().unwrap().remove(&(outer + 0x2_0000, 0x1_0000));
            super::tables_changed();
            super::buffers().remove(&(outer, 0x9001));
        }
    }

    #[test]
    fn quick_find_caches_and_invalidates_on_buffers_change_and_card_hashed_uses_own_len() {
        let _serial = BUFFERS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        let addr = 0x7FF0_5000_0000u64;
        let mut backing = vec![0u8; 64 << 10];
        let mut s = Shadow::new(buffer, backing.as_mut_ptr(), 0, addr, 290);
        s.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            side: std::sync::RwLock::new(()),
            size: 64 << 10,
            me: std::sync::OnceLock::new(),
        });
        assert_eq!(s.card_hash_size(), 304);
        assert_eq!(s.card_hashed(0), (0, 304));
        // Clear `card` before `Drop` so global `CARD_BYTES` is untouched.
        s.card = None;

        let shadow = std::sync::Arc::new(s);
        {
            let mut map = super::buffers();
            map.insert(
                (addr, 0x9101),
                super::Entry {
                    end: addr + 290,
                    shadow: Some(shadow.clone()),
                },
            );
        }
        // First lookup populates thread-local cache; retry if another parallel
        // test outside this module mutates `BUFFERS` mid-check.
        let mut hit = false;
        for _ in 0..16 {
            let before = super::BUFFERS_CHANGED.load(std::sync::atomic::Ordering::SeqCst);
            let got1 = super::find(addr, 64).expect("first lookup");
            assert!(std::sync::Arc::ptr_eq(&got1, &shadow));
            let cached = super::found(addr, 64);
            let after = super::BUFFERS_CHANGED.load(std::sync::atomic::Ordering::SeqCst);
            if before == after {
                let cached = cached.expect("thread-local cache hit with unchanged BUFFERS_CHANGED");
                assert!(std::sync::Arc::ptr_eq(&cached, &shadow));
                let got2 = super::find(addr, 64).expect("cached lookup");
                assert!(std::sync::Arc::ptr_eq(&got2, &shadow));
                hit = true;
                break;
            }
        }
        assert!(hit, "expected a stable BUFFERS_CHANGED window for quick_find cache hit");

        // Any mutation of `BUFFERS` via `buffers()` invalidates the cache.
        super::unregister(0x9101, addr);
        assert!(super::found(addr, 64).is_none());
    }

    #[test]
    fn lookup_cache_caches_buffers_and_windows_and_invalidates_on_all_mutations() {
        use std::sync::atomic::Ordering;
        let _serial = BUFFERS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(super::parse_lookup_cache_mode(None), 0);
        assert_eq!(super::parse_lookup_cache_mode(Some("0")), 0);
        assert_eq!(super::parse_lookup_cache_mode(Some("1")), 1);
        assert_eq!(super::parse_lookup_cache_mode(Some(" 1 ")), 1);
        assert_eq!(super::parse_lookup_cache_mode(Some("alt")), 2);
        assert_eq!(super::parse_lookup_cache_mode(Some(" alt ")), 2);
        assert_eq!(super::parse_lookup_cache_mode(Some("bad")), 0);

        let Some(buffer) = warp_buffer() else {
            return;
        };
        let outer = 0x7FF0_6000_0000u64;
        let inner = 0x7FF0_6000_1000u64;
        let win_start = 0x7FF0_7000_0000u64;
        let mut backing = vec![0u8; 64 << 10];
        let shadow_outer = std::sync::Arc::new(Shadow::new(
            buffer.clone(),
            backing.as_mut_ptr(),
            0,
            outer,
            0x1_0000,
        ));
        let shadow_inner = std::sync::Arc::new(Shadow::new(
            buffer.clone(),
            backing.as_mut_ptr(),
            0,
            inner,
            0x1000,
        ));
        let mut win_shadow = Shadow::new(buffer, backing.as_mut_ptr(), 0, win_start, super::WINDOW);
        win_shadow.window = true;
        let win_shadow = std::sync::Arc::new(win_shadow);

        // 1) Cache miss -> hit on `BUFFERS`, then invalidation when an inner buffer is registered.
        {
            let mut map = super::buffers();
            map.insert(
                (outer, 0x9201),
                super::Entry {
                    end: outer + 0x1_0000,
                    shadow: Some(shadow_outer.clone()),
                },
            );
        }
        let misses0 = super::LOOKUPS[1].load(Ordering::Relaxed);
        let hits0 = super::LOOKUPS[0].load(Ordering::Relaxed);
        let first = super::lookup_cached(inner, 64, false).expect("outer buffer miss->populate");
        assert!(std::sync::Arc::ptr_eq(&first, &shadow_outer));
        assert!(super::LOOKUPS[1].load(Ordering::Relaxed) > misses0);

        let second = super::lookup_cached(inner, 64, false).expect("outer buffer cache hit");
        assert!(std::sync::Arc::ptr_eq(&second, &shadow_outer));
        assert!(super::LOOKUPS[0].load(Ordering::Relaxed) > hits0);

        // Registering an inner buffer bumps `TABLES_CHANGED` and invalidates the stale outer entry.
        let gen_before = super::TABLES_CHANGED.load(Ordering::Acquire);
        {
            let mut map = super::buffers();
            map.insert(
                (inner, 0x9202),
                super::Entry {
                    end: inner + 0x1000,
                    shadow: Some(shadow_inner.clone()),
                },
            );
        }
        assert!(super::TABLES_CHANGED.load(Ordering::Acquire) > gen_before);
        let after_insert = super::lookup_cached(inner, 64, false).expect("inner buffer after invalidation");
        assert!(std::sync::Arc::ptr_eq(&after_insert, &shadow_inner));

        // Unregistering the inner buffer bumps `TABLES_CHANGED` and returns to the outer buffer.
        let gen_before = super::TABLES_CHANGED.load(Ordering::Acquire);
        super::unregister(0x9202, inner);
        assert!(super::TABLES_CHANGED.load(Ordering::Acquire) > gen_before);
        let after_unreg = super::lookup_cached(inner, 64, false).expect("outer buffer after unregister");
        assert!(std::sync::Arc::ptr_eq(&after_unreg, &shadow_outer));

        // 2) Cache miss -> hit on `LOOSE` window updates `asked` & `unsized_asks` on hits.
        let gen_before = super::TABLES_CHANGED.load(Ordering::Acquire);
        {
            let mut loose = super::LOOSE.write().unwrap();
            loose.insert((win_start, super::WINDOW), win_shadow.clone());
            super::tables_changed();
        }
        assert!(super::TABLES_CHANGED.load(Ordering::Acquire) > gen_before);
        let win_ask = win_start + 0x1234;
        let w1 = super::lookup_cached(win_ask, 1, false).expect("loose window miss->populate");
        assert!(std::sync::Arc::ptr_eq(&w1, &win_shadow));
        let unsized_before = win_shadow.unsized_asks.load(Ordering::Relaxed);
        let hits_before = super::LOOKUPS[0].load(Ordering::Relaxed);
        let w2 = super::lookup_cached(win_ask, 1, false).expect("loose window cache hit");
        assert!(std::sync::Arc::ptr_eq(&w2, &win_shadow));
        assert!(super::LOOKUPS[0].load(Ordering::Relaxed) > hits_before);
        assert_eq!(win_shadow.unsized_asks.load(Ordering::Relaxed), unsized_before + 1);
        assert!(win_shadow.asked[0].load(Ordering::Relaxed) <= win_ask);
        assert!(win_shadow.asked[1].load(Ordering::Relaxed) > win_ask);

        // Cleanup tables and bump generation so no test entries remain.
        {
            super::LOOSE.write().unwrap().remove(&(win_start, super::WINDOW));
            super::tables_changed();
            super::unregister(0x9201, outer);
        }
    }

    #[test]
    fn merge_gave_up_counters_take_and_reset() {
        use std::sync::atomic::Ordering;
        let _serial = BUFFERS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = super::merge_gave_up();
        for (i, n) in [2u64, 5, 7, 11].into_iter().enumerate() {
            super::MERGE_GAVE_UP[i].fetch_add(n, Ordering::Relaxed);
        }
        assert_eq!(super::merge_gave_up(), [2, 5, 7, 11]);
        assert_eq!(super::merge_gave_up(), [0, 0, 0, 0]);
    }

    #[test]
    fn copy_stale_and_fused_merge_under_writes_on_write_watched_card_copy() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 2 * super::PAGE;
        // SAFETY: private write-watched allocation freed before returning.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!title.is_null());
        // SAFETY: committed allocation of `LEN` bytes.
        unsafe { std::ptr::write_bytes(title, 0x3f, LEN) };
        let mut side = vec![0u8; LEN];
        let mut shadow = Shadow::new(buffer, side.as_mut_ptr(), 0, title as u64, LEN as u64);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            side: std::sync::RwLock::new(()),
            size: 0,
            me: std::sync::OnceLock::new(),
        });
        // Initial refresh copies both 64 KiB pages whole.
        let addr = title as usize;
        assert_eq!(shadow.update(addr, LEN), LEN);
        assert!(side.iter().all(|&b| b == 0x3f));
        shadow.card.as_ref().unwrap().dirty.lock().unwrap().clear();

        // Write a single byte inside 4 KiB sub-page 5 of page 0: refresh copies only 4 KiB (not 64 KiB).
        // SAFETY: inside allocation.
        unsafe { title.add(5 * 4096 + 12).write_volatile(0x99) };
        let copied = shadow.update(addr, LEN);
        assert_eq!(copied, 4096);
        assert_eq!(side[5 * 4096 + 12], 0x99);
        assert_eq!(*shadow.card.as_ref().unwrap().dirty.lock().unwrap(), vec![(5 * 4096, 6 * 4096)]);
        shadow.card.as_ref().unwrap().dirty.lock().unwrap().clear();

        // Now mark page 1 pending (shader writes in flight) and modify 64 consecutive
        // dwords (only byte 0 of each dword) in title memory: fused merge collapses
        // them into ONE contiguous 256-byte run and updates `side` + `card.dirty`.
        shadow.pending.lock().unwrap().push((addr + super::PAGE, addr + 2 * super::PAGE));
        for w in 0..64 {
            // SAFETY: inside page 1 of our allocation.
            unsafe { title.add(super::PAGE + 128 + w * 4).write_volatile(0x77) };
        }
        let mut runs_left = 16;
        let gave_up = shadow.merge_under_writes(addr + super::PAGE, super::PAGE, &mut runs_left);
        assert!(!gave_up);
        assert_eq!(runs_left, 15, "64 consecutive changed dwords collapse to 1 run");
        for w in 0..64 {
            assert_eq!(side[super::PAGE + 128 + w * 4], 0x77);
        }
        assert_eq!(
            *shadow.card.as_ref().unwrap().dirty.lock().unwrap(),
            vec![(super::PAGE + 128, super::PAGE + 128 + 256)]
        );
        shadow.card = None;
        drop(shadow);
        // SAFETY: free our private write-watched allocation.
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
        crate::watch::forget_test_pages(addr, addr + LEN);
    }

    #[test]
    fn parallel_merge_under_writes_updates_all_pages_and_honors_run_budget() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const PAGES: usize = 24;
        const LEN: usize = PAGES * super::PAGE;
        // SAFETY: private write-watched allocation freed before returning.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!title.is_null());
        // SAFETY: committed allocation of `LEN` bytes.
        unsafe { std::ptr::write_bytes(title, 0x22, LEN) };
        let mut side = vec![0u8; LEN];
        let mut shadow = Shadow::new(buffer, side.as_mut_ptr(), 0, title as u64, LEN as u64);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            side: std::sync::RwLock::new(()),
            size: 0,
            me: std::sync::OnceLock::new(),
        });
        let addr = title as usize;
        assert_eq!(shadow.update(addr, LEN), LEN);
        shadow.card.as_ref().unwrap().dirty.lock().unwrap().clear();

        // Mark all 24 pages pending so `merge_under_writes` takes the parallel path (>= 16 pages).
        // Even pages get same-byte echo writes (AVX2 first_diff fast path);
        // odd pages get two disjoint 64-byte modified runs each (12 * 2 = 24 runs total).
        shadow.pending.lock().unwrap().push((addr, addr + LEN));
        for p in 0..PAGES {
            let base = p * super::PAGE;
            if p % 2 == 0 {
                // SAFETY: inside page `p`.
                unsafe { title.add(base + 16).write_volatile(0x22) };
            } else {
                for b in 0..64 {
                    // SAFETY: inside page `p`.
                    unsafe {
                        title.add(base + 256 + b).write_volatile(0xA0 | (p as u8));
                        title.add(base + 4096 + b).write_volatile(0xB0 | (p as u8));
                    }
                }
            }
        }
        let mut runs_left = 128;
        let gave_up = shadow.merge_under_writes(addr, LEN, &mut runs_left);
        assert!(!gave_up);
        assert_eq!(runs_left, 128 - 24);
        let mut dirty = shadow.card.as_ref().unwrap().dirty.lock().unwrap().clone();
        dirty.sort_unstable();
        assert_eq!(dirty.len(), 24);
        for p in 0..PAGES {
            let base = p * super::PAGE;
            if p % 2 == 0 {
                assert_eq!(side[base + 16], 0x22);
            } else {
                assert_eq!(side[base + 256], 0xA0 | (p as u8));
                assert_eq!(side[base + 4096], 0xB0 | (p as u8));
                assert_eq!(side[base + 256 + 64], 0x22);
            }
        }

        // Budget exhaustion across >= 16 dirty pages signals `gave_up = true`.
        shadow.card.as_ref().unwrap().dirty.lock().unwrap().clear();
        for p in 0..PAGES {
            // SAFETY: inside page `p`.
            unsafe { title.add(p * super::PAGE + 512).write_volatile(0xCC) };
        }
        let mut tiny_budget = 4;
        assert!(shadow.merge_under_writes(addr, LEN, &mut tiny_budget));
        assert_eq!(tiny_budget, 0);

        shadow.card = None;
        drop(shadow);
        // SAFETY: free our private write-watched allocation.
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
        crate::watch::forget_test_pages(addr, addr + LEN);
    }

    #[test]
    fn gpu_readonly_pool_skips_gpu_write_tracking_and_preserves_cpu_uploads() {
        struct ClearGuard;
        impl Drop for ClearGuard {
            fn drop(&mut self) {
                crate::title_hints::clear_gpu_readonly_ranges();
            }
        }

        let _serial = BUFFERS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::title_hints::clear_gpu_readonly_ranges();
        let _clear = ClearGuard;
        let _ = super::readonly_stats();

        let (Some(buffer), Some(staging), Some(twin)) = (warp_buffer(), warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 2 * super::PAGE;
        let mut title = vec![0x42u8; LEN];
        let base = title.as_ptr() as u64;
        let mut side = vec![0u8; LEN];

        // Register the entire buffer as Xbox GPU-read-only.
        crate::title_hints::set_gpu_memory_readonly(base, LEN as u64, true);

        // 1) PC-side shadow: `writing()` skips RO range, bumps READONLY_SKIPPED,
        //    leaves `pending` empty and `before` unallocated.
        let pc_shadow = std::sync::Arc::new(Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, base, LEN as u64));
        assert!(!pc_shadow.writing(base as usize, 4096));
        assert!(!pc_shadow.writes_pending());
        assert!(pc_shadow.before.lock().unwrap().is_none());
        assert_eq!(super::readonly_stats(), [1, 4096]);
        assert_eq!(super::readonly_stats(), [0, 0]);

        // `gpu_writes()` also skips RO range before deferring and bumps stats.
        if let Some(host) = crate::host::host()
            && let Ok(list) = host.new_list()
        {
            pc_shadow.gpu_writes(&list, base + 512, 2048);
            assert_eq!(super::readonly_stats(), [1, 2048]);
            // Clamped unsized write inside RO range is also skipped with clamped length.
            pc_shadow.gpu_writes(&list, base + (LEN as u64 - 1024), u64::MAX);
            assert_eq!(super::readonly_stats(), [1, 1024]);
        }

        // 2) Carded shadow wholly inside RO range: `card_make_before` skips
        //    allocating the "before" twin, while CPU->GPU `update()` still works!
        let mut card_shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, base, LEN as u64);
        card_shadow.card = Some(Card {
            staging: staging.clone(),
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            side: std::sync::RwLock::new(()),
            size: 0,
            me: std::sync::OnceLock::new(),
        });
        let mut alloc_called = false;
        assert!(!card_shadow.card_make_before(|_| {
            alloc_called = true;
            Some(twin.clone())
        }));
        assert!(!alloc_called, "wholly read-only card shadow must not allocate a before twin");
        assert!(card_shadow.card_sides().unwrap().1.is_none());

        // CPU->GPU upload (`update`) MUST continue working normally on RO pools.
        title[128] = 0xAB;
        let copied = card_shadow.update(base as usize, 4096);
        assert!(copied >= 4096);
        assert_eq!(side[128], 0xAB);
        assert!(!card_shadow.card.as_ref().unwrap().dirty.lock().unwrap().is_empty());

        // 3) Carve out second page as RW (`set_gpu_memory_readonly(..., false)`):
        //    - First page (still RO) is skipped.
        //    - Second page (now RW) is NOT skipped.
        //    - Range spanning across the RO/RW boundary is NOT skipped.
        //    - Partially-RO card shadow CAN allocate its "before" twin.
        let page1 = base + super::PAGE as u64;
        crate::title_hints::set_gpu_memory_readonly(page1, super::PAGE as u64, false);
        assert!(!pc_shadow.writing(base as usize, 1024));
        assert_eq!(super::readonly_stats(), [1, 1024]);

        assert!(pc_shadow.writing(page1 as usize, 1024));
        assert!(pc_shadow.writes_pending());
        assert_eq!(super::readonly_stats(), [0, 0]);
        pc_shadow.copy_back_done(&[(page1 as usize, page1 as usize + 1024)]);
        assert!(!pc_shadow.writes_pending());

        // Spanning RO->RW boundary must NOT be skipped.
        assert!(pc_shadow.writing((page1 - 512) as usize, 1024));
        assert!(pc_shadow.writes_pending());
        assert_eq!(super::readonly_stats(), [0, 0]);
        pc_shadow.copy_back_done(&[((page1 - 512) as usize, (page1 + 512) as usize)]);

        // Partially-RO card shadow allocates its "before" twin normally.
        assert!(card_shadow.card_make_before(|_| {
            alloc_called = true;
            Some(twin.clone())
        }));
        assert!(alloc_called);
        assert!(card_shadow.card_sides().unwrap().1.is_some());

        // `card.size == 0`, so `Drop` subtracts `card_hash_bytes(0)` cleanly and leaves global CARD_BYTES unchanged.
        drop(card_shadow);
        drop(pc_shadow);
        crate::title_hints::clear_gpu_readonly_ranges();
    }

    #[test]
    fn changed_list_collects_dirty_card_copies_from_single_and_batch_runs() {
        assert_eq!(super::parse_changed_list_mode(None), 0);
        assert_eq!(super::parse_changed_list_mode(Some("0")), 0);
        assert_eq!(super::parse_changed_list_mode(Some("1")), 1);
        assert_eq!(super::parse_changed_list_mode(Some(" 1 ")), 1);
        assert_eq!(super::parse_changed_list_mode(Some("alt")), 2);
        assert_eq!(super::parse_changed_list_mode(Some(" alt ")), 2);
        assert_eq!(super::parse_changed_list_mode(Some("bogus")), 0);

        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        let mut backing_a = vec![0u8; 64 << 10];
        let mut backing_b = vec![0u8; 64 << 10];
        let mut sa = Shadow::new(buffer.clone(), backing_a.as_mut_ptr(), 0, 0x1000_0000, 64 << 10);
        sa.card = Some(Card {
            staging: staging.clone(),
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            side: std::sync::RwLock::new(()),
            size: 0,
            me: std::sync::OnceLock::new(),
        });
        let sa = std::sync::Arc::new(sa);
        let _ = sa.card.as_ref().unwrap().me.set(std::sync::Arc::downgrade(&sa));

        let mut sb = Shadow::new(buffer, backing_b.as_mut_ptr(), 0, 0x2000_0000, 64 << 10);
        sb.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            side: std::sync::RwLock::new(()),
            size: 0,
            me: std::sync::OnceLock::new(),
        });
        let sb = std::sync::Arc::new(sb);
        let _ = sb.card.as_ref().unwrap().me.set(std::sync::Arc::downgrade(&sb));

        // Dirty `sa` via `card_changed` (overlapping + disjoint) and `sb` via `card_changed_runs`.
        sa.card_changed(0x100, 0x300);
        sa.card_changed(0x200, 0x500);
        sa.card_changed(0x1000, 0x1200);
        sb.card_changed_runs([(0x400, 0x600), (0x600, 0x800), (0x2000, 0x2100)]);

        let taken = super::take_changed_from(
            vec![
                std::sync::Arc::downgrade(&sa),
                std::sync::Arc::downgrade(&sb),
                std::sync::Arc::downgrade(&sa), // duplicate entry yields empty on second visit
            ],
            1,
        );
        assert_eq!(taken.len(), 2);
        assert!(std::sync::Arc::ptr_eq(&taken[0].0, &sa));
        assert_eq!(taken[0].1, vec![(0x100, 0x500), (0x1000, 0x1200)]);
        assert!(std::sync::Arc::ptr_eq(&taken[1].0, &sb));
        assert_eq!(taken[1].1, vec![(0x400, 0x800), (0x2000, 0x2100)]);

        // Draining again finds nothing left.
        assert!(super::take_changed_from(vec![std::sync::Arc::downgrade(&sa)], 2).is_empty());
    }

    #[test]
    fn copy_profile_ranks_top_10_uploads_and_copy_backs_concisely() {
        let mut entries = std::collections::HashMap::new();
        assert!(super::format_copy_profile(&entries).is_none());

        for i in 1u64..=12 {
            entries.insert(
                (0x1000_0000 + i * 0x10_0000, i * 0x1_0000),
                super::CopyProfileEntry {
                    upload_bytes: i * (1 << 20),
                    upload_count: i * 2,
                    back_bytes: (13 - i) * (512 << 10),
                    back_count: 13 - i,
                },
            );
        }
        let line = super::format_copy_profile(&entries).expect("non-empty summary");
        assert!(line.starts_with("copy-profile: top-upload=["), "{line}");
        assert!(line.contains("] top-back=["), "{line}");
        assert!(line.ends_with(']'), "{line}");

        let (up_part, back_part) = line
            .strip_prefix("copy-profile: top-upload=[")
            .unwrap()
            .split_once("] top-back=[")
            .unwrap();
        let back_part = back_part.strip_suffix(']').unwrap();
        let up_items: Vec<&str> = up_part.split(", ").collect();
        let back_items: Vec<&str> = back_part.split(", ").collect();
        assert_eq!(up_items.len(), 10, "top-upload capped at 10");
        assert_eq!(back_items.len(), 10, "top-back capped at 10");
        // Highest upload offender is i=12 (12.00MB/24); lowest two (i=1, i=2) are omitted.
        assert_eq!(up_items[0], "0x10c00000+0xc0000:12.00MB/24");
        assert!(!up_part.contains("0x10100000+"));
        assert!(!up_part.contains("0x10200000+"));
        // Highest copy-back offender is i=1 (6.00MB/12); lowest two (i=11, i=12) are omitted.
        assert_eq!(back_items[0], "0x10100000+0x10000:6.00MB/12");
        assert!(!back_part.contains("0x10b00000+"));
        assert!(!back_part.contains("0x10c00000+"));
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers_read()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
