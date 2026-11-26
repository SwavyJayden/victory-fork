//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! Submissions are sent ahead of the wait for the card by default, so CPU
//! and GPU overlap ([`late`], `VICTORY_LATE_WAIT=0` waits for each);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_FAILED, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    _rtvs: ID3D12DescriptorHeap,
    rtv_start: usize,
    rtv_size: u32,
    /// A ring of depth views, used as `_rtvs`.
    _dsvs: ID3D12DescriptorHeap,
    dsv_start: usize,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
static RETIRED: Mutex<Vec<(u64, Retired)>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Microseconds of a submission's preparing by part, since the frame-rate
/// line last took them: the asking and the look under flights, the lists'
/// steps, the flush of refreshes, the snapshots and the wait for card
/// results, `card_before`, the hand-over to the card. An instrument: the
/// frame line's "preparing" had 19 to 23 ms a frame no line split (run 282).
static PREP_PARTS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Adds the time since `since` to part `i` of [`PREP_PARTS`]; now.
fn prep_part(i: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    PREP_PARTS[i].fetch_add((now - since).as_micros() as u64, Ordering::Relaxed);
    now
}

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((now_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. High-VRAM adapters (> 12 GiB budget,
/// e.g. 16 GB RTX 5070 Ti) retain textures longer while multi-GB headroom
/// remains, preventing eviction/recreation thrashing. Never returns less
/// than 1,000 ms so handles not yet recorded into a list or slot remain
/// protected.
///
/// With room on the card nothing has to go: in session play-1125-0127 88,000
/// of them (31 GB) went after 5 s with 7 GB of the card free, and 24,700
/// (12 GB) had to be made and uploaded again.
/// `VICTORY_TEXTURE_KEEP=0`: disable extended retention (5 s / 15 s as before).
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    static KEEP: OnceLock<bool> = OnceLock::new();
    let keep = *KEEP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_KEEP").is_none_or(|v| v != "0"));
    vram_idle_for(video_memory, keep)
}

fn vram_idle_for(video_memory: Option<[u64; 4]>, keep: bool) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const KEPT_MS: u64 = 600_000;
    const LONG_MS: u64 = 30_000;
    const RELAXED_MS: u64 = 15_000;
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if budget > 12 * GIB {
        let free = budget.saturating_sub(used);
        if used_pct >= 92 || free < GIB {
            HIGH_MS
        } else if used_pct >= 85 || free < 2 * GIB {
            MODERATE_MS
        } else if used_pct >= 75 || free < 3 * GIB {
            NORMAL_MS
        } else if !keep {
            RELAXED_MS
        } else if used_pct >= 68 {
            LONG_MS
        } else {
            KEPT_MS
        }
    } else if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else if !keep {
        NORMAL_MS
    } else if used_pct >= 68 {
        LONG_MS
    } else {
        KEPT_MS
    }
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low and capped at 8,000 ms even on abundant-RAM
/// systems so idle WDDM-pinned upload heaps do not accumulate across scene cuts.
/// Never returns less than 2,000 ms.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const ABUNDANT_MS: u64 = 8_000;
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 3_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else if avail >= 16 * GIB {
        ABUNDANT_MS
    } else {
        NORMAL_MS
    }
}

/// Effective system-RAM idle timeout in milliseconds, honouring an explicit
/// `VICTORY_RAM_IDLE_MS` override (clamped to at least 1,000 ms) or scaling via
/// [`ram_idle_ms`].
pub(crate) fn effective_ram_idle_ms() -> u64 {
    static OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();
    let override_ms = *OVERRIDE.get_or_init(|| {
        std::env::var("VICTORY_RAM_IDLE_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|ms| ms.max(1_000))
    });
    override_ms.unwrap_or_else(|| ram_idle_ms(available_physical_ram()))
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((started, Retired(Box::new(object))));
}

/// Submissions started and finished so far.
pub(crate) fn submissions() -> (u64, u64) {
    (STARTED.load(Ordering::Acquire), FINISHED.load(Ordering::Acquire))
}

/// Drops what waited for submissions up to `finished`, and reveals texture
/// slots queued by `VICTORY_SHOW_AFTER=1` whose upload submission has run.
fn bury(finished: u64) {
    crate::descriptor::host::finish_flips(finished);
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired)
            .into_iter()
            .partition(|(at, _)| *at <= finished);
        *retired = keep;
        done
    };
    drop(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    // Sweeps also go to the log as they happen: the trail is only said at
    // a loss, and the driver's errors in the system's event log (run 224's
    // first fell inside a sweep of 4,711 host textures) need the sweeps of
    // clean runs beside them.
    if entry.2.starts_with("sweep:") {
        tracing::info!("{}", entry.2);
    }
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !trail_on() || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Time waiting to reuse texture staging read by a late submission.
static STAGING_WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Adds to [`SUBMISSION_US`] and to the frame log's totals.
fn submission_us(i: usize, us: u64) {
    SUBMISSION_US[i].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::PrepareUs, crate::framelog::T::CardWaitUs][i], us);
    if i == 0 {
        crate::framelog::add(crate::framelog::T::Submissions, 1);
    }
}

/// Adds to [`PREPARING`]'s copies (time, bytes) and to the frame log's totals.
fn preparing(i: usize, n: u64) {
    PREPARING[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::CopyUs, crate::framelog::T::CopyBytes][i], n);
}

/// Adds to [`COPY_BACK`] and to the frame log's totals.
fn copy_back(i: usize, n: u64) {
    COPY_BACK[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::CopyBackUs, crate::framelog::T::CopyBackBytes][i], n);
}

/// Adds to [`CARD_CLOCK_US`] and, for the card's idle times, to the frame
/// log's totals.
fn card_clock_us(i: usize, us: u64) {
    CARD_CLOCK_US[i].fetch_add(us, Ordering::Relaxed);
    match i {
        0 => crate::framelog::add(crate::framelog::T::CardIdleUs, us),
        1 => crate::framelog::add(crate::framelog::T::CardIdleBehindUs, us),
        _ => {}
    }
}

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    card_clock_us(2, since.elapsed().as_micros() as u64);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
fn card_clock_line(frames: f64, ms: f64) {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    crate::drvwatch::start();
    crate::spread::start();
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            device = Some(create_hardware_device()?);
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_start = rtvs.GetCPUDescriptorHandleForHeapStart().ptr;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_start = dsvs.GetCPUDescriptorHandleForHeapStart().ptr;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        if let Some(desc) = adapter.as_ref().and_then(|a| a.GetDesc1().ok()) {
            tracing::info!(
                adapter = %adapter_name(&desc),
                vendor = format_args!("{:#06x}", desc.VendorId),
                device = format_args!("{:#06x}", desc.DeviceId),
                vram_mb = desc.DedicatedVideoMemory >> 20,
                "host D3D12 adapter"
            );
        }
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            _rtvs: rtvs,
            rtv_start,
            rtv_size,
            _dsvs: dsvs,
            dsv_start,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

fn adapter_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let len = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..len])
}

fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)
}

/// Resolves `VICTORY_D3D12_ADAPTER` against DXGI adapter names: a 0-based
/// index or a case-insensitive substring of the adapter description.
fn match_adapter(spec: &str, names: &[String]) -> Option<usize> {
    let spec = spec.trim();
    if spec.is_empty() {
        return None;
    }
    if let Ok(index) = spec.parse::<usize>() {
        return (index < names.len()).then_some(index);
    }
    let needle = spec.to_ascii_lowercase();
    names
        .iter()
        .position(|name| name.to_ascii_lowercase().contains(&needle))
}

/// Picks the host hardware D3D12 device: `VICTORY_D3D12_ADAPTER` (0-based DXGI
/// index or case-insensitive name substring) when set, else the first hardware
/// adapter returned by `DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE`, falling back to
/// DXGI's default adapter (`None`).
fn create_hardware_device() -> windows::core::Result<ID3D12Device> {
    let spec = std::env::var("VICTORY_D3D12_ADAPTER")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    // SAFETY: plain DXGI adapter enumeration and D3D12CreateDevice calls.
    unsafe {
        if let Some(spec) = spec {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let mut adapters = Vec::new();
            let mut names = Vec::new();
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(i) else {
                    break;
                };
                let name = adapter
                    .GetDesc1()
                    .map(|d| adapter_name(&d))
                    .unwrap_or_default();
                adapters.push(adapter);
                names.push(name);
            }
            let Some(chosen) = match_adapter(&spec, &names) else {
                tracing::warn!(
                    spec,
                    available = ?names,
                    "VICTORY_D3D12_ADAPTER matched no DXGI adapter"
                );
                return Err(windows::core::Error::from_hresult(DXGI_ERROR_NOT_FOUND));
            };
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapters[chosen], D3D_FEATURE_LEVEL_12_0, &mut device)?;
            return device.ok_or_else(windows::core::Error::empty);
        }
        if let Ok(factory6) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) {
            for i in 0.. {
                let Ok(adapter) = factory6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    i,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) else {
                    break;
                };
                if adapter
                    .GetDesc1()
                    .is_ok_and(|desc| is_software_adapter(&desc))
                {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_ok()
                    && let Some(device) = device
                {
                    return Ok(device);
                }
            }
        }
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        device.ok_or_else(windows::core::Error::empty)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn adapter_selector_matches_index_or_substring() {
        let names = vec![
            "Intel(R) UHD Graphics".to_owned(),
            "NVIDIA GeForce RTX 3050 6GB Laptop GPU".to_owned(),
            "Microsoft Basic Render Driver".to_owned(),
        ];
        assert_eq!(super::match_adapter("0", &names), Some(0));
        assert_eq!(super::match_adapter(" 1 ", &names), Some(1));
        assert_eq!(super::match_adapter("3", &names), None);
        assert_eq!(super::match_adapter("nvidia", &names), Some(1));
        assert_eq!(super::match_adapter("RTX 3050", &names), Some(1));
        assert_eq!(super::match_adapter("uhd", &names), Some(0));
        assert_eq!(super::match_adapter("radeon", &names), None);
        assert_eq!(super::match_adapter("   ", &names), None);
    }
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn sparse_copies_use_plaincopy_on_committed_islands_around_holes() {
        use windows_sys::Win32::System::Memory::*;
        let _plain = crate::plaincopy::force_on_for_test();
        const N: usize = 5 << 16;
        // SAFETY: private reservation with committed islands, uncommitted holes,
        // and a committed PAGE_NOACCESS page, freed before return.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0u8; N];
            // Committed islands at start (block 0), middle (block 2), and end (block 4),
            // plus a committed PAGE_NOACCESS island at block 3.
            for (offset, len, value) in [(0, 4096, 0x11), (2 << 16, 2 * 4096, 0x22), (N - 4096, 4096, 0x33)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), len, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, len);
                expected[offset..offset + len].fill(value);
            }
            let noaccess = base.add(3 << 16);
            assert_eq!(VirtualAlloc(noaccess.cast(), 4096, MEM_COMMIT, PAGE_NOACCESS), noaccess.cast());

            // 1. Large sparse copy (>= 64 KiB -> `read_sparse`): even after the outer
            //    bulk `plaincopy::copy` faults and marks blocks 0 and 4 in `RECENT`,
            //    `read_sparse` copies all three committed islands in place via `plaincopy`.
            let (copies_before, bytes_before, _) = crate::plaincopy::copy_stats();
            let mut out = vec![0xEEu8; N];
            super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
            assert_eq!(out[27..N - 27], expected[27..N - 27]);
            assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xEE));
            let (copies_after, bytes_after, _) = crate::plaincopy::copy_stats();
            let committed_bytes = (4096 - 27) + 2 * 4096 + (4096 - 27);
            assert!(
                copies_after >= copies_before + 3,
                "expected at least 3 in-place sparse island copies, got {} -> {}",
                copies_before,
                copies_after
            );
            assert!(
                bytes_after >= bytes_before + committed_bytes as u64,
                "expected at least {committed_bytes} bytes copied in place, got {} -> {}",
                bytes_before,
                bytes_after
            );

            // 2. Small sparse copy (< 64 KiB -> `read_pieces`) across committed + reserved pages.
            let mut small = vec![0xDDu8; 3 * 4096];
            super::copy_mapped(small.as_mut_ptr(), base.add(2 << 16), 3 * 4096, false);
            assert_eq!(&small[..2 * 4096], &expected[2 << 16..(2 << 16) + 2 * 4096]);
            assert!(small[2 * 4096..].iter().all(|&b| b == 0));

            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: OnceLock<Option<Prelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted, and after
    /// it what the clamp notes about mesh records. In memory the CPU maps.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The same words in the card's own memory ([`card_budget`]), which the
    /// clamp uses instead when there is one: zeroed in the list before its
    /// first clamp, its notes copied to `budget` when the list closes.
    card_budget: OnceLock<Option<ID3D12Resource>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// With [`fill_on_card`]: a device-local buffer constant buffer views of
    /// the list read, filled by copies on the card ([`HostList::card_constants`]):
    /// the buffer, the bytes used, and whether it is in the constant buffer
    /// state (it decays to the common state when a submission has run).
    card_constants: Mutex<Option<(ID3D12Resource, u64, bool)>>,
    /// The buffers of `crate::indirect::window_check`, made at its first use.
    window_check: OnceLock<Option<crate::indirect::WindowCheck>>,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The pair of timestamps its run is timed with since its last reset
    /// (`crate::cardmeter`), or none.
    meter: AtomicU32,
    /// With `crate::cardmeter::pipelines`: the pair of the pipeline bound
    /// last, open until the next one is bound or the list closes.
    pipeline_meter: AtomicU32,
    /// The records its draws read from slots that its prelude fills
    /// (`crate::indirect::gather`).
    gather: Mutex<Gather>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// What a list gathers for its prelude to clamp (`crate::indirect::gather`).
#[derive(Default)]
struct Gather {
    /// Something recorded in the list may write arguments: no record is
    /// gathered from here on (the ones gathered before are read before
    /// the list runs, which is before that).
    closed: bool,
    /// Host copy IDs whose bytes a shader recorded in this list was declared
    /// to write (`VICTORY_GATHER_BY_COPY`).
    written: std::collections::HashSet<u64>,
    /// One set per argument buffer, kind and stride.
    sets: Vec<crate::indirect::GatherSet>,
    /// The budget words the gathered records are charged to.
    budget: Option<ID3D12Resource>,
    /// Scratch buffers of earlier recordings, kept for the next sets.
    spare: Vec<ID3D12Resource>,
}

/// The most argument buffers a list gathers from.
const GATHER_SETS: usize = 8;

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a list's indirect-work budget lives in the card's own memory
/// (`VICTORY_INDIRECT_CARD_BUDGET=0` keeps it in memory the CPU maps, as
/// before). The clamp charges it with atomics, at least two in a row for
/// every clamp; in mapped memory on a card of its own each of them
/// crosses the bus.
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping and its host GPU address.
struct Chunk {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        _buffer: buffer,
        data: data as usize,
        gpu,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, SPARE_MOST)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// The fetch inside flights ([`inflight_fetch`]), if it is on and was
    /// made.
    inflight: Option<CardInflight>,
}

/// What the fetch inside flights needs ([`inflight_fetch`], [`CARD_FETCH`]).
struct CardInflight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// PC memory, mapped (buffer, mapping, bytes): dword 0 the counter of
    /// tickets, then one dword per dword of `CardWork::differs`, the entry
    /// of that piece.
    table: (ID3D12Resource, usize, u64),
    /// PC memory, mapped (buffer, mapping, slots of [`INFLIGHT_SLOT`]
    /// bytes): where the pieces come down. With [`inflight_on_card`] the
    /// card's own memory, no mapping (0): the landing copies slots down.
    fetched: (ID3D12Resource, usize, u32),
    /// Every ticket below it belongs to a landed flight: its slot may be
    /// written over. Raised by the thread that lands flights; read when a
    /// compare's list is recorded (an older value only gives fewer slots).
    landed: u32,
    /// What fetches the old way brought (copy, offsets, and the highest
    /// send whose lists were surely on the queue ahead of the fetch: its
    /// order of the queue less one). A flight sent at or before that ran
    /// ahead of the fetch, so the CPU's side holds those bytes as the card
    /// had them after the flight, or newer: the flight's own slot is older
    /// there and is not merged there. Dropped once every flight sent up to
    /// it has landed.
    brought: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)>,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one thread per 64 KiB piece (cut at multiples of
/// 64 KiB of title memory, as `Shadow::card_piece` cuts them): `differs`
/// gets a dword per piece, 1 where a byte differs. The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged for a
/// byte just outside the range costs a fetch that finds nothing.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

/// As [`CARD_COMPARE`] (the same constants, cuts and results; another
/// hash), with a group of 64 threads per piece: the threads take the
/// piece's 16-byte steps in turn, 64 each, and their hashes are summed.
/// One thread per piece made 4,096 loads one after the other, so a pass
/// took milliseconds however few its pieces, and a wide one left the
/// card's load units waiting. A change in the steps of one thread always
/// changes that thread's hash, as before; the last mixing keeps changes in
/// several threads from cancelling out in the sum.
pub(crate) const CARD_COMPARE_WIDE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;
    uint skew;
    uint size;
    uint pieces;
    uint out_at;
    uint compare;
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 acc[256];
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    uint piece = first + group.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) ^ ((t + 1u) * 0x9E3779B1u);
    uint at = lo + t * 16u;
    // Eight loads at once: one after the other, each waits for memory.
    [loop]
    for (; at + 28672u < hi; at += 32768u) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + 4096u);
        uint4 v2 = now.Load4(at + 8192u);
        uint4 v3 = now.Load4(at + 12288u);
        uint4 v4 = now.Load4(at + 16384u);
        uint4 v5 = now.Load4(at + 20480u);
        uint4 v6 = now.Load4(at + 24576u);
        uint4 v7 = now.Load4(at + 28672u);
        h = (h ^ v0) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v1) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v2) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v3) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v4) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v5) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v6) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v7) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += 4096u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    // Two rounds in which every word goes into every other.
    h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
    h += h.yzwx;
    h = (h ^ (h >> 13)) * 0xC2B2AE35u;
    h += h.zwxy;
    h ^= h >> 16;
    // Atomic-free parallel tree reduction across all 256 threads.
    acc[t] = h;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (t < stride) {
            acc[t] += acc[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t != 0u || group.x >= pieces) return;
    uint4 all = acc[0];
    if (compare != 0u) {
        differs.Store(out_at + group.x * 4u, any(all != hashes.Load4(piece * 16u)) ? 1u : 0u);
        if (compare == 2u) hashes.Store4(piece * 16u, all);
    } else {
        hashes.Store4(piece * 16u, all);
    }
}
"#;

/// Whether the passes over copies in the card's memory hash with a group
/// per piece ([`CARD_COMPARE_WIDE`]), and those before a submission's lists
/// come after all its copies, behind one barrier (`VICTORY_CARD_HASH_WIDE=0`:
/// a thread per piece, copies and passes in turn, as before).
fn hash_wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_WIDE").is_none_or(|v| v != "0"))
}

/// The groups that hash `pieces` pieces.
fn hash_groups(pieces: u32) -> u32 {
    if hash_wide() { pieces } else { pieces.div_ceil(64) }
}

/// Whether the pieces a flight's compare flags come down to the PC inside
/// the flight itself (`VICTORY_CARD_FETCH_INFLIGHT=2`; off by default, and
/// only with [`relay`]).
///
/// With the relay the thread that lands flights fetched the flagged pieces
/// with a list of its own on the host's queue, the one the submitting
/// thread keeps filling: the fetch ran behind every flight sent meanwhile.
///
/// With `VICTORY_CARD_FETCH_INFLIGHT=2` the compare's list copies each
/// flagged piece into a slot in the card's own memory ([`CardInflight`]),
/// and the landing thread copies the landed flight's slots down to the PC on
/// a copy queue of its own (`inflight_copy_down`). Mode `1` (direct shader
/// writes to PC memory) stalls ~600 ms/frame on RTX 5070 Ti / Blackwell and
/// is refused with a warning.
fn inflight_fetch() -> bool {
    inflight_mode() != 0
}

fn parse_inflight_mode(relay_on: bool, val: Option<&str>) -> u8 {
    if !relay_on {
        return 0;
    }
    match val.map(str::trim) {
        Some("1") => {
            tracing::warn!(
                "VICTORY_CARD_FETCH_INFLIGHT=1 (direct shader writes to PC memory) stalls ~600 ms/frame on RTX 5070 Ti / \
                 Blackwell and is refused; use VICTORY_CARD_FETCH_INFLIGHT=2 for VRAM-staged slots"
            );
            0
        }
        Some("2") => 2,
        _ => 0,
    }
}

/// [`inflight_fetch`]'s mode: 0 off, 2 slots in the card's memory
/// ([`inflight_on_card`]). Mode 1 is refused.
fn inflight_mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| parse_inflight_mode(relay(), std::env::var("VICTORY_CARD_FETCH_INFLIGHT").ok().as_deref()))
}

/// Whether the slots of the fetch inside flights are in the card's own
/// memory (`VICTORY_CARD_FETCH_INFLIGHT=2`; off by default), the landing
/// thread copying the landed flights' slots down to the PC on a copy queue
/// of its own.
fn inflight_on_card() -> bool {
    inflight_mode() == 2
}

/// The bytes of memory pieces fetched inside flights go to
/// ([`inflight_fetch`]; `VICTORY_CARD_FETCH_INFLIGHT_MB`; 256 with
/// the slots in the card's memory): 4,096 slots ([`inflight_slots`]). A
/// flight whose flagged pieces find no free slot lands the old way.
fn inflight_fetch_bytes() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        let default = if inflight_on_card() { 256 } else { 64 };
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default).max(1)
            << 20
    })
}

/// A slot of [`CARD_FETCH`]: a 64 KiB piece as the compare cuts it, widened
/// to 16 bytes at both ends (one more 16-byte step where the copy does not
/// start at a multiple of 16 in its piece).
pub(crate) const INFLIGHT_SLOT: u64 = (64 << 10) + 16;

/// The slots for `bytes` asked of [`inflight_fetch_bytes`]: one per 64 KiB,
/// rounded down to a power of two (at least 1, at most 2^15: the shader's
/// byte offsets into the slots are 32-bit), so that
/// `ticket % slots` runs on without a jump when the 32-bit ticket wraps
/// (2^32 is a multiple of it; with 1,023 slots it would jump by 4 and two
/// tickets in the air could share a slot). The buffer is `slots` times
/// [`INFLIGHT_SLOT`], a little over `bytes`.
pub(crate) fn inflight_slots(bytes: u64) -> u32 {
    let wanted = (bytes >> 16).clamp(1, 1 << 15);
    1u32 << (63 - wanted.leading_zeros())
}

/// [`CARD_FETCH`]'s table entry for a piece the compare did not flag.
pub(crate) const INFLIGHT_NONE: u32 = 0xffff_ffff;

/// Where piece `i` of `span` starts in its [`CARD_FETCH`] slot, as an
/// offset of the copy: the compare's lower bound of the piece, widened down
/// to 16 bytes (`CARD_COMPARE`, before clamping to the declared range).
/// What `Shadow::card_piece` gives for the piece lies at or above it, and
/// within [`INFLIGHT_SLOT`] of it.
pub(crate) fn inflight_piece_from(span: &crate::shadow::CardSpan, i: u32) -> usize {
    let lo = (span.first + i) as usize * 65536;
    let lo = lo.saturating_sub(span.skew as usize);
    lo & !15
}

/// Whether the ticket `k` of [`CARD_FETCH`]'s counter gets a slot, with
/// every ticket below `floor` landed and `slots` slots: the slot is
/// `k % slots`. The tickets of flights still in the air are at or above
/// `floor` (they are taken in the queue's order and flights land in that
/// order), so two that get a slot are fewer than `slots` apart and never
/// share one. The arithmetic of the shader, for the tests.
pub(crate) fn inflight_gets_slot(k: u32, floor: u32, slots: u32) -> bool {
    k.wrapping_sub(floor) < slots
}

/// A piece [`Host::relay_card_inflight`] merges: its copy, offsets, its
/// place in the slots' buffer, and its "fetched after".
type InflightPiece = (std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64);

/// With [`inflight_on_card`]: the copies that bring pieces (their place in
/// the slots' buffer, length) down one after the other: (from, to,
/// length), a piece that starts where the one before ended in the slots
/// joining its copy.
pub(crate) fn slots_down(pieces: impl IntoIterator<Item = (u64, u64)>) -> Vec<(u64, u64, u64)> {
    let mut copies: Vec<(u64, u64, u64)> = Vec::new();
    let mut to = 0u64;
    for (from, len) in pieces {
        if len == 0 {
            continue;
        }
        match copies.last_mut() {
            Some(last) if last.0 + last.2 == from && last.1 + last.2 == to => last.2 += len,
            _ => copies.push((from, to, len)),
        }
        to += len;
    }
    copies
}

/// Records [`slots_down`]'s `copies` from the slots' buffer `ring` into
/// `down`, on an open list (the landing thread's copy queue's, or a
/// test's).
pub(crate) fn record_slots_down(list: &ID3D12GraphicsCommandList, down: &ID3D12Resource, ring: &ID3D12Resource, copies: &[(u64, u64, u64)]) {
    for &(from, to, len) in copies {
        // SAFETY: an open list; live buffers; the caller sized `down`.
        unsafe { list.CopyBufferRegion(down, to, ring, from, len) };
    }
}

/// The fetch inside a flight ([`inflight_fetch`]), run in the compare's list
/// right behind the compare of a declared range (a UAV barrier between):
/// one group per piece (looping where a range has more than 65,535); where
/// `differs` flags the piece, its first thread takes a ticket of the
/// counter (`table`'s dword 0) and, if [`inflight_gets_slot`], the group
/// copies the piece, cut and widened as the compare hashes it, into slot
/// `ticket % slots` of `fetched`. Every piece's entry in `table` (dword 1 +
/// its dword in `differs`) is written: the ticket, also one that got no
/// slot (the landing repeats the test with the plan's `landed`), or
/// [`INFLIGHT_NONE`].
pub(crate) const CARD_FETCH: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
#define SLOT 65552u
#define NONE 0xffffffffu
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint landed;   // every ticket below it is landed
    uint slots;    // slots of `fetched`
    uint groups;   // groups dispatched
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer table : register(u1);
RWByteAddressBuffer fetched : register(u2);
groupshared uint ticket;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    [loop]
    for (uint i = group.x; i < pieces; i += groups) {
        if (t == 0u) {
            uint entry = NONE;
            if (differs.Load(out_at + i * 4u) != 0u) {
                uint k;
                table.InterlockedAdd(0, 1u, k);
                entry = k;
            }
            table.Store(4u + out_at + i * 4u, entry);
            ticket = entry;
        }
        GroupMemoryBarrierWithGroupSync();
        uint k = ticket;
        GroupMemoryBarrierWithGroupSync();
        if (k != NONE && (k - landed) < slots) {
            uint piece = first + i;
            uint lo = piece * 65536u;
            lo = lo > skew ? lo - skew : 0u;
            uint hi = min((piece + 1u) * 65536u - skew, size);
            lo = lo & ~15u;
            hi = (hi + 15u) & ~15u;
            uint into = (k % slots) * SLOT;
            [loop]
            for (uint at = lo + t * 16u; at < hi; at += 1024u) {
                fetched.Store4(into + (at - lo), now.Load4(at));
            }
        }
    }
}
"#;
/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists, and no timing so far names them: each walks its 64 KiB pieces
/// 16 bytes at a time, 4,096 steps one after the other, so a pass takes
/// the time of one piece however few its pieces are. Run 102 sent up 514
/// ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
///
/// `VICTORY_CARD_REHASH_RUNS=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log: one run then measures both on the
/// same scene (the card meter's "send-and-hash lists" in the frame log).
/// Runs 105-110 found no change in the frame, but the card was not the
/// frame's limit then.
fn rehash_runs() -> bool {
    /// 0 off, 1 on (default), 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_CARD_REHASH_RUNS").as_deref() {
        Ok("0") => 0,
        Ok("alt") => 2,
        _ => 1,
    }) {
        0 => false,
        1 => true,
        _ => {
            let joined = (now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(joined), Ordering::Relaxed) != u8::from(joined) {
                tracing::info!(
                    "card rehash passes from now (VICTORY_CARD_REHASH_RUNS=alt): {}",
                    if joined { "JOINED, one pass a run of touching pieces" } else { "one pass a range sent up, as usual" }
                );
            }
            joined
        }
    }
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (on by default;
/// `VICTORY_WIDE_MERGE=0` turns it off). The pieces the card's compare
/// found differing come down in one fetch and are coalesced per 64 KiB
/// piece into strictly disjoint ranges, so worker threads merge them in
/// parallel without overlap.
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_none_or(|v| v != "0"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`relay`], the number of this plan's send of changed ranges
    /// (`QUEUE_ORDER`); else 0.
    sent_as: u64,
    /// Whether the compare's list fetches the flagged pieces itself
    /// ([`inflight_fetch`]): every compared range has its [`CARD_FETCH`].
    inflight: bool,
    /// The `CardInflight::landed` its [`CARD_FETCH`] was told: which of its
    /// tickets got a slot ([`inflight_gets_slot`]).
    inflight_landed: u32,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture {
        staging: std::sync::Arc<crate::texture::Staging>,
        upload_hold: Option<crate::texture::UploadHold>,
    },
}

/// Fast non-cryptographic hasher for addresses, pointers, and integer keys in
/// per-list and per-submission maps (`Deferred`, `Batch`, `fill_ahead`).
#[derive(Default)]
pub(crate) struct AddressHash(u64);

impl std::hash::Hasher for AddressHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_usize(usize::from(b));
        }
    }
    fn write_usize(&mut self, n: usize) {
        let h = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }
    fn write_u64(&mut self, n: u64) {
        self.write_usize(n as usize);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_usize(n as usize);
    }
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// The pipelines [`HostList::bind_pipeline`] kept, by address.
    bound: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Ranges declared written by shaders since the list last wrote title
    /// memory (host copy, address, bytes): declaring them again would
    /// change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Scratch memory [`HostList::fill_scratch`] fills since the list last
    /// wrote title memory (source, bytes): its host address.
    filled: std::collections::HashMap<(usize, usize), u64, std::hash::BuildHasherDefault<AddressHash>>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Last GPU source copied into each aliased host texture in this list.
    texture_aliases: std::collections::HashMap<usize, usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// Title textures of the kinds copied back ([`crate::texback`]) that
    /// the list's shaders write, copied when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Those copies, whose texels go to title memory once the list has run.
    texture_backs: Vec<crate::texback::Back>,
    /// Whether the indirect-work budget is zeroed for this recording (a
    /// step on the CPU, or in the list for the card's copy).
    budget_zeroed: bool,
    /// The indirect scratch buffer (`HostList::indirect_args`, by its
    /// interface pointer) that a clamp in this recording left readable as
    /// indirect arguments; 0 if none. It stays readable until the next
    /// clamp into it ([`HostList::indirect_state`]).
    indirect_readable: usize,
    /// Table constant buffer ranges (`address`, `len`) bound directly in a
    /// host buffer copy rather than filled into upload memory.
    on_card: Vec<(usize, usize)>,
}

impl Deferred {
    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }

    /// The steps recorded last write title memory: what was brought up to
    /// date before them is not any more.
    fn title_written(&mut self) {
        self.refreshed.clear();
        self.gpu_written.clear();
        self.filled.clear();
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether a queue's submissions go to the card without a wait for each
/// (on by default at stage 3; `VICTORY_LATE_WAIT=0` waits for every
/// submission as before, and the timing run waits for every list on its
/// own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, 2 ([`late_exact`]) or 3
/// ([`late_free_present`]); 3 by default.
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("0") => 0,
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 3,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (on by default at its second stage, the one the
/// fast path measures; `VICTORY_RELAY=0` turns it off, `VICTORY_RELAY=1`
/// keeps the first stage's waits; it is off with [`late`] off, in the
/// timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (the default; `VICTORY_RELAY=1` keeps the
/// first): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Since the last frame-rate line, with [`inflight_fetch`]: pieces merged
/// from the slots their flights fetched them to, their bytes, landings that
/// went the old way because a plan's pieces overflowed the slots, landings
/// that fetched the old way (for any reason, overflow too), and with
/// [`inflight_on_card`]: copy-downs, bytes, microseconds, and copies.
static INFLIGHT_STATS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// After a fetch the old way taken with the queue's count at `after`, for
/// plans whose highest send is `landed`: up to which send the notes of
/// sends to card copies may go (`crate::shadow::relay_fetched`).
///
/// Without [`inflight_fetch`] every later merge is of a later fetch, whose
/// count is `after` or more: notes up to `after` are of no use, as before.
/// With it, a flight still in the air, sent before the fetch, merges its
/// own slot with "fetched after" its own send, which may be below `after`:
/// it needs the notes of the sends between, or it takes the title's bytes
/// that went up in them for its shaders' writes and puts its older card
/// bytes over them. (Flights 5, 6, 7 in the air; 5 lands the old way with
/// the count at 7; a title byte went up in send 7 into a piece 6's shaders
/// write; 6 lands from its slot, which holds that byte from before send 7.)
/// So the mark goes only to the landed plans' highest send: every flight
/// still to land was sent after it, and every later fetch the old way
/// takes a count at least that high. (`CardInflight::brought` is another
/// matter: the card's bytes the fetch put in the CPU's side, not the
/// title's.)
pub(crate) fn fallback_fetched_mark(inflight: bool, after: u64, landed: Option<u64>) -> u64 {
    match (inflight, landed) {
        (false, _) => after,
        (true, Some(landed)) => landed.min(after),
        (true, None) => 0,
    }
}

/// The landed mark of [`CARD_FETCH`]'s tickets after ticket `k` landed: one
/// past `k` if that is ahead of `landed`, in wrapping order (tickets in the
/// air are always within 2^31 of the mark), else `landed`.
pub(crate) fn inflight_raise(landed: u32, k: u32) -> u32 {
    let next = k.wrapping_add(1);
    if (next.wrapping_sub(landed) as i32) > 0 { next } else { landed }
}

/// With [`inflight_fetch`], after a landing merged (either way): the tickets
/// these plans' fetches took are landed, and their slots may be written
/// over. Plans land in the line's order and tickets are taken in the
/// queue's (a UAV barrier opens every compare's list), so every ticket of a
/// plan in the air is ahead of these.
///
/// The counter is never reset and wraps after 2^32 tickets: the mark is
/// raised in wrapping order ([`inflight_raise`]), and the slots are a power
/// of two, so `k % slots` runs on across the wrap. shortcut: ticket
/// 0xffff_ffff reads as [`INFLIGHT_NONE`]; a flagged piece that draws it
/// copies nothing and its landing goes the old way, once every 2^32.
fn inflight_tickets_landed(plans: &[&CardPlan]) {
    let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(work) = work.as_mut() else {
        return;
    };
    let dwords = (work.differs.2 / 4) as usize;
    let Some(inflight) = work.inflight.as_mut() else {
        return;
    };
    // SAFETY: the mapping of the tickets; these plans' fetches have run.
    let table = unsafe { std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords) };
    for plan in plans.iter().filter(|p| p.inflight) {
        for r in plan.ranges.iter().filter(|r| r.compared) {
            for p in 0..r.span.pieces {
                if let Some(&k) = table.get((r.at + p) as usize)
                    && k != INFLIGHT_NONE
                {
                    inflight.landed = inflight_raise(inflight.landed, k);
                }
            }
        }
    }
    // Every flight still to land was sent after these: what fetches the old
    // way brought up to them concerns none of them.
    if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
        inflight.brought.retain(|b| b.3 > last);
    }
}

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range: all of them
/// flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> std::collections::HashSet<(usize, u32)> {
    flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect()
}

/// Whether a landing tells the title about flights with nothing to copy
/// back before the copy-back of the later flights it lands with
/// (`VICTORY_EARLY_SIGNAL=1`; off by default; only with [`relay`]).
///
/// With memory in place (or flights with no shader writes), a flight has
/// nothing for the relay to fetch, compare or merge ([`Flight::covered`]).
/// But the relay lands every flight the card has run in one go, the
/// copy-back of all of them first: a covered flight's fence signals, events
/// and bottom-of-pipe writes waited for the copy-back of the flights after
/// it. With the switch the flights ahead of the first one that still has
/// copies to bring back, and what follows each, are told at once; the rest
/// as before. The line's order is kept: nothing behind a flight with a
/// copy-back is told before it.
fn early_signal() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_EARLY_SIGNAL").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: flights told before the copy-back of
/// later ones ([`early_signal`]); title fence signals set by the relay,
/// their microseconds from when it took their flights, and the longest.
static EARLY: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// With [`early_signal`]: of the things a landing takes, in the line's
/// order (a flight: whether it has nothing to copy back; a write or a
/// signal behind one: `None`), how many from the front are told before
/// the copy-back: everything ahead of the first flight that has one.
fn told_ahead(line: &[Option<bool>]) -> usize {
    line.iter().position(|f| *f == Some(false)).unwrap_or(line.len())
}

impl Flight {
    /// Whether nothing of it is left to copy back once the card has run
    /// it: no ranges of copies in the PC's memory, no pieces of copies in
    /// the card's memory to fetch.
    fn covered(&self) -> bool {
        self.back.is_empty() && self.card.as_ref().is_none_or(|c| c.ranges.is_empty())
    }
}

/// Whether pieces fetched on a copy queue beside later flights are fetched
/// again when those flights land (`VICTORY_QUEUE2_REFETCH=0`: not, as
/// before; on by default with `VICTORY_QUEUE2` bit 1).
fn refetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE2_REFETCH").is_none_or(|v| v != "0"))
}

/// With fetches on a copy queue (`crate::queue2`, bit 1): the pieces (copy,
/// piece number) a fetch brought down while later submissions may have been
/// running, each with the number of the last submission started when the
/// fetch had run ([`STARTED`]).
///
/// Such a fetch stands behind the flights being landed and nothing else,
/// so it runs beside whatever was sent after them. A later flight that
/// writes the piece (a buffer cleared and filled again) may be half way
/// through it: the bytes that come down are neither the earlier flight's
/// nor the later one's, and they go into title memory. If the later flight
/// leaves the piece as the earlier one left it (a still scene does, every
/// frame), its own compare finds the stored hash unchanged and flags
/// nothing: title memory kept the half-written bytes until the piece next
/// changed. On the direct queue a fetch only ever sees whole flights.
///
/// So a flight that declared such a piece and had started before the fetch
/// had run fetches it at its own landing whatever its compare says
/// ([`fetched_beside`]); that fetch stands behind the flight, and is noted
/// here in turn if still later flights were in the air. Between the two
/// landings title memory may hold the half-written bytes, as it does on
/// the console, where the title reads the memory the card is writing: what
/// the title is promised is the piece as the flight left it once that
/// flight's fence is signalled, and the landing comes before the signal.
static BESIDE: Mutex<Option<std::collections::HashMap<(usize, u32), u64>>> = Mutex::new(None);

/// Since the last frame-rate line: pieces fetched because an earlier fetch
/// of them ran beside their flight ([`BESIDE`]), and pieces noted there.
static BESIDE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`BESIDE`]: whether the flight numbered `number` (among the submissions
/// started) must fetch a piece that an earlier landing's fetch brought
/// down when submissions up to `started` had been started: it had started
/// by then, so it may have been writing the piece under that fetch.
fn fetched_beside(started: Option<u64>, number: u64) -> bool {
    started.is_some_and(|started| started >= number)
}

/// [`BESIDE`], after a landing of flights up to number `landed` whose
/// fetch brought `fetched` down while submissions up to `started` had
/// been started: what is noted of the pieces afterwards, and how many were
/// noted now. A note is done with once every submission it names has
/// landed (each fetched the piece again if it declared it); the pieces
/// just fetched are noted if anything later than the flights landed had
/// started.
fn note_beside(noted: &mut std::collections::HashMap<(usize, u32), u64>, fetched: &[(usize, u32)], landed: u64, started: u64) -> u64 {
    noted.retain(|_, upto| *upto > landed);
    if started <= landed {
        return 0;
    }
    for piece in fetched {
        noted.insert(*piece, started);
    }
    fetched.len() as u64
}

#[cfg(test)]
mod relay_tests {
    use super::{fetched_beside, first_landing, flagged_pieces, landable, note_beside, parse_inflight_mode, told_ahead};

    /// A piece fetched on the copy queue while a later flight was in the
    /// air is fetched again by that flight, and by no flight sent after
    /// the fetch had run; the note goes when its flights have landed.
    #[test]
    fn a_piece_fetched_beside_a_later_flight_is_fetched_again_by_it() {
        let mut noted = std::collections::HashMap::new();
        let (a, p) = (0x1000usize, 7u32);
        // Flight 5 lands; its fetch ran while 6 and 7 had started.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 5, 7), 1);
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 6));
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 7));
        // Flight 8 was sent after the fetch had run: its compare decides.
        assert!(!fetched_beside(noted.get(&(a, p)).copied(), 8));
        // Another piece was never noted.
        assert!(!fetched_beside(noted.get(&(a, p + 1)).copied(), 6));
        // Flight 6 lands and fetches it again, with 7 still in the air and
        // nothing newer: the note stays for 7.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 6, 7), 1);
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 7));
        // Flight 7 lands with nothing behind it: nothing is noted any more.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 7, 7), 0);
        assert!(noted.is_empty());
        // A fetch with nothing later in the air notes nothing.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 9, 9), 0);
        assert!(noted.is_empty());
    }

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // One flight a landing: the first flight with what follows it up
        // to the next one (and what comes ahead of it).
        assert_eq!(first_landing(&[Some(3), None, Some(4), None]), 2);
        assert_eq!(first_landing(&[None, Some(3), Some(4)]), 2);
        assert_eq!(first_landing(&[Some(3), None, None]), 3);
        assert_eq!(first_landing(&[None, None]), 2);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// With early signals: a flight with nothing to copy back is told as
    /// soon as the card has run it, with what follows it; a flight that
    /// still has a copy-back, and everything behind it (a covered flight
    /// too), waits for that copy-back, so the line's order is kept.
    #[test]
    fn covered_flights_are_told_before_the_copy_back_of_later_ones() {
        // Covered flight (fence 5), its write, a signal; a flight with a
        // copy-back (fence 6), its signal; a covered flight (fence 7).
        let fences = [Some(5), None, None, Some(6), None, Some(7)];
        let covered = [Some(true), None, None, Some(false), None, Some(true)];
        // The card has run the first only: it and what follows it go.
        let n = landable(&fences, 5);
        assert_eq!(n, 3);
        assert_eq!(told_ahead(&covered[..n]), 3);
        // It has run all three in one landing: the first at once, the
        // second after its copy-back, and the third behind it.
        let n = landable(&fences, 7);
        assert_eq!(n, 6);
        assert_eq!(told_ahead(&covered[..n]), 3);
        // A landing that begins with a copy-back tells nothing ahead.
        assert_eq!(told_ahead(&covered[3..]), 0);
        // All covered: all at once; signals alone too.
        assert_eq!(told_ahead(&[Some(true), None, Some(true), None]), 4);
        assert_eq!(told_ahead(&[None, None]), 2);
        assert_eq!(told_ahead(&[]), 0);
    }

    #[test]
    fn inflight_mode_refuses_mode_1_and_accepts_mode_2() {
        assert_eq!(parse_inflight_mode(false, Some("2")), 0);
        assert_eq!(parse_inflight_mode(true, None), 0);
        assert_eq!(parse_inflight_mode(true, Some("0")), 0);
        assert_eq!(parse_inflight_mode(true, Some("1")), 0);
        assert_eq!(parse_inflight_mode(true, Some("2")), 2);
        assert_eq!(parse_inflight_mode(true, Some(" 2 ")), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// Title-memory ranges written when its shader textures are copied back.
    texture_back_ranges: Vec<(usize, usize)>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    texture_back_ranges: Vec<(usize, usize)>,
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders may write (address, end), in
    /// order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Whether title memory `[from, to)` lies in what flights in the air may
/// write (with [`late`]).
pub(crate) fn flights_write(from: usize, to: usize) -> bool {
    touches(&line().written, from, to)
}

/// Whether a constant buffer view of a table whose title memory flights may
/// write is filled on the card, behind them ([`HostList::card_constants`]),
/// instead of in upload memory at submission, which waits for them to land
/// (`VICTORY_FILL_ON_CARD=0` fills it as before).
pub(crate) fn fill_on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_ON_CARD").is_none_or(|v| v != "0"))
}

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let written = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        line.written.clone()
    };
    blocked_by(&written, runs, true)
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    let mut pending: Vec<(usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if touches(written, src, src + len) || touches(written, dst, dst + len) {
                        // A copy that only READS where flights may write is
                        // not waited for either, and checked like a fill
                        // (`fill_ahead`): run 210 looked at 197 such waits
                        // (20,480 bytes out of the 174 MB buffer into a small
                        // buffer of its own, four a frame, 43 to 60 ms a
                        // frame) and neither side had other bytes after
                        // one. A copy INTO such a range still waits.
                        if !touches(written, dst, dst + len) && fill_ahead::can_read_ahead(src, len) {
                            if count && len > 0 {
                                pending.push((src, len));
                            }
                            continue;
                        }
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            fill_check::blocked_copy(written, dst, src, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(_, src, len) => {
                    if touches(written, src, src + len) {
                        // Not waited for, and checked when the flights land
                        // (`fill_ahead`).
                        if fill_ahead::can_read_ahead(src, len) {
                            if count && len > 0 {
                                pending.push((src, len));
                            }
                            continue;
                        }
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            static NAMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                            if NAMED.fetch_add(1, Ordering::Relaxed) < 12 {
                                let copy = crate::shadow::find(src as u64, len as u64).map(|s| s.span());
                                tracing::info!(
                                    src = format_args!("{src:#x}"),
                                    len,
                                    copy = ?copy.map(|(a, l)| format!("{a:#x}+{l:#x}")),
                                    "a fill of upload memory reads what a flight writes"
                                );
                            }
                            #[cfg(feature = "diagnostics")]
                            fill_check::blocked(written, src, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture { staging, .. } => {
                    let (at, len) = staging.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => {}
            }
        }
    }
    if !pending.is_empty() {
        fill_ahead::read_batch(&pending);
    }
    None
}

/// Fills of upload memory (a constant buffer a table names, copied into
/// the list's own memory when the list is submitted) that read title memory
/// flights "may write" are no longer waited for; what the flights then
/// write is checked against them, and the first fill that read bytes a
/// flight changed afterwards turns the waiting back on for good.
///
/// Why: a flight declares every range its shaders can store to, and the
/// title has an unordered view over the whole of its 174 MB buffer, whose
/// copy is in the card's memory. Its per-draw constants lie in the same
/// buffer, at new addresses every frame. So the first such fill of a frame
/// waited until those flights had landed: one wait a frame, 56 to 95 ms of
/// the submitting thread standing still (runs 202 and 206; 145 ms while
/// the world loaded in 208), the largest single wait of the frame. Run 208
/// looked at every one of them: 44 fills of 480 bytes, and after the wait
/// the bytes were the same 44 times.
///
/// What it risks, and how it is bounded: a fill that reads bytes a flight
/// in the air is about to change takes the old bytes, and the lists of
/// that one submission draw with them. `wrote` sees it when the flight
/// lands (the bytes going into title memory are compared with the ones
/// there, over every fill that read ahead of them) and from then on fills
/// wait as before. `VICTORY_FILL_WAIT=1`: they wait from the start.
mod fill_ahead {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock, RwLock};

    /// Fills are looked up and blacklisted by the 64 KiB piece of title memory
    /// they read.
    const PIECE: usize = 1 << 16;
    /// Shift from a 64 KiB piece index (`addr >> 16`) to its 1 MiB cell index
    /// (`addr >> 20`).
    const CELL_SHIFT_FROM_PIECE: usize = 4;
    /// A 1 MiB cell holds 16 pieces of 64 KiB; if more than this many distinct
    /// pieces in one 1 MiB cell trip, the entire 1 MiB cell is blacklisted.
    const CELL_ESCALATE_PIECES: usize = 4;
    /// Global safety valve: if more than this many distinct 64 KiB pieces trip
    /// across the whole address space, flip global `WAIT = true`.
    const GLOBAL_TRIP_PIECES: usize = 64;
    /// Cap per-region blacklist warning messages so logs stay clean.
    const WARN_CAP: u64 = 16;
    /// A fill is checked against flights that land this many frames after
    /// it at most (flights are settled at 256 lists in the air, and by
    /// every present that waits).
    const KEEP: u64 = 16;
    /// Cap distinct tracked `(from, to)` fills per 64 KiB piece so `wrote()`
    /// stays fast even under heavy ring-buffer churn (256 covers every
    /// 256-byte-aligned CBV slot in a 64 KiB page).
    const MAX_FILLS_PER_PIECE: usize = 256;

    type ReadMap = HashMap<usize, Piece, std::hash::BuildHasherDefault<super::AddressHash>>;

    /// Global kill switch: a flight changed bytes in legacy mode, or more than
    /// [`GLOBAL_TRIP_PIECES`] distinct 64 KiB pieces tripped in scoped mode.
    static WAIT: AtomicBool = AtomicBool::new(false);
    /// Fast-path gate: `false` while no piece has been blacklisted in scoped
    /// mode, so `can_read_ahead` pays zero lock cost in the common empty case.
    static HAS_WAIT_PIECES: AtomicBool = AtomicBool::new(false);
    /// Blacklisted 64 KiB piece indices (`addr / PIECE`) in scoped mode.
    static WAIT_PIECES: OnceLock<RwLock<HashSet<usize>>> = OnceLock::new();
    /// Blacklisted 1 MiB cell indices (`addr >> 20`) escalated when more than
    /// [`CELL_ESCALATE_PIECES`] distinct pieces in one cell trip.
    static WAIT_CELLS: OnceLock<RwLock<HashSet<usize>>> = OnceLock::new();
    /// Number of scoped blacklist warning messages emitted so far (capped at
    /// [`WARN_CAP`]).
    static WARNED: AtomicU64 = AtomicU64::new(0);
    /// Since the last frame line: fills that read ahead; landings' writes
    /// that touched one's piece. Since the start: fills that read bytes a
    /// flight then changed.
    static COUNT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    #[derive(Default)]
    struct Piece {
        /// The frame its old fills were last dropped in.
        pruned: u64,
        /// Sorted by `(from, to)`: from, to, frame, and the number of the last
        /// submission started when it read: the flights in the air then have
        /// that number or a lower one, and only their writes can be ones it
        /// missed.
        fills: Vec<(usize, usize, u64, u64)>,
    }

    impl Piece {
        /// Records `(from, to)` at `(frame, started)`, deduplicating identical
        /// `(from, to)` ranges in place (updating `frame` and `started` to the
        /// latest) and capping per-piece entries at [`MAX_FILLS_PER_PIECE`] by
        /// evicting the oldest entry.
        fn record(&mut self, from: usize, to: usize, frame: u64, started: u64) {
            if self.pruned != frame {
                self.pruned = frame;
                self.fills.retain(|f| f.2 + KEEP >= frame);
            }
            match self.fills.binary_search_by_key(&(from, to), |f| (f.0, f.1)) {
                Ok(idx) => {
                    let entry = &mut self.fills[idx];
                    entry.2 = entry.2.max(frame);
                    entry.3 = entry.3.max(started);
                }
                Err(idx) => {
                    if self.fills.len() < MAX_FILLS_PER_PIECE {
                        self.fills.insert(idx, (from, to, frame, started));
                    } else if let Some((oldest_idx, oldest)) = self
                        .fills
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, f)| (f.2, f.3))
                        && (frame, started) >= (oldest.2, oldest.3)
                    {
                        self.fills.remove(oldest_idx);
                        let insert_at = if oldest_idx < idx { idx - 1 } else { idx };
                        self.fills.insert(insert_at, (from, to, frame, started));
                    }
                }
            }
        }
    }

    /// The lowest submission number among the flights the landing thread
    /// is landing now; 0: not known (a landing by another way), and then
    /// every write counts. (Run 211: a read of frame 1381 was taken for
    /// missed because shaders wrote the same 80 bytes in frame 1397, in a
    /// flight sent sixteen frames after it.)
    static LANDING: AtomicU64 = AtomicU64::new(0);

    pub(super) fn landing(first: u64) {
        LANDING.store(first, Ordering::Relaxed);
    }

    static READ: Mutex<Option<ReadMap>> = Mutex::new(None);

    /// Records a batch of `(src, len)` fills into `read_map` under a single
    /// mutex acquisition, deduplicating locally first so duplicate entries
    /// within one submission never touch the map twice.
    fn record_batch(read_map: &Mutex<Option<ReadMap>>, reads: &[(usize, usize)], frame: u64, started: u64) {
        let mut unique: Vec<(usize, usize)> = reads.iter().copied().filter(|&(_, len)| len > 0).collect();
        if unique.is_empty() {
            return;
        }
        if unique.len() > 1 {
            unique.sort_unstable();
            unique.dedup();
        }
        let mut read = read_map.lock().unwrap_or_else(|e| e.into_inner());
        let map = read.get_or_insert_with(ReadMap::default);
        for (src, len) in unique {
            let end = src.saturating_add(len);
            if end <= src {
                continue;
            }
            for piece in src / PIECE..=(end - 1) / PIECE {
                map.entry(piece).or_default().record(src, end, frame, started);
            }
        }
    }

    fn wait_pieces() -> &'static RwLock<HashSet<usize>> {
        WAIT_PIECES.get_or_init(|| RwLock::new(HashSet::new()))
    }

    fn wait_cells() -> &'static RwLock<HashSet<usize>> {
        WAIT_CELLS.get_or_init(|| RwLock::new(HashSet::new()))
    }

    fn parse_fill_wait(val: Option<&str>) -> bool {
        val.is_some_and(|v| v.trim() == "1")
    }

    fn parse_scoped(val: Option<&str>) -> bool {
        val.is_none_or(|v| v.trim() != "0")
    }

    /// Whether `VICTORY_FILL_WAIT=1` forces all fills and read-copies to wait.
    fn global_wait_override() -> bool {
        static FORCE_WAIT: OnceLock<bool> = OnceLock::new();
        *FORCE_WAIT.get_or_init(|| parse_fill_wait(std::env::var("VICTORY_FILL_WAIT").ok().as_deref()))
    }

    /// Whether mismatches blacklist only the affected 64 KiB pieces / 1 MiB
    /// cells (`VICTORY_FILL_AHEAD_SCOPED`, default ON; `=0` restores the
    /// legacy global kill switch).
    fn scoped() -> bool {
        static SCOPED: OnceLock<bool> = OnceLock::new();
        *SCOPED.get_or_init(|| parse_scoped(std::env::var("VICTORY_FILL_AHEAD_SCOPED").ok().as_deref()))
    }

    #[derive(Debug, Default, PartialEq, Eq)]
    struct TripOutcome {
        new_pieces: usize,
        new_cells: usize,
        total_pieces: usize,
        total_cells: usize,
        tripped_global: bool,
    }

    struct Blacklist<'a> {
        wait: &'a AtomicBool,
        has_pieces: &'a AtomicBool,
        pieces: &'a RwLock<HashSet<usize>>,
        cells: &'a RwLock<HashSet<usize>>,
    }

    fn global_blacklist() -> Blacklist<'static> {
        Blacklist {
            wait: &WAIT,
            has_pieces: &HAS_WAIT_PIECES,
            pieces: wait_pieces(),
            cells: wait_cells(),
        }
    }

    impl Blacklist<'_> {
        fn can_read_ahead(&self, force_wait: bool, src: usize, len: usize) -> bool {
            if force_wait || self.wait.load(Ordering::Relaxed) {
                return false;
            }
            if len == 0 {
                return true;
            }
            if !self.has_pieces.load(Ordering::Acquire) {
                return true;
            }
            let end = src.saturating_add(len);
            let last = end - 1;
            let first_cell = src >> 20;
            let last_cell = last >> 20;
            {
                let cells = self.cells.read().unwrap_or_else(|e| e.into_inner());
                if !cells.is_empty() && (first_cell..=last_cell).any(|cell| cells.contains(&cell)) {
                    return false;
                }
            }
            let first_piece = src / PIECE;
            let last_piece = last / PIECE;
            let pieces = self.pieces.read().unwrap_or_else(|e| e.into_inner());
            !(first_piece..=last_piece).any(|piece| pieces.contains(&piece))
        }

        fn trip(&self, scoped_on: bool, fill: (usize, usize), overlap: (usize, usize)) -> TripOutcome {
            if !scoped_on {
                return TripOutcome {
                    tripped_global: !self.wait.swap(true, Ordering::Relaxed),
                    ..Default::default()
                };
            }
            let mut new_pieces = 0usize;
            let mut escalated: Vec<usize> = Vec::new();
            let total_pieces = {
                let mut pieces = self.pieces.write().unwrap_or_else(|e| e.into_inner());
                let mut touched_cells: Vec<usize> = Vec::new();
                for (lo, hi) in [fill, overlap] {
                    if lo >= hi {
                        continue;
                    }
                    for piece in lo / PIECE..=(hi - 1) / PIECE {
                        if pieces.insert(piece) {
                            new_pieces += 1;
                            let cell = piece >> CELL_SHIFT_FROM_PIECE;
                            if !touched_cells.contains(&cell) {
                                touched_cells.push(cell);
                            }
                        }
                    }
                }
                for cell in touched_cells {
                    let base = cell << CELL_SHIFT_FROM_PIECE;
                    let in_cell = (base..base + (1 << CELL_SHIFT_FROM_PIECE))
                        .filter(|p| pieces.contains(p))
                        .count();
                    if in_cell > CELL_ESCALATE_PIECES {
                        escalated.push(cell);
                    }
                }
                pieces.len()
            };
            let (new_cells, total_cells) = {
                if escalated.is_empty() {
                    let cells = self.cells.read().unwrap_or_else(|e| e.into_inner());
                    (0, cells.len())
                } else {
                    let mut cells = self.cells.write().unwrap_or_else(|e| e.into_inner());
                    let mut added = 0usize;
                    for cell in escalated {
                        if cells.insert(cell) {
                            added += 1;
                        }
                    }
                    (added, cells.len())
                }
            };
            if total_pieces > 0 {
                self.has_pieces.store(true, Ordering::Release);
            }
            let tripped_global = total_pieces > GLOBAL_TRIP_PIECES && !self.wait.swap(true, Ordering::Relaxed);
            TripOutcome {
                new_pieces,
                new_cells,
                total_pieces,
                total_cells,
                tripped_global,
            }
        }
        /// Core mismatch detector shared by [`wrote`] and unit tests.
        #[cfg(test)]
        fn check_wrote(
            &self,
            (force_wait, scoped_on): (bool, bool),
            read_map: &Mutex<Option<ReadMap>>,
            (landing, frame): (u64, u64),
            (at, new): (usize, &[u8]),
            on_mismatch: impl Fn(usize, usize, u64, usize, &TripOutcome),
        ) -> usize {
            self.check_wrote_on_hits((force_wait, scoped_on), read_map, (landing, frame), (at, new), || {}, on_mismatch)
        }

        fn check_wrote_on_hits(
            &self,
            (force_wait, scoped_on): (bool, bool),
            read_map: &Mutex<Option<ReadMap>>,
            (landing, frame): (u64, u64),
            (at, new): (usize, &[u8]),
            on_hits: impl FnOnce(),
            on_mismatch: impl Fn(usize, usize, u64, usize, &TripOutcome),
        ) -> usize {
            if new.is_empty() || force_wait || self.wait.load(Ordering::Relaxed) {
                return 0;
            }
            let mut hits: Vec<(usize, usize, u64)> = {
                let read = read_map.lock().unwrap_or_else(|e| e.into_inner());
                let Some(map) = read.as_ref() else { return 0 };
                let end = at + new.len();
                (at / PIECE..=(end - 1) / PIECE)
                    .filter_map(|piece| map.get(&piece))
                    .flat_map(|p| p.fills.iter().copied())
                    .filter(|f| f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3))
                    .map(|f| (f.0, f.1, f.2))
                    .collect()
            };
            if hits.is_empty() {
                return 0;
            }
            if hits.len() > 1 {
                hits.sort_unstable();
                hits.dedup();
            }
            on_hits();
            let mut mismatches = 0usize;
            let mut now = Vec::new();
            for (from, to, read_in) in hits {
                let (a, b) = (from.max(at), to.min(at + new.len()));
                now.clear();
                now.resize(b - a, 0);
                // Title memory as it is, which is what the fill read unless the
                // title wrote it since; not there: nothing to say.
                if !super::read_into(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
                    continue;
                }
                mismatches += 1;
                let first = now.iter().zip(&new[a - at..b - at]).position(|(x, y)| x != y).unwrap_or(0);
                let outcome = self.trip(scoped_on, (from, to), (a, b));
                on_mismatch(from, to, read_in, a + first, &outcome);
                if !scoped_on || outcome.tripped_global {
                    break;
                }
            }
            mismatches
        }
    }

    /// Whether a fill or read-only title copy of `len` bytes at `src` may read
    /// ahead of flights in the air without waiting for them to land.
    pub(super) fn can_read_ahead(src: usize, len: usize) -> bool {
        global_blacklist().can_read_ahead(global_wait_override(), src, len)
    }

    /// Fills of a submission read `(src, len)` ranges ahead of flights that
    /// may write them: recorded in one mutex acquisition with deduplication
    /// and per-piece capping.
    pub(super) fn read_batch(reads: &[(usize, usize)]) {
        let valid = reads.iter().filter(|&&(_, len)| len > 0).count();
        if valid == 0 {
            return;
        }
        COUNT[0].fetch_add(valid as u64, Ordering::Relaxed);
        let frame = crate::frame::current();
        let started = super::STARTED.load(Ordering::Relaxed);
        record_batch(&READ, reads, frame, started);
    }

    /// A landing is about to put `new` into title memory at `at`: bytes
    /// shaders wrote, merged into the title's. Still checks active fills even
    /// after `HAS_WAIT_PIECES` is true; skipped only once global `WAIT` (or
    /// `VICTORY_FILL_WAIT=1`) is set.
    pub(crate) fn wrote(at: usize, new: &[u8]) {
        let force_wait = global_wait_override();
        if new.is_empty() || force_wait || WAIT.load(Ordering::Relaxed) {
            return;
        }
        let frame = crate::frame::current();
        let landing = LANDING.load(Ordering::Relaxed);
        let scoped_on = scoped();
        let bl = global_blacklist();
        bl.check_wrote_on_hits(
            (force_wait, scoped_on),
            &READ,
            (landing, frame),
            (at, new),
            || {
                COUNT[1].fetch_add(1, Ordering::Relaxed);
            },
            |from, to, read_in, first_diff, outcome| {
            let n = COUNT[2].fetch_add(1, Ordering::Relaxed) + 1;
            if !scoped_on {
                if outcome.tripped_global {
                    tracing::warn!(
                        "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                         them (first at {first_diff:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                         Fills wait for flights again from here on, as with VICTORY_FILL_WAIT=1 ({n} such fills)",
                        to - from,
                    );
                }
            } else if outcome.tripped_global {
                tracing::warn!(
                    "scoped fill_ahead tripped global safety valve ({} distinct 64 KiB pieces > {GLOBAL_TRIP_PIECES}, {} 1 MiB cells) \
                     after fill of {} bytes at {from:#x} (frame {read_in}, first diff at {first_diff:#x}, landed in frame {frame}): all fills \
                     wait for flights from here on ({n} such fills)",
                    outcome.total_pieces,
                    outcome.total_cells,
                    to - from,
                );
            } else if (outcome.new_pieces > 0 || outcome.new_cells > 0) && WARNED.fetch_add(1, Ordering::Relaxed) < WARN_CAP {
                tracing::warn!(
                    "scoped fill_ahead blacklisted +{} 64 KiB piece(s) / +{} 1 MiB cell(s) (total {} pieces, {} cells) after fill of \
                     {} bytes at {from:#x} in frame {read_in} read ahead of shader write (first diff at {first_diff:#x}, landed in frame {frame}; \
                     {n} such fills): only blacklisted regions wait for flights",
                    outcome.new_pieces,
                    outcome.new_cells,
                    outcome.total_pieces,
                    outcome.total_cells,
                    to - from,
                );
            }
        });
    }

    /// With the frame line.
    pub(super) fn line(frames: f64) {
        let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Ordering::Relaxed));
        let (blacklisted_pieces, blacklisted_cells) = if HAS_WAIT_PIECES.load(Ordering::Acquire) {
            (
                wait_pieces().read().unwrap_or_else(|e| e.into_inner()).len(),
                wait_cells().read().unwrap_or_else(|e| e.into_inner()).len(),
            )
        } else {
            (0, 0)
        };
        if ahead == 0 && looked == 0 && blacklisted_pieces == 0 {
            return;
        }
        let kept = {
            let frame = crate::frame::current();
            let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
            read.as_mut().map_or(0, |map| {
                map.retain(|_, p| {
                    p.fills.retain(|f| f.2 + KEEP >= frame);
                    !p.fills.is_empty()
                });
                map.values().map(|p| p.fills.len()).sum::<usize>()
            })
        };
        tracing::info!(
            "fills of upload memory that read title memory flights may write, not waited for (VICTORY_FILL_WAIT=1 waits, \
             VICTORY_FILL_AHEAD_SCOPED={}): {:.1} per frame; {:.1} landings' writes per frame fell in the same 64 KiB as one and \
             were compared with it; fills that read bytes a flight then changed, since the start: {}{}; blacklisted: \
             {blacklisted_pieces} 64 KiB pieces, {blacklisted_cells} 1 MiB cells; {kept} fills kept to check",
            u8::from(scoped()),
            ahead as f64 / frames,
            looked as f64 / frames,
            COUNT[2].load(Ordering::Relaxed),
            if WAIT.load(Ordering::Relaxed) { " (fills wait again)" } else { "" }
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        struct Fixture {
            wait: AtomicBool,
            has_pieces: AtomicBool,
            pieces: RwLock<HashSet<usize>>,
            cells: RwLock<HashSet<usize>>,
            read_map: Mutex<Option<ReadMap>>,
        }

        impl Fixture {
            fn new() -> Self {
                Self {
                    wait: AtomicBool::new(false),
                    has_pieces: AtomicBool::new(false),
                    pieces: RwLock::new(HashSet::new()),
                    cells: RwLock::new(HashSet::new()),
                    read_map: Mutex::new(None),
                }
            }

            fn blacklist(&self) -> Blacklist<'_> {
                Blacklist {
                    wait: &self.wait,
                    has_pieces: &self.has_pieces,
                    pieces: &self.pieces,
                    cells: &self.cells,
                }
            }

            fn can_read(&self, force_wait: bool, src: usize, len: usize) -> bool {
                self.blacklist().can_read_ahead(force_wait, src, len)
            }

            fn trip(&self, scoped_on: bool, from: usize, to: usize, a: usize, b: usize) -> TripOutcome {
                self.blacklist().trip(scoped_on, (from, to), (a, b))
            }

            fn record_read(&self, src: usize, len: usize, frame: u64, started: u64) {
                record_batch(&self.read_map, &[(src, len)], frame, started);
            }
        }

        #[test]
        fn scoped_isolation_blocks_tripped_piece_only() {
            let f = Fixture::new();
            let piece_a = 0x0000_dbe0_0000_usize;
            let piece_b = 0x0000_3b81_0000_usize;
            assert!(f.can_read(false, piece_a + 0xdf60, 2048));
            assert!(f.can_read(false, piece_b, 20_480));
            assert!(!f.has_pieces.load(Ordering::Relaxed));

            let out = f.trip(true, piece_a + 0xdf60, piece_a + 0xdf60 + 2048, piece_a + 0xdf60, piece_a + 0xdf60 + 2048);
            assert_eq!(out.new_pieces, 1);
            assert_eq!(out.new_cells, 0);
            assert_eq!(out.total_pieces, 1);
            assert!(!out.tripped_global);
            assert!(f.has_pieces.load(Ordering::Relaxed));
            assert!(!f.wait.load(Ordering::Relaxed));

            // Tripped 64 KiB piece waits; unrelated piece and sibling piece in the same 1 MiB cell still read ahead.
            assert!(!f.can_read(false, piece_a, 64));
            assert!(!f.can_read(false, piece_a + 0xdf60, 2048));
            assert!(f.can_read(false, piece_a + PIECE, 2048));
            assert!(f.can_read(false, piece_b, 20_480));
            // Zero-length read is always allowed when not globally waiting.
            assert!(f.can_read(false, piece_a, 0));
        }

        #[test]
        fn cross_piece_range_touching_tripped_piece_returns_false() {
            let f = Fixture::new();
            let base = 0x1000_0000usize;
            let piece_a = base + 3 * PIECE;
            f.trip(true, piece_a + 128, piece_a + 256, piece_a + 128, piece_a + 256);

            // Starts in piece A-1, extends into piece A.
            assert!(!f.can_read(false, piece_a - 16, 32));
            // Starts in piece A, extends into piece A+1.
            assert!(!f.can_read(false, piece_a + PIECE - 16, 32));
            // Spans across piece A from A-1 to A+2.
            assert!(!f.can_read(false, piece_a - PIECE, 3 * PIECE));
            // Adjacent ranges right before and right after piece A remain free.
            assert!(f.can_read(false, piece_a - 64, 64));
            assert!(f.can_read(false, piece_a + PIECE, 64));
        }

        #[test]
        fn cell_escalation_after_four_distinct_pieces_in_same_mib() {
            let f = Fixture::new();
            let cell0 = 0x2000_0000usize; // 1 MiB aligned
            let cell1 = cell0 + (1 << 20);
            // Trip 4 distinct 64 KiB pieces in cell0: should NOT escalate yet.
            for p in 0..4 {
                let addr = cell0 + p * PIECE;
                let out = f.trip(true, addr, addr + 256, addr, addr + 256);
                assert_eq!(out.new_cells, 0, "piece {p} must not escalate cell0");
                assert_eq!(out.total_cells, 0);
            }
            // Untouched piece 15 in cell0 still reads ahead at 4 tripped pieces.
            assert!(f.can_read(false, cell0 + 15 * PIECE, 256));

            // 5th distinct piece (> 4) escalates the entire 1 MiB cell.
            let addr4 = cell0 + 4 * PIECE;
            let out = f.trip(true, addr4, addr4 + 256, addr4, addr4 + 256);
            assert_eq!(out.new_pieces, 1);
            assert_eq!(out.new_cells, 1);
            assert_eq!(out.total_cells, 1);
            assert!(!out.tripped_global);

            // Every piece in cell0 (even untouched piece 15) now returns false.
            for p in 0..16 {
                assert!(!f.can_read(false, cell0 + p * PIECE, 256), "cell0 piece {p} should wait");
            }
            // Adjacent 1 MiB cell1 stays unaffected.
            assert!(f.can_read(false, cell1, 256));
            // Cross-cell range touching cell0 returns false.
            assert!(!f.can_read(false, cell1 - 16, 32));
        }

        #[test]
        fn global_safety_valve_trips_after_sixty_four_distinct_pieces() {
            let f = Fixture::new();
            let untouched = 0x9000_0000usize;
            // Spread 64 tripped pieces across 64 separate 1 MiB cells so cell escalation never interferes.
            for i in 0..64 {
                let addr = 0x3000_0000usize + i * (1 << 20);
                let out = f.trip(true, addr, addr + 64, addr, addr + 64);
                assert!(!out.tripped_global, "piece {i} must not trip global valve");
                assert!(!f.wait.load(Ordering::Relaxed));
            }
            assert!(f.can_read(false, untouched, 64));

            // 65th distinct piece (> 64) trips global WAIT safety valve.
            let addr64 = 0x3000_0000usize + 64 * (1 << 20);
            let out = f.trip(true, addr64, addr64 + 64, addr64, addr64 + 64);
            assert!(out.tripped_global);
            assert_eq!(out.total_pieces, 65);
            assert!(f.wait.load(Ordering::Relaxed));
            assert!(!f.can_read(false, untouched, 64));
        }

        #[test]
        fn fill_wait_override_and_legacy_scoped_off_behavior() {
            assert!(!parse_fill_wait(None));
            assert!(!parse_fill_wait(Some("0")));
            assert!(parse_fill_wait(Some("1")));
            assert!(parse_fill_wait(Some(" 1 ")));

            assert!(parse_scoped(None));
            assert!(parse_scoped(Some("1")));
            assert!(!parse_scoped(Some("0")));
            assert!(!parse_scoped(Some(" 0 ")));

            // VICTORY_FILL_WAIT=1 forces wait even with no tripped pieces.
            let f = Fixture::new();
            assert!(!f.can_read(true, 0x4000_0000, 256));
            assert!(!f.can_read(true, 0x4000_0000, 0));

            // VICTORY_FILL_AHEAD_SCOPED=0 trips global WAIT on the very first mismatch.
            let out = f.trip(false, 0x4000_0000, 0x4000_0100, 0x4000_0000, 0x4000_0100);
            assert!(out.tripped_global);
            assert!(f.wait.load(Ordering::Relaxed));
            assert!(!f.has_pieces.load(Ordering::Relaxed));
            assert!(!f.can_read(false, 0x5000_0000, 256));
        }

        #[test]
        fn wrote_keeps_checking_active_fills_after_first_scoped_blacklist() {
            let f = Fixture::new();
            // Allocate 3 pieces so we can pick two guaranteed distinct 64 KiB pieces.
            let buf = vec![0x11u8; 3 * PIECE];
            let base = buf.as_ptr() as usize;
            let p1 = (base + PIECE) & !(PIECE - 1);
            let p2 = p1 + PIECE;
            assert!(p1 >= base && p2 + 64 <= base + buf.len());

            f.record_read(p1, 64, 10, 5);
            f.record_read(p2, 64, 10, 5);

            let changed = [0x22u8; 64];
            let bl = f.blacklist();
            // First mismatch blacklists p1 only; p2 stays readable ahead.
            let hits1 = bl.check_wrote((false, true), &f.read_map, (5, 10), (p1, &changed), |_, _, _, _, _| {});
            assert_eq!(hits1, 1);
            assert!(f.has_pieces.load(Ordering::Relaxed));
            assert!(!f.wait.load(Ordering::Relaxed));
            assert!(!f.can_read(false, p1, 64));
            assert!(f.can_read(false, p2, 64));

            // A newer submission (landing=6 > started=5) must NOT falsely trip p2.
            let false_hits = bl.check_wrote((false, true), &f.read_map, (6, 10), (p2, &changed), |_, _, _, _, _| {});
            assert_eq!(false_hits, 0);
            assert!(f.can_read(false, p2, 64));

            // With HAS_WAIT_PIECES=true, a real mismatch on p2 is STILL caught and blacklisted.
            let hits2 = bl.check_wrote((false, true), &f.read_map, (5, 10), (p2, &changed), |_, _, _, _, _| {});
            assert_eq!(hits2, 1);
            assert!(!f.can_read(false, p2, 64));
            assert!(!f.wait.load(Ordering::Relaxed));
        }

        #[test]
        fn trip_covers_both_fill_and_overlap_ranges_and_stays_idempotent() {
            let f = Fixture::new();
            let base = 0x6000_0000usize;
            // Repeat trip on the same piece does not inflate counts or double-count pieces.
            let first = f.trip(true, base, base + 128, base + PIECE, base + PIECE + 128);
            assert_eq!(first.new_pieces, 2);
            assert_eq!(first.total_pieces, 2);
            let again = f.trip(true, base, base + 128, base + PIECE, base + PIECE + 128);
            assert_eq!(again.new_pieces, 0);
            assert_eq!(again.total_pieces, 2);
            assert!(!f.can_read(false, base, 16));
            assert!(!f.can_read(false, base + PIECE, 16));
            assert!(f.can_read(false, base + 2 * PIECE, 16));
        }

        #[test]
        fn read_batch_deduplicates_updates_in_place_and_caps_per_piece() {
            let f = Fixture::new();
            let base = 0x7000_0000usize;
            // Batch with duplicates and zero-length reads collapses to 2 distinct entries.
            record_batch(
                &f.read_map,
                &[(base + 64, 256), (base + 64, 256), (base + 512, 128), (base + 64, 0), (base + 512, 128)],
                10,
                100,
            );
            {
                let guard = f.read_map.lock().unwrap();
                let piece = &guard.as_ref().unwrap()[&(base / PIECE)];
                assert_eq!(piece.fills, vec![(base + 64, base + 320, 10, 100), (base + 512, base + 640, 10, 100)]);
            }
            // Re-recording the same range on a newer frame/submission updates in place without growing.
            record_batch(&f.read_map, &[(base + 64, 256)], 12, 140);
            {
                let guard = f.read_map.lock().unwrap();
                let piece = &guard.as_ref().unwrap()[&(base / PIECE)];
                assert_eq!(piece.fills, vec![(base + 64, base + 320, 12, 140), (base + 512, base + 640, 10, 100)]);
            }
            // Flooding one piece with more than `MAX_FILLS_PER_PIECE` distinct ranges stays capped,
            // evicting older entries in favor of newer ones.
            let piece2 = base + PIECE;
            let flood: Vec<(usize, usize)> = (0..MAX_FILLS_PER_PIECE + 40).map(|i| (piece2 + i * 16, 16)).collect();
            record_batch(&f.read_map, &flood[..MAX_FILLS_PER_PIECE], 20, 200);
            record_batch(&f.read_map, &flood[MAX_FILLS_PER_PIECE..], 21, 210);
            {
                let guard = f.read_map.lock().unwrap();
                let piece = &guard.as_ref().unwrap()[&(piece2 / PIECE)];
                assert_eq!(piece.fills.len(), MAX_FILLS_PER_PIECE);
                // All 40 newest entries from frame 21 are retained.
                assert_eq!(piece.fills.iter().filter(|entry| entry.2 == 21 && entry.3 == 210).count(), 40);
                // Remains sorted by `(from, to)`.
                assert!(piece.fills.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)));
            }
        }
    }
}

pub(crate) use fill_ahead::wrote as shaders_write;

/// Diagnostics: the fills of upload memory that made a submission wait for
/// flights (`blocked_by`), which one they are, and whether the wait was
/// needed: the bytes the fill reads are kept when it blocks and compared
/// with what it reads once the flights have landed.
///
/// Runs 202 and 206: one such wait a frame, 56 to 95 ms of the submitting
/// thread standing still, the largest single wait of the frame. The ranges
/// flights "may write" are whole views shaders can store to; if the bytes
/// a fill reads never change under the wait, the wait is for nothing.
#[cfg(feature = "diagnostics")]
mod fill_check {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// The fill waited for now: source, length, its bytes when it blocked.
    static PENDING: Mutex<Option<(usize, usize, Vec<u8>)>> = Mutex::new(None);
    /// Fills that blocked by (source page, length): how often.
    static SEEN: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());
    /// Fills checked after their wait; those whose bytes had changed; the
    /// bytes that had.
    static CHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    /// A fill of `len` bytes at `src` blocks on `written`.
    pub(super) fn blocked(written: &[(usize, usize)], src: usize, len: usize) {
        let at = written.partition_point(|r| r.1 <= src);
        let touched = written.get(at).copied().unwrap_or((0, 0));
        let key = (src & !0xfff, len);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1) == key);
        match known {
            Some(k) => seen[k].2 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, 1));
                if seen.len() <= 12 {
                    tracing::info!(
                        "a fill of upload memory waits for flights: {len} bytes at {src:#x}; the range shaders may write that it                          touches: {:#x}+{:#x} ({} ranges in the air); frame {}",
                        touched.0,
                        touched.1 - touched.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let mut bytes = vec![0u8; len.min(1 << 20)];
        if super::read_into(src as *const u8, &mut bytes) {
            *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((src, len, bytes));
        }
    }

    /// The fill of `len` bytes at `src` runs now.
    pub(super) fn runs(src: usize, len: usize) {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1) == (src, len)) {
            return;
        }
        let Some((_, _, before)) = pending.take() else { return };
        drop(pending);
        let mut now = vec![0u8; before.len()];
        if !super::read_into(src as *const u8, &mut now) {
            return;
        }
        let differ = before.iter().zip(&now).filter(|(a, b)| a != b).count() as u64;
        let n = CHECKED[0].fetch_add(1, Relaxed) + 1;
        if differ > 0 {
            let bad = CHECKED[1].fetch_add(1, Relaxed) + 1;
            CHECKED[2].fetch_add(differ, Relaxed);
            if bad <= 12 {
                let first = before.iter().zip(&now).position(|(a, b)| a != b).unwrap_or(0);
                tracing::info!(
                    "a fill of upload memory that waited for flights read other bytes after the wait: {differ} of {len} bytes at                      {src:#x} changed, the first at +{first:#x} (check {n}, frame {})",
                    crate::frame::current()
                );
            }
        }
    }

    /// The title buffer copy waited for now: destination, source, length,
    /// the bytes of both when it blocked.
    static PENDING_COPY: Mutex<Option<(usize, usize, usize, Vec<u8>, Vec<u8>)>> = Mutex::new(None);
    /// Copies checked after their wait; those whose source had changed;
    /// those whose destination had.
    static COPIES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    /// Copies that blocked by (source page, destination page, length).
    static SEEN_COPIES: Mutex<Vec<(usize, usize, usize, u64)>> = Mutex::new(Vec::new());

    /// A title buffer copy of `len` bytes from `src` to `dst` blocks on
    /// `written`.
    pub(super) fn blocked_copy(written: &[(usize, usize)], dst: usize, src: usize, len: usize) {
        let by_src = super::touches(written, src, src + len);
        let by_dst = super::touches(written, dst, dst + len);
        let key = (src & !0xfff, dst & !0xfff, len);
        let mut seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1, s.2) == key);
        match known {
            Some(k) => seen[k].3 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, key.2, 1));
                if seen.len() <= 16 {
                    let range = |at: usize| {
                        let k = written.partition_point(|r| r.1 <= at);
                        written.get(k).copied().unwrap_or((0, 0))
                    };
                    let (s, d) = (range(src), range(dst));
                    tracing::info!(
                        "a title buffer copy waits for flights: {len} bytes from {src:#x} to {dst:#x}; source in a range shaders may                          write: {by_src} ({:#x}+{:#x}); destination: {by_dst} ({:#x}+{:#x}); {} ranges in the air; frame {}",
                        s.0,
                        s.1 - s.0,
                        d.0,
                        d.1 - d.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let n = len.min(1 << 20);
        let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
        if super::read_into(src as *const u8, &mut a) && super::read_into(dst as *const u8, &mut b) {
            *PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner()) = Some((dst, src, len, a, b));
        }
    }

    /// The title buffer copy runs now.
    pub(super) fn copy_runs(dst: usize, src: usize, len: usize) {
        let mut pending = PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1, p.2) == (dst, src, len)) {
            return;
        }
        let Some((_, _, _, src_before, dst_before)) = pending.take() else { return };
        drop(pending);
        let (mut a, mut b) = (vec![0u8; src_before.len()], vec![0u8; dst_before.len()]);
        if !super::read_into(src as *const u8, &mut a) || !super::read_into(dst as *const u8, &mut b) {
            return;
        }
        let n = COPIES[0].fetch_add(1, Relaxed) + 1;
        let (src_changed, dst_changed) = (a != src_before, b != dst_before);
        if src_changed {
            COPIES[1].fetch_add(1, Relaxed);
        }
        if dst_changed {
            COPIES[2].fetch_add(1, Relaxed);
        }
        if (src_changed || dst_changed) && COPIES[1].load(Relaxed) + COPIES[2].load(Relaxed) <= 16 {
            let differ = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p != q).count();
            tracing::info!(
                "a title buffer copy that waited for flights found other bytes after the wait: {len} bytes from {src:#x} to {dst:#x};                  {} source bytes changed, {} destination bytes changed (check {n}, frame {})",
                differ(&a, &src_before),
                differ(&b, &dst_before),
                crate::frame::current()
            );
        }
    }

    /// With the frame line.
    pub(super) fn line() {
        let copies = COPIES[0].load(Relaxed);
        if copies > 0 {
            let seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
            let mut most: Vec<_> = seen.clone();
            most.sort_unstable_by_key(|s| std::cmp::Reverse(s.3));
            let most: Vec<String> =
                most.iter().take(6).map(|s| format!("{:#x}->{:#x}+{:#x} {}x", s.0, s.1, s.2, s.3)).collect();
            tracing::info!(
                "title buffer copies that waited for flights, since the start: {copies} checked after their wait; the source had                  other bytes after it {} times, the destination {} times; {} distinct copies, the most frequent: {}",
                COPIES[1].load(Relaxed),
                COPIES[2].load(Relaxed),
                seen.len(),
                most.join(", ")
            );
        }
        let n = CHECKED[0].load(Relaxed);
        if n == 0 {
            return;
        }
        let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let mut most: Vec<_> = seen.clone();
        most.sort_unstable_by_key(|s| std::cmp::Reverse(s.2));
        let most: Vec<String> = most.iter().take(6).map(|s| format!("{:#x}+{:#x} {}x", s.0, s.1, s.2)).collect();
        tracing::info!(
            "fills of upload memory that waited for flights, since the start: {n} checked after their wait, {} of them read other              bytes than before it ({} bytes in all); {} distinct fills (page, length), the most frequent: {}",
            CHECKED[1].load(Relaxed),
            CHECKED[2].load(Relaxed),
            seen.len(),
            most.join(", ")
        );
    }
}

/// Since the last frame-rate line, ns spent scanning lists' refreshes for
/// copies shaders may be writing (`stale_under_flights`).
static UNDER_SCAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> bool {
    // Per copy, the disjoint refreshed spans (a flush brings those ranges up
    // to date without spanning unrelated gaps between them).
    let scanning = std::time::Instant::now();
    let mut spans = Batch::default();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && shadow.writes_pending()
            {
                spans.add(shadow, *address, *len);
            }
        }
    }
    UNDER_SCAN.fetch_add(scanning.elapsed().as_nanos() as u64, Ordering::Relaxed);
    stale_under_flights_spans(&spans)
}

fn stale_under_flights_spans(spans: &Batch) -> bool {
    if spans.refreshes.is_empty() {
        return false;
    }
    if late_exact() {
        static RUNS: OnceLock<usize> = OnceLock::new();
        let mut runs_left = *RUNS.get_or_init(|| {
            std::env::var("VICTORY_MERGE_RUNS").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(512, |n| n.clamp(1, 1 << 16))
        });
        return spans
            .iter()
            .any(|(shadow, from, to)| shadow.merge_under_writes(from, to - from, &mut runs_left));
    }
    spans
        .iter()
        .any(|(shadow, from, to)| shadow.stale_under_writes(from, to - from))
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return false;
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return false;
        }
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// [`late_met`] without its settle: whether a bottom-of-pipe write of a
/// flight in the air, or of one being landed, is known to meet `wait`.
/// False says nothing about what else the flights may write there.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let line = line();
    let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
        Late::Write(at, what, n) => Some((at, what, n)),
        _ => None,
    }));
    for (at, what, n) in writes {
        if let crate::queue::Written::Value(v) = what
            && *at == address
            && *n >= bytes
            && wait.holds_for(*v)
        {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let landed = std::mem::take(&mut line().landed);
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    let Some(host) = host() else {
        return;
    };
    crate::framelog::keep_on_fast_cores("victory-late-wait");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            land();
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT, default 3; 0 waits for each)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY, default 2; 0 turns it off)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    submission_us(1, us);
}

/// Whether the landing thread lands one flight at a time (with what
/// follows it in the line) rather than every flight the card has run
/// (`VICTORY_RELAY_LAND_ONE`: on with the slots of the fetch inside
/// flights in the card's memory, [`inflight_on_card`], unless `=0`; `=1`
/// on in any case).
fn land_one() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_RELAY_LAND_ONE").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => inflight_on_card(),
    })
}

/// Of the first `fences.len()` things of the line, all landable
/// ([`landable`]): how many make the first landing with [`land_one`],
/// the first flight and what follows it up to the next flight.
pub(crate) fn first_landing(fences: &[Option<u64>]) -> usize {
    match fences.iter().position(Option::is_some) {
        Some(first) => fences[first + 1..].iter().position(Option::is_some).map_or(fences.len(), |next| first + 1 + next),
        None => fences.len(),
    }
}

/// [`RelayFetch::copy`]: a copy queue, an allocator, and a command list for
/// copying card slots down.
type CopyFetch = (ID3D12CommandQueue, ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// With [`inflight_on_card`]: the copy queue, its allocator and list,
    /// made at the first fetch there (they signal `fence` too).
    copy: Option<CopyFetch>,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    let Some(host) = host() else {
        return;
    };
    crate::framelog::keep_on_fast_cores("victory-relay");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    let mut fences: Vec<Option<u64>> = Vec::new();
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                fences.clear();
                fences.extend(line.line.iter().map(|l| match l {
                    Late::Run(flight) => Some(flight.fence),
                    _ => None,
                }));
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                let n = if land_one() { first_landing(&fences[..n]) } else { n };
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards).
fn relay_hand_back(back: &[Declared]) {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    if work.is_empty() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    let scanning = std::time::Instant::now();
    // Looked through without a lock: a piece that changes meanwhile (a
    // later flight writing, the title's bytes going in) is found now or by
    // that flight's landing, and the merge looks again under the lock.
    let found = crate::shadow::find_all(&work);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    let (mut now, mut title) = (Vec::new(), Vec::new());
    for ((shadow, from, to), found) in work.iter().zip(found) {
        let mut written = 0;
        for (at, n) in found {
            written += shadow.relay_merge(at, n, &mut now, &mut title);
        }
        copy_back(1, written as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::compared(*from, to - from, written);
        let _ = (from, to);
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    fill_check::line();
    fill_ahead::line(frames);
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        let [again, noted] = [0, 1].map(|i| BESIDE_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        tracing::info!(
            "the relay per frame (VICTORY_RELAY={}): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it); pieces fetched on the copy queue beside later \
             submissions: {noted:.1}, pieces those fetched again at their own landing whatever the compare said: {again:.1} \
             (VICTORY_QUEUE2_REFETCH)",
            if relay_exact() { 2 } else { 1 },
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if inflight_fetch() {
            let i: Vec<f64> = INFLIGHT_STATS.iter().map(|i| i.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "card copies' pieces fetched inside their flights per frame (VICTORY_CARD_FETCH_INFLIGHT={}): {:.1} pieces \
                 ({:.2} MB) merged with no fetch of their own; {:.1} landings fetched the old way (the wait above), {:.1} of \
                 them because a flight's pieces overflowed the slots; slots in the card's memory copied down on the landing \
                 thread's copy queue: {:.1}x {:.2} MB in {:.0} copies, {:.1} ms",
                inflight_mode(),
                i[0],
                i[1] / (1 << 20) as f64,
                i[3],
                i[2],
                i[4],
                i[5] / (1 << 20) as f64,
                i[7],
                i[6] / 1000.0,
            );
        }
        let [ahead, signals, us, longest] = [0, 1, 2, 3].map(|i| EARLY[i].swap(0, Ordering::Relaxed) as f64);
        tracing::info!(
            "title fence signals set by the relay per frame: {:.1}, {:.2} ms on average after it took their flights from the card \
             (the longest {:.1} ms); {:.1} flights told before the copy-back of later ones (VICTORY_EARLY_SIGNAL=1: now {})",
            signals / frames,
            if signals > 0.0 { us / signals / 1000.0 } else { 0.0 },
            longest / 1000.0,
            ahead / frames,
            if early_signal() { "on" } else { "off" }
        );
    }
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        let gave_up = crate::shadow::merge_gave_up();
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB; in these frames the merge gave up (the flights are then settled whole): {} times \
             because the copy's memory is not watched, {} because the write tracking had no answer, {} for more changed runs than \
             VICTORY_MERGE_RUNS leaves, {} for a copy in PC memory without its snapshot",
            bytes / (1 << 20) as f64,
            gave_up[0],
            gave_up[1],
            gave_up[2],
            gave_up[3]
        );
        let u = crate::shadow::under_stats().map(|n| n as f64 / frames);
        let ms = |i: usize| u[i] / 1e6;
        tracing::info!(
            "of those, ms a frame: {:.2} finding copies in the lists' refreshes, {:.0} calls: {:.2} asking what was written, \
             {:.2} finding the pages, {:.2} waiting for the locks, {:.2} comparing pages (threads together: {:.2} reading title \
             memory, {:.2} comparing and putting in), {:.2} noting the runs; {:.0} stale pages",
            UNDER_SCAN.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames,
            u[9],
            ms(0),
            ms(1),
            ms(2),
            ms(3),
            ms(5),
            ms(6),
            ms(4),
            u[8],
        );
    }
    crate::shadow::lookup_cache_line(frames);
    crate::gpu_detile::line(frames);
    {
        let parts = [0, 1, 2, 3, 4, 5].map(|i| PREP_PARTS[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
        tracing::info!(
            "a submission's preparing by part, ms per frame: the write-watch asking and the look for title bytes under flights \
             (settles for them included) {:.1}; the lists' steps (title copies, fills, preludes; refreshes are only noted) {:.1}; \
             the refreshes (the flush) {:.1}; snapshots and the wait for card results {:.1}; the card copies' send and hash lists \
             (card_before) {:.1}; handing the lists to the card and signalling {:.1}",
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
            parts[5]
        );
    }
}

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// See [`HostList::defer_gpu_write`].
fn declare_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DECLARE_ONCE").map_or(true, |v| v != "0"))
}

/// Merges declared write lists from multiple flights into one deduplicated
/// `Vec<Declared>` in O(N log N) instead of O(N^2) linear scans.
fn merge_declared<'a>(sets: impl IntoIterator<Item = &'a [Declared]>) -> Vec<Declared> {
    let mut index: std::collections::HashMap<usize, usize, std::hash::BuildHasherDefault<AddressHash>> =
        std::collections::HashMap::default();
    let mut back: Vec<Declared> = Vec::new();
    for declared in sets {
        for (shadow, ranges) in declared {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            match index.get(&key) {
                Some(&at) => back[at].1.extend_from_slice(ranges),
                None => {
                    index.insert(key, back.len());
                    back.push((shadow.clone(), ranges.clone()));
                }
            }
        }
    }
    for (_, ranges) in &mut back {
        if ranges.len() > 1 {
            ranges.sort_unstable();
            ranges.dedup();
        }
    }
    back
}

/// Coalesces overlapping or touching sub-ranges `(lo, hi)` within a single
/// 64 KiB piece in place, leaving disjoint ranges untouched (so undeclared
/// gaps are never fetched or merged).
fn coalesce_piece_ranges(ranges: &mut Vec<(usize, usize)>) {
    if ranges.len() < 2 {
        return;
    }
    ranges.sort_unstable();
    let mut write = 0usize;
    for read in 1..ranges.len() {
        let (lo, hi) = ranges[read];
        if lo <= ranges[write].1 {
            ranges[write].1 = ranges[write].1.max(hi);
        } else {
            write += 1;
            ranges[write] = (lo, hi);
        }
    }
    ranges.truncate(write + 1);
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    // Textures shaders wrote that the title reads on the CPU ([`crate::texback`]).
    for back in &list.deferred().texture_backs {
        crate::texback::land(back);
    }
    for (buffer, at, [row, rows, pitch]) in &list.deferred().backs {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            copy_back(1, written as u64);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            copy_back(1, written as u64);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host's pixel shaders read `SV_Barycentrics` and each
    /// vertex's attribute values (`GetAttributeAtVertex`), which a pixel
    /// shader linked to a mesh shader whose outputs are packed needs
    /// (`victory_shader::emit::Link`); logged once.
    pub(crate) fn barycentrics(&self) -> bool {
        static BARYCENTRICS: OnceLock<bool> = OnceLock::new();
        *BARYCENTRICS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let barycentrics = ok && options.BarycentricsSupported.as_bool();
            tracing::info!(barycentrics, "host barycentrics support");
            barycentrics
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                card_budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                card_constants: Mutex::default(),
                window_check: OnceLock::new(),
                _alive: Counted::new(),
                meter: AtomicU32::new(u32::MAX),
                pipeline_meter: AtomicU32::new(u32::MAX),
                gather: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            card_clock_us(0, since.elapsed().as_micros() as u64);
            card_clock_us(1, worked().saturating_sub(worked_then));
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        let source = if hash_wide() { CARD_COMPARE_WIDE } else { CARD_COMPARE };
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                inflight: if inflight_fetch() { self.make_card_inflight(RESULTS) } else { None },
            })
        }
    }

    /// [`CardInflight`] for `results` bytes of the compare's results; `None`
    /// (and the old fetch) if the shader or the memory cannot be had.
    fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<CardInflight> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out
            // pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_FETCH.as_ptr().cast(),
                    CARD_FETCH.len(),
                    windows::core::s!("card fetch in flight"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let slots = inflight_slots(inflight_fetch_bytes());
            let (table, table_data) = self.shared_buffer(4 + results).ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let fetched_bytes = u64::from(slots) * INFLIGHT_SLOT;
            let (fetched, fetched_data) = if inflight_on_card() {
                (self.uav_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?, std::ptr::null_mut())
            } else {
                self.shared_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?
            };
            name(&fetched, || "card fetch in flight: pieces".into());
            // SAFETY: the mapping holds the counter; nothing of ours ran
            // on the card with it yet.
            unsafe { table_data.cast::<u32>().write_volatile(0) };
            // SAFETY: the blob's own bytes; the root signature is inside
            // the shader.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, 4 + results),
                    fetched: (fetched, fetched_data as usize, slots),
                    landed: 0,
                    brought: Vec::new(),
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.fetched.2,
                    on_card = inflight_on_card(),
                    "card copies' flagged pieces come down inside their flights (VICTORY_CARD_FETCH_INFLIGHT={}{})",
                    inflight_mode(),
                    if inflight_on_card() {
                        ": slots in the card's memory, copied down at the landing on a copy queue"
                    } else {
                        ""
                    }
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no fetch inside flights (VICTORY_CARD_FETCH_INFLIGHT={}): pieces are fetched the old way",
                    inflight_mode()
                );
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    ///
    /// `sent`: the last value signalled on the host's fence; the caller
    /// signals the next one behind this submission's lists.
    ///
    /// With `VICTORY_QUEUE2` bit 2 (`crate::queue2`; off by default) the
    /// changed bytes cross from the PC's memory to the card's on a copy
    /// queue, on the card's copy engine, while the direct queue still runs
    /// what was sent before: they go into a landing buffer in the card's
    /// memory that nothing else uses. The direct queue is told to wait for
    /// that copy and then moves the bytes from the landing buffer into the
    /// copies, inside the card's memory, in the same place of its order as
    /// before. So the copies change exactly when they did on one queue; what
    /// leaves the direct queue is the crossing from the PC's memory.
    fn card_before(&self, declared_all: &[Declared], flown: bool, sent: u64) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per card-local copy.
        let declared: Vec<&Declared> = declared_all.iter().filter(|(shadow, _)| shadow.carded()).collect();
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        // With fetches on a copy queue (`crate::queue2`, bit 1 of
        // `VICTORY_QUEUE2`): the send's number is signalled on the direct
        // queue behind the send, so that a fetch knows which sends the card
        // has finished (`Second::order_done`).
        let numbered = || {
            if sent_as != 0
                && let Some(second) = crate::queue2::second().filter(|s| s.fetches())
            {
                second.sent_up(&self.queue, sent_as);
            }
        };
        if declared.is_empty() && dirty.is_empty() {
            numbered();
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        // The changed bytes through the copy queue (`VICTORY_QUEUE2` bit
        // 2): the landing buffer they are in, in the order of `dirty`'s
        // ranges, once the direct queue has waited. `None`: straight from
        // the PC's memory on the direct queue, as before.
        let mut landed: Option<ID3D12Resource> = None;
        if let Some(second) = crate::queue2::second().filter(|s| s.uploads()) {
            let bytes: u64 = dirty
                .iter()
                .filter(|(shadow, _)| shadow.carded())
                .flat_map(|(_, ranges)| ranges.iter().map(|&(a, b)| (b - a) as u64))
                .sum();
            // SAFETY: a plain query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            // Held until the card has passed this submission's signal: the
            // direct queue's copy out of it runs ahead of that.
            if bytes > 0
                && let Some(landing) = second.landing(bytes, completed, sent + 1)
            {
                // SAFETY: an open copy list of the lane's; live buffers,
                // and the ranges lie in them; the host's queue.
                let sent_up = unsafe {
                    (|| -> windows::core::Result<()> {
                        let copy = second.up.list()?;
                        let mut at = 0u64;
                        for (shadow, ranges) in &dirty {
                            let Some((staging, _, _)) = shadow.card_sides() else {
                                continue;
                            };
                            for &(a, b) in ranges {
                                copy.1.CopyBufferRegion(&landing, at, staging, a as u64, (b - a) as u64);
                                at += (b - a) as u64;
                            }
                        }
                        let job = second.up.send(copy, &[], bytes)?;
                        // Nothing sent to the direct queue from here on
                        // runs before the bytes have landed.
                        second.hold(&self.queue, job)
                    })()
                };
                if let Err(e) = sent_up {
                    fail(format_args!(
                        "host copies in the card's memory could not be brought up to date through the copy queue ({e}, VICTORY_QUEUE2)"
                    ));
                }
                landed = Some(landing);
            }
        }
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Sync);
                    // Where the next range lies in the landing buffer.
                    let mut at = 0u64;
                    let mut copies = |shadow: &crate::shadow::Shadow, staging: &ID3D12Resource, ranges: &[(usize, usize)]| {
                        for &(a, b) in ranges {
                            match &landed {
                                Some(landing) => list.CopyBufferRegion(&shadow.buffer, a as u64, landing, at, (b - a) as u64),
                                None => list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64),
                            }
                            at += (b - a) as u64;
                            crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                    };
                    // All copies first, then one barrier, then all passes
                    // ([`hash_wide`]).
                    let wide = hash_wide();
                    if wide {
                        let mut barriers = Vec::new();
                        for (shadow, ranges) in &dirty {
                            let Some((staging, before, _)) = shadow.card_sides() else {
                                continue;
                            };
                            copies(shadow, staging, ranges);
                            if before.is_some() && !ranges.is_empty() {
                                barriers.push(transition(
                                    &shadow.buffer,
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                ));
                            }
                        }
                        if !barriers.is_empty() {
                            list.ResourceBarrier(&barriers);
                        }
                    }
                    let mut compute_bound = false;
                    for (shadow, ranges) in &dirty {
                        let (Some((staging, before, _)), size) = (shadow.card_sides(), shadow.card_hash_size()) else {
                            continue;
                        };
                        if !wide {
                            copies(shadow, staging, ranges);
                        }
                        let Some(before) = before else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        if !wide && !ranges.is_empty() {
                            list.ResourceBarrier(&[transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )]);
                        }
                        if !compute_bound {
                            list.SetComputeRootSignature(&work.root);
                            list.SetPipelineState(&work.pso);
                            list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                            compute_bound = true;
                        }
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            ranges.iter().map(|&(a, b)| shadow.card_run(a, b)).collect()
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            let constants = [run.first, run.skew, size as u32, run.pieces, 0, 0];
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.Dispatch(hash_groups(run.pieces), 1, 1);
                        }
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        numbered();
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            sent_as,
            inflight: false,
            inflight_landed: 0,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Check);
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let (Some((_, Some(before), _)), size) = (r.shadow.card_sides(), r.shadow.card_hash_size()) else {
                            continue;
                        };
                        let constants = [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, if flown { 2 } else { 1 }];
                        list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.Dispatch(hash_groups(r.span.pieces), 1, 1);
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    // The fetch inside the flight (`inflight_fetch`): behind
                    // every compare of the list, the pieces they flagged go
                    // to slots in the PC's memory.
                    if flown && let Some(inflight) = &work.inflight {
                        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        }]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, inflight.table.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, inflight.fetched.0.GetGPUVirtualAddress());
                        let mut every = true;
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let groups = r.span.pieces.clamp(1, 65535);
                            let constants =
                                [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, inflight.landed, inflight.fetched.2, groups];
                            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.Dispatch(groups, 1, 1);
                        }
                        plan.inflight = every;
                        plan.inflight_landed = inflight.landed;
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        let list = self.make_prelude()?;
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                fence,
                event,
                sent: 0,
                buffer: None,
                copy: None,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up.
    ///
    /// With [`inflight_fetch`] the pieces are first looked for where the
    /// flights' own compare lists put them ([`Host::relay_card_inflight`]);
    /// only a landing that cannot take them from there fetches, as here.
    /// `behind`: the value signalled on the host's fence behind the last of
    /// the flights. With `VICTORY_QUEUE2` bit 1 (`crate::queue2`) the fetch
    /// goes to a copy queue, behind that value and nothing else: it does
    /// not stand behind what the submitting thread sent meanwhile (run
    /// 118: 114-157 ms a frame of this thread's time). It then reads the
    /// card's side while later flights may be writing it: a byte is the
    /// old one or the new one, and the flight that writes it has a compare
    /// and a landing of its own behind it.
    ///
    /// `plans` come with their flights' numbers and `landed` is the number
    /// of the last flight being landed: a piece an earlier fetch on the
    /// copy queue brought down beside one of these flights is fetched
    /// whatever the compare says ([`BESIDE`]).
    fn relay_card_back(&self, plans: &[(u64, &CardPlan)], fetch: &mut Option<RelayFetch>, behind: u64, landed: u64) {
        if inflight_fetch() {
            let only: Vec<&CardPlan> = plans.iter().map(|p| p.1).collect();
            let taken = self.relay_card_inflight(plans, fetch, behind, landed);
            inflight_tickets_landed(&only);
            if taken {
                return;
            }
        }
        let lane = crate::queue2::second().filter(|s| s.fetches());
        // The pieces fetched beside flights that had started, as noted by
        // earlier landings: looked at for these plans, renewed below.
        let beside = lane.is_some() && refetch();
        let noted: std::collections::HashMap<(usize, u32), u64> = if beside {
            BESIDE.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default()
        } else {
            Default::default()
        };
        let mut again = 0u64;
        // Every piece fetched now: (copy, piece number).
        let mut numbers: Vec<(usize, u32)> = Vec::new();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        if let Some((differs, dwords)) = differs {
            // SAFETY: the mapping of the results. The compares of these
            // plans have run, and their dwords are given to no other plan
            // before these are done with (`card_done`).
            let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
            // Group flagged sub-ranges by (copy, 64 KiB piece) and coalesce
            // overlapping/touching spans so each overlapping region is fetched
            // once, every span remains inside a single 64 KiB page, and all
            // spans are strictly pairwise disjoint for parallel merging.
            let mut by_piece: std::collections::HashMap<
                (usize, u32),
                (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>),
            > = std::collections::HashMap::new();
            for &(number, plan) in plans {
                let mut flags: Vec<(usize, u32, bool)> = plan
                    .ranges
                    .iter()
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                        })
                    })
                    .collect();
                // A piece an earlier fetch brought down while this flight
                // may have been writing it: fetched, flagged or not.
                for flag in flags.iter_mut().filter(|f| !f.2) {
                    if fetched_beside(noted.get(&(flag.0, flag.1)).copied(), number) {
                        flag.2 = true;
                        again += 1;
                    }
                }
                let fetched = flagged_pieces(&flags);
                numbers.extend(fetched.iter().copied());
                for r in &plan.ranges {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece_num = r.span.first + p;
                        if !fetched.contains(&(copy, piece_num)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo < hi {
                            by_piece
                                .entry((copy, piece_num))
                                .or_insert_with(|| (r.shadow.clone(), Vec::new()))
                                .1
                                .push((lo, hi));
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
            }
            for (_, (shadow, mut spans)) in by_piece {
                coalesce_piece_ranges(&mut spans);
                for (lo, hi) in spans {
                    pieces.push((shadow.clone(), lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
        }
        BESIDE_STATS[0].fetch_add(again, Ordering::Relaxed);
        if pieces.is_empty() {
            if inflight_fetch() {
                // The next merge may be of a flight in the air now, with
                // bytes from when it ran: only sends up to these plans' are
                // older than it.
                if let Some(last) = plans.iter().map(|p| p.1.sent_as).max() {
                    crate::shadow::relay_fetched(last);
                }
                return;
            }
            if beside {
                // Notes whose flights have all landed are done with.
                let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                note_beside(all.get_or_insert_with(Default::default), &[], landed, landed);
            }
            // Nothing to fetch: every send so far is older than the next
            // fetch (on a copy queue: every send the card has finished).
            crate::shadow::relay_fetched(match lane {
                Some(second) => second.order_done(),
                None => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            });
            return;
        }
        if inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let waiting = std::time::Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                if let Some(second) = lane {
                    // The sends of title bytes the card has finished are in
                    // what comes down. A later one may be too, if the card
                    // runs it before this fetch: the merge leaves its bytes
                    // out as a send that came after, and they are the
                    // title's either way.
                    let after = second.order_done();
                    let copy = second.down.list()?;
                    for (shadow, lo, hi, at) in &pieces {
                        copy.1.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                    }
                    let job = second.down.send(copy, &[(&self.fence, behind)], total)?;
                    second.down.wait(job, event)?;
                    if beside {
                        // Whatever had started by now may have been
                        // writing these pieces while they came down
                        // (counted before a submission's lists go to the
                        // queue, `Host::submit`: nothing later ran yet).
                        let started = STARTED.load(Ordering::Acquire);
                        let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                        let noted = note_beside(all.get_or_insert_with(Default::default), &numbers, landed, started);
                        BESIDE_STATS[1].fetch_add(noted, Ordering::Relaxed);
                    }
                    second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                    return Ok(after);
                }
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in &pieces {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        // This fetch may bring bytes of flights in the air now: their own
        // slots are older than what it puts in the CPU's side there
        // (`relay_card_inflight`). Sends up to `after` are numbered, and the
        // submission numbered `after` may not have its lists on the queue
        // yet; every one before it has (the submission lock).
        if inflight_fetch()
            && after > 1
            && let Some(inflight) = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|w| w.inflight.as_mut())
        {
            inflight.brought.extend(pieces.iter().map(|(shadow, lo, hi, _)| (shadow.clone(), *lo, *hi, after - 1)));
        }
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        let written = if wide_merge() && pieces.len() >= 8 {
            // Every span in `pieces` is strictly pairwise disjoint (coalesced
            // per (copy, 64 KiB piece) above), and `card_merge_relay` takes a
            // shared read lock on `Card::side`: merge in parallel across workers.
            let total_written = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let each = pieces.len().div_ceil(8);
            let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                .chunks(each)
                .map(|chunk| {
                    let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = chunk.to_vec();
                    let total_written = total_written.clone();
                    Box::new(move || {
                        let mut title = Vec::new();
                        let mut local = 0usize;
                        for (shadow, lo, hi, at) in chunk {
                            // SAFETY: the mapping of the fetched bytes; the
                            // copy into it has run, and this caller waits for
                            // all jobs before returning.
                            let gpu = unsafe { std::slice::from_raw_parts((data + at as usize) as *const u8, hi - lo) };
                            let merged = shadow.card_merge_relay(lo, gpu, after, &mut title);
                            shadow.walls_merged(gpu.len(), merged);
                            local += merged;
                        }
                        total_written.fetch_add(local, Ordering::Relaxed);
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
            total_written.load(Ordering::Relaxed)
        } else {
            let mut title = Vec::new();
            let mut written = 0usize;
            for (shadow, lo, hi, at) in &pieces {
                // SAFETY: the mapping of the fetched bytes, `total` of them;
                // the copy into it has run.
                let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
                let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
                shadow.walls_merged(gpu.len(), merged);
                written += merged;
            }
            written
        };
        copy_back(1, written as u64);
        // Sends up to `after` are of no use to any later merge; with the
        // fetch inside flights only those up to these plans' are
        // (`fallback_fetched_mark`).
        let landed = plans.iter().map(|p| p.1.sent_as).max();
        crate::shadow::relay_fetched(fallback_fetched_mark(inflight_fetch(), after, landed));
    }

    /// [`Host::relay_card_back`] with [`inflight_fetch`]: the pieces these
    /// plans' compares flagged are taken from the slots their own compare
    /// lists copied them to ([`CARD_FETCH`]) and merged, plan by plan in the
    /// line's order, with no list or wait of ours. `false` (nothing merged)
    /// if any plan cannot land so; the caller then lands them all the old
    /// way, as without the switch.
    ///
    /// The merge's "fetched after" (`Shadow::card_merge_relay`) is the
    /// plan's own `sent_as`. `QUEUE_ORDER` is raised only in
    /// `Host::card_before`, under the submission lock, and that lock is held
    /// until the submission's lists and its compare's list are on the queue
    /// behind its sync list: so when the compare's list ran, the card had
    /// received exactly the sends 1..=`sent_as` of title bytes and no later
    /// one, which is what the merge's skip of later sends asks. (The old
    /// fetch, by contrast, ran whenever it reached the queue, and took the
    /// count at its own ExecuteCommandLists.)
    ///
    /// When a landing goes the old way instead, all of it does: a fetch the
    /// old way brings the card's bytes as they are now, newer than the slots
    /// of plans in the air then, and merging an older slot after it would
    /// put a flight's bytes back over a later flight's. For the same reason
    /// the bytes such a fetch brought (`CardInflight::brought`) are left
    /// out of the slots of flights whose lists were on the queue ahead of
    /// it: the CPU's side holds what the card held after them there. The
    /// rest of their slots is merged, newer than anything merged there
    /// before. (Left out whole, a flight sent before such a fetch would have
    /// had to land the old way, its fetch taking a later count still: one
    /// overflow kept every landing on the old way while flights were in
    /// the air.) Plans land in the line's order, each plan's slot as it was
    /// right behind its own lists, so every merge here is of bytes at least
    /// as new as the CPU's side holds.
    ///
    /// shortcut: all or nothing per landing, and a plan whose pieces overflow
    /// the slots lands the old way for all its pieces, not only the ones
    /// left out.
    ///
    /// With [`inflight_on_card`] the slots are in the card's memory: the
    /// pieces to merge are copied from them to a buffer in the PC's memory
    /// on the landing thread's copy queue, behind `behind` (the host fence
    /// value behind the last of these flights, reached already), and
    /// waited for, before the merge and before the tickets are landed.
    fn relay_card_inflight(&self, plans: &[(u64, &CardPlan)], fetch: &mut Option<RelayFetch>, behind: u64, landed: u64) -> bool {
        use std::sync::Arc;
        // If a prior fallback copy-queue fetch ran beside later flights
        // (`VICTORY_QUEUE2` bit 1 + `VICTORY_CARD_FETCH_INFLIGHT=2`), any
        // unflagged piece noted in `BESIDE` for one of these flights is not
        // in its inflight slot and must fall back so `BESIDE` refetches it.
        let beside = crate::queue2::second().is_some_and(|s| s.fetches()) && refetch();
        let noted: std::collections::HashMap<(usize, u32), u64> = if beside {
            BESIDE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .filter(|m| !m.is_empty())
                .cloned()
                .unwrap_or_default()
        } else {
            Default::default()
        };
        // (copy, offsets, the piece's place in the slots' buffer, "fetched after").
        let mut pieces: Vec<InflightPiece> = Vec::new();
        let ring: (ID3D12Resource, usize);
        {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(work) = work.as_ref() else {
                return false;
            };
            let Some(inflight) = &work.inflight else {
                return false;
            };
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: the mappings of the results and the tickets. The
            // compares and fetches of these plans have run, and their dwords
            // are given to no other plan before these are done with
            // (`card_done`).
            let (differs, table) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords),
                )
            };
            let slots = inflight.fetched.2;
            ring = (inflight.fetched.0.clone(), inflight.fetched.1);
            for &(number, plan) in plans {
                if plan.ranges.is_empty() {
                    continue;
                }
                if !plan.inflight || plan.ranges.iter().any(|r| !r.compared) {
                    return false;
                }
                // Pieces (copy, number) the compare flagged, with their ticket.
                let mut tickets: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let (Some(&flag), Some(&entry)) = (differs.get(at), table.get(at)) else {
                            return false;
                        };
                        if flag != 0 {
                            if entry == INFLIGHT_NONE {
                                return false;
                            }
                            if !inflight_gets_slot(entry, plan.inflight_landed, slots) {
                                INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                                return false;
                            }
                            tickets.insert((copy, r.span.first + p), entry);
                        }
                        flags.push((copy, r.span.first + p, flag != 0));
                    }
                }
                if !noted.is_empty()
                    && flags.iter().any(|&(copy, piece, flagged)| {
                        !flagged
                            && !tickets.contains_key(&(copy, piece))
                            && fetched_beside(noted.get(&(copy, piece)).copied(), number)
                    })
                {
                    return false;
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let Some(&k) = tickets.get(&(copy, piece)) else {
                            return false;
                        };
                        if lo < from || (hi - from) as u64 > INFLIGHT_SLOT {
                            return false;
                        }
                        let slot = (k % slots) as usize * INFLIGHT_SLOT as usize;
                        // Bytes a fetch the old way brought after this flight ran.
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| b.3 >= plan.sent_as && Arc::ptr_eq(&b.0, &r.shadow) && b.1 < hi && b.2 > lo)
                            .map(|b| (b.1, b.2))
                            .collect();
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            pieces.push((r.shadow.clone(), a, b, slot + (a - from), plan.sent_as));
                        }
                    }
                }
            }
        }
        // Outside the lock: no slot of these plans is written over before
        // their tickets are landed (`inflight_tickets_landed`, after this).
        // With the slots in the card's memory they come down first, into
        // `base` (mapped), one after the other; else `base` is the slots'
        // own mapping.
        let base = if inflight_on_card() {
            match self.inflight_copy_down(&ring.0, &pieces, fetch, behind) {
                Some(data) => data,
                None => return false,
            }
        } else {
            ring.1
        };
        let mut title = Vec::new();
        let (mut written, mut bytes) = (0, 0u64);
        for (shadow, lo, hi, at, after) in &pieces {
            let at = if inflight_on_card() { base + bytes as usize } else { base + at };
            // SAFETY: inside the mapping of the slots (checked above), or of
            // the buffer they were copied down to (`inflight_copy_down`
            // puts them one after the other, and has waited); the copy into
            // the slot ran in the plan's flight, which the card has run.
            let gpu = unsafe { std::slice::from_raw_parts(at as *const u8, hi - lo) };
            written += shadow.card_merge_relay(*lo, gpu, *after, &mut title);
            bytes += (hi - lo) as u64;
        }
        for &(_, plan) in plans {
            for r in &plan.ranges {
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            }
        }
        copy_back(1, written as u64);
        INFLIGHT_STATS[0].fetch_add(pieces.len() as u64, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        if !noted.is_empty() {
            let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(map) = all.as_mut() {
                note_beside(map, &[], landed, landed);
            }
        }
        // The next merge is of a later plan, or a fetch the old way sent now.
        if let Some(last) = plans.iter().map(|p| p.1.sent_as).max() {
            crate::shadow::relay_fetched(last);
        }
        true
    }

    /// [`Host::relay_card_inflight`] with [`inflight_on_card`]: `pieces`'
    /// bytes, at their places in the slots' buffer `ring`, are copied one
    /// after the other into the landing thread's buffer in the PC's memory
    /// ([`RelayFetch::buffer`]) on its copy queue (sharing `queue2::second`'s
    /// down lane when `VICTORY_QUEUE2` bit 1 is on), which first waits on the
    /// host fence for `behind`; and waited for. The mapping of the buffer,
    /// or `None` (the landing then goes the old way) if the queue or the
    /// memory cannot be had.
    fn inflight_copy_down(
        &self,
        ring: &ID3D12Resource,
        pieces: &[InflightPiece],
        fetch: &mut Option<RelayFetch>,
        behind: u64,
    ) -> Option<usize> {
        if pieces.is_empty() {
            return Some(0);
        }
        let started = std::time::Instant::now();
        let copies = slots_down(pieces.iter().map(|(_, lo, hi, at, _)| (*at as u64, (hi - lo) as u64)));
        let total: u64 = copies.iter().map(|c| c.2).sum();
        if total == 0 {
            return Some(0);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let f = fetch.as_mut()?;
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let (buffer, data, _) = f.buffer.clone()?;
        let ran = if let Some(second) = crate::queue2::second().filter(|s| s.fetches()) {
            let event = f.event;
            (|| -> windows::core::Result<()> {
                let copy = second.down.list()?;
                record_slots_down(&copy.1, &buffer, ring, &copies);
                let job = second.down.send(copy, &[(&self.fence, behind)], total)?;
                second.down.wait(job, event)?;
                second.down.stats.waited_us.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                Ok(())
            })()
        } else {
            let (queue, allocator, list) = self.relay_copy_lane(f)?;
            f.sent += 1;
            let (fence, event, sent) = (f.fence.clone(), f.event, f.sent);
            // SAFETY: the list's last run was waited for by this thread; live
            // buffers; the queue, fence and event are this thread's; the host
            // fence is ours.
            unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(&allocator, None)?;
                    record_slots_down(&list, &buffer, ring, &copies);
                    list.Close()?;
                    queue.Wait(&self.fence, behind)?;
                    queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
                    queue.Signal(&fence, sent)?;
                    fence.SetEventOnCompletion(sent, event)?;
                    WaitForSingleObject(event, INFINITE);
                    Ok(())
                })()
            }
        };
        if let Err(e) = ran {
            fail(format_args!("the slots of the fetch inside flights could not be copied down ({e}, VICTORY_CARD_FETCH_INFLIGHT=2)"));
        }
        let us = started.elapsed().as_micros() as u64;
        INFLIGHT_STATS[4].fetch_add(1, Ordering::Relaxed);
        INFLIGHT_STATS[5].fetch_add(total, Ordering::Relaxed);
        INFLIGHT_STATS[6].fetch_add(us, Ordering::Relaxed);
        INFLIGHT_STATS[7].fetch_add(copies.len() as u64, Ordering::Relaxed);
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        Some(data)
    }

    /// The landing thread's copy queue ([`RelayFetch::copy`]), made the
    /// first time.
    fn relay_copy_lane(&self, f: &mut RelayFetch) -> Option<CopyFetch> {
        if f.copy.is_none() {
            // SAFETY: plain object creation on our device.
            let made = unsafe {
                (|| -> windows::core::Result<CopyFetch> {
                    let queue: ID3D12CommandQueue = self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })?;
                    let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                    let list: ID3D12GraphicsCommandList =
                        self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
                    list.Close()?;
                    Ok((queue, allocator, list))
                })()
            };
            match made {
                Ok(made) => {
                    name(&made.0, || "victory relay copy queue".into());
                    tracing::info!(
                        "the landing thread copies card pieces down on a copy queue of its own (VICTORY_CARD_FETCH_INFLIGHT=2)"
                    );
                    f.copy = Some(made);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no copy queue for the landing thread");
                }
            }
        }
        f.copy.clone()
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    ///
    /// With [`early_signal`], the flights before the first one that has
    /// something to copy back, and what follows each of them, are told
    /// first, before that copy-back.
    fn relay_land(&self, mut items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        let mut landed = Vec::new();
        // SAFETY: a plain query (a removed device fails below, with what
        // the flights ran).
        if early_signal() && unsafe { self.device.GetDeviceRemovedReason() }.is_ok() {
            let covered: Vec<Option<bool>> = items
                .iter()
                .map(|l| match l {
                    Late::Run(flight) => Some(flight.covered()),
                    _ => None,
                })
                .collect();
            let rest = items.split_off(told_ahead(&covered));
            let ahead = std::mem::replace(&mut items, rest);
            let plans: Vec<(u64, &CardPlan)> = ahead
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => flight.card.as_ref().map(|card| (flight.number, card)),
                    _ => None,
                })
                .collect();
            if !plans.is_empty() {
                let behind = ahead
                    .iter()
                    .filter_map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                let landed_num = ahead
                    .iter()
                    .filter_map(|l| match l {
                        Late::Run(flight) => Some(flight.number),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(0);
                // Nothing to fetch for these: their sends are noted as seen.
                self.relay_card_back(&plans, fetch, behind, landed_num);
            }
            let flights = ahead.iter().filter(|l| matches!(l, Late::Run(_))).count() as u64;
            EARLY[0].fetch_add(flights, Ordering::Relaxed);
            RELAY_STATS[1].fetch_add(flights, Ordering::Relaxed);
            if flights > 0 && !items.iter().any(|l| matches!(l, Late::Run(_))) {
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
            }
            let has_more = !items.is_empty();
            landed.extend(self.relay_tell(ahead, landing, has_more));
        }
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                fill_ahead::landing(flights.iter().map(|f| f.number).min().unwrap_or(0));
                let plans: Vec<(u64, &CardPlan)> = flights.iter().filter_map(|f| f.card.as_ref().map(|card| (f.number, card))).collect();
                let behind = flights.iter().map(|f| f.fence).max().unwrap_or(0);
                let landed_num = flights.iter().map(|f| f.number).max().unwrap_or(0);
                self.relay_card_back(&plans, fetch, behind, landed_num);
                let back = merge_declared(flights.iter().map(|f| f.back.as_slice()));
                relay_hand_back(&back);
                copy_back(0, landing.elapsed().as_micros() as u64);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        landed.extend(self.relay_tell(items, landing, false));
        self.log_messages();
        fill_ahead::landing(0);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                    ranges.extend(flight.texture_back_ranges.iter().copied());
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// [`relay`], landing: what tells the title the card is done with
    /// `items`, in the order given (each flight's read-backs, its
    /// bottom-of-pipe writes, the fence signals behind it). `landing` is
    /// when the relay took them. `flush_in_hand` trims completed writes from
    /// `LINE.in_hand` when more flights remain in this landing so early
    /// signals cannot leave stale writes visible to `late_met`. Returns the
    /// flights' host lists.
    fn relay_tell(
        &self,
        items: Vec<Late>,
        landing: std::time::Instant,
        flush_in_hand: bool,
    ) -> Vec<(Vec<crate::list::Detached>, Vec<usize>)> {
        let mut last = 0;
        let mut landed = Vec::new();
        let mut writes_done = 0usize;
        let drain_writes = |writes_done: &mut usize| {
            if *writes_done > 0 {
                let mut line = line();
                let n = (*writes_done).min(line.in_hand.len());
                line.in_hand.drain(..n);
                *writes_done = 0;
            }
        };
        // Keep each flight's declared ranges (without deduplicating across
        // flights so `Shadow::flights` refcounts decrement once per flight)
        // and release `shadow.pending` only after all read-backs, writes, and
        // signals complete, strictly before `LINE_LANDED.notify_all()`.
        let mut back_done: Vec<Declared> = Vec::new();
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    back_done.extend(flight.back);
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => {
                    what.write(at as *mut u8, bytes);
                    writes_done += 1;
                }
                Late::Signal(fence, value) => {
                    drain_writes(&mut writes_done);
                    crate::fence::set(fence as crate::object::This, value);
                    let us = landing.elapsed().as_micros() as u64;
                    EARLY[1].fetch_add(1, Ordering::Relaxed);
                    EARLY[2].fetch_add(us, Ordering::Relaxed);
                    EARLY[3].fetch_max(us, Ordering::Relaxed);
                }
            }
        }
        if flush_in_hand {
            drain_writes(&mut writes_done);
        }
        for (shadow, ranges) in &back_done {
            shadow.copy_back_done(ranges);
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        landed
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    // With `VICTORY_QUEUE2` bit 1 the fetch goes to the copy
                    // queue here too, behind everything sent to the direct
                    // queue. Nothing is gained (the card has run it all
                    // when this is reached); the path is the same one the
                    // relay uses, in the plainest run there is.
                    if let Some(second) = crate::queue2::second().filter(|s| s.fetches()) {
                        let copy = second.down.list()?;
                        for &(i, lo, hi, at) in &pieces {
                            copy.1.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                        }
                        let job = second.down.send(copy, &[(&self.fence, *submitted)], total)?;
                        let was_working = working(false);
                        second.down.wait(job, self.event)?;
                        stood(waiting);
                        working(was_working);
                        second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                        return Ok(());
                    }
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            copy_back(1, written as u64);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // Behind what the queues were given before it and have not run
        // yet (`crate::queue::behind`).
        crate::queue::before_sync();
        let _ = self.submit(runs, false);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Waits for a direct queue fence with the shared event, under the submission lock.
    fn wait_for_submission(&self, target: u64, started: std::time::Instant, context: &str) {
        // SAFETY: the fence and event are ours, and the submission lock
        // prevents another wait from using this event.
        unsafe {
            if let Err(e) = self.fence.SetEventOnCompletion(target, self.event) {
                fail(format_args!("could not register a direct fence wait for {context} ({e})"));
            }
            working(false);
            match WaitForSingleObject(self.event, INFINITE) {
                WAIT_OBJECT_0 => {}
                WAIT_FAILED => {
                    let error = windows::core::Error::from_thread();
                    fail(format_args!("could not wait for {context} ({error})"));
                }
                status => fail(format_args!("unexpected direct fence wait result for {context} ({:#x})", status.0)),
            }
            stood(started);
            working(true);
        }
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                self.wait_for_submission(*submitted, waiting, "submissions sent ahead");
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                submission_us(1, gpu.as_micros() as u64);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let first_flight = items
                    .iter()
                    .filter_map(|item| match item {
                        Late::Run(flight) => Some(flight.number),
                        _ => None,
                    })
                    .min()
                    .unwrap_or(0);
                fill_ahead::landing(first_flight);
                let mut seen = std::collections::HashSet::new();
                for item in &items {
                    if let Late::Run(flight) = item
                        && let Some(card) = &flight.card
                    {
                        self.card_back(card, submitted, &mut seen);
                    }
                }
                let back = merge_declared(items.iter().filter_map(|item| match item {
                    Late::Run(flight) => Some(flight.back.as_slice()),
                    _ => None,
                }));
                hand_back(&back);
                fill_ahead::landing(0);
                copy_back(0, copying.elapsed().as_micros() as u64);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// A CPU detile must wait for the last GPU read of its staging buffer.
    /// Called with the submission lock held, so no newer upload can race it.
    pub(crate) fn wait_for_staging(&self, target: u64) {
        // SAFETY: the fence belongs to this host. A null event waits inline.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            fail(format_args!("device removed while waiting to reuse texture staging"));
        }
        if completed < target {
            let started = std::time::Instant::now();
            let was_working = working(false);
            // SAFETY: target was signalled on the host queue before preparation began.
            let waited = unsafe { self.fence.SetEventOnCompletion(target, HANDLE::default()) };
            working(was_working);
            if let Err(error) = waited {
                fail(format_args!("could not wait to reuse texture staging ({error})"));
            }
            // SAFETY: a plain query. Device removal can also wake the wait.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed == u64::MAX || completed < target {
                fail(format_args!("texture staging upload did not reach fence {target}"));
            }
            stood(started);
            STAGING_WAIT_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    /// Whether the host GPU has finished reading the staging buffer from
    /// submission `target` (non-blocking fence poll).
    pub(crate) fn staging_completed(&self, target: u64) -> bool {
        if target == 0 {
            return true;
        }
        // SAFETY: plain fence query on this host's fence.
        let completed = unsafe { self.fence.GetCompletedValue() };
        completed != u64::MAX && completed >= target
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    fn submit<'a>(&'a self, runs: &[&HostList], late: bool) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        #[cfg(feature = "diagnostics")]
        crate::profile::seen(crate::profile::Seen::Submit);
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let was_working = working(true);
        submitting(1, runs.len());
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if let Some(why) = late_blocked(runs) {
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| blocked_by(&line.written, runs, false).is_none());
            } else {
                self.settle(&mut submitted, why);
            }
        }
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted.
        crate::descriptor::sync_heaps();
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        crate::descriptor::make_waiting();
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // The parts of the preparing, for the line that splits it
        // ([`PREP_PARTS`]).
        let mut part = std::time::Instant::now();
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        // Also collect pending-write spans in the same single pass when behind flights.
        if crate::watch::once() {
            let scanning = behind.then(std::time::Instant::now);
            let mut spans = Batch::default();
            let mut pending_spans = Batch::default();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step {
                        if !shadow.unwatched() {
                            spans.add(shadow, *address, *len);
                        }
                        if behind && shadow.writes_pending() {
                            pending_spans.add(shadow, *address, *len);
                        }
                    }
                }
            }
            if let Some(scanning) = scanning {
                UNDER_SCAN.fetch_add(scanning.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            let ranges: Vec<(usize, usize)> = spans.iter().map(|(_, from, to)| (from, to)).collect();
            crate::watch::ask_once(&ranges);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if behind && stale_under_flights_spans(&pending_spans) {
                let settling = std::time::Instant::now();
                self.settle(&mut submitted, WHY_STALE);
                settled += settling.elapsed();
                crate::watch::ask_once(&ranges);
            }
        } else if behind && stale_under_flights(runs) {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        part = prep_part(0, part);
        let detile_batch = crate::gpu_detile::BatchGuard::begin();
        let mut pending_fills = Vec::new();
        for list in runs {
            list.run_deferred_into(&mut batch, &mut pending_fills);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        flush_fills(&mut pending_fills);
        part = prep_part(1, part);
        batch.flush();
        crate::watch::ask_once_end();
        part = prep_part(2, part);
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission. Reuse the writes accepted
        // by `Shadow::writing` during `run_deferred` so skipped read-only
        // ranges never enter `card_before`, `hand_back`, or `line.written`.
        let back = batch.take_declared();
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        part = prep_part(3, part);
        let card = self.card_before(&back, flown, *submitted);
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        prep_part(4, part);
        detile_batch.flush(&self.queue);
        let prepared = std::time::Instant::now();
        if late {
            *submitted += 1;
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            prep_part(5, prepared);
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            for staging in &batch.uploaded {
                staging.note_upload_fence(*submitted);
            }
            let mut texture_back_ranges = Vec::new();
            for list in runs {
                texture_back_ranges.extend(
                    list.deferred()
                        .texture_backs
                        .iter()
                        .map(crate::texback::Back::range),
                );
            }
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            submission_us(0, (prepared - started).saturating_sub(settled).as_micros() as u64);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                texture_back_ranges,
                writes,
            };
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence is ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                if let Err(e) = self.queue.Signal(&self.fence, *submitted) {
                    fail(format_args!("could not signal a synchronous host submission ({e})"));
                }
            }
            self.wait_for_submission(*submitted, started, "a synchronous host submission");
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        submission_us(0, (prepared - started).as_micros() as u64);
        submission_us(1, gpu.as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // Every earlier flight already settled before this synchronous
        // submission ran: tag `LANDING` with `this_submission` so kept fills
        // from earlier frames are not falsely blamed on this submission.
        fill_ahead::landing(this_submission);
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        hand_back(&back);
        fill_ahead::landing(0);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        copy_back(0, copying.elapsed().as_micros() as u64);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // The runtime can classify a removal even when it supplies no fault
        // address. Older runtimes still have the DRED1 report below.
        match dred.cast::<ID3D12DeviceRemovedExtendedData2>() {
            Ok(dred2) => {
                // SAFETY: a query on the live DRED interface.
                let state = unsafe { dred2.GetDeviceState() };
                let name = match state {
                    D3D12_DRED_DEVICE_STATE_HUNG => "hung",
                    D3D12_DRED_DEVICE_STATE_FAULT => "fault",
                    D3D12_DRED_DEVICE_STATE_PAGEFAULT => "page fault",
                    D3D12_DRED_DEVICE_STATE_UNKNOWN => "unknown",
                    _ => "unrecognized",
                };
                text.push_str(&format!("; DRED device state {name} ({})", state.0));
            }
            Err(e) => text.push_str(&format!("; DRED device state unavailable ({:#010X})", e.code().0)),
        }
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// With the frame line (`VICTORY_D3D12_DEBUG=1`, diagnostics builds):
    /// the debug layer's messages since the last call go to our log, the
    /// first five of each kind in full, the rest counted, with one line of
    /// the counts by kind.
    ///
    /// They were only read when the device was lost ([`fail`]): a run that
    /// survived (run 205) said nothing of what the layer had found, and
    /// what it finds is what a driver's fault would have to be traced back
    /// to (an object let go while the card uses it, a resource in the
    /// wrong state, a view of something gone).
    pub(crate) fn drain_messages(&self) {
        let Some(queue) = &self.messages else {
            return;
        };
        static SEEN: Mutex<std::collections::BTreeMap<i32, (u64, i32)>> = Mutex::new(std::collections::BTreeMap::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        let stored = unsafe {
            let stored = queue.GetNumStoredMessages();
            for i in 0..stored {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let kind = seen.entry(m.ID.0).or_insert((0, m.Severity.0));
                    kind.0 += 1;
                    if kind.0 <= 5 {
                        let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text);
                        let text = text.trim_end_matches('\0');
                        tracing::warn!(id = m.ID.0, severity = m.Severity.0, nth = kind.0, "D3D12 debug layer: {text}");
                    }
                }
            }
            queue.ClearStoredMessages();
            stored
        };
        if stored > 0 {
            let counts: Vec<String> = seen.iter().map(|(id, (n, severity))| format!("id {id} (severity {severity}) x{n}")).collect();
            tracing::info!(
                "D3D12 debug layer messages so far by kind (severity 0 corruption, 1 error, 2 warning, 3 info, 4 message): {}",
                counts.join(", ")
            );
        }
    }

    /// The host device (`crate::cardmeter`).
    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// Ticks a second of the queue's timestamps; 0 if it has none.
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0)
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.rtv_start + (slot * self.rtv_size) as usize,
        };
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
        }
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.dsv_start + (slot * self.dsv_size) as usize,
        };
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
        }
        handle
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        self.twins_to_depth();
        self.textures_back();
        let scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((scratch, _)) = scratch {
            self.indirect_common(&scratch);
        }
        self.card_budget_out();
        // The scratch buffers its prelude fills and leaves readable as
        // arguments: common again behind the list, as every scratch buffer.
        for set in self.gather.lock().unwrap_or_else(|e| e.into_inner()).sets.iter().filter(|s| s.records() > 0) {
            // SAFETY: an open list and a live buffer.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    &set.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
        crate::cardmeter::end(&self.list, crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)));
        crate::cardmeter::end_named(
            &self.list,
            crate::cardmeter::Span::from_raw(self.meter.swap(u32::MAX, Ordering::Relaxed)),
            &self.pipelines.lock().unwrap_or_else(|e| e.into_inner()),
        );
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self._alive.used();
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        if let Some(c) = self.card_constants.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            // Every submission of the list has run: the buffer is free, and
            // in the common state again.
            (c.1, c.2) = (0, false);
        }
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            let Gather { closed, written, sets, spare, .. } = &mut *gather;
            *closed = false;
            written.clear();
            for set in sets.drain(..) {
                if spare.len() < GATHER_SETS {
                    spare.push(set.into_scratch());
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        self.meter
            .store(crate::cardmeter::begin(&self.list, crate::cardmeter::Kind::Lists).raw(), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        if !deferred
            .back_wanted
            .iter()
            .any(|target| target.owner() == resource as usize)
            && let Some(target) = crate::texback::Target::new(resource)
        {
            deferred.back_wanted.push(target);
        }
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            self.settle(&target.copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        // The records the list's draws read from slots (`crate::indirect::gather`).
        let gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        let gathered = gather.sets.iter().any(|s| s.records() > 0);
        if barriers.is_empty() && syncs.is_empty() && !gathered {
            return None;
        }
        let Some((allocator, list)) = self
            .prelude
            .get_or_init(|| host().and_then(|h| h.make_prelude()))
            .as_ref()
        else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                if gathered {
                    // The slots must hold clamped records before a draw
                    // reads them: a prelude that cannot record the passes
                    // is no prelude.
                    let filled = match &gather.budget {
                        Some(budget) => crate::indirect::record_gathers(list, &gather.sets, budget),
                        None => false,
                    };
                    if !filled {
                        fail(format_args!("a list's gathered indirect records could not be clamped in its prelude"));
                    }
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// From here on the list gathers no records (`crate::indirect::gather`):
    /// what is about to be recorded may write arguments.
    pub(crate) fn gather_close(&self) {
        self.gather.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
    }

    /// Notes that a shader recorded in this list may write host copy `copy`
    /// (`VICTORY_GATHER_BY_COPY`): further indirect records from `copy` are
    /// not gathered.
    pub(crate) fn gather_wrote(&self, copy: u64) {
        if crate::indirect::gather_by_copy_kept() {
            self.gather.lock().unwrap_or_else(|e| e.into_inner()).written.insert(copy);
        }
    }

    /// A slot for the record at `offset` of the host copy `src` in the
    /// scratch buffer the list's prelude fills (`crate::indirect::gather`,
    /// which checked the record): the buffer and the slot's place in it.
    pub(crate) fn gather_slot(&self, src: &ID3D12Resource, src_id: u64, offset: u32, (kind, stride): (u32, u32)) -> Option<(ID3D12Resource, u64)> {
        let none = |why: usize| {
            crate::indirect::GATHER_NONE[why].fetch_add(1, Ordering::Relaxed);
            None
        };
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        if gather.closed || gather.written.contains(&src_id) {
            return none(1);
        }
        let found = gather
            .sets
            .iter()
            .position(|s| s.src_id == src_id && s.kind == kind && s.stride == stride && s.src.as_raw() == src.as_raw());
        let at = match found {
            Some(at) => at,
            None => {
                if gather.sets.len() >= GATHER_SETS {
                    return none(2);
                }
                // The prelude zeroes the budget as the list zeroes its own.
                if self.list.cast::<ID3D12GraphicsCommandList2>().is_err() {
                    return none(4);
                }
                let Some(host) = host() else {
                    return none(4);
                };
                if gather.budget.is_none() {
                    let Some(budget) = host.uav_buffer(256) else {
                        return none(4);
                    };
                    name(&budget, || "indirect budget of gathered records".to_owned());
                    gather.budget = Some(budget);
                }
                let scratch = match gather.spare.pop() {
                    Some(scratch) => scratch,
                    None => {
                        let Some(scratch) = host.uav_buffer(u64::from(crate::indirect::GATHER_MOST) * u64::from(crate::indirect::GATHER_STRIDE)) else {
                            return none(4);
                        };
                        name(&scratch, || "indirect arguments, gathered".to_owned());
                        scratch
                    }
                };
                let Some(table) = self.scratch(u64::from(crate::indirect::GATHER_MOST) * 4) else {
                    gather.spare.push(scratch);
                    return none(4);
                };
                gather.sets.push(crate::indirect::GatherSet::new(src.clone(), src_id, (kind, stride), scratch, table));
                gather.sets.len() - 1
            }
        };
        let set = &mut gather.sets[at];
        let Some(slot) = set.slot(offset) else {
            return none(3);
        };
        Some((set.scratch.clone(), u64::from(slot) * u64::from(stride)))
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty()
                && deferred.texture_backs.is_empty()
                && deferred.on_card.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes a pipeline the list binds; called where it is bound.
    pub(crate) fn used_pipeline(&self, name: &str) {
        {
            let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if !pipelines.contains(name) {
                pipelines.insert(name.to_owned());
            }
        }
        if crate::cardmeter::pipelines() {
            // The pipeline before it ends here on the card's clock.
            crate::cardmeter::end(&self.list, crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)));
            self.pipeline_meter.store(crate::cardmeter::begin_named(&self.list, name).raw(), Ordering::Relaxed);
        }
    }

    /// The list binds a pipeline: it and its root signature are kept until
    /// the list is reset, and its name noted, the first time only (a kept
    /// pipeline's address names no other).
    pub(crate) fn bind_pipeline(
        &self,
        name: &str,
        pso: &ID3D12PipelineState,
        root: &ID3D12RootSignature,
    ) {
        {
            let mut deferred = self.deferred();
            if !deferred.bound.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.title_written();
    }

    /// With [`fill_on_card`]: `len` bytes of the host copy `shadow` from title
    /// address `address`, copied on the card where the list stands into its
    /// device-local constants buffer: the GPU address of the bytes there (on
    /// 256 bytes, for a constant buffer view), or `None` (no buffer, no room).
    ///
    /// A fill of upload memory reads title memory on the CPU at submission,
    /// and a submission whose fill reads what a flight's shaders write waits
    /// for the flight to land (52-80 ms a frame in the world, one table
    /// constant buffer of 512 bytes). Read on the card, behind the flight on
    /// the same queue, those bytes are what the shaders wrote without the
    /// wait; title writes to them reach the copy first, through its refresh.
    pub(crate) fn card_constants(&self, shadow: &crate::shadow::Shadow, address: u64, len: u64) -> Option<u64> {
        const SIZE: u64 = 1 << 20;
        let host = host()?;
        let offset = shadow.offset_in(address)?;
        if len == 0 || shadow.left(address) < len {
            return None;
        }
        let room = len.next_multiple_of(256);
        let mut c = self.card_constants.lock().unwrap_or_else(|e| e.into_inner());
        if c.is_none() {
            let buffer = host.uav_buffer(SIZE)?;
            name(&buffer, || "list constants on the card".into());
            *c = Some((buffer, 0, false));
        }
        let (buffer, used, readable) = c.as_mut()?;
        let buffer: &ID3D12Resource = buffer;
        if *used + room > SIZE {
            return None;
        }
        let at = *used;
        *used += room;
        let constants = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        // SAFETY: an open list; both buffers are live and the ranges inside
        // them (checked above).
        unsafe {
            if *readable {
                self.list.ResourceBarrier(&[transition(buffer, constants, D3D12_RESOURCE_STATE_COPY_DEST)]);
            }
            // What shaders of this list wrote to the copy before, first.
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
            self.list.CopyBufferRegion(buffer, at, &shadow.buffer, offset, len);
            self.list.ResourceBarrier(&[transition(buffer, D3D12_RESOURCE_STATE_COPY_DEST, constants)]);
            *readable = true;
            Some(buffer.GetGPUVirtualAddress() + at)
        }
    }

    /// Notes that a table constant view (`address`, `len` bytes) was bound
    /// directly in a host buffer copy instead of filled into upload memory.
    pub(crate) fn note_table_on_card(&self, address: usize, len: usize) {
        self.deferred().on_card.push((address, len));
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// The host address of `len` bytes of scratch memory that hold the title
    /// memory at `src` as of the list's submission: one fill for every ask
    /// until the list next writes title memory.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        let known = self.deferred().filled.get(&key).copied();
        if known.is_some() {
            return known;
        }
        let (gpu, to) = self.scratch(len)?;
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Fill(to as usize, src as usize, len as usize));
        deferred.filled.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    /// Declared once until the list next writes title memory (unless
    /// `VICTORY_DECLARE_ONCE=0`).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) || !declare_once() {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Whether the list, once it has run, stores at `address` a value of
    /// `bytes` bytes or more for which `holds` says yes (one of its
    /// bottom-of-pipe writes, as [`late_met`] reads them of a flight).
    pub(crate) fn will_write(&self, address: usize, bytes: usize, holds: impl Fn(u64) -> bool) -> bool {
        self.deferred()
            .writes
            .iter()
            .any(|(at, what, n)| *at == address && *n >= bytes && matches!(what, crate::queue::Written::Value(v) if holds(*v)))
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().hold_upload(buffer, len);
            return Some((gpu, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, gpu) = (s.chunks[s.current].0.data, s.chunks[s.current].0.gpu);
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let card = self.card_budget.get_or_init(|| self.make_card_budget()).as_ref();
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all; with the card's copy, as
            // the list's end copied them here): taken, then zeroed.
            deferred.copies.push(Step::Host(Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                for k in 0..3 {
                    words.add(k).write_volatile(0);
                }
            })));
            drop(deferred);
            if let Some(card) = card {
                self.zero_card_budget(card);
            }
        }
        // SAFETY: a live buffer.
        Some(unsafe { card.unwrap_or(buffer).GetGPUVirtualAddress() })
    }

    /// The budget's words in the card's own memory ([`card_budget`]), if
    /// the list can zero them itself (`WriteBufferImmediate`).
    fn make_card_budget(&self) -> Option<ID3D12Resource> {
        if !card_budget() {
            return None;
        }
        self.list.cast::<ID3D12GraphicsCommandList2>().ok()?;
        let buffer = host()?.uav_buffer(256)?;
        name(&buffer, || "indirect budget".to_owned());
        Some(buffer)
    }

    /// Records, before the list's first clamp, the zeroing of the card's
    /// copy of the budget: common, written as a copy, then left for
    /// unordered access until the list closes ([`HostList::card_budget_out`]).
    fn zero_card_budget(&self, card: &ID3D12Resource) {
        let Ok(list2) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        // SAFETY: a live buffer.
        let at = unsafe { card.GetGPUVirtualAddress() };
        let zeros = [0u64, 4, 8].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + offset,
            Value: 0,
        });
        // SAFETY: an open list; the buffer is the list's own and in the
        // common state at the start of every recording.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }

    /// Before the list closes, if this recording charged the card's copy
    /// of the budget: its notes on mesh records go to the mapped words
    /// (read at the next submission), and it goes back to common.
    fn card_budget_out(&self) {
        if !self.deferred().budget_zeroed {
            return;
        }
        let (Some(Some(card)), Some(Some((mapped, _)))) =
            (self.card_budget.get(), self.budget.get())
        else {
            return;
        };
        // SAFETY: an open list; both buffers are the list's own. The
        // mapped one is promoted from common to the copy destination.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list.CopyBufferRegion(mapped, 4, card, 4, 8);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            // It ends the list common, as every scratch buffer does.
            self.indirect_common(&old_buf);
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Moves `scratch` back to the common state if a clamp left it readable
    /// as indirect arguments ([`HostList::indirect_readable`]): where the
    /// list's first clamp expects it. Done before the list closes and before
    /// a bigger scratch buffer replaces it: one barrier per list (where
    /// there was one per indirect execution), and the list ends as it
    /// begins however it is submitted.
    fn indirect_common(&self, scratch: &ID3D12Resource) {
        {
            let mut deferred = self.deferred();
            if deferred.indirect_readable != scratch.as_raw() as usize {
                return;
            }
            deferred.indirect_readable = 0;
        }
        let barrier = transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        // SAFETY: an open list and a live buffer this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }

    /// The state the list's indirect scratch buffer `scratch` is in at this
    /// point of the recording: indirect arguments if a clamp in this
    /// recording left it so ([`HostList::indirect_readable`]), otherwise
    /// common: the state a buffer is made in, and the one a recording
    /// leaves it in when it closes or replaces it ([`HostList::indirect_common`]).
    pub(crate) fn indirect_state(&self, scratch: &ID3D12Resource) -> D3D12_RESOURCE_STATES {
        if self.deferred().indirect_readable == scratch.as_raw() as usize {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        }
    }

    /// A clamp recorded in this list left `scratch` readable as indirect
    /// arguments. It stays so for every execution that reads it, until the
    /// next clamp moves it back to unordered access.
    pub(crate) fn indirect_readable(&self, scratch: &ID3D12Resource) {
        self.deferred().indirect_readable = scratch.as_raw() as usize;
    }

    /// The list's buffers for `crate::indirect::window_check`.
    pub(crate) fn window_check(&self) -> Option<&crate::indirect::WindowCheck> {
        self.window_check.get_or_init(crate::indirect::WindowCheck::new).as_ref()
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Whether `view` has not yet been copied from `source` in this list.
    pub(crate) fn alias_source_changed(&self, view: &ID3D12Resource, source: &ID3D12Resource) -> bool {
        let view = view.as_raw() as usize;
        let source = source.as_raw() as usize;
        self.deferred().texture_aliases.insert(view, source) != Some(source)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.title_written();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        upload_hold: Option<crate::texture::UploadHold>,
    ) {
        self.deferred().copies.push(Step::Texture { staging, upload_hold });
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    #[cfg(test)]
    fn run_deferred(&self, batch: &mut Batch) {
        let mut pending_fills = Vec::new();
        self.run_deferred_into(batch, &mut pending_fills);
        flush_fills(&mut pending_fills);
    }

    /// Runs the list's submission-time steps, collecting `Step::Fill` copies
    /// into `pending_fills` across side-effect-free steps (`Step::Refresh`,
    /// `Step::GpuWrite`) and flushing them before any barrier step
    /// (`Step::Copy`, `Step::Run`, `Step::Host`, `Step::Texture`).
    ///
    /// Callers must call [`flush_fills`] after the last list to execute any
    /// trailing fills.
    fn run_deferred_into(&self, batch: &mut Batch, pending_fills: &mut Vec<(usize, usize, usize)>) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let deferred = self.deferred();
        if deferred.copies.is_empty() {
            return;
        }
        // Time `Step::Copy` work once per list rather than per step to avoid
        // thousands of `QueryPerformanceCounter` + atomic RMW calls per frame.
        // `Step::Fill` batches are timed once per batch inside `flush_fills`.
        let mut copy_started: Option<std::time::Instant> = None;
        let mut copy_before_us = 0u64;
        let mut copy_bytes = 0usize;
        #[cfg(feature = "diagnostics")]
        let mut copy_count = 0u64;
        let mut excluded_us = 0u64;

        for step in &deferred.copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    flush_fills(pending_fills);
                    if copy_started.is_none() {
                        copy_before_us = PREPARING[0].load(Ordering::Relaxed);
                        copy_started = Some(std::time::Instant::now());
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        copy_count += 1;
                    }
                    batch.title_wrote(dst, len);
                    #[cfg(feature = "diagnostics")]
                    fill_check::copy_runs(dst, src, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    copy_bytes += len;
                }
                &Step::Fill(dst, src, len) => {
                    // Reads title memory and writes host upload memory only:
                    // commutes with `Step::Refresh` and `Step::GpuWrite`, and is
                    // flushed before any barrier step or submission handoff.
                    pending_fills.push((dst, src, len));
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                }
                Step::Run(work) | Step::Host(work) => {
                    flush_fills(pending_fills);
                    #[cfg(feature = "diagnostics")]
                    let run_started = std::time::Instant::now();
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    let excl_started = copy_started.is_some().then(std::time::Instant::now);
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                    }
                    if let Some(excl) = excl_started {
                        excluded_us += excl.elapsed().as_micros() as u64;
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, run_started.elapsed().as_micros() as u64);
                    }
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame). Neither `flush_one` nor `writing`
                    // mutates title memory or list upload buffers, so collected
                    // `Step::Fill`s remain valid across it.
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    let excl_started = copy_started.is_some().then(std::time::Instant::now);
                    if shadow.writing(*address, *len) {
                        batch.note_gpu_write(shadow, *address, *len);
                        #[cfg(feature = "diagnostics")]
                        {
                            add(C::GpuWrites, 1);
                            add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                        }
                    }
                    if let Some(excl) = excl_started {
                        excluded_us += excl.elapsed().as_micros() as u64;
                    }
                }
                Step::Texture { staging, upload_hold } => {
                    flush_fills(pending_fills);
                    if upload_hold.is_some() {
                        staging.note_upload();
                        batch.uploaded.push(staging.clone());
                    }
                    if batch.texture_due(staging) {
                        let excl_started = copy_started.is_some().then(std::time::Instant::now);
                        staging.refresh();
                        if let Some(excl) = excl_started {
                            excluded_us += excl.elapsed().as_micros() as u64;
                        }
                    }
                }
            }
        }

        if let Some(started) = copy_started {
            #[cfg(feature = "diagnostics")]
            {
                add(C::Copies, copy_count);
                add(C::CopyBytes, copy_bytes as u64);
            }
            let nested_us = PREPARING[0].load(Ordering::Relaxed).saturating_sub(copy_before_us);
            let net_us = (started.elapsed().as_micros() as u64)
                .saturating_sub(nested_us)
                .saturating_sub(excluded_us);
            preparing(0, net_us);
            preparing(1, copy_bytes as u64);
        }
    }
}

/// Minimum fill count and total bytes to dispatch a disjoint `Step::Fill` batch
/// across worker threads.
const WIDE_FILL_MIN_COUNT: usize = 8;
const WIDE_FILL_MIN_BYTES: usize = 128 << 10;
/// Single (or very few) fills whose total size reaches this threshold also
/// split across workers to saturate memory bandwidth.
const WIDE_FILL_SINGLE_BYTES: usize = 1 << 20;
/// Minimum piece size (bytes) when splitting an oversized `Step::Fill` across
/// worker chunks, matching `copy_mapped`'s `read_sparse` threshold.
const FILL_SPLIT_MIN: usize = 64 << 10;
/// Alignment (bytes) for split offsets inside an oversized `Step::Fill`.
const FILL_SPLIT_ALIGN: usize = 4 << 10;

/// Whether batches of disjoint `Step::Fill` copies run across worker threads
/// (on by default; `VICTORY_WIDE_FILL=0` turns it off).
fn wide_fill() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_FILL").is_none_or(|v| v != "0"))
}

/// Target worker chunk count for parallel `Step::Fill` batches, matching
/// `crate::shadow::wide_run`'s worker pool plus the submitting thread (3..=6).
fn wide_fill_chunks() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        (std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 4)
            .clamp(2, 6)
            + 1)
        .min(6)
    })
}

/// Returns whether all non-empty destination intervals `[dst, dst + len)` in
/// `fills` (`(dst, src, len)`) are pairwise disjoint without address overflow.
///
/// Uses an O(N) zero-allocation fast path for monotonically non-overlapping
/// bump-allocated upload rings, falling back to a sorted interval check when
/// destinations arrive out of order (for example across multiple upload chunks
/// or command lists).
fn fills_disjoint(fills: &[(usize, usize, usize)]) -> bool {
    let mut prev_end = 0usize;
    let mut monotonic = true;
    let mut non_empty = 0usize;
    for &(dst, _, len) in fills {
        if len == 0 {
            continue;
        }
        let Some(end) = dst.checked_add(len) else {
            return false;
        };
        non_empty += 1;
        if dst < prev_end {
            monotonic = false;
            break;
        }
        prev_end = end;
    }
    if monotonic || non_empty <= 1 {
        return true;
    }
    let mut intervals: Vec<(usize, usize)> = Vec::with_capacity(fills.len());
    for &(dst, _, len) in fills {
        if len == 0 {
            continue;
        }
        let Some(end) = dst.checked_add(len) else {
            return false;
        };
        intervals.push((dst, end));
    }
    intervals.sort_unstable_by_key(|&(start, _)| start);
    intervals.windows(2).all(|w| w[0].1 <= w[1].0)
}

/// Partitions `fills` (`(dst, src, len)`) into at most `max_chunks` non-empty
/// byte-balanced chunks, preserving sequential order for upload write-combining
/// locality and splitting oversized single fills at 4 KiB boundaries (with
/// pieces >= 64 KiB) so one multi-MiB fill and hundreds of tiny fills balance
/// evenly across worker threads.
fn partition_fills_by_bytes(
    fills: &[(usize, usize, usize)],
    max_chunks: usize,
) -> Vec<Vec<(usize, usize, usize)>> {
    let max_chunks = max_chunks.max(1);
    let total_bytes: usize = fills.iter().map(|&(_, _, len)| len).sum();
    if total_bytes == 0 || max_chunks == 1 {
        let non_empty: Vec<_> = fills.iter().copied().filter(|&(_, _, len)| len > 0).collect();
        return if non_empty.is_empty() {
            Vec::new()
        } else {
            vec![non_empty]
        };
    }

    let mut chunks: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(max_chunks);
    let mut current: Vec<(usize, usize, usize)> = Vec::new();
    let mut current_bytes = 0usize;
    let mut remaining_bytes = total_bytes;

    let flush_chunk = |chunks: &mut Vec<Vec<(usize, usize, usize)>>,
                       current: &mut Vec<(usize, usize, usize)>,
                       current_bytes: &mut usize| {
        if !current.is_empty() {
            chunks.push(std::mem::take(current));
            *current_bytes = 0;
        }
    };

    for &(mut dst, mut src, mut len) in fills {
        if len == 0 {
            continue;
        }
        while len > 0 {
            let chunks_left = max_chunks - chunks.len();
            if chunks_left <= 1 {
                current.push((dst, src, len));
                current_bytes += len;
                remaining_bytes -= len;
                break;
            }
            let goal = (current_bytes + remaining_bytes).div_ceil(chunks_left).max(1);
            let room = goal.saturating_sub(current_bytes);

            let min_add = if len >= 2 * FILL_SPLIT_MIN {
                FILL_SPLIT_MIN
            } else {
                len
            };
            if !current.is_empty() && min_add > room {
                let overshoot = min_add - room;
                if overshoot > room && current_bytes >= room {
                    flush_chunk(&mut chunks, &mut current, &mut current_bytes);
                    continue;
                }
            }

            if len > room && len >= 2 * FILL_SPLIT_MIN {
                let desired = room.max(FILL_SPLIT_MIN);
                let take = (desired & !(FILL_SPLIT_ALIGN - 1)).max(FILL_SPLIT_MIN);
                if len > take && len - take >= FILL_SPLIT_MIN {
                    current.push((dst, src, take));
                    current_bytes += take;
                    remaining_bytes -= take;
                    dst += take;
                    src += take;
                    len -= take;
                    flush_chunk(&mut chunks, &mut current, &mut current_bytes);
                    continue;
                }
            }

            current.push((dst, src, len));
            current_bytes += len;
            remaining_bytes -= len;
            len = 0;
            if current_bytes >= goal {
                flush_chunk(&mut chunks, &mut current, &mut current_bytes);
            }
        }
    }
    flush_chunk(&mut chunks, &mut current, &mut current_bytes);
    chunks
}

/// Executes all buffered `Step::Fill` copies, dispatching pairwise-disjoint
/// batches across `crate::shadow::wide_run` in byte-balanced chunks when
/// [`wide_fill`] is enabled and work exceeds the parallel threshold.
fn flush_fills(fills: &mut Vec<(usize, usize, usize)>) {
    if fills.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let total_bytes: usize = fills.iter().map(|&(_, _, len)| len).sum();
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Copies, fills.len() as u64);
        add(C::CopyBytes, total_bytes as u64);
        for &(_, src, len) in fills.iter() {
            fill_check::runs(src, len);
        }
    }
    let run_wide = wide_fill()
        && ((fills.len() >= WIDE_FILL_MIN_COUNT && total_bytes >= WIDE_FILL_MIN_BYTES)
            || total_bytes >= WIDE_FILL_SINGLE_BYTES)
        && fills_disjoint(fills);
    if run_wide {
        let max_chunks = wide_fill_chunks().min((total_bytes / (32 << 10)).max(1));
        let chunks = partition_fills_by_bytes(fills, max_chunks);
        if chunks.len() >= 2 {
            let jobs: Vec<Box<dyn FnOnce() + Send>> = chunks
                .into_iter()
                .map(|chunk| {
                    Box::new(move || {
                        for (dst, src, len) in chunk {
                            copy_mapped(dst as *mut u8, src as *const u8, len, false);
                        }
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
        } else {
            for &(dst, src, len) in fills.iter() {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
        }
    } else {
        for &(dst, src, len) in fills.iter() {
            copy_mapped(dst as *mut u8, src as *const u8, len, false);
        }
    }
    preparing(0, started.elapsed().as_micros() as u64);
    preparing(1, total_bytes as u64);
    fills.clear();
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=1`; off by default).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (two or more copies
/// totalling at least 64 KiB, sorted largest-first for LPT load balancing)
/// are spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_REFRESH").is_some_and(|v| v == "1"))
}

/// Minimum combined refresh span bytes for parallel [`Batch::flush`].
const WIDE_REFRESH_MIN_BYTES: usize = 64 << 10;

/// Returns whether a [`Batch::flush`] with `count` distinct shadows and
/// `total_bytes` combined span bytes should dispatch across `wide_run`.
fn should_wide_refresh(wide_on: bool, count: usize, total_bytes: usize) -> bool {
    wide_on && count >= 2 && total_bytes >= WIDE_REFRESH_MIN_BYTES
}

/// Whether a range of a host copy brought up to date in a submission is
/// left alone for the rest of that submission, until a step of ours writes
/// title memory there (`VICTORY_REFRESH_ONCE=1`; off by default). After
/// upstream VIctory 6d021fe ("a copy's range brought up to date is not
/// copied again in the submission until a step writes title memory").
///
/// A list of dispatches records "refresh this range, shaders write it" for
/// every dispatch, and each of those refreshes was done on its own
/// (`Batch::flush_one`): a walk of the guards or a failed kernel asking
/// for a copy of view memory, some 1,800 to 2,700 a frame and 23 to 29 ms
/// (runs 222 and 225, "watches"), nearly all finding nothing new.
///
/// What changes: what the title's other threads write into a range while
/// its submission is being prepared is taken by the next submission, not
/// by a later step of this one, as `crate::watch::once` already has it for
/// write-watched memory. What this submission's lists read was written
/// before they were submitted. A title buffer copy of ours takes its
/// bytes out of what counts as up to date ([`Batch::title_wrote`]); any
/// other step that writes title memory takes everything out.
///
/// `VICTORY_REFRESH_ONCE=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log, so that one run measures both in
/// the same scene (two runs do not stand in the same scene at the same
/// second: runs 228 and 229).
fn refresh_once() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_REFRESH_ONCE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            let on = (now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "refreshes once a submission from now (VICTORY_REFRESH_ONCE=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
/// Per-shadow sorted, disjoint `[from, to)` refresh spans in a [`Batch`].
/// Overlapping and touching intervals coalesce; far-apart intervals stay
/// separate so refreshes do not copy or stall on unrelated gaps.
const MAX_BATCH_SPANS: usize = 64;

enum Spans {
    One((usize, usize)),
    Many(Vec<(usize, usize)>),
}

impl Spans {
    fn add(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        match self {
            Self::One((a, b)) => {
                if from <= *b && to >= *a {
                    *a = (*a).min(from);
                    *b = (*b).max(to);
                } else {
                    let v = if to < *a {
                        vec![(from, to), (*a, *b)]
                    } else {
                        vec![(*a, *b), (from, to)]
                    };
                    *self = Self::Many(v);
                }
            }
            Self::Many(v) => coalesce_span(v, from, to),
        }
    }

    fn as_slice(&self) -> &[(usize, usize)] {
        match self {
            Self::One(range) => std::slice::from_ref(range),
            Self::Many(ranges) => ranges.as_slice(),
        }
    }

    fn total_bytes(&self) -> usize {
        self.as_slice()
            .iter()
            .map(|&(from, to)| to.saturating_sub(from))
            .sum()
    }
}

fn coalesce_span(spans: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    if from >= to {
        return;
    }
    let pos = spans.partition_point(|&(_, end)| end < from);
    let mut lo = from;
    let mut hi = to;
    let mut erase_end = pos;
    while erase_end < spans.len() && spans[erase_end].0 <= hi {
        lo = lo.min(spans[erase_end].0);
        hi = hi.max(spans[erase_end].1);
        erase_end += 1;
    }
    if pos < erase_end {
        spans[pos] = (lo, hi);
        spans.drain(pos + 1..erase_end);
    } else {
        spans.insert(pos, (lo, hi));
    }
    if spans.len() > MAX_BATCH_SPANS
        && let Some(best) = (0..spans.len() - 1).min_by_key(|&i| spans[i + 1].0 - spans[i].1)
    {
        spans[best].1 = spans[best + 1].1;
        spans.remove(best + 1);
    }
}

#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, Spans)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: std::collections::HashMap<usize, usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// With [`refresh_once`], by copy (its address; the copy is held, so
    /// that the address names no other): the range of title addresses
    /// brought up to date in this submission that no step of ours has
    /// written since.
    current: std::collections::HashMap<
        usize,
        (std::sync::Arc<crate::shadow::Shadow>, usize, usize),
        std::hash::BuildHasherDefault<AddressHash>,
    >,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Texture stagings with `uploads == true` seen during `run_deferred`,
    /// avoiding a second full scan of all lists' deferred steps after `Signal`.
    uploaded: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// GPU-written ranges accepted by [`Shadow::writing`] during `run_deferred`,
    /// deduplicated per copy when taken via [`Batch::take_declared`].
    declared: Vec<Declared>,
    declared_index: std::collections::HashMap<usize, usize, std::hash::BuildHasherDefault<AddressHash>>,
}

impl Batch {
    fn note_gpu_write(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let range = (address, address + len);
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        match self.declared_index.get(&key) {
            Some(&at) => self.declared[at].1.push(range),
            None => {
                self.declared_index.insert(key, self.declared.len());
                self.declared.push((shadow.clone(), vec![range]));
            }
        }
    }

    fn take_declared(&mut self) -> Vec<Declared> {
        self.declared_index.clear();
        let mut back = std::mem::take(&mut self.declared);
        for (_, ranges) in &mut back {
            if ranges.len() > 1 {
                ranges.sort_unstable();
                ranges.dedup();
            }
        }
        back
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let Some(end) = address.checked_add(len).filter(|&e| e > address) else {
            return;
        };
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        // Brought up to date in this submission already ([`refresh_once`];
        // nothing is in `current` without the switch).
        if let Some(&(_, from, to)) = self.current.get(&key)
            && from <= address
            && end <= to
        {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, len as u64);
            }
            return;
        }
        match self.index.get(&key) {
            Some(&at) => self.refreshes[at].1.add(address, end),
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes
                    .push((shadow.clone(), Spans::One((address, end))));
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (&std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
        self.refreshes.iter().flat_map(|(shadow, spans)| {
            spans
                .as_slice()
                .iter()
                .map(move |&(from, to)| (shadow, from, to))
        })
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refreshes are done, the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let Some(at) = self.index.remove(&(std::sync::Arc::as_ptr(shadow) as usize)) else {
            return;
        };
        let (shadow, spans) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        for &(from, to) in spans.as_slice() {
            Self::update(&shadow, from, to);
            Self::brought(&mut self.current, &shadow, from, to);
        }
    }

    /// With [`refresh_once`]: `[from, to)` of `shadow`'s copy was brought
    /// up to date. Kept: its union with what was up to date if they touch,
    /// else the larger of the two.
    fn brought(
        current: &mut std::collections::HashMap<
            usize,
            (std::sync::Arc<crate::shadow::Shadow>, usize, usize),
            std::hash::BuildHasherDefault<AddressHash>,
        >,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        from: usize,
        to: usize,
    ) {
        if !refresh_once() || from >= to {
            return;
        }
        let entry = current
            .entry(std::sync::Arc::as_ptr(shadow) as usize)
            .or_insert_with(|| (shadow.clone(), from, to));
        if from <= entry.2 && entry.1 <= to {
            (entry.1, entry.2) = (entry.1.min(from), entry.2.max(to));
        } else if to - from > entry.2 - entry.1 {
            (entry.1, entry.2) = (from, to);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        preparing(0, us);
        preparing(1, copied as u64);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        // Where it writes is not known: no copy counts as up to date.
        self.current.clear();
    }

    /// [`Batch::title_written`] for a write of ours whose place is known
    /// (a title buffer copy into `[at, at + len)`): of a copy up to date
    /// in this submission that holds some of those bytes, the larger part
    /// beside them stays so.
    fn title_wrote(&mut self, at: usize, len: usize) {
        self.flush();
        self.textures.clear();
        let end = at + len;
        self.current.retain(|_, entry| {
            let (from, to) = (entry.1, entry.2);
            if end <= from || to <= at {
                return true;
            }
            let (left, right) = (at.saturating_sub(from), to.saturating_sub(end));
            if left >= right {
                entry.2 = from + left;
            } else {
                entry.1 = end;
            }
            entry.1 < entry.2
        });
    }

    fn flush(&mut self) {
        self.index.clear();
        if self.refreshes.is_empty() {
            return;
        }
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        let ahead = self.refreshes.len() > 1
            || self
                .refreshes
                .first()
                .is_some_and(|(_, s)| s.as_slice().len() > 1);
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = self
                .refreshes
                .iter()
                .filter(|(shadow, _)| !shadow.unwatched())
                .flat_map(|(_, spans)| spans.as_slice().iter().copied())
                .collect();
            crate::watch::ask_ahead(&ranges);
            preparing(0, started.elapsed().as_micros() as u64);
        }
        let total_span_bytes: usize = self
            .refreshes
            .iter()
            .map(|(_, spans)| spans.total_bytes())
            .sum();
        if should_wide_refresh(wide_refresh(), self.refreshes.len(), total_span_bytes) {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // Keep per-shadow exclusivity (all disjoint spans of one shadow
            // stay in one job) and sort largest-first (LPT scheduling) so the
            // biggest buffer starts immediately on the first worker while
            // smaller copies backfill remaining cores.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            let before = PREPARING[0].load(Ordering::Relaxed);
            self.refreshes
                .sort_unstable_by_key(|(_, spans)| std::cmp::Reverse(spans.total_bytes()));
            // Noted now: nothing reads the notes before the jobs are done.
            for (shadow, spans) in &self.refreshes {
                for &(from, to) in spans.as_slice() {
                    Self::brought(&mut self.current, shadow, from, to);
                }
            }
            let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                .refreshes
                .drain(..)
                .map(|(shadow, spans)| {
                    Box::new(move || {
                        for &(from, to) in spans.as_slice() {
                            Self::update(&shadow, from, to);
                        }
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_run(jobs);
            let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
            let over = summed.saturating_sub(started.elapsed().as_micros() as u64);
            PREPARING[0].fetch_sub(over, Ordering::Relaxed);
            crate::framelog::sub(crate::framelog::T::CopyUs, over);
        } else {
            for (shadow, spans) in self.refreshes.drain(..) {
                for &(from, to) in spans.as_slice() {
                    Self::update(&shadow, from, to);
                    Self::brought(&mut self.current, &shadow, from, to);
                }
            }
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    // Into host memory: in place first, if asked for (`crate::plaincopy`);
    // through the kernel if that is off or met a page that was not there.
    let whole = (!title_dst && crate::plaincopy::copy(dst, src, len)) || read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len, title_dst);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining, false);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !crate::plaincopy::copy_committed(to, at, n) && !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n, false);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize, title_dst: bool) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    let copy_piece = |at: *const u8, to: *mut u8, n: usize| {
        (!title_dst && crate::plaincopy::copy(to, at, n)) || read(at, to, n)
    };
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !copy_piece(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !copy_piece(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    // In place first, if asked for (`crate::plaincopy`): the destination is
    // memory of ours, as in `copy_mapped`.
    crate::plaincopy::copy(dst.as_mut_ptr(), src, dst.len()) || read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || crate::plaincopy::write(dst, src, len) || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        crate::fpsbox::set(fps);
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame"
        );
        // (Run 200 counted indirect compute records asking for 77,000 to
        // 98,000 groups in x, 0.2 to 0.8 a frame, cut to the 65,535 one
        // dispatch runs; the counter went when the clamp's budget moved to
        // the card's memory.)
        {
            let [texture_us, slots] = [0, 1].map(|i| crate::draw::HEAP_PASS_PARTS[i].swap(0, Ordering::Relaxed));
            tracing::info!(
                "the pass over the heap per frame (the frame line's \"heap buffers ... ms\" is both parts): {:.1} ms in its                  pass over textures picked by index; {:.0} buffer slots looked at by its pass over buffer views",
                texture_us as f64 / 1e3 / frames,
                slots as f64 / frames
            );
        }
        crate::shadow::exact_line(frames);
        crate::descriptor::sync_line(frames);
        late_line(frames);
        card_clock_line(frames, ms);
        crate::queue2::line(frames);
        crate::plaincopy::line(frames);
        if let Some(host) = host() {
            host.drain_messages();
        }
        {
            let held = crate::pipeline::WALKS_HELD.swap(0, Ordering::Relaxed);
            if held > 0 {
                tracing::info!(
                    "dispatches of programs that walk ray-tracing structures held back per frame (VICTORY_BVH_HOLD=1; the                      builder has not built them): {:.1}",
                    held as f64 / frames
                );
            }
        }
        crate::shadow::back_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let [own, asking, still, all] =
                [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
            let staging = STAGING_WAIT_US.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames;
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it), {staging:.1} ms waiting for the staging upload; \
                     the rest is reading, hashing and detiling"
                );
            }
            let [through, left] = [0, 1].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there)"
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        // Of it: looking at slots, spans and refreshes, textures (ms per
        // frame); slots looked at and copies brought up to date per frame.
        let [picked_look, picked_refresh, picked_tex] =
            [0, 1, 2].map(|i| crate::draw::HEAP_PASS_MICROS[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
        let [picked_slots, picked_copies] =
            [0, 1].map(|i| crate::draw::HEAP_BUFFER_VISITS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        let picked_mode = if crate::heap_pass::full_pass() { "full" } else { "marked" };
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        let readonly = crate::shadow::readonly_stats();
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
            );
            tracing::info!(
                "copies left out since the start because both sides were one host texture: {}",
                crate::list::SELF_COPIES.load(Ordering::Relaxed)
            );
            if crate::drvwatch::errors() > 0 {
                tracing::warn!(
                    "errors the graphics driver reported to the system since the start: {} (the paths under suspicion are off)",
                    crate::drvwatch::errors()
                );
            }
            crate::tiles::status();
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} dropped), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, dropped, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({dropped} dropped);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({})                      (VICTORY_PAGE_WRITES: unset or guards opens the guarded blocks, 0 counts only, 1, slots, all)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 { "told as well" } else { "counted only" },
                    if mode & 2 != 0 { "made due again" } else { "left as they were" }
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            let bar = crate::shadow::bar_copies();
            if crate::shadow::gpu_upload_copies().0 > 0 || bar[0] > 0 {
                tracing::info!(
                    "host copies in GPU upload heaps (VICTORY_GPU_UPLOAD_COPIES={}): {} copies {} MB ({} MB of declared shader writes read back over the bus)",
                    crate::shadow::gpu_upload_copies().0,
                    bar[0],
                    bar[1] >> 20,
                    bar[2] >> 20,
                );
            }
            let (ro_ranges, ro_bytes) = crate::title_hints::gpu_readonly_stats();
            if ro_ranges > 0 || readonly[0] > 0 {
                tracing::info!(
                    "xmem gpu-readonly: {} ranges ({} MB), skipped {} write declarations ({:.1} MB/frame) (VICTORY_XMEM_READONLY=0 turns it off)",
                    ro_ranges,
                    ro_bytes >> 20,
                    readonly[0],
                    readonly[1] as f64 / (1 << 20) as f64 / frames,
                );
            }
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms ({picked_mode}: {picked_slots:.0} slots looked at {picked_look:.1} ms, \
                 {picked_copies:.0} copies refreshed {picked_refresh:.1} ms, textures {picked_tex:.1} ms);                  texture slots written so far: {placed} placed, {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{}; depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let shader_note = match crate::pipeline::loading_status() {
            Some((done, total, _)) if total > 0 => {
                let pct = ((done.saturating_mul(100)) / total).min(99);
                let left = total.saturating_sub(done);
                format!(" | Shaders {pct}% ({left} left)")
            }
            _ => String::new(),
        };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame{shader_note}");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

/// Parses `VICTORY_WORLD_ENTRY_STREAM_MS`: defaults to `VICTORY_WORLD_ENTRY_WAIT_MS`
/// (8,000 ms, clamped `0..=15_000`) when `VICTORY_LOADING_WARM` is active; `0` disables.
fn parse_world_entry_stream_ms(
    stream_raw: Option<&str>,
    wait_raw: Option<&str>,
    loading_warm: bool,
) -> u64 {
    if let Some(ms) = stream_raw.and_then(|v| v.trim().parse::<u64>().ok()) {
        return ms.min(15_000);
    }
    if !loading_warm {
        return 0;
    }
    wait_raw
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(8_000, |v| v.min(15_000))
}

/// Budget in milliseconds to hold the loading-screen -> first 3D world
/// transition while DirectStorage reads drain into RAM (`VICTORY_WORLD_ENTRY_STREAM_MS`).
fn world_entry_stream_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| {
        let loading_warm = std::env::var_os("VICTORY_LOADING_WARM").is_none_or(|v| v != "0");
        parse_world_entry_stream_ms(
            std::env::var("VICTORY_WORLD_ENTRY_STREAM_MS").ok().as_deref(),
            std::env::var("VICTORY_WORLD_ENTRY_WAIT_MS").ok().as_deref(),
            loading_warm,
        )
    })
}

/// Outcome of a world-entry DirectStorage drain hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamDrainOutcome {
    waited_ms: u64,
    timed_out: bool,
    dstorage_pending: usize,
    dstorage_completed_ops: u64,
    dstorage_completed_bytes: u64,
}

/// Polls DirectStorage backlog in 10 ms slices up to `budget` until `pending`
/// stays 0 for 150 ms (or `completed_ops` stops advancing for 250 ms while
/// `pending == 0`), returning `None` immediately when `budget` is zero or no
/// DirectStorage activity is present.
fn drain_dstorage_until_settled(
    budget: std::time::Duration,
    mut elapsed: impl FnMut() -> std::time::Duration,
    mut sleep: impl FnMut(std::time::Duration),
    mut pending_fn: impl FnMut() -> usize,
    mut completed_fn: impl FnMut() -> (u64, u64),
) -> Option<StreamDrainOutcome> {
    use std::time::Duration;
    const POLL_SLICE: Duration = Duration::from_millis(10);
    const ZERO_DEBOUNCE: Duration = Duration::from_millis(150);
    const QUIET_OPS_DEBOUNCE: Duration = Duration::from_millis(250);

    if budget.is_zero() {
        return None;
    }
    let mut pending = pending_fn();
    let (mut last_ops, mut last_bytes) = completed_fn();
    if pending == 0 && last_ops == 0 {
        return None;
    }

    let mut quiet_since: Option<Duration> = None;
    let mut saw_ops_advance_at_zero = false;
    let mut timed_out = true;

    loop {
        let now = elapsed();
        if pending > 0 {
            quiet_since = None;
        } else {
            let since = *quiet_since.get_or_insert(now);
            let settle = if saw_ops_advance_at_zero {
                QUIET_OPS_DEBOUNCE
            } else {
                ZERO_DEBOUNCE
            };
            if now.saturating_sub(since) >= settle {
                timed_out = false;
                break;
            }
        }
        let rem = budget.saturating_sub(now);
        if rem.is_zero() {
            break;
        }
        sleep(POLL_SLICE.min(rem));
        pending = pending_fn();
        let (ops, bytes) = completed_fn();
        if ops != last_ops {
            if pending == 0 && quiet_since.is_some() {
                saw_ops_advance_at_zero = true;
            }
            last_ops = ops;
            last_bytes = bytes;
            quiet_since = if pending == 0 { Some(elapsed()) } else { None };
        }
    }

    Some(StreamDrainOutcome {
        waited_ms: elapsed().as_millis() as u64,
        timed_out,
        dstorage_pending: pending,
        dstorage_completed_ops: last_ops,
        dstorage_completed_bytes: last_bytes,
    })
}

/// On the loading-screen -> first 3D world transition (`delta >= 60` draws,
/// matching `pipeline::make::frame_drew`), holds up to `VICTORY_WORLD_ENTRY_STREAM_MS`
/// while active DirectStorage disk/decompression streams finish draining into RAM.
fn world_entry_stream_hold(drawn: u64) {
    static WORLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static LAST_DRAWN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let delta = drawn.saturating_sub(LAST_DRAWN.swap(drawn, Ordering::Relaxed));
    if WORLD.load(Ordering::Relaxed) || delta < 60 {
        return;
    }
    if WORLD.swap(true, Ordering::AcqRel) {
        return;
    }
    let budget_ms = world_entry_stream_ms();
    let budget = std::time::Duration::from_millis(budget_ms);
    let started = std::time::Instant::now();
    let Some(outcome) = drain_dstorage_until_settled(
        budget,
        || started.elapsed(),
        std::thread::sleep,
        crate::title_hints::dstorage_pending,
        crate::title_hints::dstorage_completed,
    ) else {
        return;
    };
    tracing::info!(
        waited_ms = outcome.waited_ms,
        timed_out = outcome.timed_out,
        dstorage_pending = outcome.dstorage_pending,
        dstorage_completed_ops = outcome.dstorage_completed_ops,
        dstorage_completed_mb = outcome.dstorage_completed_bytes >> 20,
        pipelines_waiting = crate::pipeline::making(),
        "world-entry DirectStorage streaming wait {} (VICTORY_WORLD_ENTRY_STREAM_MS={budget_ms})",
        if outcome.timed_out { "timed out" } else { "complete" },
    );
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    #[allow(dead_code)]
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        self.present_over(texture, state, None, frame)
    }

    /// [`Self::present`], with the title's second plane laid over `texture`
    /// when there is one (`crate::compose`). Its state is left as it was.
    pub(crate) fn present_over(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        over: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        let showing = std::time::Instant::now();
        let shown = self.present_now(texture, state, over, frame);
        crate::framelog::add(crate::framelog::T::PresentShowUs, showing.elapsed().as_micros() as u64);
        world_entry_stream_hold(crate::draw::drawn());
        shown
    }

    fn present_now(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        over: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let read_now = crate::capture::wanted(frame) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            let list = &copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            let laid = over.and_then(|(ui, ui_state)| {
                let composed = crate::compose::record(&self.device, list, texture, ui, ui_state)?;
                copy.keep(&composed);
                copy.keep(ui);
                Some(composed)
            });
            list.CopyResource(&back, laid.as_ref().unwrap_or(texture));
            // The frame rate, into the corner of what is shown (not of the
            // title's own frame, which the lines below read back).
            crate::fpsbox::draw(self, list, &back, (width, height, format));
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() { None } else { crate::capture::probe(&self.device, list, texture) };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            copy.close();
            match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                }
                None => self.execute(&[copy]),
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// Whether the card and the runtime have GPU upload heaps (the card's
    /// memory mapped for the CPU: resizable BAR), asked once and said.
    fn gpu_upload_heap(&self) -> bool {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| {
            let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: a structure of the size passed.
            let asked = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut o16).cast(),
                    size_of_val(&o16) as u32,
                )
            };
            let has = asked.is_ok() && o16.GPUUploadHeapSupported.as_bool();
            tracing::info!(
                asked = asked.is_ok(),
                has,
                "GPU upload heaps (the card's memory mapped for the CPU; resizable BAR): {}",
                if has {
                    "supported (VICTORY_GPU_UPLOAD_COPIES can place copies there)"
                } else {
                    "not supported; falling back to PC shared buffers"
                }
            );
            has
        })
    }

    /// [`Self::shared_buffer`] in the card's own memory: a buffer in a GPU
    /// upload heap (the card's memory mapped for the CPU through its window
    /// on the bus, write-combined), shaders may write it, in the common
    /// state. `None` where the card has no such heap
    /// ([`Self::gpu_upload_heap`]) or it could not be made. The CPU writes
    /// it as fast as upload memory; reading it back crosses the bus,
    /// uncached (`crate::shadow::gpu_upload_copies`).
    pub(crate) fn card_shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.gpu_upload_heap() {
            return None;
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                static WARNED: AtomicU32 = AtomicU32::new(0);
                if WARNED.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::warn!(error = %e, size, "no shared buffer in a GPU upload heap: in the PC's memory instead");
                }
                None
            }
        }
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        let vram_idle_ms = |memory| super::vram_idle_for(memory, false);
        let keeps = |memory| super::vram_idle_for(memory, true);
        let gib = 1024 * 1024 * 1024;
        // Extended retention (`VICTORY_TEXTURE_KEEP` on, default) on <= 12 GiB
        assert_eq!(keeps(Some([100, 67, 0, 0])), 600_000);
        assert_eq!(keeps(Some([100, 68, 0, 0])), 30_000);
        assert_eq!(keeps(Some([100, 74, 0, 0])), 30_000);
        assert_eq!(keeps(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(keeps(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(keeps(None), 5_000);
        assert_eq!(keeps(Some([0, 100, 0, 0])), 5_000);
        // Extended retention on > 12 GiB preserves tiered pressure thresholds
        assert_eq!(keeps(Some([15 * gib, 8 * gib, 0, 0])), 600_000);
        assert_eq!(keeps(Some([15 * gib, 10 * gib + gib / 2, 0, 0])), 30_000);
        assert_eq!(keeps(Some([15 * gib, 12 * gib, 0, 0])), 5_000);
        assert_eq!(keeps(Some([15 * gib, 13 * gib, 0, 0])), 2_500);
        assert_eq!(keeps(Some([15 * gib, 14 * gib + gib / 2, 0, 0])), 1_000);
        // `VICTORY_TEXTURE_KEEP=0` preserves original tiers
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
        // High-VRAM adapter (> 12 GiB, e.g. 16 GiB RTX 5070 Ti): retains
        // textures longer while multi-GB headroom exists.
        assert_eq!(vram_idle_ms(Some([15 * gib, 8 * gib, 0, 0])), 15_000);
        assert_eq!(vram_idle_ms(Some([15 * gib, 12 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([15 * gib, 13 * gib, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([15 * gib, 14 * gib + gib / 2, 0, 0])), 1_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Abundant (>= 16 GiB)
        assert_eq!(ram_idle_ms(Some(32 * gib)), 8_000);
        assert_eq!(ram_idle_ms(Some(16 * gib)), 8_000);
        // Normal (4..16 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 5_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 5_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 3_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 3_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 5_000);
    }

    #[test]
    fn coalesce_piece_ranges_merges_overlapping_and_preserves_gaps() {
        use super::coalesce_piece_ranges;
        let mut r = vec![(100, 300), (0, 100), (200, 400), (600, 700), (650, 680)];
        coalesce_piece_ranges(&mut r);
        assert_eq!(r, vec![(0, 400), (600, 700)]);
        let mut empty: Vec<(usize, usize)> = Vec::new();
        coalesce_piece_ranges(&mut empty);
        assert!(empty.is_empty());
        let mut single = vec![(16, 32)];
        coalesce_piece_ranges(&mut single);
        assert_eq!(single, vec![(16, 32)]);
    }

    #[test]
    fn coalesced_spans_keep_disjoint_ranges_separate_and_merge_overlaps() {
        use super::{MAX_BATCH_SPANS, Spans};

        let mut spans = Spans::One((100, 160));
        // Subset and touching/overlapping merges stay a single span.
        spans.add(110, 140);
        assert_eq!(spans.as_slice(), &[(100, 160)]);
        spans.add(160, 200);
        assert_eq!(spans.as_slice(), &[(100, 200)]);
        spans.add(80, 120);
        assert_eq!(spans.as_slice(), &[(80, 200)]);
        spans.add(300, 300); // empty span ignored
        assert_eq!(spans.as_slice(), &[(80, 200)]);

        // Disjoint out-of-order inserts stay sorted and separate.
        spans.add(10, 30);
        spans.add(500, 560);
        spans.add(300, 360);
        assert_eq!(
            spans.as_slice(),
            &[(10, 30), (80, 200), (300, 360), (500, 560)]
        );

        // Bridging adjacent/overlapping spans coalesces across them.
        spans.add(200, 300);
        assert_eq!(spans.as_slice(), &[(10, 30), (80, 360), (500, 560)]);

        // Exceeding MAX_BATCH_SPANS merges only the smallest gap.
        let mut bounded = Spans::One((0, 10));
        for i in 1..MAX_BATCH_SPANS {
            let start = i * 100;
            bounded.add(start, start + 10);
        }
        assert_eq!(bounded.as_slice().len(), MAX_BATCH_SPANS);
        // Insert one more very close to slot 1 ([100, 110)) so gap (110..120 = 10) is smallest.
        bounded.add(120, 125);
        assert_eq!(bounded.as_slice().len(), MAX_BATCH_SPANS);
        assert!(bounded.as_slice().contains(&(100, 125)));
    }

    #[test]
    fn stale_under_flights_and_batch_flush_ignore_unrequested_gaps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        let Some(host) = super::host() else {
            return;
        };
        const PAGE: usize = 64 << 10;
        let bytes = 4 * PAGE;
        // Allocate non-write-watched memory so Shadow tracks refreshes across the full span.
        // SAFETY: plain private anonymous RW pages freed before return.
        let start = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(start, 0);
        // SAFETY: zero-initialized allocation of `bytes`.
        unsafe { std::ptr::write_bytes(start as *mut u8, 0, bytes) };
        crate::shadow::register(start, start as u64, bytes as u64);
        let shadow = crate::shadow::find(start as u64, 1).unwrap();

        // Populate guest pages 0, 2, and 3 AFTER creating the shadow so only refreshed ranges update.
        // SAFETY: distinct pages inside the 4-page allocation.
        unsafe {
            std::ptr::write_bytes(start as *mut u8, 0x11, 64);
            std::ptr::write_bytes((start + 2 * PAGE) as *mut u8, 0x22, 64);
            std::ptr::write_bytes((start + 3 * PAGE) as *mut u8, 0x33, 64);
        }
        // Mark page 1 as an in-flight shader write.
        shadow.writing(start + PAGE, 64);

        let list1 = host.new_list().unwrap();
        let list2 = host.new_list().unwrap();
        let list3 = host.new_list().unwrap();
        list1.defer_refresh(&shadow, start, 64);
        list2.defer_refresh(&shadow, start + 3 * PAGE, 64);
        list3.defer_refresh(&shadow, start + PAGE, 64);

        // Disjoint refreshes on pages 0 and 3 do not intersect page 1's pending write.
        assert!(!super::stale_under_flights(&[&list1, &list2]));
        // A refresh covering page 1 itself does trigger stale detection.
        assert!(super::stale_under_flights(&[&list1, &list3]));

        // Flushing disjoint spans on pages 0 and 3 copies those endpoints without copying page 2.
        let mut batch = Batch::default();
        batch.add(&shadow, start, 64);
        batch.add(&shadow, start + 3 * PAGE, 64);
        batch.flush();
        let (p0, p2, p3) = (
            shadow.byte_at(0),
            shadow.byte_at(2 * PAGE),
            shadow.byte_at(3 * PAGE),
        );
        assert_eq!(p0, 0x11);
        assert_eq!(p2, 0x00, "unrequested middle page must not be copied");
        assert_eq!(p3, 0x33);

        drop(batch);
        drop(list1);
        drop(list2);
        drop(list3);
        shadow.copy_back_done(&[(start + PAGE, start + PAGE + 64)]);
        drop(shadow);
        crate::shadow::unregister(start, start as u64);
        assert_ne!(unsafe { VirtualFree(start as _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn world_entry_stream_ms_parsing_defaults_and_overrides() {
        use super::parse_world_entry_stream_ms;
        // Inherits `WORLD_ENTRY_WAIT_MS` (default 8000) when loading warm is on.
        assert_eq!(parse_world_entry_stream_ms(None, None, true), 8_000);
        assert_eq!(parse_world_entry_stream_ms(None, Some("5000"), true), 5_000);
        assert_eq!(parse_world_entry_stream_ms(None, Some("0"), true), 0);
        assert_eq!(parse_world_entry_stream_ms(None, Some("30000"), true), 15_000);
        // `VICTORY_LOADING_WARM=0` disables default stream hold unless explicitly set.
        assert_eq!(parse_world_entry_stream_ms(None, None, false), 0);
        assert_eq!(parse_world_entry_stream_ms(None, Some("5000"), false), 0);
        assert_eq!(parse_world_entry_stream_ms(Some("4000"), None, false), 4_000);
        // Explicit `VICTORY_WORLD_ENTRY_STREAM_MS` overrides `WORLD_ENTRY_WAIT_MS`.
        assert_eq!(parse_world_entry_stream_ms(Some("0"), Some("8000"), true), 0);
        assert_eq!(parse_world_entry_stream_ms(Some(" 3500 "), Some("8000"), true), 3_500);
        assert_eq!(parse_world_entry_stream_ms(Some("99999"), None, true), 15_000);
    }

    #[test]
    fn world_entry_dstorage_drain_settles_chains_and_times_out() {
        use super::drain_dstorage_until_settled;
        use std::cell::Cell;
        use std::time::Duration;

        // 1) Zero budget -> no wait.
        assert!(
            drain_dstorage_until_settled(
                Duration::ZERO,
                || Duration::ZERO,
                |_| unreachable!(),
                || 5,
                || (10, 1 << 20),
            )
            .is_none()
        );

        // 2) No DirectStorage activity (pending == 0 && completed_ops == 0) -> no wait.
        assert!(
            drain_dstorage_until_settled(
                Duration::from_millis(8_000),
                || Duration::ZERO,
                |_| unreachable!(),
                || 0,
                || (0, 0),
            )
            .is_none()
        );

        // 3) Pending drains at 100 ms, then stays quiet -> settles after 150 ms debounce (250 ms total).
        let now = Cell::new(Duration::ZERO);
        let outcome = drain_dstorage_until_settled(
            Duration::from_millis(8_000),
            || now.get(),
            |d| now.set(now.get() + d),
            || if now.get() < Duration::from_millis(100) { 4 } else { 0 },
            || {
                if now.get() < Duration::from_millis(100) {
                    (10, 10 << 20)
                } else {
                    (14, 14 << 20)
                }
            },
        )
        .expect("should wait until settled");
        assert!(!outcome.timed_out);
        assert_eq!(outcome.waited_ms, 250);
        assert_eq!(outcome.dstorage_pending, 0);
        assert_eq!(outcome.dstorage_completed_ops, 14);
        assert_eq!(outcome.dstorage_completed_bytes, 14 << 20);

        // 4) Chained batch arrives 50 ms after first batch finishes -> resets debounce and settles after second batch.
        let now = Cell::new(Duration::ZERO);
        let outcome = drain_dstorage_until_settled(
            Duration::from_millis(8_000),
            || now.get(),
            |d| now.set(now.get() + d),
            || {
                let ms = now.get().as_millis();
                if ms < 100 || (150..200).contains(&ms) { 3 } else { 0 }
            },
            || {
                let ms = now.get().as_millis();
                if ms < 100 {
                    (10, 10 << 20)
                } else if ms < 200 {
                    (13, 13 << 20)
                } else {
                    (16, 16 << 20)
                }
            },
        )
        .expect("should wait through chained batch");
        assert!(!outcome.timed_out);
        assert_eq!(outcome.waited_ms, 350);
        assert_eq!(outcome.dstorage_pending, 0);
        assert_eq!(outcome.dstorage_completed_ops, 16);

        // 5) Rapid micro-batches advancing `completed_ops` while `pending == 0` settle after 250 ms quiet period.
        let now = Cell::new(Duration::ZERO);
        let outcome = drain_dstorage_until_settled(
            Duration::from_millis(8_000),
            || now.get(),
            |d| now.set(now.get() + d),
            || 0,
            || {
                let ms = now.get().as_millis();
                if ms < 60 {
                    (20, 20 << 20)
                } else {
                    (25, 25 << 20)
                }
            },
        )
        .expect("should wait 250 ms after ops stop advancing");
        assert!(!outcome.timed_out);
        assert_eq!(outcome.waited_ms, 310);
        assert_eq!(outcome.dstorage_completed_ops, 25);

        // 6) Budget timeout while pending remains > 0.
        let now = Cell::new(Duration::ZERO);
        let outcome = drain_dstorage_until_settled(
            Duration::from_millis(500),
            || now.get(),
            |d| now.set(now.get() + d),
            || 7,
            || (100, 128 << 20),
        )
        .expect("should time out at budget");
        assert!(outcome.timed_out);
        assert_eq!(outcome.waited_ms, 500);
        assert_eq!(outcome.dstorage_pending, 7);
    }

    #[test]
    fn readonly_gpu_writes_excluded_from_list_and_batch_declared() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        struct ClearGuard;
        impl Drop for ClearGuard {
            fn drop(&mut self) {
                crate::title_hints::clear_gpu_readonly_ranges();
            }
        }
        crate::title_hints::clear_gpu_readonly_ranges();
        let _clear = ClearGuard;
        let _ = crate::shadow::readonly_stats();

        let Some(host) = super::host() else {
            return;
        };
        const LEN: usize = 2 * (64 << 10);
        // SAFETY: private anonymous RW allocation freed before returning.
        let start = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(start, 0);
        // SAFETY: committed allocation of `LEN` bytes.
        unsafe { std::ptr::write_bytes(start as *mut u8, 0, LEN) };
        crate::shadow::register(start, start as u64, LEN as u64);
        let shadow = crate::shadow::find(start as u64, 1).unwrap();

        // Mark first page RO, leave second page RW.
        crate::title_hints::set_gpu_memory_readonly(start as u64, 64 << 10, true);
        let list = host.new_list().unwrap();
        // RO write skipped at `gpu_writes` before deferring.
        shadow.gpu_writes(&list, start as u64, 4096);
        assert_eq!(crate::shadow::readonly_stats(), [1, 4096]);
        // RW write accepted and deferred into the list.
        let rw_addr = start + (64 << 10);
        shadow.gpu_writes(&list, rw_addr as u64, 4096);
        assert_eq!(crate::shadow::readonly_stats(), [0, 0]);

        let mut batch = Batch::default();
        list.run_deferred(&mut batch);
        let declared = batch.take_declared();
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].1, vec![(rw_addr, rw_addr + 4096)]);
        shadow.copy_back_done(&declared[0].1);

        // Even if a write was deferred BEFORE its range became RO, `run_deferred`
        // skips it and excludes it from `batch.take_declared()`.
        let list2 = host.new_list().unwrap();
        shadow.gpu_writes(&list2, rw_addr as u64, 2048);
        crate::title_hints::set_gpu_memory_readonly(rw_addr as u64, 64 << 10, true);
        let mut batch2 = Batch::default();
        list2.run_deferred(&mut batch2);
        assert!(batch2.take_declared().is_empty());
        assert!(!shadow.writes_pending());
        assert_eq!(crate::shadow::readonly_stats(), [1, 2048]);

        drop(list2);
        drop(list);
        drop(shadow);
        crate::shadow::unregister(start, start as u64);
        assert_ne!(unsafe { VirtualFree(start as _, 0, MEM_RELEASE) }, 0);
        crate::title_hints::clear_gpu_readonly_ranges();
    }

    #[test]
    fn fills_disjoint_accepts_disjoint_and_rejects_overlaps() {
        use super::fills_disjoint;

        assert!(fills_disjoint(&[]));
        assert!(fills_disjoint(&[(100, 0, 64)]));
        // Strictly ascending and touching half-open intervals.
        assert!(fills_disjoint(&[(100, 1, 64), (164, 2, 64), (300, 3, 128)]));
        // Zero-length fills never write memory and do not conflict.
        assert!(fills_disjoint(&[(100, 1, 200), (150, 2, 0), (300, 3, 64)]));
        // Unsorted disjoint intervals (e.g. multiple upload chunks or lists).
        assert!(fills_disjoint(&[
            (2000, 10, 256),
            (100, 20, 256),
            (800, 30, 512),
            (356, 40, 444),
        ]));

        // Rejects ascending overlap, unsorted overlap, identical destination,
        // containment, and address overflow.
        assert!(!fills_disjoint(&[(100, 1, 64), (163, 2, 64)]));
        assert!(!fills_disjoint(&[(200, 1, 64), (150, 2, 64)]));
        assert!(!fills_disjoint(&[(100, 1, 64), (100, 2, 64)]));
        assert!(!fills_disjoint(&[(100, 1, 200), (120, 2, 16)]));
        assert!(!fills_disjoint(&[(120, 1, 16), (100, 2, 200)]));
        assert!(!fills_disjoint(&[(usize::MAX - 8, 1, 16)]));
    }

    #[test]
    fn partition_fills_by_bytes_balances_giant_and_tiny_fills() {
        use super::partition_fills_by_bytes;

        assert!(partition_fills_by_bytes(&[], 4).is_empty());
        assert!(partition_fills_by_bytes(&[(100, 200, 0), (300, 400, 0)], 4).is_empty());

        let coalesce = |items: &[(usize, usize, usize)]| -> Vec<(usize, usize, usize)> {
            let mut merged: Vec<(usize, usize, usize)> = Vec::new();
            for &(dst, src, len) in items {
                if len == 0 {
                    continue;
                }
                if let Some(last) = merged.last_mut()
                    && last.0 + last.2 == dst
                    && last.1 + last.2 == src
                {
                    last.2 += len;
                } else {
                    merged.push((dst, src, len));
                }
            }
            merged
        };

        let assert_exact_coverage = |input: &[(usize, usize, usize)], chunks: &[Vec<(usize, usize, usize)>]| {
            // Reconstruct merged `(dst, src, len)` spans from all chunks and verify
            // exact byte-for-byte identity with the non-empty input fills.
            let got: Vec<(usize, usize, usize)> = chunks.iter().flatten().copied().collect();
            assert!(got.iter().all(|&(_, _, len)| len > 0));
            assert_eq!(coalesce(&got), coalesce(input));
        };

        // 1) One 4 MiB fill + 200 tiny 256 B fills in three orderings across 4 and 6 workers.
        const GIANT: usize = 4 << 20;
        const TINY_COUNT: usize = 200;
        const TINY_LEN: usize = 256;

        let build = |giant_pos: usize| {
            let mut fills = Vec::with_capacity(TINY_COUNT + 1);
            let mut dst = 0x1000_0000usize;
            let mut src = 0x2000_0000usize;
            for i in 0..=TINY_COUNT {
                let len = if i == giant_pos { GIANT } else { TINY_LEN };
                fills.push((dst, src, len));
                dst += len.next_multiple_of(256);
                src += len + 64; // Distinct src-dst offset per fill so only split pieces coalesce.
            }
            fills
        };

        for giant_pos in [0, TINY_COUNT / 2, TINY_COUNT] {
            let fills = build(giant_pos);
            let total: usize = fills.iter().map(|&(_, _, l)| l).sum();
            for max_chunks in [4usize, 6] {
                let chunks = partition_fills_by_bytes(&fills, max_chunks);
                assert_eq!(chunks.len(), max_chunks);
                assert_exact_coverage(&fills, &chunks);
                let avg = total / max_chunks;
                for chunk in &chunks {
                    let bytes: usize = chunk.iter().map(|&(_, _, l)| l).sum();
                    assert!(
                        bytes >= avg * 9 / 10 && bytes <= avg * 11 / 10,
                        "unbalanced chunk {bytes} vs avg {avg} (giant_pos={giant_pos}, max_chunks={max_chunks})"
                    );
                }
            }
        }

        // 2) Uniform small fills (16 x 8 KiB) partition cleanly without splitting.
        let uniform: Vec<_> = (0..16)
            .map(|i| (0x10000 + i * 8192, 0x80000 + i * 8192, 8192))
            .collect();
        let chunks = partition_fills_by_bytes(&uniform, 4);
        assert_eq!(chunks.len(), 4);
        assert_exact_coverage(&uniform, &chunks);
        for chunk in &chunks {
            let bytes: usize = chunk.iter().map(|&(_, _, l)| l).sum();
            assert_eq!(bytes, 32768);
        }

        // 3) Single chunk returns all non-empty fills intact.
        let single = partition_fills_by_bytes(&uniform, 1);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], uniform);
    }

    #[test]
    fn should_wide_refresh_threshold_and_spans_total_bytes() {
        use super::{Spans, WIDE_REFRESH_MIN_BYTES, should_wide_refresh};

        let mut spans = Spans::One((1000, 1000 + 4096));
        assert_eq!(spans.total_bytes(), 4096);
        spans.add(100_000, 100_000 + 8192);
        assert_eq!(spans.total_bytes(), 4096 + 8192);

        assert!(!should_wide_refresh(false, 4, 1 << 20));
        assert!(!should_wide_refresh(true, 0, 0));
        assert!(!should_wide_refresh(true, 1, 174 << 20));
        assert!(!should_wide_refresh(true, 2, WIDE_REFRESH_MIN_BYTES - 1));
        assert!(should_wide_refresh(true, 2, WIDE_REFRESH_MIN_BYTES));
        assert!(should_wide_refresh(true, 3, 10 << 20));
    }

    #[test]
    fn flush_fills_parallel_disjoint_and_serial_overlap() {
        use super::flush_fills;

        // Disjoint batch >= 8 fills and >= 128 KiB exercises `wide_run` directly.
        const PIECE: usize = 16 << 10;
        const COUNT: usize = 16;
        let src: Vec<u8> = (0..COUNT * PIECE).map(|i| ((i * 13 + 7) & 0xFF) as u8).collect();
        let mut dst = vec![0u8; COUNT * PIECE];
        let mut fills: Vec<(usize, usize, usize)> = (0..COUNT)
            .map(|i| {
                // Reverse order to also exercise unsorted disjoint verification.
                let idx = COUNT - 1 - i;
                (
                    dst.as_mut_ptr() as usize + idx * PIECE,
                    src.as_ptr() as usize + idx * PIECE,
                    PIECE,
                )
            })
            .collect();
        flush_fills(&mut fills);
        assert!(fills.is_empty());
        assert_eq!(dst, src);

        // Overlapping fills to the same destination must stay strictly serial (last wins).
        let sources: Vec<Vec<u8>> = (0..16u8).map(|k| vec![0x40 + k; PIECE]).collect();
        let mut overlap_dst = vec![0u8; PIECE];
        let mut overlap_fills: Vec<(usize, usize, usize)> = sources
            .iter()
            .map(|s| (overlap_dst.as_mut_ptr() as usize, s.as_ptr() as usize, PIECE))
            .collect();
        flush_fills(&mut overlap_fills);
        assert!(overlap_fills.is_empty());
        assert!(overlap_dst.iter().all(|&b| b == 0x4F));
    }

    #[test]
    fn run_deferred_interleaved_fills_refreshes_and_copy_barrier_order() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        let Some(host) = super::host() else {
            return;
        };

        const PIECE: usize = 16 << 10;
        const COUNT: usize = 16;
        const TOTAL: usize = PIECE * COUNT; // 256 KiB (> 128 KiB wide-fill threshold)

        // SAFETY: private anonymous RW pages freed before returning.
        let title_src = unsafe {
            VirtualAlloc(std::ptr::null(), TOTAL, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        } as *mut u8;
        assert!(!title_src.is_null());
        // Pre-copy title bytes: 0xAA; post-copy patch bytes: 0x55.
        // SAFETY: committed allocation of `TOTAL` bytes.
        unsafe { std::ptr::write_bytes(title_src, 0xAA, TOTAL) };
        let patch = vec![0x55u8; TOTAL];

        // Register a shadow over `title_src` to verify interleaved `Step::Refresh`
        // commutes with `Step::Fill` and flushes before `Step::Copy`.
        crate::shadow::register(title_src as usize, title_src as u64, TOTAL as u64);
        let shadow = crate::shadow::find(title_src as u64, 1).unwrap();
        // Overwrite title_src with 0xCC after shadow registration so a refresh must
        // actually copy `0xCC` before `Step::Copy` overwrites it with `0x55`.
        // SAFETY: committed allocation of `TOTAL` bytes.
        unsafe { std::ptr::write_bytes(title_src, 0xCC, TOTAL) };

        let mut pre_dst = vec![0u8; TOTAL];
        let mut post_dst = vec![0u8; TOTAL];

        let list = host.new_list().unwrap();
        // Segment 1: 16 fills interleaved with refreshes BEFORE `Step::Copy`.
        for i in 0..COUNT {
            // SAFETY: distinct `PIECE`-byte slices inside `pre_dst` and `title_src`.
            unsafe {
                list.defer_fill(pre_dst.as_mut_ptr().add(i * PIECE), title_src.add(i * PIECE), PIECE);
            }
            list.defer_refresh(&shadow, title_src as usize + i * PIECE, PIECE);
        }
        // Barrier: `Step::Copy` overwrites `title_src` with `0x55`.
        // Must flush Segment 1 fills AND refreshes BEFORE mutating `title_src`.
        list.defer_copy(title_src, patch.as_ptr(), TOTAL);
        // Segment 2: 16 fills interleaved with refreshes AFTER `Step::Copy`.
        for i in 0..COUNT {
            // SAFETY: distinct `PIECE`-byte slices inside `post_dst` and `title_src`.
            unsafe {
                list.defer_fill(post_dst.as_mut_ptr().add(i * PIECE), title_src.add(i * PIECE), PIECE);
            }
            list.defer_refresh(&shadow, title_src as usize + i * PIECE, PIECE);
        }

        let mut batch = Batch::default();
        list.run_deferred(&mut batch);

        // Segment 1 fills observed pre-copy bytes (`0xCC`).
        assert!(
            pre_dst.iter().all(|&b| b == 0xCC),
            "fills before Step::Copy must observe pre-copy bytes"
        );
        // Segment 2 fills observed post-copy bytes (`0x55`).
        assert!(
            post_dst.iter().all(|&b| b == 0x55),
            "fills after Step::Copy must observe post-copy bytes"
        );
        // Shadow refresh from Segment 1 flushed before `Step::Copy` -> currently holds `0xCC`;
        // flushing `batch` applies Segment 2's refresh -> updates to `0x55`.
        assert_eq!(shadow.byte_at(0), 0xCC);
        batch.flush();
        assert_eq!(shadow.byte_at(0), 0x55);
        assert_eq!(shadow.byte_at(TOTAL - 1), 0x55);

        drop(batch);
        drop(list);
        drop(shadow);
        crate::shadow::unregister(title_src as usize, title_src as u64);
        assert_ne!(unsafe { VirtualFree(title_src as _, 0, MEM_RELEASE) }, 0);
    }
}

#[cfg(test)]
mod fill_on_card_tests {
    use super::*;

    /// A readback buffer of `len` bytes.
    fn readback(host: &Host, len: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
                .unwrap();
        }
        buffer.unwrap()
    }

    /// What a flight's shaders wrote to a host copy (bytes the title's memory
    /// does not have) reaches constants filled on the card, byte for byte,
    /// for two fills in one list.
    #[test]
    fn constants_filled_on_the_card_read_what_the_card_wrote() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        const LEN: usize = 1 << 16;
        // Title memory: a buffer the title fills with 0xAA.
        let title: &'static mut [u8] = Box::leak(vec![0xAAu8; LEN].into_boxed_slice());
        let start = title.as_ptr() as u64;
        let resource = 0x7e57_0001usize;
        crate::shadow::register(resource, start, LEN as u64);
        let shadow = crate::shadow::find(start + 64, 512).expect("a host copy");
        let offset = shadow.offset_in(start).expect("the copy starts at the buffer");
        // A "flight" writes a pattern into the copy on the card.
        let pattern: Vec<u8> = (0..1024u32).map(|i| (i * 7 + 3) as u8).collect();
        let (upload, data) = host.upload_buffer(1024).unwrap();
        // SAFETY: the upload buffer holds 1024 mapped bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 1024) };
        let flight = host.new_list().unwrap();
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe { flight.list.CopyBufferRegion(&shadow.buffer, offset + 64, &upload, 0, 1024) };
        flight.close();
        host.execute(&[&flight]);
        // The next list fills two constant buffers from it on the card.
        let list = host.new_list().unwrap();
        let first = list.card_constants(&shadow, start + 64, 512).expect("room");
        let second = list.card_constants(&shadow, start + 64 + 512, 300).expect("room");
        assert_eq!(first % 256, 0);
        assert_eq!(second % 256, 0);
        let constants = list.card_constants.lock().unwrap().as_ref().unwrap().0.clone();
        // SAFETY: a live buffer.
        let base = unsafe { constants.GetGPUVirtualAddress() };
        let back = readback(host, 1024);
        // SAFETY: an open list; the constants buffer is in the constant
        // buffer state after the fills; ranges inside both buffers.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &constants,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&back, 0, &constants, first - base, 512);
            list.list.CopyBufferRegion(&back, 512, &constants, second - base, 300);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); 1024 bytes.
        let read = unsafe {
            back.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024).to_vec()
        };
        assert_eq!(read[..512], pattern[..512], "the first fill");
        assert_eq!(read[512..812], pattern[512..812], "the second fill");
        // The title's memory was never read: it still holds 0xAA.
        assert!(title[64..1088].iter().all(|&b| b == 0xAA));
        crate::shadow::unregister(resource, start);
    }
}

/// The host's own paths for a copy in the card's memory, end to end, on
/// the software adapter only: the title's bytes go up, a shader reads them
/// and writes, and what it wrote comes back into title memory. Run it with
/// the switches under test in the environment (they are read once per
/// process); it does nothing unless the host device is the software
/// adapter (`VICTORY_D3D12_WARP=1` in a diagnostics build) and the copy is
/// in the card's memory (`VICTORY_CARD_BUFFERS_MIN_MB=1`):
///
/// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 [VICTORY_QUEUE2=n]
///  [VICTORY_RELAY=2 | VICTORY_LATE_WAIT=3] cargo test -p victory-gpu
///  --features diagnostics queue2_tests`
///
/// The same title memory must come out whatever the switches: through the
/// direct queue alone, and through the copy queues.
#[cfg(test)]
mod queue2_tests {
    use super::*;

    /// `memory[to] = memory[from] + add`, after `spin` steps.
    const ADD: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint from; uint add; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint h = 0u;
    [loop]
    for (uint i = 0u; i < spin; i++) {
        h = (h ^ memory.Load(from)) * 0x01000193u;
    }
    memory.Store(to, memory.Load(from) + add + (h & 0u));
}
"#;

    #[test]
    fn warp_title_bytes_go_up_and_shader_writes_come_back_whatever_the_queues() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        const ROUNDS: usize = 12;
        // "Title memory": write-watched, as the title's is.
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x51_5151, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(ADD, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // Round k: the title writes a dword in piece 1+k; a submission's
        // shader adds k to it and stores the sum in piece 30+k, a range
        // declared as shader-written. Nothing is waited for between the
        // rounds where submissions are sent ahead.
        let from = |k: usize| (1 + k) * PIECE + 4 * k + 64;
        let to = |k: usize| (30 + k) * PIECE + 8 * k + 128;
        let titles = |k: usize| 0x0100_0000 + 0x1000 * k as u32 + 7;
        let mut lists = Vec::new();
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            unsafe { dword(from(k)).write_volatile(titles(k)) };
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + (30 + k) * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [to(k) as u32, from(k) as u32, k as u32, 30_000];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            if late() {
                host.execute_own(&list);
            } else {
                host.execute(&[&list]);
            }
            lists.push(list);
        }
        settle_now(WHY_SYNC);
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            let (sum, kept) = unsafe { (dword(to(k)).read_volatile(), dword(from(k)).read_volatile()) };
            assert_eq!(kept, titles(k), "round {k}: the title's own dword changed");
            assert_eq!(
                sum,
                titles(k) + k as u32,
                "round {k}: the shader read {:#x} where the title had written {:#x}, or its write did not come back",
                sum.wrapping_sub(k as u32),
                titles(k)
            );
        }
        // Nothing else moved.
        let touched: std::collections::HashSet<usize> = (0..ROUNDS).flat_map(|k| [from(k), to(k)]).collect();
        for i in (0..LEN / 4).filter(|i| !touched.contains(&(i * 4))) {
            // SAFETY: inside the allocation.
            let got = unsafe { dword(i * 4).read_volatile() };
            assert_eq!(got, 0x5A00_0000 | i as u32, "dword {i} of title memory changed");
        }
        // The queues the switches name did the work.
        if let Some(second) = crate::queue2::second() {
            let (down, up) = (second.down.stats.jobs.load(Ordering::Relaxed), second.up.stats.jobs.load(Ordering::Relaxed));
            // (Sends that found no landing buffer went the old way.)
            let no_room = second.landing[1].load(Ordering::Relaxed);
            eprintln!(
                "queue2_tests: VICTORY_QUEUE2={} late {} relay {}: {down} fetches and {up} sends through the copy queues ({no_room} more found no landing buffer), the direct queue told to wait {} times",
                crate::queue2::bits(),
                late(),
                relay(),
                second.held[0].load(Ordering::Relaxed)
            );
            assert_eq!(down > 0, second.fetches(), "fetches through the copy queue");
            assert_eq!(up + no_room > 0, second.uploads(), "sends through the copy queue");
            assert_eq!(up, second.held[0].load(Ordering::Relaxed), "the direct queue waits for every send");
        } else {
            eprintln!("queue2_tests: the direct queue alone (late {}, relay {})", late(), relay());
        }
        drop(lists);
    }

    /// `memory[to] = first` at once; then `spin` steps, each a store to
    /// `memory[to + 4]`; then `memory[to + 4] = 0` and `memory[to] = last`.
    /// Whatever `spin` is, the bytes it leaves are the same.
    const TWICE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint first; uint last; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    memory.Store(to, first);
    [loop]
    for (uint i = 0u; i < spin; i++) {
        memory.Store(to + 4u, i);
    }
    memory.Store(to + 4u, 0u);
    memory.Store(to, last);
}
"#;

    /// The hole a fetch on the copy queue had (`VICTORY_QUEUE2` bit 1 with
    /// the relay): flight A writes a piece; flight B, sent behind it,
    /// writes the same piece through a half-way value and leaves it as A
    /// left it. A is landed while B is half way: the fetch, beside B on
    /// the copy queue, brings B's half-way bytes into title memory. B's
    /// compare finds the piece as A's compare left it and flags nothing,
    /// so B's landing fetched nothing and title memory kept the half-way
    /// bytes. With the fix B's landing fetches the piece all the same
    /// (`BESIDE`).
    ///
    /// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_QUEUE2=1
    ///  VICTORY_RELAY=1 cargo test -p victory-gpu --features diagnostics
    ///  fetched_again -- --nocapture` (alone: it lands its two flights
    /// itself; `VICTORY_QUEUE2_REFETCH=0` shows the fault).
    #[test]
    fn warp_a_piece_fetched_while_a_later_flight_writes_it_is_fetched_again() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        if !relay() || !crate::queue2::second().is_some_and(|s| s.fetches()) || inflight_fetch() {
            eprintln!(
                "queue2_tests: skipped, fetches do not read live card copies on a copy queue beside the relay \
                 (needs VICTORY_RELAY=1/2 VICTORY_QUEUE2=1/3 without VICTORY_CARD_FETCH_INFLIGHT=2)"
            );
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x52_5252, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(TWICE, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // A list whose shader writes the dword at 256 of `piece`, declared.
        let record = |piece: usize, first: u32, last: u32, spin: u32| {
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + piece * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [(piece * PIECE + 256) as u32, first, last, spin];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            list
        };
        // SAFETY: a plain query.
        let completed = || unsafe { host.fence.GetCompletedValue() };
        let wait_for = |value: u64| {
            while completed() < value {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        // How many steps keep the card busy for some seconds: timed on
        // another piece, waited for.
        const TRIAL: u32 = 1 << 20;
        let trial = record(40, 1, 2, TRIAL);
        let timing = std::time::Instant::now();
        host.execute(&[&trial]);
        let took = timing.elapsed().as_secs_f64().max(1e-4);
        let spin = (f64::from(TRIAL) * 3.0 / took).clamp(f64::from(TRIAL), 2.0e9) as u32;
        // SAFETY: inside the allocation.
        assert_eq!(unsafe { dword(40 * PIECE + 256).read_volatile() }, 2, "the trial's write did not come back");
        const AT: usize = 20 * PIECE + 256;
        const HALF_WAY: u32 = 0xDEAD_0001;
        const LEFT: u32 = 0x600D_0002;
        // Flight A, and the card has run it.
        let a = record(20, HALF_WAY, LEFT, 16);
        let (submitted, mut sent_a) = host.submit(&[&a], true).expect("flight A sent ahead");
        let fence_a = *submitted;
        drop(submitted);
        wait_for(fence_a);
        // Flight B: the same piece, through the half-way value, for seconds.
        let b = record(20, HALF_WAY, LEFT, spin);
        let (submitted, mut sent_b) = host.submit(&[&b], true).expect("flight B sent ahead");
        let fence_b = *submitted;
        drop(submitted);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            completed() < fence_b,
            "flight B had run after 0.3 s ({spin} steps; the trial's {TRIAL} took {took:.3} s): nothing to show"
        );
        // A lands, as `Host::relay_land` lands it, while B is half way.
        let mut fetch = None;
        {
            let plan = sent_a.card.as_ref().expect("flight A has a plan for the card's copies");
            host.relay_card_back(&[(sent_a.number, plan)], &mut fetch, fence_a, sent_a.number);
        }
        for (copy, ranges) in &sent_a.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let seen = unsafe { dword(AT).read_volatile() };
        let b_ran = completed() >= fence_b;
        eprintln!(
            "queue2_tests: after A's landing title memory holds {seen:#x} (half way {HALF_WAY:#x}, left {LEFT:#x}); B had run: {b_ran}; {spin} steps"
        );
        assert_eq!(
            seen, HALF_WAY,
            "the fetch for A did not come down beside B (it waited for B, or B had not begun): this machine cannot show the fault"
        );
        // B has run; it left the piece as A left it, so its compare flags
        // nothing. Its landing must bring the piece down all the same.
        wait_for(fence_b);
        {
            let plan = sent_b.card.as_ref().expect("flight B has a plan for the card's copies");
            host.relay_card_back(&[(sent_b.number, plan)], &mut fetch, fence_b, sent_b.number);
        }
        for (copy, ranges) in &sent_b.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let (left, beside) = unsafe { (dword(AT).read_volatile(), dword(AT + 4).read_volatile()) };
        assert_eq!(
            left, LEFT,
            "title memory kept flight B's half-way bytes: B's compare found the piece as A left it and B's landing fetched nothing"
        );
        assert_eq!(beside, 0, "the dword beside it holds a half-way count");
        for sent in [&mut sent_a, &mut sent_b] {
            if let Some(plan) = sent.card.take() {
                host.card_done(plan);
            }
        }
        drop((a, b, trial));
    }

    /// The fetch inside flights with its slots in the card's memory
    /// (`VICTORY_CARD_FETCH_INFLIGHT=2`): flight A leaves a value in a
    /// piece; flight B, sent behind it, writes the same piece through a
    /// half-way value for seconds. A lands while B is half way: its pieces
    /// come down from A's own slot on the landing thread's copy queue,
    /// without waiting for B, and hold what A left, neither B's half-way
    /// bytes (the tear of a copy-queue fetch of the copy itself) nor B's
    /// last ones. Then B lands with its own.
    #[test]
    fn warp_slots_on_the_card_hold_their_flights_bytes_while_a_later_flight_writes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("inflight_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        if !relay() || !inflight_on_card() {
            eprintln!("inflight_tests: skipped, needs VICTORY_RELAY=1 VICTORY_CARD_FETCH_INFLIGHT=2");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x52_5253, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("inflight_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(TWICE, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        let record = |piece: usize, first: u32, last: u32, spin: u32| {
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + piece * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [(piece * PIECE + 256) as u32, first, last, spin];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            list
        };
        // SAFETY: a plain query.
        let completed = || unsafe { host.fence.GetCompletedValue() };
        let wait_for = |value: u64| {
            while completed() < value {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        const TRIAL: u32 = 1 << 20;
        let trial = record(40, 1, 2, TRIAL);
        let timing = std::time::Instant::now();
        host.execute(&[&trial]);
        let took = timing.elapsed().as_secs_f64().max(1e-4);
        let spin = (f64::from(TRIAL) * 3.0 / took).clamp(f64::from(TRIAL), 2.0e9) as u32;
        const AT: usize = 20 * PIECE + 256;
        const A_LEFT: u32 = 0xA000_0001;
        const B_HALF_WAY: u32 = 0xDEAD_0002;
        const B_LEFT: u32 = 0xB000_0003;
        let stats = || [3, 4].map(|i| INFLIGHT_STATS[i].load(Ordering::Relaxed));
        let before = stats();
        let a = record(20, A_LEFT, A_LEFT, 16);
        let (submitted, mut sent_a) = host.submit(&[&a], true).expect("flight A sent ahead");
        let fence_a = *submitted;
        drop(submitted);
        let b = record(20, B_HALF_WAY, B_LEFT, spin);
        let (submitted, mut sent_b) = host.submit(&[&b], true).expect("flight B sent ahead");
        let fence_b = *submitted;
        drop(submitted);
        wait_for(fence_a);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(completed() < fence_b, "flight B had run after 0.3 s ({spin} steps): nothing to show");
        let mut fetch = None;
        {
            let plan = sent_a.card.as_ref().expect("flight A has a plan for the card's copies");
            assert!(plan.inflight, "flight A's compare list has no fetch inside it");
            host.relay_card_back(&[(sent_a.number, plan)], &mut fetch, fence_a, sent_a.number);
        }
        for (copy, ranges) in &sent_a.back {
            copy.copy_back_done(ranges);
        }
        let b_ran = completed() >= fence_b;
        // SAFETY: inside the allocation.
        let seen = unsafe { dword(AT).read_volatile() };
        eprintln!("inflight_tests: after A's landing title memory holds {seen:#x} (A left {A_LEFT:#x}); B had run: {b_ran}");
        assert!(!b_ran, "A's landing waited for the later flight B");
        assert_eq!(seen, A_LEFT, "A's landing did not bring A's own bytes (B's half way {B_HALF_WAY:#x}, B's last {B_LEFT:#x})");
        let after_a = stats();
        assert_eq!(after_a[0], before[0], "A landed the old way");
        assert_eq!(after_a[1], before[1] + 1, "A's slot did not come down on the copy queue");
        wait_for(fence_b);
        {
            let plan = sent_b.card.as_ref().expect("flight B has a plan for the card's copies");
            host.relay_card_back(&[(sent_b.number, plan)], &mut fetch, fence_b, sent_b.number);
        }
        for (copy, ranges) in &sent_b.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let (left, beside) = unsafe { (dword(AT).read_volatile(), dword(AT + 4).read_volatile()) };
        assert_eq!(left, B_LEFT, "B's landing did not bring B's bytes");
        assert_eq!(beside, 0, "the dword beside it holds a half-way count");
        assert_eq!(stats()[0], before[0], "B landed the old way");
        for sent in [&mut sent_a, &mut sent_b] {
            if let Some(plan) = sent.card.take() {
                host.card_done(plan);
            }
        }
        drop((a, b, trial));
    }

    /// The copies down of slots: pieces that follow each other in the slots
    /// join one copy, the others start one of their own; empty ones none.
    #[test]
    fn slots_down_join_pieces_that_follow_each_other() {
        use super::slots_down;
        assert!(slots_down([]).is_empty());
        assert_eq!(
            slots_down([(100, 10), (110, 5), (0, 0), (500, 7), (507, 1), (0, 3)]),
            vec![(100, 0, 15), (500, 15, 8), (0, 23, 3)]
        );
    }
}
