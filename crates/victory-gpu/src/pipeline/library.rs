//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Library calls overlap: the host documents the library as safe to use
//! from several threads ("The pipeline library is thread-safe to use, and
//! will internally synchronize as necessary", `CreatePipelineLibrary`,
//! Thread Safety), with one exception kept here: two loads of the same
//! pipeline are made one after the other ([`same_name`]). Under one lock of
//! ours for every call, the eight workers made pipelines one at a time:
//! 87 to 96 of 110 looks at each worker found it waiting for that lock
//! (run 49), and an earlier run's 2,300 pipelines, all in the library, were
//! still being loaded 46 s into the opening cutscene.
//! `VICTORY_LIBRARY_LOCK=1`: one call at a time, as before.
//!
//! The library's file is mapped, not read into memory ([`Blob`]): the host
//! keeps pointing into the bytes it was made from for as long as it lives,
//! and read into a buffer they were 1,352 MB of the process's own memory
//! for the whole run (run 60), all of it in physical memory. Mapped, the
//! pages are the file's: only those the host reads are in memory, and the
//! system can drop them without writing them anywhere. A save writes the
//! new library through a mapping of the new file in the same way (a buffer
//! of the library's size was made, filled and written out at each save).
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.
//!
//! The mapped file's handle stays open for as long as the mapping lives
//! ([`holds`]). A save renames the new library over the file this run has
//! mapped, and with the handle closed Windows then gives up the old file's
//! contents: every page of the mapping that is not in memory at that
//! moment, or leaves memory later (the trim below, or Windows taking pages
//! back), reads as zeros from then on. The host's library reads such pages
//! as stored pipelines; the driver, handed a pipeline of zeros, removes the
//! device with DXGI_ERROR_DRIVER_INTERNAL_ERROR (runs 72, 73 and 82, each
//! within 5 ms of the first library call after a save and a trim; and the
//! library run 77 saved from such pages was refused by run 78). Measured
//! with a 64 MB file written past the system's cache, mapped and replaced:
//! 16,352 of 16,384 pages read zeros with the handle closed, none with it
//! open (the test below). `VICTORY_LIBRARY_HANDLE=0`: closed, as before.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The folder the library is kept in.
pub(super) fn folder() -> Option<PathBuf> {
    DIR.get().cloned()
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// For the memory lines: the bytes of the library's file, kept in memory
/// while the library lives, and the bytes the last save wrote (a buffer of
/// that size is made, filled and written every time pipelines were added,
/// at most every 20 s).
pub(crate) static BLOB: AtomicU64 = AtomicU64::new(0);
pub(crate) static SAVED: AtomicU64 = AtomicU64::new(0);

/// Whether the host pipeline library is split into 64 lazy on-demand shards
/// under `<dir>/shards/<mode>/shard-<00..3f>.bin`. On by default;
/// `VICTORY_LIBRARY_SHARDS=0` uses the legacy monolithic `host-pipelines.bin`.
fn sharded() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_SHARDS").is_none_or(|v| v != "0"))
}

const SHARD_COUNT: usize = 64;
const SHARDS_DIR: &str = "shards";

fn shard_mode(vram_cbv: bool) -> &'static str {
    if vram_cbv { "vram_cbv" } else { "classic" }
}

fn shard_dir(root: &std::path::Path, vram_cbv: bool) -> PathBuf {
    root.join(SHARDS_DIR).join(shard_mode(vram_cbv))
}

fn shard_path(root: &std::path::Path, vram_cbv: bool, shard: usize) -> PathBuf {
    shard_dir(root, vram_cbv).join(format!("shard-{shard:02x}.bin"))
}

fn shard_next_path(root: &std::path::Path, vram_cbv: bool, shard: usize) -> PathBuf {
    shard_dir(root, vram_cbv).join(format!("shard-{shard:02x}.next"))
}

/// Deterministic shard index `0..64` for pipeline name `name`.
/// Uses the first byte of the hex BLAKE3 digest (`& 63`) when `name` starts
/// with two ASCII hex digits, with an FNV-1a fallback for arbitrary names.
fn shard_index(name: &str) -> usize {
    if name.len() >= 2
        && name.is_char_boundary(2)
        && let Ok(byte) = u8::from_str_radix(&name[..2], 16)
    {
        return usize::from(byte) & (SHARD_COUNT - 1);
    }
    let hash = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (hash as usize) & (SHARD_COUNT - 1)
}

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// Whether the mapped library file's handle is kept open while it is
/// mapped (see the module's notes: without it a save in the same run turns
/// the mapping's pages into zeros as they leave memory). On by default;
/// `VICTORY_LIBRARY_HANDLE=0` closes it after mapping, as before.
fn holds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_HANDLE").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`] ([`holds`]): open
    /// until after the view is unmapped, so that the file a rename took
    /// the name from keeps its contents for the view.
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the file's pages
        // after the mapping object and the file are closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(mapping, if write { FILE_MAP_WRITE } else { FILE_MAP_READ }, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped { at, len, _file: None })
    }

    /// The file at `path`, mapped for reading; `None` if it is not there,
    /// is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        Mapped::open_holding(path, holds())
    }

    /// As [`Mapped::open`]; `hold` says whether the file stays open with
    /// the mapping.
    fn open_holding(path: &std::path::Path, hold: bool) -> Option<Mapped> {
        // Opened as the standard library opens a file to read: others may
        // read, write, rename and delete it meanwhile, so a save still
        // renames the new library over this one.
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        if hold {
            mapped._file = Some(file);
        }
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, tests, or a file
    /// that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    fn of(dir: &std::path::Path) -> Blob {
        Blob::of_paths(&dir.join(LIBRARY_FILE), &dir.join(NEXT_FILE))
    }

    /// The library file at `path`: a pending `next` file from an earlier save
    /// takes its place first.
    fn of_paths(path: &std::path::Path, next: &std::path::Path) -> Blob {
        if next.exists() && std::fs::rename(next, path).is_err() {
            tracing::warn!(path = %path.display(), "the host pipeline library saved last could not take the old one's place");
        }
        Blob::of_readonly(path)
    }

    /// Maps or reads an existing library file read-only without renaming or
    /// mutating anything on disk.
    fn of_readonly(path: &std::path::Path) -> Blob {
        if mapped()
            && let Some(file) = Mapped::open(path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(path) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

/// Live mapped library views (`(base, len)` — either the monolithic file or
/// lazily mapped shards + read-only legacy fallback). Never unmapped while the
/// process lives, so [`trim`] and [`removal_note`] can safely inspect them.
static MAPPED_VIEWS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
static LOADED_AT: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// The bytes of the views let go so far, for the memory line.
pub(crate) static TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the views' pages were last let go (`crate::host::now_ms`), and how
/// often and when a save of this run last renamed a new library over a
/// mapped file: for [`removal_note`].
static TRIMMED_AT: AtomicU64 = AtomicU64::new(0);
static REPLACED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static SAVE_SEQ: AtomicU64 = AtomicU64::new(0);

fn register_mapped(file: &Mapped) {
    MAPPED_VIEWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((file.at.Value as usize, file.len));
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
}

/// The library calls running now: thread, what (load, make, store,
/// serialise), the pipeline, and since when (`crate::host::now_us`).
static CALLS: Mutex<Vec<(u32, &'static str, String, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// The pipeline this thread is making (its shaders), for [`CALLS`].
    static LABEL: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Names the pipeline this thread makes next.
pub(crate) fn label(text: String) {
    LABEL.with(|l| *l.borrow_mut() = text);
}

/// A library call of this thread, listed in [`CALLS`] until dropped.
struct Call(u32, u64);

fn call(what: &'static str, name: &str) -> Call {
    let (thread, since) = (crate::host::thread_id(), crate::host::now_us());
    let label = LABEL.with(|l| l.borrow().clone());
    CALLS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((thread, what, format!("{name} {label}"), since));
    Call(thread, since)
}

impl Drop for Call {
    fn drop(&mut self) {
        CALLS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|c| (c.0, c.3) != (self.0, self.1));
    }
}

/// For the report when the host device is removed: the library calls
/// running at that moment, whether this run replaced the mapped file and
/// let its pages go, and how many pages of the mapping read as zeros now.
/// A library holds no page of zeros (the 259 MB one of run 83 has no run of
/// 2,048 zero bytes): each such page is one whose contents were lost (see
/// the module's notes), and a pipeline stored there is what the driver was
/// handed.
pub(crate) fn removal_note() -> String {
    use std::fmt::Write;
    let now = crate::host::now_us();
    let mut text = String::from("pipeline library: ");
    {
        let calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        if calls.is_empty() {
            text.push_str("no call running");
        }
        for (i, (thread, what, name, since)) in calls.iter().enumerate() {
            let _ = write!(
                text,
                "{}{what} of {name} running for {:.1} ms on thread {thread}",
                if i == 0 { "" } else { ", " },
                now.saturating_sub(*since) as f64 / 1000.0
            );
        }
    }
    let views: Vec<(usize, usize)> = MAPPED_VIEWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if views.is_empty() {
        text.push_str("; its file is not mapped");
        return text;
    }
    let total_len: usize = views.iter().map(|&(_, len)| len).sum();
    let ago = |ms: u64| crate::host::now_ms().saturating_sub(ms) as f64 / 1000.0;
    let _ = write!(
        text,
        "; {} mapped library file(s) ({total_len} bytes), the handle {}",
        views.len(),
        if holds() { "kept open" } else { "closed (VICTORY_LIBRARY_HANDLE=0)" }
    );
    match REPLACED[0].load(Ordering::Relaxed) {
        0 => text.push_str("; this run has not replaced a mapped file"),
        n => {
            let _ = write!(
                text,
                "; this run replaced a mapped file {n} times, last {:.1} s ago",
                ago(REPLACED[1].load(Ordering::Relaxed))
            );
        }
    }
    if TRIMMED_BYTES.load(Ordering::Relaxed) > 0 {
        let _ = write!(text, "; its pages were last let go {:.1} s ago", ago(TRIMMED_AT.load(Ordering::Relaxed)));
    }
    // Within two seconds: pages not in memory come from the disk.
    let started = std::time::Instant::now();
    let (mut zero, mut looked, mut first) = (0u64, 0u64, None);
    'scan: for &(at, len) in &views {
        // SAFETY: each registered view lives for the process lifetime.
        let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, len) };
        for page in bytes.chunks(4096) {
            if looked.is_multiple_of(4096) && started.elapsed() > std::time::Duration::from_secs(2) {
                break 'scan;
            }
            if page.iter().all(|&b| b == 0) {
                zero += 1;
                if first.is_none() {
                    first = Some(looked);
                }
            }
            looked += 1;
        }
    }
    let _ = write!(text, "; {zero} of the {looked} pages looked at read as zeros now");
    if let Some(first) = first {
        let _ = write!(text, " (the first is page {first}): the library's contents were lost there");
    }
    text
}

/// Lets the mapped library file's pages go from the process's memory once
/// no pipeline has been loaded for ten seconds. The host reads nearly the
/// whole file while the early pipelines load (run 65: all 1,490 MB of it
/// were in the process's memory a minute in, and 1,300 MB still at the
/// end, on a PC with 2 GB free); after that it reads the file again only
/// when a new pipeline loads. Pages let go stay in Windows' file cache
/// while nothing else wants the memory, and come back on the next read.
/// Called once a second by the sweep; does its work on a thread of its own
/// (hundreds of thousands of pages), once after each burst of loads.
/// `VICTORY_LIBRARY_TRIM=0`: the pages stay until Windows takes them.
pub(crate) fn trim() {
    static ON: OnceLock<bool> = OnceLock::new();
    if TRIMMED.load(Ordering::Relaxed)
        || !*ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_TRIM").is_none_or(|v| v != "0"))
    {
        return;
    }
    let views: Vec<(usize, usize)> = MAPPED_VIEWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if views.is_empty() {
        return;
    }
    let quiet_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    if crate::host::now_ms() < LOADED_AT.load(Ordering::Relaxed) + quiet_ms
        || TRIMMED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let work = move || {
        let mut total = 0u64;
        for (at, len) in views {
            // SAFETY: each registered view lives for the process lifetime.
            // Unlocking pages that are not locked fails, as expected, and
            // takes them out of the working set: the documented way to let
            // pages of a mapped file go.
            unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len) };
            total += len as u64;
        }
        TRIMMED_BYTES.fetch_add(total, Ordering::Relaxed);
        TRIMMED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::host::trail(|| format!("pipeline library: mapped files' pages let go from memory ({total} bytes)"));
    };
    if std::thread::Builder::new().name("victory-library-trim".into()).spawn(work).is_err() {
        TRIMMED.store(false, Ordering::Relaxed);
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// Stores share it and a save takes it alone: the size the library
    /// asks for must still be its size when it is written out.
    saving: RwLock<()>,
}

// SAFETY: D3D12 objects are free-threaded, and the library synchronises
// its own calls (see the module's notes); what it does not, two loads of
// one name, is serialised by `same_name`.
unsafe impl Send for Library {}
// SAFETY: as above; the other fields are an atomic and a lock.
unsafe impl Sync for Library {}

/// Held by a load of the pipeline named `name`: loads of the same name
/// wait for each other (the host's one exception to the library's thread
/// safety: such a load may change that pipeline's entry). Names share 64
/// locks, so two different names seldom wait.
fn same_name(name: &str) -> MutexGuard<'static, ()> {
    static LOADING: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];
    let at = name.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b)));
    LOADING[at % LOADING.len()].lock().unwrap_or_else(|e| e.into_inner())
}

/// With `VICTORY_LIBRARY_LOCK=1`: held by every library call, as before.
fn one_call() -> Option<MutexGuard<'static, ()>> {
    static ON: OnceLock<bool> = OnceLock::new();
    static CALLING: Mutex<()> = Mutex::new(());
    ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_LOCK").is_some_and(|v| v == "1"))
        .then(|| CALLING.lock().unwrap_or_else(|e| e.into_inner()))
}

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 100_000;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR.get().map_or(Blob::None, |d| Blob::of(d));
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library
        })
        .as_ref()
}

/// 64 lazily opened mode-separated pipeline library shards with a read-only
/// fallback to legacy monolithic `host-pipelines.bin` (zero-loss migration).
struct ShardedLibrary {
    device: ID3D12Device1,
    dir: Option<PathBuf>,
    vram_cbv: bool,
    track_globals: bool,
    shards: [OnceLock<Option<Library>>; SHARD_COUNT],
    legacy: OnceLock<Option<Library>>,
    migrated: AtomicU64,
}

// SAFETY: D3D12 devices and pipeline libraries are free-threaded; per-shard
// access is synchronised by `same_name` and each `Library`'s `saving` RwLock.
unsafe impl Send for ShardedLibrary {}
// SAFETY: as above.
unsafe impl Sync for ShardedLibrary {}

impl ShardedLibrary {
    fn new(device: ID3D12Device1, dir: Option<PathBuf>, vram_cbv: bool, track_globals: bool) -> Self {
        Self {
            device,
            dir,
            vram_cbv,
            track_globals,
            shards: [const { OnceLock::new() }; SHARD_COUNT],
            legacy: OnceLock::new(),
            migrated: AtomicU64::new(0),
        }
    }

    /// Lazily opens and maps shard `idx` on first access. Unvisited shards
    /// remain uninitialized (0 RAM, 0 open handles).
    fn shard(&self, idx: usize) -> Option<&Library> {
        self.shards[idx]
            .get_or_init(|| {
                let blob = self.dir.as_deref().map_or(Blob::None, |d| {
                    Blob::of_paths(&shard_path(d, self.vram_cbv, idx), &shard_next_path(d, self.vram_cbv, idx))
                });
                open_with(&self.device, blob, self.track_globals, Some(idx))
            })
            .as_ref()
    }

    /// Lazily opens existing legacy `host-pipelines.bin` read-only on a shard
    /// miss. Returns `None` if absent, empty, or rejected by the host driver.
    fn legacy(&self) -> Option<&Library> {
        self.legacy
            .get_or_init(|| {
                let dir = self.dir.as_deref()?;
                let blob = Blob::of_readonly(&dir.join(LIBRARY_FILE));
                open_readonly_fallback(&self.device, blob, self.track_globals)
            })
            .as_ref()
    }

    fn any_dirty(&self) -> bool {
        self.shards.iter().any(|slot| {
            slot.get()
                .and_then(Option::as_ref)
                .is_some_and(|s| s.added.load(Ordering::Relaxed) > 0)
        })
    }

    fn load_or_create(
        &self,
        name: &str,
        mut load: impl FnMut(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
        create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
    ) -> windows::core::Result<ID3D12PipelineState> {
        let idx = shard_index(name);
        let wide = HSTRING::from(name);
        let Some(shard) = self.shard(idx) else {
            if self.track_globals {
                MADE.fetch_add(1, Ordering::Relaxed);
            }
            return create();
        };
        let began = std::time::Instant::now();
        // Step 1: check the target shard first (lock order: OnceLock -> one_call -> same_name).
        let loaded = {
            let _all = one_call();
            let _one = same_name(name);
            let _call = call("load", name);
            load(&shard.library, PCWSTR(wide.as_ptr()))
        };
        if self.track_globals {
            LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
            TRIMMED.store(false, Ordering::Relaxed);
        }
        let refused = match loaded {
            Ok(pso) => {
                if self.track_globals {
                    LOADED.fetch_add(1, Ordering::Relaxed);
                    crate::host::trail(|| {
                        let label = LABEL.with(|l| l.borrow().clone());
                        format!(
                            "pipeline library: loaded {label} from shard-{idx:02x} in {:.1} ms",
                            began.elapsed().as_secs_f64() * 1e3
                        )
                    });
                }
                return Ok(pso);
            }
            Err(e) => e,
        };
        // Step 2: on shard miss, lazily check read-only legacy `host-pipelines.bin`
        // (outside any lock) and migrate hits into the target shard with zero recompile.
        if let Some(legacy) = self.legacy() {
            let migrated = {
                let _all = one_call();
                let _one = same_name(name);
                let _call = call("load-legacy", name);
                // Re-check shard in case a racing thread just migrated `name`.
                match load(&shard.library, PCWSTR(wide.as_ptr())) {
                    Ok(pso) => Some(pso),
                    Err(_) => match load(&legacy.library, PCWSTR(wide.as_ptr())) {
                        Ok(pso) => {
                            let _stores = shard.saving.read().unwrap_or_else(|e| e.into_inner());
                            // SAFETY: live PSO and library; duplicate name is harmlessly ignored.
                            if unsafe { shard.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
                                shard.added.fetch_add(1, Ordering::Relaxed);
                                self.migrated.fetch_add(1, Ordering::Relaxed);
                            }
                            Some(pso)
                        }
                        Err(_) => None,
                    },
                }
            };
            if self.track_globals {
                LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
                TRIMMED.store(false, Ordering::Relaxed);
            }
            if let Some(pso) = migrated {
                if self.track_globals {
                    LOADED.fetch_add(1, Ordering::Relaxed);
                    crate::host::trail(|| {
                        let label = LABEL.with(|l| l.borrow().clone());
                        format!(
                            "pipeline library: migrated {label} from legacy into shard-{idx:02x} in {:.1} ms",
                            began.elapsed().as_secs_f64() * 1e3
                        )
                    });
                }
                return Ok(pso);
            }
        }
        // Step 3: compile outside any lock, then store into the target shard.
        let pso = {
            let _call = call("make", name);
            create()?
        };
        if self.track_globals {
            MADE.fetch_add(1, Ordering::Relaxed);
            crate::host::trail(|| {
                let label = LABEL.with(|l| l.borrow().clone());
                format!(
                    "pipeline library: made {label} for shard-{idx:02x} in {:.1} ms (not loaded: {refused})",
                    began.elapsed().as_secs_f64() * 1e3
                )
            });
        }
        let _all = one_call();
        let _one = same_name(name);
        let _stores = shard.saving.read().unwrap_or_else(|e| e.into_inner());
        let _call = call("store", name);
        // SAFETY: a live pipeline; a name already stored (a description that
        // differs) is refused, which leaves the library as it was.
        if unsafe { shard.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
            shard.added.fetch_add(1, Ordering::Relaxed);
        }
        Ok(pso)
    }

    /// Serializes and atomically replaces ONLY shards marked dirty (`added > 0`),
    /// preserving open mapped file handles (`holds()`) and never touching clean
    /// shards or legacy `host-pipelines.bin`.
    fn save_dirty(&self) -> (usize, usize, std::io::Result<()>) {
        let Some(dir) = self.dir.as_deref() else {
            return (0, 0, Ok(()));
        };
        let sdir = shard_dir(dir, self.vram_cbv);
        let mut dir_created = false;
        let mut total_bytes = 0usize;
        let mut shards_saved = 0usize;
        let mut first_err: Option<std::io::Error> = None;

        for (idx, slot) in self.shards.iter().enumerate() {
            let Some(Some(shard)) = slot.get() else {
                continue;
            };
            if shard.added.load(Ordering::Relaxed) == 0 {
                continue;
            }
            if !dir_created {
                if let Err(e) = std::fs::create_dir_all(&sdir) {
                    return (0, 0, Err(e));
                }
                dir_created = true;
            }
            let _all = one_call();
            // Hold saving.write() across serialize + atomic replace so concurrent
            // periodic/shutdown saves never race an older rename over a newer one.
            let _alone = shard.saving.write().unwrap_or_else(|e| e.into_inner());
            if shard.added.load(Ordering::Relaxed) == 0 {
                continue;
            }
            // SAFETY: a plain query.
            let size = unsafe { shard.library.GetSerializedSize() };
            let seq = SAVE_SEQ.fetch_add(1, Ordering::Relaxed);
            let tmp = sdir.join(format!("shard-{idx:02x}.bin.{}.{seq}.tmp", std::process::id()));
            let target = shard_path(dir, self.vram_cbv, idx);
            let next = shard_next_path(dir, self.vram_cbv, idx);
            let label = format!("shard-{idx:02x}");
            let _call = call("serialise", &label);
            if let Err(e) = serialise(&shard.library, &tmp, size) {
                tracing::warn!(shard = idx, error = %e, "host pipeline shard not serialised");
                let _ = std::fs::remove_file(&tmp);
                if first_err.is_none() {
                    first_err = Some(e);
                }
                continue;
            }
            match replace_file(&tmp, &target, &next) {
                Ok(in_place) => {
                    shard.added.store(0, Ordering::Relaxed);
                    total_bytes += size;
                    shards_saved += 1;
                    if self.track_globals && in_place && matches!(shard._blob, Blob::Mapped(_)) {
                        REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
                        REPLACED[0].fetch_add(1, Ordering::Relaxed);
                    }
                    if self.track_globals {
                        crate::host::trail(|| {
                            format!(
                                "pipeline library: saved shard-{idx:02x} ({size} bytes), {}",
                                if in_place {
                                    "renamed over shard file"
                                } else {
                                    "left beside shard file for next start"
                                }
                            )
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!(shard = idx, error = %e, "host pipeline shard not replaced");
                    let _ = std::fs::remove_file(&tmp);
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        if self.track_globals && total_bytes > 0 {
            SAVED.store(total_bytes as u64, Ordering::Relaxed);
        }
        (total_bytes, shards_saved, first_err.map_or(Ok(()), Err))
    }

    #[cfg(test)]
    fn opened_shards(&self) -> usize {
        self.shards.iter().filter(|s| s.get().is_some()).count()
    }

    #[cfg(test)]
    fn is_shard_opened(&self, idx: usize) -> bool {
        self.shards[idx].get().is_some()
    }

    #[cfg(test)]
    fn is_legacy_opened(&self) -> bool {
        self.legacy.get().is_some()
    }

    #[cfg(test)]
    fn migrated_count(&self) -> u64 {
        self.migrated.load(Ordering::Relaxed)
    }
}

fn sharded_library() -> Option<&'static ShardedLibrary> {
    static SHARDED: OnceLock<Option<ShardedLibrary>> = OnceLock::new();
    SHARDED
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let vram_cbv = victory_shader::emit::vram_cbv();
            let dir = DIR.get().cloned();
            tracing::info!(
                mode = shard_mode(vram_cbv),
                shards = SHARD_COUNT,
                "sharded on-demand host pipeline library enabled"
            );
            Some(ShardedLibrary::new(device, dir, vram_cbv, true))
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    open_with(device, blob, true, None)
}

fn open_with(device: &ID3D12Device1, blob: Blob, track_globals: bool, shard: Option<usize>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.bytes().is_empty() {
        match make(blob.bytes()) {
            Ok(library) => {
                let bytes = blob.bytes().len();
                if track_globals {
                    if shard.is_some() {
                        BLOB.fetch_add(bytes as u64, Ordering::Relaxed);
                    } else {
                        BLOB.store(bytes as u64, Ordering::Relaxed);
                    }
                    if let Blob::Mapped(file) = &blob {
                        register_mapped(file);
                    }
                }
                match shard {
                    Some(idx) => tracing::debug!(shard = idx, bytes, mapped = matches!(blob, Blob::Mapped(_)), "host pipeline shard loaded"),
                    None => tracing::info!(bytes, mapped = matches!(blob, Blob::Mapped(_)), "host pipeline library loaded"),
                }
                return Some(Library {
                    library,
                    _blob: blob,
                    added: AtomicU64::new(0),
                    saving: RwLock::new(()),
                });
            }
            Err(e) => match shard {
                Some(idx) => tracing::info!(shard = idx, error = %e, "host pipeline shard rejected; starting an empty one"),
                None => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
            },
        }
    } else if shard.is_none() {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        added: AtomicU64::new(0),
        saving: RwLock::new(()),
    })
}

/// Opens a read-only fallback library from `blob`, returning `None` if `blob`
/// is empty or rejected by the host (never allocates an empty fallback library).
fn open_readonly_fallback(device: &ID3D12Device1, blob: Blob, track_globals: bool) -> Option<Library> {
    if blob.bytes().is_empty() {
        return None;
    }
    // SAFETY: the bytes outlive the library (kept beside it in `Library::_blob`).
    let made: windows::core::Result<ID3D12PipelineLibrary1> = unsafe {
        device
            .CreatePipelineLibrary::<ID3D12PipelineLibrary>(blob.bytes())
            .and_then(|l| l.cast())
    };
    match made {
        Ok(library) => {
            let bytes = blob.bytes().len();
            if track_globals {
                BLOB.fetch_add(bytes as u64, Ordering::Relaxed);
                if let Blob::Mapped(file) = &blob {
                    register_mapped(file);
                }
            }
            tracing::info!(
                bytes,
                mapped = matches!(blob, Blob::Mapped(_)),
                "legacy host-pipelines.bin opened read-only for zero-loss shard migration"
            );
            Some(Library {
                library,
                _blob: blob,
                added: AtomicU64::new(0),
                saving: RwLock::new(()),
            })
        }
        Err(e) => {
            tracing::info!(error = %e, "legacy host-pipelines.bin rejected; skipping read-only fallback");
            drop(blob);
            None
        }
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    if victory_shader::emit::vram_cbv() {
        let tag = b"vram_cbv=1";
        h.update(&(tag.len() as u64).to_le_bytes());
        h.update(tag);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    mut load: impl FnMut(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    if sharded() {
        let Some(store) = sharded_library() else {
            MADE.fetch_add(1, Ordering::Relaxed);
            return create();
        };
        return store.load_or_create(name, load, create);
    }
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    let began = std::time::Instant::now();
    let loaded = {
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load", name);
        load(&library.library, PCWSTR(wide.as_ptr()))
    };
    // The load read the library's file (see `trim`).
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
    let refused = match loaded {
        Ok(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            crate::host::trail(|| {
                let label = LABEL.with(|l| l.borrow().clone());
                format!("pipeline library: loaded {label} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3)
            });
            return Ok(pso);
        }
        Err(e) => e,
    };
    // Compiled outside any lock: compiles take up to seconds.
    let pso = {
        let _call = call("make", name);
        create()?
    };
    MADE.fetch_add(1, Ordering::Relaxed);
    crate::host::trail(|| {
        let label = LABEL.with(|l| l.borrow().clone());
        format!(
            "pipeline library: made {label} in {:.1} ms (the library did not load it: {refused})",
            began.elapsed().as_secs_f64() * 1e3
        )
    });
    let _all = one_call();
    let _stores = library.saving.read().unwrap_or_else(|e| e.into_inner());
    let _call = call("store", name);
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { library.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        library.added.fetch_add(1, Ordering::Relaxed);
    }
    Ok(pso)
}

/// Writes the library (or dirty shards) and the early list if pipelines were
/// added since the last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let Some(dir) = DIR.get() else { return };
    if sharded() {
        let Some(store) = sharded_library() else { return };
        {
            let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
            if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
                return;
            }
            *last = Some(std::time::Instant::now());
        }
        let dirty = store.any_dirty();
        let used_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
        if !dirty && !used_changed {
            return;
        }
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::warn!(error = %e, "host pipeline library not saved");
            return;
        }
        let (bytes, shards_saved, library_written) = if dirty {
            store.save_dirty()
        } else {
            (0, 0, Ok(()))
        };
        let used: Vec<String> = used().iter().flatten().cloned().collect();
        let slow: Vec<String> = slow_ones().iter().flatten().map(|(key, ms)| format!("{key} {ms}")).collect();
        let written = library_written
            .and_then(|()| write_atomically(&dir.join(SLOW_FILE), slow.join("\n").as_bytes()))
            .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
        match written {
            Ok(()) => tracing::info!(
                bytes,
                shards_saved,
                migrated = store.migrated.load(Ordering::Relaxed),
                early = used.len(),
                made = MADE.load(Ordering::Relaxed),
                loaded = LOADED.load(Ordering::Relaxed),
                started_early = EARLY.load(Ordering::Relaxed),
                "host pipeline library saved"
            ),
            Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
        }
        return;
    }
    let Some(library) = library() else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    let seq = SAVE_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!("{LIBRARY_FILE}.{}.{seq}.tmp", std::process::id()));
    // The bytes of the library written to `tmp`; 0 when no pipeline was
    // added: the library on disk is then what it was, and only the early
    // list is written (a title pipeline first drawn with wrote all 1,352 MB
    // of the library again: run 60's second save).
    let bytes = {
        let _all = one_call();
        // No store between asking the size and writing; loads go on.
        let _alone = library.saving.write().unwrap_or_else(|e| e.into_inner());
        let added = library.added.load(Ordering::Relaxed);
        if !USED_CHANGED.swap(false, Ordering::Relaxed) && added == 0 {
            return;
        }
        if added == 0 {
            0
        } else {
            // SAFETY: a plain query.
            let size = unsafe { library.library.GetSerializedSize() };
            let _call = call("serialise", "the whole library");
            if let Err(e) = serialise(&library.library, &tmp, size) {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            library.added.store(0, Ordering::Relaxed);
            SAVED.store(size as u64, Ordering::Relaxed);
            size
        }
    };
    // Through `used`: a process that asked for none yet (`victory shaders`)
    // must not write the earlier runs' list over with nothing.
    let used: Vec<String> = used().iter().flatten().cloned().collect();
    let library_written = if bytes > 0 { replace_library(dir, &tmp) } else { Ok(()) };
    if bytes > 0 && library_written.is_ok() {
        // In place unless the system refused and the new one waits beside it.
        let in_place = !dir.join(NEXT_FILE).exists();
        if in_place && matches!(library._blob, Blob::Mapped(_)) {
            REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
            REPLACED[0].fetch_add(1, Ordering::Relaxed);
        }
        crate::host::trail(|| {
            format!(
                "pipeline library: saved {bytes} bytes, {}",
                if in_place { "renamed over the library's file" } else { "left beside the library's file for the next start" }
            )
        });
    }
    let slow: Vec<String> = slow_ones().iter().flatten().map(|(key, ms)| format!("{key} {ms}")).collect();
    let written = library_written
        .and_then(|()| write_atomically(&dir.join(SLOW_FILE), slow.join("\n").as_bytes()))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes the library's `size` serialised bytes to the file `tmp`: through
/// a mapping of that file, so that the bytes are the file's pages and not
/// 1.3 GB more of the process's own memory for the seconds a save takes;
/// through a buffer where the file cannot be mapped (and with
/// `VICTORY_LIBRARY_MAP=0`).
fn serialise(library: &ID3D12PipelineLibrary1, tmp: &std::path::Path, size: usize) -> std::io::Result<()> {
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    if mapped()
        && let Some(mut file) = Mapped::create(tmp, size)
    {
        // SAFETY: a buffer of the size the library asks for.
        return unsafe { library.Serialize(file.bytes_mut()) }.map_err(failed);
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: a buffer of the size the library asks for.
    unsafe { library.Serialize(&mut bytes) }.map_err(failed)?;
    std::fs::write(tmp, &bytes)
}

/// Renames `tmp` over `target`, falling back to `next` if the OS refuses to
/// replace a mapped file in place. Returns `Ok(true)` when replaced in place.
fn replace_file(
    tmp: &std::path::Path,
    target: &std::path::Path,
    next: &std::path::Path,
) -> std::io::Result<bool> {
    match std::fs::rename(tmp, target) {
        Ok(()) => Ok(true),
        Err(_) => {
            std::fs::rename(tmp, next)?;
            Ok(false)
        }
    }
}

/// Renames the written library `tmp` over the library's file in `dir`. This
/// run may have that file mapped: Windows 11 replaces a mapped file, and
/// the mapping goes on reading the old one for as long as a handle on it
/// is open ([`holds`]; with none, only the pages in memory at that moment
/// stay: the rest read as zeros). Where a system refuses, the new library
/// is left beside it as [`NEXT_FILE`] for the next start.
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    replace_file(tmp, &dir.join(LIBRARY_FILE), &dir.join(NEXT_FILE)).map(|_| ())
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::AtomicU64;
    static WRITING: Mutex<()> = Mutex::new(());
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    let tmp = path.with_file_name(name);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// How long each title pipeline that took long to make took, in ms (beside
/// the early list; a key may be there more than once).
const SLOW_FILE: &str = "slow-pipelines.txt";
/// A pipeline that takes this long is one ([`slow`]).
pub(crate) const SLOW_MS: u64 = 1000;
static SLOW: Mutex<Option<std::collections::HashMap<String, u64>>> = Mutex::new(None);

fn slow_ones() -> std::sync::MutexGuard<'static, Option<std::collections::HashMap<String, u64>>> {
    let mut slow = SLOW.lock().unwrap_or_else(|e| e.into_inner());
    if slow.is_none() {
        let listed = DIR.get().and_then(|d| std::fs::read_to_string(d.join(SLOW_FILE)).ok()).unwrap_or_default();
        *slow = Some(
            listed
                .lines()
                .filter_map(|l| l.split_once(' '))
                .filter_map(|(key, ms)| Some((key.to_owned(), ms.parse().ok()?)))
                .collect(),
        );
    }
    slow
}

/// Whether making the title pipeline with `key` took [`SLOW_MS`] in this or
/// an earlier run.
pub(crate) fn slow(key: &str) -> bool {
    slow_ones().as_ref().is_some_and(|s| s.contains_key(key))
}

/// The title pipeline with `key` took `ms` to make.
pub(crate) fn note_time(key: &str, ms: u64) {
    if ms < SLOW_MS {
        return;
    }
    let mut slow = slow_ones();
    let known = slow.get_or_insert_default().entry(key.to_owned()).or_insert(0);
    if ms > *known {
        *known = ms;
        // With the next save of the early list.
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

/// Title pipeline keys requested by draws or dispatches (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was requested in this or an earlier run.
///
/// `VICTORY_EARLY_ALL=1`: every title pipeline is started at creation,
/// requested before or not, so that the driver's own cache holds it for
/// later runs.
pub(crate) fn wanted_early(key: &str) -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    if *ALL.get_or_init(|| std::env::var_os("VICTORY_EARLY_ALL").is_some_and(|v| v == "1")) {
        return true;
    }
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// A draw or dispatch requested the title pipeline with `key`.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16))).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Blob::None));
    }

    /// A mapped file reads as the file's bytes, a file written through a
    /// mapping holds what was written, and the library's file can be
    /// replaced while it is mapped (or the new one waits as the next).
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let tmp = path.with_extension("tmp");
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        {
            let mut file = Mapped::create(&tmp, new.len()).expect("a new file, mapped");
            file.bytes_mut().copy_from_slice(&new);
        }
        assert_eq!(std::fs::read(&tmp).unwrap(), new);
        replace_library(dir.path(), &tmp).unwrap();
        assert!(!tmp.exists());
        // The mapping still reads the old library.
        assert_eq!(blob.bytes(), &old[..]);
        drop(blob);
        // The next start reads the new one, wherever the save left it.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.chunks_exact_mut(4).enumerate() {
            let v = ((at / 4 + i) as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
            word.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The device loss of runs 72, 73 and 82: the library's file is
    /// replaced while it is mapped, and pages of the mapping that were not
    /// in memory then are read afterwards (in the runs: pages the trim let
    /// go and Windows gave to something else). With the file's handle kept
    /// open they still read as the old file. (With it closed, as before,
    /// all but the first few read as zeros on the PC this was found on.)
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        if !write_uncached(&path, LEN) {
            return; // no uncached writes here: nothing to show
        }
        let mapped = Mapped::open_holding(&path, true).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        let tmp = path.with_extension("tmp");
        {
            let mut file = Mapped::create(&tmp, LEN).expect("a new file, mapped");
            file.bytes_mut().fill(0xEE);
        }
        replace_library(dir.path(), &tmp).unwrap();
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(bytes == &expected[..], "page {page} of the replaced library no longer reads as the old file");
        }
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }

    #[test]
    fn library_name_preserves_baseline_when_vram_cbv_off_and_separates_when_on() {
        let off = victory_shader::emit::with_vram_cbv(false, || {
            name("graphics", b"root", &[b"vs", b"ps"], "state")
        });
        let on = victory_shader::emit::with_vram_cbv(true, || {
            name("graphics", b"root", &[b"vs", b"ps"], "state")
        });
        assert_ne!(off, on, "VRAM CBV pipelines must not collide with classic CBV entries");

        let mut h = blake3::Hasher::new();
        for part in [b"graphics".as_slice(), b"root", b"vs", b"ps", b"state"] {
            h.update(&(part.len() as u64).to_le_bytes());
            h.update(part);
        }
        assert_eq!(off, h.finalize().to_hex()[..32]);
    }

    /// Minimal valid SM 6.6 compute DXIL with `[numthreads(threads_x, 1, 1)]`
    /// and an empty root signature for real D3D12 pipeline library tests.
    fn test_compute_pso_desc(
        device: &ID3D12Device1,
        threads_x: u32,
    ) -> (ID3D12RootSignature, Vec<u8>) {
        let code_bytes = 0xBF81_0000u32.to_le_bytes().to_vec(); // s_endpgm
        let native = victory_shader::NativeProgram {
            stage: victory_shader::HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: Vec::new(),
            used: Vec::new(),
            io: victory_shader::native::StageIo::Cs(victory_shader::native::ComputeIo {
                rsrc2: 0,
                threads: [threads_x, 1, 1],
            }),
            code: code_bytes,
        };
        let c = victory_shader::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let p = victory_shader::ir::build(&native.code, false, native.vgprs).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let res = victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();

        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
                    ..Default::default()
                },
            },
        };
        // SAFETY: valid root signature description and device.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            let mut err = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut err)).unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            device.CreateRootSignature(0, bytes).unwrap()
        };
        (root, shader.dxil)
    }

    fn make_compute_desc(root: &ID3D12RootSignature, dxil: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
        D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: borrowed pointer outlived by caller's `root`.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn shard_index_is_deterministic_and_covers_all_64_shards() {
        let mut seen = [false; SHARD_COUNT];
        for b in 0..=255u8 {
            let name = format!("{b:02x}0123456789abcdef0123456789abcd");
            let idx = shard_index(&name);
            assert_eq!(idx, usize::from(b) & 63);
            assert_eq!(idx, shard_index(&name));
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "all 64 shards must be reachable");
        // Non-hex arbitrary names also map deterministically into 0..64.
        for label in ["", "a", "pso-alpha", "pso-beta", "mesh_triangles"] {
            let idx = shard_index(label);
            assert!(idx < SHARD_COUNT);
            assert_eq!(idx, shard_index(label));
        }
    }

    #[test]
    fn sharded_library_opens_lazily_and_saves_only_dirty_shards() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let (root1, dxil1) = test_compute_pso_desc(&device, 1);
        let (root2, dxil2) = test_compute_pso_desc(&device, 2);
        let (root3, dxil3) = test_compute_pso_desc(&device, 4);
        let desc1 = make_compute_desc(&root1, &dxil1);
        let desc2 = make_compute_desc(&root2, &dxil2);
        let desc3 = make_compute_desc(&root3, &dxil3);

        // Two names routed into distinct shards (0x05 and 0x2a), plus a third into 0x2a.
        let name1 = "05aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let name2 = "2abbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let name3 = "2acccccccccccccccccccccccccccccc";
        let s1 = shard_index(name1);
        let s2 = shard_index(name2);
        assert_ne!(s1, s2);
        assert_eq!(s2, shard_index(name3));

        // Session 1: compile name1 + name2 -> only 2 of 64 shards open & saved.
        {
            let store = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), true, false);
            assert_eq!(store.opened_shards(), 0);
            assert!(!store.is_legacy_opened());
            let mut created = 0usize;
            store
                .load_or_create(
                    name1,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc1) },
                    || {
                        created += 1;
                        unsafe { device.CreateComputePipelineState(&desc1) }
                    },
                )
                .unwrap();
            assert_eq!(created, 1);
            assert_eq!(store.opened_shards(), 1);
            assert!(store.is_shard_opened(s1));
            assert!(!store.is_shard_opened(s2));

            store
                .load_or_create(
                    name2,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc2) },
                    || {
                        created += 1;
                        unsafe { device.CreateComputePipelineState(&desc2) }
                    },
                )
                .unwrap();
            assert_eq!(created, 2);
            assert_eq!(store.opened_shards(), 2);

            let (bytes, saved, res) = store.save_dirty();
            res.unwrap();
            assert_eq!(saved, 2);
            assert!(bytes > 0);
            assert!(shard_path(dir.path(), true, s1).is_file());
            assert!(shard_path(dir.path(), true, s2).is_file());
            assert!(!shard_path(dir.path(), true, 0).exists());
            // A second save with no new PSOs writes nothing.
            let (bytes2, saved2, res2) = store.save_dirty();
            res2.unwrap();
            assert_eq!((bytes2, saved2), (0, 0));
        }

        // Session 2: load name1 (clean hit!), compile name3 into s2 -> save_dirty
        // writes ONLY s2 while s1 stays clean & untouched.
        let s1_path = shard_path(dir.path(), true, s1);
        let s1_bytes_before = std::fs::read(&s1_path).unwrap();
        {
            let store = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), true, false);
            let mut created = 0usize;
            store
                .load_or_create(
                    name1,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc1) },
                    || {
                        created += 1;
                        unsafe { device.CreateComputePipelineState(&desc1) }
                    },
                )
                .unwrap();
            assert_eq!(created, 0, "name1 must hit shard-{s1:02x} without recompiling");
            assert_eq!(store.opened_shards(), 1);
            assert!(!store.is_legacy_opened());

            store
                .load_or_create(
                    name3,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc3) },
                    || {
                        created += 1;
                        unsafe { device.CreateComputePipelineState(&desc3) }
                    },
                )
                .unwrap();
            assert_eq!(created, 1);

            let (bytes, saved, res) = store.save_dirty();
            res.unwrap();
            assert_eq!(saved, 1, "only dirty shard-{s2:02x} may be written");
            assert!(bytes > 0);
        }
        assert_eq!(std::fs::read(&s1_path).unwrap(), s1_bytes_before);
    }

    #[test]
    fn sharded_library_migrates_from_legacy_host_pipelines_without_recompile_or_modifying_it() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let (root, dxil) = test_compute_pso_desc(&device, 8);
        let desc = make_compute_desc(&root, &dxil);
        let pso_name = "13feedface00000013feedface000000";
        let s = shard_index(pso_name);

        // Seed a legacy monolithic `host-pipelines.bin` with `pso_name`.
        let legacy_path = dir.path().join(LIBRARY_FILE);
        {
            let lib = open_with(&device, Blob::None, false, None).unwrap();
            // SAFETY: valid device, desc, and live pipeline library.
            let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&desc) }.unwrap();
            let wide = HSTRING::from(pso_name);
            unsafe { lib.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.unwrap();
            let size = unsafe { lib.library.GetSerializedSize() };
            serialise(&lib.library, &legacy_path, size).unwrap();
        }
        let legacy_before = std::fs::read(&legacy_path).unwrap();
        assert!(!legacy_before.is_empty());

        // Session 1: empty shard misses -> lazily opens read-only `host-pipelines.bin`
        // -> hits with ZERO recompile -> migrates into shard -> saves tiny shard.
        {
            let store = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), true, false);
            let mut created = 0usize;
            store
                .load_or_create(
                    pso_name,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc) },
                    || {
                        created += 1;
                        unsafe { device.CreateComputePipelineState(&desc) }
                    },
                )
                .unwrap();
            assert_eq!(created, 0, "migrated PSO must load from legacy without recompiling");
            assert!(store.is_legacy_opened());
            assert_eq!(store.migrated_count(), 1);
            let (bytes, saved, res) = store.save_dirty();
            res.unwrap();
            assert_eq!(saved, 1);
            assert!(bytes > 0);
            assert!(shard_path(dir.path(), true, s).is_file());
        }
        // Legacy monolithic file must stay 100% intact and unmodified.
        assert_eq!(std::fs::read(&legacy_path).unwrap(), legacy_before);

        // Session 2: fresh store hits the migrated shard directly and NEVER opens `host-pipelines.bin`.
        {
            let store = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), true, false);
            let mut created = 0usize;
            store
                .load_or_create(
                    pso_name,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc) },
                    || {
                        created += 1;
                        unsafe { device.CreateComputePipelineState(&desc) }
                    },
                )
                .unwrap();
            assert_eq!(created, 0, "migrated shard must serve PSO directly");
            assert_eq!(store.opened_shards(), 1);
            assert!(!store.is_legacy_opened(), "legacy host-pipelines.bin must not be opened on shard hit");
        }
    }

    #[test]
    fn sharded_library_separates_vram_cbv_and_classic_modes() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let (root_classic, dxil_classic) = test_compute_pso_desc(&device, 16);
        let (root_vram, dxil_vram) = test_compute_pso_desc(&device, 32);
        let desc_classic = make_compute_desc(&root_classic, &dxil_classic);
        let desc_vram = make_compute_desc(&root_vram, &dxil_vram);
        let pso_name = "09deadbeef00000009deadbeef000000";
        let s = shard_index(pso_name);

        // Store in classic mode (`shards/classic/shard-09.bin`).
        {
            let classic = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), false, false);
            classic
                .load_or_create(
                    pso_name,
                    |lib, w| unsafe { lib.LoadComputePipeline(w, &desc_classic) },
                    || unsafe { device.CreateComputePipelineState(&desc_classic) },
                )
                .unwrap();
            assert_eq!(classic.save_dirty().1, 1);
        }
        assert!(shard_path(dir.path(), false, s).is_file());
        assert!(!shard_path(dir.path(), true, s).exists());

        // vram_cbv mode uses `shards/vram_cbv/shard-09.bin` and does not collide with classic.
        {
            let vram = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), true, false);
            let mut created = 0usize;
            vram.load_or_create(
                pso_name,
                |lib, w| unsafe { lib.LoadComputePipeline(w, &desc_vram) },
                || {
                    created += 1;
                    unsafe { device.CreateComputePipelineState(&desc_vram) }
                },
            )
            .unwrap();
            assert_eq!(created, 1, "vram_cbv shard must be separate from classic shard");
            assert_eq!(vram.save_dirty().1, 1);
        }
        assert!(shard_path(dir.path(), true, s).is_file());

        // Reloading both modes independently hits each mode's own shard with 0 compiles.
        let classic_re = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), false, false);
        let mut c_made = 0usize;
        classic_re
            .load_or_create(
                pso_name,
                |lib, w| unsafe { lib.LoadComputePipeline(w, &desc_classic) },
                || {
                    c_made += 1;
                    unsafe { device.CreateComputePipelineState(&desc_classic) }
                },
            )
            .unwrap();
        assert_eq!(c_made, 0);

        let vram_re = ShardedLibrary::new(device.clone(), Some(dir.path().to_path_buf()), true, false);
        let mut v_made = 0usize;
        vram_re
            .load_or_create(
                pso_name,
                |lib, w| unsafe { lib.LoadComputePipeline(w, &desc_vram) },
                || {
                    v_made += 1;
                    unsafe { device.CreateComputePipelineState(&desc_vram) }
                },
            )
            .unwrap();
        assert_eq!(v_made, 0);
    }

    #[test]
    fn replaced_shard_file_retains_live_mapped_pages_and_promotes_next() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let sdir = shard_dir(dir.path(), true);
        std::fs::create_dir_all(&sdir).unwrap();
        let target = shard_path(dir.path(), true, 7);
        let next = shard_next_path(dir.path(), true, 7);
        if !write_uncached(&target, LEN) {
            return;
        }
        // Keep the open handle (`holds() == true`) while replacing the shard file.
        let mapped_old = Mapped::open_holding(&target, true).expect("mapped shard");
        let tmp = sdir.join("shard-07.bin.tmp");
        {
            let mut file = Mapped::create(&tmp, LEN).expect("mapped tmp shard");
            file.bytes_mut().fill(0xCC);
        }
        let in_place = replace_file(&tmp, &target, &next).unwrap();
        assert!(!tmp.exists());

        // Every unpaged 4 KiB page of the old shard mapping must still read the original data.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped_old.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert_eq!(
                bytes,
                &expected[..],
                "page {page} of replaced shard lost its contents"
            );
        }
        drop(mapped_old);

        // Next start reads the replacement (promoting `.next` if it could not replace in place).
        let reloaded = Blob::of_paths(&target, &next);
        assert_eq!(reloaded.bytes().len(), LEN);
        assert!(reloaded.bytes().iter().all(|&b| b == 0xCC));
        if !in_place {
            assert!(!next.exists());
        }
    }

    #[test]
    fn sharded_library_concurrent_migration_and_store_is_thread_safe() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let (root, dxil) = test_compute_pso_desc(&device, 64);
        let desc = make_compute_desc(&root, &dxil);
        let pso_name = "3fcafebabe0000003fcafebabe000000";

        // Seed legacy `host-pipelines.bin`.
        let legacy_path = dir.path().join(LIBRARY_FILE);
        {
            let lib = open_with(&device, Blob::None, false, None).unwrap();
            let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&desc) }.unwrap();
            let wide = HSTRING::from(pso_name);
            unsafe { lib.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.unwrap();
            let size = unsafe { lib.library.GetSerializedSize() };
            serialise(&lib.library, &legacy_path, size).unwrap();
        }

        let store = std::sync::Arc::new(ShardedLibrary::new(
            device.clone(),
            Some(dir.path().to_path_buf()),
            true,
            false,
        ));
        let created = std::sync::Arc::new(AtomicU64::new(0));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(4));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let store = store.clone();
                let created = created.clone();
                let barrier = barrier.clone();
                let device = device.clone();
                let root = root.clone();
                let dxil = dxil.clone();
                scope.spawn(move || {
                    let desc = make_compute_desc(&root, &dxil);
                    barrier.wait();
                    store
                        .load_or_create(
                            pso_name,
                            |lib, w| unsafe { lib.LoadComputePipeline(w, &desc) },
                            || {
                                created.fetch_add(1, Ordering::Relaxed);
                                unsafe { device.CreateComputePipelineState(&desc) }
                            },
                        )
                        .unwrap();
                });
            }
        });
        assert_eq!(created.load(Ordering::Relaxed), 0, "all threads must load/migrate with 0 compiles");
        assert_eq!(store.migrated_count(), 1, "exactly one thread should store the migrated entry");
        let (_, saved, res) = store.save_dirty();
        res.unwrap();
        assert_eq!(saved, 1);
    }
}
