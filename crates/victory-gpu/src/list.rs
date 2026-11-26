//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (the title asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: the title's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Vec<u8>>>,
    #[cfg(windows)]
    host: Option<crate::host::HostList>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
    /// The parts of the recording before `host`, in order, each closed: the
    /// list is cut at its waits for memory values ([`cut`]).
    #[cfg(windows)]
    earlier: Vec<Part>,
    /// The waits recorded before the first work of `host`: the queue stands
    /// at them before it runs `host`.
    #[cfg(windows)]
    gate: Vec<ValueWait>,
}

/// One part of a title list cut at its waits: the waits the queue stands
/// at, then the host list that runs once they are met.
#[cfg(windows)]
struct Part {
    waits: Vec<ValueWait>,
    host: crate::host::HostList,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(feature = "diagnostics")]
    crate::copystat::recorded(started);
    #[cfg(feature = "diagnostics")]
    crate::profile::seen(crate::profile::Seen::Record);
}

/// The submission of `lists` has run: each gives its descriptor tables'
/// slots back ([`crate::draw::State::tables_done`]). A list cut at waits
/// keeps them: its state goes with its last part.
#[cfg(windows)]
pub(crate) fn tables_done(lists: &[This]) {
    for &this in lists {
        if this.is_null() {
            continue;
        }
        // SAFETY: the title passes our lists.
        let list = unsafe { &*this.cast::<List>() };
        if list.earlier.is_empty() {
            list.state.lock().unwrap_or_else(|e| e.into_inner()).tables_done();
        }
    }
}

/// The host list behind a title list, if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    unsafe { (*this.cast::<List>()).host.as_ref() }
}

/// The host side of a submitted list, taken from it: a queue that waits
/// for a fence runs it later, and the title may reset the list meanwhile.
/// With it go the upload memory and the dynamic descriptor slots its
/// commands name, kept until it has run.
#[cfg(windows)]
pub(crate) struct Detached {
    host: crate::host::HostList,
    _uploads: Vec<Vec<u8>>,
    _state: crate::draw::State,
}

#[cfg(windows)]
impl Detached {
    pub(crate) fn host(&self) -> &crate::host::HostList {
        &self.host
    }
}

/// A host list that has run and waits to be used again, and the bytes of
/// upload memory it keeps.
#[cfg(windows)]
struct Idle(crate::host::HostList, u64);

// SAFETY: D3D12 objects are free-threaded, and an idle list is touched by
// nobody until it is taken out again.
#[cfg(windows)]
unsafe impl Send for Idle {}

/// Host lists that have run, reset and ready to record: the host side of
/// a list a queue held, or of a part of a list cut at a wait, is taken
/// from the title's list at its submission, and the title's list needs
/// another. Each was made new and thrown away after its run: a hundred a
/// frame in gameplay, each two command allocators and two lists, and for
/// four in ten a budget and an argument buffer (run 61: "104 host lists
/// made 21.2 ms" a frame, and the graphics kernel's time to free them).
#[cfg(windows)]
static IDLE: std::sync::Mutex<(Vec<Idle>, u64)> = std::sync::Mutex::new((Vec::new(), 0));

/// The most host lists kept idle, and the most upload memory they may keep
/// between them (a list keeps scratch chunks over a reset only while lists
/// do not share them, `VICTORY_SCRATCH_POOL=0`).
#[cfg(windows)]
// (512 since run 221: with submissions prepared by a thread of ours the
// title has its lists back at once and resets them while the flights of
// the frame before are still in the air, so more than 128 are out at a
// time: 80 host lists were made and as many thrown away every frame,
// 8 ms to make and more to free, on the thread that submits.)
const IDLE_MOST: (usize, u64) = (512, 256 << 20);

/// Whether host lists that have run are used again (on by default;
/// `VICTORY_LIST_POOL=0`: each is made new and thrown away, as before).
#[cfg(windows)]
fn pools() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIST_POOL").is_none_or(|v| v != "0"))
}

/// A host list open for recording: one that has run, as a title's list is
/// after its reset, or a new one.
#[cfg(windows)]
fn fresh_list() -> Option<crate::host::HostList> {
    let idle = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = idle.0.pop();
        if let Some(Idle(_, bytes)) = &taken {
            idle.1 -= bytes;
        }
        taken
    };
    match idle {
        Some(Idle(list, _)) => Some(list),
        None => crate::host::host().and_then(|h| h.new_list().ok()),
    }
}

/// The host sides of lists whose submission has run (submissions are
/// waited for; `ran` names the ones it ran, by their place in `lists`):
/// what they held is let go, as before, and the host lists are reset and
/// kept for [`fresh_list`].
#[cfg(windows)]
pub(crate) fn recycle(lists: Vec<Detached>, ran: &[usize]) {
    if !pools() {
        return;
    }
    for (i, Detached { host, _uploads, _state }) in lists.into_iter().enumerate() {
        // The descriptor slots and the upload memory its commands named.
        drop(_state);
        drop(_uploads);
        if !ran.contains(&i) || IDLE.lock().unwrap_or_else(|e| e.into_inner()).0.len() >= IDLE_MOST.0 {
            continue;
        }
        // As the title's reset does: what the list kept for its commands
        // goes, its allocators stay.
        host.reset();
        let bytes = host.scratch_kept();
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        if idle.0.len() < IDLE_MOST.0 && idle.1 + bytes <= IDLE_MOST.1 {
            idle.0.push(Idle(host, bytes));
            idle.1 += bytes;
        }
    }
}

/// A wait for a memory value recorded into list `this`, with waits cut
/// into lists (`crate::queue::cuts`). On the console the queue runs the
/// list up to the wait, stands there until the value is in memory, and
/// runs on; the list's bottom-of-pipe writes before the wait are done by
/// then, and other queues may be waiting for exactly those. So the
/// recording is cut here: the host list so far is closed and kept as a
/// part of its own ([`Part`]), the title's calls record into a new host
/// list from here on, and the wait stands before that one. A submission
/// then gives the queue the parts in order ([`detach_parts`]).
///
/// A wait before any work (the first call after a reset, or a wait right
/// after another) cuts nothing: it joins the waits before the current
/// host list.
/// shortcut: a bottom-of-pipe write is done when its part has run, not at
/// its place in the part: a queue waiting for it starts later than on the
/// console, never earlier, and nothing can wait for itself that way (a
/// part has no wait after its first work).
#[cfg(windows)]
fn cut(this: This, wait: ValueWait) {
    if this.is_null() {
        return;
    }
    // SAFETY: the title passes one of our lists; a list is recorded by one
    // thread at a time and is not submitted while it records.
    let list = unsafe { &mut *this.cast_mut().cast::<List>() };
    let Some(current) = list.host.as_ref() else {
        return;
    };
    if current.untouched() {
        list.gate.push(wait);
        return;
    }
    let Some(fresh) = fresh_list() else {
        // shortcut: without a new host list the wait stands before the part
        // it was recorded in.
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no host list for the part after a wait: the wait moves to the part's start"));
        list.gate.push(wait);
        return;
    };
    list.state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .end_batch(current);
    current.close();
    static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        tracing::info!(parts = list.earlier.len() + 2, "a list is cut at a wait for a memory value");
    }
    if let Some(done) = list.host.replace(fresh) {
        let waits = std::mem::replace(&mut list.gate, vec![wait]);
        list.earlier.push(Part { waits, host: done });
    }
    // What the title set so far, on the new host list.
    if let Some(host) = list.host.as_ref() {
        list.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .replay(host);
    }
}

/// Whether list `this` recorded a wait for a memory value (with waits cut
/// into lists).
#[cfg(windows)]
pub(crate) fn cut_list(this: This) -> bool {
    if this.is_null() {
        return false;
    }
    // SAFETY: one of our lists, closed and submitted: nothing records
    // into it.
    let list = unsafe { &*this.cast::<List>() };
    !list.earlier.is_empty() || !list.gate.is_empty()
}

/// What a queue is given for submitted lists that were cut at their waits:
/// the waits it stands at, the host lists it runs once they are met, and
/// the order those run in.
#[cfg(windows)]
pub(crate) type Parts = Vec<(Vec<ValueWait>, Vec<Detached>, Vec<usize>)>;

/// Takes the host sides of the submitted `lists`, part by part ([`cut`]),
/// and leaves each list a new, closed host list for its next reset, as
/// [`detach`] does. What comes back is in the order the queue runs it: a
/// new entry starts at every part that has waits; host lists without
/// waits join the entry before them. The list's upload memory and draw
/// state (its dynamic descriptor slots) go with its last part, which runs
/// last.
/// shortcut: a list the submission names a second time runs again only if
/// it has one part and that part is in the entry being filled; otherwise
/// the second naming is dropped (logged once).
#[cfg(windows)]
pub(crate) fn detach_parts(lists: &[This]) -> Parts {
    let mut parts: Parts = vec![(Vec::new(), Vec::new(), Vec::new())];
    // The lists taken: where its one host list is (entry, place), or none
    // for a list of several parts.
    let mut seen: Vec<(This, Option<(usize, usize)>)> = Vec::new();
    for &this in lists {
        if this.is_null() {
            continue;
        }
        if let Some(&(_, at)) = seen.iter().find(|(t, _)| *t == this) {
            match at {
                Some((entry, place)) if entry + 1 == parts.len() => parts[entry].2.push(place),
                _ => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!("a list named twice by a submission that waits for memory values runs once")
                    });
                }
            }
            continue;
        }
        let Some(fresh) = fresh_list() else {
            continue;
        };
        // A new host list is open; the title's next call on its list is a
        // reset, which wants it closed.
        fresh.close();
        // SAFETY: one of our lists, closed and submitted: nothing records
        // into it until the title resets it.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        let Some(host) = list.host.replace(fresh) else {
            continue;
        };
        let earlier = std::mem::take(&mut list.earlier);
        let gate = std::mem::take(&mut list.gate);
        let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
        let state = std::mem::take(&mut *list.state.lock().unwrap_or_else(|e| e.into_inner()));
        let single = earlier.is_empty() && gate.is_empty();
        let mut pieces: Vec<(Vec<ValueWait>, Detached)> = earlier
            .into_iter()
            .map(|part| {
                (
                    part.waits,
                    Detached {
                        host: part.host,
                        _uploads: Vec::new(),
                        _state: Default::default(),
                    },
                )
            })
            .collect();
        pieces.push((
            gate,
            Detached {
                host,
                _uploads: uploads,
                _state: state,
            },
        ));
        let mut last = None;
        for (waits, detached) in pieces {
            if !waits.is_empty() {
                parts.push((waits, Vec::new(), Vec::new()));
            }
            let entry = parts.len() - 1;
            let (_, taken, order) = &mut parts[entry];
            let place = taken.len();
            order.push(place);
            taken.push(detached);
            last = Some((entry, place));
        }
        seen.push((this, if single { last } else { None }));
    }
    // (The first entry is empty if the first list begins with a wait.)
    parts.retain(|(_, taken, _)| !taken.is_empty());
    parts
}

/// Takes the host sides of submitted lists (each once, however often the
/// submission names it) and leaves each list a new, closed host list for
/// its next reset: what was taken, and which of them run in what order.
#[cfg(windows)]
pub(crate) fn detach(lists: &[This]) -> (Vec<Detached>, Vec<usize>) {
    let mut taken: Vec<(This, Detached)> = Vec::new();
    let mut order = Vec::new();
    for &this in lists {
        if this.is_null() {
            continue;
        }
        if let Some(i) = taken.iter().position(|(t, _)| *t == this) {
            order.push(i);
            continue;
        }
        let Some(fresh) = fresh_list() else {
            continue;
        };
        // A new host list is open; the title's next call on its list is a
        // reset, which wants it closed.
        fresh.close();
        let list = this.cast_mut().cast::<List>();
        // SAFETY: one of our lists, closed and submitted: nothing records
        // into it until the title resets it.
        let (host, uploads, state) = unsafe {
            let Some(host) = (*list).host.replace(fresh) else {
                continue;
            };
            let uploads =
                std::mem::take(&mut *(*list).uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let state =
                std::mem::take(&mut *(*list).state.lock().unwrap_or_else(|e| e.into_inner()));
            (host, uploads, state)
        };
        order.push(taken.len());
        taken.push((
            this,
            Detached {
                host,
                _uploads: uploads,
                _state: state,
            },
        ));
    }
    (taken.into_iter().map(|(_, d)| d).collect(), order)
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

/// `pipeline` is the list's initial pipeline (null: none), bound until the
/// title sets another, as after a reset.
fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        host: crate::host::host().and_then(|h| match h.new_list() {
            Ok(list) => Some(list),
            Err(e) => {
                tracing::warn!(error = %e, "no host command list");
                None
            }
        }),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
        #[cfg(windows)]
        earlier: Vec::new(),
        #[cfg(windows)]
        gate: Vec::new(),
    });
    #[cfg(not(windows))]
    let _ = pipeline;
    object::hand_out(LIST, list, iid, &LIST_IIDS, out)
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        with_state(this, |list, s| s.end_batch(list));
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // A wait recorded and never submitted goes with the recording.
    VALUE_WAITS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|&(list, _)| list != this as usize);
    // Every submission of the list has run (submissions are waited for).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).clear();
    // The parts of a recording that was cut and never submitted go with it
    // (a submission takes the parts it runs).
    #[cfg(windows)]
    {
        // SAFETY: the title passes one of our lists, and neither records
        // into it nor submits it during its reset.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        list.earlier.clear();
        list.gate.clear();
    }
    #[cfg(windows)]
    with_state(this, |host, state| {
        host.reset();
        state.reset(pipeline);
    });
    #[cfg(not(windows))]
    let _ = (this, pipeline);
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                // A depth texture's unordered access is its colour twin's
                // (the host gives the depth texture none): the barrier is
                // on the texture shaders write.
                match crate::resource::twin_resource(b.first) {
                    Some(twin) => twin.as_raw().cast_const().cast(),
                    None => host(b.first)?,
                }
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Notifies [`crate::draw::State::title_barrier`] when the title orders its
/// work via a barrier, barrier group, or pipeline sync.
#[cfg(windows)]
fn title_ordered(this: This, waits: bool) {
    let rules = crate::indirect::run_rules();
    if rules.title_barriers || (rules.order && waits) {
        with_state(this, |list, s| s.title_barrier(list, waits));
    }
}

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    title_ordered(this, false);
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WaitsForBuffers, 1);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

/// Whether one of two resources is a target in a stand-in format and the
/// other is not ([`crate::resource::target_format`]): their texels differ
/// in size, so the host cannot copy between them (logged once).
#[cfg(windows)]
fn unlike(dst: This, src: This) -> bool {
    let unlike = crate::resource::stands_in(dst) != crate::resource::stands_in(src);
    if unlike {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(?dst, ?src, "copy between a stand-in target and title-format texels skipped")
        });
    }
    unlike
}

/// Copies left out because both sides were one host texture
/// ([`copy_resource`], [`copy_texture_region`]), since the start.
#[cfg(windows)]
pub(crate) static SELF_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if unlike(dst, src) {
                return;
            }
            // Two title textures over the same memory have one host
            // texture between them: the copy would be of that texture onto
            // itself, which the host does not allow, and it would change
            // nothing. It was recorded all the same, with the texture left
            // in the source state as the copy's destination (run 205's
            // debug layer: "CopyResource with its destination in the
            // COPY_SOURCE state", 37 times, a 2560x1440 depth texture).
            // Only the two state changes and the copy are left out: what
            // is noted of the textures stays as it was. After
            // triippywhale 6e502b3b.
            let same = windows::core::Interface::as_raw(to) == windows::core::Interface::as_raw(from);
            // A source aliased over a GPU-written texture gets those texels;
            // otherwise a source the title fills itself is uploaded first
            // (upstream PR #671): a copy read whatever the host texture held.
            let aliased = crate::resource::alias_of(src)
                .is_some_and(|gpu_src| crate::texture::alias(list, src, gpu_src));
            if !aliased && crate::texture::title_written(from) {
                crate::texture::upload(list, src, from);
            }
            // Depth textures shaders also write: what they wrote, first.
            list.twin_use(dst, false, true);
            list.twin_use(src, false, false);
            // A destination made level by level gets its levels' memory
            // ahead of this list (`crate::tiles`).
            crate::texture::card_writes(list, dst, to, None);
            crate::texture::host_written(dst);
            if same {
                SELF_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        ?dst,
                        ?src,
                        "a copy between two title textures that share one host texture is left out: the first one"
                    );
                });
                return;
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: live textures.
            let (dst_desc, src_desc) = unsafe { (to.GetDesc(), from.GetDesc()) };
            let bridged = needs_cross_format_bridge(&dst_desc, &src_desc);
            if crate::resource::ped_tex_diag()
                && (crate::resource::is_bc_format(dst_desc.Format.0 as u32)
                    || crate::resource::is_bc_format(src_desc.Format.0 as u32))
            {
                log_ped_copy_resource(
                    &dst_desc,
                    crate::resource::host_tiled(dst).is_some(),
                    &src_desc,
                    bridged,
                );
            }
            if bridged {
                bridge_copy_resource(list, to, &dst_desc, from, &src_desc);
            } else {
                // SAFETY: an open list and live textures of compatible format family.
                unsafe { list.list.CopyResource(to, from) };
            }
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// What unwritten texels of a title-filled texture hold on the first GPU
/// `CopyTextureRegion` into it (`VICTORY_COPY_OVER_TITLE`).
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopyOverTitle {
    /// Leave pool block bytes untouched (`VICTORY_COPY_OVER_TITLE=0`).
    Off,
    /// Upload current title memory first (`VICTORY_COPY_OVER_TITLE=title`).
    Title,
    /// Zero every subresource first (default: prevents stale pool bytes from
    /// bleeding into unwritten regions of partially baked atlases).
    Zeros,
}

#[cfg(any(windows, test))]
fn copy_over_title_from(raw: Option<&str>) -> CopyOverTitle {
    match raw.map(str::trim) {
        Some("0") => CopyOverTitle::Off,
        Some("title") => CopyOverTitle::Title,
        _ => CopyOverTitle::Zeros,
    }
}

#[cfg(windows)]
fn copy_over_title() -> CopyOverTitle {
    static MODE: std::sync::OnceLock<CopyOverTitle> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        copy_over_title_from(std::env::var("VICTORY_COPY_OVER_TITLE").ok().as_deref())
    })
}

/// Records copies of zeros into every subresource of `texture` (or, if
/// `level_mask` is `Some(mask)`, only mip levels whose bit is set in `mask`),
/// which must already be in `COPY_DEST`. Returns `true` if at least one copy
/// was recorded.
#[cfg(windows)]
fn zero_texture_levels(
    list: &crate::host::HostList,
    host: &crate::host::Host,
    texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    level_mask: Option<u32>,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.SampleDesc.Count > 1 {
        return false;
    }
    let mips = u32::from(d.MipLevels);
    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(d.DepthOrArraySize)
    };
    let count = mips * layers;
    if count == 0 {
        return false;
    }
    if let Some(mask) = level_mask {
        let mut wrote = false;
        for level in (0..mips.min(32)).filter(|&l| mask & (1u32 << l) != 0) {
            for layer in 0..layers {
                let sub = layer * mips + level;
                let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let mut bytes = 0u64;
                // SAFETY: outputs for one subresource.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &d,
                        sub,
                        1,
                        0,
                        Some(&mut footprint),
                        None,
                        None,
                        Some(&mut bytes),
                    );
                }
                if bytes == 0 || bytes == u64::MAX {
                    continue;
                }
                let Some((buffer, data)) = list.upload(host, bytes) else {
                    continue;
                };
                // SAFETY: a mapped buffer of `bytes` bytes, just made.
                unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: a borrowed pointer the location never releases.
                    pResource: unsafe { std::mem::transmute_copy(texture) },
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: sub,
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: as above; the list holds the buffer until it is reset.
                    pResource: unsafe { std::mem::transmute_copy(&buffer) },
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: footprint,
                    },
                };
                // SAFETY: an open list; both locations name live resources.
                unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
                wrote = true;
            }
        }
        return wrote;
    }
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut bytes = 0u64;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            None,
            None,
            Some(&mut bytes),
        );
    }
    if bytes == 0 || bytes == u64::MAX {
        return false;
    }
    let Some((buffer, data)) = list.upload(host, bytes) else {
        return false;
    };
    // SAFETY: a mapped buffer of `bytes` bytes, just made.
    unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above; the list holds the buffer until it is reset.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    true
}

/// Records copies of zeros into every subresource of `texture`, which must
/// already be in `COPY_DEST`.
#[cfg(windows)]
fn zero_texture_subresources(
    list: &crate::host::HostList,
    host: &crate::host::Host,
    texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
) {
    zero_texture_levels(list, host, texture, None);
}

/// Whether two host textures belong to different DXGI typeless format families
/// but share the same bytes-per-element (or block) so a cross-format GPU copy
/// must go through an intermediate device-local buffer (e.g., `(W/4 x H/4)`
/// `R32G32B32A32_UINT` -> `(W x H)` `BC3`/`BC5`/`BC7`, `R32G32_UINT` ->
/// `BC1`/`BC4`, `BC3` -> `BC7`, or `R32_UINT` -> `R9G9B9E5`).
#[cfg(windows)]
fn needs_cross_format_bridge(
    dst: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    src: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
) -> bool {
    let (dst_fmt, src_fmt) = (dst.Format.0 as u32, src.Format.0 as u32);
    if crate::resource::family(dst_fmt) == crate::resource::family(src_fmt) {
        return false;
    }
    if dst.SampleDesc.Count > 1
        || src.SampleDesc.Count > 1
        || crate::resource::is_depth_stencil_format(dst_fmt)
        || crate::resource::is_depth_stencil_format(src_fmt)
    {
        return false;
    }
    match (
        crate::resource::element(dst_fmt),
        crate::resource::element(src_fmt),
    ) {
        (Some((db, dk)), Some((sb, sk))) => db == sb && matches!(dk, 1 | 4) && matches!(sk, 1 | 4),
        _ => false,
    }
}

/// Computes source and destination buffer footprints for a cross-format
/// texture-to-texture region copy through a transient GPU buffer.
/// Returns `(buf_bytes, src_buf_fp, dst_buf_fp, clamped_src_box)` where
/// `clamped_src_box` is the source `D3D12_BOX` to read (or `None` when copying
/// the entire source subresource).
#[cfg(any(windows, test))]
#[allow(clippy::too_many_arguments)]
fn cross_format_footprints(
    dst_fmt: u32,
    dst_mip_w: u32,
    dst_mip_h: u32,
    dst_mip_d: u32,
    dst_xyz: [u32; 3],
    src_fmt: u32,
    src_mip_w: u32,
    src_mip_h: u32,
    src_mip_d: u32,
    src_box: Option<[u32; 6]>,
) -> Option<(u64, crate::resource::PlacedFootprint, crate::resource::PlacedFootprint, Option<[u32; 6]>)> {
    let (dst_bpe, dst_blk) = crate::resource::element(dst_fmt)?;
    let (src_bpe, src_blk) = crate::resource::element(src_fmt)?;
    if dst_bpe != src_bpe || !matches!(dst_blk, 1 | 4) || !matches!(src_blk, 1 | 4) {
        return None;
    }
    let (dst_blk_u32, src_blk_u32) = (dst_blk as u32, src_blk as u32);
    let (copy_w, copy_h, copy_d, use_box) = match src_box {
        Some(b) => {
            if b[3] <= b[0] || b[4] <= b[1] || b[5] <= b[2] {
                return None;
            }
            (b[3] - b[0], b[4] - b[1], b[5] - b[2], Some(b))
        }
        None => {
            // Clamp if an oversized scratch UAV is copied without a source box
            // into a smaller mip subresource.
            let dst_avail_blocks_w = dst_mip_w
                .div_ceil(dst_blk_u32)
                .saturating_sub(dst_xyz[0] / dst_blk_u32);
            let dst_avail_blocks_h = dst_mip_h
                .div_ceil(dst_blk_u32)
                .saturating_sub(dst_xyz[1] / dst_blk_u32);
            let dst_avail_d = dst_mip_d.saturating_sub(dst_xyz[2]);
            let src_blocks_w = src_mip_w.div_ceil(src_blk_u32);
            let src_blocks_h = src_mip_h.div_ceil(src_blk_u32);
            if dst_avail_blocks_w == 0 || dst_avail_blocks_h == 0 || dst_avail_d == 0 {
                return None;
            }
            if src_blocks_w > dst_avail_blocks_w
                || src_blocks_h > dst_avail_blocks_h
                || src_mip_d > dst_avail_d
            {
                let cw = (dst_avail_blocks_w * src_blk_u32).min(src_mip_w);
                let ch = (dst_avail_blocks_h * src_blk_u32).min(src_mip_h);
                let cd = dst_avail_d.min(src_mip_d);
                (cw, ch, cd, Some([0, 0, 0, cw, ch, cd]))
            } else {
                (src_mip_w, src_mip_h, src_mip_d, None)
            }
        }
    };
    let blocks_w = u64::from(copy_w.div_ceil(src_blk_u32));
    let blocks_h = u64::from(copy_h.div_ceil(src_blk_u32));
    let depth = u64::from(copy_d);
    let row_bytes = blocks_w.checked_mul(src_bpe)?;
    let row_pitch = row_bytes.max(1).next_multiple_of(256);
    let lines = blocks_h.checked_mul(depth)?;
    let buf_bytes = row_pitch
        .checked_mul(lines.saturating_sub(1))?
        .checked_add(row_bytes)?;
    let row_pitch_u32 = u32::try_from(row_pitch).ok()?;
    let src_fp = crate::resource::PlacedFootprint {
        offset: 0,
        format: src_fmt,
        width: u32::try_from(blocks_w.checked_mul(src_blk)?).ok()?,
        height: u32::try_from(blocks_h.checked_mul(src_blk)?).ok()?,
        depth: copy_d,
        row_pitch: row_pitch_u32,
    };
    let dst_fp = crate::resource::PlacedFootprint {
        offset: 0,
        format: dst_fmt,
        width: u32::try_from(blocks_w.checked_mul(dst_blk)?).ok()?,
        height: u32::try_from(blocks_h.checked_mul(dst_blk)?).ok()?,
        depth: copy_d,
        row_pitch: row_pitch_u32,
    };
    Some((buf_bytes, src_fp, dst_fp, use_box))
}

#[cfg(windows)]
fn subresource_dims(
    desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    sub: u32,
) -> (u32, u32, u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let mips = u32::from(desc.MipLevels).max(1);
    let mip = sub % mips;
    let w = (desc.Width >> mip).max(1) as u32;
    let h = (desc.Height >> mip).max(1);
    let d = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        (u32::from(desc.DepthOrArraySize) >> mip).max(1)
    } else {
        1
    };
    (w, h, d)
}

/// Bridges a cross-format texture-to-texture `CopyTextureRegion` through a
/// transient device-local buffer (both textures must already be in
/// `COPY_DEST` / `COPY_SOURCE`). Returns `true` if recorded.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn bridge_copy_texture_region(
    list: &crate::host::HostList,
    to: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    dst_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    dst_sub: u32,
    x: u32,
    y: u32,
    z: u32,
    from: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    src_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    src_sub: u32,
    src_box: Option<[u32; 6]>,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    let Some(host) = crate::host::host() else {
        return false;
    };
    let (dw, dh, dd) = subresource_dims(dst_desc, dst_sub);
    let (sw, sh, sd) = subresource_dims(src_desc, src_sub);
    let Some((buf_bytes, src_fp, dst_fp, use_box)) = cross_format_footprints(
        dst_desc.Format.0 as u32,
        dw,
        dh,
        dd,
        [x, y, z],
        src_desc.Format.0 as u32,
        sw,
        sh,
        sd,
        src_box,
    ) else {
        return false;
    };
    let Some(buffer) = host.uav_buffer(buf_bytes.next_multiple_of(256)) else {
        return false;
    };
    let common = D3D12_RESOURCE_STATE_COMMON;
    let copy_dst = D3D12_RESOURCE_STATE_COPY_DEST;
    let copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let to_sub_loc = |tex: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed pointer never released by the location.
        pResource: unsafe { std::mem::transmute_copy(tex) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: sub,
        },
    };
    let to_buf_loc = |fp: crate::resource::PlacedFootprint| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed pointer kept alive by `list.keep(&buffer)`.
        pResource: unsafe { std::mem::transmute_copy(&buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            // SAFETY: `PlacedFootprint` matches `D3D12_PLACED_SUBRESOURCE_FOOTPRINT` layout (asserted in `resource.rs`).
            PlacedFootprint: unsafe { std::mem::transmute::<crate::resource::PlacedFootprint, D3D12_PLACED_SUBRESOURCE_FOOTPRINT>(fp) },
        },
    };
    let d3d_box = use_box.map(|b| D3D12_BOX {
        left: b[0],
        top: b[1],
        front: b[2],
        right: b[3],
        bottom: b[4],
        back: b[5],
    });
    // SAFETY: an open list and live resources; buffer is sized for the footprints.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&buffer, common, copy_dst)]);
        list.list.CopyTextureRegion(
            &to_buf_loc(src_fp),
            0,
            0,
            0,
            &to_sub_loc(from, src_sub),
            d3d_box.as_ref().map(std::ptr::from_ref),
        );
        list.list
            .ResourceBarrier(&[crate::host::transition(&buffer, copy_dst, copy_src)]);
        list.list.CopyTextureRegion(
            &to_sub_loc(to, dst_sub),
            x,
            y,
            z,
            &to_buf_loc(dst_fp),
            None,
        );
        list.list
            .ResourceBarrier(&[crate::host::transition(&buffer, copy_src, common)]);
    }
    list.keep(&buffer);
    true
}

/// Bridges a cross-format `CopyResource` between two textures (including
/// `(W/4 x H/4)` uncompressed UAV -> `(W x H)` BC) per subresource through a
/// single transient device-local buffer.
#[cfg(windows)]
fn bridge_copy_resource(
    list: &crate::host::HostList,
    to: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    dst_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    from: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    src_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
) {
    use windows::Win32::Graphics::Direct3D12::*;
    let layers = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            1
        } else {
            u32::from(d.DepthOrArraySize)
        }
    };
    let (dst_mips, src_mips) = (u32::from(dst_desc.MipLevels), u32::from(src_desc.MipLevels));
    let mips = dst_mips.min(src_mips);
    let slice_count = layers(dst_desc).min(layers(src_desc));
    for slice in 0..slice_count {
        for mip in 0..mips {
            let dst_sub = slice * dst_mips + mip;
            let src_sub = slice * src_mips + mip;
            bridge_copy_texture_region(
                list, to, dst_desc, dst_sub, 0, 0, 0, from, src_desc, src_sub, None,
            );
        }
    }
}

/// Translates a buffer-to-texture `PlacedFootprint` to satisfy PC D3D12 rules
/// (256-byte row pitch, BC 4x4 block-aligned dimensions, and format
/// compatibility with `dst_fmt`, including `(W/4 x H/4)` uncompressed block
/// buffers copied into `(W x H)` BC textures).
/// Returns `(host_footprint, adjusted_box, needs_repitch, [host_bytes, row_bytes, lines, src_pitch])`.
#[cfg(any(windows, test))]
fn prepare_upload_footprint(
    src: crate::resource::PlacedFootprint,
    dst_fmt: u32,
    src_box: Option<[u32; 6]>,
) -> Option<(
    crate::resource::PlacedFootprint,
    Option<[u32; 6]>,
    bool,
    [u64; 4],
)> {
    let (src_bpe, src_blk) = crate::resource::element(src.format)?;
    let (mut host_fp, [_, row, lines, src_pitch]) = src.host_rows()?;
    let needs_repitch = src.row_pitch == 0
        || !u64::from(src.row_pitch).is_multiple_of(256)
        || u64::from(src.row_pitch) < row;
    let host_pitch = if needs_repitch {
        u64::from(host_fp.row_pitch)
    } else {
        u64::from(src.row_pitch)
    };
    host_fp.row_pitch = u32::try_from(host_pitch).ok()?;
    let host_bytes = host_pitch
        .checked_mul(lines.saturating_sub(1))?
        .checked_add(row)?;
    let mut adj_box = src_box;
    // Align/translate footprint format to match the destination texture family
    // when both share the same element/block byte width.
    if crate::resource::family(src.format) != crate::resource::family(dst_fmt)
        && !crate::resource::is_depth_stencil_format(dst_fmt)
        && let Some((dst_bpe, dst_blk)) = crate::resource::element(dst_fmt)
        && dst_bpe == src_bpe
    {
        match (src_blk, dst_blk) {
            (1, 4) => {
                host_fp.format = dst_fmt;
                host_fp.width = host_fp.width.checked_mul(4)?;
                host_fp.height = host_fp.height.checked_mul(4)?;
                if let Some(b) = adj_box.as_mut() {
                    b[0] = b[0].checked_mul(4)?;
                    b[1] = b[1].checked_mul(4)?;
                    b[3] = b[3].checked_mul(4)?;
                    b[4] = b[4].checked_mul(4)?;
                }
            }
            (4, 1) => {
                host_fp.format = dst_fmt;
                host_fp.width = host_fp.width.div_ceil(4);
                host_fp.height = host_fp.height.div_ceil(4);
                if let Some(b) = adj_box.as_mut() {
                    b[0] /= 4;
                    b[1] /= 4;
                    b[3] = b[3].div_ceil(4);
                    b[4] = b[4].div_ceil(4);
                }
            }
            (1, 1) | (4, 4) => {
                host_fp.format = dst_fmt;
            }
            _ => {}
        }
    }
    Some((host_fp, adj_box, needs_repitch, [host_bytes, row, lines, src_pitch]))
}

#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn log_ped_copy_texture_region(
    dst_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    dst_sub: u32,
    dst_tiled: bool,
    dst_staged: bool,
    dst_host_wrote: bool,
    src_kind: u32,
    src_sub: u32,
    src_mips: u16,
    src_w: u32,
    src_h: u32,
    src_fmt: u32,
    src_pitch: u32,
    x: u32,
    y: u32,
    src_box: Option<[u32; 6]>,
    bridged: bool,
    bridge_ok: bool,
) {
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<[u64; 15]>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let b = src_box.unwrap_or([u32::MAX; 6]);
    let key = [
        dst_desc.Width,
        u64::from(dst_desc.Height),
        u64::from(dst_desc.Format.0 as u32),
        (u64::from(dst_desc.MipLevels) << 16) | u64::from(dst_desc.DepthOrArraySize),
        u64::from(dst_sub),
        u64::from(dst_tiled)
            | (u64::from(dst_staged) << 1)
            | (u64::from(dst_host_wrote) << 2)
            | (u64::from(bridged) << 3)
            | (u64::from(bridge_ok) << 4),
        u64::from(src_kind),
        (u64::from(src_sub) << 16) | u64::from(src_mips),
        u64::from(src_w),
        u64::from(src_h),
        u64::from(src_fmt),
        u64::from(src_pitch),
        (u64::from(x) << 32) | u64::from(y),
        (u64::from(b[0]) << 32) | u64::from(b[1]),
        (u64::from(b[3]) << 32) | u64::from(b[4]),
    ];
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let cap = crate::resource::ped_tex_diag_cap();
    if seen.len() < cap && seen.insert(key) {
        if bridged && !bridge_ok {
            tracing::warn!(
                frame = crate::frame::current(),
                dst_w = dst_desc.Width,
                dst_h = dst_desc.Height,
                dst_fmt = dst_desc.Format.0,
                dst_mips = dst_desc.MipLevels,
                dst_layers = dst_desc.DepthOrArraySize,
                dst_sub,
                dst_tiled,
                dst_staged,
                dst_host_wrote,
                src_kind,
                src_sub,
                src_mips,
                src_w,
                src_h,
                src_fmt,
                src_pitch,
                x,
                y,
                src_box = ?src_box,
                bridged,
                bridge_ok,
                "VICTORY_PED_TEX_DIAG CopyTextureRegion BC bridge failed"
            );
        } else {
            tracing::info!(
                frame = crate::frame::current(),
                dst_w = dst_desc.Width,
                dst_h = dst_desc.Height,
                dst_fmt = dst_desc.Format.0,
                dst_mips = dst_desc.MipLevels,
                dst_layers = dst_desc.DepthOrArraySize,
                dst_sub,
                dst_tiled,
                dst_staged,
                dst_host_wrote,
                src_kind,
                src_sub,
                src_mips,
                src_w,
                src_h,
                src_fmt,
                src_pitch,
                x,
                y,
                src_box = ?src_box,
                bridged,
                bridge_ok,
                "VICTORY_PED_TEX_DIAG CopyTextureRegion BC"
            );
        }
    }
}

#[cfg(windows)]
fn log_ped_copy_resource(
    dst_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    dst_tiled: bool,
    src_desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    bridged: bool,
) {
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<[u64; 9]>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let key = [
        dst_desc.Width,
        u64::from(dst_desc.Height),
        u64::from(dst_desc.Format.0 as u32),
        (u64::from(dst_desc.MipLevels) << 16) | u64::from(dst_desc.DepthOrArraySize),
        src_desc.Width,
        u64::from(src_desc.Height),
        u64::from(src_desc.Format.0 as u32),
        (u64::from(src_desc.MipLevels) << 16) | u64::from(src_desc.DepthOrArraySize),
        u64::from(dst_tiled) | (u64::from(bridged) << 1),
    ];
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < crate::resource::ped_tex_diag_cap() && seen.insert(key) {
        tracing::info!(
            frame = crate::frame::current(),
            dst_w = dst_desc.Width,
            dst_h = dst_desc.Height,
            dst_fmt = dst_desc.Format.0,
            dst_mips = dst_desc.MipLevels,
            dst_layers = dst_desc.DepthOrArraySize,
            dst_tiled,
            src_w = src_desc.Width,
            src_h = src_desc.Height,
            src_fmt = src_desc.Format.0,
            src_mips = src_desc.MipLevels,
            src_layers = src_desc.DepthOrArraySize,
            bridged,
            "VICTORY_PED_TEX_DIAG CopyResource BC"
        );
    }
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission. Texture to buffer (read back) is not done
/// yet.
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            // Texture to buffer: a read back. The list copies the texels
            // into a host buffer of ours; once it has run they go to the
            // title's buffer (`HostList::defer_read_back`).
            // The title reads two tiny float targets back every frame (5x1
            // and 1x1 RGBA32F, 80 and 16 bytes: run 90), and read a buffer
            // nobody wrote. `VICTORY_READ_BACK=0` leaves them unmade, as
            // before.
            static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let made = read_back(list, dst, x, y, z, src, source_box);
            if n <= 16 || n.is_multiple_of(4096) || made.is_err() {
                static REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if made.is_err() && REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 16 {
                    return;
                }
                let source = crate::resource::host_resource(src.resource).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{}x{} f{} mips {}", d.Width, d.Height, d.Format.0, d.MipLevels)
                });
                tracing::info!(
                    copies = n,
                    made = ?made,
                    source = ?source,
                    source_kind = src.kind,
                    destination_kind = dst.kind,
                    destination_offset = dst.footprint.offset,
                    destination_bytes = ?dst.footprint.bytes(),
                    "a texture to buffer copy (a read back)"
                );
            }
            return;
        };
        if unlike(dst.resource, src.resource) {
            return;
        }
        // SAFETY: a live texture and optional title box.
        let dst_desc = unsafe { to.GetDesc() };
        let dst_fmt = dst_desc.Format.0 as u32;
        let raw_box = (!source_box.is_null()).then(|| unsafe { *source_box });
        let mut adjusted_box = raw_box;
        let mut from = src;
        // Both sides are the same subresource of one host texture (set below).
        let mut same = false;
        let mut src_texture: Option<&windows::Win32::Graphics::Direct3D12::ID3D12Resource> = None;
        let mut src_tex_desc: Option<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC> =
            None;
        let mut bridged = false;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let (Some(host), Some(src_bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some((host_fp, adj_box, needs_repitch, [host_bytes, row, lines, src_pitch])) =
                prepare_upload_footprint(src.footprint, dst_fmt, raw_box)
            else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, src_bytes) else {
                return;
            };
            let Some((buffer, staging)) = list.upload(host, host_bytes) else {
                return;
            };
            if needs_repitch {
                // Non-256B console RowPitch: zero padding once and copy every
                // row into its 256-byte-aligned destination row at submission.
                // SAFETY: `staging` is a freshly mapped upload buffer of `host_bytes` bytes.
                unsafe { std::ptr::write_bytes(staging, 0, host_bytes as usize) };
                let host_pitch = u64::from(host_fp.row_pitch);
                for r in 0..lines {
                    // SAFETY: both offsets stay within `host_bytes` and `src_bytes` (validated above).
                    let dst_row = unsafe { staging.add((r * host_pitch) as usize) };
                    let src_row = unsafe { data.add((r * src_pitch) as usize) };
                    list.defer_fill(dst_row, src_row, row as usize);
                }
                bridged = true;
            } else {
                // The list owns this mapped D3D12 upload buffer. Wine can report
                // its address as MEM_FREE, so use the host-upload path.
                list.defer_fill(staging, data, src_bytes.min(host_bytes) as usize);
                if host_fp.format != src.footprint.format
                    || host_fp.width != src.footprint.width
                    || host_fp.height != src.footprint.height
                {
                    bridged = true;
                }
            }
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint = host_fp;
            adjusted_box = adj_box;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() }, host_bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            // The same subresource of one host texture on both sides (two
            // title textures over one memory): not a copy the host allows.
            // The states and the copy are left out below; the rest is
            // noted as before.
            same = texture.as_raw() == to.as_raw()
                && src.kind != FOOTPRINT
                && dst.kind != FOOTPRINT
                && src.footprint.offset as u32 == dst.footprint.offset as u32;
            list.twin_use(src.resource, false, false);
            let aliased = crate::resource::alias_of(src.resource)
                .is_some_and(|gpu_src| crate::texture::alias(list, src.resource, gpu_src));
            if !aliased && crate::texture::title_written(texture) {
                crate::texture::upload(list, src.resource, texture);
            }
            if !same {
                ensure_state(list, src.resource, COPY_SOURCE);
            }
            // SAFETY: a live texture.
            let sdesc = unsafe { texture.GetDesc() };
            bridged = !same && needs_cross_format_bridge(&dst_desc, &sdesc);
            src_texture = Some(texture);
            src_tex_desc = Some(sdesc);
            from.resource = texture.as_raw().cast_const().cast();
        }
        let dst_sub = dst.footprint.offset as u32;
        let dst_tiled = crate::resource::host_tiled(dst.resource);
        let dst_staged = crate::texture::is_staged(dst.resource);
        let dst_host_wrote = crate::texture::host_wrote(dst.resource as usize);
        // A depth texture shaders also write: what they wrote, first (the
        // source above too).
        list.twin_use(dst.resource, false, true);
        // The first copy into a texture of the title's: what it does not
        // write is zeros, as in a texture that was an allocation of its own
        // (`VICTORY_COPY_OVER_TITLE=title`: what title memory holds; `=0`:
        // whatever the pool block held). A host copy cut from a pool block
        // otherwise holds stale bytes from earlier textures, and no upload
        // runs once a copy wrote it — partially baked atlases (e.g., 2048x2048
        // BC3 character decoration bakes written in 256x256 regions) would
        // composite stale bytes over unwritten areas.
        let over = copy_over_title();
        if over != CopyOverTitle::Off && crate::texture::title_written(to) && !dst_host_wrote {
            // A texture a list uploaded or a draw sampled already holds the
            // title's bytes; on-demand tiled textures have freshly mapped
            // levels zeroed by `card_writes` below, while any already-mapped
            // tiles (e.g. from an earlier unstaged upload of stale pool memory)
            // are zeroed here before the title's copy writes over them.
            let mut touched = false;
            if over == CopyOverTitle::Zeros && !dst_staged {
                if let Some(host) = crate::host::host() {
                    if let Some(ref on_demand) = dst_tiled {
                        if let Some(tiled) = on_demand.tiled() {
                            let all = tiled.every_level();
                            let already = all & !tiled.without(all);
                            if already != 0 && list.first_use(dst.resource as usize) {
                                ensure_state(list, dst.resource, COPY_DEST);
                                touched = zero_texture_levels(list, host, to, Some(already));
                            }
                        }
                    } else if list.first_use(dst.resource as usize) {
                        ensure_state(list, dst.resource, COPY_DEST);
                        zero_texture_subresources(list, host, to);
                        touched = true;
                    } else {
                        crate::texture::upload(list, dst.resource, to);
                        touched = true;
                    }
                }
            } else {
                crate::texture::upload(list, dst.resource, to);
                touched = true;
            }
            // Done before the title's copy writes over it (forces a
            // COPY_DEST -> COPY_SOURCE -> COPY_DEST barrier pair).
            if touched {
                ensure_state(list, dst.resource, COPY_SOURCE);
            }
        }
        // A destination made level by level gets the level's memory ahead
        // of this list (`crate::tiles`): the title's bake copies each level
        // of its result into a block-compressed texture of that kind, and
        // a copy into a level without memory is dropped.
        crate::texture::card_writes(list, dst.resource, to, Some(dst_sub));
        crate::texture::host_written(dst.resource);
        if same {
            SELF_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        ensure_state(list, dst.resource, COPY_DEST);
        let mut bridged_ran = false;
        if bridged
            && src.kind != FOOTPRINT
            && let (Some(src_tex), Some(sdesc)) = (src_texture, src_tex_desc)
            && bridge_copy_texture_region(
                list,
                to,
                &dst_desc,
                dst_sub,
                x,
                y,
                z,
                src_tex,
                &sdesc,
                src.footprint.offset as u32,
                raw_box,
            )
        {
            bridged_ran = true;
        }
        if crate::resource::ped_tex_diag() {
            let (src_w, src_h, src_fmt, src_pitch, src_sub, src_mips) = if src.kind == FOOTPRINT {
                (
                    src.footprint.width,
                    src.footprint.height,
                    src.footprint.format,
                    src.footprint.row_pitch,
                    0u32,
                    0u16,
                )
            } else if let Some(sd) = src_tex_desc {
                (
                    sd.Width as u32,
                    sd.Height,
                    sd.Format.0 as u32,
                    0u32,
                    src.footprint.offset as u32,
                    sd.MipLevels,
                )
            } else {
                (0, 0, 0, 0, 0, 0)
            };
            if crate::resource::is_bc_format(dst_fmt) || crate::resource::is_bc_format(src_fmt) {
                let bridge_ok = !bridged || src.kind == FOOTPRINT || bridged_ran;
                log_ped_copy_texture_region(
                    &dst_desc,
                    dst_sub,
                    dst_tiled.is_some(),
                    dst_staged,
                    dst_host_wrote,
                    src.kind,
                    src_sub,
                    src_mips,
                    src_w,
                    src_h,
                    src_fmt,
                    src_pitch,
                    x,
                    y,
                    raw_box,
                    bridged,
                    bridge_ok,
                );
            }
        }
        if bridged_ran {
            return;
        }
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let d3d_box = adjusted_box.map(|b| D3D12_BOX {
            left: b[0],
            top: b[1],
            front: b[2],
            right: b[3],
            bottom: b[4],
            back: b[5],
        });
        let host_box = d3d_box.as_ref().map(std::ptr::from_ref);
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {d3d_box:?}",
                    dst.resource,
                    dst_desc.Width,
                    dst_desc.Height,
                    dst_desc.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(from.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is constructed in D3D12_BOX layout.
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

/// Records a texture to buffer copy (a read back) of [`copy_texture_region`]
/// into `list`, or says why not.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)] // the method's own arguments
fn read_back(
    list: &crate::host::HostList,
    dst: CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: CopyLocation,
    source_box: *const [u32; 6],
) -> Result<(), &'static str> {
    use windows::Win32::Graphics::Direct3D12::D3D12_BOX;
    use windows::core::Interface;
    const FOOTPRINT: u32 = 1;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_READ_BACK").is_none_or(|v| v != "0")) {
        return Err("switched off");
    }
    if dst.kind != FOOTPRINT || src.kind == FOOTPRINT {
        return Err("not a texture subresource into a placed footprint");
    }
    if (x, y, z) != (0, 0, 0) {
        return Err("a destination corner other than the footprint's");
    }
    let host = crate::host::host().ok_or("no host")?;
    let (footprint, [size, row, rows, pitch]) = dst.footprint.host_rows().ok_or("an unknown format")?;
    // The title's buffer holds every row at its own pitch.
    let bytes = pitch * (rows - 1) + row;
    let data = buffer_range(dst.resource, dst.footprint.offset, bytes).ok_or("a destination outside its buffer")?;
    let texture = crate::resource::host_resource(src.resource).ok_or("no host texture")?;
    let buffer = host.read_back_buffer(size).ok_or("no read-back buffer")?;
    // As a texture source of any copy: what shaders wrote to a depth twin
    // and what the title wrote itself first, then the state.
    list.twin_use(src.resource, false, false);
    if crate::texture::title_written(texture) {
        crate::texture::upload(list, src.resource, texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let mut from = src;
    from.resource = texture.as_raw().cast_const().cast();
    let mut to = dst;
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint = footprint;
    let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "read back of {:?} sub {} into buffer {:?}+{:#x}: {rows} rows of {row} bytes",
            src.resource, src.footprint.offset as u32, dst.resource, dst.footprint.offset
        )
    });
    // SAFETY: an open list; both locations in the host's layout (asserted
    // in `copy_texture_region`), naming a live texture and our buffer, which
    // the list keeps until it is reset; the box is the title's.
    unsafe {
        list.list
            .CopyTextureRegion((&raw const to).cast(), 0, 0, 0, (&raw const from).cast(), host_box)
    };
    list.defer_read_back(buffer, data, [row as usize, rows as usize, pitch as usize]);
    Ok(())
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        if unlike(dst, src) {
            return;
        }
        // Depth textures shaders also write: what they wrote, first.
        list.twin_use(dst, false, true);
        list.twin_use(src, false, false);
        // As for a copy (`crate::tiles`).
        crate::texture::card_writes(list, dst, to, Some(dst_sub));
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_written(dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = if crate::resource::stands_in(dst) { a.Format } else { DXGI_FORMAT(format as i32) };
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(Some(texture), record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        // A depth texture shaders also write: what they wrote, first (the
        // clear may be of one slice or of some rectangles).
        list.twin_use(record.resource, false, true);
        ensure_state(list, record.resource, DEPTH_WRITE);
        if crate::resource::depth_and_unordered(record.resource) {
            crate::resource::DEPTH_UNORDERED[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: keep console flags so thread-count dispatches
/// (`flags & 0x20`, used by the ray-tracing builder) convert to group counts
/// and populate `USER_DISPATCH_THREADS_X`.
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, flags: u32) {
    count_dispatch();
    #[cfg(windows)]
    state::dispatch_with_flags(this, x, y, z, Some(flags));
    #[cfg(not(windows))]
    let _ = (this, x, y, z, flags);
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones (time spans come out as the submission's length).
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
        return;
    };
    if end > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do.
extern "system" fn sync(this: This, _flags: u32, _control: u32, _address: u64, _size: u64) {
    #[cfg(windows)]
    title_ordered(this, false);
    #[cfg(not(windows))]
    let _ = this;
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. The title's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let mut block = vec![0u8; len];
    let start = block.as_mut_ptr();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    // Moving the vector keeps its heap block where it is.
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth order and shading rate hints): any arguments,
/// nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        // (Through the draw state, which keeps them for the host list
        // that continues this one after a cut: `State::replay`.)
        with_state(this, |list, s| s.set_viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.set_scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        with_state(this, |list, s| {
            // SAFETY: the title passes four floats or null.
            s.set_blend_factor(list, unsafe { factor.as_ref() })
        });
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.set_stencil_ref(list, reference));
    }
    pub(super) extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
        with_state(this, |list, s| s.set_depth_bounds(list, min, max));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        dispatch_with_flags(this, x, y, z, None);
    }
    pub(super) fn dispatch_with_flags(this: This, x: u32, y: u32, z: u32, flags: Option<u32>) {
        with_state(this, |list, s| s.dispatch(list, x, y, z, flags));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as depth_bounds;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. The title's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

/// A GPU wait for a memory value: the list's commands after it run when
/// the 32 or 64 bits at `address` compare as asked with `reference`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueWait {
    address: u64,
    comparison: u32,
    reference: u64,
    wide: bool,
}

impl ValueWait {
    /// Whether the memory holds what is waited for now.
    /// shortcut: the comparison codes are taken to be the comparison
    /// functions' (1 never .. 8 always, value against reference); one not
    /// known counts as met, and so does memory that is not mapped.
    ///
    /// The value is read through the kernel, which reports memory that is
    /// not there instead of faulting (as the copies of title memory are
    /// read, `crate::host::copy_mapped`). Asking Windows first whether the
    /// memory is committed (`VirtualQuery`) walks the page tables to the
    /// end of the region: the title keeps these values in a 1.1 GB
    /// allocation (0xd00000000), where one question took about 8 ms, and a
    /// queue asks at every submission and every signal, for each queue that
    /// stands at a wait (the title has five queues). That was
    /// 1.0-2.2 s of every cutscene frame on the thread that submits (runs
    /// 36-49: "between its submissions ... kernel"; 47 of 110 looks at
    /// that thread in run 49 found it in this question).
    /// `VICTORY_WAIT_ASK=1` asks first, as before.
    pub(crate) fn met(&self) -> bool {
        #[cfg(all(windows, feature = "diagnostics"))]
        let started = std::time::Instant::now();
        let met = self.holds();
        #[cfg(all(windows, feature = "diagnostics"))]
        {
            use crate::copystat::{C, add};
            add(C::WaitChecks, 1);
            add(C::WaitCheckNs, started.elapsed().as_nanos() as u64);
        }
        met
    }

    fn holds(&self) -> bool {
        let bytes = if self.wide { 8 } else { 4 };
        if self.address == 0 {
            return true;
        }
        static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ASK.get_or_init(|| std::env::var_os("VICTORY_WAIT_ASK").is_some_and(|v| v == "1"))
            && !crate::texture::committed(self.address as *const u8, bytes)
        {
            return true;
        }
        // The low `bytes` bytes; the rest stays zero.
        let mut raw = [0u8; 8];
        if !crate::host::read_into(self.address as *const u8, &mut raw[..bytes]) {
            return true;
        }
        self.holds_for(u64::from_le_bytes(raw))
    }

    /// Where the value waited for is: address and bytes.
    #[cfg(windows)]
    pub(crate) fn place(&self) -> (usize, usize) {
        (self.address as usize, if self.wide { 8 } else { 4 })
    }

    /// Whether the wait is met with `value` at its address (its low half
    /// for a 32-bit wait).
    pub(crate) fn holds_for(&self, value: u64) -> bool {
        let (value, reference) = if self.wide {
            (value, self.reference)
        } else {
            (value & 0xFFFF_FFFF, self.reference & 0xFFFF_FFFF)
        };
        match self.comparison {
            1 => false,
            2 => value < reference,
            3 => value == reference,
            4 => value <= reference,
            5 => value > reference,
            6 => value != reference,
            7 => value >= reference,
            _ => true,
        }
    }
}

/// The waits lists recorded since their last reset, by list.
static VALUE_WAITS: std::sync::Mutex<Vec<(usize, ValueWait)>> = std::sync::Mutex::new(Vec::new());

/// The waits of the submitted `lists` that are not met yet (taken from
/// them): what the console's GPU would stop at. The title orders work
/// between its queues this way (the title: the compute queue's
/// lighting passes wait for a value the graphics queue's depth pass writes
/// at the bottom of the pipe; run at once, they read a cleared depth
/// buffer and nothing was lit).
/// shortcut: a wait holds its whole list (and submission) back, not only
/// the commands after it.
pub(crate) fn unmet_waits(lists: &[This]) -> Vec<ValueWait> {
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut unmet = Vec::new();
    waits.retain(|&(list, wait)| {
        if !lists.iter().any(|&l| l as usize == list) {
            return true;
        }
        if !wait.met() {
            unmet.push(wait);
        }
        false
    });
    unmet
}

/// The waits each of the submitted `lists` recorded (taken from them),
/// met or not, at the list's first place in the submission.
#[cfg(windows)]
pub(crate) fn waits_by_list(lists: &[This]) -> Vec<Vec<ValueWait>> {
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut by = vec![Vec::new(); lists.len()];
    waits.retain(|&(list, wait)| {
        match lists.iter().position(|&l| l as usize == list) {
            Some(i) => {
                by[i].push(wait);
                false
            }
            None => true,
        }
    });
    by
}

fn wait_for(this: This, address: u64, comparison: u32, reference: u64, wide: bool) {
    let wait = ValueWait {
        address,
        comparison,
        reference,
        wide,
    };
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
        tracing::info!(
            address = format_args!("{address:#x}"),
            comparison,
            reference,
            wide,
            met = wait.met(),
            "a list waits for a memory value"
        );
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some(host) = host_list(this) {
        crate::capture::op(std::ptr::from_ref(host) as usize, || {
            format!(
                "WAIT VALUE at {address:#x} comparison {comparison} reference {reference:#x} wide {wide} met {}",
                wait.met()
            )
        });
    }
    // The list is cut here, and the wait stands before what follows.
    #[cfg(windows)]
    if crate::queue::cuts() {
        cut(this, wait);
        return;
    }
    VALUE_WAITS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((this as usize, wait));
}
extern "system" fn wait_value(this: This, address: u64, comparison: u32, reference: u64) {
    wait_for(this, address, comparison, reference, false);
}
extern "system" fn wait_value64(this: This, address: u64, comparison: u32, reference: u64) {
    wait_for(this, address, comparison, reference, true);
}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    #[cfg(windows)]
    if count > 0 {
        title_ordered(this, true);
    }
    // Diagnostics: counted, and left out under `VICTORY_BARRIER_TEST=groups`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if count > 0 {
        crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
        if crate::copystat::barrier_test() & 1 != 0 {
            return;
        }
    }
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
    value_written(this, address, u64::from(value));
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
    if let Written::Value(value) = what {
        value_written(this, address, value);
    }
}

/// The title wrote a value at the bottom of the pipe (not a timestamp, not
/// a query result): listed (diagnostics), and with `VICTORY_CUT_WRITES=1`
/// the host list is cut right behind it, so that the value is stored when
/// the card has run the list up to there, as on the console, and not when
/// the whole list has run. Lists of other queues wait for these values
/// (the title: the compute list that builds the depth pyramid
/// waits for one, and the meshlet culling of the same frame reads that
/// pyramid). Off by default: every such write then ends a host list, and
/// each part is a submission of its own.
fn value_written(this: This, address: u64, value: u64) {
    #[cfg(windows)]
    {
        if address == 0 || this.is_null() {
            return;
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = value;
        #[cfg(feature = "diagnostics")]
        if let Some(host) = host_list(this) {
            crate::capture::op(std::ptr::from_ref(host) as usize, || {
                format!("WRITE VALUE at {address:#x} value {value:#x}")
            });
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WRITES").is_some_and(|v| v == "1"))
            || !crate::queue::cuts()
        {
            return;
        }
        // SAFETY: the title passes one of our lists; a list is recorded by
        // one thread at a time and is not submitted while it records.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        let Some(current) = list.host.as_ref() else {
            return;
        };
        let Some(fresh) = fresh_list() else {
            return;
        };
        list.state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .end_batch(current);
    current.close();
        if let Some(done) = list.host.replace(fresh) {
            // The part so far keeps its waits; nothing stands before the
            // next one but this part itself.
            let waits = std::mem::take(&mut list.gate);
            list.earlier.push(Part { waits, host: done });
        }
        if let Some(host) = list.host.as_ref() {
            list.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .replay(host);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, address, value);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, state::depth_bounds as *const ()),       // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait_value as *const ()),                // 0x238, 32 bits
    (72, wait_value64 as *const ()),              // 0x240, 64 bits
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 9] = [
    (28, query_mark as *const ()),             // 0xe0 BeginQuery
    (29, query_mark as *const ()),             // 0xe8 EndQuery
    (30, resolve_queries as *const ()),        // 0xf0
    (78, dispatch_x as *const ()),             // 0x270
    (98, state::dispatch_mesh_x as *const ()), // 0x310
    (99, driver_hint as *const ()),            // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// The title's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from the title's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as the title asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    #[test]
    fn copy_over_title_defaults_to_zeros_and_parses_overrides() {
        assert_eq!(copy_over_title_from(None), CopyOverTitle::Zeros);
        assert_eq!(copy_over_title_from(Some("")), CopyOverTitle::Zeros);
        assert_eq!(copy_over_title_from(Some("1")), CopyOverTitle::Zeros);
        assert_eq!(copy_over_title_from(Some("zeros")), CopyOverTitle::Zeros);
        assert_eq!(copy_over_title_from(Some("title")), CopyOverTitle::Title);
        assert_eq!(copy_over_title_from(Some(" title ")), CopyOverTitle::Title);
        assert_eq!(copy_over_title_from(Some("0")), CopyOverTitle::Off);
        assert_eq!(copy_over_title_from(Some(" 0 ")), CopyOverTitle::Off);
    }

    #[cfg(windows)]
    #[test]
    fn zero_texture_subresources_clears_texture_cleanly() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        let Some(host) = crate::host::host() else {
            return;
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 16,
            Height: 16,
            DepthOrArraySize: 1,
            MipLevels: 3,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let mut tex: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap 2D texture creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut tex,
                )
                .expect("create 2D texture");
        }
        let tex = tex.expect("texture created");
        let list = host.new_list().expect("host list");
        zero_texture_subresources(&list, host, &tex);
        list.close();
        host.execute(&[&list]);
        if let Some(err) = host.log_messages() {
            panic!("D3D12 debug layer error in zero_texture_subresources: {err}");
        }
    }

    #[test]
    fn prepare_upload_footprint_repitches_aligns_bc_and_bridges_uav_format() {
        // Unaligned 64-byte RowPitch on 16x16 BC7 (4 block rows of 64 bytes):
        // must re-pitch to 256 bytes and total 3*256 + 64 = 832 bytes.
        let unaligned = crate::resource::PlacedFootprint {
            offset: 128,
            format: 98, // BC7_UNORM
            width: 16,
            height: 16,
            depth: 1,
            row_pitch: 64,
        };
        let (fp, adj_box, repitch, [host_bytes, row, lines, src_pitch]) =
            prepare_upload_footprint(unaligned, 98, None).expect("repitch footprint");
        assert!(repitch);
        assert_eq!(adj_box, None);
        assert_eq!((fp.offset, fp.format, fp.width, fp.height, fp.row_pitch), (0, 98, 16, 16, 256));
        assert_eq!((host_bytes, row, lines, src_pitch), (3 * 256 + 64, 64, 4, 64));

        // 2x2 BC7 tail mip with 256B pitch: width/height must pad up to 4x4 without re-pitch.
        let tail = crate::resource::PlacedFootprint {
            offset: 0,
            format: 98,
            width: 2,
            height: 2,
            depth: 1,
            row_pitch: 256,
        };
        let (fp, _, repitch, [host_bytes, row, lines, _]) =
            prepare_upload_footprint(tail, 98, None).expect("tail footprint");
        assert!(!repitch);
        assert_eq!((fp.width, fp.height, fp.row_pitch), (4, 4, 256));
        assert_eq!((host_bytes, row, lines), (16, 16, 1));

        // (W/4 x H/4) R32G32B32A32_UINT buffer copied into (W x H) BC7_UNORM:
        // format becomes 98, footprint & source box scale by 4.
        let uav_buf = crate::resource::PlacedFootprint {
            offset: 0,
            format: 3, // R32G32B32A32_UINT
            width: 16,
            height: 16,
            depth: 1,
            row_pitch: 256,
        };
        let (fp, adj_box, repitch, [host_bytes, row, lines, _]) =
            prepare_upload_footprint(uav_buf, 98, Some([2, 3, 0, 10, 11, 1]))
                .expect("uav->bc footprint");
        assert!(!repitch);
        assert_eq!((fp.format, fp.width, fp.height, fp.row_pitch), (98, 64, 64, 256));
        assert_eq!(adj_box, Some([8, 12, 0, 40, 44, 1]));
        assert_eq!((host_bytes, row, lines), (15 * 256 + 256, 256, 16));
    }

    #[test]
    fn cross_format_footprints_computes_uav_to_bc_and_bc_to_bc_layouts() {
        // 16x16 R32G32B32A32_UINT sub-box [4,4..12,12] -> 64x64 BC7 at (16,16).
        let (bytes, sfp, dfp, use_box) = cross_format_footprints(
            98,
            64,
            64,
            1,
            [16, 16, 0],
            3,
            16,
            16,
            1,
            Some([4, 4, 0, 12, 12, 1]),
        )
        .expect("uav->bc partial");
        assert_eq!(use_box, Some([4, 4, 0, 12, 12, 1]));
        assert_eq!((sfp.format, sfp.width, sfp.height, sfp.row_pitch), (3, 8, 8, 256));
        assert_eq!((dfp.format, dfp.width, dfp.height, dfp.row_pitch), (98, 32, 32, 256));
        assert_eq!(bytes, 7 * 256 + 128);

        // 1x1 R32G32B32A32_UINT -> 2x2 BC7 mip tail.
        let (bytes, sfp, dfp, use_box) =
            cross_format_footprints(98, 2, 2, 1, [0, 0, 0], 3, 1, 1, 1, None)
                .expect("uav->bc 2x2 tail");
        assert_eq!(use_box, None);
        assert_eq!((sfp.width, sfp.height, dfp.width, dfp.height, sfp.row_pitch), (1, 1, 4, 4, 256));
        assert_eq!(bytes, 16);

        // Oversized 32x32 scratch UAV without box into 32x32 BC7 (8x8 blocks): clamped to 8x8.
        let (_, sfp, dfp, use_box) =
            cross_format_footprints(98, 32, 32, 1, [0, 0, 0], 3, 32, 32, 1, None)
                .expect("clamped scratch uav");
        assert_eq!(use_box, Some([0, 0, 0, 8, 8, 1]));
        assert_eq!((sfp.width, sfp.height, dfp.width, dfp.height), (8, 8, 32, 32));

        // 64x64 BC3 (77) -> 64x64 BC7 (98).
        let (bytes, sfp, dfp, use_box) =
            cross_format_footprints(98, 64, 64, 1, [0, 0, 0], 77, 64, 64, 1, None)
                .expect("bc3->bc7");
        assert_eq!(use_box, None);
        assert_eq!((sfp.format, dfp.format, sfp.width, dfp.width, sfp.row_pitch), (77, 98, 64, 64, 256));
        assert_eq!(bytes, 15 * 256 + 256);
    }

    /// End-to-end GPU test (D3D12 debug layer clean):
    /// 1) Buffer -> BC7 CopyTextureRegion with non-256B console RowPitch (64B)
    /// 2) R32G32B32A32_UINT (16x16) UAV -> BC7 (64x64) partial CopyTextureRegion
    ///    (with unwritten outer region cleanly zeroed)
    /// 3) R32G32B32A32_UINT (16x16) UAV -> BC7 (64x64) whole CopyResource
    #[cfg(windows)]
    #[test]
    fn gpu_bc7_repitch_and_uav_to_bc_copies_pass_debug_layer() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let res_iid = crate::object::guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let queue_iid = crate::object::guid(
            0x0ec8_70a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let alloc_iid = crate::object::guid(
            0x6102_dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = crate::object::guid(
            0x5b16_0d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let qdesc = [0u32; 4];
        assert_eq!(crate::queue::create(null, qdesc.as_ptr().cast(), &queue_iid, &mut queue), S_OK);
        assert_eq!(create_allocator(null, 0, &alloc_iid, &mut allocator), S_OK);
        assert_eq!(create_list(null, 0, 0, allocator, null, &list_iid, &mut list), S_OK);

        let readback_tex = |tex: &ID3D12Resource| -> (Vec<u8>, u32, u32, u64) {
            // SAFETY: live D3D12 texture.
            let d = unsafe { tex.GetDesc() };
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut rows = 0u32;
            let mut row_size = 0u64;
            let mut total = 0u64;
            // SAFETY: plain query.
            unsafe {
                host.device.GetCopyableFootprints(
                    &d,
                    0,
                    1,
                    0,
                    Some(&mut fp),
                    Some(&mut rows),
                    Some(&mut row_size),
                    Some(&mut total),
                );
            }
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let bdesc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total.max(256),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            // SAFETY: plain readback buffer creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &bdesc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut rb,
                    )
                    .unwrap();
            }
            let rb = rb.unwrap();
            let blist = host.new_list().unwrap();
            let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed pointer.
                pResource: unsafe { std::mem::transmute_copy(&rb) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
            };
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed pointer.
                pResource: unsafe { std::mem::transmute_copy(tex) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            // SAFETY: open list; caller transitioned `tex` to COPY_SOURCE.
            unsafe { blist.list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None) };
            blist.close();
            host.execute(&[&blist]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: completed readback buffer.
            let out = unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let slice = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize).to_vec();
                rb.Unmap(0, None);
                slice
            };
            (out, fp.Footprint.RowPitch, rows, row_size)
        };

        // --- Sub-test 1: Buffer -> 16x16 BC7 with non-256B RowPitch = 64 ---
        let bc16_words: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, (1 << 16) | 1, 98, 1, 0, 0, 0, 0];
        // SAFETY: 14 words = 56-byte ResourceDesc.
        let bc16_desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(bc16_words) };
        let mut bc16_backing = vec![0u8; 64 << 10];
        let mut bc16: This = null;
        assert_eq!(
            crate::resource::create_at_address(
                null,
                bc16_backing.as_mut_ptr() as u64,
                &bc16_desc,
                0,
                null.cast(),
                &res_iid,
                &mut bc16
            ),
            S_OK
        );
        let src_buf = crate::resource::test_buffer(512);
        let (buf_ptr, _) = crate::resource::buffer_bytes(src_buf).unwrap();
        let pattern: Vec<u8> = (0..256u32).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
        // SAFETY: 256 bytes inside 512-byte test buffer.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), buf_ptr, 256) };
        let dst_loc = CopyLocation {
            resource: bc16,
            kind: 0,
            footprint: crate::resource::PlacedFootprint {
                offset: 0,
                format: 0,
                width: 0,
                height: 0,
                depth: 0,
                row_pitch: 0,
            },
        };
        let src_loc = CopyLocation {
            resource: src_buf,
            kind: 1,
            footprint: crate::resource::PlacedFootprint {
                offset: 0,
                format: 98,
                width: 16,
                height: 16,
                depth: 1,
                row_pitch: 64, // intentionally non-256B aligned!
            },
        };
        type CopyRegion =
            extern "system" fn(This, *const CopyLocation, u32, u32, u32, *const CopyLocation, *const [u32; 6]);
        slot::<CopyRegion>(list, 0x80)(list, &dst_loc, 0, 0, 0, &src_loc, null.cast());
        ensure_state(host_list(list).unwrap(), bc16, COPY_SOURCE);
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(reset(list, allocator, null), S_OK);
        if let Some(err) = host.log_messages() {
            panic!("D3D12 debug layer error (unaligned BC7 repitch): {err}");
        }
        let (read, pitch, rows, row_size) = readback_tex(crate::resource::host_resource(bc16).unwrap());
        assert_eq!((rows, row_size), (4, 64));
        for r in 0..4usize {
            assert_eq!(
                &read[r * pitch as usize..r * pitch as usize + 64],
                &pattern[r * 64..(r + 1) * 64],
                "unaligned BC7 row {r}"
            );
        }

        // --- Sub-test 2 & 3: 16x16 R32G32B32A32_UINT UAV -> 64x64 BC7 ---
        let uav_words: [u32; 14] = [3, 0, 0, 0, 16, 0, 16, (1 << 16) | 1, 3, 1, 0, 0, 4, 0];
        let bc64_words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, (1 << 16) | 1, 98, 1, 0, 0, 0, 0];
        // SAFETY: 14 words = 56-byte ResourceDesc.
        let uav_desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(uav_words) };
        let bc64_desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(bc64_words) };
        let mut uav_backing = vec![0u8; 64 << 10];
        let mut bc64_a_backing = vec![0u8; 64 << 10];
        let mut bc64_b_backing = vec![0u8; 64 << 10];
        let (mut uav, mut bc64_partial, mut bc64_whole) = (null, null, null);
        assert_eq!(
            crate::resource::create_at_address(
                null,
                uav_backing.as_mut_ptr() as u64,
                &uav_desc,
                0,
                null.cast(),
                &res_iid,
                &mut uav
            ),
            S_OK
        );
        assert_eq!(
            crate::resource::create_at_address(
                null,
                bc64_a_backing.as_mut_ptr() as u64,
                &bc64_desc,
                0,
                null.cast(),
                &res_iid,
                &mut bc64_partial
            ),
            S_OK
        );
        assert_eq!(
            crate::resource::create_at_address(
                null,
                bc64_b_backing.as_mut_ptr() as u64,
                &bc64_desc,
                0,
                null.cast(),
                &res_iid,
                &mut bc64_whole
            ),
            S_OK
        );
        // Seed the 16x16 R32G32B32A32_UINT UAV with unique 16-byte payloads per texel.
        let uav_buf = crate::resource::test_buffer(4096);
        let (uav_ptr, _) = crate::resource::buffer_bytes(uav_buf).unwrap();
        let mut expected_blocks = vec![0u8; 16 * 16 * 16];
        for y in 0..16usize {
            for x in 0..16usize {
                let block = [
                    (x as u8).wrapping_add(1),
                    (y as u8).wrapping_add(1),
                    ((x * 13 + y * 7) & 0xff) as u8,
                    0xa5,
                    0x11,
                    0x22,
                    0x33,
                    0x44,
                    0x55,
                    0x66,
                    0x77,
                    0x88,
                    0x99,
                    0xaa,
                    0xbb,
                    0xcc,
                ];
                expected_blocks[(y * 16 + x) * 16..(y * 16 + x + 1) * 16].copy_from_slice(&block);
            }
        }
        // SAFETY: 4096 bytes inside 4096-byte test buffer (row_pitch = 256 = 16*16).
        unsafe { std::ptr::copy_nonoverlapping(expected_blocks.as_ptr(), uav_ptr, 4096) };
        let seed_dst = CopyLocation {
            resource: uav,
            kind: 0,
            footprint: crate::resource::PlacedFootprint {
                offset: 0,
                format: 0,
                width: 0,
                height: 0,
                depth: 0,
                row_pitch: 0,
            },
        };
        let seed_src = CopyLocation {
            resource: uav_buf,
            kind: 1,
            footprint: crate::resource::PlacedFootprint {
                offset: 0,
                format: 3,
                width: 16,
                height: 16,
                depth: 1,
                row_pitch: 256,
            },
        };
        slot::<CopyRegion>(list, 0x80)(list, &seed_dst, 0, 0, 0, &seed_src, null.cast());

        // Partial copy: 8x8 blocks [4,4..12,12] from UAV -> BC7 at (16, 16) (4x4 blocks).
        let partial_dst = CopyLocation {
            resource: bc64_partial,
            kind: 0,
            footprint: seed_dst.footprint,
        };
        let uav_src = CopyLocation {
            resource: uav,
            kind: 0,
            footprint: seed_dst.footprint,
        };
        let sbox = [4u32, 4, 0, 12, 12, 1];
        slot::<CopyRegion>(list, 0x80)(list, &partial_dst, 16, 16, 0, &uav_src, &sbox);

        // Whole copy: CopyResource UAV (16x16) -> BC7 (64x64).
        slot::<extern "system" fn(This, This, This)>(list, 0x88)(list, bc64_whole, uav);

        ensure_state(host_list(list).unwrap(), bc64_partial, COPY_SOURCE);
        ensure_state(host_list(list).unwrap(), bc64_whole, COPY_SOURCE);
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(reset(list, allocator, null), S_OK);
        if let Some(err) = host.log_messages() {
            panic!("D3D12 debug layer error (UAV->BC7 bridge): {err}");
        }

        let (partial_rb, p_pitch, p_rows, p_row_size) =
            readback_tex(crate::resource::host_resource(bc64_partial).unwrap());
        assert_eq!((p_rows, p_row_size), (16, 256));
        for by in 0..16usize {
            for bx in 0..16usize {
                let off = by * p_pitch as usize + bx * 16;
                let got = &partial_rb[off..off + 16];
                if (4..12).contains(&by) && (4..12).contains(&bx) {
                    let want = &expected_blocks[(by * 16 + bx) * 16..(by * 16 + bx + 1) * 16];
                    assert_eq!(got, want, "partial UAV->BC7 block ({bx},{by})");
                } else {
                    assert_eq!(got, &[0u8; 16], "unwritten BC7 block ({bx},{by}) must be zero");
                }
            }
        }

        let (whole_rb, w_pitch, w_rows, w_row_size) =
            readback_tex(crate::resource::host_resource(bc64_whole).unwrap());
        assert_eq!((w_rows, w_row_size), (16, 256));
        for by in 0..16usize {
            assert_eq!(
                &whole_rb[by * w_pitch as usize..by * w_pitch as usize + 256],
                &expected_blocks[by * 256..(by + 1) * 256],
                "whole CopyResource UAV->BC7 block row {by}"
            );
        }

        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for obj in [bc16, src_buf, uav, bc64_partial, bc64_whole, uav_buf, list, allocator, queue] {
            let _ = release(obj);
        }
    }
}
