//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Mapped `D3D12_HEAP_TYPE_UPLOAD` (or pooled `DEFAULT` UAV for GPU detiling)
/// staging buffer held by a [`Staging`] entry while needed for an upload and
/// detached once the texture settles in VRAM.
struct UploadBuf {
    buffer: ID3D12Resource,
    /// Persistent CPU mapping for `UPLOAD`-heap staging, or null when `vram`.
    data: *mut u8,
    /// Offset in `buffer` of what `data` points at (non-zero only for a
    /// sub-allocated [`Piece`] of a staging block); copy footprints add this.
    base: u64,
    /// Pool bucket size if taken from [`SPARE`] (or `gpu_detile` VRAM pool),
    /// or 0 if unpooled / block-backed.
    pooled: u64,
    /// Staging footprint size counted in [`STAGED`].
    bytes: u64,
    /// Whether `buffer` is a `DEFAULT`-heap UAV buffer filled by
    /// [`crate::gpu_detile`].
    vram: bool,
    /// Committed VRAM bytes tracked by `gpu_detile` (0 when `!vram`).
    vram_alloc: u64,
    /// For `vram`: whether a recycled buffer still holds previous contents
    /// until detiled or explicitly zeroed on the GPU.
    needs_zero: bool,
    /// Sub-allocated range of a [`BLOCKS`] upload block (`VICTORY_STAGING_BLOCKS=1`).
    /// Declared after `buffer` so `buffer` releases before `piece` returns its range.
    piece: Option<Piece>,
}

// SAFETY: a D3D12 upload or VRAM staging buffer and its persistent mapping;
// mutated only under `Staging::upload` (and `Staging::last`) while no GPU copy
// reads it.
unsafe impl Send for UploadBuf {}

impl UploadBuf {
    fn new(host: &crate::host::Host, total: u64) -> Option<Self> {
        let (buffer, data, pooled, piece) = staging_buffer(host, total)?;
        let base = piece.as_ref().map_or(0, |p| p.offset);
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_add(1, Relaxed);
        STAGED[1].fetch_add(total, Relaxed);
        Some(Self {
            buffer,
            data,
            base,
            pooled,
            bytes: total,
            vram: false,
            vram_alloc: 0,
            needs_zero: false,
            piece,
        })
    }

    fn new_for(host: &crate::host::Host, total: u64, try_vram: bool) -> Option<Self> {
        if try_vram
            && let Some((buffer, vram_alloc, pooled)) =
                crate::gpu_detile::take_vram_buf(host, total)
        {
            use std::sync::atomic::Ordering::Relaxed;
            STAGED[0].fetch_add(1, Relaxed);
            STAGED[1].fetch_add(total, Relaxed);
            return Some(Self {
                buffer,
                data: std::ptr::null_mut(),
                base: 0,
                pooled,
                bytes: total,
                vram: true,
                vram_alloc,
                // Always mark `needs_zero` until GPU detile or explicit GPU
                // zero-fill runs: both recycled pool buffers and WARP's
                // freshly created default-heap allocations can contain prior
                // contents if `refresh_timed` exits early without detiling.
                needs_zero: true,
                piece: None,
            });
        }
        Self::new(host, total)
    }
}

impl Drop for UploadBuf {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // Nothing holds the upload buffer any more (command lists keep the COM
        // resource until reset, and sweeps retire after GPU completion).
        if self.vram {
            crate::gpu_detile::release_vram_buf(
                self.buffer.clone(),
                self.vram_alloc,
                self.pooled,
            );
            return;
        }
        if self.pooled != 0 {
            let cap = spare_budget();
            let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            if all.1 + self.pooled <= cap {
                all.1 += self.pooled;
                all.0
                    .entry(self.pooled)
                    .or_default()
                    .push(Spare(self.buffer.clone(), self.data as usize));
            }
        }
    }
}

/// RAII hold preventing a [`Staging`] entry's upload buffer from being
/// released while a command list holds a recorded `CopyTextureRegion`
/// referencing it.
pub(crate) struct UploadHold(Arc<Staging>);

impl Drop for UploadHold {
    fn drop(&mut self) {
        self.0
            .pending_uploads
            .fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
///
/// Lock order DAG: `STAGING` -> `last` -> `upload` -> { `SPARE`, `BLOCKS` }.
/// NEVER acquire `last` or `STAGING` while holding `upload`.
pub(crate) struct Staging {
    upload: Mutex<Option<UploadBuf>>,
    /// Fast-path mirror of `upload.is_some()` for lock-free sweep filtering.
    has_upload: std::sync::atomic::AtomicBool,
    /// Number of live [`UploadHold`] guards held by recorded command lists.
    pending_uploads: std::sync::atomic::AtomicU32,
    /// When a submission last noted an upload (`crate::host::moment`).
    uploaded_at: std::sync::atomic::AtomicU64,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    total: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    row_sizes: Vec<u64>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// The first of them that is uploaded, and the bytes of title memory
    /// read for it and the smaller ones.
    first: u32,
    size: usize,
    /// The staging buffer's bytes (counted in `STAGED` while its [`UploadBuf`] lives).
    bytes: u64,
    /// Made by the pass over textures picked by index, for one upload
    /// ([`once`]): read when the list is submitted, and never looked at again.
    once: std::sync::atomic::AtomicBool,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// The fence of the last late submission that actually reads this
    /// staging buffer. Checked only when new contents will be written.
    last_upload: std::sync::atomic::AtomicU64,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// Whether this staging entry qualifies for GPU compute detiling
    /// ([`crate::gpu_detile`]).
    gpu_detile: bool,
    /// Fast-path mirror of whether a recycled VRAM `UploadBuf` still needs
    /// GPU zeroing if a refresh exits without detiling.
    vram_needs_zero: std::sync::atomic::AtomicBool,
}

impl Drop for Staging {
    fn drop(&mut self) {
        let fence = *self.last_upload.get_mut();
        if let Some(upload) = self
            .upload
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            && fence != 0
            && host().is_some_and(|h| !h.staging_completed(fence))
        {
            // Keep the upload buffer and any sub-allocated block range alive
            // until the GPU finishes the submission reading it.
            crate::host::retire(upload);
        }
    }
}

/// A staging buffer nothing holds: the buffer and where it is mapped.
struct Spare(ID3D12Resource, usize);

// SAFETY: a D3D12 buffer and its mapping, touched by nobody while it is
// spare.
unsafe impl Send for Spare {}

/// Spare staging buffers by size (a power of two), and their bytes in all.
static SPARE: Mutex<(std::collections::BTreeMap<u64, Vec<Spare>>, u64)> =
    Mutex::new((std::collections::BTreeMap::new(), 0));
/// Last millisecond [`staging_buffer_alone`] was called.
static SPARE_USED_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The largest staging buffer that is pooled (16 MiB covers 4K BC7/BC1 mips).
const SPARE_MAX_SINGLE: u64 = 16 << 20;
/// Maximum pooled upload bytes retained after an idle period (`SPARE` / `BLOCKS`).
const SPARE_IDLE_KEEP: u64 = 64 << 20;

/// The most bytes kept spare (`VICTORY_STAGING_POOL_MB`, else 192 MiB on
/// systems with >= 16 GiB free RAM, 128 MiB otherwise).
fn spare_budget() -> u64 {
    static OVERRIDE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    if let Some(bytes) = *OVERRIDE.get_or_init(|| {
        std::env::var("VICTORY_STAGING_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|mb| mb << 20)
    }) {
        return bytes;
    }
    if crate::host::available_physical_ram().is_some_and(|avail| avail >= 16 << 30) {
        192 << 20
    } else {
        128 << 20
    }
}
/// Staging buffers taken from the spare ones / blocks, and made, since the start.
pub(crate) static SPARE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether staging buffers are ranges of big blocks (`VICTORY_STAGING_BLOCKS=1`;
/// off by default: buffers of their own, kept by size). After shadowdog00's
/// vi-pc-combo `0781f61` (GPL-3.0). The spare buffers help only what was
/// used before: a burst of a thousand textures in a frame, held until the
/// frame's lists are reset, makes most of its buffers anew.
fn staging_blocks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| staging_blocks_from(std::env::var("VICTORY_STAGING_BLOCKS").ok().as_deref()))
}

fn staging_blocks_from(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// The size of a block of upload memory and the unit ranges are cut on
/// (a page; copies want 512).
const BLOCK: u64 = 64 << 20;
const GRAIN: u64 = 4096;

/// A block of upload memory, mapped for its life, and what is free in it.
struct Block {
    buffer: ID3D12Resource,
    data: usize,
    ranges: crate::ranges::Ranges,
}
// SAFETY: a free-threaded D3D12 object and the address of its mapping.
unsafe impl Send for Block {}

/// The blocks (`None` where one was given back) and their free bytes.
static BLOCKS: Mutex<(Vec<Option<Block>>, u64)> = Mutex::new((Vec::new(), 0));
/// Last millisecond [`cut_from_blocks`] was called.
static BLOCKS_USED_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A staging buffer's range of a block; dropping it gives the range back
/// (and retires the block if that leaves it completely empty and free bytes
/// exceed `spare_budget() + BLOCK`).
struct Piece {
    block: usize,
    offset: u64,
    len: u64,
}

impl Drop for Piece {
    fn drop(&mut self) {
        let cap = spare_budget();
        let mut gone = None;
        {
            let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
            let (all, free) = &mut *blocks;
            if let Some(Some(block)) = all.get_mut(self.block) {
                block.ranges.give(self.offset, self.len);
                *free += self.len;
                if block.ranges.room() == BLOCK && *free >= cap + BLOCK {
                    *free -= BLOCK;
                    gone = all[self.block].take();
                }
            }
        }
        // Retire outside the lock so the block buffer is never destroyed while
        // any GPU submission is in flight.
        if let Some(gone) = gone {
            crate::host::retire(gone);
        }
    }
}

/// A range of a block for a staging buffer of `total` bytes, if the switch
/// is on ([`staging_blocks`], [`cut_from_blocks`]).
fn staging_range(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, *mut u8, Piece)> {
    if staging_blocks() {
        cut_from_blocks(host, total)
    } else {
        None
    }
}

/// A range of a block for a staging buffer of `total` bytes, all zeros as a
/// new buffer is (levels not there yet must read as zeros): the block's
/// buffer, where the range is mapped, and the range. `None` for a quarter of
/// a block and more, and when the host refuses.
fn cut_from_blocks(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, *mut u8, Piece)> {
    let len = total.max(1).next_multiple_of(GRAIN);
    if len > BLOCK / 4 {
        return None;
    }
    BLOCKS_USED_MS.store(crate::host::now_ms(), std::sync::atomic::Ordering::Relaxed);
    let (buffer, data, piece, used_before, reused_block) = {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        let (all, free) = &mut *blocks;
        let found = all.iter_mut().enumerate().find_map(|(i, b)| {
            let b = b.as_mut()?;
            b.ranges.take(len, GRAIN).map(|(offset, used)| (i, offset, used))
        });
        let (index, offset, used, reused_block) = match found {
            Some((index, offset, used)) => (index, offset, used, true),
            None => {
                let (buffer, data) = host.upload_buffer(BLOCK)?;
                crate::host::name(&buffer, || "staging block".to_owned());
                let mut ranges = crate::ranges::Ranges::new(BLOCK);
                let (offset, used) = ranges.take(len, GRAIN)?;
                let fresh = Block {
                    buffer,
                    data: data as usize,
                    ranges,
                };
                *free += BLOCK;
                let index = match all.iter().position(Option::is_none) {
                    Some(index) => {
                        all[index] = Some(fresh);
                        index
                    }
                    None => {
                        all.push(Some(fresh));
                        all.len() - 1
                    }
                };
                (index, offset, used, false)
            }
        };
        *free -= len;
        let block = all[index].as_ref()?;
        (
            block.buffer.clone(),
            (block.data + offset as usize) as *mut u8,
            Piece {
                block: index,
                offset,
                len,
            },
            used,
            reused_block,
        )
    };
    SPARE_STATS[usize::from(!reused_block)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if used_before {
        // SAFETY: the range's own part of the block's mapping, which nothing
        // else holds.
        unsafe { std::ptr::write_bytes(data, 0, total as usize) };
    }
    Some((buffer, data, piece))
}

/// Frees completely empty upload blocks above `SPARE_IDLE_KEEP` once the block
/// pool has sat idle for `idle_ms`, retiring popped blocks outside the lock.
fn trim_blocks_idle(idle_ms: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let last = BLOCKS_USED_MS.load(Relaxed);
    if last == 0 || crate::host::now_ms().saturating_sub(last) < idle_ms {
        return;
    }
    let keep = SPARE_IDLE_KEEP.min(spare_budget());
    let mut retired = Vec::new();
    {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        let (all, free) = &mut *blocks;
        if *free < keep + BLOCK {
            return;
        }
        for slot in all.iter_mut() {
            if *free < keep + BLOCK {
                break;
            }
            if slot.as_ref().is_some_and(|b| b.ranges.room() == BLOCK) {
                *free -= BLOCK;
                if let Some(block) = slot.take() {
                    retired.push(block);
                }
            }
        }
    }
    for block in retired {
        crate::host::retire(block);
    }
}

/// Frees pooled upload buffers above `SPARE_IDLE_KEEP` once the spare pool has
/// sat idle for `idle_ms`, retiring popped D3D12 resources outside the lock.
fn trim_spare_idle(idle_ms: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    trim_blocks_idle(idle_ms);
    let last = SPARE_USED_MS.load(Relaxed);
    if last == 0 || crate::host::now_ms().saturating_sub(last) < idle_ms {
        return;
    }
    let keep = SPARE_IDLE_KEEP.min(spare_budget());
    let mut retired = Vec::new();
    {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let (buckets, total) = &mut *all;
        if *total <= keep {
            return;
        }
        let sizes: Vec<u64> = buckets.keys().copied().rev().collect();
        for size in sizes {
            if *total <= keep {
                break;
            }
            if let Some(bucket) = buckets.get_mut(&size) {
                while *total > keep {
                    let Some(Spare(buffer, _)) = bucket.pop() else {
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
    for buffer in retired {
        crate::host::retire(buffer);
    }
}

/// A mapped upload buffer of `total` bytes for a texture's staging, all
/// zeros, its pool size (0: not pooled), and its block range if cut from
/// [`BLOCKS`].
///
/// Every texture the title streams in got a buffer of its own from the
/// host and gave it back when its staging went: about a quarter of a
/// millisecond to make (run 213's profile: 25 ms a frame on the recording
/// threads for some 100 uploads, most of it the graphics kernel waiting)
/// and more to free on the thread that submits. A streaming burst is 200
/// to 500 uploads in a frame (run 222), and those are the frames that
/// dip. Buffers of up to 16 MiB are kept by size and used again,
/// zeroed as a new one is (levels not there yet must read as zeros); with
/// `VICTORY_STAGING_BLOCKS=1`, ranges are sub-allocated from 64 MiB blocks.
/// `VICTORY_STAGING_POOL=0`: made and freed each time, as before.
fn staging_buffer(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, *mut u8, u64, Option<Piece>)> {
    match staging_range(host, total) {
        Some((buffer, data, piece)) => Some((buffer, data, 0, Some(piece))),
        None => staging_buffer_alone(host, total).map(|(buffer, data, size)| (buffer, data, size, None)),
    }
}

fn staging_buffer_alone(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let size = total.max(64 << 10).next_power_of_two();
    if size > SPARE_MAX_SINGLE || !*ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_POOL").is_none_or(|v| v != "0")) {
        return host.upload_buffer(total).map(|(buffer, data)| (buffer, data, 0));
    }
    SPARE_USED_MS.store(crate::host::now_ms(), std::sync::atomic::Ordering::Relaxed);
    let spare = {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = all.0.get_mut(&size).and_then(Vec::pop);
        if taken.is_some() {
            all.1 -= size;
        }
        taken
    };
    if let Some(Spare(buffer, data)) = spare {
        SPARE_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: the buffer's own mapping, `size` bytes, which nothing
        // else holds.
        unsafe { std::ptr::write_bytes(data as *mut u8, 0, total as usize) };
        return Some((buffer, data as *mut u8, size));
    }
    SPARE_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    host.upload_buffer(size).map(|(buffer, data)| (buffer, data, size))
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Digest>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// Until this frame the texture is hashed at every check: its memory
    /// changed behind the guard's back (a write through another view of
    /// the same pages, as a video decoder's planes get) when last hashed.
    hot_until: u64,
    /// The frame the hash last found the texture changed in. The texture's
    /// next check hashes it once more (`verify`): if it changed again it is
    /// hot. A texture that streams in changes once and is not hashed on. A
    /// video's planes cycle through a ring of textures, each sampled only
    /// every third frame, so "again" cannot mean "the next frame".
    changed_at: u64,
    verify: bool,
}

/// How many frames a texture that keeps changing stays hashed at every
/// check after its last change.
const HOT_FRAMES: u64 = 24;
/// How long after a change the next check still counts as "again".
const VERIFY_FRAMES: u64 = 24;

impl Last {
    /// The hash found the texture changed in `frame`.
    fn changed(&mut self, frame: u64) {
        if (self.verify && frame <= self.changed_at + VERIFY_FRAMES) || frame < self.hot_until {
            self.hot_until = frame + HOT_FRAMES;
        }
        self.changed_at = frame;
        self.verify = true;
    }

    /// The hash found the texture as it was.
    fn unchanged(&mut self) {
        self.verify = false;
    }

    /// Whether the hash is due at `frame` whatever the turns say.
    fn hash_due(&self, frame: u64) -> bool {
        frame < self.hot_until || (self.verify && frame > self.changed_at && frame <= self.changed_at + VERIFY_FRAMES)
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: RwLock<Option<HashMap<usize, Arc<Staging>>>> = RwLock::new(None);

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    // An upload from title memory can have already used this view. Only
    // a copy from the same GPU source can satisfy this read.
    list.first_use(placed as usize);
    let new_source = list.alias_source_changed(to, from);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !new_source && before.is_none() {
        return true;
    }
    crate::resource::note_draw_epoch_bump();
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (on; `VICTORY_TEXTURE_NO_WAIT=0` waits).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0). On an RX 6800 XT the same wait was
/// 3.1-4.5 s of a 4.4-5.6 s frame during the opening cutscene; with the
/// wait gone, "textures" fell to 10-46 ms and the frame to 1.2-1.5 s
/// (issue #178 has the same measurement on an RX 7900 XT).
///
/// The refresh asks for the lock without waiting. If the runtime has it,
/// the texture is read through the kernel instead, which needs no lock: it
/// fails if a page is not there, and then the texture is left as it is and
/// read at its next refresh, a frame later. A read through the kernel
/// costs about 0.6 ms a megabyte against 0.3 in place, and only when a
/// remap is under way; the in-place read is unchanged otherwise.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| no_wait_from(std::env::var("VICTORY_TEXTURE_NO_WAIT").ok().as_deref()))
}

/// [`no_wait`] from the variable's value: on unless it is `0` (surrounding
/// spaces ignored), so an unset variable means on.
fn no_wait_from(value: Option<&str>) -> bool {
    value.is_none_or(|value| value.trim() != "0")
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Since the frame line last took them, of texture refreshes on every
/// thread, how each ended: refreshes; passed over as the write watch saw
/// no write; read as the write watch saw one; passed over as guarded and
/// unwritten; partly mapped and passed over (read at most every 30 frames),
/// of them found so before waiting for title memory to hold still
/// ([`partial_early`]); partly mapped and read; read in place and found the
/// same; read in place and detiled; waits for title memory to hold still of
/// a millisecond or more; the longest such wait, in microseconds; and the
/// microseconds of detiling (all refreshes).
#[allow(dead_code)]
pub(crate) static REFRESH_ENDS: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];

/// Places in [`REFRESH_ENDS`].
#[allow(dead_code)]
pub(crate) mod ends {
    pub(crate) const ALL: usize = 0;
    pub(crate) const UNWRITTEN: usize = 1;
    pub(crate) const WATCHED: usize = 2;
    pub(crate) const GUARDED: usize = 3;
    pub(crate) const PARTIAL_SKIPPED: usize = 4;
    pub(crate) const PARTIAL_EARLY: usize = 5;
    pub(crate) const PARTIAL_READ: usize = 6;
    pub(crate) const SAME: usize = 7;
    pub(crate) const DETILED: usize = 8;
    pub(crate) const LONG_WAITS: usize = 9;
    pub(crate) const LONGEST_WAIT: usize = 10;
    pub(crate) const DETILE_US: usize = 11;
}

fn ended(place: usize) {
    REFRESH_ENDS[place].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a partly mapped texture that is not due for a read (read in the
/// last 30 frames) is passed over before the refresh waits for title
/// memory to hold still (`VICTORY_PARTIAL_EARLY=0` waits first, as before;
/// on by default).
///
/// Such a texture is passed over either way: under the lock the refresh
/// asks whether all of it is mapped and, told no, returns. The question
/// needs no lock (it reads nothing in place: `QueryWorkingSetEx` and reads
/// through the kernel), so it is put before the wait as well; told yes, the
/// refresh waits and asks again under the lock as before. A texture whose
/// last pages arrive while the runtime is mapping them may then be read at
/// its next refresh instead of after the wait.
fn partial_early() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PARTIAL_EARLY").is_none_or(|v| v != "0"))
}

fn parse_partial_cooldown(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(6)
}

/// Frames to wait before re-reading a partly-mapped streamed texture
/// (`VICTORY_PARTIAL_FRAMES`, default 6).
fn partial_cooldown() -> u64 {
    static FRAMES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FRAMES.get_or_init(|| parse_partial_cooldown(std::env::var("VICTORY_PARTIAL_FRAMES").ok().as_deref()))
}

/// [`crate::watch::title_memory_reading`], its wait counted in
/// [`REFRESH_WAITS`] and, when long, in [`REFRESH_ENDS`].
fn wait_for_title_memory() -> crate::watch::Reading {
    use std::sync::atomic::Ordering::Relaxed;
    let started = std::time::Instant::now();
    let reading = crate::watch::title_memory_reading();
    let us = started.elapsed().as_micros() as u64;
    REFRESH_WAITS[2].fetch_add(us, Relaxed);
    crate::framelog::add(crate::framelog::T::RefreshStillUs, us);
    if us >= 1000 {
        ended(ends::LONG_WAITS);
        REFRESH_ENDS[ends::LONGEST_WAIT].fetch_max(us, Relaxed);
    }
    reading
}

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    let us = started.elapsed().as_micros() as u64;
    REFRESH_WAITS[slot].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
    if slot == 2 {
        crate::framelog::add(crate::framelog::T::RefreshStillUs, us);
    }
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left), shrinking oversized
/// allocations back to 4 MiB after 4K uploads so worker threads do not pin
/// tens of megabytes each.
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    const READ_KEEP_MAX: usize = 4 << 20;
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        let out = f(&mut read[..size]);
        if read.capacity() > READ_KEEP_MAX {
            read.truncate(READ_KEEP_MAX);
            read.shrink_to_fit();
        }
        out
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether active settled textures release their `D3D12_HEAP_TYPE_UPLOAD`
/// staging buffers while retaining lightweight [`Staging`] metadata
/// (`VICTORY_STAGING_RELEASE=0` disables; on by default).
fn staging_release_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| staging_release_from(std::env::var("VICTORY_STAGING_RELEASE").ok().as_deref()))
}

fn staging_release_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds, and detaches the mapped
/// `D3D12_HEAP_TYPE_UPLOAD` buffer from active settled textures once their
/// last upload has completed on the GPU: a staging buffer is system memory
/// the size of its texture, kept so that an unchanged texture is not uploaded
/// again, and retaining upload buffers for active static textures pinned
/// gigabytes of `PAGE_WRITECOMBINE` memory across the run. Active entries
/// keep their lightweight [`Staging`] metadata (hash/watch state) and lazily
/// reallocate an [`UploadBuf`] only if the texture's memory later changes.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::{Acquire, Relaxed};
    const IDLE_FRAMES: u64 = 4;
    const RELEASE_IDLE_MS: u64 = 2_000;
    const RELEASE_IDLE_FRAMES: u64 = 4;
    let idle_ms = crate::host::effective_ram_idle_ms();
    trim_spare_idle(idle_ms);
    crate::gpu_detile::trim_idle(idle_ms);
    let release_on = staging_release_on();
    let host = host();
    let (mut went, mut bytes) = (0u64, 0u64);
    let mut released: Vec<UploadBuf> = Vec::new();
    {
        let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
        let Some(map) = map.as_mut() else {
            return;
        };
        if map.is_empty() {
            return;
        }
        // A list that recorded an upload or a check holds the staging until it
        // is reset: one only the table holds is named by no list, recorded or
        // running.
        map.retain(|_, s| {
            let keep = Arc::strong_count(s) > 1
                || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
            if !keep {
                if s.has_upload.load(Relaxed) {
                    LET_GO[0].fetch_add(1, Relaxed);
                    LET_GO[1].fetch_add(s.bytes, Relaxed);
                    went += 1;
                    bytes += s.bytes;
                }
            } else if release_on
                && s.has_upload.load(Relaxed)
                && s.pending_uploads.load(Acquire) == 0
                && s.uploaded.load(Acquire)
                && !s.dirty.load(Acquire)
                && (!s.on_demand || s.waiting.load(Relaxed) == 0)
                && crate::host::long_ago(s.uploaded_at.load(Relaxed), RELEASE_IDLE_MS, RELEASE_IDLE_FRAMES)
                && host.is_none_or(|h| h.staging_completed(s.last_upload.load(Relaxed)))
                && let Some(u) = s.release_upload()
            {
                released.push(u);
            }
            keep
        });
    }
    if !released.is_empty() {
        let rel_count = released.len() as u64;
        let rel_bytes: u64 = released.iter().map(|u| u.bytes).sum();
        for u in released {
            crate::host::retire(u);
        }
        LET_GO[0].fetch_add(rel_count, Relaxed);
        LET_GO[1].fetch_add(rel_bytes, Relaxed);
        crate::host::trail(|| {
            format!(
                "sweep: {rel_count} settled texture upload buffers released ({} KiB)",
                rel_bytes >> 10
            )
        });
    }
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: RwLock<Option<std::collections::HashSet<usize>>> = RwLock::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    if HOST_WRITTEN
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)))
    {
        return;
    }
    let new = {
        let mut set = HOST_WRITTEN.write().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    {
        let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut() {
            map.remove(&resource);
        }
    }
    {
        let mut set = HOST_WRITTEN.write().unwrap_or_else(|e| e.into_inner());
        if let Some(set) = set.as_mut() {
            set.remove(&resource);
        }
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

thread_local! {
    /// The thread records the upload of a texture picked by index, of which
    /// that many bytes were found mapped just now (`u64::MAX`: not asked).
    pub(crate) static PICKING: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Whether a staging buffer made for one upload of a texture picked by index
/// is filled without asking the write watch and the guards
/// (`VICTORY_TEXTURE_ONCE=0`: with, as before). What they say is for the next
/// refresh, and there is none: the buffer goes with the upload. Asking guards
/// the texture's blocks (a change of protection, under the lock every other
/// refresh wants), and the title's next write there faults to open them.
fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| once_from(std::env::var("VICTORY_TEXTURE_ONCE").ok().as_deref()))
}

fn once_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// A 128-bit digest of `bytes`, to tell a texture's memory changed: four
/// lanes of folded multiplications over 64 bytes at a time. (Not a
/// cryptographic one, which took 0.2 ms a megabyte, four times the detiling
/// it was to spare; nobody makes up textures to collide.)
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| {
        let mut w = [0; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8; 64]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane).wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41)) ^ b;
        }
    };
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        step(block.try_into().unwrap_or(&[0; 64]));
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len) ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    upload_from(list, resource, texture, 0);
}

/// [`upload`] of the mips from `first` on, the ones a view shows: of a
/// placed texture the title may have no memory for the larger ones, and
/// what lies there is another texture's.
#[allow(dead_code)]
pub(crate) fn upload_from(list: &HostList, resource: This, texture: &ID3D12Resource, first: u32) {
    let host_written = HOST_WRITTEN
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) && staged_from(resource) <= first {
        return;
    }
    let Some(staging) = staging(resource, texture, first) else {
        return;
    };
    use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
    if PICKING.get().is_none() {
        // A draw samples it through a table: kept and checked from here on.
        staging.once.store(false, Relaxed);
    }
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Release);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.dirty.store(true, Release);
                staging.waiting.store(0, Release);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    if staging.uploaded.load(Acquire) && !staging.dirty.load(Acquire) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.defer_texture(staging, None);
        return;
    }
    let hold = staging.acquire_upload_hold();
    let (buf, base, reallocated) = {
        let mut guard = staging.upload.lock().unwrap_or_else(|e| e.into_inner());
        let mut reallocated = false;
        if guard.is_none() {
            let Some(host) = host() else {
                return;
            };
            let Some(upload) = UploadBuf::new_for(host, staging.bytes, staging.gpu_detile) else {
                return;
            };
            if !upload.vram && upload.piece.is_none() {
                crate::host::name(&upload.buffer, || {
                    format!("staging reallocated of {resource:?} ({} B)", staging.bytes)
                });
            }
            staging
                .vram_needs_zero
                .store(upload.vram && upload.needs_zero, Release);
            *guard = Some(upload);
            staging.has_upload.store(true, Release);
            reallocated = true;
        }
        let upload = guard.as_ref().expect("upload buffer ensured above");
        (upload.buffer.clone(), upload.base, reallocated)
    };
    if reallocated {
        staging.forget();
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    crate::framelog::add(crate::framelog::T::Uploads, 1);
    const COPY_DEST: u32 = 0x400;
    crate::resource::note_draw_epoch_bump();
    crate::list::ensure_state(list, resource, COPY_DEST);
    // Keep the upload buffer COM reference alive until the list is reset.
    list.keep(&buf);
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buf.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = subresource_bytes(
                footprint.Offset,
                f.RowPitch,
                staging.rows[i],
                f.Depth,
                staging.row_sizes[i],
                staging.bytes,
            ) as u64;
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.total,
                        base + footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + base + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: staging.subresource(i),
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above; `list.keep(&buf)` keeps the resource alive until reset.
            pResource: unsafe { std::mem::transmute_copy(&buf) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: base + footprint.Offset,
                    ..*footprint
                },
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    let Some(on_demand) = on_demand else {
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging, Some(hold));
        return;
    };
    list.defer_texture(staging.clone(), Some(hold));
    list.defer_host(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Acquires an RAII hold preventing this entry's upload buffer from being
    /// released until the recording list is reset or dropped.
    fn acquire_upload_hold(self: &Arc<Self>) -> UploadHold {
        self.pending_uploads
            .fetch_add(1, std::sync::atomic::Ordering::Acquire);
        UploadHold(Arc::clone(self))
    }

    /// Detaches the mapped upload buffer if no command list holds a pending
    /// upload, the texture has been uploaded and is not dirty, and no
    /// on-demand level is waiting for VRAM room.
    fn release_upload(&self) -> Option<UploadBuf> {
        use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
        let mut guard = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            return None;
        }
        if self.pending_uploads.load(Acquire) != 0
            || !self.uploaded.load(Acquire)
            || self.dirty.load(Acquire)
            || (self.on_demand && self.waiting.load(Relaxed) != 0)
        {
            return None;
        }
        let buf = guard.take();
        self.has_upload.store(false, Release);
        buf
    }

    /// Note an upload before checking for changed contents. The check sets
    /// dirty again if the next list needs to upload newer contents.
    pub(crate) fn note_upload(&self) {
        use std::sync::atomic::Ordering::{Relaxed, Release};
        self.uploaded_at.store(crate::host::moment(), Relaxed);
        self.dirty.store(false, Release);
        self.uploaded.store(true, Release);
    }

    pub(crate) fn note_upload_fence(&self, fence: u64) {
        self.last_upload.store(fence, std::sync::atomic::Ordering::Relaxed);
    }

    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.size)
    }

    /// The mip and the array slice of the `i`th part of the staging buffer.
    fn part(&self, i: usize) -> (u32, u32) {
        let mips = self.mips - self.first;
        (self.first + i as u32 % mips, i as u32 / mips)
    }

    /// The texture's subresource the `i`th part of the staging buffer is.
    fn subresource(&self, i: usize) -> u32 {
        let (mip, slice) = self.part(i);
        slice * self.mips + mip
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        if self
            .vram_needs_zero
            .load(std::sync::atomic::Ordering::Acquire)
        {
            self.ensure_vram_zeroed();
        }
        let us = whole.elapsed().as_micros() as u64;
        REFRESH_WAITS[3].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
        crate::framelog::add(crate::framelog::T::RefreshUs, us);
        read
    }

    /// Ensures a recycled VRAM linear buffer is zero-filled if `refresh_timed`
    /// returned early without calling `detile` (e.g., all-zero uncommitted
    /// title memory).
    fn ensure_vram_zeroed(&self) {
        let mut guard = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        let cleared = if let Some(upload) = guard.as_mut()
            && upload.vram
            && upload.needs_zero
            && let Some(host) = host()
        {
            if crate::gpu_detile::zero_buffer(host, &upload.buffer, upload.base, self.bytes) {
                upload.needs_zero = false;
                true
            } else {
                false
            }
        } else {
            true
        };
        if cleared {
            self.vram_needs_zero
                .store(false, std::sync::atomic::Ordering::Release);
        }
    }

    fn refresh_timed(&self) -> usize {
        ended(ends::ALL);
        let size = self.size;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        // Only skip write-watch/guard queries on the first upload of a one-shot
        // picked texture (`last.hash.is_none()`). Because Wave 6 retains lightweight
        // `Staging` metadata in `STAGING` after releasing its `UploadBuf`, later
        // refreshes (e.g. when higher mips finish streaming in) must still digest
        // and detile rather than returning early.
        let skip_watch = self.once.load(std::sync::atomic::Ordering::Relaxed) && last.hash.is_none();
        if !skip_watch && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                ended(ends::UNWRITTEN);
                return 0;
            }
            ended(ends::WATCHED);
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken every eighth frame only, by the
        // texture's address, so the textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !skip_watch
            && *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
        {
            const HASH_EVERY: u64 = 8;
            let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
            last.asked = now;
            // A texture the hash found changed lately is hashed at every
            // check: the Bink video planes of the intro are written through
            // the CPU view of their pages, which the guard does not see, and
            // every eighth frame showed them up to seven frames late (the
            // picture jumped between three older frames).
            let frame = crate::frame::current();
            let turn = frame.wrapping_add((start >> 16) as u64).is_multiple_of(HASH_EVERY) || last.hash_due(frame);
            if unwritten && !turn {
                ended(ends::GUARDED);
                return 0;
            }
        }
        // A partly mapped texture not due for a read is passed over below,
        // once title memory holds still; told so without the wait
        // (`partial_early`).
        if partial_early()
            && last.partial.is_some_and(|at| crate::frame::current() < at + partial_cooldown())
            && !committed(self.memory, size)
        {
            ended(ends::PARTIAL_SKIPPED);
            ended(ends::PARTIAL_EARLY);
            return 0;
        }
        // Kernel copy reads safely without locking against remaps.
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = digest(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    ended(ends::SAME);
                    last.unchanged();
                    return size;
                }
                ended(ends::DETILED);
                if last.hash.is_some() {
                    last.changed(crate::frame::current());
                }
                last.hash = Some(hash);
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now() else {
                return self.refresh_unlocked(last, size);
            };
            mapped
        } else {
            wait_for_title_memory()
        };
        if !committed(self.memory, size) {
            // Kernel `copy_mapped` safely reads pages without holding the remap lock.
            drop(mapped);
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // Re-check every `partial_cooldown()` frames (`VICTORY_PARTIAL_FRAMES`,
            // default 6) so streamed cutscene mips pop in promptly without
            // hashing on every frame.
            let frame = crate::frame::current();
            if last.partial.is_some_and(|at| frame < at + partial_cooldown()) {
                ended(ends::PARTIAL_SKIPPED);
                return 0;
            }
            ended(ends::PARTIAL_READ);
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = digest(read);
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if all_zero(read) {
                    return false;
                }
                self.detile(read);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = digest(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            ended(ends::SAME);
            last.unchanged();
            return size;
        }
        ended(ends::DETILED);
        if last.hash.is_some() {
            last.changed(crate::frame::current());
        }
        last.hash = Some(hash);
        self.detile(tiled);
        drop(mapped);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. If a staging buffer has never been filled
    /// yet, kernel `copy_mapped` reads every page that is mapped (zeros for
    /// the rest) so first uploads and bakes never stall or upload empty memory;
    /// otherwise nothing is taken and the texture is looked at again at its
    /// next refresh. Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                if last.hash.is_some() {
                    return false;
                }
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                if all_zero(read) {
                    return false;
                }
                last.hash = Some(digest(read));
                last.partial = Some(crate::frame::current());
                last.asked = 0;
                self.detile(read);
                return true;
            }
            let hash = digest(read);
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        let started = std::time::Instant::now();
        self.detile_timed(tiled);
        REFRESH_ENDS[ends::DETILE_US].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    fn detile_timed(&self, tiled: &[u8]) {
        use std::sync::atomic::Ordering::{Relaxed, Release};
        let fence = self.last_upload.swap(0, Relaxed);
        if fence != 0 {
            host().expect("an upload has a host").wait_for_staging(fence);
        }
        let mut guard = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let Some(host) = host() else {
                return;
            };
            let Some(upload) = UploadBuf::new_for(host, self.bytes, self.gpu_detile) else {
                return;
            };
            self.vram_needs_zero
                .store(upload.vram && upload.needs_zero, Release);
            *guard = Some(upload);
            self.has_upload.store(true, Release);
        }
        let upload = guard.as_mut().expect("upload buffer ensured above");
        self.dirty.store(true, Release);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        if all_zero(tiled) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        if upload.vram {
            let Some(host) = host() else {
                return;
            };
            if crate::gpu_detile::detile(
                host,
                &upload.buffer,
                upload.base,
                self.bytes,
                &self.layout,
                tiled,
                &self.footprints,
                &self.rows,
                &self.row_sizes,
                self.first,
                self.mips,
                held,
            ) {
                upload.needs_zero = false;
                self.vram_needs_zero.store(false, Release);
            }
            return;
        }
        crate::gpu_detile::note_cpu_detile();
        let data = upload.data;
        for (i, ((footprint, &rows), &row_size)) in self
            .footprints
            .iter()
            .zip(&self.rows)
            .zip(&self.row_sizes)
            .enumerate()
        {
            let (mip, slice) = self.part(i);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_bytes(
                footprint.Offset,
                f.RowPitch,
                rows,
                f.Depth,
                row_size,
                self.bytes,
            );
            // SAFETY: `footprint.Offset + len <= self.bytes`, inside the mapped staging buffer,
            // and `guard` holds `self.upload` for the duration of the write.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Byte span of one placed subresource inside a D3D12 staging buffer of `total`
/// bytes. `GetCopyableFootprints` omits trailing row-pitch padding after the
/// last row of the last subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_bytes(
    offset: u64,
    pitch: u32,
    rows: u32,
    depth: u32,
    row_size: u64,
    total: u64,
) -> usize {
    if rows == 0 || depth == 0 {
        return 0;
    }
    let pitch = u64::from(pitch);
    let rows = u64::from(rows);
    let depth = u64::from(depth);
    let exact = (depth - 1) * pitch * rows + (rows - 1) * pitch + row_size;
    exact.min(total.saturating_sub(offset)) as usize
}

/// Vectorizable all-zero check, 32 bytes at a time.
fn all_zero(bytes: &[u8]) -> bool {
    let (chunks, tail) = bytes.as_chunks::<32>();
    chunks.iter().all(|c| *c == [0u8; 32]) && tail.iter().all(|&b| b == 0)
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
}

/// Where a staged title texture's memory is: address and own bytes
/// (stopping before the next placed texture's start).
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    let at = at as u64;
    Some((at, crate::descriptor::host::texture_own_bytes(at, len as u64)))
}

/// A fingerprint of the texture's own title memory within `[at, at + len)`
/// (stopping before the next placed texture's start so neighbours packed in
/// its large levels do not trigger false re-uploads), never 0: of all of it
/// up to 48 KiB, else of 16 KiB at its start, its middle and its end (a
/// texture streams in whole mips; the small ones lie first). Read through
/// the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = crate::descriptor::host::texture_own_bytes(at, len) as usize;
    let mut print = 0u128;
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            print = print.rotate_left(37) ^ digest(&read[..bytes]).0;
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let first = (print as u64 ^ (print >> 64) as u64).to_le_bytes();
    u64::from_le_bytes(first) | 1
}

/// Has the next list that uploads a title texture with a staging buffer
/// copy it whole, changed or not since: what the check at a submission
/// finds changed is copied by the next list that samples the texture
/// through a table, and there may be none for one that shaders now pick by
/// index alone.
#[allow(dead_code)]
pub(crate) fn upload_again(resource: This) {
    let staging = {
        let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
        map.as_ref()
            .and_then(|map| map.get(&(resource as usize)).cloned())
    };
    if let Some(staging) = staging {
        staging
            .uploaded
            .store(false, std::sync::atomic::Ordering::Release);
        staging.last.lock().unwrap_or_else(|e| e.into_inner()).partial = None;
    }
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
/// (Whether it is mapped is asked only if `ask`; else it counts as mapped.)
pub(crate) fn staged(resource: This, ask: bool) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    Some((s.size as u64, !ask || committed(s.memory, s.size)))
}

/// The first mip a title texture's staging buffer holds; `u32::MAX` without
/// one.
fn staged_from(resource: This) -> u32 {
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .map_or(u32::MAX, |s| s.first)
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
/// Queries up to 256 arbitrary 4 KiB page-aligned addresses in one
/// `K32QueryWorkingSetEx` call, falling back to a one-byte kernel read
/// only for pages not currently in the working set.
fn query_pages(pages: &[usize], out: &mut [bool]) {
    use std::mem::MaybeUninit;
    use windows_sys::Win32::System::ProcessStatus::{
        K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    debug_assert!(pages.len() <= 256 && pages.len() == out.len());
    let n = pages.len();
    if n == 0 {
        return;
    }
    let mut info: [MaybeUninit<PSAPI_WORKING_SET_EX_INFORMATION>; 256] =
        [const { MaybeUninit::uninit() }; 256];
    for (entry, &page_addr) in info[..n].iter_mut().zip(pages) {
        // SAFETY: plain zeroed POD struct with VirtualAddress populated.
        let mut item: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
        item.VirtualAddress = page_addr as *mut _;
        entry.write(item);
    }
    // SAFETY: `info[..n]` is initialized; our own process handle.
    let asked = unsafe {
        K32QueryWorkingSetEx(
            GetCurrentProcess(),
            info.as_mut_ptr().cast(),
            (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
        )
    } != 0;
    for i in 0..n {
        // SAFETY: `info[i]` was initialized above and filled by the query.
        let valid = asked && unsafe { info[i].assume_init_ref().VirtualAttributes.Flags } & 1 != 0;
        let mut byte = [0u8];
        out[i] = valid || crate::host::read_into(pages[i] as *const u8, &mut byte);
    }
}

fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use std::mem::MaybeUninit;
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    let mut info: [MaybeUninit<PSAPI_WORKING_SET_EX_INFORMATION>; 256] =
        [const { MaybeUninit::uninit() }; 256];
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            // SAFETY: plain zeroed POD struct with VirtualAddress populated.
            let mut item: PSAPI_WORKING_SET_EX_INFORMATION = unsafe { std::mem::zeroed() };
            item.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
            entry.write(item);
        }
        // SAFETY: `info[..n]` is initialized; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: `entry` was initialized above and filled by the query.
            let valid = asked && unsafe { entry.assume_init_ref().VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    if !by_query() {
        const KERNEL_PAGE: usize = 4096;
        let first_page = start & !(KERNEL_PAGE - 1);
        let last_page = (end - 1) & !(KERNEL_PAGE - 1);
        // Probe both endpoints in a single K32QueryWorkingSetEx syscall.
        if first_page == last_page {
            let mut ok = [false];
            query_pages(&[first_page], &mut ok);
            return if ok[0] { len as u64 } else { 0 };
        }
        let mut end_ok = [false; 2];
        query_pages(&[first_page, last_page], &mut end_ok);
        // Neither endpoint committed means the texture is absent: walking every
        // 64 KiB block across uncommitted reservations stalls on RTX GPUs.
        if !end_ok[0] && !end_ok[1] {
            return 0;
        }
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each, batched up to 256 blocks
        // (16 MiB) per K32QueryWorkingSetEx call.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        let mut addrs = [0usize; 256];
        let mut spans = [0u64; 256];
        let mut ok = [false; 256];
        while block < end {
            let mut count = 0usize;
            while block < end && count < 256 {
                let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
                addrs[count] = lo & !(KERNEL_PAGE - 1);
                spans[count] = (hi - lo) as u64;
                count += 1;
                block = block.saturating_add(BLOCK);
            }
            query_pages(&addrs[..count], &mut ok[..count]);
            for i in 0..count {
                if ok[i] {
                    there += spans[i];
                }
            }
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// The staging state of a title texture from mip `first` on, made on first
/// use (and made anew for larger mips than the one there holds).
fn staging(resource: This, texture: &ID3D12Resource, first: u32) -> Option<Arc<Staging>> {
    // Fast read path: avoid exclusive contention across recording threads.
    if let Some(s) = STAGING
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .filter(|s| s.first <= first)
    {
        return Some(s.clone());
    }
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order. Perform
    // footprint queries, D3D12 resource creation, and buffer zeroing outside
    // the global STAGING lock so other recording threads never stall on them.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let s = new_staging(resource, texture, on_demand, first)?;
    let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(existing) = map.get(&(resource as usize))
        && existing.first <= s.first
    {
        return Some(existing.clone());
    }
    map.insert(resource as usize, s.clone());
    Some(s)
}

/// A staging buffer for `resource` (host copy `texture`) made and detiled
/// off the submitting thread (`VICTORY_ASYNC_TEXTURES`): not in the table
/// yet, so that the pass over the heap's textures still tells a texture
/// draws read through a table (it has a staging buffer) from one it
/// uploads itself; [`adopt`] puts it there when the pass records the
/// upload.
///
/// Read from title memory now, as the submission's check would: the same
/// write-watch asking (or hash, or guard) is noted in it, so the check that
/// still runs at the submission that uploads it (`HostList::defer_texture`)
/// reads again only what the title wrote since, as for any staging buffer.
#[allow(dead_code)]
pub(crate) fn prepare(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let s = new_staging(resource, texture, on_demand, 0)?;
    s.refresh_now();
    Some(s)
}

/// Puts a staging buffer made by [`prepare`] in the table, for the upload
/// the caller records next to find. False if the texture has one already
/// (a draw staged it meanwhile): that one is used, and `staging` let go.
#[allow(dead_code)]
pub(crate) fn adopt(resource: This, staging: Arc<Staging>) -> bool {
    let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if map.contains_key(&(resource as usize)) {
        return false;
    }
    staging
        .used
        .store(crate::host::moment(), std::sync::atomic::Ordering::Relaxed);
    map.insert(resource as usize, staging);
    true
}

impl Staging {
    /// The bytes of its upload buffer's staging layout.
    #[allow(dead_code)]
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// A title texture's staging state from mip `first` on, made and not put in
/// the table. `on_demand`: the host texture's levels are made on demand
/// (`crate::resource::host_tiled`), asked before the table is locked.
fn new_staging(
    resource: This,
    texture: &ID3D12Resource,
    on_demand: bool,
    first: u32,
) -> Option<Arc<Staging>> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    // Only where the title's layout can end behind the smaller mips (and
    // not with levels on demand, which judge each level themselves).
    let first = if !on_demand && first < u32::from(d.MipLevels) && layout.bytes_from(first) < layout.size {
        first
    } else {
        0
    };
    let size = layout.bytes_from(first) as usize;
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize) - first
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut row_sizes = vec![0u64; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let gpu_detile = d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D
        && crate::gpu_detile::eligible(
            &layout,
            size,
            total,
            &footprints,
            &rows,
            &row_sizes,
            first,
            u32::from(d.MipLevels),
        );
    let upload = UploadBuf::new_for(host, total, gpu_detile)?;
    let vram_needs_zero = upload.vram && upload.needs_zero;
    if !upload.vram && upload.piece.is_none() {
        crate::host::name(&upload.buffer, || {
            format!(
                "staging of {resource:?} {}x{}x{} f{}",
                d.Width, d.Height, d.DepthOrArraySize, d.Format.0
            )
        });
    }
    tracing::debug!(?desc, total, first, gpu_detile, vram = upload.vram, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        upload: Mutex::new(Some(upload)),
        has_upload: std::sync::atomic::AtomicBool::new(true),
        pending_uploads: std::sync::atomic::AtomicU32::new(0),
        uploaded_at: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "diagnostics")]
        total,
        footprints,
        rows,
        row_sizes,
        layout,
        mips: u32::from(d.MipLevels),
        first,
        size,
        bytes: total,
        once: std::sync::atomic::AtomicBool::new(once() && !on_demand && PICKING.get().is_some()),
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        last_upload: std::sync::atomic::AtomicU64::new(0),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        gpu_detile,
        vram_needs_zero: std::sync::atomic::AtomicBool::new(vram_needs_zero),
    });
    Some(s)
}

#[cfg(test)]
mod staging_tests {
    #[test]
    fn staging_blocks_toggle_defaults_off_and_one_enables() {
        use super::staging_blocks_from;
        assert!(!staging_blocks_from(None));
        assert!(!staging_blocks_from(Some("0")));
        assert!(!staging_blocks_from(Some("yes")));
        assert!(staging_blocks_from(Some("1")));
        assert!(staging_blocks_from(Some(" 1 ")));
    }

    /// A range of a block used before reads as zeros again, as a buffer the
    /// host made would; two ranges held at once do not overlap.
    #[test]
    fn a_range_used_before_is_zeros_again() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let cut = |total| super::cut_from_blocks(host, total).unwrap();
        let (buffer, data, piece) = cut(10_000);
        let (_, other, second) = cut(5_000);
        assert_eq!(piece.offset % super::GRAIN, 0);
        assert!(
            piece.block != second.block
                || piece.offset + piece.len <= second.offset
                || second.offset + second.len <= piece.offset
        );
        // SAFETY: the ranges' own mapped bytes.
        unsafe {
            std::ptr::write_bytes(data, 0xee, 10_000);
            std::ptr::write_bytes(other, 0xdd, 5_000);
        }
        let (block, offset) = (piece.block, piece.offset);
        drop((buffer, piece));
        let (_, again, third) = cut(10_000);
        // SAFETY: as above.
        let bytes = unsafe { std::slice::from_raw_parts(again, 10_000) };
        assert!(bytes.iter().all(|&b| b == 0), "old bytes in a range cut again");
        assert_eq!((third.block, third.offset), (block, offset));
        // The one still held is untouched.
        // SAFETY: as above.
        assert!(unsafe { std::slice::from_raw_parts(other, 5_000) }.iter().all(|&b| b == 0xdd));
        // Too long for a block: the caller makes a buffer of its own.
        assert!(super::cut_from_blocks(host, super::BLOCK / 4 + 1).is_none());
    }

    /// A block-backed `UploadBuf` integrates cleanly with Wave 6 `release_upload`
    /// and `STAGED` accounting, returning its sub-allocated range upon drop.
    #[test]
    fn block_backed_upload_buf_releases_and_reuses_range() {
        use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
        use std::sync::{Arc, Mutex};
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let words: [u32; 14] = [3, 0, 0, 0, 16, 0, 4, 65537, 10, 1, 0, 256, 3_145_728, 0];
        // SAFETY: 14 words = 56-byte ResourceDesc.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = crate::resource::texture_layout(&desc)
            .expect("a console layout")
            .planes[0]
            .layout
            .clone();
        const BYTES: u64 = 4096;
        let (buffer, data, piece) = super::cut_from_blocks(host, BYTES).expect("block range");
        let expected = (piece.block, piece.offset);
        let before_count = super::STAGED[0].load(Relaxed);
        let before_bytes = super::STAGED[1].load(Relaxed);
        super::STAGED[0].fetch_add(1, Relaxed);
        super::STAGED[1].fetch_add(BYTES, Relaxed);
        let upload = super::UploadBuf {
            buffer,
            data,
            base: piece.offset,
            pooled: 0,
            bytes: BYTES,
            vram: false,
            vram_alloc: 0,
            needs_zero: false,
            piece: Some(piece),
        };
        let staging = Arc::new(super::Staging {
            upload: Mutex::new(Some(upload)),
            has_upload: AtomicBool::new(true),
            pending_uploads: AtomicU32::new(0),
            uploaded_at: AtomicU64::new(0),
            #[cfg(feature = "diagnostics")]
            total: BYTES,
            footprints: Vec::new(),
            rows: Vec::new(),
            row_sizes: Vec::new(),
            size: layout.size as usize,
            layout,
            mips: 1,
            first: 0,
            bytes: BYTES,
            once: AtomicBool::new(false),
            memory: std::ptr::null(),
            last: Mutex::default(),
            dirty: AtomicBool::new(false),
            uploaded: AtomicBool::new(true),
            last_upload: AtomicU64::new(0),
            used: AtomicU64::new(0),
            on_demand: false,
            detiled: AtomicU32::new(0),
            judged: [const { AtomicU64::new(0) }; 2],
            waiting: AtomicU64::new(0),
            gpu_detile: false,
            vram_needs_zero: AtomicBool::new(false),
        });
        let hold = staging.acquire_upload_hold();
        assert!(staging.release_upload().is_none());
        drop(hold);
        let released = staging.release_upload().expect("settled block UploadBuf releases");
        assert!(!staging.has_upload.load(Relaxed));
        drop(released);
        assert_eq!(super::STAGED[0].load(Relaxed), before_count);
        assert_eq!(super::STAGED[1].load(Relaxed), before_bytes);
        let (_, _, reused) = super::cut_from_blocks(host, BYTES).expect("reused range");
        assert_eq!((reused.block, reused.offset), expected);
    }

    /// End-to-end `Staging` integration test comparing `VICTORY_GPU_DETILE=1`
    /// vs `=0` across a full mip chain (including mip tails), Wave 6 upload
    /// buffer release + reallocation, and content mutation re-upload.
    #[test]
    fn gpu_detile_staging_end_to_end_matches_cpu() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let res_iid = crate::object::guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );

        let read_all_mips = |tex: &ID3D12Resource| -> Vec<u8> {
            // SAFETY: live D3D12 texture.
            let d = unsafe { tex.GetDesc() };
            let count = u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize);
            let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
            let mut sub_rows = vec![0u32; count as usize];
            let mut sub_row_sizes = vec![0u64; count as usize];
            let mut total = 0u64;
            // SAFETY: plain footprint query.
            unsafe {
                host.device.GetCopyableFootprints(
                    &d,
                    0,
                    count,
                    0,
                    Some(fps.as_mut_ptr()),
                    Some(sub_rows.as_mut_ptr()),
                    Some(sub_row_sizes.as_mut_ptr()),
                    Some(&mut total),
                );
            }
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total.max(4),
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
                        &buf_desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut rb,
                    )
                    .unwrap();
            }
            let rb = rb.unwrap();
            let list = host.new_list().unwrap();
            for (sub, fp) in fps.iter().enumerate() {
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: borrowed COM pointer.
                    pResource: unsafe { std::mem::transmute_copy(tex) },
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: sub as u32,
                    },
                };
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: borrowed COM pointer.
                    pResource: unsafe { std::mem::transmute_copy(&rb) },
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp },
                };
                // SAFETY: open list; `tex` was left in COPY_SOURCE by caller.
                unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
            }
            list.close();
            host.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: readback completed via `host.execute`. Extract only
            // active row bytes so uninitialized readback row-pitch padding
            // never causes false mismatches.
            unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let raw = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize);
                let mut out = Vec::new();
                for ((fp, &rows), &row_size) in
                    fps.iter().zip(&sub_rows).zip(&sub_row_sizes)
                {
                    let base = fp.Offset as usize;
                    let pitch = fp.Footprint.RowPitch as usize;
                    let active = row_size as usize;
                    for y in 0..rows as usize {
                        out.extend_from_slice(&raw[base + y * pitch..base + y * pitch + active]);
                    }
                }
                rb.Unmap(0, None);
                out
            }
        };

        let run_case = |words: [u32; 14], mutate: bool| {
            // SAFETY: 14 u32 words = 56-byte ResourceDesc.
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("valid console layout")
                .planes[0]
                .layout
                .clone();
            let initial: Vec<u8> = (0..layout.size as usize)
                .map(|i| ((i.wrapping_mul(131) ^ (i >> 7)) & 0xff) as u8)
                .collect();
            let updated: Vec<u8> = (0..layout.size as usize)
                .map(|i| ((i.wrapping_mul(197).wrapping_add(41) ^ (i >> 5)) & 0xff) as u8)
                .collect();

            let run_mode = |use_gpu: bool| -> (Vec<u8>, Option<Vec<u8>>) {
                let _guard = crate::gpu_detile::OverrideGuard::set(use_gpu);
                let mut backing = initial.clone();
                let mut res: crate::object::This = std::ptr::null();
                let hr = crate::resource::create_at_address(
                    std::ptr::null(),
                    backing.as_ptr() as u64,
                    &desc,
                    0,
                    std::ptr::null(),
                    &res_iid,
                    &mut res,
                );
                assert_eq!(hr, 0, "create_at_address failed");
                let tex = crate::resource::host_resource(res).expect("host texture");

                let list = host.new_list().unwrap();
                super::upload(&list, res, tex);
                crate::list::ensure_state(&list, res, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
                list.close();
                host.execute(&[&list]);
                if let Some(err) = host.log_messages() {
                    panic!("D3D12 debug layer error (use_gpu={use_gpu}, initial): {err}");
                }

                // Verify Staging used the expected mode (VRAM when GPU detile is on).
                let staging = {
                    let map = super::STAGING.read().unwrap_or_else(|e| e.into_inner());
                    map.as_ref().unwrap().get(&(res as usize)).unwrap().clone()
                };
                assert_eq!(staging.gpu_detile, use_gpu);
                {
                    let guard = staging.upload.lock().unwrap_or_else(|e| e.into_inner());
                    assert_eq!(guard.as_ref().unwrap().vram, use_gpu);
                }

                let first_read = read_all_mips(tex);
                drop(list);

                let second_read = if mutate {
                    // First upload's `refresh()` runs after `note_upload()` and
                    // marks `dirty = true`; settle it cleanly before testing
                    // Wave 6 upload buffer release and pool reallocation.
                    staging.note_upload();
                    let released = staging.release_upload();
                    assert!(released.is_some(), "settled upload buffer should release");
                    drop(released);

                    backing.copy_from_slice(&updated);
                    super::upload_again(res);
                    let list2 = host.new_list().unwrap();
                    super::upload(&list2, res, tex);
                    crate::list::ensure_state(
                        &list2,
                        res,
                        D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32,
                    );
                    list2.close();
                    host.execute(&[&list2]);
                    if let Some(err) = host.log_messages() {
                        panic!("D3D12 debug layer error (use_gpu={use_gpu}, reupload): {err}");
                    }
                    Some(read_all_mips(tex))
                } else {
                    None
                };

                // Release the title resource cleanly.
                super::forget(res as usize);
                // SAFETY: valid COM object returned by `create_at_address`.
                unsafe {
                    type Release = unsafe extern "system" fn(crate::object::This) -> u32;
                    let vtable = *res.cast::<*const [Release; 3]>();
                    ((*vtable)[2])(res);
                }
                (first_read, second_read)
            };

            let (gpu_first, gpu_second) = run_mode(true);
            let (cpu_first, cpu_second) = run_mode(false);
            assert_eq!(gpu_first, cpu_first, "initial GPU vs CPU upload mismatch");
            assert_eq!(gpu_second, cpu_second, "re-upload GPU vs CPU mismatch");
        };

        // Case 1: 256x256 BC7, 9 mips (with mip tail), SW_64KB_D (0x10a) + reupload.
        run_case(
            [3, 0, 0, 0, 256, 0, 256, (9 << 16) | 1, 98, 1, 0, 0x10a, 0, 0],
            true,
        );
        // Case 2: 131x79 RGBA8, 8 mips (NPOT with mip tail), SW_4KB_S (0x105).
        run_case(
            [3, 0, 0, 0, 131, 0, 79, (8 << 16) | 1, 28, 1, 0, 0x105, 0, 0],
            true,
        );
    }

    /// End-to-end batched `Staging` test exercising `BatchGuard::begin()` ->
    /// multiple textures (including 4–16 KiB textures newly eligible under
    /// `MIN_BYTES = 4 KiB`) refreshed in ONE batch with zero CPU waits ->
    /// `BatchGuard::flush(host.queue())` -> direct-queue `CopyTextureRegion` ->
    /// readback 100% bit-exact against CPU detiling with zero D3D12 debug errors.
    #[test]
    fn gpu_detile_batched_staging_end_to_end_matches_cpu() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let res_iid = crate::object::guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );

        let read_all_mips = |tex: &ID3D12Resource| -> Vec<u8> {
            // SAFETY: live D3D12 texture.
            let d = unsafe { tex.GetDesc() };
            let count = u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize);
            let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
            let mut sub_rows = vec![0u32; count as usize];
            let mut sub_row_sizes = vec![0u64; count as usize];
            let mut total = 0u64;
            // SAFETY: plain footprint query.
            unsafe {
                host.device.GetCopyableFootprints(
                    &d,
                    0,
                    count,
                    0,
                    Some(fps.as_mut_ptr()),
                    Some(sub_rows.as_mut_ptr()),
                    Some(sub_row_sizes.as_mut_ptr()),
                    Some(&mut total),
                );
            }
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total.max(4),
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
                        &buf_desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut rb,
                    )
                    .unwrap();
            }
            let rb = rb.unwrap();
            let list = host.new_list().unwrap();
            for (sub, fp) in fps.iter().enumerate() {
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: borrowed COM pointer.
                    pResource: unsafe { std::mem::transmute_copy(tex) },
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: sub as u32,
                    },
                };
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    // SAFETY: borrowed COM pointer.
                    pResource: unsafe { std::mem::transmute_copy(&rb) },
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp },
                };
                // SAFETY: open list; `tex` was left in COPY_SOURCE by caller.
                unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
            }
            list.close();
            host.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: readback completed via `host.execute`.
            unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let raw = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize);
                let mut out = Vec::new();
                for ((fp, &rows), &row_size) in
                    fps.iter().zip(&sub_rows).zip(&sub_row_sizes)
                {
                    let base = fp.Offset as usize;
                    let pitch = fp.Footprint.RowPitch as usize;
                    let active = row_size as usize;
                    for y in 0..rows as usize {
                        out.extend_from_slice(&raw[base + y * pitch..base + y * pitch + active]);
                    }
                }
                rb.Unmap(0, None);
                out
            }
        };

        let cases: [[u32; 14]; 3] = [
            // 32x32 RGBA8 (4 KiB tiled, SW_4KB_S 0x105) -> verifies lowered 4 KiB MIN_BYTES!
            [3, 0, 0, 0, 32, 0, 32, (1 << 16) | 1, 28, 1, 0, 0x105, 0, 0],
            // 256x256 BC7 9-mip with mip tail (SW_64KB_D 0x10a)
            [3, 0, 0, 0, 256, 0, 256, (9 << 16) | 1, 98, 1, 0, 0x10a, 0, 0],
            // 131x79 RGBA8 8-mip NPOT with mip tail (SW_4KB_S 0x105)
            [3, 0, 0, 0, 131, 0, 79, (8 << 16) | 1, 28, 1, 0, 0x105, 0, 0],
        ];

        let run_all = |use_gpu: bool| -> Vec<Vec<u8>> {
            let _guard = crate::gpu_detile::OverrideGuard::set(use_gpu);
            let mut backings: Vec<Vec<u8>> = Vec::new();
            let mut resources: Vec<(crate::object::This, &ID3D12Resource)> = Vec::new();
            for (idx, words) in cases.iter().enumerate() {
                // SAFETY: 14 u32 words = 56-byte ResourceDesc.
                let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(*words) };
                let layout = crate::resource::texture_layout(&desc)
                    .expect("valid console layout")
                    .planes[0]
                    .layout
                    .clone();
                let backing: Vec<u8> = (0..layout.size as usize)
                    .map(|i| {
                        ((i.wrapping_mul(131 + idx * 17).wrapping_add(idx * 29) ^ (i >> 7)) & 0xff)
                            as u8
                    })
                    .collect();
                backings.push(backing);
            }
            let list = host.new_list().unwrap();
            for (words, backing) in cases.iter().zip(&backings) {
                // SAFETY: 14 u32 words = 56-byte ResourceDesc.
                let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(*words) };
                let mut res: crate::object::This = std::ptr::null();
                let hr = crate::resource::create_at_address(
                    std::ptr::null(),
                    backing.as_ptr() as u64,
                    &desc,
                    0,
                    std::ptr::null(),
                    &res_iid,
                    &mut res,
                );
                assert_eq!(hr, 0, "create_at_address failed");
                let tex = crate::resource::host_resource(res).expect("host texture");
                super::upload(&list, res, tex);
                crate::list::ensure_state(&list, res, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
                resources.push((res, tex));
            }
            list.close();

            if use_gpu {
                // Refresh all 3 textures inside ONE async compute batch and
                // flush via GPU-side `direct_queue.Wait` right before executing `list`!
                let batch = crate::gpu_detile::BatchGuard::begin();
                for &(res, _) in &resources {
                    let staging = {
                        let map = super::STAGING.read().unwrap_or_else(|e| e.into_inner());
                        map.as_ref().unwrap().get(&(res as usize)).unwrap().clone()
                    };
                    assert!(staging.gpu_detile, "all 3 cases (>= 4 KiB) must use gpu_detile");
                    staging.refresh_now();
                }
                batch.flush(host.queue());
            }

            host.execute(&[&list]);
            if let Some(err) = host.log_messages() {
                panic!("D3D12 debug layer error (batched use_gpu={use_gpu}): {err}");
            }

            let out: Vec<Vec<u8>> = resources.iter().map(|&(_, tex)| read_all_mips(tex)).collect();
            drop(list);
            for (res, _) in resources {
                super::forget(res as usize);
                // SAFETY: valid COM object returned by `create_at_address`.
                unsafe {
                    type Release = unsafe extern "system" fn(crate::object::This) -> u32;
                    let vtable = *res.cast::<*const [Release; 3]>();
                    ((*vtable)[2])(res);
                }
            }
            out
        };

        let gpu_out = run_all(true);
        let cpu_out = run_all(false);
        assert_eq!(gpu_out.len(), cpu_out.len());
        for (i, (g, c)) in gpu_out.iter().zip(&cpu_out).enumerate() {
            assert_eq!(g, c, "batched end-to-end GPU vs CPU mismatch on case {i}");
        }
    }
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    #[test]
    fn staging_release_toggle_defaults_on_and_zero_disables() {
        use super::staging_release_from;
        assert!(staging_release_from(None));
        assert!(staging_release_from(Some("1")));
        assert!(staging_release_from(Some("yes")));
        assert!(!staging_release_from(Some("0")));
        assert!(!staging_release_from(Some(" 0 ")));
    }

    #[test]
    fn upload_hold_refcounts_and_release_upload_gating() {
        use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
        use std::sync::{Arc, Mutex};
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        assert!(host.staging_completed(0));
        let words: [u32; 14] = [3, 0, 0, 0, 16, 0, 4, 65537, 10, 1, 0, 256, 3_145_728, 0];
        // SAFETY: 14 words = 56-byte ResourceDesc.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = crate::resource::texture_layout(&desc)
            .expect("a console layout")
            .planes[0]
            .layout
            .clone();
        const BYTES: u64 = 4096;
        let before_count = super::STAGED[0].load(Relaxed);
        let before_bytes = super::STAGED[1].load(Relaxed);
        let upload = super::UploadBuf::new(host, BYTES).expect("host upload buffer");
        assert_eq!(super::STAGED[0].load(Relaxed), before_count + 1);
        assert_eq!(super::STAGED[1].load(Relaxed), before_bytes + BYTES);
        let staging = Arc::new(super::Staging {
            upload: Mutex::new(Some(upload)),
            has_upload: AtomicBool::new(true),
            pending_uploads: AtomicU32::new(0),
            uploaded_at: AtomicU64::new(0),
            #[cfg(feature = "diagnostics")]
            total: BYTES,
            footprints: Vec::new(),
            rows: Vec::new(),
            row_sizes: Vec::new(),
            size: layout.size as usize,
            layout,
            mips: 1,
            first: 0,
            bytes: BYTES,
            once: AtomicBool::new(false),
            memory: std::ptr::null(),
            last: Mutex::default(),
            dirty: AtomicBool::new(true),
            uploaded: AtomicBool::new(false),
            last_upload: AtomicU64::new(0),
            used: AtomicU64::new(0),
            on_demand: false,
            detiled: AtomicU32::new(0),
            judged: [const { AtomicU64::new(0) }; 2],
            waiting: AtomicU64::new(0),
            gpu_detile: false,
            vram_needs_zero: AtomicBool::new(false),
        });
        // Not uploaded yet and dirty: must block release.
        assert!(staging.release_upload().is_none());
        assert!(staging.has_upload.load(Relaxed));

        // Note upload: marks uploaded=true, dirty=false.
        staging.note_upload();
        assert!(staging.uploaded.load(Relaxed));
        assert!(!staging.dirty.load(Relaxed));

        // UploadHold RAII refcounting blocks release while pending_uploads > 0.
        let hold1 = staging.acquire_upload_hold();
        let hold2 = staging.acquire_upload_hold();
        assert_eq!(staging.pending_uploads.load(Relaxed), 2);
        assert!(staging.release_upload().is_none());
        drop(hold1);
        assert_eq!(staging.pending_uploads.load(Relaxed), 1);
        assert!(staging.release_upload().is_none());
        drop(hold2);
        assert_eq!(staging.pending_uploads.load(Relaxed), 0);

        // Dirty flag blocks release even when pending_uploads == 0 and uploaded == true.
        staging.dirty.store(true, Relaxed);
        assert!(staging.release_upload().is_none());
        staging.dirty.store(false, Relaxed);

        // Once pending_uploads == 0 && uploaded && !dirty, release succeeds cleanly.
        let released = staging.release_upload();
        assert!(released.is_some());
        assert!(!staging.has_upload.load(Relaxed));
        assert!(staging.release_upload().is_none());

        // Dropping the released UploadBuf decrements STAGED once; dropping
        // Staging afterward must not double-decrement.
        drop(released);
        assert_eq!(super::STAGED[0].load(Relaxed), before_count);
        assert_eq!(super::STAGED[1].load(Relaxed), before_bytes);
        drop(staging);
        assert_eq!(super::STAGED[0].load(Relaxed), before_count);
        assert_eq!(super::STAGED[1].load(Relaxed), before_bytes);
    }

    #[test]
    fn texture_refreshes_do_not_wait_unless_asked_to() {
        use super::no_wait_from;
        assert!(no_wait_from(None));
        assert!(no_wait_from(Some("1")));
        assert!(no_wait_from(Some("yes")));
        assert!(!no_wait_from(Some("0")));
        assert!(!no_wait_from(Some(" 0 ")));
    }

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }

    #[test]
    fn subresource_bytes_stays_within_staging_buffer_for_unpadded_last_row() {
        use super::subresource_bytes;
        // 16x4 RGBA16F: RowPitch = 256, row_size = 128, D3D12 total = 3*256 + 128 = 896 (< 1024).
        let span = subresource_bytes(0, 256, 4, 1, 128, 896);
        assert_eq!(span, 896);
        // 2-mip 4x4 RGBA8: mip 0 at 0 (row_size=16), mip 1 (1x1) at 1024 (row_size=4), total = 1028.
        let mip0 = subresource_bytes(0, 256, 4, 1, 16, 1028);
        let mip1 = subresource_bytes(1024, 256, 1, 1, 4, 1028);
        assert_eq!(mip0, 784);
        assert_eq!(mip1, 4);
        assert!(1024 + mip1 <= 1028);
        // 3D 4x4x2 RGBA8: depth = 2, RowPitch = 256, rows = 4, row_size = 16, total = 1808.
        assert_eq!(subresource_bytes(0, 256, 4, 2, 16, 1808), 1808);
        // Detiling the 16x4 RGBA16F texture into a slice sized by `subresource_bytes` succeeds.
        let words: [u32; 14] = [3, 0, 0, 0, 16, 0, 4, 65537, 10, 1, 0, 256, 3_145_728, 0];
        // SAFETY: 14 words = 56-byte ResourceDesc.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = crate::resource::texture_layout(&desc)
            .expect("a console layout")
            .planes[0]
            .layout
            .clone();
        let tiled = vec![0x3Cu8; layout.size as usize];
        let mut buf = vec![0u8; 896];
        layout
            .detile(&tiled, 0, 0, &mut buf[..span], 256, 256 * 4)
            .expect("exact unpadded staging slice suffices");
    }

    #[test]
    fn all_zero_matches_naive_across_lengths() {
        use super::all_zero;
        for len in [0, 1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257] {
            let mut buf = vec![0u8; len];
            assert!(all_zero(&buf), "len {len}");
            for pos in [0, len / 2, len.saturating_sub(1)] {
                if pos < len {
                    buf[pos] = 1;
                    assert!(!all_zero(&buf), "len {len} pos {pos}");
                    buf[pos] = 0;
                }
            }
        }
    }

    #[test]
    fn batched_mapped_bytes_handles_unmapped_partial_and_large_spans() {
        use super::mapped_bytes;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const BLOCK: usize = 64 << 10;
        // 260 blocks (> 256 batch size, 16.25 MiB) to exercise multi-batch querying.
        const TOTAL: usize = 260 * BLOCK;
        // SAFETY: reserve and commit selective blocks inside our own address space.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), TOTAL, MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            // Entirely uncommitted: 0.
            assert_eq!(mapped_bytes(base, TOTAL), 0);
            assert_eq!(mapped_bytes(base, 4096), 0);
            // Commit block 0, block 128, and block 259 (endpoints + interior across batch boundary).
            for b in [0usize, 128, 259] {
                let p = VirtualAlloc(base.add(b * BLOCK).cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE);
                assert!(!p.is_null());
                base.add(b * BLOCK).write_volatile(0x5a);
            }
            // Single-page range inside committed block 0.
            assert_eq!(mapped_bytes(base.add(100), 500), 500);
            // Full 260-block span: exactly 3 committed blocks.
            assert_eq!(mapped_bytes(base, TOTAL), (3 * BLOCK) as u64);
            // Unaligned start/end spanning block 0 tail + uncommitted block 1 + block 128.
            let unaligned_start = base.add(BLOCK - 1024);
            let unaligned_len = 1024 + 127 * BLOCK + 4096;
            assert_eq!(mapped_bytes(unaligned_start, unaligned_len), (1024 + 4096) as u64);
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn fingerprint_stops_at_next_placed_texture() {
        let mut backing = vec![0x11u8; 700_416];
        let base = backing.as_ptr() as u64;
        // Build a view heap with a placed texture at `base` and a neighbour at `base + 0x4000`.
        let Some(heap) = crate::descriptor::host::HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let mut title = [0u64; 32];
        let heap_ptr = title.as_mut_ptr() as u64;
        let resource: [u32; 14] = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let placed_at = |addr: u64, slot: u32| {
            let mut desc = [0u32; 16];
            desc[0] = 98;
            desc[1] = 4;
            desc[5] = 8;
            (desc[10], desc[11]) = (addr as u32, (addr >> 32) as u32);
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, heap_ptr + u64::from(slot) * 32);
            heap.mirror(slot, heap_ptr + u64::from(slot) * 32, false);
        };
        placed_at(base, 1);
        placed_at(base + 0x4000, 2);
        let before = super::fingerprint(base, backing.len() as u64);
        // Modifying bytes inside the neighbour (>= 0x4000) must not change the fingerprint.
        backing[0x4000..0x8000].fill(0x99);
        let tail = backing.len() - 4096;
        backing[tail..].fill(0xaa);
        assert_eq!(super::fingerprint(base, backing.len() as u64), before);
        // Modifying bytes inside this texture's own bytes (< 0x4000) must change it.
        backing[0x100] = 0x22;
        assert_ne!(super::fingerprint(base, backing.len() as u64), before);
    }
}

#[cfg(test)]
mod hot_texture_tests {
    use super::{HOT_FRAMES, Last, VERIFY_FRAMES};

    /// A video's planes cycle through a ring of three textures, each sampled
    /// and checked every third frame, and each found changed at every check:
    /// from the second change on the texture is hot, and stays so.
    #[test]
    fn a_texture_changing_at_every_check_goes_hot() {
        let mut last = Last::default();
        last.changed(10);
        assert!(!last.hash_due(10), "the check that found the change is done");
        assert!(last.hash_due(13), "the next check verifies it");
        last.changed(13);
        assert!(last.hash_due(14), "changed twice: hot");
        last.changed(16);
        assert!(last.hash_due(16 + HOT_FRAMES - 1));
    }

    /// A texture that streams in changes once: it is hashed once more at its
    /// next check, found as it was, and not hashed again between turns.
    #[test]
    fn a_texture_that_changed_once_is_not_hot() {
        let mut last = Last::default();
        last.changed(100);
        assert!(last.hash_due(101), "verified at the next check");
        last.unchanged();
        assert!(!last.hash_due(102));
        assert!(!last.hash_due(100 + VERIFY_FRAMES));
        // A change long after the first is a first change again.
        last.changed(500);
        assert!(!last.hash_due(500));
        assert!(last.hash_due(501));
    }

    /// A hot texture that stops changing falls back to the turns.
    #[test]
    fn a_hot_texture_cools_down() {
        let mut last = Last::default();
        last.changed(1);
        last.changed(4);
        assert!(last.hash_due(4 + HOT_FRAMES - 1));
        last.unchanged();
        assert!(!last.hash_due(4 + HOT_FRAMES + 1));
    }

    #[test]
    fn oversized_thread_read_buffer_shrinks_after_use() {
        super::with_read(8 << 20, |buf| {
            assert_eq!(buf.len(), 8 << 20);
            buf[0] = 0x5a;
        });
        super::READ.with(|read| {
            assert!(read.borrow().capacity() <= 4 << 20);
        });
    }

    #[test]
    fn partial_cooldown_defaults_to_six_and_honors_env() {
        assert_eq!(super::parse_partial_cooldown(None), 6);
        assert_eq!(super::parse_partial_cooldown(Some("0")), 6);
        assert_eq!(super::parse_partial_cooldown(Some("bad")), 6);
        assert_eq!(super::parse_partial_cooldown(Some(" 4 ")), 4);
        assert_eq!(super::parse_partial_cooldown(Some("30")), 30);
    }

    #[test]
    fn texture_once_toggle_defaults_on_and_zero_disables() {
        use super::once_from;
        assert!(once_from(None));
        assert!(once_from(Some("1")));
        assert!(once_from(Some("yes")));
        assert!(!once_from(Some("0")));
        assert!(!once_from(Some(" 0 ")));
    }

    /// The digest tells every length and every changed byte apart, zeros
    /// among them.
    #[test]
    fn a_digest_tells_bytes_apart() {
        use super::digest;
        let zeros = [0u8; 400];
        let mut seen: Vec<super::Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }
}
