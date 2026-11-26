//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT,
    MEM_EXTENDED_PARAMETER, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
    MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile, MapViewOfFile3, MemExtendedParameterAddressRequirements, PAGE_NOACCESS,
    PAGE_READWRITE, SEC_COMMIT, SEC_RESERVE, UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc,
    VirtualAlloc2, VirtualFree, VirtualQuery, VirtualQueryEx,
};

/// `winnt.h` `MEM_COALESCE_PLACEHOLDERS` (not exported by `windows-sys`).
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x0000_0001;
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);
/// Title-owned virtual memory committed via `XMemVirtualAlloc`.
static TITLE_VIRTUAL_COMMITTED: AtomicUsize = AtomicUsize::new(0);
/// Title-owned 64 KiB physical pages committed across `XMemAllocatePhysicalPages` sections.
static TITLE_PHYSICAL_COMMITTED: AtomicUsize = AtomicUsize::new(0);
/// Title-owned heap allocations served by `XMemAllocDefault` (non-32-bit).
static TITLE_HEAP_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

fn saturating_sub_atomic(counter: &AtomicUsize, bytes: usize) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        Some(cur.saturating_sub(bytes))
    });
}

/// Whether `XMemGetWorkingSetStatistics` reports title-only memory usage
/// instead of the whole host process commit (`VICTORY_XMEM_TITLE_USAGE=0`
/// restores raw `PagefileUsage`).
fn title_usage_tracking() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_XMEM_TITLE_USAGE").is_none_or(|v| v != "0")
    })
}

/// Whether physical-page sections keep one persistent writable view for
/// DirectStorage writes (`VICTORY_XMEM_PERSISTENT_VIEW=0` maps/unmaps per write).
fn persistent_views_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_XMEM_PERSISTENT_VIEW").is_none_or(|v| v != "0")
    })
}

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if !at.is_null() && !placeholder && (kind & MEM_COMMIT != 0) {
        TITLE_VIRTUAL_COMMITTED.fetch_add(size.next_multiple_of(4096), Ordering::Relaxed);
    }
    if placeholder && !at.is_null() {
        let alloc_base = at as usize;
        let alloc_len = size.next_multiple_of(PAGE);
        let mut p = physical();
        // Drop any stale placeholder entries if the OS reused freed VA space.
        let stale: Vec<usize> = p
            .placeholders
            .range(..alloc_base + alloc_len)
            .filter(|&(&start, &len)| start + len > alloc_base)
            .map(|(&start, _)| start)
            .collect();
        for start in stale {
            p.placeholders.remove(&start);
        }
        p.placeholders.insert(alloc_base, alloc_len);
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        TITLE_HEAP_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            let mut mbi = MEMORY_BASIC_INFORMATION::default();
            if VirtualQuery(address, &mut mbi, size_of::<MEMORY_BASIC_INFORMATION>()) != 0
                && mbi.State == MEM_COMMIT
            {
                saturating_sub_atomic(&TITLE_VIRTUAL_COMMITTED, mbi.RegionSize);
            }
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        saturating_sub_atomic(&TITLE_HEAP_ALLOCATED, size);
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// Persistent whole-section writable mapping used by DirectStorage workers so
/// physical-page writes do not call `MapViewOfFile`/`UnmapViewOfFile` (and
/// trigger cross-core TLB shootdowns) on every streamed texture chunk.
struct PersistentView {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
}

// SAFETY: the mapped address range belongs to the process and stays valid until
// the owning `Section` drops (which drops `PersistentView` before closing the handle).
unsafe impl Send for PersistentView {}
// SAFETY: concurrent disjoint page writes into the section are synchronized by the title.
unsafe impl Sync for PersistentView {}

impl Drop for PersistentView {
    fn drop(&mut self) {
        // SAFETY: mapped by `MapViewOfFile` in `Section::persistent_base`.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

struct Section {
    handle: HANDLE,
    size: usize,
    pages: usize,
    persistent: OnceLock<Option<PersistentView>>,
    /// 1 bit per 64 KiB page committed in this section.
    committed: Mutex<Vec<u64>>,
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    fn new(handle: HANDLE, pages: usize, initially_committed: bool) -> Self {
        let mut bits = vec![0u64; pages.div_ceil(64)];
        if initially_committed && pages > 0 {
            for idx in 0..pages {
                bits[idx / 64] |= 1u64 << (idx % 64);
            }
            TITLE_PHYSICAL_COMMITTED.fetch_add(pages * PAGE, Ordering::Relaxed);
        }
        Self {
            handle,
            size: pages * PAGE,
            pages,
            persistent: OnceLock::new(),
            committed: Mutex::new(bits),
        }
    }

    fn persistent_base(&self) -> Option<usize> {
        if !persistent_views_enabled() || self.size == 0 {
            return None;
        }
        self.persistent
            .get_or_init(|| {
                // SAFETY: maps the full section we own; `SEC_RESERVE` sections
                // consume virtual address space only until pages are committed.
                let at = unsafe { MapViewOfFile(self.handle, FILE_MAP_WRITE, 0, 0, self.size) };
                if at.Value.is_null() {
                    tracing::debug!(
                        size = self.size,
                        error = %std::io::Error::last_os_error(),
                        "xmem: persistent section view unavailable; falling back to per-write views"
                    );
                    None
                } else {
                    Some(PersistentView { at })
                }
            })
            .as_ref()
            .map(|v| v.at.Value as usize)
    }

    fn all_committed(&self, first: usize, count: usize) -> bool {
        let end = first.saturating_add(count).min(self.pages);
        if first >= end {
            return true;
        }
        let bits = self.committed.lock().unwrap_or_else(|e| e.into_inner());
        (first..end).all(|idx| (bits[idx / 64] >> (idx % 64)) & 1 != 0)
    }

    fn mark_committed(&self, first: usize, count: usize) {
        let end = first.saturating_add(count).min(self.pages);
        if first >= end {
            return;
        }
        let mut newly = 0usize;
        {
            let mut bits = self.committed.lock().unwrap_or_else(|e| e.into_inner());
            for idx in first..end {
                let mask = 1u64 << (idx % 64);
                let word = &mut bits[idx / 64];
                if *word & mask == 0 {
                    *word |= mask;
                    newly += 1;
                }
            }
        }
        if newly > 0 {
            TITLE_PHYSICAL_COMMITTED.fetch_add(newly * PAGE, Ordering::Relaxed);
        }
    }

    fn release_page_accounting(&self, index: usize) {
        if index >= self.pages {
            return;
        }
        let was_set = {
            let mut bits = self.committed.lock().unwrap_or_else(|e| e.into_inner());
            let mask = 1u64 << (index % 64);
            let word = &mut bits[index / 64];
            let had = *word & mask != 0;
            *word &= !mask;
            had
        };
        if was_set {
            saturating_sub_atomic(&TITLE_PHYSICAL_COMMITTED, PAGE);
        }
    }

    /// Returns a writable pointer to `count` consecutive 64 KiB pages starting
    /// at `first` inside the section's persistent view, committing only pages
    /// not yet marked committed.
    fn ensure_persistent_slice(&self, first: usize, count: usize) -> Option<*mut u8> {
        if count == 0 || first.checked_add(count).is_none_or(|end| end > self.pages) {
            return None;
        }
        let base = self.persistent_base()?;
        let ptr = base + first * PAGE;
        if !self.all_committed(first, count) {
            if !commit(ptr, count * PAGE) {
                return None;
            }
            self.mark_committed(first, count);
        }
        Some(ptr as *mut u8)
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        let remaining: usize = {
            let bits = self.committed.lock().unwrap_or_else(|e| e.into_inner());
            bits.iter().map(|w| w.count_ones() as usize).sum()
        };
        if remaining > 0 {
            saturating_sub_atomic(&TITLE_PHYSICAL_COMMITTED, remaining * PAGE);
        }
        // Unmap the persistent view before closing the section handle.
        let _ = std::mem::take(&mut self.persistent);
        // SAFETY: our handle; any title views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Unmapped placeholder intervals by start address: their exact kernel length.
    placeholders: BTreeMap<usize, usize>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
}

impl Physical {
    /// What the physical page currently mapped at `at` shows.
    fn slice_at(&self, at: usize) -> Option<Slice> {
        self.views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
    }
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    let section = Arc::new(Section::new(handle, n, on_wine()));
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        if let Some(slice) = physical.pages.remove(&unsafe { *pages.add(i) }) {
            slice.section.release_page_accounting(slice.index);
        }
    }
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped (fallback when persistent section views
/// are disabled or unavailable).
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        if !commit(at.Value as usize, len) {
            return None;
        }
        slice.section.mark_committed(slice.index, n);
        Some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Uses each section's persistent view when available (zero per-request map/unmap
/// TLB shootdowns), falling back to one temporary view per run of consecutive pages.
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let k = (n * PAGE - within).min(data.len() - done);
        if let Some(ptr) = slice.section.ensure_persistent_slice(slice.index, *n) {
            // SAFETY: `ptr` covers `n * PAGE` committed bytes in the section's
            // persistent view, kept alive by `slice.section`.
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr.add(within), k) };
            dst.copy_from_slice(&data[done..done + k]);
        } else {
            let Some(mut view) = View::map(slice, *n) else {
                return false;
            };
            view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        }
        (within, done) = (0, done + k);
    }
    true
}

enum WindowBacking {
    Persistent {
        _section: Arc<Section>,
        ptr: *mut u8,
        len: usize,
    },
    Temp(View),
}

// SAFETY: `ptr` points into `_section`'s persistent view and stays valid for `_section`'s lifetime.
unsafe impl Send for WindowBacking {}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    backing: WindowBacking,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        match &mut self.backing {
            WindowBacking::Persistent { ptr, len: total, .. } => {
                debug_assert!(within + len <= *total);
                // SAFETY: `[within, within + len)` lies inside the committed
                // persistent slice held alive by `_section`.
                unsafe { std::slice::from_raw_parts_mut(ptr.add(within), len) }
            }
            WindowBacking::Temp(view) => &mut view.bytes()[within..within + len],
        }
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    let backing = if let Some(ptr) = slice.section.ensure_persistent_slice(slice.index, *n) {
        WindowBacking::Persistent {
            _section: slice.section.clone(),
            ptr,
            len: n * PAGE,
        }
    } else {
        WindowBacking::Temp(View::map(slice, *n)?)
    };
    Some(PageWindow {
        backing,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let mut physical = physical();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    if side_by_side() > 1 && n > 1 {
        // Read before any lock is taken, as in `XMemMapPhysicalPages`.
        let wanted: Vec<(usize, usize)> = (0..n)
            // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
            .map(|i| unsafe {
                (
                    *addresses.add(i) as usize,
                    if pages.is_null() { 0 } else { *pages.add(i) },
                )
            })
            .collect();
        if wanted.iter().all(|w| w.0 % PAGE == 0) {
            return scatter(&wanted).into();
        }
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Pages changed under the locks at once in [`scatter`] and [`map_aliases`].
const GROUP: usize = 64;

/// How many threads map a group of pages side by side (`VICTORY_MAP_THREADS`;
/// default `(logical_cpus / 2).clamp(2, 8)` -> 6 on 6C/12T; `1` disables).
///
/// Mapping and unmapping a view contend with the kernel's VAD/address-creation
/// lock alongside write-guard protection changes (`0.2-4 ms` per page on one
/// thread). Standing in that queue from multiple threads boosts streaming page
/// mapping throughput up to ~20x without oversubscribing mid-core CPUs.
fn side_by_side() -> usize {
    static THREADS: LazyLock<usize> = LazyLock::new(|| {
        let default = std::thread::available_parallelism()
            .map_or(6, |n| (n.get() / 2).clamp(2, 8));
        std::env::var("VICTORY_MAP_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(default)
            .clamp(1, 32)
    });
    *THREADS
}

/// [`map_alias`] of many ranges in order, mapping disjoint 64 KiB pages in
/// parallel across [`side_by_side`] worker threads (upstream `565f4a56`).
pub fn map_aliases(ranges: &[(u64, u64, u64)]) -> bool {
    if side_by_side() <= 1 || ranges.is_empty() {
        return ranges.iter().all(|&(to, from, len)| map_alias(to, from, len));
    }
    let pages: Vec<(usize, Option<usize>)> = ranges
        .iter()
        .flat_map(|&(to, from, len)| {
            (0..len as usize)
                .step_by(PAGE)
                .map(move |k| (to as usize + k, (from != 0).then_some(from as usize + k)))
        })
        .collect();
    if pages.len() < 3
        || pages
            .iter()
            .any(|p| p.0 % PAGE != 0 || p.1.is_some_and(|from| from % PAGE != 0))
    {
        return ranges.iter().all(|&(to, from, len)| map_alias(to, from, len));
    }
    let mut ok = true;
    let mut at = 0;
    while at < pages.len() {
        let mut physical = physical();
        let mut ops: Vec<Op> = Vec::new();
        let mut touched = std::collections::HashSet::new();
        for &(to, from) in &pages[at..] {
            // A page the group changes already, or one it would be mapped from:
            // push into the next group to preserve sequential semantics.
            if ops.len() == GROUP
                || touched.contains(&to)
                || from.is_some_and(|from| touched.contains(&from))
            {
                break;
            }
            let slice = from.map(|from| physical.slice_at(from));
            ok &= !matches!(slice, Some(None));
            touched.insert(to);
            ops.push(Op::new(to, slice.flatten()));
            at += 1;
        }
        let _changing = victory_gpu::title_memory_remapping();
        ok &= change(&mut physical, ops);
    }
    ok
}

/// Maps (page number != 0) or unmaps each address in parallel batches; false
/// if a page is unknown or the host refuses one.
fn scatter(wanted: &[(usize, usize)]) -> bool {
    let mut ok = true;
    let mut at = 0;
    while at < wanted.len() {
        let mut physical = physical();
        let mut ops: Vec<Op> = Vec::new();
        let mut touched = std::collections::HashSet::new();
        for &(to, page) in &wanted[at..] {
            if ops.len() == GROUP || !touched.insert(to) {
                break;
            }
            let slice = if page == 0 {
                None
            } else {
                let s = physical.pages.get(&page).cloned();
                if s.is_none() {
                    tracing::warn!(page, "XMemMapPhysicalPagesScatter: unknown page");
                    ok = false;
                }
                s
            };
            ops.push(Op::new(to, slice));
            at += 1;
        }
        let _changing = victory_gpu::title_memory_remapping();
        ok &= change(&mut physical, ops);
    }
    ok
}

/// A single 64 KiB page operation executed on a worker thread by [`pool::all`].
struct Op {
    at: usize,
    slice: Option<Slice>,
    /// A single-page view currently occupies `[at, at + PAGE)` and must be unmapped first.
    had: bool,
}

impl Op {
    fn new(at: usize, slice: Option<Slice>) -> Self {
        Self {
            at,
            slice,
            had: false,
        }
    }

    fn run(&self) -> bool {
        if self.had {
            // SAFETY: a single-page view we mapped; returns to an exact 64 KiB placeholder.
            let gone = unsafe {
                UnmapViewOfFile2(
                    GetCurrentProcess(),
                    MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: self.at as *mut c_void,
                    },
                    MEM_PRESERVE_PLACEHOLDER,
                )
            };
            if gone == 0 {
                return false;
            }
        }
        let Some(slice) = &self.slice else {
            return true;
        };
        show_page(self.at, slice)
    }
}

/// Executes the host syscalls to map a single 64 KiB `slice` over an exact
/// placeholder at `base` and commit it.
fn show_page(base: usize, slice: &Slice) -> bool {
    let map_view = || {
        // SAFETY: `[base, base + PAGE)` is an exact placeholder prepared under `PHYSICAL`.
        unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                base as *const c_void,
                (slice.index * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        }
    };
    let mut view = map_view();
    if view.Value as usize != base {
        // Fallback if an external split left an unsplit/coalescable placeholder.
        // SAFETY: acts only on placeholders covering `[base, base + PAGE)`.
        unsafe {
            VirtualFree(
                base as *mut c_void,
                PAGE,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        view = map_view();
    }
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len = PAGE,
            error = %std::io::Error::last_os_error(),
            "parallel mapping physical page failed (not a placeholder?)"
        );
        return false;
    }
    if !commit(base, PAGE) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    true
}

/// Applies `ops` (all distinct 64 KiB pages) with `PHYSICAL` and
/// `title_memory_remapping` held: updates metadata and placeholders on the
/// caller thread, runs kernel unmap/map/commit syscalls in parallel via
/// [`pool::all`], then commits tracking & GPU notifications on the caller thread.
fn change(physical: &mut Physical, mut ops: Vec<Op>) -> bool {
    for op in &mut ops {
        let there = physical
            .views
            .range(..=op.at)
            .next_back()
            .filter(|(base, view)| op.at < **base + view.0)
            .map(|(base, view)| (*base, view.0));
        match there {
            Some((base, PAGE)) if base == op.at => {
                if let Some((size, slice)) = physical.views.remove(&op.at) {
                    for k in 0..size.div_ceil(PAGE) {
                        let key = slice.key(k);
                        if let Some(addresses) = physical.shown.get_mut(&key) {
                            addresses.retain(|&a| a != op.at + k * PAGE);
                            if addresses.is_empty() {
                                physical.shown.remove(&key);
                            }
                        }
                    }
                }
                victory_gpu::title_memory_remapped(op.at, PAGE);
                physical.placeholders.remove(&op.at);
                op.had = true;
            }
            Some(_) => {
                // Part of a larger multi-page view: split/unmap cleanly on caller thread.
                unmap(physical, op.at, PAGE);
            }
            None => {}
        }
        if !op.had && op.slice.is_some() {
            // Prepare exact 64 KiB placeholder serially so concurrent worker
            // threads never race splitting the same enclosing placeholder.
            prepare_placeholder(physical, op.at, PAGE);
        }
    }
    let done = pool::all(ops);
    let mut ok = true;
    for (op, succeeded) in done {
        match op.slice {
            Some(slice) if succeeded => {
                slice.section.mark_committed(slice.index, 1);
                physical.placeholders.remove(&op.at);
                physical.shown.entry(slice.key(0)).or_default().push(op.at);
                victory_gpu::title_memory_remapped(op.at, PAGE);
                victory_gpu::title_memory_view(
                    op.at,
                    PAGE,
                    slice.section.handle as usize,
                    (slice.index * PAGE) as u64,
                );
                physical.views.insert(op.at, (PAGE, slice));
            }
            None if op.had && succeeded => {
                physical.placeholders.insert(op.at, PAGE);
            }
            _ => {
                if op.had && !succeeded {
                    physical.placeholders.insert(op.at, PAGE);
                }
                ok &= succeeded;
            }
        }
    }
    ok
}

/// Worker pool that runs kernel `UnmapViewOfFile2` / `MapViewOfFile3` / `VirtualAlloc`
/// syscalls side by side for [`change`].
mod pool {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, OnceLock};

    use super::{Op, side_by_side};

    struct Batch {
        ops: Vec<(Op, AtomicBool)>,
        next: AtomicUsize,
        left: AtomicUsize,
        over: (Mutex<bool>, Condvar),
    }

    impl Batch {
        fn work(&self) {
            loop {
                let i = self.next.fetch_add(1, Ordering::Relaxed);
                let Some((op, done)) = self.ops.get(i) else {
                    return;
                };
                done.store(op.run(), Ordering::Release);
                if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
                    *self.over.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
                    self.over.1.notify_all();
                }
            }
        }
    }

    struct Pool {
        batches: Mutex<VecDeque<Arc<Batch>>>,
        more: Condvar,
    }

    fn pool() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        static STARTED: std::sync::Once = std::sync::Once::new();
        let pool = POOL.get_or_init(|| Pool {
            batches: Mutex::default(),
            more: Condvar::new(),
        });
        STARTED.call_once(|| {
            let started = (1..side_by_side())
                .filter(|i| {
                    std::thread::Builder::new()
                        .name(format!("victory-map-{i}"))
                        .spawn(|| serve(pool))
                        .is_ok()
                })
                .count();
            tracing::info!(
                started,
                "threads mapping physical pages side by side (VICTORY_MAP_THREADS=1: none)"
            );
        });
        pool
    }

    fn serve(pool: &Pool) {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
        };
        // SAFETY: plain priority call on our own worker thread.
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
        let mut batches = pool.batches.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let Some(batch) = batches.front().cloned() else {
                batches = pool.more.wait(batches).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            if batch.next.load(Ordering::Relaxed) >= batch.ops.len() {
                if batches.front().is_some_and(|b| Arc::ptr_eq(b, &batch)) {
                    batches.pop_front();
                }
                continue;
            }
            drop(batches);
            batch.work();
            batches = pool.batches.lock().unwrap_or_else(|e| e.into_inner());
        }
    }

    pub(super) fn all(ops: Vec<Op>) -> Vec<(Op, bool)> {
        if ops.len() < 3 {
            return ops
                .into_iter()
                .map(|op| {
                    let done = op.run();
                    (op, done)
                })
                .collect();
        }
        let n = ops.len();
        let batch = Arc::new(Batch {
            ops: ops.into_iter().map(|op| (op, AtomicBool::new(false))).collect(),
            next: AtomicUsize::new(0),
            left: AtomicUsize::new(n),
            over: Default::default(),
        });
        let pool = pool();
        pool.batches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(batch.clone());
        if n > side_by_side() {
            pool.more.notify_all();
        } else {
            (1..n).for_each(|_| pool.more.notify_one());
        }
        batch.work();
        let mut over = batch.over.0.lock().unwrap_or_else(|e| e.into_inner());
        while !*over {
            over = batch.over.1.wait(over).unwrap_or_else(|e| e.into_inner());
        }
        drop(over);
        batch
            .ops
            .iter()
            .map(|(op, done)| {
                let op = Op {
                    at: op.at,
                    slice: op.slice.clone(),
                    had: op.had,
                };
                (op, done.load(Ordering::Acquire))
            })
            .collect()
    }
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    // Snapshot the source slices before unmapping `to` so overlapping aliases
    // keep their source pages and a missing source page leaves `to` intact.
    let sources: Vec<Slice> = if from == 0 {
        Vec::new()
    } else {
        let mut out = Vec::with_capacity(len / PAGE);
        for k in (0..len).step_by(PAGE) {
            let at = from + k;
            let Some(slice) = physical
                .views
                .range(..=at)
                .next_back()
                .filter(|(base, (size, _))| at < *base + size)
                .map(|(base, (_, s))| Slice {
                    index: s.index + (at - base) / PAGE,
                    section: s.section.clone(),
                })
            else {
                return false;
            };
            out.push(slice);
        }
        out
    };
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, to, len);
    // Coalesce consecutive pages of the same section into one `MapViewOfFile3`
    // call so `CopyPageMappingsBatchX` avoids per-64 KiB-page kernel remaps.
    let mut i = 0;
    while i < sources.len() {
        let first = sources[i].clone();
        let run = 1 + sources[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(s, k)| {
                Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
            })
            .count();
        if !map(&mut physical, to + i * PAGE, run * PAGE, first) {
            return false;
        }
        i += run;
    }
    true
}

/// Ensures `[base, base + len)` is a single exact placeholder: splits any
/// larger tracked placeholder straddling `base` or `base + len`, then
/// coalesces fragmented placeholders inside `[base, base + len)`.
fn prepare_placeholder(physical: &mut Physical, base: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    let first = physical
        .placeholders
        .range(..=base)
        .next_back()
        .filter(|&(&start, &size)| start + size > base)
        .map(|(&start, &size)| (start, size));
    let mut covered: Vec<(usize, usize)> = first
        .into_iter()
        .chain(
            physical
                .placeholders
                .range(base + 1..end)
                .map(|(&start, &size)| (start, size)),
        )
        .collect();
    let contiguous = !covered.is_empty()
        && covered[0].0 <= base
        && covered.last().is_some_and(|&(s, n)| s + n >= end)
        && covered.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0);
    if !contiguous {
        // Fallback for placeholders created outside `XMemVirtualAlloc`.
        // SAFETY: splits a single enclosing placeholder if needed.
        unsafe {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        return;
    }
    if covered.len() == 1 {
        let (p0, s0) = covered[0];
        if p0 < base || p0 + s0 > end {
            // SAFETY: splits the single enclosing placeholder at `base` and/or `end`.
            unsafe {
                VirtualFree(
                    base as *mut c_void,
                    len,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
            }
            physical.placeholders.remove(&p0);
            if p0 < base {
                physical.placeholders.insert(p0, base - p0);
            }
            physical.placeholders.insert(base, len);
            if p0 + s0 > end {
                physical.placeholders.insert(end, p0 + s0 - end);
            }
        }
        return;
    }
    let (p0, s0) = covered[0];
    if p0 < base {
        // SAFETY: splits `[p0, p0 + s0)` into `[p0, base)` and `[base, p0 + s0)`.
        unsafe {
            VirtualFree(
                p0 as *mut c_void,
                base - p0,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        physical.placeholders.insert(p0, base - p0);
        physical.placeholders.insert(base, p0 + s0 - base);
        covered[0] = (base, p0 + s0 - base);
    }
    let last_idx = covered.len() - 1;
    let (pk, sk) = covered[last_idx];
    if pk + sk > end {
        // SAFETY: splits `[pk, pk + sk)` into `[pk, end)` and `[end, pk + sk)`.
        unsafe {
            VirtualFree(
                pk as *mut c_void,
                end - pk,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        physical.placeholders.insert(pk, end - pk);
        physical.placeholders.insert(end, pk + sk - end);
        covered[last_idx] = (pk, end - pk);
    }
    // SAFETY: all tracked placeholders covering `[base, end)` now match exact boundaries.
    unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
        );
    }
    for (p, _) in covered {
        physical.placeholders.remove(&p);
    }
    physical.placeholders.insert(base, len);
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in placeholder memory (splitting/coalescing placeholders as needed).
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    prepare_placeholder(physical, base, len);
    // SAFETY: `[base, base + len)` is now a single exact placeholder (if valid
    // placeholder memory); `MapViewOfFile3` replaces it with the section view.
    let mut view = unsafe {
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        // If an external split left exact placeholders covering `[base, base + len)`,
        // try coalescing them once before giving up.
        // SAFETY: acts only on placeholders matching `[base, base + len)`.
        unsafe {
            if VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
            ) != 0
            {
                view = MapViewOfFile3(
                    slice.section.handle,
                    GetCurrentProcess(),
                    base as *const c_void,
                    (slice.index * PAGE) as u64,
                    len,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
            }
        }
    }
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    slice.section.mark_committed(slice.index, len.div_ceil(PAGE));
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(
        base,
        len,
        slice.section.handle as usize,
        (slice.index * PAGE) as u64,
    );
    physical.placeholders.remove(&base);
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        physical.placeholders.insert(at, size);
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let phys_used = TITLE_PHYSICAL_COMMITTED.load(Ordering::Relaxed);
    let virt_used = TITLE_VIRTUAL_COMMITTED.load(Ordering::Relaxed);
    let heap_used = TITLE_HEAP_ALLOCATED.load(Ordering::Relaxed);
    let tracked_title = phys_used
        .saturating_add(virt_used)
        .saturating_add(heap_used);
    let title_used = if title_usage_tracking() {
        tracked_title.min(counters.PagefileUsage)
    } else {
        counters.PagefileUsage
    };
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used,
        physical_pages: (phys_used / PAGE) as u32,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every 1,024th.
    // Reporting title-only allocations avoids inflating `title_used` with host
    // shadow copies, D3D12 driver commit, and caches (which previously reported
    // ~27 GB against Xbox Series X's 13.5 GB limit and triggered eviction thrash).
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            host_commit_mb = counters.PagefileUsage >> 20,
            phys_mb = phys_used >> 20,
            virt_mb = virt_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let hold = XMemAllocDefault(PAGE, 0);
        assert!(!hold.is_null());
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
        XMemFreeDefault(hold, 0);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn remapping_contiguous_run_over_fragmented_placeholders_succeeds() {
        let pages = allocate(3);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());

        // Map page 0 alone, splitting the 3-page placeholder into [a, a+PAGE) and [a+PAGE, a+3*PAGE).
        assert_eq!(XMemMapPhysicalPages(a, 1, &pages[0]), a);
        // Now remap 2 contiguous pages at `a`: unmapping page 0 leaves two separate
        // placeholders [a, a+PAGE) and [a+PAGE, a+3*PAGE) that must be split at
        // a+2*PAGE and coalesced across [a, a+2*PAGE) before MapViewOfFile3.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);

        // SAFETY: both pages are mapped read-write.
        unsafe {
            a.cast::<u8>().write(0x11);
            a.wrapping_byte_add(PAGE).cast::<u8>().write(0x22);
            assert_eq!(a.cast::<u8>().read(), 0x11);
            assert_eq!(a.wrapping_byte_add(PAGE).cast::<u8>().read(), 0x22);
        }

        // The trailing placeholder at a + 2*PAGE must still be mappable.
        let third = a.wrapping_byte_add(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(third, 1, &pages[2]), third);
        // SAFETY: third page is mapped read-write.
        unsafe {
            third.cast::<u8>().write(0x33);
            assert_eq!(third.cast::<u8>().read(), 0x33);
        }

        // Remapping all 3 contiguous pages over the two views [a, a+2*PAGE) + [third, a+3*PAGE)
        // must coalesce across both unmapped placeholders too.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: all 3 pages are mapped read-write.
        unsafe {
            assert_eq!(a.cast::<u8>().read(), 0x11);
            assert_eq!(a.wrapping_byte_add(PAGE).cast::<u8>().read(), 0x22);
            assert_eq!(third.cast::<u8>().read(), 0x33);
        }

        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn map_alias_handles_overlapping_ranges_and_missing_sources() {
        let pages = allocate(3);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: pages 0 and 1 are mapped read-write.
        unsafe {
            a.cast::<u8>().write(0xA1);
            a.wrapping_byte_add(PAGE).cast::<u8>().write(0xB2);
        }

        // Overlapping shift [a, a+2*PAGE) -> [a+PAGE, a+3*PAGE): page 1 overlaps
        // between source and destination, so snapshotting before unmap preserves it.
        let second = a.wrapping_byte_add(PAGE);
        let third = a.wrapping_byte_add(2 * PAGE);
        assert!(map_alias(second as u64, a as u64, (2 * PAGE) as u64));
        // SAFETY: all three pages are now mapped (page0, page0, page1).
        unsafe {
            assert_eq!(a.cast::<u8>().read(), 0xA1);
            assert_eq!(second.cast::<u8>().read(), 0xA1);
            assert_eq!(third.cast::<u8>().read(), 0xB2);
        }

        // Missing source page must fail without unmapping destination `a`.
        assert!(!map_alias(a as u64, 0x10000, PAGE as u64));
        // SAFETY: `a` was not unmapped on failed alias.
        unsafe {
            assert_eq!(a.cast::<u8>().read(), 0xA1);
        }

        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn persistent_section_view_reuses_mapping_and_tracks_working_set() {
        let before = TITLE_PHYSICAL_COMMITTED.load(Ordering::Relaxed);
        let pages = allocate(4);
        // First write commits pages 1..3 through the persistent view.
        let mut w1 = page_window(&pages, PAGE + 16, PAGE).expect("first window");
        let p1 = w1.bytes().as_ptr() as usize;
        w1.bytes().fill(0x3c);
        drop(w1);
        let after_commit = TITLE_PHYSICAL_COMMITTED.load(Ordering::Relaxed);
        assert!(after_commit >= before + 2 * PAGE);

        // Second write reuses the exact same persistent view address (zero remap).
        let mut w2 = page_window(&pages, PAGE + 16, PAGE).expect("second window");
        let p2 = w2.bytes().as_ptr() as usize;
        assert_eq!(p1, p2);
        assert!(w2.bytes().iter().all(|&b| b == 0x3c));
        drop(w2);

        // Working set reports tracked title memory rather than inflated host commit.
        let mut ws = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0, &mut ws), 1);
        assert!(ws.title_used >= 2 * PAGE);
        assert!(ws.physical_pages >= 2);

        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        assert!(TITLE_PHYSICAL_COMMITTED.load(Ordering::Relaxed) <= before);
    }

    /// Many pages in one call: mapped in parallel, moved over each other's
    /// addresses, duplicated in one list, aliased via `map_aliases`, and unmapped.
    #[test]
    fn a_scatter_of_many_pages_is_what_one_after_the_other_is() {
        const N: usize = 200;
        let pages = allocate(N);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * N * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        ) as usize;
        assert_ne!(a, 0);
        let at = |k: usize| (a + k * PAGE) as *mut c_void;
        let first = |k: usize| at(k).cast::<u64>();
        for round in 0..4_usize {
            let to: Vec<*mut c_void> = (0..N).map(|i| at((i + 7 * round) % N)).collect();
            assert_eq!(XMemMapPhysicalPagesScatter(to.as_ptr(), N, pages.as_ptr()), 1);
            for i in 0..N {
                // SAFETY: mapped read-write.
                unsafe {
                    if round > 0 {
                        assert_eq!(
                            first((i + 7 * round) % N).read(),
                            (1000 * (round - 1) + i) as u64
                        );
                    }
                    first((i + 7 * round) % N).write((1000 * round + i) as u64);
                }
            }
        }
        // Batch GPU page-mapping copies including self-dependent ranges.
        let half = (N / 2 * PAGE) as u64;
        let (from, to) = (a as u64, at(N) as u64);
        assert!(map_aliases(&[(to, from, half), (to + half, to, half)]));
        for k in 0..N / 2 {
            // SAFETY: mapped read-write.
            unsafe {
                assert_eq!(first(N + k).read(), first(k).read());
                assert_eq!(first(N + N / 2 + k).read(), first(k).read());
                first(N + k).write(5);
                assert_eq!(first(k).read(), 5);
            }
        }
        // Repeated target address in one scatter list: last wins; page 0 unmaps.
        let to = [at(0), at(1), at(0), at(2), at(3), at(0)];
        let what = [pages[5], pages[6], 0, 0, pages[7], pages[8]];
        assert_eq!(XMemMapPhysicalPagesScatter(to.as_ptr(), 6, what.as_ptr()), 1);
        assert_eq!(query(at(2)).host.State, MEM_RESERVE);
        // SAFETY: mapped read-write; same page mapped at three addresses.
        unsafe {
            first(0).write(77);
            assert_eq!(
                XMemMapPhysicalPagesScatter([at(2), at(4), at(6)].as_ptr(), 3, [pages[8]; 3].as_ptr()),
                1
            );
            assert_eq!((first(2).read(), first(4).read(), first(6).read()), (77, 77, 77));
        }
        // Unknown page number fails cleanly.
        assert_eq!(
            XMemMapPhysicalPagesScatter([at(0), at(1)].as_ptr(), 2, [pages[0], 1].as_ptr()),
            0
        );
        // Batch unmap across all 2*N pages including parts of multi-page views.
        assert_eq!(XMemMapPhysicalPages(at(0), 4, std::ptr::null()), at(0));
        // SAFETY: placeholders of ours, nothing mapped in them.
        unsafe { VirtualFree(at(0), 4 * PAGE, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) };
        assert_eq!(XMemMapPhysicalPages(at(0), 4, pages.as_ptr()), at(0));
        assert!(map_aliases(&[(a as u64, 0, (2 * N * PAGE) as u64)]));
        assert!((0..2 * N).all(|k| query(at(k)).host.State == MEM_RESERVE));
        assert!(physical().views.range(a..a + 2 * N * PAGE).next().is_none());
        assert_eq!(XMemFreePhysicalPages(N, pages.as_ptr()), 1);
    }
}
