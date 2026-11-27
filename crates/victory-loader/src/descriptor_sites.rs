//! Renderer port #2 (`VICTORY_PORT_DESCRIPTORS=1`, read in `victory-gpu`'s
//! `port_descriptors`): the title's own code that copies descriptor records
//! into its shader-visible heap as bytes tells the layer which slots it wrote.
//!
//! What the title does (game_scarlett_master.exe): each draw's and each
//! dispatch's descriptor tables are allocated from its frame allocator
//! (image +0x4d9f400; the allocator's memory is the placed shader-visible
//! view heap) and filled by two functions, +0x4d9efd0 (graphics) and
//! +0x4eec8e0 (compute), three loops each, one record at a time:
//! `and r, 7FE0h` (the slot's byte offset in the table), `vmovups ymm0,
//! [record]`, `vmovups [table + r], ymm0`. Found with `dumpbin /disasm` from
//! the write faults of run faults-6 (book records "the instruction at a made
//! the fault"); a search of all of `.text` for the same instruction shape
//! finds these six stores and no other.
//!
//! Each store becomes a jump to code of ours that does the store, then, if
//! the address lies in the heap `victory-gpu` names in [`Marks`], sets the
//! slot's byte in a map of slots and its 4 KiB chunk's byte in a map of
//! chunks, and jumps back. The flags are clobbered: at every site the next
//! instructions are a `movzx` and a `cmp`, which sets them before any use
//! (checked as part of the original bytes). Every register is put back.
//!
//! Each site's original bytes (from the `and` to the `cmp`) are compared
//! before anything is written; a site whose bytes differ is left alone and
//! said so (another build of the title).

/// The cells the stubs read, `#[repr(C)]` in `victory-gpu`: the heap's first
/// byte, its length in bytes (0: none, nothing is marked), the map of slots
/// (a byte each), the map of chunks (a byte for 128 slots).
pub const MARKS_LEN: usize = 32;

/// A store of the title's to hook: the window's offset in the image, the
/// store's offset in the window and its length, and the window's bytes.
pub struct Site {
    pub window: usize,
    pub store: usize,
    pub len: usize,
    pub bytes: &'static [u8],
}

/// The six record stores of the table fillers (game_scarlett_master.exe
/// 1.0.2.1, image offsets).
pub const SITES: [Site; 6] = [
    // +0x4d9efd0, loop 1: [rcx+rsi]
    Site { window: 0x4d9f04f, store: 10, len: 5, bytes: &[0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x31, 0x0f, 0xb6, 0x47, 0x0b, 0x39, 0xc5] },
    // loop 2: [rcx+rsi]
    Site { window: 0x4d9f10d, store: 10, len: 5, bytes: &[0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x31, 0x0f, 0xb6, 0x47, 0x0f, 0x41, 0x39, 0xc5] },
    // loop 3: [r9+rsi]
    Site { window: 0x4d9f19c, store: 12, len: 6, bytes: &[0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc4, 0xc1, 0x7c, 0x10, 0x02, 0xc4, 0xc1, 0x7c, 0x11, 0x04, 0x31, 0x44, 0x0f, 0xb6, 0x4f, 0x0d, 0x44, 0x39, 0xc8] },
    // +0x4eec8e0, loop 1: [r9+r8]
    Site { window: 0x4eec95d, store: 11, len: 6, bytes: &[0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c, 0x11, 0x04, 0x01, 0x0f, 0xb6, 0x42, 0x0b, 0x39, 0xc5] },
    // loop 2: [rdi+r8]
    Site { window: 0x4eeca08, store: 10, len: 6, bytes: &[0x81, 0xe7, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x06, 0xc4, 0xa1, 0x7c, 0x11, 0x04, 0x07, 0x0f, 0xb6, 0x72, 0x0d, 0x39, 0xf0] },
    // loop 3: [r10+r8]
    Site { window: 0x4eeca82, store: 11, len: 6, bytes: &[0x41, 0x81, 0xe2, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c, 0x11, 0x04, 0x02, 0x0f, 0xb6, 0x42, 0x0f, 0x41, 0x39, 0xc6] },
];

/// `lea rax, [base+index]` for a `vmovups [base+index], ymm0` (VEX, ModRM
/// 04h with a SIB byte, no displacement, the stack pointer not its base):
/// `None` for another form.
fn lea_of(store: &[u8]) -> Option<[u8; 4]> {
    let (x, b, op_at) = match *store.first()? {
        0xc5 => (false, false, 2),
        0xc4 => {
            let p = *store.get(1)?;
            if p & 0x1f != 1 {
                return None;
            }
            (p & 0x40 == 0, p & 0x20 == 0, 3)
        }
        _ => return None,
    };
    let (op, modrm, sib) = (*store.get(op_at)?, *store.get(op_at + 1)?, *store.get(op_at + 2)?);
    if op != 0x11 || modrm != 0x04 || store.len() != op_at + 3 {
        return None;
    }
    // No base (mode 0, base 101), or the stack pointer as base, or rsp as index (none).
    if sib & 7 == 5 || (sib & 7 == 4 && !b) || (sib >> 3 & 7 == 4 && !x) {
        return None;
    }
    Some([0x48 | u8::from(x) << 1 | u8::from(b), 0x8d, 0x04, sib])
}

/// The code at `to` for the store `store` at `at`:
/// - Check whether `[base+index]` falls inside `[marks.start, marks.start + marks.len)`.
/// - If outside: restore registers, run `<store>`, and jump back (fast path).
/// - If inside: **pre-mark** `slots[slot] = 1` and `chunks[chunk] = 1` BEFORE
///   `<store>` (closing the store-before-mark race against `sync_heaps`
///   fallback scans), restore all registers + `rsp` so `<store>` runs in its
///   exact original context (VEH-safe on guard-page faults), run `<store>`,
///   then **post-mark** `slots[slot] = 1` and `chunks[chunk] = 1` AFTER
///   `<store>` (so a concurrent `take()` that drained the pre-mark before
///   `<store>` landed still observes the new bytes on the next `take()`), and
///   jump back to `at + store.len()`.
fn stub(store: &[u8], at: usize, to: usize, marks: usize) -> Option<Vec<u8>> {
    let lea = lea_of(store)?;
    let mark_body: [u8; 24] = [
        0x48, 0xc1, 0xe8, 0x05, // shr rax, 5: the slot
        0x48, 0x8b, 0x4a, 0x10, // mov rcx, [rdx+16]
        0xc6, 0x04, 0x01, 0x01, // mov byte [rcx+rax], 1
        0x48, 0xc1, 0xe8, 0x07, // shr rax, 7: the chunk
        0x48, 0x8b, 0x4a, 0x18, // mov rcx, [rdx+24]
        0xc6, 0x04, 0x01, 0x01, // mov byte [rcx+rax], 1
    ];
    let mut out = Vec::with_capacity(144);
    // Phase 1: range check + pre-mark before the store.
    out.push(0x50); // push rax
    out.extend(lea); // lea rax, [base+index]
    out.extend([0x52, 0x51]); // push rdx ; push rcx
    out.extend([0x48, 0xba]); // mov rdx, marks
    out.extend((marks as u64).to_le_bytes());
    out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]
    out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
    out.extend([0x73, 0x00]); // jae .outside (patched below)
    let jae_outside_disp = out.len() - 1;
    let jae_outside_next = out.len();
    out.extend(mark_body);
    out.extend([0x59, 0x5a, 0x58]); // pop rcx ; pop rdx ; pop rax
    out.extend(store); // original store with pristine registers & rsp
    // Phase 2: post-mark after the store (re-checks marks.len for let_go safety).
    out.push(0x50); // push rax
    out.extend(lea); // lea rax, [base+index]
    out.extend([0x52, 0x51]); // push rdx ; push rcx
    out.extend([0x48, 0xba]); // mov rdx, marks
    out.extend((marks as u64).to_le_bytes());
    out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]
    out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
    out.extend([0x73, mark_body.len() as u8]); // jae .done
    out.extend(mark_body);
    // .done:
    out.extend([0x59, 0x5a, 0x58]); // pop rcx ; pop rdx ; pop rax
    let back_in = crate::console_insn::jump(to + out.len(), at + store.len())?;
    out.extend(back_in);
    // .outside:
    let outside = out.len();
    let outside_disp = i8::try_from(outside.checked_sub(jae_outside_next)?).ok()?;
    out[jae_outside_disp] = outside_disp as u8;
    out.extend([0x59, 0x5a, 0x58]); // pop rcx ; pop rdx ; pop rax
    out.extend(store);
    let back_out = crate::console_insn::jump(to + out.len(), at + store.len())?;
    out.extend(back_out);
    Some(out)
}

/// What became of a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hooked {
    Done,
    /// Its bytes are not the ones expected: left alone.
    Differs,
    /// Outside the image, no room for code near it, or Windows refused.
    Failed(&'static str),
}

/// Hooks `sites` in the image of `size` bytes at `base`; the stubs read the
/// cells at `marks` ([`MARKS_LEN`] bytes). What became of each, said in the
/// log too.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet; `marks` stays
/// valid for as long as the image runs.
pub unsafe fn apply(base: usize, size: usize, marks: usize, sites: &[Site]) -> Vec<Hooked> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut room: Option<(usize, usize)> = None;
    let mut out = Vec::new();
    for site in sites {
        let done = (|| {
            if site.window.checked_add(site.bytes.len()).is_none_or(|end| end > size) || site.store + site.len > site.bytes.len() {
                return Hooked::Failed("outside the image");
            }
            let window = base + site.window;
            // SAFETY: inside the mapped image (checked above).
            let there = unsafe { std::slice::from_raw_parts(window as *const u8, site.bytes.len()) };
            if there != site.bytes {
                return Hooked::Differs;
            }
            let at = window + site.store;
            if room.is_none() {
                room = crate::console_insn::near(at).map(|place| (place, 0));
            }
            let Some((place, used)) = room.as_mut() else {
                return Hooked::Failed("no room for code near the title's");
            };
            let to = *place + *used;
            let store = &site.bytes[site.store..site.store + site.len];
            let (Some(body), Some(jump)) = (stub(store, at, to, marks), crate::console_insn::jump(at, to)) else {
                return Hooked::Failed("not a store of the expected form, or out of reach");
            };
            if *used + body.len() > crate::console_insn::STUBS_LEN {
                return Hooked::Failed("no room for code near the title's");
            }
            let mut old = 0;
            // SAFETY: our own code memory, which nothing runs yet; then the
            // store, whose page is made writable for it and put back, and
            // which nothing runs either.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), to as *mut u8, body.len());
                if VirtualProtect(at as *const _, site.len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    return Hooked::Failed("Windows would not make it writable");
                }
                std::ptr::write_bytes(at as *mut u8, 0x90, site.len);
                std::ptr::copy_nonoverlapping(jump.as_ptr(), at as *mut u8, jump.len());
                VirtualProtect(at as *const _, site.len, old, &mut old);
                FlushInstructionCache(GetCurrentProcess(), at as *const _, site.len);
            }
            *used += body.len().next_multiple_of(16);
            Hooked::Done
        })();
        let offset = format_args!("{:#x}", site.window + site.store);
        match done {
            Hooked::Done => tracing::info!(site = %offset, "renderer port 2: the title's descriptor record store at title+site patched to tell the layer its slot (VICTORY_PORT_DESCRIPTORS)"),
            Hooked::Differs => tracing::warn!(site = %offset, "renderer port 2: not patched, the title's bytes there are not the ones expected (another build?)"),
            Hooked::Failed(why) => tracing::warn!(site = %offset, why, "renderer port 2: not patched"),
        }
        out.push(done);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sites_lea_their_stores_address() {
        let leas: Vec<[u8; 4]> = SITES.iter().map(|s| lea_of(&s.bytes[s.store..s.store + s.len]).unwrap()).collect();
        assert_eq!(leas[0], [0x48, 0x8d, 0x04, 0x31]); // lea rax, [rcx+rsi]
        assert_eq!(leas[1], [0x48, 0x8d, 0x04, 0x31]);
        assert_eq!(leas[2], [0x49, 0x8d, 0x04, 0x31]); // [r9+rsi]
        assert_eq!(leas[3], [0x4b, 0x8d, 0x04, 0x01]); // [r9+r8]
        assert_eq!(leas[4], [0x4a, 0x8d, 0x04, 0x07]); // [rdi+r8]
        assert_eq!(leas[5], [0x4b, 0x8d, 0x04, 0x02]); // [r10+r8]
        // Each window: the store follows the record's load and is followed
        // by a movzx and the cmp that sets the flags.
        for s in &SITES {
            let after = &s.bytes[s.store + s.len..];
            let movzx = if after[0] == 0x0f { 0 } else { 1 };
            assert_eq!(&after[movzx..movzx + 2], &[0x0f, 0xb6]);
            assert_eq!(after[after.len() - 2], 0x39, "cmp");
        }
        // A store with a displacement, of the stack, of another kind: not taken.
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x44, 0x31, 0x08]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x04, 0x24]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x10, 0x04, 0x31]).is_none());
    }

    /// The marks `victory-gpu` keeps, for a test.
    #[repr(C)]
    struct Marks {
        start: u64,
        len: u64,
        slots: *mut u8,
        chunks: *mut u8,
    }

    /// Each site's window in code of a test's, twice, one copy hooked, both
    /// run: the record is stored as before, the registers the window leaves
    /// are the same, and the slot and its chunk are marked when the table is
    /// in the heap, not otherwise.
    #[test]
    fn a_hooked_site_stores_and_marks_its_slot() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // mov to, from (64 bits)
        let mov = |to: u8, from: u8| [0x48 | (from >> 3) << 2 | (to >> 3), 0x89, 0xc0 | (from & 7) << 3 | (to & 7)];
        for site in &SITES {
            let store = &site.bytes[site.store..site.store + site.len];
            let lea = lea_of(store).unwrap();
            let base = lea[3] & 7 | (lea[0] & 1) << 3;
            let index = lea[3] >> 3 & 7 | (lea[0] & 2) << 2;
            let load_at = site.store - if site.bytes[site.store - 5] == 0xc4 { 5 } else { 4 };
            let load = &site.bytes[load_at..site.store];
            let load_reg = if load[0] == 0xc4 { load[4] & 7 | 8 } else { load[3] & 7 };
            // fn(offset: rcx, table: rdx, record: r8, scratch: r9) -> rax:
            // the store's address as the window leaves its registers.
            let mut code: Vec<u8> = vec![0x56, 0x57, 0x53]; // push rsi ; push rdi ; push rbx
            code.extend(mov(11, 1)); // r11 = offset
            code.extend(mov(3, 2)); // rbx = table
            code.extend(mov(6, 8)); // rsi = record
            for scratch in [7, 10, 2] {
                code.extend(mov(scratch, 9)); // rdi, r10, rdx: what the movzx reads
            }
            // The window's `and` masks the offset's register; the table is in
            // the other. (The record first: rsi may be one of them.)
            let rex = usize::from(site.bytes[0] == 0x41);
            let offset = site.bytes[rex + 1] & 7 | if rex == 1 { 8 } else { 0 };
            assert!(offset == base || offset == index);
            let table = if offset == base { index } else { base };
            code.extend(mov(load_reg, 6));
            code.extend(mov(table, 3));
            code.extend(mov(offset, 11));
            let at = code.len();
            code.extend(site.bytes);
            code.extend(lea); // lea rax, [base+index]
            code.extend([0x5b, 0x5f, 0x5e]); // pop rbx ; pop rdi ; pop rsi
            code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
            type F = unsafe extern "system" fn(u64, *mut u8, *const u8, *mut u8) -> u64;
            // SAFETY: fresh memory for code and data, one copy changed before it runs.
            unsafe {
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                assert_ne!(place, 0);
                let (plain, hooked) = (place + 0x100 - at, place + 0x1100 - at);
                std::ptr::copy_nonoverlapping(code.as_ptr(), plain as *mut u8, code.len());
                std::ptr::copy_nonoverlapping(code.as_ptr(), hooked as *mut u8, code.len());
                let heap = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
                let scratch = [0u8; 64];
                let (mut slots, mut chunks) = (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
                let marks = Box::new(Marks { start: heap as u64 + 0x2000, len: 0x8000, slots: slots.as_mut_ptr(), chunks: chunks.as_mut_ptr() });
                let mine = [Site { window: 0x1100, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), [Hooked::Done]);
                assert_eq!(*((place + 0x1100 + site.store) as *const u8), 0xe9);
                let (plain, hooked): (F, F) = (
                    std::mem::transmute::<usize, F>(plain),
                    std::mem::transmute::<usize, F>(hooked),
                );
                let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x5a);
                // In the heap: slot 0x40 + 3 of its start.
                let table = heap.add(0x2000 + 0x40 * 32);
                let first = plain(0x60 | 0x8000_0000, table.add(0x4000), record.as_ptr(), scratch.as_ptr().cast_mut());
                let second = hooked(0x60 | 0x8000_0000, table, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(first.wrapping_sub(0x4000), second, "the registers the window leaves");
                assert_eq!(std::slice::from_raw_parts(table.add(0x60), 32), &record);
                assert_eq!(std::slice::from_raw_parts(table.add(0x4060), 32), &record);
                let slot = 0x40 + 3;
                assert_eq!(slots[slot], 1);
                assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 1);
                assert_eq!(chunks[slot / 128], 1);
                // Before the heap: stored, not marked.
                slots.fill(0);
                chunks.fill(0);
                let below = heap.add(0x1000);
                hooked(0x20, below, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(std::slice::from_raw_parts(below.add(0x20), 32), &record);
                assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
                drop(marks);
            }
        }
    }

    #[test]
    fn other_bytes_are_left_alone() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // SAFETY: fresh memory of our own.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            let mut bytes = SITES[0].bytes.to_vec();
            bytes[3] = 0x7e; // and ecx, 7EE0h
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), place.add(0x100), bytes.len());
            let mine = [Site { window: 0x100, ..SITES[0] }];
            assert_eq!(apply(place as usize, 0x10000, 0, &mine), [Hooked::Differs]);
            assert_eq!(std::slice::from_raw_parts(place.add(0x100), bytes.len()), &bytes[..]);
            let outside = [Site { window: 0xfff0, ..SITES[0] }];
            assert!(matches!(apply(place as usize, 0x10000, 0, &outside)[..], [Hooked::Failed(_)]));
        }
    }

    /// Proves on real x86-64 execution that every site stub **pre-marks** the
    /// slot and chunk BEFORE `<original vmovups>` faults on a guarded page,
    /// runs `<original vmovups>` with pristine registers/stack, and
    /// **post-marks** the slot and chunk again AFTER `<original vmovups>` even
    /// if a concurrent `take()` drained the pre-mark right at the store.
    #[test]
    fn stub_pre_marks_before_store_and_post_marks_after_store() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, EXCEPTION_POINTERS, RemoveVectoredExceptionHandler,
        };
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualProtect,
        };

        const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
        const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

        static TARGET_ADDR: AtomicUsize = AtomicUsize::new(0);
        static SLOT_PTR: AtomicUsize = AtomicUsize::new(0);
        static CHUNK_PTR: AtomicUsize = AtomicUsize::new(0);
        static SAW_FAULT: AtomicBool = AtomicBool::new(false);
        static PRE_MARKED_OK: AtomicBool = AtomicBool::new(false);

        unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: Windows passes valid exception pointers.
            let rec = unsafe { &*(*info).ExceptionRecord };
            if rec.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || rec.NumberParameters < 2 {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let fault_addr = rec.ExceptionInformation[1];
            let target = TARGET_ADDR.load(Ordering::Acquire);
            if target == 0 || fault_addr != target {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let slot_p = SLOT_PTR.load(Ordering::Acquire) as *mut u8;
            let chunk_p = CHUNK_PTR.load(Ordering::Acquire) as *mut u8;
            // SAFETY: valid test vectors alive for the duration of the hooked call.
            unsafe {
                let pre_ok = *slot_p == 1 && *chunk_p == 1;
                PRE_MARKED_OK.store(pre_ok, Ordering::Release);
                // Simulate concurrent `port_descriptors::take()` draining the pre-mark
                // right before the store lands; post-mark must re-set both to 1.
                *slot_p = 0;
                *chunk_p = 0;
                let mut old = 0u32;
                let page = target & !0xfff;
                assert_ne!(VirtualProtect(page as *const _, 0x1000, PAGE_READWRITE, &mut old), 0);
            }
            SAW_FAULT.store(true, Ordering::Release);
            EXCEPTION_CONTINUE_EXECUTION
        }

        let mov = |to: u8, from: u8| [0x48 | (from >> 3) << 2 | (to >> 3), 0x89, 0xc0 | (from & 7) << 3 | (to & 7)];
        // SAFETY: scoped VEH handler removed before returning.
        let veh = unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
        assert!(!veh.is_null());
        struct RemoveVeh(*mut std::ffi::c_void);
        impl Drop for RemoveVeh {
            fn drop(&mut self) {
                TARGET_ADDR.store(0, Ordering::Release);
                // SAFETY: handle returned by `AddVectoredExceptionHandler`.
                unsafe { RemoveVectoredExceptionHandler(self.0) };
            }
        }
        let _guard = RemoveVeh(veh);

        for site in &SITES {
            let store = &site.bytes[site.store..site.store + site.len];
            let lea = lea_of(store).unwrap();
            let base = lea[3] & 7 | (lea[0] & 1) << 3;
            let index = lea[3] >> 3 & 7 | (lea[0] & 2) << 2;
            let load_at = site.store - if site.bytes[site.store - 5] == 0xc4 { 5 } else { 4 };
            let load = &site.bytes[load_at..site.store];
            let load_reg = if load[0] == 0xc4 { load[4] & 7 | 8 } else { load[3] & 7 };
            let mut code: Vec<u8> = vec![0x56, 0x57, 0x53];
            code.extend(mov(11, 1));
            code.extend(mov(3, 2));
            code.extend(mov(6, 8));
            for scratch in [7, 10, 2] {
                code.extend(mov(scratch, 9));
            }
            let rex = usize::from(site.bytes[0] == 0x41);
            let offset = site.bytes[rex + 1] & 7 | if rex == 1 { 8 } else { 0 };
            let table = if offset == base { index } else { base };
            code.extend(mov(load_reg, 6));
            code.extend(mov(table, 3));
            code.extend(mov(offset, 11));
            let at = code.len();
            code.extend(site.bytes);
            code.extend(lea);
            code.extend([0x5b, 0x5f, 0x5e]);
            code.extend([0xc5, 0xf8, 0x77, 0xc3]);
            type F = unsafe extern "system" fn(u64, *mut u8, *const u8, *mut u8) -> u64;

            // SAFETY: fresh executable/data allocations owned by this test.
            unsafe {
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                assert_ne!(place, 0);
                let (plain_addr, hooked_addr) = (place + 0x100 - at, place + 0x1100 - at);
                std::ptr::copy_nonoverlapping(code.as_ptr(), plain_addr as *mut u8, code.len());
                std::ptr::copy_nonoverlapping(code.as_ptr(), hooked_addr as *mut u8, code.len());
                let heap = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u8;
                let scratch = [0u8; 64];
                let (mut slots, mut chunks) = (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
                let marks = Box::new(Marks {
                    start: heap as u64 + 0x2000,
                    len: 0x8000,
                    slots: slots.as_mut_ptr(),
                    chunks: chunks.as_mut_ptr(),
                });
                let mine = [Site { window: 0x1100, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), [Hooked::Done]);
                let (plain, hooked): (F, F) = (
                    std::mem::transmute::<usize, F>(plain_addr),
                    std::mem::transmute::<usize, F>(hooked_addr),
                );
                let record: [u8; 32] = std::array::from_fn(|k| (k as u8).wrapping_mul(3) ^ 0xa5);
                let table = heap.add(0x2000 + 0x40 * 32);
                let expected = plain(0x60, table.add(0x4000), record.as_ptr(), scratch.as_ptr().cast_mut());
                let target = table.add(0x60);
                let slot = 0x40 + 3;
                SLOT_PTR.store(slots.as_mut_ptr().add(slot) as usize, Ordering::Release);
                CHUNK_PTR.store(chunks.as_mut_ptr().add(slot / 128) as usize, Ordering::Release);
                SAW_FAULT.store(false, Ordering::Release);
                PRE_MARKED_OK.store(false, Ordering::Release);
                TARGET_ADDR.store(target as usize, Ordering::Release);
                let mut old = 0u32;
                assert_ne!(VirtualProtect((target as usize & !0xfff) as *const _, 0x1000, PAGE_READONLY, &mut old), 0);

                let ret = hooked(0x60, table, record.as_ptr(), scratch.as_ptr().cast_mut());
                TARGET_ADDR.store(0, Ordering::Release);

                assert!(SAW_FAULT.load(Ordering::Acquire), "store must have faulted on read-only page");
                assert!(PRE_MARKED_OK.load(Ordering::Acquire), "slot & chunk must be pre-marked BEFORE store executes");
                assert_eq!(ret, expected.wrapping_sub(0x4000), "registers must survive VEH fault and post-mark");
                assert_eq!(std::slice::from_raw_parts(target, 32), &record);
                assert_eq!(slots[slot], 1, "post-mark must re-set slot after concurrent take() drained pre-mark");
                assert_eq!(chunks[slot / 128], 1, "post-mark must re-set chunk after concurrent take() drained pre-mark");
                drop(marks);
            }
        }
    }
}
