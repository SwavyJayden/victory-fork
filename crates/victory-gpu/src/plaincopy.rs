//! Copies of title memory made in place, by this process, with a way back
//! when a page is not there (`VICTORY_COPY_PLAIN=1`; off by default).
//!
//! Every copy of title memory into a host buffer goes through the kernel
//! (`ReadProcessMemory` on our own process, `crate::host::copy_mapped`):
//! the title unmaps and remaps its memory while we read it, a read in
//! place of a page that is gone is an access violation, and the kernel's
//! copy fails instead. It costs what a system call that locks and probes
//! every page costs: run 225's profile has the submitting thread 35 ms a
//! frame inside `ZwReadVirtualMemory` for about 100 MB (3 GB a second),
//! where the processor's own copy moves that much in a third of the time.
//!
//! Here the copy runs in place in a leaf assembly routine that uses no
//! stack and modifies no callee-saved registers: 256-bit AVX2 unrolled
//! `vmovdqu` loops when AVX2 is available (`VICTORY_COPY_AVX2`, on by
//! default unless `=0`, saturating 64-byte write-combining burst lines on
//! Zen 3 instead of `rep movsb`), with a `rep movsb` fallback otherwise.
//! An exception handler, first in the line, looks at every access
//! violation: one that is a READ inside our copy routine's guarded range
//! continues at the routine's fault exit, which returns 0, and the
//! caller falls back to the kernel's copy as before (it says which page
//! is gone and fills in zeros). A write that faults during a read copy is
//! not ours to take: the destination is host memory, which is always
//! there; such a fault goes on to the handlers after this one.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_copy_guarded",
    ".globl victory_copy_guarded_at",
    ".globl victory_copy_guarded_fault",
    ".globl victory_write_guarded",
    ".globl victory_write_guarded_at",
    ".globl victory_write_guarded_fault",
    ".globl victory_copy_avx2",
    ".globl victory_copy_avx2_try_start",
    ".globl victory_copy_avx2_try_end",
    ".globl victory_copy_avx2_fault",
    ".globl victory_write_avx2",
    ".globl victory_write_avx2_try_start",
    ".globl victory_write_avx2_try_end",
    ".globl victory_write_avx2_fault",
    ".globl victory_diff_avx2",
    ".globl victory_diff_avx2_try_start",
    ".globl victory_diff_avx2_try_end",
    ".globl victory_diff_avx2_fault",
    "victory_copy_guarded:",
    // rcx: to, rdx: from, r8: bytes. rsi and rdi are the caller's: kept in
    // r9 and r10, which are ours to use, so nothing is on the stack.
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_copy_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_copy_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",
    "victory_write_guarded:",
    "    mov r9, rsi",
    "    mov r10, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    "victory_write_guarded_at:",
    "    rep movsb",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    mov eax, 1",
    "    ret",
    "victory_write_guarded_fault:",
    "    mov rsi, r9",
    "    mov rdi, r10",
    "    xor eax, eax",
    "    ret",

    // =========================================================================
    // Fault-guarded 256-bit AVX2 read copy: title memory (`rdx`) -> host (`rcx`).
    // Windows x64 leaf function: touches ONLY volatile registers (rax, rcx, rdx,
    // r8, ymm0..ymm3), ZERO callee-saved registers, ZERO stack adjustments.
    //
    // Aligns `from` (`rdx`) to 8 bytes then 32 bytes before vector loads so no
    // load ever straddles a 4096-byte page boundary; stores execute in strict
    // ascending order, preserving exact prefix bytes before any faulting page.
    // =========================================================================
    ".p2align 4",
    "victory_copy_avx2:",
    "    cmp r8, 32",
    "    jb .Lcopy_small_entry",
    "victory_copy_avx2_try_start:",
    "    test dl, 31",
    "    jz .Lcopy_avx_check128",
    // Align `from` (`rdx`) to 8 bytes (<= 7 bytes; r8 >= 32 -> r8 >= 25).
    "    test dl, 7",
    "    jz .Lcopy_align32_loop",
    ".Lcopy_align8_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    test dl, 7",
    "    jnz .Lcopy_align8_loop",
    "    test dl, 31",
    "    jz .Lcopy_after_align32",
    // Align `from` (`rdx`) to 32 bytes using 8-byte aligned loads (<= 24 bytes; r8 >= 1).
    ".Lcopy_align32_loop:",
    "    mov rax, qword ptr [rdx]",
    "    mov qword ptr [rcx], rax",
    "    add rdx, 8",
    "    add rcx, 8",
    "    sub r8, 8",
    "    test dl, 31",
    "    jnz .Lcopy_align32_loop",
    ".Lcopy_after_align32:",
    // `from` is now 32-byte aligned and no YMM register has been touched yet.
    "    cmp r8, 32",
    "    jb .Lcopy_small_qword_check",
    ".Lcopy_avx_check128:",
    "    cmp r8, 128",
    "    jb .Lcopy_avx32_loop",
    ".p2align 4",
    ".Lcopy_avx128_loop:",
    "    vmovdqu ymm0, ymmword ptr [rdx]",
    "    vmovdqu ymmword ptr [rcx], ymm0",
    "    vmovdqu ymm1, ymmword ptr [rdx + 32]",
    "    vmovdqu ymmword ptr [rcx + 32], ymm1",
    "    vmovdqu ymm2, ymmword ptr [rdx + 64]",
    "    vmovdqu ymmword ptr [rcx + 64], ymm2",
    "    vmovdqu ymm3, ymmword ptr [rdx + 96]",
    "    vmovdqu ymmword ptr [rcx + 96], ymm3",
    "    add rdx, 128",
    "    add rcx, 128",
    "    sub r8, 128",
    "    cmp r8, 128",
    "    jae .Lcopy_avx128_loop",
    "    test r8, r8",
    "    jz .Lcopy_done_avx",
    "    cmp r8, 32",
    "    jb .Lcopy_avx_tail8_check",
    ".Lcopy_avx32_loop:",
    "    vmovdqu ymm0, ymmword ptr [rdx]",
    "    vmovdqu ymmword ptr [rcx], ymm0",
    "    add rdx, 32",
    "    add rcx, 32",
    "    sub r8, 32",
    "    cmp r8, 32",
    "    jae .Lcopy_avx32_loop",
    "    test r8, r8",
    "    jz .Lcopy_done_avx",
    ".Lcopy_avx_tail8_check:",
    "    cmp r8, 8",
    "    jb .Lcopy_avx_tail1_loop",
    ".Lcopy_avx_tail8_loop:",
    "    mov rax, qword ptr [rdx]",
    "    mov qword ptr [rcx], rax",
    "    add rdx, 8",
    "    add rcx, 8",
    "    sub r8, 8",
    "    cmp r8, 8",
    "    jae .Lcopy_avx_tail8_loop",
    "    test r8, r8",
    "    jz .Lcopy_done_avx",
    ".Lcopy_avx_tail1_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    jnz .Lcopy_avx_tail1_loop",
    "    jmp .Lcopy_done_avx",
    // Non-AVX small-copy path (< 32 bytes): avoids touching YMM & skips vzeroupper.
    ".Lcopy_small_entry:",
    "    cmp r8, 8",
    "    jb .Lcopy_small_byte_check",
    "    test dl, 7",
    "    jz .Lcopy_small_qword_loop",
    ".Lcopy_small_align8_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    test dl, 7",
    "    jnz .Lcopy_small_align8_loop",
    ".Lcopy_small_qword_check:",
    "    cmp r8, 8",
    "    jb .Lcopy_small_byte_check",
    ".Lcopy_small_qword_loop:",
    "    mov rax, qword ptr [rdx]",
    "    mov qword ptr [rcx], rax",
    "    add rdx, 8",
    "    add rcx, 8",
    "    sub r8, 8",
    "    cmp r8, 8",
    "    jae .Lcopy_small_qword_loop",
    ".Lcopy_small_byte_check:",
    "    test r8, r8",
    "    jz .Lcopy_done_no_avx",
    ".Lcopy_small_byte_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    jnz .Lcopy_small_byte_loop",
    "victory_copy_avx2_try_end:",
    ".Lcopy_done_no_avx:",
    "    mov eax, 1",
    "    ret",
    ".Lcopy_done_avx:",
    "    vzeroupper",
    "    mov eax, 1",
    "    ret",
    "victory_copy_avx2_fault:",
    "    vzeroupper",
    "    xor eax, eax",
    "    ret",

    // =========================================================================
    // Fault-guarded 256-bit AVX2 write-back: host (`rdx`) -> title memory (`rcx`).
    // Windows x64 leaf function: touches ONLY volatile registers (rax, rcx, rdx,
    // r8, ymm0..ymm3), ZERO callee-saved registers, ZERO stack adjustments.
    //
    // Aligns `to` (`rcx`) to 8 bytes then 32 bytes before vector stores so no
    // store ever straddles a 4096-byte page boundary; writes execute in strict
    // ascending order, preserving exact prefix bytes before any faulting page.
    // =========================================================================
    ".p2align 4",
    "victory_write_avx2:",
    "    cmp r8, 32",
    "    jb .Lwrite_small_entry",
    "victory_write_avx2_try_start:",
    "    test cl, 31",
    "    jz .Lwrite_avx_check128",
    // Align `to` (`rcx`) to 8 bytes (<= 7 bytes; r8 >= 32 -> r8 >= 25).
    "    test cl, 7",
    "    jz .Lwrite_align32_loop",
    ".Lwrite_align8_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    test cl, 7",
    "    jnz .Lwrite_align8_loop",
    "    test cl, 31",
    "    jz .Lwrite_after_align32",
    // Align `to` (`rcx`) to 32 bytes using 8-byte aligned stores (<= 24 bytes; r8 >= 1).
    ".Lwrite_align32_loop:",
    "    mov rax, qword ptr [rdx]",
    "    mov qword ptr [rcx], rax",
    "    add rdx, 8",
    "    add rcx, 8",
    "    sub r8, 8",
    "    test cl, 31",
    "    jnz .Lwrite_align32_loop",
    ".Lwrite_after_align32:",
    // `to` is now 32-byte aligned and no YMM register has been touched yet.
    "    cmp r8, 32",
    "    jb .Lwrite_small_qword_check",
    ".Lwrite_avx_check128:",
    "    cmp r8, 128",
    "    jb .Lwrite_avx32_loop",
    ".p2align 4",
    ".Lwrite_avx128_loop:",
    "    vmovdqu ymm0, ymmword ptr [rdx]",
    "    vmovdqu ymmword ptr [rcx], ymm0",
    "    vmovdqu ymm1, ymmword ptr [rdx + 32]",
    "    vmovdqu ymmword ptr [rcx + 32], ymm1",
    "    vmovdqu ymm2, ymmword ptr [rdx + 64]",
    "    vmovdqu ymmword ptr [rcx + 64], ymm2",
    "    vmovdqu ymm3, ymmword ptr [rdx + 96]",
    "    vmovdqu ymmword ptr [rcx + 96], ymm3",
    "    add rdx, 128",
    "    add rcx, 128",
    "    sub r8, 128",
    "    cmp r8, 128",
    "    jae .Lwrite_avx128_loop",
    "    test r8, r8",
    "    jz .Lwrite_done_avx",
    "    cmp r8, 32",
    "    jb .Lwrite_avx_tail8_check",
    ".Lwrite_avx32_loop:",
    "    vmovdqu ymm0, ymmword ptr [rdx]",
    "    vmovdqu ymmword ptr [rcx], ymm0",
    "    add rdx, 32",
    "    add rcx, 32",
    "    sub r8, 32",
    "    cmp r8, 32",
    "    jae .Lwrite_avx32_loop",
    "    test r8, r8",
    "    jz .Lwrite_done_avx",
    ".Lwrite_avx_tail8_check:",
    "    cmp r8, 8",
    "    jb .Lwrite_avx_tail1_loop",
    ".Lwrite_avx_tail8_loop:",
    "    mov rax, qword ptr [rdx]",
    "    mov qword ptr [rcx], rax",
    "    add rdx, 8",
    "    add rcx, 8",
    "    sub r8, 8",
    "    cmp r8, 8",
    "    jae .Lwrite_avx_tail8_loop",
    "    test r8, r8",
    "    jz .Lwrite_done_avx",
    ".Lwrite_avx_tail1_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    jnz .Lwrite_avx_tail1_loop",
    "    jmp .Lwrite_done_avx",
    // Non-AVX small-write path (< 32 bytes): avoids touching YMM & skips vzeroupper.
    ".Lwrite_small_entry:",
    "    cmp r8, 8",
    "    jb .Lwrite_small_byte_check",
    "    test cl, 7",
    "    jz .Lwrite_small_qword_loop",
    ".Lwrite_small_align8_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    test cl, 7",
    "    jnz .Lwrite_small_align8_loop",
    ".Lwrite_small_qword_check:",
    "    cmp r8, 8",
    "    jb .Lwrite_small_byte_check",
    ".Lwrite_small_qword_loop:",
    "    mov rax, qword ptr [rdx]",
    "    mov qword ptr [rcx], rax",
    "    add rdx, 8",
    "    add rcx, 8",
    "    sub r8, 8",
    "    cmp r8, 8",
    "    jae .Lwrite_small_qword_loop",
    ".Lwrite_small_byte_check:",
    "    test r8, r8",
    "    jz .Lwrite_done_no_avx",
    ".Lwrite_small_byte_loop:",
    "    mov al, byte ptr [rdx]",
    "    mov byte ptr [rcx], al",
    "    inc rdx",
    "    inc rcx",
    "    dec r8",
    "    jnz .Lwrite_small_byte_loop",
    "victory_write_avx2_try_end:",
    ".Lwrite_done_no_avx:",
    "    mov eax, 1",
    "    ret",
    ".Lwrite_done_avx:",
    "    vzeroupper",
    "    mov eax, 1",
    "    ret",
    "victory_write_avx2_fault:",
    "    vzeroupper",
    "    xor eax, eax",
    "    ret",

    // =========================================================================
    // Fault-guarded 256-bit AVX2 in-place compare: title (`rcx`) vs host (`rdx`),
    // length `r8`. Windows x64 leaf function: touches ONLY volatile registers
    // (rax, rcx, rdx, r8, r9, r10, ymm0..ymm3), ZERO stack, ZERO callee-saved.
    //
    // Returns:
    //   - `len` (`r8`) if all bytes in `[0..len)` are equal
    //   - `0..len` exact byte offset of the first differing byte
    //   - `usize::MAX` (`-1`) if reading `title` faulted (caught by VEH)
    // =========================================================================
    ".p2align 4",
    "victory_diff_avx2:",
    "    xor r9, r9",
    "    mov r10, r8",
    "    cmp r10, 32",
    "    jb .Ldiff_small_entry",
    "victory_diff_avx2_try_start:",
    "    cmp r10, 128",
    "    jb .Ldiff_avx32_loop",
    ".p2align 4",
    ".Ldiff_avx128_loop:",
    "    vmovdqu ymm0, ymmword ptr [rcx + r9]",
    "    vpcmpeqb ymm0, ymm0, ymmword ptr [rdx + r9]",
    "    vmovdqu ymm1, ymmword ptr [rcx + r9 + 32]",
    "    vpcmpeqb ymm1, ymm1, ymmword ptr [rdx + r9 + 32]",
    "    vmovdqu ymm2, ymmword ptr [rcx + r9 + 64]",
    "    vpcmpeqb ymm2, ymm2, ymmword ptr [rdx + r9 + 64]",
    "    vmovdqu ymm3, ymmword ptr [rcx + r9 + 96]",
    "    vpcmpeqb ymm3, ymm3, ymmword ptr [rdx + r9 + 96]",
    "    vpand ymm0, ymm0, ymm1",
    "    vpand ymm2, ymm2, ymm3",
    "    vpand ymm0, ymm0, ymm2",
    "    vpmovmskb eax, ymm0",
    "    cmp eax, -1",
    "    jne .Ldiff_avx32_loop",
    "    add r9, 128",
    "    sub r10, 128",
    "    cmp r10, 128",
    "    jae .Ldiff_avx128_loop",
    "    test r10, r10",
    "    jz .Ldiff_equal_avx",
    "    cmp r10, 32",
    "    jb .Ldiff_avx_tail8_check",
    ".Ldiff_avx32_loop:",
    "    vmovdqu ymm0, ymmword ptr [rcx + r9]",
    "    vpcmpeqb ymm0, ymm0, ymmword ptr [rdx + r9]",
    "    vpmovmskb eax, ymm0",
    "    not eax",
    "    test eax, eax",
    "    jnz .Ldiff_found_mask",
    "    add r9, 32",
    "    sub r10, 32",
    "    cmp r10, 32",
    "    jae .Ldiff_avx32_loop",
    "    test r10, r10",
    "    jz .Ldiff_equal_avx",
    ".Ldiff_avx_tail8_check:",
    "    cmp r10, 8",
    "    jb .Ldiff_avx_tail1_loop",
    ".Ldiff_avx_tail8_loop:",
    "    mov rax, qword ptr [rcx + r9]",
    "    xor rax, qword ptr [rdx + r9]",
    "    jnz .Ldiff_found_qword_avx",
    "    add r9, 8",
    "    sub r10, 8",
    "    cmp r10, 8",
    "    jae .Ldiff_avx_tail8_loop",
    "    test r10, r10",
    "    jz .Ldiff_equal_avx",
    ".Ldiff_avx_tail1_loop:",
    "    mov al, byte ptr [rcx + r9]",
    "    cmp al, byte ptr [rdx + r9]",
    "    jne .Ldiff_found_here_avx",
    "    inc r9",
    "    dec r10",
    "    jnz .Ldiff_avx_tail1_loop",
    ".Ldiff_equal_avx:",
    "    vzeroupper",
    "    mov rax, r8",
    "    ret",
    ".Ldiff_found_mask:",
    "    bsf eax, eax",
    "    add rax, r9",
    "    vzeroupper",
    "    ret",
    ".Ldiff_found_qword_avx:",
    "    bsf rax, rax",
    "    shr rax, 3",
    "    add rax, r9",
    "    vzeroupper",
    "    ret",
    ".Ldiff_found_here_avx:",
    "    vzeroupper",
    "    mov rax, r9",
    "    ret",
    // Small-length path (< 32 bytes): avoids touching YMM registers & vzeroupper.
    ".Ldiff_small_entry:",
    "    cmp r10, 8",
    "    jb .Ldiff_small_byte_check",
    ".Ldiff_small_qword_loop:",
    "    mov rax, qword ptr [rcx + r9]",
    "    xor rax, qword ptr [rdx + r9]",
    "    jnz .Ldiff_found_qword_no_avx",
    "    add r9, 8",
    "    sub r10, 8",
    "    cmp r10, 8",
    "    jae .Ldiff_small_qword_loop",
    ".Ldiff_small_byte_check:",
    "    test r10, r10",
    "    jz .Ldiff_equal_no_avx",
    ".Ldiff_small_byte_loop:",
    "    mov al, byte ptr [rcx + r9]",
    "    cmp al, byte ptr [rdx + r9]",
    "    jne .Ldiff_found_here_no_avx",
    "    inc r9",
    "    dec r10",
    "    jnz .Ldiff_small_byte_loop",
    "victory_diff_avx2_try_end:",
    ".Ldiff_equal_no_avx:",
    "    mov rax, r8",
    "    ret",
    ".Ldiff_found_qword_no_avx:",
    "    bsf rax, rax",
    "    shr rax, 3",
    "    add rax, r9",
    "    ret",
    ".Ldiff_found_here_no_avx:",
    "    mov rax, r9",
    "    ret",
    "victory_diff_avx2_fault:",
    "    vzeroupper",
    "    mov rax, -1",
    "    ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_copy_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_guarded_at: u8;
    static victory_copy_guarded_fault: u8;
    fn victory_write_guarded(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_write_guarded_at: u8;
    static victory_write_guarded_fault: u8;
    fn victory_copy_avx2(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_copy_avx2_try_start: u8;
    static victory_copy_avx2_try_end: u8;
    static victory_copy_avx2_fault: u8;
    fn victory_write_avx2(to: *mut u8, from: *const u8, len: usize) -> u32;
    static victory_write_avx2_try_start: u8;
    static victory_write_avx2_try_end: u8;
    static victory_write_avx2_fault: u8;
    fn victory_diff_avx2(title: *const u8, held: *const u8, len: usize) -> usize;
    static victory_diff_avx2_try_start: u8;
    static victory_diff_avx2_try_end: u8;
    static victory_diff_avx2_fault: u8;
}

#[cfg(test)]
thread_local! {
    /// Forces [`on`] to return true on the current thread during tests without
    /// mutating process-wide environment variables or [`OnceLock`].
    static FORCE_ON_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) struct ForceOnForTest(bool);

#[cfg(test)]
impl Drop for ForceOnForTest {
    fn drop(&mut self) {
        FORCE_ON_IN_TEST.set(self.0);
    }
}

#[cfg(test)]
pub(crate) fn force_on_for_test() -> ForceOnForTest {
    let prev = FORCE_ON_IN_TEST.replace(true);
    if cfg!(target_arch = "x86_64") {
        install();
    }
    ForceOnForTest(prev)
}

#[cfg(test)]
pub(crate) fn copy_stats() -> (u64, u64, u64) {
    (COUNT[0].load(Relaxed), COUNT[1].load(Relaxed), COUNT[2].load(Relaxed))
}

/// Whether 256-bit AVX2 fault-guarded copies/writes are enabled by env
/// (`VICTORY_COPY_AVX2`, default ON unless `=0`).
#[cfg(target_arch = "x86_64")]
fn parse_avx2(val: Option<&str>) -> bool {
    val.is_none_or(|v| v.trim() != "0")
}

/// Whether the 256-bit AVX2 path is active on this CPU (`VICTORY_COPY_AVX2`,
/// default ON when AVX2 is supported unless `=0`).
#[cfg(target_arch = "x86_64")]
fn avx2_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::is_x86_feature_detected!("avx2")
            && parse_avx2(std::env::var("VICTORY_COPY_AVX2").ok().as_deref())
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_on() -> bool {
    false
}

/// Whether copies of title memory into host memory are made in place
/// (`VICTORY_COPY_PLAIN=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if cfg!(target_arch = "x86_64") && FORCE_ON_IN_TEST.get() {
        install();
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = cfg!(target_arch = "x86_64") && std::env::var_os("VICTORY_COPY_PLAIN").is_some_and(|v| v == "1");
        if on {
            install();
            let mode = if avx2_on() { ", avx2" } else { "" };
            tracing::info!(
                "copies of title memory into host buffers are made in place, with the kernel's copy behind them for a \
                 page that is not there (VICTORY_COPY_PLAIN=1{mode})"
            );
        }
        on
    })
}

/// Copies made in place, their bytes, the copies that met a page that was
/// not there (and went to the kernel's copy), and the copies not tried in
/// place because their source faulted lately ([`RECENT`]) or they were
/// under [`least`] bytes, since the last line.
static COUNT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Default fewest bytes copied in place (`VICTORY_COPY_PLAIN_MIN`, 16). A
/// read of a byte or a word (1-8 bytes) is how the layer ASKS whether a page
/// is there (a texture's first page, a record): it is expected to fail, and a
/// failed kernel read costs a microsecond where a fault and its handler cost
/// tens. Live benchmarks showed ~1,500 16-31 byte constant copies per frame
/// still falling back to `ReadProcessMemory` at 32 bytes, so the default
/// threshold is 16 bytes.
const DEFAULT_LEAST: usize = 16;

fn parse_least(val: Option<&str>) -> usize {
    val.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(DEFAULT_LEAST)
}

/// Minimum byte length copied in place (`VICTORY_COPY_PLAIN_MIN`, default 16).
fn least() -> usize {
    static LEAST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LEAST.get_or_init(|| parse_least(std::env::var("VICTORY_COPY_PLAIN_MIN").ok().as_deref()))
}

/// The 64 KiB blocks of title memory a copy in place faulted in lately:
/// their copies go straight to the kernel's, which reads round the hole.
/// By the block's number, in the place its low bits name (a later block
/// with the same low bits takes the place). Forgotten every two seconds:
/// the title maps the memory in the end.
static RECENT: [std::sync::atomic::AtomicUsize; 1024] = [const { std::sync::atomic::AtomicUsize::new(0) }; 1024];
/// When [`RECENT`] was last emptied (milliseconds of [`now_ms`]).
static EMPTIED: AtomicU64 = AtomicU64::new(0);

fn now_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// The numbers of the first and last 64 KiB block of `[from, from + len)`,
/// never 0 (0 is an empty place).
fn blocks(from: usize, len: usize) -> (usize, usize) {
    ((from >> 16) | 1 << 63, ((from + len.max(1) - 1) >> 16) | 1 << 63)
}

/// Whether a copy from there faulted lately.
fn faulted_lately(from: usize, len: usize) -> bool {
    let now = now_ms();
    let emptied = EMPTIED.load(Relaxed);
    if now.saturating_sub(emptied) >= 2_000 && EMPTIED.compare_exchange(emptied, now, Relaxed, Relaxed).is_ok() {
        for place in &RECENT {
            place.store(0, Relaxed);
        }
        return false;
    }
    let (first, last) = blocks(from, len);
    RECENT[first & 1023].load(Relaxed) == first || RECENT[last & 1023].load(Relaxed) == last
}

/// A copy from there faulted.
fn note_fault(from: usize, len: usize) {
    let (first, last) = blocks(from, len);
    RECENT[first & 1023].store(first, Relaxed);
    RECENT[last & 1023].store(last, Relaxed);
}

/// The handler that turns a read fault inside [`guarded`] into its return
/// of false; put first in the line, once.
#[cfg(target_arch = "x86_64")]
fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // context it will continue with.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        let rip = context.Rip;
        const READ: usize = 0;
        const WRITE: usize = 1;

        // Only at our own copy/compare instructions (read faults from title memory;
        // a write fault to host memory MUST NOT be swallowed).
        let in_copy_rep = rip == (&raw const victory_copy_guarded_at) as u64;
        let in_copy_avx2 = !in_copy_rep
            && rip >= (&raw const victory_copy_avx2_try_start) as u64
            && rip < (&raw const victory_copy_avx2_try_end) as u64;
        let in_diff_avx2 = !in_copy_rep
            && !in_copy_avx2
            && rip >= (&raw const victory_diff_avx2_try_start) as u64
            && rip < (&raw const victory_diff_avx2_try_end) as u64;
        if in_copy_rep || in_copy_avx2 || in_diff_avx2 {
            let read = r.NumberParameters >= 1 && r.ExceptionInformation[0] == READ;
            let gone =
                (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && read;
            let guard =
                r.ExceptionCode == EXCEPTION_GUARD_PAGE && (r.NumberParameters == 0 || r.ExceptionInformation[0] == READ);
            if gone || guard {
                context.Rip = if in_diff_avx2 {
                    (&raw const victory_diff_avx2_fault) as u64
                } else if in_copy_avx2 {
                    (&raw const victory_copy_avx2_fault) as u64
                } else {
                    (&raw const victory_copy_guarded_fault) as u64
                };
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }

        // Only at our own write-back instructions (write faults to title memory;
        // a read fault from host memory MUST NOT be swallowed).
        let in_write_rep = rip == (&raw const victory_write_guarded_at) as u64;
        let in_write_avx2 = !in_write_rep
            && rip >= (&raw const victory_write_avx2_try_start) as u64
            && rip < (&raw const victory_write_avx2_try_end) as u64;
        if in_write_rep || in_write_avx2 {
            let write = r.NumberParameters >= 1 && r.ExceptionInformation[0] == WRITE;
            let gone =
                (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) && write;
            let guard =
                r.ExceptionCode == EXCEPTION_GUARD_PAGE && (r.NumberParameters == 0 || r.ExceptionInformation[0] == WRITE);
            if gone || guard {
                context.Rip = if in_write_avx2 {
                    (&raw const victory_write_avx2_fault) as u64
                } else {
                    (&raw const victory_write_guarded_fault) as u64
                };
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: a handler that reads the record it is given and nothing else.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn install() {}

/// Copies `len` bytes from `from` to `to` in place; false if a page of the
/// source was not there (then the prefix before the faulting page has been
/// written into `to`).
///
/// `to` must be writable memory of `len` bytes that is ours (host memory);
/// `from` may be anything.
#[cfg(target_arch = "x86_64")]
fn guarded(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    if avx2_on() {
        // SAFETY: CPU supports AVX2 (verified by [`avx2_on`]); `to` is the
        // caller's writable host buffer and any unmapped/inaccessible source
        // page ends the copy with 0 via [`install`].
        unsafe { victory_copy_avx2(to, from, len) != 0 }
    } else {
        // SAFETY: the destination is the caller's to write; a source page that
        // is not there ends the copy with 0 instead of a fault ([`install`]).
        unsafe { victory_copy_guarded(to, from, len) != 0 }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

/// Copies `len` bytes from host memory `from` into title memory `to` in place;
/// false if a destination page is not mapped or not writable.
#[cfg(target_arch = "x86_64")]
fn guarded_write(to: *mut u8, from: *const u8, len: usize) -> bool {
    install();
    if avx2_on() {
        // SAFETY: CPU supports AVX2 (verified by [`avx2_on`]); `from` is our
        // readable host buffer and any unmapped or read-only destination page
        // ends the write with 0 via [`install`].
        unsafe { victory_write_avx2(to, from, len) != 0 }
    } else {
        // SAFETY: `from` is our readable host buffer; any unmapped or read-only
        // destination page ends the write with 0 via [`install`].
        unsafe { victory_write_guarded(to, from, len) != 0 }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn guarded_write(_: *mut u8, _: *const u8, _: usize) -> bool {
    false
}

fn write_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var_os("VICTORY_WRITE_PLAIN").is_none_or(|v| v != "0"))
}

/// Writes made in place, their bytes, and the writes that met an unmapped or
/// read-only destination page (falling back to the kernel's write).
static WRITE_COUNT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// With [`on`] (and `VICTORY_WRITE_PLAIN!=0`): writes host memory at `from`
/// into title memory at `to` in place; false if disabled or a destination page
/// is not writable (caller falls back to `NtWriteVirtualMemory`).
pub(crate) fn write(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !write_on() {
        return false;
    }
    let whole = guarded_write(to, from, len);
    if whole {
        WRITE_COUNT[0].fetch_add(1, Relaxed);
        WRITE_COUNT[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITE_COUNT[2].fetch_add(1, Relaxed);
    }
    whole
}

/// With [`on`]: copies title memory at `from` into host memory at `to`;
/// false if it is off or a source page was not there (the caller then
/// copies through the kernel, as before).
pub(crate) fn copy(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() {
        return false;
    }
    if len < least() {
        COUNT[4].fetch_add(1, Relaxed);
        return false;
    }
    if faulted_lately(from as usize, len) {
        COUNT[3].fetch_add(1, Relaxed);
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        COUNT[0].fetch_add(1, Relaxed);
        COUNT[1].fetch_add(len as u64, Relaxed);
    } else {
        COUNT[2].fetch_add(1, Relaxed);
        note_fault(from as usize, len);
    }
    whole
}

/// Compares title memory `[title, title + len)` directly against mapped host
/// memory `[held, held + len)` in place using 256-bit AVX2 without allocating
/// or writing a scratch buffer.
///
/// Returns `Some(len)` when all bytes are equal, `Some(offset)` (`offset < len`)
/// at the exact first differing byte, or `None` if disabled, `VICTORY_BACK_READS`
/// is active, the source faulted recently, or a page of `title` is not mapped.
#[cfg(target_arch = "x86_64")]
pub(crate) fn first_diff(title: *const u8, held: *const u8, len: usize) -> Option<usize> {
    if len == 0 {
        return Some(0);
    }
    if !on() || !avx2_on() || crate::backreads::on() {
        return None;
    }
    if len < least() || faulted_lately(title as usize, len) {
        return None;
    }
    install();
    // SAFETY: CPU supports AVX2 (verified by `avx2_on`); `held` is mapped host
    // memory of `len` bytes, and any read fault on `title` redirects to
    // `victory_diff_avx2_fault` returning `usize::MAX`.
    let res = unsafe { victory_diff_avx2(title, held, len) };
    if res == usize::MAX {
        COUNT[2].fetch_add(1, Relaxed);
        note_fault(title as usize, len);
        None
    } else {
        Some(res)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn first_diff(_: *const u8, _: *const u8, _: usize) -> Option<usize> {
    None
}

/// With [`on`]: copies title memory at `from` into host memory at `to` in
/// place without checking [`RECENT`] or [`least`]. Intended for regions the
/// caller has already confirmed are committed (e.g. `VirtualQuery` in
/// `host::read_sparse`) so committed islands in a sparse buffer whose outer
/// bulk copy faulted into [`RECENT`] still avoid `ReadProcessMemory`. If a
/// concurrent unmap races before the copy finishes, our VEH handler catches
/// the fault cleanly and returns `false` so the caller can fall back.
pub(crate) fn copy_committed(to: *mut u8, from: *const u8, len: usize) -> bool {
    if !on() {
        return false;
    }
    let whole = guarded(to, from, len);
    if whole {
        COUNT[0].fetch_add(1, Relaxed);
        COUNT[1].fetch_add(len as u64, Relaxed);
    } else {
        COUNT[2].fetch_add(1, Relaxed);
        note_fault(from as usize, len);
    }
    whole
}

/// With the frame line: what [`copy`] did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let min = least();
    let mode = if avx2_on() { ", avx2" } else { "" };
    let [copies, bytes, gone, lately, small] = [0, 1, 2, 3, 4].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    let [writes, wbytes, wgone] = [0, 1, 2].map(|i| WRITE_COUNT[i].swap(0, Relaxed) as f64 / frames);
    tracing::info!(
        "copies of title memory made in place per frame (VICTORY_COPY_PLAIN=1{mode}): {copies:.0}, {:.1} MB; {gone:.1} met a page \
         that was not there and went through the kernel's copy; left to the kernel's copy untried: {lately:.0} whose source \
         faulted in the last two seconds, {small:.0} of under {min} bytes; writes back in place: {writes:.0}, {:.1} MB, \
         {wgone:.1} fell back",
        bytes / 1e6,
        wbytes / 1e6
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PAGE: usize = 4096;

    #[test]
    fn memory_that_is_there_is_copied() {
        let from: Vec<u8> = (0..100_000u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut to = vec![0u8; from.len()];
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), from.len()));
        assert_eq!(to, from);
        // Nothing to copy is a copy.
        assert!(guarded(to.as_mut_ptr(), from.as_ptr(), 0));
    }

    /// Three pages reserved, the first committed: a copy that stays in the
    /// first page is whole; one that runs into the second ends with false,
    /// the bytes before the hole copied, and the thread goes on.
    #[test]
    fn a_page_that_is_not_there_ends_the_copy_without_a_fault() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xAB);
        let mut to = vec![0u8; 2 * PAGE];
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        to.fill(0);
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE / 2), PAGE));
        assert!(to[..PAGE / 2].iter().all(|&b| b == 0xAB), "the bytes before the hole");
        assert!(to[PAGE / 2..].iter().all(|&b| b == 0), "nothing after it");
        // A source that is not there at all.
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(PAGE), PAGE));
        // Committed but closed to reads.
        // SAFETY: the third page of the reservation.
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        assert!(!guarded(to.as_mut_ptr(), base.wrapping_add(2 * PAGE), 16));
        // And the registers the caller owns came back: this still works.
        assert!(guarded(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xAB));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// A source that faulted is remembered by its 64 KiB blocks, and only
    /// those.
    #[test]
    fn a_source_that_faulted_is_remembered_by_its_blocks() {
        // (Addresses no other test uses; the table is shared.)
        let at = 0x7654_3210_0000usize;
        assert!(!faulted_lately(at, 4096));
        note_fault(at + 100, 4096);
        assert!(faulted_lately(at, 16));
        assert!(faulted_lately(at + 0xffff, 1));
        // A copy that ends in the block, and one that begins in it.
        assert!(faulted_lately(at - 4096, 8192));
        // The blocks beside it are not remembered.
        assert!(!faulted_lately(at + 0x10000, 4096));
        assert!(!faulted_lately(at - 0x10000, 4096));
        assert_eq!(blocks(0, 0), (1 << 63, 1 << 63));
    }

    /// Many threads at once, half of them faulting: each gets its own answer.
    #[test]
    fn threads_fault_and_copy_side_by_side() {
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(base, 0);
        // SAFETY: the first page of that reservation.
        assert!(!unsafe { VirtualAlloc(base as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let threads: Vec<_> = (0..8)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut to = vec![0u8; PAGE];
                    for _ in 0..2_000 {
                        let from = (base + if t % 2 == 0 { 0 } else { PAGE }) as *const u8;
                        assert_eq!(guarded(to.as_mut_ptr(), from, PAGE), t % 2 == 0);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn a_readonly_or_unmapped_destination_ends_the_write_without_a_fault() {
        use windows_sys::Win32::System::Memory::PAGE_READONLY;
        // SAFETY: memory of the test's own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: page 0 writable, page 1 uncommitted, page 2 read-only.
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READONLY) }.is_null());
        let src = vec![0x5Au8; 2 * PAGE];
        assert!(guarded_write(base, src.as_ptr(), PAGE));
        // SAFETY: committed and readable.
        assert!(unsafe { std::slice::from_raw_parts(base, PAGE) }.iter().all(|&b| b == 0x5A));
        // Spanning into uncommitted page 1 fails cleanly.
        assert!(!guarded_write(base.wrapping_add(PAGE / 2), src.as_ptr(), PAGE));
        // Writing to read-only page 2 fails cleanly.
        assert!(!guarded_write(base.wrapping_add(2 * PAGE), src.as_ptr(), 16));
        // Caller-saved registers intact: valid write still succeeds.
        assert!(guarded_write(base, src.as_ptr(), PAGE));
        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn copy_plain_min_parsing_defaults_to_16_and_honors_overrides() {
        assert_eq!(parse_least(None), 16);
        assert_eq!(parse_least(Some("")), 16);
        assert_eq!(parse_least(Some("not-a-number")), 16);
        assert_eq!(parse_least(Some("32")), 32);
        assert_eq!(parse_least(Some("64")), 64);
        assert_eq!(parse_least(Some(" 256 ")), 256);
        assert_eq!(parse_least(Some("0")), 0);
    }

    #[test]
    fn avx2_env_parsing_defaults_on_unless_zero() {
        assert!(parse_avx2(None));
        assert!(parse_avx2(Some("")));
        assert!(parse_avx2(Some("1")));
        assert!(parse_avx2(Some("true")));
        assert!(!parse_avx2(Some("0")));
        assert!(!parse_avx2(Some(" 0 ")));
    }

    #[test]
    fn small_16_32_and_64_byte_guarded_copies_and_writes_work_and_handle_faults() {
        use windows_sys::Win32::System::Memory::PAGE_READONLY;
        for size in [16usize, 32usize, 64usize] {
            let src: Vec<u8> = (0..size).map(|i| (i * 13 + 5) as u8).collect();
            let mut dst = vec![0u8; size];
            assert!(guarded(dst.as_mut_ptr(), src.as_ptr(), size), "copy {size}B");
            assert_eq!(dst, src);

            // SAFETY: memory of the test's own (page 0 RW, page 1 uncommitted, page 2 RO).
            let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
            assert!(!base.is_null());
            assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
            assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_READONLY) }.is_null());

            // Valid 16B/32B/64B write into page 0 and read back.
            assert!(guarded_write(base, src.as_ptr(), size), "write {size}B");
            dst.fill(0);
            assert!(guarded(dst.as_mut_ptr(), base, size), "read back {size}B");
            assert_eq!(dst, src);

            // Read spanning page 0 -> uncommitted page 1 fails cleanly with partial prefix copied.
            // SAFETY: page 0 is committed and writable.
            unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xAB);
            dst.fill(0);
            assert!(!guarded(dst.as_mut_ptr(), base.wrapping_add(PAGE - size / 2), size));
            assert!(dst[..size / 2].iter().all(|&b| b == 0xAB));
            assert!(dst[size / 2..].iter().all(|&b| b == 0));

            // Read from uncommitted page 1 fails cleanly.
            assert!(!guarded(dst.as_mut_ptr(), base.wrapping_add(PAGE), size));

            // Write spanning page 0 -> uncommitted page 1 fails cleanly.
            assert!(!guarded_write(base.wrapping_add(PAGE - size / 2), src.as_ptr(), size));

            // Write to read-only page 2 fails cleanly.
            assert!(!guarded_write(base.wrapping_add(2 * PAGE), src.as_ptr(), size));

            // Valid 16B/32B/64B copy/write still work after faults.
            assert!(guarded_write(base, src.as_ptr(), size));
            assert!(guarded(dst.as_mut_ptr(), base, size));
            assert_eq!(dst, src);

            // SAFETY: the test's reservation.
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn avx2_try_ranges_are_well_formed_and_disjoint() {
        let c_start = (&raw const victory_copy_avx2_try_start) as usize;
        let c_end = (&raw const victory_copy_avx2_try_end) as usize;
        let c_fault = (&raw const victory_copy_avx2_fault) as usize;
        let w_start = (&raw const victory_write_avx2_try_start) as usize;
        let w_end = (&raw const victory_write_avx2_try_end) as usize;
        let w_fault = (&raw const victory_write_avx2_fault) as usize;

        assert!(c_start < c_end, "copy try range must be non-empty");
        assert!(w_start < w_end, "write try range must be non-empty");
        assert!(c_fault < c_start || c_fault >= c_end, "copy fault label outside try range");
        assert!(w_fault < w_start || w_fault >= w_end, "write fault label outside try range");
        assert!(c_end <= w_start || w_end <= c_start, "copy and write try ranges must be disjoint");
    }

    #[test]
    fn avx2_copy_and_write_exact_across_sizes_and_unaligned_offsets() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        install();

        // 1. Exhaustive unaligned (src_off, dst_off) in 0..32 across all loop boundaries.
        const GUARD: usize = 64;
        const MATRIX_SIZES: &[usize] = &[
            0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 127, 128, 129, 159, 160, 255,
            256, 257, 513,
        ];
        let max_small = *MATRIX_SIZES.last().unwrap() + 32 + 2 * GUARD;
        let src_buf: Vec<u8> = (0..max_small).map(|i| ((i * 29 + 11) & 0xFF) as u8).collect();
        let mut dst_buf = vec![0xA5u8; max_small];

        for &len in MATRIX_SIZES {
            for src_off in 0..32usize {
                for dst_off in 0..32usize {
                    let s = GUARD + src_off;
                    let d = GUARD + dst_off;
                    // Test `victory_copy_avx2`.
                    dst_buf[d - 8..d + len + 8].fill(0xA5);
                    // SAFETY: `s..s+len` and `d..d+len` are in bounds of distinct allocations; AVX2 checked.
                    let ok = unsafe { victory_copy_avx2(dst_buf.as_mut_ptr().add(d), src_buf.as_ptr().add(s), len) };
                    assert_eq!(ok, 1, "copy_avx2 failed len={len} src_off={src_off} dst_off={dst_off}");
                    assert_eq!(&dst_buf[d..d + len], &src_buf[s..s + len]);
                    assert!(dst_buf[d - 8..d].iter().all(|&b| b == 0xA5), "copy underrun");
                    assert!(dst_buf[d + len..d + len + 8].iter().all(|&b| b == 0xA5), "copy overrun");

                    // Test `victory_write_avx2`.
                    dst_buf[d - 8..d + len + 8].fill(0x5A);
                    // SAFETY: distinct valid allocations; AVX2 checked.
                    let ok = unsafe { victory_write_avx2(dst_buf.as_mut_ptr().add(d), src_buf.as_ptr().add(s), len) };
                    assert_eq!(ok, 1, "write_avx2 failed len={len} src_off={src_off} dst_off={dst_off}");
                    assert_eq!(&dst_buf[d..d + len], &src_buf[s..s + len]);
                    assert!(dst_buf[d - 8..d].iter().all(|&b| b == 0x5A), "write underrun");
                    assert!(dst_buf[d + len..d + len + 8].iter().all(|&b| b == 0x5A), "write overrun");
                }
            }
        }

        // 2. Every length 0..=260 plus key sizes up to 131_072 with rotating unaligned offsets 0..31.
        const MAX_LEN: usize = 131_072;
        let big_src: Vec<u8> = (0..MAX_LEN + 64 + 2 * GUARD).map(|i| ((i * 53 + 7) & 0xFF) as u8).collect();
        let mut big_dst = vec![0xCCu8; MAX_LEN + 64 + 2 * GUARD];
        let mut sizes: Vec<usize> = (0..=260).collect();
        sizes.extend_from_slice(&[
            300, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 16_384, 32_767,
            32_768, 32_769, 65_535, 65_536, 65_537, 131_071, 131_072,
        ]);
        for (idx, &len) in sizes.iter().enumerate() {
            let src_off = idx % 32;
            let dst_off = (idx * 7 + 3) % 32;
            let s = GUARD + src_off;
            let d = GUARD + dst_off;

            big_dst[d - 16..d + len + 16].fill(0xCC);
            // SAFETY: distinct valid buffers; AVX2 checked.
            let ok = unsafe { victory_copy_avx2(big_dst.as_mut_ptr().add(d), big_src.as_ptr().add(s), len) };
            assert_eq!(ok, 1, "big copy_avx2 len={len}");
            assert_eq!(&big_dst[d..d + len], &big_src[s..s + len], "big copy mismatch len={len}");
            assert!(big_dst[d - 16..d].iter().all(|&b| b == 0xCC));
            assert!(big_dst[d + len..d + len + 16].iter().all(|&b| b == 0xCC));

            big_dst[d - 16..d + len + 16].fill(0x33);
            // SAFETY: distinct valid buffers; AVX2 checked.
            let ok = unsafe { victory_write_avx2(big_dst.as_mut_ptr().add(d), big_src.as_ptr().add(s), len) };
            assert_eq!(ok, 1, "big write_avx2 len={len}");
            assert_eq!(&big_dst[d..d + len], &big_src[s..s + len], "big write mismatch len={len}");
            assert!(big_dst[d - 16..d].iter().all(|&b| b == 0x33));
            assert!(big_dst[d + len..d + len + 16].iter().all(|&b| b == 0x33));
        }

        // 3. Also verify legacy `rep movsb` routines still match cleanly.
        let mut rep_dst = vec![0u8; 4096];
        // SAFETY: valid buffers.
        assert_eq!(unsafe { victory_copy_guarded(rep_dst.as_mut_ptr(), big_src.as_ptr(), 4096) }, 1);
        assert_eq!(&rep_dst[..], &big_src[..4096]);
        rep_dst.fill(0);
        // SAFETY: valid buffers.
        assert_eq!(unsafe { victory_write_guarded(rep_dst.as_mut_ptr(), big_src.as_ptr(), 4096) }, 1);
        assert_eq!(&rep_dst[..], &big_src[..4096]);
    }

    #[test]
    fn avx2_fault_recovery_all_loops_and_page_protections() {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY, VirtualProtect};
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        install();

        // Pairs of (valid_prefix_bytes_before_fault_page, total_len) covering every loop stage:
        // - immediate fault (prefix=0) in 1B, 8B, 32B, 128B loops
        // - 128B unrolled AVX2 loop at lanes +0, +32, +64, +96 (1st & 2nd iter, aligned & unaligned)
        // - 32B AVX2 loop (direct & after 128B loop, aligned & unaligned)
        // - 8B loop (small non-AVX path, after align32 prologue, and AVX tail)
        // - 1B tail (small non-AVX path, align8 prologue, and AVX tail)
        const CASES: &[(usize, usize)] = &[
            (0, 1),
            (0, 7),
            (0, 8),
            (0, 16),
            (0, 31),
            (0, 32),
            (0, 64),
            (0, 128),
            (0, 512),
            // 1B / 8B small non-AVX paths
            (1, 5),
            (3, 7),
            (5, 12),
            (8, 16),
            (11, 24),
            (16, 31),
            // Align prologue into small non-AVX tail (< 32B remaining)
            (16, 32),
            (24, 32),
            (31, 35),
            (31, 40),
            // 32B AVX2 loop (direct & unaligned)
            (32, 64),
            (64, 96),
            (37, 90),
            // 128B AVX2 loop: all 4 lanes (+0, +32, +64, +96) in 1st & 2nd iterations
            (32, 256),
            (64, 256),
            (96, 256),
            (128, 256),
            (128 + 32, 384),
            (128 + 64, 384),
            (128 + 96, 384),
            // Unaligned entry into 128B AVX2 loop
            (131, 300),
            (199, 400),
            // AVX2 32B / 8B / 1B tails after 128B / 32B loops
            (32, 35),
            (32, 40),
            (128, 133),
            (128, 136),
            (128 + 8, 139),
            (128 + 8, 144),
            (128 + 32, 192),
            (128 + 32 + 8, 175),
        ];

        // Mode: 0 = uncommitted (reserved), 1 = committed PAGE_NOACCESS, 2 = PAGE_GUARD, 3 = PAGE_READONLY (write only).
        for mode in 0..=3u32 {
            // SAFETY: private 2-page reservation; page 0 is RW, page 1 has the fault mode under test.
            let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
            assert!(!base.is_null());
            assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
            let page1 = base.wrapping_add(PAGE);
            match mode {
                0 => {} // leave page 1 reserved/uncommitted
                1 => {
                    assert!(!unsafe { VirtualAlloc(page1.cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
                }
                2 => {
                    assert!(!unsafe { VirtualAlloc(page1.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
                }
                3 => {
                    assert!(!unsafe { VirtualAlloc(page1.cast(), PAGE, MEM_COMMIT, PAGE_READONLY) }.is_null());
                }
                _ => unreachable!(),
            }

            let pattern: Vec<u8> = (0..PAGE).map(|i| ((i * 31 + 17) & 0xFF) as u8).collect();
            const PAD: usize = 32;
            let mut host_buf = vec![0u8; 1024 + 2 * PAD];

            for &(prefix, len) in CASES {
                let page_off = PAGE - prefix;

                // 1. Test `victory_copy_avx2` (read fault) for modes 0 (unmapped), 1 (NOACCESS), 2 (GUARD).
                if mode != 3 {
                    if mode == 2 {
                        let mut old = 0u32;
                        // SAFETY: re-arm one-shot PAGE_GUARD on page 1 before each fault test.
                        assert_ne!(
                            unsafe { VirtualProtect(page1.cast(), PAGE, PAGE_READWRITE | PAGE_GUARD, &mut old) },
                            0
                        );
                    }
                    // SAFETY: page 0 is committed RW.
                    unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.copy_from_slice(&pattern);
                    host_buf.fill(0xFE);
                    let dst_off = PAD + (prefix * 5) % 31;
                    // SAFETY: `host_buf` is writable; fault at `page1` handled by VEH.
                    let ok = unsafe {
                        victory_copy_avx2(host_buf.as_mut_ptr().add(dst_off), base.add(page_off), len)
                    };
                    assert_eq!(ok, 0, "copy_avx2 should fault: mode={mode} prefix={prefix} len={len}");
                    assert_eq!(
                        &host_buf[dst_off..dst_off + prefix],
                        &pattern[page_off..PAGE],
                        "copy_avx2 prefix mismatch: mode={mode} prefix={prefix} len={len}"
                    );
                    assert!(
                        host_buf[dst_off + prefix..dst_off + len].iter().all(|&b| b == 0xFE),
                        "copy_avx2 wrote past fault boundary: mode={mode} prefix={prefix} len={len}"
                    );
                    assert!(host_buf[dst_off - 16..dst_off].iter().all(|&b| b == 0xFE));
                    assert!(host_buf[dst_off + len..dst_off + len + 16].iter().all(|&b| b == 0xFE));
                }

                // 2. Test `victory_write_avx2` (write fault) for all 4 modes (unmapped, NOACCESS, GUARD, READONLY).
                if mode == 2 {
                    let mut old = 0u32;
                    // SAFETY: re-arm one-shot PAGE_GUARD on page 1 before each fault test.
                    assert_ne!(
                        unsafe { VirtualProtect(page1.cast(), PAGE, PAGE_READWRITE | PAGE_GUARD, &mut old) },
                        0
                    );
                }
                // SAFETY: page 0 is committed RW.
                unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xEE);
                let src_off = PAD + (prefix * 3) % 31;
                for (i, b) in host_buf[src_off..src_off + len].iter_mut().enumerate() {
                    *b = ((i * 43 + 9) & 0xFF) as u8;
                }
                // SAFETY: `host_buf` is readable; fault at `page1` handled by VEH.
                let ok = unsafe {
                    victory_write_avx2(base.add(page_off), host_buf.as_ptr().add(src_off), len)
                };
                assert_eq!(ok, 0, "write_avx2 should fault: mode={mode} prefix={prefix} len={len}");
                // SAFETY: page 0 is committed RW.
                let page0 = unsafe { std::slice::from_raw_parts(base, PAGE) };
                assert_eq!(
                    &page0[page_off..PAGE],
                    &host_buf[src_off..src_off + prefix],
                    "write_avx2 prefix mismatch: mode={mode} prefix={prefix} len={len}"
                );
                assert!(
                    page0[..page_off].iter().all(|&b| b == 0xEE),
                    "write_avx2 corrupted bytes before destination: mode={mode} prefix={prefix} len={len}"
                );
            }

            // Verify normal copy & write still work cleanly after all faults (callee-saved registers intact).
            // SAFETY: page 0 is committed RW.
            unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.copy_from_slice(&pattern);
            let mut verify = vec![0u8; PAGE];
            // SAFETY: valid buffers.
            assert_eq!(unsafe { victory_copy_avx2(verify.as_mut_ptr(), base, PAGE) }, 1);
            assert_eq!(verify, pattern);
            let write_pat: Vec<u8> = (0..PAGE).map(|i| (i ^ 0xA5) as u8).collect();
            // SAFETY: valid buffers.
            assert_eq!(unsafe { victory_write_avx2(base, write_pat.as_ptr(), PAGE) }, 1);
            // SAFETY: page 0 is committed RW.
            assert_eq!(unsafe { std::slice::from_raw_parts(base, PAGE) }, &write_pat[..]);

            // SAFETY: private test reservation.
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn avx2_non_overlapping_and_throughput_sanity() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        install();

        // Non-overlapping regions within a single larger buffer: neighboring slices stay intact.
        let mut arena: Vec<u8> = (0..32_768usize).map(|i| ((i * 19 + 3) & 0xFF) as u8).collect();
        let expected_src = arena[4096..12_288].to_vec();
        let expected_before = arena[12_288..16_384].to_vec();
        let expected_after = arena[24_576..32_768].to_vec();
        // SAFETY: `[4096..12_288)` and `[16_384..24_576)` are non-overlapping in bounds.
        assert_eq!(
            unsafe { victory_copy_avx2(arena.as_mut_ptr().add(16_384), arena.as_ptr().add(4096), 8192) },
            1
        );
        assert_eq!(&arena[16_384..24_576], &expected_src[..]);
        assert_eq!(&arena[12_288..16_384], &expected_before[..]);
        assert_eq!(&arena[24_576..32_768], &expected_after[..]);

        // Multi-megabyte throughput sanity check (16 * 4 MiB = 64 MiB).
        const CHUNK: usize = 4 << 20;
        const ITERS: usize = 16;
        let src: Vec<u8> = (0..CHUNK).map(|i| ((i * 11 + 5) & 0xFF) as u8).collect();
        let mut dst = vec![0u8; CHUNK];
        // Warm up pages.
        // SAFETY: distinct 4 MiB buffers.
        assert_eq!(unsafe { victory_copy_avx2(dst.as_mut_ptr(), src.as_ptr(), CHUNK) }, 1);
        assert_eq!(dst, src);

        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            // SAFETY: distinct 4 MiB buffers.
            assert_eq!(unsafe { victory_copy_avx2(dst.as_mut_ptr(), src.as_ptr(), CHUNK) }, 1);
        }
        let elapsed = started.elapsed().as_secs_f64().max(1e-9);
        let gb_per_s = ((CHUNK * ITERS) as f64 / 1e9) / elapsed;
        assert_eq!(dst[0], src[0]);
        assert_eq!(dst[CHUNK - 1], src[CHUNK - 1]);
        assert!(gb_per_s > 0.1, "unexpectedly low AVX2 copy throughput: {gb_per_s:.2} GB/s");
    }

    #[test]
    fn copy_committed_bypasses_recent_and_handles_concurrent_unmapped_faults() {
        let _on = force_on_for_test();
        // SAFETY: memory of the test's own (page 0 committed, page 1 uncommitted).
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: page 0 is committed and writable.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0xCD);
        let mut to = vec![0u8; 2 * PAGE];

        // Bulk copy across page 0 + uncommitted page 1 faults and marks the 64 KiB block in RECENT.
        assert!(!copy(to.as_mut_ptr(), base, 2 * PAGE));
        assert!(faulted_lately(base as usize, PAGE));
        // Regular `copy` now skips page 0 because its 64 KiB block faulted lately.
        assert!(!copy(to.as_mut_ptr(), base, PAGE));

        // `copy_committed` bypasses RECENT and copies the committed page in place (even < least()).
        to.fill(0);
        let (copies_before, bytes_before, _) = copy_stats();
        assert!(copy_committed(to.as_mut_ptr(), base, PAGE));
        assert!(to[..PAGE].iter().all(|&b| b == 0xCD));
        assert!(copy_committed(to.as_mut_ptr(), base, 8));
        let (copies_after, bytes_after, _) = copy_stats();
        assert!(copies_after >= copies_before + 2);
        assert!(bytes_after >= bytes_before + PAGE as u64 + 8);

        // If a mapping races and turns out unmapped, `copy_committed` still returns false cleanly.
        assert!(!copy_committed(to.as_mut_ptr(), base.wrapping_add(PAGE), PAGE));

        // SAFETY: the test's reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn diff_avx2_finds_exact_offset_and_handles_faults() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let _on = force_on_for_test();
        install();
        // Exact match and exact mismatch offset across all AVX2/qword/byte tails.
        for len in [0usize, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 4096, 65536] {
            let a: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(37).wrapping_add(11)) as u8).collect();
            let mut b = a.clone();
            // SAFETY: valid equal buffers of `len` bytes.
            assert_eq!(unsafe { victory_diff_avx2(a.as_ptr(), b.as_ptr(), len) }, len, "equal len={len}");
            for &pos in &[0usize, 1, 7, 8, 31, 32, 63, 64, 95, 96, 127, 128, 129, len.saturating_sub(1)] {
                if pos >= len {
                    continue;
                }
                b[pos] ^= 0x80;
                // SAFETY: valid buffers of `len` bytes.
                assert_eq!(
                    unsafe { victory_diff_avx2(a.as_ptr(), b.as_ptr(), len) },
                    pos,
                    "len={len} pos={pos}"
                );
                b[pos] ^= 0x80;
            }
        }
        // Fault handling: page 0 committed RW, page 1 uncommitted, page 2 NOACCESS.
        // Clear `RECENT` in case Windows recycles a 64 KiB VA freed by an earlier test.
        for place in &RECENT {
            place.store(0, Relaxed);
        }
        // SAFETY: private test reservation.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as *mut u8;
        assert!(!base.is_null());
        assert!(!unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(!unsafe { VirtualAlloc(base.wrapping_add(2 * PAGE).cast(), PAGE, MEM_COMMIT, PAGE_NOACCESS) }.is_null());
        let held = vec![0x5Au8; 2 * PAGE];
        // SAFETY: page 0 is committed RW.
        unsafe { std::slice::from_raw_parts_mut(base, PAGE) }.fill(0x5A);
        assert_eq!(first_diff(base, held.as_ptr(), PAGE), Some(PAGE));
        // SAFETY: page 0 is committed RW.
        unsafe { *base.add(2049) = 0x5B };
        assert_eq!(first_diff(base, held.as_ptr(), PAGE), Some(2049));
        // Spanning into uncommitted page 1 faults cleanly and returns None.
        // SAFETY: page 0 is committed RW.
        unsafe { *base.add(2049) = 0x5A };
        assert_eq!(first_diff(base.wrapping_add(PAGE / 2), held.as_ptr(), PAGE), None);
        // NOACCESS page 2 returns usize::MAX directly from asm.
        // SAFETY: VEH catches the read fault cleanly.
        assert_eq!(unsafe { victory_diff_avx2(base.wrapping_add(2 * PAGE), held.as_ptr(), 64) }, usize::MAX);
        // SAFETY: private test reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }
}
