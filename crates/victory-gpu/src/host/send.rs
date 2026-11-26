//! What goes up to copies in the card's memory goes up as it was when its
//! submission was prepared (`VICTORY_SEND_OWN`, on by default; `0`: as
//! before).
//!
//! Each submission's dirty bytes are copied out of the CPU sides into upload
//! memory owned by the submission, then uploaded to card-local copies via three
//! tiers tuned for PCIe bandwidth and command-buffer density:
//!
//! 1. **1–2 dword tiny runs (`4..=8 B`)**: packed as `(offset, value)` word
//!    pairs (`8 B/dword`) and scattered by one `SCATTER` compute dispatch.
//! 2. **Dense small/medium runs (`12 B..=16 KiB`, 4-byte aligned,
//!    `VICTORY_CARD_SCATTER=1` default)**: packed as dense contiguous payload
//!    plus 12-byte `[dst_off, src_off, dwords]` job headers (`1.0x` PCIe
//!    bandwidth instead of `2.0x` pairs, eliminating `CopyBufferRegion` calls
//!    up to 16 KiB) and scattered by one `RUN_SCATTER` compute dispatch.
//! 3. **Large (`> 16 KiB`) or unaligned runs**: uploaded via `CopyBufferRegion`.

use super::*;
use crate::shadow::Shadow;

pub(super) fn own() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_OWN").is_none_or(|v| v != "0"))
}

/// Whether dense small/medium 4-byte-aligned dirty runs (`12 B..=16 KiB`) use
/// the packed `RUN_SCATTER` compute uploader (`VICTORY_CARD_SCATTER`, on by
/// default; `0` falls back to `<= 256 B` word pairs + `CopyBufferRegion`).
pub(super) fn card_scatter() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_card_scatter(std::env::var("VICTORY_CARD_SCATTER").ok().as_deref()))
}

fn parse_card_scatter(val: Option<&str>) -> bool {
    val.is_none_or(|v| v.trim() != "0")
}

/// Legacy word-pair ceiling when `VICTORY_CARD_SCATTER=0`.
const SHORT_LEGACY: usize = 256;
/// Word-pair ceiling when `VICTORY_CARD_SCATTER=1`: 1–2 dwords (`<= 8 B`) are
/// smaller as pairs (`8 B/16 B`) than a 12 B job header + dense payload.
const PAIR_MAX: usize = 8;
/// Dense `RUN_SCATTER` ceiling (`16 KiB`): eliminates `CopyBufferRegion` overhead
/// for all runs up to 16 KiB while streaming large ranges via DMA copies.
const SCATTER_MAX_RUN: usize = 16 << 10;
/// Maximum dwords copied per `RUN_SCATTER` thread job (`64 dwords = 256 B`) so
/// multi-KiB scatter runs fan out cleanly across NVIDIA Wave32 warps.
const SCATTER_MAX_JOB_DWORDS: u32 = 64;
/// Maximum jobs per single `Dispatch` (`65535` groups x `64` threads).
const SCATTER_MAX_DISPATCH_JOBS: u32 = 64 * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Pair,
    DenseScatter,
    Copy,
}

fn classify_with(at: usize, len: usize, dense_on: bool) -> Route {
    if len == 0 || !at.is_multiple_of(4) || !len.is_multiple_of(4) || at.saturating_add(len) > u32::MAX as usize {
        return Route::Copy;
    }
    if dense_on {
        if len <= PAIR_MAX {
            Route::Pair
        } else if len <= SCATTER_MAX_RUN {
            Route::DenseScatter
        } else {
            Route::Copy
        }
    } else if len <= SHORT_LEGACY {
        Route::Pair
    } else {
        Route::Copy
    }
}

fn classify(at: usize, len: usize) -> Route {
    classify_with(at, len, card_scatter())
}

/// Whether a run of `len` bytes at offset `at` of a copy in the card's memory
/// is handled by a compute scatter dispatch rather than its own copy command.
pub(crate) fn goes_as_words(at: usize, len: usize) -> bool {
    own() && !matches!(classify(at, len), Route::Copy)
}

/// Whether the title's bytes that go in under flights' writes
/// (`Shadow::merge_under_writes`) go in by whole words
/// (`VICTORY_UNDER_WORDS=0`: by the bytes that changed, as before).
pub(crate) fn words_under() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| own() && std::env::var_os("VICTORY_UNDER_WORDS").is_none_or(|v| v != "0"))
}

const SCATTER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Part : register(b0) { uint at; uint count; };
ByteAddressBuffer pairs : register(t0);
RWByteAddressBuffer card : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint2 p = pairs.Load2(at + id.x * 8u);
    card.Store(p.x, p.y);
}
"#;

/// Dense-payload compute scatter uploader (`CARD_SCATTER`): each thread reads
/// one 12-byte job `(dst_off, src_off, dwords)` (`1..=64` dwords) and copies
/// contiguous dwords with vectorized `Load4`/`Store4`.
const RUN_SCATTER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Meta : register(b0) {
    uint job_base;
    uint job_count;
};
ByteAddressBuffer jobs : register(t0);
ByteAddressBuffer payload : register(t1);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= job_count) return;
    uint j = (job_base + id.x) * 12u;
    uint3 job = jobs.Load3(j);
    uint d = job.x;
    uint s = job.y;
    uint n = job.z;
    uint i = 0u;
    [loop]
    for (; i + 4u <= n; i += 4u) {
        dst.Store4(d + i * 4u, payload.Load4(s + i * 4u));
    }
    [loop]
    for (; i < n; i += 1u) {
        dst.Store(d + i * 4u, payload.Load(s + i * 4u));
    }
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    run_root: ID3D12RootSignature,
    run_pso: ID3D12PipelineState,
}

fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile(SCATTER, windows::core::s!("main"), windows::core::s!("cs_5_1")).ok()?;
        let run_code =
            compile(RUN_SCATTER, windows::core::s!("main"), windows::core::s!("cs_5_1")).ok()?;
        // SAFETY: both DXBC blobs live through creation and embed their root signatures.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).ok()?;
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            name(&pso, || "victory: changed words sent to the card".into());

            let run_root: ID3D12RootSignature =
                host.device.CreateRootSignature(0, &run_code).ok()?;
            let run_pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&run_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: run_code.as_ptr().cast(),
                        BytecodeLength: run_code.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            name(&run_pso, || "victory: CARD_SCATTER dense runs sent to the card".into());

            Some(Pass {
                root,
                pso,
                run_root,
                run_pso,
            })
        }
    })
    .as_ref()
}

/// Memory a submission's sends come from, mapped.
pub(super) struct Memory {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

// SAFETY: a D3D12 object (free-threaded) and its mapping, with one owner.
unsafe impl Send for Memory {}

/// What submissions that have run gave back.
static SPARE: Mutex<Vec<Memory>> = Mutex::new(Vec::new());

/// At most this much is kept (of the card's memory, with [`on_card`]).
fn kept() -> u64 {
    if on_card() {
        256 << 20
    } else {
        match crate::residency::host_ram_tier() {
            crate::residency::HostRamTier::Tight => 256 << 20,
            crate::residency::HostRamTier::Moderate => 384 << 20,
            crate::residency::HostRamTier::Plentiful => 768 << 20,
        }
    }
}

fn on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_BAR").is_some_and(|v| v == "1"))
}

fn memory(host: &Host, size: u64) -> Option<Memory> {
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let fit = spare
            .iter()
            .enumerate()
            .filter(|(_, m)| m.size >= size)
            .min_by_key(|(_, m)| m.size)
            .map(|(i, _)| i);
        if let Some(i) = fit {
            return Some(spare.swap_remove(i));
        }
    }
    let size = size.max(8 << 20).next_power_of_two();
    let mapped = if on_card() {
        host.mapped_card_buffer_private(size, false)
    } else {
        None
    };
    let (buffer, data) = match mapped {
        Some(made) => made,
        // Write-combined `UPLOAD` heap avoids L0 `WRITE_BACK` cache-snooping
        // stalls on GPU reads; `take` streams payloads and batched job headers
        // contiguously so WC fill buffers flush cleanly.
        None => host.upload_buffer_private(size)?,
    };
    name(&buffer, || "bytes a submission sends up".into());
    Some(Memory {
        buffer,
        data: data as usize,
        size,
    })
}

/// The submission has run.
pub(super) fn give_back(memory: Memory) {
    let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    spare.push(memory);
    spare.sort_by_key(|m| std::cmp::Reverse(m.size));
    while spare.iter().map(|m| m.size).sum::<u64>() > kept() {
        spare.remove(0);
    }
}

/// Since the last frame-rate line:
/// `[0]` CopyBufferRegion bytes, `[1]` CopyBufferRegion runs,
/// `[2]` word-pair scatter words, `[3]` microseconds taking out of CPU sides,
/// `[4]` dense `CARD_SCATTER` runs, `[5]` dense `CARD_SCATTER` bytes.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 6] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// One copy's dense-payload `RUN_SCATTER` section inside `Memory`.
struct DenseScatter {
    jobs_at: u64,
    payload_at: u64,
    jobs: u32,
}

/// One copy's sends: `CopyBufferRegion` ranges, word-pair scatter section, and
/// dense-payload `RUN_SCATTER` section.
struct Copy {
    runs: Vec<(usize, u64, usize)>,
    pairs: (u64, u32),
    scatter: DenseScatter,
}

impl Copy {
    fn has_compute(&self) -> bool {
        self.pairs.1 > 0 || self.scatter.jobs > 0
    }
}

/// Where a submission's bytes are taken to.
pub(super) struct Own {
    memory: Memory,
    copies: Vec<Copy>,
}

/// Runs whose pairs/scatter jobs are one parallel chunk of work.
const RUNS_AT_A_TIME: usize = 4096;

fn jobs_for_run(len_bytes: usize) -> u32 {
    let dwords = (len_bytes / 4) as u32;
    dwords.div_ceil(SCATTER_MAX_JOB_DWORDS)
}

/// Takes what `dirty` sends up out of the CPU sides.
pub(super) fn take(
    host: &Host,
    dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
) -> Option<Own> {
    let began = std::time::Instant::now();
    pass(host)?;
    let dense_on = card_scatter();
    let route = |&(a, b): &(usize, usize)| classify_with(a, b - a, dense_on);
    let mut at = 0u64;
    let mut copies = Vec::with_capacity(dirty.len());
    let mut dense_runs_total = 0u64;
    let mut dense_bytes_total = 0u64;

    for (_, ranges) in dirty {
        let mut runs = Vec::new();
        let mut pair_words = 0usize;
        let mut scatter_jobs = 0u64;
        let mut scatter_bytes = 0u64;
        for &r in ranges {
            let len = r.1 - r.0;
            match route(&r) {
                Route::Copy => {
                    runs.push((r.0, at, len));
                    at += (len as u64).next_multiple_of(16);
                }
                Route::Pair => {
                    pair_words += len / 4;
                }
                Route::DenseScatter => {
                    let j = u64::from(jobs_for_run(len));
                    // Guard against `u32` payload offset overflow (4 GiB in a single shadow).
                    if scatter_bytes.saturating_add(len as u64) <= u64::from(u32::MAX - 256)
                        && scatter_jobs.saturating_add(j) <= u64::from(u32::MAX)
                    {
                        scatter_jobs += j;
                        scatter_bytes += len as u64;
                        dense_runs_total += 1;
                        dense_bytes_total += len as u64;
                    } else {
                        runs.push((r.0, at, len));
                        at += (len as u64).next_multiple_of(16);
                    }
                }
            }
        }
        let pairs_at = at;
        if pair_words > 0 {
            at += (pair_words as u64 * 8).next_multiple_of(16);
        }
        let jobs_at = if scatter_jobs > 0 {
            at = at.next_multiple_of(256);
            let p = at;
            at += (scatter_jobs * 12).next_multiple_of(256);
            p
        } else {
            0
        };
        let payload_at = if scatter_jobs > 0 {
            let p = at;
            at += scatter_bytes.next_multiple_of(256);
            p
        } else {
            0
        };
        copies.push(Copy {
            runs,
            pairs: (pairs_at, pair_words as u32),
            scatter: DenseScatter {
                jobs_at,
                payload_at,
                jobs: scatter_jobs as u32,
            },
        });
    }
    if at == 0 {
        return None;
    }
    let memory = memory(host, at)?;
    let mut moves: Vec<(usize, usize, usize)> = Vec::new();
    let mut pair_work: Vec<(usize, &[(usize, usize)], usize)> = Vec::new();
    let mut scatter_work: Vec<(usize, &[(usize, usize)], usize, usize, u32, u32)> = Vec::new();

    for ((shadow, ranges), copy) in dirty.iter().zip(&copies) {
        let from = shadow.card_data() as usize;
        moves.extend(
            copy.runs
                .iter()
                .map(|&(a, to, n)| (from + a, memory.data + to as usize, n)),
        );
        if copy.pairs.1 > 0 {
            let mut to = memory.data + copy.pairs.0 as usize;
            for chunk in ranges.chunks(RUNS_AT_A_TIME) {
                let words: usize = chunk
                    .iter()
                    .filter(|r| route(r) == Route::Pair)
                    .map(|r| (r.1 - r.0) / 4)
                    .sum();
                if words > 0 {
                    pair_work.push((from, chunk, to));
                    to += words * 8;
                }
            }
        }
        if copy.scatter.jobs > 0 {
            let mut job_ptr = memory.data + copy.scatter.jobs_at as usize;
            let mut pay_ptr = memory.data + copy.scatter.payload_at as usize;
            let mut pay_rel = 0u32;
            let mut jobs_left = copy.scatter.jobs;
            for chunk in ranges.chunks(RUNS_AT_A_TIME) {
                if jobs_left == 0 {
                    break;
                }
                let mut chunk_jobs = 0u32;
                let mut chunk_bytes = 0u32;
                for &r in chunk {
                    if route(&r) != Route::DenseScatter {
                        continue;
                    }
                    let len = (r.1 - r.0) as u32;
                    let j = jobs_for_run(len as usize);
                    if j > jobs_left.saturating_sub(chunk_jobs) {
                        break;
                    }
                    chunk_jobs += j;
                    chunk_bytes += len;
                }
                if chunk_jobs > 0 {
                    scatter_work.push((from, chunk, job_ptr, pay_ptr, pay_rel, chunk_jobs));
                    job_ptr += chunk_jobs as usize * 12;
                    pay_ptr += chunk_bytes as usize;
                    pay_rel += chunk_bytes;
                    jobs_left -= chunk_jobs;
                }
            }
        }
    }

    thread_local! {
        static SCATTER_JOBS: std::cell::RefCell<Vec<[u32; 3]>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    let pair_count = pair_work.len();
    let do_compute = |i: usize| {
        if i < pair_count {
            let (from, ranges, to) = pair_work[i];
            let mut to = to as *mut [u32; 2];
            for &(a, b) in ranges.iter().filter(|r| route(r) == Route::Pair) {
                for offset in (a..b).step_by(4) {
                    // SAFETY: inside the copy's CPU side and within `memory`'s
                    // uniquely reserved pair slice.
                    unsafe {
                        let word = ((from + offset) as *const u32).read_unaligned();
                        to.write([offset as u32, word]);
                        to = to.add(1);
                    }
                }
            }
        } else {
            let (from, ranges, job_ptr, mut pay_ptr, mut pay_rel, chunk_jobs) =
                scatter_work[i - pair_count];
            // Stage 12-byte `[dst_off, src_off, dwords]` job headers in cached
            // RAM while streaming payloads contiguously to `pay_ptr`, then
            // flush all job headers to WC `job_ptr` in one sequential copy.
            SCATTER_JOBS.with(|cell| {
                let mut jobs = cell.borrow_mut();
                jobs.clear();
                jobs.reserve(chunk_jobs as usize);
                for &(a, b) in ranges.iter().filter(|r| route(r) == Route::DenseScatter) {
                    let len = b - a;
                    let run_jobs = jobs_for_run(len) as usize;
                    if jobs.len() + run_jobs > chunk_jobs as usize {
                        break;
                    }
                    // SAFETY: disjoint reserved payload slice inside `memory` and
                    // live CPU-side shadow bytes `[from + a, from + b)`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (from + a) as *const u8,
                            pay_ptr as *mut u8,
                            len,
                        );
                    }
                    let mut rem_dwords = (len / 4) as u32;
                    let mut dst_off = a as u32;
                    let mut src_off = pay_rel;
                    while rem_dwords > 0 {
                        let take_dw = rem_dwords.min(SCATTER_MAX_JOB_DWORDS);
                        jobs.push([dst_off, src_off, take_dw]);
                        let step_bytes = take_dw * 4;
                        dst_off += step_bytes;
                        src_off += step_bytes;
                        rem_dwords -= take_dw;
                    }
                    pay_ptr += len;
                    pay_rel += len as u32;
                }
                debug_assert_eq!(jobs.len(), chunk_jobs as usize);
                if !jobs.is_empty() {
                    // SAFETY: disjoint reserved `chunk_jobs * 12`-byte job slice
                    // inside `memory`, written sequentially in one burst.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            jobs.as_ptr().cast::<u8>(),
                            job_ptr as *mut u8,
                            jobs.len() * size_of::<[u32; 3]>(),
                        );
                    }
                }
                if jobs.capacity() > 16384 {
                    jobs.clear();
                    jobs.shrink_to(4096);
                }
            });
        }
    };
    crate::land::each_under(pair_count + scatter_work.len(), &do_compute);

    STATS[0].fetch_add(moves.iter().map(|m| m.2 as u64).sum(), Ordering::Relaxed);
    STATS[1].fetch_add(moves.len() as u64, Ordering::Relaxed);
    STATS[2].fetch_add(
        copies.iter().map(|c| u64::from(c.pairs.1)).sum(),
        Ordering::Relaxed,
    );
    STATS[4].fetch_add(dense_runs_total, Ordering::Relaxed);
    STATS[5].fetch_add(dense_bytes_total, Ordering::Relaxed);
    crate::shadow::copy_all(moves);
    STATS[3].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    Some(Own { memory, copies })
}

impl Own {
    /// Records the sends into `list`; the copies that `looked` says are in
    /// `COPY_DEST`, the others in the common state. Those with a "before" are
    /// left readable by compute shaders.
    ///
    /// # Safety
    /// An open list of ours, whose compute state is set anew behind this.
    pub(super) unsafe fn record(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
        looked: impl Fn(&Shadow) -> bool,
    ) {
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let (mut first, mut last) = (Vec::new(), Vec::new());
        for ((shadow, _), copy) in dirty.iter().zip(&self.copies) {
            for &(a, from, n) in &copy.runs {
                // SAFETY: the caller's; live buffers, ranges inside them.
                unsafe {
                    list.CopyBufferRegion(
                        &shadow.buffer,
                        a as u64,
                        &self.memory.buffer,
                        from,
                        n as u64,
                    );
                }
            }
            let mut state = (looked(shadow) || !copy.runs.is_empty()).then_some(dest);
            if copy.has_compute() {
                if let Some(was) = state {
                    first.push(transition(&shadow.buffer, was, write));
                }
                state = Some(write);
            }
            let hashed = shadow
                .card_sides()
                .is_some_and(|(_, before, _)| before.is_some());
            if let Some(was) = state
                && hashed
                && (!copy.runs.is_empty() || copy.has_compute())
            {
                last.push(transition(&shadow.buffer, was, read));
            }
        }
        // SAFETY: the caller's; live objects.
        unsafe {
            if !first.is_empty() {
                list.ResourceBarrier(&first);
            }
            if let Some(pass) = pass(host) {
                let mem_va = self.memory.buffer.GetGPUVirtualAddress();
                if self.copies.iter().any(|c| c.pairs.1 > 0) {
                    list.SetComputeRootSignature(&pass.root);
                    list.SetPipelineState(&pass.pso);
                    list.SetComputeRootShaderResourceView(1, mem_va);
                    for ((shadow, _), copy) in
                        dirty.iter().zip(&self.copies).filter(|(_, c)| c.pairs.1 > 0)
                    {
                        list.SetComputeRootUnorderedAccessView(
                            2,
                            shadow.buffer.GetGPUVirtualAddress(),
                        );
                        const MOST: u32 = 64 * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
                        let (at, words) = copy.pairs;
                        let mut done = 0;
                        while done < words {
                            let n = (words - done).min(MOST);
                            let constants = [at as u32 + done * 8, n];
                            list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                            list.Dispatch(n.div_ceil(64), 1, 1);
                            done += n;
                        }
                    }
                }
                if self.copies.iter().any(|c| c.scatter.jobs > 0) {
                    // Notice that ranges per shadow are strictly disjoint, so
                    // pair writes and dense scatter writes never overlap bytes
                    // (no UAV barrier needed between the two dispatches).
                    list.SetComputeRootSignature(&pass.run_root);
                    list.SetPipelineState(&pass.run_pso);
                    for ((shadow, _), copy) in dirty
                        .iter()
                        .zip(&self.copies)
                        .filter(|(_, c)| c.scatter.jobs > 0)
                    {
                        list.SetComputeRootShaderResourceView(1, mem_va + copy.scatter.jobs_at);
                        list.SetComputeRootShaderResourceView(2, mem_va + copy.scatter.payload_at);
                        list.SetComputeRootUnorderedAccessView(
                            3,
                            shadow.buffer.GetGPUVirtualAddress(),
                        );
                        let mut done = 0u32;
                        while done < copy.scatter.jobs {
                            let n = (copy.scatter.jobs - done).min(SCATTER_MAX_DISPATCH_JOBS);
                            let constants = [done, n];
                            list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                            list.Dispatch(n.div_ceil(64), 1, 1);
                            done += n;
                        }
                    }
                }
            }
            if !last.is_empty() {
                list.ResourceBarrier(&last);
            }
        }
    }

    /// The memory, to be kept until the submission has run.
    pub(super) fn memory(self) -> Memory {
        self.memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_classification_matches_dense_and_legacy_modes() {
        assert!(parse_card_scatter(None));
        assert!(parse_card_scatter(Some("1")));
        assert!(!parse_card_scatter(Some("0")));
        assert!(!parse_card_scatter(Some(" 0 ")));

        // Dense mode: 4..=8 B -> Pair; 12..=16 KiB -> DenseScatter; >16 KiB or unaligned -> Copy.
        assert_eq!(classify_with(0, 4, true), Route::Pair);
        assert_eq!(classify_with(4, 8, true), Route::Pair);
        assert_eq!(classify_with(0, 12, true), Route::DenseScatter);
        assert_eq!(classify_with(64, 256, true), Route::DenseScatter);
        assert_eq!(classify_with(4096, 16 << 10, true), Route::DenseScatter);
        assert_eq!(classify_with(0, (16 << 10) + 4, true), Route::Copy);
        assert_eq!(classify_with(2, 16, true), Route::Copy);
        assert_eq!(classify_with(0, 15, true), Route::Copy);

        // Legacy mode: <= 256 B -> Pair; > 256 B -> Copy.
        assert_eq!(classify_with(0, 256, false), Route::Pair);
        assert_eq!(classify_with(0, 260, false), Route::Copy);
    }

    /// End-to-end GPU test verifying mixed word-pair (`4 B`, `8 B`), dense
    /// `CARD_SCATTER` (`12 B`, `64 B`, `256 B`, `4 KiB`, `16 KiB`), large
    /// `CopyBufferRegion` (`20 KiB`), and unaligned (`3 B`) runs land byte-exact
    /// on the RTX 5070 Ti without disturbing any sentinel neighbor byte.
    #[test]
    fn card_scatter_and_pairs_write_exact_runs_without_touching_neighbors() {
        let Some(host) = crate::host::host() else { return };
        let Some(p) = pass(host) else {
            panic!("send compute scatter pipelines failed to compile");
        };

        const BUF_LEN: usize = 96 << 10;
        const SENTINEL: u8 = 0x5A;
        let dst_buf = host.uav_buffer(BUF_LEN as u64).unwrap();
        let (init_up, init_ptr) = host.upload_buffer(BUF_LEN as u64).unwrap();
        // SAFETY: mapped upload buffer of `BUF_LEN` bytes.
        unsafe { std::ptr::write_bytes(init_ptr, SENTINEL, BUF_LEN) };

        let mut expected = vec![SENTINEL; BUF_LEN];
        let mut src = vec![0u8; BUF_LEN];
        for (i, b) in src.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37) ^ (i >> 3) ^ 0xA3) as u8).max(1);
            if *b == SENTINEL {
                *b = 0x5B;
            }
        }

        // Sorted disjoint ranges spanning every route:
        // Pair (4B, 8B), DenseScatter (12B, 64B, 256B, 4KiB, 16KiB), Copy (unaligned 3B, 20KiB).
        let ranges: Vec<(usize, usize)> = vec![
            (16, 20),                       // 4B Pair
            (40, 48),                       // 8B Pair
            (64, 76),                       // 12B DenseScatter (3 dwords, tail loop)
            (128, 192),                     // 64B DenseScatter (16 dwords, vectorized)
            (512, 768),                     // 256B DenseScatter (exact 1 job)
            (1025, 1028),                   // 3B unaligned Copy
            (4096, 8192),                   // 4KiB DenseScatter (16 jobs)
            (12288, 12288 + (16 << 10)),    // 16KiB DenseScatter (64 jobs)
            (40960, 40960 + (20 << 10)),    // 20KiB Copy
        ];
        for &(a, b) in &ranges {
            expected[a..b].copy_from_slice(&src[a..b]);
        }

        // Pack into one upload buffer using the exact layout `send::take` uses.
        let mut up_bytes = 0usize;
        let mut copy_runs = Vec::new();
        let mut pairs: Vec<[u32; 2]> = Vec::new();
        let mut jobs: Vec<[u32; 3]> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();
        for &(a, b) in &ranges {
            let len = b - a;
            match classify_with(a, len, true) {
                Route::Copy => {
                    copy_runs.push((a, up_bytes as u64, len));
                    up_bytes += len.next_multiple_of(16);
                }
                Route::Pair => {
                    for off in (a..b).step_by(4) {
                        let w = u32::from_le_bytes(src[off..off + 4].try_into().unwrap());
                        pairs.push([off as u32, w]);
                    }
                }
                Route::DenseScatter => {
                    let mut rem = (len / 4) as u32;
                    let mut dst_off = a as u32;
                    let mut src_off = payload.len() as u32;
                    payload.extend_from_slice(&src[a..b]);
                    while rem > 0 {
                        let take_dw = rem.min(SCATTER_MAX_JOB_DWORDS);
                        jobs.push([dst_off, src_off, take_dw]);
                        dst_off += take_dw * 4;
                        src_off += take_dw * 4;
                        rem -= take_dw;
                    }
                }
            }
        }
        let pairs_at = up_bytes as u64;
        up_bytes += (pairs.len() * 8).next_multiple_of(16);
        up_bytes = up_bytes.next_multiple_of(256);
        let jobs_at = up_bytes as u64;
        up_bytes += (jobs.len() * 12).next_multiple_of(256);
        let payload_at = up_bytes as u64;
        up_bytes += payload.len().next_multiple_of(256);

        let (up_buf, up_ptr) = host.upload_buffer_private(up_bytes as u64).unwrap();
        // SAFETY: `up_ptr` is mapped for `up_bytes`.
        unsafe {
            for &(a, to, n) in &copy_runs {
                std::ptr::copy_nonoverlapping(src[a..].as_ptr(), up_ptr.add(to as usize), n);
            }
            for (i, pair) in pairs.iter().enumerate() {
                up_ptr
                    .add(pairs_at as usize + i * 8)
                    .cast::<[u32; 2]>()
                    .write_unaligned(*pair);
            }
            for (i, job) in jobs.iter().enumerate() {
                up_ptr
                    .add(jobs_at as usize + i * 12)
                    .cast::<[u32; 3]>()
                    .write_unaligned(*job);
            }
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                up_ptr.add(payload_at as usize),
                payload.len(),
            );
        }

        let rb = host.read_back_buffer(BUF_LEN as u64).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: open list, live resources, valid barriers & dispatches.
        unsafe {
            list.list.CopyBufferRegion(&dst_buf, 0, &init_up, 0, BUF_LEN as u64);
            for &(a, from, n) in &copy_runs {
                list.list
                    .CopyBufferRegion(&dst_buf, a as u64, &up_buf, from, n as u64);
            }
            list.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            let up_va = up_buf.GetGPUVirtualAddress();
            let dst_va = dst_buf.GetGPUVirtualAddress();
            // Pairs dispatch
            list.list.SetComputeRootSignature(&p.root);
            list.list.SetPipelineState(&p.pso);
            list.list.SetComputeRootShaderResourceView(1, up_va);
            list.list.SetComputeRootUnorderedAccessView(2, dst_va);
            let c_pairs = [pairs_at as u32, pairs.len() as u32];
            list.list
                .SetComputeRoot32BitConstants(0, 2, c_pairs.as_ptr().cast(), 0);
            list.list.Dispatch((pairs.len() as u32).div_ceil(64), 1, 1);

            // Dense RUN_SCATTER dispatch
            list.list.SetComputeRootSignature(&p.run_root);
            list.list.SetPipelineState(&p.run_pso);
            list.list.SetComputeRootShaderResourceView(1, up_va + jobs_at);
            list.list
                .SetComputeRootShaderResourceView(2, up_va + payload_at);
            list.list.SetComputeRootUnorderedAccessView(3, dst_va);
            let c_jobs = [0u32, jobs.len() as u32];
            list.list
                .SetComputeRoot32BitConstants(0, 2, c_jobs.as_ptr().cast(), 0);
            list.list.Dispatch((jobs.len() as u32).div_ceil(64), 1, 1);

            list.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&rb, 0, &dst_buf, 0, BUF_LEN as u64);
        }
        list.close();
        host.execute(&[&list]);
        if let Some(err) = host.log_messages() {
            panic!("D3D12 debug layer error during CARD_SCATTER test: {err}");
        }
        let mut out_ptr = std::ptr::null_mut();
        // SAFETY: readback buffer finished on GPU.
        let got = unsafe {
            rb.Map(0, None, Some(&mut out_ptr)).unwrap();
            let s = std::slice::from_raw_parts(out_ptr.cast::<u8>(), BUF_LEN).to_vec();
            rb.Unmap(0, None);
            s
        };
        assert_eq!(got, expected, "CARD_SCATTER / pair / CopyBufferRegion output mismatch");
    }
}
