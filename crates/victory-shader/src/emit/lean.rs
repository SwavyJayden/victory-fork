//! Cheaper exact lowerings and wave-uniform / class-range helpers, extracted
//! from `lower.rs` to keep file sizes small.
//!
//! - `VICTORY_CLASS_RANGES=1`: `v_cmp_class_f32/f16` with a constant class
//!   mask tests the ranges of the magnitude's bits those classes cover
//!   (zero, denormals, normals, infinity, signalling and quiet NaNs lie in
//!   that order), with the sign where the mask asks for one: `±inf` is two
//!   instructions instead of some twenty (the general form works out the
//!   class number and shifts the mask by it).
//! - `VICTORY_WAVE_UNIFORM`: more values a console wave64 program in two host
//!   waves of 32 finds alike in every lane ([`lane_uniform`]), so that a
//!   `v_readfirstlane` of them takes the host wave's own first lane instead
//!   of the lower half's through LDS (two group barriers each). Bit 1: an
//!   operand is judged by its own registers only (not by the old value of
//!   the destination, which the instruction reads for lanes a guard masks).
//!   Bit 2: buffer loads at an address alike in every lane through a view
//!   the title declares read-only (SRV or CBV: nothing in the dispatch
//!   writes it, so both halves read the same bytes).

use std::collections::HashSet;

use super::bitcode::{Attr, Bin, Cast, Module, V, pred};
use super::shader::{Shader, flags};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::resource::Site;
use crate::structure::Structured;

/// Whether class tests with a constant mask test ranges
/// (`VICTORY_CLASS_RANGES=1`).
pub(super) fn class_ranges_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_CLASS_RANGES.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLASS_RANGES").is_some_and(|v| v == "1"))
}

/// `VICTORY_WAVE_UNIFORM` (default 0): bitmask (`& 3`) enabling cheaper
/// wave-uniformity detection in [`lane_uniform`].
pub(super) fn wave_uniform() -> u8 {
    #[cfg(test)]
    if let Some(bits) = TEST_WAVE_UNIFORM.get() {
        return bits & 3;
    }
    static BITS: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| {
        std::env::var("VICTORY_WAVE_UNIFORM")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(0)
            & 3
    })
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`class_ranges_on`], for its thread only.
    pub(super) static TEST_CLASS_RANGES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// A test's choice for [`wave_uniform`], for its thread only.
    pub(super) static TEST_WAVE_UNIFORM: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// A quad permutation as one quad operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QuadRead {
    /// Each lane reads itself.
    Same,
    /// Every lane reads quad lane n (`QuadReadLaneAt`).
    Lane(u32),
    /// `QuadOp`: 0 across x (lane ^ 1), 1 across y (lane ^ 2), 2 the
    /// diagonal (lane ^ 3), quad lanes numbered as D3D numbers them.
    Across(u8),
}

/// `quad_perm` `ctrl` (0..=0xFF: lane i of a quad reads lane `ctrl >> 2i &
/// 3`) as one quad operation, if it is one.
pub(super) fn quad_read(ctrl: u16) -> Option<QuadRead> {
    if ctrl > 0xFF {
        return None;
    }
    let p = [0, 1, 2, 3].map(|i| u32::from(ctrl >> (2 * i) & 3));
    Some(match p {
        [0, 1, 2, 3] => QuadRead::Same,
        [n, b, c, d] if n == b && n == c && n == d => QuadRead::Lane(n),
        [1, 0, 3, 2] => QuadRead::Across(0),
        [2, 3, 0, 1] => QuadRead::Across(1),
        [3, 2, 1, 0] => QuadRead::Across(2),
        _ => return None,
    })
}

/// `x` (an i32) as quad read `q` reads it, in block `c`.
pub(super) fn quad_op(s: &mut Shader, c: usize, x: V, q: QuadRead) -> V {
    let i32t = s.t.i32;
    match q {
        QuadRead::Same => x,
        QuadRead::Lane(n) => {
            s.flags |= flags::WAVE_OPS;
            let n = s.m.i32(n);
            s.op(c, "quadReadLaneAt.i32", Attr::NoUnwind, i32t, 122, &[x, n])
        }
        QuadRead::Across(k) => {
            s.flags |= flags::WAVE_OPS;
            let k = s.m.i8(k);
            s.op(c, "quadOp.i32", Attr::NoUnwind, i32t, 123, &[x, k])
        }
    }
}

/// `x` of lane `src` (in this lane's quad) through quad reads, whose
/// lane must be a constant.
pub(super) fn quad_pick(s: &mut Shader, c: usize, x: V, src: V, pattern: Option<u8>) -> V {
    s.flags |= flags::WAVE_OPS;
    if let Some(q) = pattern.and_then(|p| quad_read(u16::from(p))) {
        return quad_op(s, c, x, q);
    }
    let three = s.m.i32(3);
    let q = s.m.bin(c, Bin::And, src, three);
    let mut r = x;
    for n in 0..4 {
        let kn = s.m.i32(n);
        let read = s.op(
            c,
            "quadReadLaneAt.i32",
            Attr::NoUnwind,
            s.t.i32,
            122,
            &[x, kn],
        );
        let is = s.m.cmp(c, pred::EQ, q, kn);
        r = s.m.select(c, is, read, r);
    }
    r
}

/// Whether `max_min` can skip the signed-zero fix under `VICTORY_LEAN_MATH`.
pub(super) fn lean_max_min_skip(m: &Module, a: V, b: V, lean: u8) -> bool {
    if lean < 1 {
        return false;
    }
    // Only two zeros need the fix: none with a constant other than ±0
    // (NaN and infinity included), and none for a value and itself,
    // whose host maximum is one of two equal operands. Level 2: the
    // host's FMax and FMin order the zeros themselves (measured).
    let nonzero = |v: V| m.float_bits(v).is_some_and(|k| k & 0x7FFF_FFFF != 0);
    a == b || nonzero(a) || nonzero(b) || lean >= 2
}

/// `0 × anything = 0` under `VICTORY_LEAN_MATH >= 1`, with fewer tests:
/// operands known not to be ±0 need no test, and `x * x` / `-x * x` / `|x| * x`
/// need one test.
pub(super) fn lean_legacy_mul(m: &mut Module, c: usize, a: V, b: V, prod: V) -> V {
    let zero = m.f32(0);
    let zero_bits = |v: V| m.float_bits(v).map(|k| k & 0x7FFF_FFFF == 0);
    let (ka, kb) = (zero_bits(a), zero_bits(b));
    if ka == Some(true) || kb == Some(true) {
        return zero;
    }
    let (ra, rb) = (m.zero_root(a), m.zero_root(b));
    let tested = match (ka, kb) {
        (Some(false), Some(false)) => return prod,
        (Some(false), None) => Some(rb),
        (None, Some(false)) => Some(ra),
        _ if ra == rb => Some(ra),
        _ => None,
    };
    if let Some(x) = tested {
        let z = m.cmp(c, pred::OEQ, x, zero);
        return m.select(c, z, zero, prod);
    }
    let az = m.cmp(c, pred::OEQ, ra, zero);
    let bz = m.cmp(c, pred::OEQ, rb, zero);
    let z = m.bin(c, Bin::Or, az, bz);
    m.select(c, z, zero, prod)
}

/// `count` dwords from byte `addr` of raw view `h`, as a scalar buffer
/// load reads them: one host load for each four (DXIL's raw loads take
/// up to four dwords at a 4-byte alignment), where one per dword was.
/// The address is the same in every lane.
pub(super) fn raw_dwords(s: &mut Shader, b: usize, h: V, addr: V, count: u32) -> Vec<V> {
    let mut out = Vec::with_capacity(count as usize);
    let mut k = 0;
    while k < count {
        let n = (count - k).min(4);
        let a = if k == 0 {
            addr
        } else {
            let off = s.m.i32(4 * k);
            s.m.bin(b, Bin::Add, addr, off)
        };
        let mask = s.m.i8((1u8 << n) - 1);
        let (u, al) = (s.m.undef(s.t.i32), s.m.i32(4));
        let r = s.op(
            b,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            s.t.res_ret,
            139,
            &[h, a, u, mask, al],
        );
        for j in 0..n {
            let x = s.m.extract(b, r, j);
            out.push(s.m.cast(b, Cast::Bitcast, x, s.t.i32));
        }
        k += n;
    }
    out
}

/// Cube-map face coordinates (AMD's ISA): the major axis is `z` when
/// `|z| >= |x|` and `|z| >= |y|`, else `y` when `|y| >= |x|`, else `x`.
/// Under `VICTORY_LEAN_MATH >= 1`, uses one sign test on the picked major
/// coordinate (shared across the four instructions of a direction).
pub(super) fn cube_face(
    s: &mut Shader,
    c: usize,
    name: &str,
    x: V,
    y: V,
    z: V,
    lean: u8,
) -> V {
    let f32t = s.t.f32;
    let abs = |s: &mut Shader, v: V| s.op(c, "unary.f32", Attr::ReadNone, f32t, 6, &[v]);
    let (ax, ay, az) = (abs(s, x), abs(s, y), abs(s, z));
    let zx = s.m.cmp(c, pred::OGE, az, ax);
    let zy = s.m.cmp(c, pred::OGE, az, ay);
    let zmaj = s.m.bin(c, Bin::And, zx, zy);
    let ymaj = s.m.cmp(c, pred::OGE, ay, ax);
    let zero = s.m.f32(0);
    let neg = |m: &mut Module, v: V| m.cmp(c, pred::OLT, v, zero);
    let minus = |m: &mut Module, v: V| {
        let sign = m.f32(0x8000_0000);
        m.bin(c, Bin::Sub, sign, v)
    };
    let pick = |m: &mut Module, zv: V, yv: V, xv: V| {
        let t = m.select(c, ymaj, yv, xv);
        m.select(c, zmaj, zv, t)
    };
    let m = &mut s.m;
    if lean >= 1 {
        let major = pick(m, z, y, x);
        return match name {
            "v_cubeid_f32" => {
                let neg = neg(m, major);
                let (f4, f2, f1, f0) = (
                    m.f32(4.0f32.to_bits()),
                    m.f32(2.0f32.to_bits()),
                    m.f32(1.0f32.to_bits()),
                    zero,
                );
                let t = m.select(c, ymaj, f2, f0);
                let base = m.select(c, zmaj, f4, t);
                let odd = m.select(c, neg, f1, f0);
                m.bin(c, Bin::Add, base, odd)
            }
            "v_cubesc_f32" => {
                let neg = neg(m, major);
                let (mx, mz) = (minus(m, x), minus(m, z));
                let zf = m.select(c, neg, mx, x);
                let xf = m.select(c, neg, z, mz);
                pick(m, zf, x, xf)
            }
            "v_cubetc_f32" => {
                let neg = neg(m, major);
                let (my, mz) = (minus(m, y), minus(m, z));
                let yf = m.select(c, neg, mz, z);
                pick(m, my, yf, my)
            }
            _ => {
                let two = m.f32(2.0f32.to_bits());
                m.bin(c, Bin::Mul, major, two)
            }
        };
    }
    match name {
        "v_cubeid_f32" => {
            let (nz, ny, nx) = (neg(m, z), neg(m, y), neg(m, x));
            let (f5, f4, f3, f2, f1, f0) = (
                m.f32(5.0f32.to_bits()),
                m.f32(4.0f32.to_bits()),
                m.f32(3.0f32.to_bits()),
                m.f32(2.0f32.to_bits()),
                m.f32(1.0f32.to_bits()),
                zero,
            );
            let zf = m.select(c, nz, f5, f4);
            let yf = m.select(c, ny, f3, f2);
            let xf = m.select(c, nx, f1, f0);
            pick(m, zf, yf, xf)
        }
        "v_cubesc_f32" => {
            let (nz, nx) = (neg(m, z), neg(m, x));
            let (mx, mz) = (minus(m, x), minus(m, z));
            let zf = m.select(c, nz, mx, x);
            let xf = m.select(c, nx, z, mz);
            pick(m, zf, x, xf)
        }
        "v_cubetc_f32" => {
            let ny = neg(m, y);
            let (my, mz) = (minus(m, y), minus(m, z));
            let yf = m.select(c, ny, mz, z);
            pick(m, my, yf, my)
        }
        _ => {
            let t = pick(m, z, y, x);
            let two = m.f32(2.0f32.to_bits());
            m.bin(c, Bin::Mul, t, two)
        }
    }
}

/// Whether no instruction reads the high 16 bits of `v`: each reader takes it
/// as an f16 low half (`v_cvt_f32_f16` without `op_sel`, a `v_fma_mix*`
/// operand with `op_sel_hi` and without `op_sel`), or keeps it as the old
/// low half of `v_fma_mixhi_f16`, or as the old high half of `v_fma_mixlo_f16`
/// whose own high half no one reads (`depth` levels of those); a join (phi) or
/// any other reader counts as a read.
pub(super) fn high_half_unread(p: &Program, st: &Structured, v: Value, depth: u32) -> bool {
    if p.blocks
        .iter()
        .any(|b| b.phis.iter().any(|ph| ph.args.contains(&v)))
    {
        return false;
    }
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if !inst.reads.iter().any(|r| r.1 == v) {
                continue;
            }
            let Fields::Valu(f) = &inst.fields else {
                return false;
            };
            if f.sdwa.is_some() || f.dpp.is_some() {
                return false;
            }
            let name = inst.name();
            let holds = |op: &Operand| {
                matches!(*op, Operand::Vector { reg, .. } if inst.read(Reg::V(reg)) == Some(v))
            };
            for (k, op) in f.src.iter().enumerate() {
                if !holds(op) {
                    continue;
                }
                let low = match name {
                    "v_cvt_f32_f16" => f.opsel & 1 == 0,
                    "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                        f.opsel_hi >> k & 1 != 0 && f.opsel >> k & 1 == 0
                    }
                    _ => false,
                };
                if !low {
                    return false;
                }
            }
            if f.dst.as_ref().is_some_and(holds) {
                let Some(Operand::Vector { reg, .. }) = f.dst else {
                    return false;
                };
                let ok = match name {
                    "v_fma_mixhi_f16" => true,
                    "v_fma_mixlo_f16" => {
                        depth > 0
                            && inst
                                .write(Reg::V(reg))
                                .is_some_and(|w| high_half_unread(p, st, w, depth - 1))
                    }
                    "v_cvt_f32_f16" | "v_fma_mix_f32" => st.guards[b][i].is_none(),
                    _ => false,
                };
                if !ok {
                    return false;
                }
            }
        }
    }
    true
}

/// VGPR values that are the same in every lane holding them (`all` is
/// `all_lane_blocks`): written by an instruction every arriving lane runs (no
/// guard), lane by lane, from scalars, constants and such values; or loaded
/// from an address made of them. An LDS read (`ds_read*`) at an address held
/// alike in every lane counts too when no LDS write of the program can run
/// while it reads ([`lds_unraced`]). With [`wave_uniform`] bit 2, a read-only
/// buffer load counts too when `read_only` confirms its view is SRV/CBV and
/// its address is held alike (or is the offset alone).
pub(super) fn lane_uniform(
    p: &Program,
    st: &Structured,
    all: &[bool],
    waits: &dyn Fn(usize) -> bool,
    read_only: &dyn Fn(Site) -> bool,
) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let own_registers = wave_uniform() & 1 != 0;
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block
            .preds
            .iter()
            .all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            if let Fields::Ds {
                addr: Some(Operand::Vector { reg, .. }),
                gds: false,
                ..
            } = inst.fields
                && name.starts_with("ds_read")
                && st.guards[b][i].is_none()
                && inst.read(Reg::V(reg)).is_some_and(|a| same.contains(&a))
                && lds_unraced(p, (b, i), waits)
            {
                same.extend(
                    inst.writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .map(|w| w.1),
                );
                continue;
            }
            if let Fields::Buffer {
                addr: Operand::Vector { reg, dwords },
                offen,
                idxen,
                lds: false,
                tfe: false,
                ..
            } = inst.fields
                && name.contains("_load_")
                && st.guards[b][i].is_none()
                && read_only((b, i))
            {
                // without `offen` and `idxen` the address register is not read
                let addressed = !(offen || idxen)
                    || (reg..reg.saturating_add(dwords))
                        .all(|r| inst.read(Reg::V(r)).is_some_and(|a| same.contains(&a)));
                if addressed {
                    same.extend(
                        inst.writes
                            .iter()
                            .filter(|w| matches!(w.0, Reg::V(_)))
                            .map(|w| w.1),
                    );
                }
                continue;
            }
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                _ => continue,
            };
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            // `VICTORY_WAVE_UNIFORM` bit 1: an operand is its own registers;
            // of the other VGPRs the instruction reads, the old values of
            // those it writes (unguarded here) do not count.
            let own = own_registers
                && !(name.contains("mac") || name.contains("dot") || name.contains("movrel"));
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, dwords } => {
                    let end = if own {
                        reg.saturating_add(dwords.max(1))
                    } else {
                        u8::MAX
                    };
                    let read: Vec<Value> = inst
                        .reads
                        .iter()
                        .filter(|r| matches!(r.0, Reg::V(x) if x >= reg && (x < end || !own)))
                        .map(|r| r.1)
                        .collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => inst
                    .read(Reg::S(code))
                    .is_none_or(|x| !is_mask(x) || same.contains(&x)),
                _ => true,
            }) && (!own || {
                let in_operand = |x: u8| {
                    ops.iter().any(|op| {
                        matches!(*op, Operand::Vector { reg, dwords } if x >= reg && x < reg.saturating_add(dwords.max(1)))
                    })
                };
                let written = |x: u8| inst.writes.iter().any(|w| w.0 == Reg::V(x));
                inst.reads.iter().all(|r| match r.0 {
                    Reg::V(x) if !in_operand(x) => written(x) || same.contains(&r.1),
                    _ => true,
                })
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

/// Whether no LDS write can run while the read at `at` does, in any wave of
/// the group: none lies in the stretch between group barriers that holds the
/// read.
fn lds_unraced(p: &Program, at: (usize, usize), waits: &dyn Fn(usize) -> bool) -> bool {
    let writes = |inst: &Inst| match &inst.fields {
        Fields::Ds { .. } => {
            let n = inst.name();
            !(n.starts_with("ds_read")
                || matches!(
                    n,
                    "ds_swizzle_b32" | "ds_permute_b32" | "ds_bpermute_b32" | "ds_nop"
                ))
        }
        Fields::Buffer { lds, .. } | Fields::Flat { lds, .. } => *lds,
        _ => false,
    };
    let barrier = |b: usize, inst: &Inst| inst.name() == "s_barrier" && waits(b);
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut seen = vec![false; p.blocks.len()];
    let mut back = vec![at];
    while let Some((b, end)) = back.pop() {
        let insts = &p.blocks[b].insts;
        let mut started = false;
        for j in (0..end.min(insts.len())).rev() {
            if barrier(b, &insts[j]) {
                starts.push((b, j + 1));
                started = true;
                break;
            }
            if writes(&insts[j]) {
                return false;
            }
        }
        if started {
            continue;
        }
        if b == 0 || p.blocks[b].preds.is_empty() {
            starts.push((b, 0));
        }
        for &q in &p.blocks[b].preds {
            if !std::mem::replace(&mut seen[q], true) {
                back.push((q, p.blocks[q].insts.len()));
            }
        }
    }
    let mut seen = vec![false; p.blocks.len()];
    let mut ahead = starts;
    while let Some((b, from)) = ahead.pop() {
        let block = &p.blocks[b];
        let mut ended = false;
        for inst in block.insts.iter().skip(from) {
            if barrier(b, inst) {
                ended = true;
                break;
            }
            if writes(inst) {
                return false;
            }
        }
        if ended {
            continue;
        }
        if matches!(block.exit, crate::ir::Exit::Indirect) {
            return false;
        }
        for s in block.exit.successors() {
            if !std::mem::replace(&mut seen[s], true) {
                ahead.push((s, 0));
            }
        }
    }
    true
}

/// A run of the magnitude's bits `lo..=hi` and the sign it needs (`None`:
/// either).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Run {
    pub lo: u32,
    pub hi: u32,
    pub negative: Option<bool>,
}

/// The magnitudes of each class of a float, in their order: zero,
/// denormals, normals, infinity, signalling NaNs, quiet NaNs; with the mask
/// bits of the positive and the negative class.
fn kinds(f16: bool) -> [(u32, u32, u32, u32); 6] {
    let (inf, quiet, top) = if f16 {
        (0x7C00, 0x7E00, 0x7FFF)
    } else {
        (0x7F80_0000, 0x7FC0_0000, 0x7FFF_FFFF)
    };
    let first_normal = if f16 { 0x0400 } else { 0x0080_0000 };
    [
        (0, 0, 6, 5),
        (1, first_normal - 1, 7, 4),
        (first_normal, inf - 1, 8, 3),
        (inf, inf, 9, 2),
        (inf + 1, quiet - 1, 0, 0),
        (quiet, top, 1, 1),
    ]
}

/// The runs of magnitudes a class mask (bits 0-9: signalling NaN, quiet
/// NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf) takes,
/// adjacent classes with the same sign rule joined.
pub(super) fn class_runs(mask: u32, f16: bool) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (lo, hi, pos, neg) in kinds(f16) {
        let (p, n) = (mask >> pos & 1 != 0, mask >> neg & 1 != 0);
        let negative = match (p, n) {
            (false, false) => continue,
            (true, true) => None,
            (true, false) => Some(false),
            (false, true) => Some(true),
        };
        match runs.last_mut() {
            Some(r) if r.hi + 1 == lo && r.negative == negative => r.hi = hi,
            _ => runs.push(Run { lo, hi, negative }),
        }
    }
    runs
}

/// The class test of `x` (an f32's bits, or an f16's in the low half) for
/// the constant `mask`, as the general form decides it: the magnitude is
/// the bits below the sign, the sign is anything at or above it.
pub(super) fn class_test(m: &mut Module, c: usize, x: V, mask: u32, f16: bool) -> V {
    let runs = class_runs(mask, f16);
    let (top, sign_shift) = if f16 {
        (0x7FFF, 15)
    } else {
        (0x7FFF_FFFF, 31)
    };
    if runs.is_empty() {
        return m.bool(false);
    }
    let k = |m: &mut Module, v: u32| m.i32(v);
    let km = k(m, top);
    let abs = m.bin(c, Bin::And, x, km);
    let mut sign = None;
    let mut result: Option<V> = None;
    for run in runs {
        let range = match (run.lo, run.hi) {
            (0, hi) if hi == top => None,
            (lo, hi) if lo == hi => {
                let v = k(m, lo);
                Some(m.cmp(c, pred::EQ, abs, v))
            }
            (0, hi) => {
                let v = k(m, hi);
                Some(m.cmp(c, pred::IULE, abs, v))
            }
            (lo, hi) if hi == top => {
                let v = k(m, lo);
                Some(m.cmp(c, pred::IUGE, abs, v))
            }
            (lo, hi) => {
                let (l, n) = (k(m, lo), k(m, hi - lo));
                let off = m.bin(c, Bin::Sub, abs, l);
                Some(m.cmp(c, pred::IULE, off, n))
            }
        };
        let signed = run.negative.map(|negative| {
            let s = *sign.get_or_insert_with(|| {
                let sh = k(m, sign_shift);
                let s = m.bin(c, Bin::LShr, x, sh);
                let z = k(m, 0);
                m.cmp(c, pred::NE, s, z)
            });
            if negative {
                s
            } else {
                let t = m.bool(true);
                m.bin(c, Bin::Xor, s, t)
            }
        });
        let term = match (range, signed) {
            (Some(a), Some(b)) => m.bin(c, Bin::And, a, b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => m.bool(true),
        };
        result = Some(match result {
            Some(r) => m.bin(c, Bin::Or, r, term),
            None => term,
        });
    }
    result.unwrap_or_else(|| m.bool(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The general form's class number of `x` (`Lower::compare`).
    fn class_of(x: u32, f16: bool) -> u32 {
        let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
            (10, 15, 0x1F, 0x03FF, 0x0200)
        } else {
            (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
        };
        let e = x >> exp_sh & exp_mask;
        let m = x & mant_mask;
        let neg = x >> sign_sh != 0;
        let t = if m == 0 { 6 } else { 7 };
        let t = if e == 0 { t } else { 8 };
        let t = if e == exp_mask { 9 } else { t };
        let t = if neg { 11 - t } else { t };
        if m != 0 && e == exp_mask {
            u32::from(x & quiet_bit != 0)
        } else {
            t
        }
    }

    /// `x` against the runs, as [`class_test`] builds it.
    fn runs_say(runs: &[Run], x: u32, f16: bool) -> bool {
        let (top, sign_shift) = if f16 {
            (0x7FFF, 15)
        } else {
            (0x7FFF_FFFF, 31)
        };
        let abs = x & top;
        let neg = x >> sign_shift != 0;
        runs.iter()
            .any(|r| abs >= r.lo && abs <= r.hi && r.negative.is_none_or(|n| n == neg))
    }

    #[test]
    fn runs_agree_with_the_general_form_for_every_half_and_mask() {
        for mask in 0..1024u32 {
            let runs = class_runs(mask, true);
            for x in 0..=0xFFFFu32 {
                let want = mask >> class_of(x, true) & 1 != 0;
                assert_eq!(runs_say(&runs, x, true), want, "mask {mask:#x} x {x:#06x}");
            }
            // The f16 form reads bits above the half as a sign, as the
            // general one does.
            for x in [0x1_0000u32, 0x1_3C00, 0xFFFF_7C00] {
                let want = mask >> class_of(x, true) & 1 != 0;
                assert_eq!(runs_say(&runs, x, true), want, "mask {mask:#x} x {x:#x}");
            }
        }
    }

    #[test]
    fn runs_agree_with_the_general_form_for_floats() {
        // Every class edge and its neighbours, both signs.
        let mut xs = Vec::new();
        for edge in [
            0u32,
            1,
            0x7F_FFFF,
            0x80_0000,
            0x3F80_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7F80_0001,
            0x7FBF_FFFF,
            0x7FC0_0000,
            0x7FFF_FFFF,
        ] {
            for d in [0u32, 1, 2] {
                for x in [edge.wrapping_sub(d), edge.wrapping_add(d)] {
                    xs.push(x & 0x7FFF_FFFF);
                    xs.push(x | 0x8000_0000);
                }
            }
        }
        let mut seed = 0x9E37_79B9u32;
        for _ in 0..4096 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            xs.push(seed);
        }
        for mask in 0..1024u32 {
            let runs = class_runs(mask, false);
            for &x in &xs {
                let want = mask >> class_of(x, false) & 1 != 0;
                assert_eq!(
                    runs_say(&runs, x, false),
                    want,
                    "mask {mask:#x} x {x:#010x}"
                );
            }
        }
    }

    #[test]
    fn common_masks_are_one_run() {
        let one = |mask, f16| class_runs(mask, f16).len();
        // ±inf, NaN, ±0, ±denormal, finite, not NaN.
        for mask in [0x204, 0x003, 0x060, 0x090, 0x1F8, 0x3FC] {
            assert_eq!(one(mask, false), 1, "{mask:#x}");
        }
        assert_eq!(
            class_runs(0x204, false),
            [Run {
                lo: 0x7F80_0000,
                hi: 0x7F80_0000,
                negative: None
            }]
        );
        assert_eq!(
            class_runs(0x200, true),
            [Run {
                lo: 0x7C00,
                hi: 0x7C00,
                negative: Some(false)
            }]
        );
        assert!(class_runs(0, false).is_empty());
        assert_eq!(
            class_runs(0x3FF, false),
            [Run {
                lo: 0,
                hi: 0x7FFF_FFFF,
                negative: None
            }]
        );
    }

    #[test]
    fn quad_reads_read_the_lanes_the_permutation_names() {
        let mut found = 0;
        for ctrl in 0..=0xFFu16 {
            let Some(q) = quad_read(ctrl) else { continue };
            found += 1;
            for i in 0..4u32 {
                let want = u32::from(ctrl) >> (2 * i) & 3;
                let got = match q {
                    QuadRead::Same => i,
                    QuadRead::Lane(n) => n,
                    QuadRead::Across(k) => i ^ (u32::from(k) + 1),
                };
                assert_eq!(got, want, "ctrl {ctrl:#x} lane {i}");
            }
        }
        // The identity, four broadcasts and three swaps.
        assert_eq!(found, 8);
        assert_eq!(quad_read(0xB1), Some(QuadRead::Across(0)));
        assert_eq!(quad_read(0x4E), Some(QuadRead::Across(1)));
        assert_eq!(quad_read(0x1B), Some(QuadRead::Across(2)));
        assert_eq!(quad_read(0xE4), Some(QuadRead::Same));
        assert_eq!(quad_read(0xAA), Some(QuadRead::Lane(2)));
        assert_eq!(quad_read(0x101), None);
    }

    /// A pixel program reading a texture coordinate's neighbour in its quad
    /// (`v_mov_b32_dpp quad_perm`) through `ctrl`, validated by `dxil.dll` when present.
    #[cfg(windows)]
    fn pixel_dpp(ctrl: u16) -> Vec<u8> {
        use super::super::container::SigElement;
        use super::super::lower::{Parts, lower};
        use crate::native::{ExportPatch, HwStage, NativeProgram, PsInput, StageIo, Usage};
        use crate::{io, ir, resource, structure};
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0x7E06_02FA, // v_mov_b32_dpp v3, v2 quad_perm row_mask:0xf bank_mask:0xf
            0xFF00_0002 | u32::from(ctrl) << 8,
            0xF800_180F, // exp mrt0, v3, v3, v3, v3 done vm
            0x0303_0303,
            0xBF81_0000, // s_endpgm
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 1,
                    mask: 3,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 1,
                }],
                input_addr: 2,
                db_shader_control: 0,
                patches: Vec::<ExportPatch>::new(),
            },
            code,
        };
        let element = |name: &str, kind, interp, row, cols, usage| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage,
            half: false,
            packed: false,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let lowered = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![element("TEXCOORD", 0, 2, 1, 2, 1)],
            outputs: vec![element("SV_Target", 16, 0, 0, 4, 0xF)],
            prims: vec![],
            mesh_stride: 0,
        });
        let dxil = lowered.unwrap_or_else(|e| panic!("ctrl {ctrl:#x}: {e}")).dxil;
        let path = super::super::validator::Validator::default_path();
        if let Ok(mut v) = super::super::validator::Validator::load(path.as_deref()) {
            let mut copy = dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("ctrl {ctrl:#x}: {e}"));
        }
        dxil
    }

    #[cfg(windows)]
    #[test]
    fn pixel_quad_permutations_validate_and_shrink() {
        let general = pixel_dpp(0x93);
        for ctrl in [0xB1, 0x4E, 0x1B, 0x00, 0xFF, 0xE4] {
            let quad = pixel_dpp(ctrl);
            assert!(
                quad.len() < general.len(),
                "ctrl {ctrl:#x}: {} bytes, general {}",
                quad.len(),
                general.len()
            );
        }
    }
}
