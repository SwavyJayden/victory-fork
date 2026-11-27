//! Clean-room, non-destructive in-memory Scaleform SWF vector patcher that
//! replaces Xbox button prompt glyphs in `input_icons.gfx` with crisp
//! PlayStation 5 (DualSense) vector glyphs on the fly.
//!
//! Controlled via `VICTORY_PROMPTS`:
//! - `ps5` / `playstation` / `dualsense` / `1` -> force PlayStation prompts on
//! - `xbox` / `0` / `off` -> force Xbox prompts (patching disabled)
//! - unset / `auto` -> enabled automatically when a Sony HID gamepad is detected

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Byte offset of the `input_icons.gfx` zlib stream inside `data.rpf`.
pub const INPUT_ICONS_RPF_OFFSET: u64 = 0x117d_7000;
/// Exact compressed byte length of `input_icons.gfx` inside `data.rpf`.
pub const INPUT_ICONS_COMPRESSED_LEN: usize = 9_644;
/// Exact uncompressed byte length of `input_icons.gfx`.
pub const INPUT_ICONS_INFLATED_LEN: usize = 28_148;

/// Symbol marker present inside the target Scaleform `input_icons.gfx` asset.
const INPUT_ICONS_MARKER: &[u8] = b"platform_xbsx.input_icons.Main_Button_30";

/// SWF tag code for `End`.
const TAG_END: u16 = 0;
/// SWF tag code for `DefineShape3`.
const TAG_DEFINE_SHAPE3: u16 = 32;

/// Target `DefineShape3` character IDs for `Main_Button_30` .. `Main_Button_39`.
const ID_CROSS: u16 = 146; // Button_30 (A -> Cross)
const ID_CIRCLE: u16 = 150; // Button_31 (B -> Circle)
const ID_SQUARE: u16 = 154; // Button_32 (X -> Square)
const ID_TRIANGLE: u16 = 163; // Button_33 (Y -> Triangle)
const ID_L1: u16 = 76; // Button_34 (LB -> L1)
const ID_L2_NUM: u16 = 85; // Button_35 (T of LT -> 2, forming L2)
const ID_R1: u16 = 93; // Button_36 (RB -> R1)
const ID_R2_NUM: u16 = 107; // Button_37 (T of RT -> 2, forming R2)
const ID_OPTIONS: u16 = 120; // Button_38 (Menu -> Options)
const ID_TOUCHPAD: u16 = 130; // Button_39 (View -> Touchpad)

/// Configured prompt override from `VICTORY_PROMPTS`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromptMode {
    /// Enable PS5 glyphs automatically whenever a Sony HID gamepad is present.
    Auto,
    /// Always patch prompts to PS5 DualSense glyphs.
    Ps5,
    /// Never patch prompts (keep native Xbox glyphs).
    Xbox,
}

/// Parses a `VICTORY_PROMPTS` environment variable value.
pub fn parse_prompts_mode(value: Option<&str>) -> PromptMode {
    let Some(raw) = value else {
        return PromptMode::Auto;
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => PromptMode::Auto,
        "ps5" | "playstation" | "dualsense" | "1" | "on" | "true" => PromptMode::Ps5,
        "xbox" | "0" | "off" | "false" => PromptMode::Xbox,
        other => {
            tracing::warn!(
                value = other,
                "prompts: unrecognized VICTORY_PROMPTS value; falling back to auto"
            );
            PromptMode::Auto
        }
    }
}

/// Returns the cached `VICTORY_PROMPTS` mode from the process environment.
fn configured_mode() -> PromptMode {
    static MODE: OnceLock<PromptMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let env_val = std::env::var("VICTORY_PROMPTS").ok();
        parse_prompts_mode(env_val.as_deref())
    })
}

/// Returns `true` if `VICTORY_PROMPTS` explicitly disabled PS5 prompt patching
/// (`xbox` / `0` / `off` / `false`), allowing `files.rs` to skip all `data.rpf`
/// handle/stream tracking with zero overhead.
pub fn patching_disabled() -> bool {
    configured_mode() == PromptMode::Xbox
}

/// Returns `true` if PS5 DualSense button prompt patching is currently enabled.
pub fn ps5_enabled() -> bool {
    match configured_mode() {
        PromptMode::Ps5 => true,
        PromptMode::Xbox => false,
        PromptMode::Auto => super::gameinput_hid::has_sony_gamepad(),
    }
}

/// Returns `true` if `path` refers to `data.rpf` (case-insensitive filename match).
pub fn is_data_rpf_path(path: &str) -> bool {
    path.rsplit(['\\', '/'])
        .next()
        .is_some_and(|name| name.eq_ignore_ascii_case("data.rpf"))
}

/// Returns `true` if `wide` (UTF-16 path without trailing NUL) ends with `data.rpf`.
pub fn is_data_rpf_wide(wide: &[u16]) -> bool {
    let tail = wide
        .rsplit(|&c| c == u16::from(b'\\') || c == u16::from(b'/'))
        .next()
        .unwrap_or(wide);
    const TARGET: &[u8] = b"data.rpf";
    if tail.len() != TARGET.len() {
        return false;
    }
    tail.iter()
        .zip(TARGET)
        .all(|(&w, &t)| u8::try_from(w).is_ok_and(|b| b.to_ascii_lowercase() == t))
}

/// Patches an uncompressed Scaleform `GFX` (`input_icons.gfx`) buffer in place,
/// replacing the 10 Xbox button inner glyph `DefineShape3` records with clean-room
/// PlayStation 5 vector glyphs while keeping the exact buffer length and header
/// `fileLength` intact.
///
/// Returns `true` if the buffer matched `input_icons.gfx` and was patched cleanly.
pub fn patch_input_icons_gfx(buf: &mut [u8]) -> bool {
    if !is_input_icons_gfx(buf) {
        return false;
    }
    let Some(tags_start) = swf_tags_start(buf) else {
        return false;
    };

    let orig_len = buf.len();
    let mut out = Vec::with_capacity(orig_len);
    out.extend_from_slice(&buf[..tags_start]);

    let mut pos = tags_start;
    let mut replaced = 0usize;
    let mut saw_end = false;

    while pos + 2 <= orig_len {
        let tag_start = pos;
        let hdr = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let code = hdr >> 6;
        let mut len = usize::from(hdr & 0x3F);
        if len == 0x3F {
            if pos + 4 > orig_len {
                return false;
            }
            len = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            pos += 4;
        }
        let Some(payload_end) = pos.checked_add(len) else {
            return false;
        };
        if payload_end > orig_len {
            return false;
        }

        if code == TAG_DEFINE_SHAPE3 && len >= 2 {
            let cid = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
            if let Some(payload) = build_ps5_shape3_payload(cid) {
                write_swf_tag(&mut out, TAG_DEFINE_SHAPE3, &payload);
                replaced += 1;
                pos = payload_end;
                continue;
            }
        }

        out.extend_from_slice(&buf[tag_start..payload_end]);
        pos = payload_end;

        if code == TAG_END {
            saw_end = true;
            break;
        }
    }

    if !saw_end || replaced == 0 || out.len() > orig_len {
        return false;
    }

    // Zero-pad any trailing bytes past the SWF End tag and preserve original fileLength.
    out.resize(orig_len, 0);
    out[4..8].copy_from_slice(&(orig_len as u32).to_le_bytes());
    buf.copy_from_slice(&out);
    true
}

/// Inflates a zlib-compressed `input_icons.gfx` stream, patches its 10 glyphs to
/// PS5 vectors, and re-compresses it to fit within `max_len` bytes (zero-padded
/// to `max_len`).
///
/// Returns `None` if the input is not a valid `input_icons.gfx` zlib stream or
/// if the re-compressed output would exceed `max_len`.
pub fn patch_compressed_input_icons(compressed: &[u8], max_len: usize) -> Option<Vec<u8>> {
    if compressed.len() < 2 || max_len == 0 {
        return None;
    }
    if !looks_like_zlib_header(compressed) {
        return None;
    }

    // Inflate with a generous ceiling (exact real asset is 28,148 bytes).
    let mut inflated = inflate_zlib(compressed, INPUT_ICONS_INFLATED_LEN.max(65_536))?;
    if !patch_input_icons_gfx(&mut inflated) {
        return None;
    }

    // Re-compress at level 9. Smaller `mem_level` block sizes allow deflate to
    // emit separate Huffman trees for binary `DefineShape3` records vs. ASCII
    // symbol/ABC tables, matching or beating optimal-parse packers.
    let mut out = vec![0u8; zlib_rs::compress_bound(inflated.len())];
    let mut best: Option<Vec<u8>> = None;
    for mem_level in [5, 4, 6, 3, 7, 8] {
        let mut cfg = zlib_rs::DeflateConfig::new(9);
        cfg.mem_level = mem_level;
        let (packed, rc) = zlib_rs::compress_slice(&mut out, &inflated, cfg);
        if rc == zlib_rs::ReturnCode::Ok
            && packed.len() <= max_len
            && best.as_ref().is_none_or(|b| packed.len() < b.len())
        {
            best = Some(packed.to_vec());
            if mem_level == 5 {
                break;
            }
        }
    }
    let packed = best?;
    let mut result = vec![0u8; max_len];
    result[..packed.len()].copy_from_slice(&packed);
    Some(result)
}

/// Patches an arbitrary raw read slice from `data.rpf` if the byte range
/// `[file_offset, file_offset + buf.len())` contains the compressed
/// `input_icons.gfx` stream at [`INPUT_ICONS_RPF_OFFSET`].
///
/// Caches the re-compressed bytes once in a [`OnceLock`] so repeated reads
/// perform zero redundant zlib work. Returns `true` if `buf` was patched.
pub fn patch_data_rpf_compressed_region(file_offset: u64, buf: &mut [u8]) -> bool {
    patch_data_rpf_compressed_region_inner(file_offset, buf, ps5_enabled)
}

fn patch_data_rpf_compressed_region_inner(
    file_offset: u64,
    buf: &mut [u8],
    is_enabled: impl FnOnce() -> bool,
) -> bool {
    let Some(read_end) = file_offset.checked_add(buf.len() as u64) else {
        return false;
    };
    let target_end = INPUT_ICONS_RPF_OFFSET + INPUT_ICONS_COMPRESSED_LEN as u64;
    if file_offset > INPUT_ICONS_RPF_OFFSET || read_end < target_end {
        return false;
    }
    if !is_enabled() {
        return false;
    }

    let rel = (INPUT_ICONS_RPF_OFFSET - file_offset) as usize;
    let slice = &mut buf[rel..rel + INPUT_ICONS_COMPRESSED_LEN];
    if !looks_like_zlib_header(slice) {
        return false;
    }

    let hash = fnv1a_64(slice);
    static CACHED: OnceLock<(u64, Vec<u8>)> = OnceLock::new();
    if let Some((cached_hash, cached_bytes)) = CACHED.get()
        && *cached_hash == hash
        && cached_bytes.len() == slice.len()
    {
        slice.copy_from_slice(cached_bytes);
        return true;
    }

    let Some(patched) = patch_compressed_input_icons(slice, INPUT_ICONS_COMPRESSED_LEN) else {
        return false;
    };
    let _ = CACHED.set((hash, patched.clone()));
    slice.copy_from_slice(&patched);
    true
}

/// Logs once when the DirectStorage or Win32 file path patches `input_icons.gfx`.
pub fn log_patched_once(source: &str) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "prompts: patched input_icons.gfx to PS5 DualSense glyphs ({source})"
        );
    }
}

fn looks_like_zlib_header(bytes: &[u8]) -> bool {
    if bytes.len() < 2 {
        return false;
    }
    let cmf = bytes[0];
    let flg = bytes[1];
    (cmf & 0x0F) == 8 && (cmf >> 4) <= 7 && (u16::from(cmf) * 256 + u16::from(flg)).is_multiple_of(31)
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn inflate_zlib(compressed: &[u8], max_out: usize) -> Option<Vec<u8>> {
    let mut state = zlib_rs::Inflate::new(true, 15);
    let mut out = vec![0u8; max_out];
    let (mut in_pos, mut out_pos) = (0usize, 0usize);
    loop {
        match state.decompress(
            &compressed[in_pos..],
            &mut out[out_pos..],
            zlib_rs::InflateFlush::NoFlush,
        ) {
            Ok(zlib_rs::Status::StreamEnd) => {
                let total = state.total_out() as usize;
                out.truncate(total);
                return Some(out);
            }
            Ok(zlib_rs::Status::Ok) => {
                let next_in = state.total_in() as usize;
                let next_out = state.total_out() as usize;
                if next_in == in_pos && next_out == out_pos {
                    return None;
                }
                in_pos = next_in;
                out_pos = next_out;
            }
            _ => return None,
        }
    }
}

fn is_input_icons_gfx(buf: &[u8]) -> bool {
    buf.len() >= 22
        && buf.starts_with(b"GFX")
        && buf
            .windows(INPUT_ICONS_MARKER.len())
            .any(|w| w == INPUT_ICONS_MARKER)
}

fn swf_tags_start(buf: &[u8]) -> Option<usize> {
    if buf.len() < 13 {
        return None;
    }
    let nbits = usize::from(buf[8] >> 3);
    let rect_bytes = (5 + 4 * nbits).div_ceil(8);
    let tags_start = 8usize.checked_add(rect_bytes)?.checked_add(4)?;
    (tags_start <= buf.len()).then_some(tags_start)
}

fn write_swf_tag(out: &mut Vec<u8>, code: u16, payload: &[u8]) {
    if payload.len() < 0x3F {
        let hdr = (code << 6) | (payload.len() as u16);
        out.extend_from_slice(&hdr.to_le_bytes());
    } else {
        let hdr = (code << 6) | 0x3F;
        out.extend_from_slice(&hdr.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    }
    out.extend_from_slice(payload);
}

// ---------------------------------------------------------------------------
// Clean-room SWF DefineShape3 bit encoder
// ---------------------------------------------------------------------------

/// MSB-first SWF bitstream writer.
struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
            acc: 0,
            bits: 0,
        }
    }

    fn write_ub(&mut self, nbits: u8, val: u32) {
        if nbits == 0 {
            return;
        }
        debug_assert!(nbits <= 32);
        let mask = if nbits == 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << nbits) - 1
        };
        self.acc = (self.acc << nbits) | (u64::from(val) & mask);
        self.bits += u32::from(nbits);
        while self.bits >= 8 {
            self.bits -= 8;
            self.buf.push((self.acc >> self.bits) as u8);
        }
    }

    fn write_sb(&mut self, nbits: u8, val: i32) {
        self.write_ub(nbits, val as u32);
    }

    fn align(&mut self) {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.buf.push((self.acc << pad) as u8);
            self.acc = 0;
            self.bits = 0;
        }
    }

    fn write_bytes_aligned(&mut self, bytes: &[u8]) {
        self.align();
        self.buf.extend_from_slice(bytes);
    }

    fn write_rect(&mut self, xmin: i32, xmax: i32, ymin: i32, ymax: i32) {
        self.align();
        let nb = sb_bits(xmin)
            .max(sb_bits(xmax))
            .max(sb_bits(ymin))
            .max(sb_bits(ymax));
        self.write_ub(5, u32::from(nb));
        self.write_sb(nb, xmin);
        self.write_sb(nb, xmax);
        self.write_sb(nb, ymin);
        self.write_sb(nb, ymax);
        self.align();
    }

    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.buf
    }
}

/// Minimum two's-complement signed bit width required to represent `val`.
fn sb_bits(val: i32) -> u8 {
    let mag_bits = if val >= 0 {
        32 - val.leading_zeros()
    } else {
        32 - (!val).leading_zeros()
    };
    (mag_bits + 1) as u8
}

/// One vector segment in SWF twip space (20 twips = 1 px; Y increases downward).
#[derive(Clone, Copy, Debug)]
enum Segment {
    Line(i32, i32),
    Quad(i32, i32, i32, i32),
}

/// A closed contour in SWF twip space.
///
/// Convention matching Scaleform `input_icons.gfx` (`fillStyle0 = 1`, `fillStyle1 = 0`):
/// outer solid contours wind **clockwise (CW)** in Y-down coordinates and inner
/// holes wind **counter-clockwise (CCW)**.
struct Contour {
    start: (i32, i32),
    segments: Vec<Segment>,
}

impl Contour {
    fn from_polygon(pts: &[(i32, i32)]) -> Self {
        debug_assert!(pts.len() >= 3);
        let mut segments = Vec::with_capacity(pts.len());
        for &p in &pts[1..] {
            segments.push(Segment::Line(p.0, p.1));
        }
        segments.push(Segment::Line(pts[0].0, pts[0].1));
        Self {
            start: pts[0],
            segments,
        }
    }

    fn update_bounds(&self, bounds: &mut (i32, i32, i32, i32)) {
        let mut include = |x: i32, y: i32| {
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.max(x);
            bounds.2 = bounds.2.min(y);
            bounds.3 = bounds.3.max(y);
        };
        let (mut cx, mut cy) = self.start;
        include(cx, cy);
        for &seg in &self.segments {
            match seg {
                Segment::Line(x, y) => {
                    include(x, y);
                    cx = x;
                    cy = y;
                }
                Segment::Quad(qx, qy, x, y) => {
                    include(x, y);
                    // Include exact quadratic Bezier extrema along X and Y.
                    if let Some(ex) = quad_extremum(cx, qx, x) {
                        include(ex.floor() as i32, cy);
                        include(ex.ceil() as i32, cy);
                    }
                    if let Some(ey) = quad_extremum(cy, qy, y) {
                        include(cx, ey.floor() as i32);
                        include(cx, ey.ceil() as i32);
                    }
                    cx = x;
                    cy = y;
                }
            }
        }
    }
}

fn quad_extremum(p0: i32, p1: i32, p2: i32) -> Option<f64> {
    let denom = f64::from(p0 - 2 * p1 + p2);
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = f64::from(p0 - p1) / denom;
    if t > 0.0 && t < 1.0 {
        let mt = 1.0 - t;
        Some(mt * mt * f64::from(p0) + 2.0 * mt * t * f64::from(p1) + t * t * f64::from(p2))
    } else {
        None
    }
}

/// Encodes a clean-room `DefineShape3` payload for `shape_id` using solid opaque
/// black (`RGBA(0, 0, 0, 255)`) fill style 1 (`fillStyle0 = 1`) and no line styles.
fn encode_shape3(shape_id: u16, contours: &[Contour]) -> Vec<u8> {
    let mut bounds = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for c in contours {
        c.update_bounds(&mut bounds);
    }

    let mut w = BitWriter::new();
    // ShapeId (u16 LE)
    w.write_bytes_aligned(&shape_id.to_le_bytes());
    // ShapeBounds (RECT)
    w.write_rect(bounds.0, bounds.1, bounds.2, bounds.3);
    // FILLSTYLEARRAY: 1 solid fill RGBA(0, 0, 0, 255)
    // LINESTYLEARRAY: 0 line styles
    w.write_bytes_aligned(&[1, 0x00, 0x00, 0x00, 0x00, 0xFF, 0]);
    // NumFillBits = 1, NumLineBits = 0
    w.write_ub(4, 1);
    w.write_ub(4, 0);

    for (idx, contour) in contours.iter().enumerate() {
        let (sx, sy) = contour.start;
        // StyleChangeRecord: set MoveTo, and set FillStyle0 = 1 on first contour.
        w.write_ub(1, 0); // TypeFlag = 0 (non-edge)
        let set_fill0 = idx == 0;
        let flags = 0x01 | if set_fill0 { 0x02 } else { 0 };
        w.write_ub(5, flags);
        let mb = sb_bits(sx).max(sb_bits(sy));
        w.write_ub(5, u32::from(mb));
        w.write_sb(mb, sx);
        w.write_sb(mb, sy);
        if set_fill0 {
            w.write_ub(1, 1); // FillStyle0 = 1
        }

        let (mut cx, mut cy) = (sx, sy);
        for &seg in &contour.segments {
            match seg {
                Segment::Line(x, y) => {
                    let dx = x - cx;
                    let dy = y - cy;
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    w.write_ub(1, 1); // TypeFlag = 1 (edge)
                    w.write_ub(1, 1); // StraightFlag = 1
                    if dx == 0 {
                        let nb = sb_bits(dy).max(2);
                        w.write_ub(4, u32::from(nb - 2));
                        w.write_ub(1, 0); // GeneralLineFlag = 0
                        w.write_ub(1, 1); // VertLineFlag = 1 (vertical)
                        w.write_sb(nb, dy);
                    } else if dy == 0 {
                        let nb = sb_bits(dx).max(2);
                        w.write_ub(4, u32::from(nb - 2));
                        w.write_ub(1, 0); // GeneralLineFlag = 0
                        w.write_ub(1, 0); // VertLineFlag = 0 (horizontal)
                        w.write_sb(nb, dx);
                    } else {
                        let nb = sb_bits(dx).max(sb_bits(dy)).max(2);
                        w.write_ub(4, u32::from(nb - 2));
                        w.write_ub(1, 1); // GeneralLineFlag = 1
                        w.write_sb(nb, dx);
                        w.write_sb(nb, dy);
                    }
                    cx = x;
                    cy = y;
                }
                Segment::Quad(qx, qy, x, y) => {
                    let cdx = qx - cx;
                    let cdy = qy - cy;
                    let adx = x - qx;
                    let ady = y - qy;
                    let nb = sb_bits(cdx)
                        .max(sb_bits(cdy))
                        .max(sb_bits(adx))
                        .max(sb_bits(ady))
                        .max(2);
                    w.write_ub(1, 1); // TypeFlag = 1 (edge)
                    w.write_ub(1, 0); // StraightFlag = 0 (curved)
                    w.write_ub(4, u32::from(nb - 2));
                    w.write_sb(nb, cdx);
                    w.write_sb(nb, cdy);
                    w.write_sb(nb, adx);
                    w.write_sb(nb, ady);
                    cx = x;
                    cy = y;
                }
            }
        }
    }

    // EndShapeRecord (6 zero bits)
    w.write_ub(6, 0);
    w.finish()
}

// ---------------------------------------------------------------------------
// Clean-room PlayStation 5 (DualSense) vector glyph definitions
// ---------------------------------------------------------------------------

fn build_ps5_shape3_payload(shape_id: u16) -> Option<Vec<u8>> {
    let contours = match shape_id {
        ID_CROSS => glyph_cross(),
        ID_CIRCLE => glyph_circle(),
        ID_SQUARE => glyph_square(),
        ID_TRIANGLE => glyph_triangle(),
        ID_L1 => glyph_l1(),
        ID_L2_NUM => glyph_digit_2(696, 344),
        ID_R1 => glyph_r1(),
        ID_R2_NUM => glyph_digit_2(678, 344),
        ID_OPTIONS => glyph_options(),
        ID_TOUCHPAD => glyph_touchpad(),
        _ => return None,
    };
    Some(encode_shape3(shape_id, &contours))
}

/// 8-segment quadratic Bezier circle centered at `(cx, cy)` with radius `r`.
/// `cw = true` winds clockwise (solid outer) in Y-down space; `cw = false` winds CCW (hole).
fn circle_contour(cx: i32, cy: i32, r: i32, cw: bool) -> Contour {
    let rf = f64::from(r);
    // Tangent intersection radius for 45-degree quadratic Bezier arc: r / cos(pi/8).
    let r_ctrl = rf / (std::f64::consts::FRAC_PI_8).cos();
    let pt = |rad: f64, deg: f64| -> (i32, i32) {
        let a = deg.to_radians();
        (
            cx + (rad * a.cos()).round() as i32,
            cy + (rad * a.sin()).round() as i32,
        )
    };

    let start = pt(rf, 0.0);
    let mut segments = Vec::with_capacity(8);
    for i in 0..8 {
        let (mid_deg, end_deg) = if cw {
            (f64::from(i) * 45.0 + 22.5, f64::from(i + 1) * 45.0)
        } else {
            (
                f64::from(8 - i) * 45.0 - 22.5,
                f64::from(7 - i) * 45.0,
            )
        };
        let (qx, qy) = pt(r_ctrl, mid_deg);
        let (ex, ey) = if i == 7 { start } else { pt(rf, end_deg) };
        segments.push(Segment::Quad(qx, qy, ex, ey));
    }
    Contour { start, segments }
}

/// Rounded rectangle contour with corner radius `r`.
/// `cw = true` winds clockwise (solid) in Y-down space; `cw = false` winds CCW (hole).
fn rounded_rect_contour(x0: i32, y0: i32, x1: i32, y1: i32, r: i32, cw: bool) -> Contour {
    if cw {
        Contour {
            start: (x0 + r, y0),
            segments: vec![
                Segment::Line(x1 - r, y0),
                Segment::Quad(x1, y0, x1, y0 + r),
                Segment::Line(x1, y1 - r),
                Segment::Quad(x1, y1, x1 - r, y1),
                Segment::Line(x0 + r, y1),
                Segment::Quad(x0, y1, x0, y1 - r),
                Segment::Line(x0, y0 + r),
                Segment::Quad(x0, y0, x0 + r, y0),
            ],
        }
    } else {
        Contour {
            start: (x0 + r, y0),
            segments: vec![
                Segment::Quad(x0, y0, x0, y0 + r),
                Segment::Line(x0, y1 - r),
                Segment::Quad(x0, y1, x0 + r, y1),
                Segment::Line(x1 - r, y1),
                Segment::Quad(x1, y1, x1, y1 - r),
                Segment::Line(x1, y0 + r),
                Segment::Quad(x1, y0, x1 - r, y0),
                Segment::Line(x0 + r, y0),
            ],
        }
    }
}

/// Button_30 (146): DualSense Cross (`✕`), centered at `(640, 640)`.
fn glyph_cross() -> Vec<Contour> {
    let (cx, cy) = (640, 640);
    let r = 264;
    let c = 64; // perpendicular bar thickness = 64 * sqrt(2) ~= 90.5 twips
    vec![Contour::from_polygon(&[
        (cx - r + c, cy - r),
        (cx, cy - c),
        (cx + r - c, cy - r),
        (cx + r, cy - r + c),
        (cx + c, cy),
        (cx + r, cy + r - c),
        (cx + r - c, cy + r),
        (cx, cy + c),
        (cx - r + c, cy + r),
        (cx - r, cy + r - c),
        (cx - c, cy),
        (cx - r, cy - r + c),
    ])]
}

/// Button_31 (150): DualSense Circle (`○`), centered at `(640, 640)`.
fn glyph_circle() -> Vec<Contour> {
    vec![
        circle_contour(640, 640, 288, true),
        circle_contour(640, 640, 198, false),
    ]
}

/// Button_32 (154): DualSense Square (`□`), centered at `(640, 640)`.
fn glyph_square() -> Vec<Contour> {
    let (cx, cy) = (640, 640);
    let o = 256;
    let i = 166;
    vec![
        // Outer square (CW)
        Contour::from_polygon(&[
            (cx - o, cy - o),
            (cx + o, cy - o),
            (cx + o, cy + o),
            (cx - o, cy + o),
        ]),
        // Inner square hole (CCW)
        Contour::from_polygon(&[
            (cx - i, cy - i),
            (cx - i, cy + i),
            (cx + i, cy + i),
            (cx + i, cy - i),
        ]),
    ]
}

/// Button_33 (163): DualSense Triangle (`△`), optically centered in `(640, 640)` disc.
fn glyph_triangle() -> Vec<Contour> {
    vec![
        // Outer equilateral triangle (CW): height 496, base 572
        Contour::from_polygon(&[(640, 366), (926, 862), (354, 862)]),
        // Inner equilateral hole (CCW): uniform 90-twip wall thickness
        Contour::from_polygon(&[(640, 546), (510, 772), (770, 772)]),
    ]
}

/// Bold geometric numeral `1` with angled top flag, spanning `y = y0..y0+440`.
fn contour_bumper_1(stem_right: i32, y0: i32) -> Contour {
    let stem_w = 94;
    let sr = stem_right;
    let sl = sr - stem_w;
    let fl = sl - 104;
    Contour::from_polygon(&[
        (sl + 10, y0),
        (sr, y0),
        (sr, y0 + 440),
        (sl, y0 + 440),
        (sl, y0 + 106),
        (fl, y0 + 168),
        (fl, y0 + 84),
    ])
}

/// Button_34 (76): DualSense `L1` inside the left bumper frame.
fn glyph_l1() -> Vec<Contour> {
    let y0 = 452;
    let y1 = y0 + 440;
    // Bold geometric `L` (CW)
    let l_contour = Contour::from_polygon(&[
        (386, y0),
        (480, y0),
        (480, y1 - 84),
        (666, y1 - 84),
        (666, y1),
        (386, y1),
    ]);
    // Bold geometric `1` (CW)
    let one_contour = contour_bumper_1(928, y0);
    vec![l_contour, one_contour]
}

/// Button_36 (93): DualSense `R1` inside the right bumper frame.
fn glyph_r1() -> Vec<Contour> {
    let y0 = 452;
    let dx = 24;
    // Bold geometric `R` (CW outer contour + CCW bowl counter)
    let r_outer = Contour {
        start: (304 + dx, y0),
        segments: vec![
            Segment::Line(488 + dx, y0),
            // Upper-right outer bowl curve
            Segment::Quad(616 + dx, y0, 616 + dx, y0 + 132),
            // Lower-right outer bowl curve into leg crotch
            Segment::Quad(616 + dx, y0 + 234, 528 + dx, y0 + 258),
            // Diagonal outer right leg
            Segment::Line(630 + dx, y0 + 440),
            Segment::Line(522 + dx, y0 + 440),
            // Diagonal inner right leg
            Segment::Line(430 + dx, y0 + 270),
            Segment::Line(398 + dx, y0 + 270),
            Segment::Line(398 + dx, y0 + 440),
            Segment::Line(304 + dx, y0 + 440),
            Segment::Line(304 + dx, y0),
        ],
    };
    let r_hole = Contour {
        start: (398 + dx, y0 + 80),
        segments: vec![
            // CCW: down left inner wall, right across bottom, curve up around bowl, left along top
            Segment::Line(398 + dx, y0 + 190),
            Segment::Line(474 + dx, y0 + 190),
            Segment::Quad(522 + dx, y0 + 190, 522 + dx, y0 + 135),
            Segment::Quad(522 + dx, y0 + 80, 474 + dx, y0 + 80),
            Segment::Line(398 + dx, y0 + 80),
        ],
    };
    let one_contour = contour_bumper_1(904 + dx, y0);
    vec![r_outer, r_hole, one_contour]
}

/// Buttons 35 (85) & 37 (107): Bold geometric numeral `2` (`368 x 552` twips)
/// matched in cap height (`344..896`) and stroke weight (`~104-112` twips) to
/// the existing untouched `L` (`84`) and `R` (`106`) trigger letters.
fn glyph_digit_2(x0: i32, y0: i32) -> Vec<Contour> {
    let p = |u: i32, v: i32| (x0 + u, y0 + v);
    let q = |cu: i32, cv: i32, eu: i32, ev: i32| Segment::Quad(x0 + cu, y0 + cv, x0 + eu, y0 + ev);
    let l = |u: i32, v: i32| Segment::Line(x0 + u, y0 + v);

    let contour = Contour {
        start: p(14, 184),
        segments: vec![
            // Outer top-left arch up to crown
            q(16, 68, 92, 22),
            q(134, 0, 188, 0),
            // Outer top-right arch down to shoulder
            q(262, 0, 314, 42),
            q(364, 84, 364, 158),
            // Outer right descending sweep to diagonal waist
            q(364, 224, 312, 286),
            l(154, 448),
            // Top of horizontal base bar -> right corner -> bottom-right -> bottom-left
            l(368, 448),
            l(368, 552),
            l(0, 552),
            l(0, 466),
            // Inner diagonal sweep up into underside of arch
            l(214, 240),
            q(254, 198, 254, 158),
            q(254, 122, 232, 104),
            q(212, 92, 186, 92),
            q(152, 92, 134, 112),
            q(116, 132, 114, 184),
            l(14, 184),
        ],
    };
    vec![contour]
}

/// Button_38 (120): DualSense Options (3 sleek stacked pill-ended horizontal bars).
fn glyph_options() -> Vec<Contour> {
    let (x0, x1) = (348, 932);
    let half_h = 38;
    let r = 38;
    [468, 640, 812]
        .into_iter()
        .map(|cy| rounded_rect_contour(x0, cy - half_h, x1, cy + half_h, r, true))
        .collect()
}

/// Button_39 (130): DualSense Touchpad (clean landscape rounded-rectangle frame).
fn glyph_touchpad() -> Vec<Contour> {
    vec![
        // Outer rounded rectangle (CW): 748 x 452 centered at (640, 640)
        rounded_rect_contour(266, 414, 1014, 866, 72, true),
        // Inner rounded cutout (CCW): uniform 76-twip frame wall
        rounded_rect_contour(342, 490, 938, 790, 26, false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple MSB-first SWF bitstream reader for round-trip verification in tests.
    struct BitReader<'a> {
        buf: &'a [u8],
        pos: usize,
        acc: u64,
        bits: u32,
    }

    impl<'a> BitReader<'a> {
        fn new(buf: &'a [u8], pos: usize) -> Self {
            Self {
                buf,
                pos,
                acc: 0,
                bits: 0,
            }
        }

        fn align(&mut self) {
            self.bits = 0;
        }

        fn ub(&mut self, n: u8) -> u32 {
            if n == 0 {
                return 0;
            }
            while self.bits < u32::from(n) {
                self.acc = (self.acc << 8) | u64::from(self.buf[self.pos]);
                self.pos += 1;
                self.bits += 8;
            }
            self.bits -= u32::from(n);
            let mask = if n == 32 {
                u64::from(u32::MAX)
            } else {
                (1u64 << n) - 1
            };
            ((self.acc >> self.bits) & mask) as u32
        }

        fn sb(&mut self, n: u8) -> i32 {
            let v = self.ub(n);
            if n > 0 && (v & (1u32 << (n - 1))) != 0 {
                (v as i32) - (1i32 << n)
            } else {
                v as i32
            }
        }

        fn rect(&mut self) -> (i32, i32, i32, i32) {
            self.align();
            let nb = self.ub(5) as u8;
            let xmin = self.sb(nb);
            let xmax = self.sb(nb);
            let ymin = self.sb(nb);
            let ymax = self.sb(nb);
            self.align();
            (xmin, xmax, ymin, ymax)
        }
    }

    /// Parses an encoded `DefineShape3` payload and verifies every record closes
    /// cleanly, winds CW for outer contours / CCW for holes, and stays within bounds.
    fn verify_shape3_payload(payload: &[u8], expected_id: u16, expected_contours: usize) {
        let cid = u16::from_le_bytes([payload[0], payload[1]]);
        assert_eq!(cid, expected_id);
        let mut r = BitReader::new(payload, 2);
        let (xmin, xmax, ymin, ymax) = r.rect();
        assert!(xmin < xmax && ymin < ymax, "non-empty bounds for {cid}");
        assert!(
            xmin >= 200 && xmax <= 1120 && ymin >= 300 && ymax <= 960,
            "shape {cid} bounds ({xmin}..{xmax}, {ymin}..{ymax}) inside button frame"
        );
        // Solid black fill style 1, 0 line styles
        let pos = r.pos;
        assert_eq!(&payload[pos..pos + 7], &[1, 0, 0, 0, 0, 0xFF, 0]);
        let mut br = BitReader::new(payload, pos + 7);
        let nfill = br.ub(4) as u8;
        let nline = br.ub(4) as u8;
        assert_eq!((nfill, nline), (1, 0));

        let mut contours: Vec<Vec<(i32, i32)>> = Vec::new();
        let mut cur: Vec<(i32, i32)> = Vec::new();
        let (mut cx, mut cy) = (0i32, 0i32);
        let mut fill0 = 0u32;

        loop {
            if br.ub(1) == 0 {
                let flags = br.ub(5);
                if flags == 0 {
                    if !cur.is_empty() {
                        contours.push(cur);
                    }
                    break;
                }
                if !cur.is_empty() {
                    contours.push(std::mem::take(&mut cur));
                }
                if flags & 0x01 != 0 {
                    let mb = br.ub(5) as u8;
                    cx = br.sb(mb);
                    cy = br.sb(mb);
                }
                if flags & 0x02 != 0 {
                    fill0 = br.ub(nfill);
                }
                assert_eq!(flags & 0x1C, 0, "unexpected style flags");
                assert_eq!(fill0, 1, "fillStyle0 must remain 1");
                cur.push((cx, cy));
            } else {
                let straight = br.ub(1) == 1;
                let nb = (br.ub(4) + 2) as u8;
                if straight {
                    if br.ub(1) == 1 {
                        cx += br.sb(nb);
                        cy += br.sb(nb);
                    } else if br.ub(1) == 1 {
                        cy += br.sb(nb);
                    } else {
                        cx += br.sb(nb);
                    }
                    cur.push((cx, cy));
                } else {
                    let qx = cx + br.sb(nb);
                    let qy = cy + br.sb(nb);
                    cx = qx + br.sb(nb);
                    cy = qy + br.sb(nb);
                    cur.push((qx, qy));
                    cur.push((cx, cy));
                }
                assert!(
                    cx >= xmin && cx <= xmax && cy >= ymin && cy <= ymax,
                    "shape {cid} anchor ({cx}, {cy}) outside bounds ({xmin}..{xmax}, {ymin}..{ymax})"
                );
            }
        }

        br.align();
        assert_eq!(
            br.pos,
            payload.len(),
            "shape {cid} has trailing unparsed bytes"
        );
        assert_eq!(contours.len(), expected_contours, "contour count for {cid}");
        for (i, pts) in contours.iter().enumerate() {
            assert_eq!(pts.first(), pts.last(), "shape {cid} contour {i} must close");
        }
    }

    fn signed_area(pts: &[(i32, i32)]) -> i64 {
        pts.windows(2)
            .map(|w| i64::from(w[0].0) * i64::from(w[1].1) - i64::from(w[1].0) * i64::from(w[0].1))
            .sum()
    }

    #[test]
    fn prompts_env_parsing() {
        assert_eq!(parse_prompts_mode(None), PromptMode::Auto);
        assert_eq!(parse_prompts_mode(Some("")), PromptMode::Auto);
        assert_eq!(parse_prompts_mode(Some("auto")), PromptMode::Auto);
        assert_eq!(parse_prompts_mode(Some(" AUTO ")), PromptMode::Auto);

        for on in ["ps5", "PS5", "PlayStation", "dualsense", "1", "on", "true"] {
            assert_eq!(parse_prompts_mode(Some(on)), PromptMode::Ps5, "{on}");
        }
        for off in ["xbox", "XBOX", "0", "off", "false"] {
            assert_eq!(parse_prompts_mode(Some(off)), PromptMode::Xbox, "{off}");
        }
        assert_eq!(parse_prompts_mode(Some("unknown")), PromptMode::Auto);
    }

    #[test]
    fn data_rpf_path_matching() {
        assert!(is_data_rpf_path(r"G:\data.rpf"));
        assert!(is_data_rpf_path(r"C:/Games/Project/DATA.RPF"));
        assert!(is_data_rpf_path("data.rpf"));
        assert!(!is_data_rpf_path(r"G:\appdata0.rpf"));
        assert!(!is_data_rpf_path(r"G:\data.rpf.bak"));

        let wide: Vec<u16> = r"G:\DATA.RPF".encode_utf16().collect();
        assert!(is_data_rpf_wide(&wide));
        let other: Vec<u16> = r"G:\other.rpf".encode_utf16().collect();
        assert!(!is_data_rpf_wide(&other));
    }

    #[test]
    fn all_ten_ps5_shapes_encode_and_parse_cleanly() {
        let specs = [
            (ID_CROSS, 1),
            (ID_CIRCLE, 2),
            (ID_SQUARE, 2),
            (ID_TRIANGLE, 2),
            (ID_L1, 2),
            (ID_L2_NUM, 1),
            (ID_R1, 3),
            (ID_R2_NUM, 1),
            (ID_OPTIONS, 3),
            (ID_TOUCHPAD, 2),
        ];
        let mut total_bytes = 0usize;
        for (cid, expected_contours) in specs {
            let payload = build_ps5_shape3_payload(cid).expect("target shape id");
            verify_shape3_payload(&payload, cid, expected_contours);
            total_bytes += payload.len();
        }
        // Must stay comfortably below the original 1,008 bytes of the 10 Xbox shapes.
        assert!(
            total_bytes <= 1008,
            "PS5 replacement shapes total {total_bytes} bytes > 1008 byte budget"
        );
    }

    #[test]
    fn hollow_shapes_have_cw_outer_and_ccw_inner_winding() {
        let contour_pts = |c: &Contour| -> Vec<(i32, i32)> {
            let mut pts = vec![c.start];
            for &s in &c.segments {
                match s {
                    Segment::Line(x, y) => pts.push((x, y)),
                    Segment::Quad(qx, qy, x, y) => {
                        pts.push((qx, qy));
                        pts.push((x, y));
                    }
                }
            }
            pts
        };
        for contours in [glyph_circle(), glyph_square(), glyph_triangle(), glyph_touchpad()] {
            assert_eq!(contours.len(), 2);
            assert!(
                signed_area(&contour_pts(&contours[0])) > 0,
                "outer contour must wind CW in Y-down space"
            );
            assert!(
                signed_area(&contour_pts(&contours[1])) < 0,
                "inner hole must wind CCW in Y-down space"
            );
        }
        // R1: outer R is CW, R bowl hole is CCW, digit 1 is CW.
        let r1 = glyph_r1();
        assert!(signed_area(&contour_pts(&r1[0])) > 0);
        assert!(signed_area(&contour_pts(&r1[1])) < 0);
        assert!(signed_area(&contour_pts(&r1[2])) > 0);
    }

    /// Builds a clean-room synthetic uncompressed `GFX` SWF containing target
    /// `DefineShape3` tags, a preserved unrelated `DefineShape3`, and a
    /// `SymbolClass` marker tag.
    fn synthetic_input_icons_swf() -> Vec<u8> {
        let mut swf = Vec::new();
        swf.extend_from_slice(b"GFX");
        swf.push(42);
        swf.extend_from_slice(&0u32.to_le_bytes()); // fileLength placeholder
        // Stage RECT (0, 1280, 0, 1280)
        let mut rw = BitWriter::new();
        rw.write_rect(0, 1280, 0, 1280);
        swf.extend_from_slice(&rw.finish());
        swf.extend_from_slice(&(30u16 << 8).to_le_bytes()); // 30 fps
        swf.extend_from_slice(&1u16.to_le_bytes()); // 1 frame

        // Unrelated background shape (ID 144) that must remain verbatim.
        let bg = encode_shape3(144, &[rounded_rect_contour(30, 30, 1250, 1250, 100, true)]);
        write_swf_tag(&mut swf, TAG_DEFINE_SHAPE3, &bg);

        // Placeholder Xbox-sized shapes for all 10 target IDs with high-entropy
        // payload bytes so zlib re-compression matches real-world size ratios.
        for cid in [
            ID_CROSS,
            ID_CIRCLE,
            ID_SQUARE,
            ID_TRIANGLE,
            ID_L1,
            ID_L2_NUM,
            ID_R1,
            ID_R2_NUM,
            ID_OPTIONS,
            ID_TOUCHPAD,
        ] {
            let mut dummy: Vec<u8> = (0..180usize)
                .map(|i| ((i * 73 + usize::from(cid) * 31) ^ (i >> 1)) as u8)
                .collect();
            dummy[0..2].copy_from_slice(&cid.to_le_bytes());
            write_swf_tag(&mut swf, TAG_DEFINE_SHAPE3, &dummy);
        }

        // SymbolClass (tag 76) containing the target marker string.
        let mut sym = Vec::new();
        sym.extend_from_slice(&1u16.to_le_bytes());
        sym.extend_from_slice(&147u16.to_le_bytes());
        sym.extend_from_slice(INPUT_ICONS_MARKER);
        sym.push(0);
        write_swf_tag(&mut swf, 76, &sym);

        // ShowFrame + End
        write_swf_tag(&mut swf, 1, &[]);
        write_swf_tag(&mut swf, TAG_END, &[]);

        let total_len = swf.len() as u32;
        swf[4..8].copy_from_slice(&total_len.to_le_bytes());
        swf
    }

    #[test]
    fn synthetic_swf_round_trip_and_zlib_recompression() {
        let orig = synthetic_input_icons_swf();
        let mut patched = orig.clone();
        assert!(patch_input_icons_gfx(&mut patched));
        assert_eq!(patched.len(), orig.len());
        assert_eq!(
            u32::from_le_bytes([patched[4], patched[5], patched[6], patched[7]]) as usize,
            orig.len()
        );
        assert_ne!(patched, orig);

        // Walk every tag in the patched SWF to verify integrity and untouched shape 144.
        let mut pos = swf_tags_start(&patched).unwrap();
        let mut seen_ids = Vec::new();
        let mut saw_end = false;
        while pos + 2 <= patched.len() {
            let hdr = u16::from_le_bytes([patched[pos], patched[pos + 1]]);
            pos += 2;
            let code = hdr >> 6;
            let mut len = usize::from(hdr & 0x3F);
            if len == 0x3F {
                len = u32::from_le_bytes([
                    patched[pos],
                    patched[pos + 1],
                    patched[pos + 2],
                    patched[pos + 3],
                ]) as usize;
                pos += 4;
            }
            let payload = &patched[pos..pos + len];
            pos += len;
            if code == TAG_DEFINE_SHAPE3 {
                let cid = u16::from_le_bytes([payload[0], payload[1]]);
                seen_ids.push(cid);
                if let Some(expected) = build_ps5_shape3_payload(cid) {
                    assert_eq!(payload, expected.as_slice());
                }
            }
            if code == TAG_END {
                saw_end = true;
                break;
            }
        }
        assert!(saw_end);
        assert!(patched[pos..].iter().all(|&b| b == 0), "trailing bytes zero-padded");
        assert_eq!(seen_ids.len(), 11);
        assert_eq!(seen_ids[0], 144);

        // Re-running on an already-patched buffer is idempotent.
        let snapshot = patched.clone();
        assert!(patch_input_icons_gfx(&mut patched));
        assert_eq!(patched, snapshot);

        // Zlib round-trip helper fits within compressed budget and inflates identically.
        let mut comp_buf = vec![0u8; zlib_rs::compress_bound(orig.len())];
        let (comp, rc) =
            zlib_rs::compress_slice(&mut comp_buf, &orig, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);
        let patched_comp = patch_compressed_input_icons(comp, comp.len()).unwrap();
        assert_eq!(patched_comp.len(), comp.len());
        let reinflated = inflate_zlib(&patched_comp, orig.len() + 1024).unwrap();
        assert_eq!(reinflated, snapshot);
    }

    #[test]
    fn compressed_region_hook_respects_offset_and_toggle() {
        let orig = synthetic_input_icons_swf();
        let mut comp_buf = vec![0u8; zlib_rs::compress_bound(orig.len())];
        let (comp, rc) =
            zlib_rs::compress_slice(&mut comp_buf, &orig, zlib_rs::DeflateConfig::new(6));
        assert_eq!(rc, zlib_rs::ReturnCode::Ok);

        // Simulate a chunked read starting 64 bytes before INPUT_ICONS_RPF_OFFSET.
        let lead = 64usize;
        let mut chunk = vec![0xCCu8; lead + INPUT_ICONS_COMPRESSED_LEN + 32];
        chunk[lead..lead + comp.len()].copy_from_slice(comp);
        let before = chunk.clone();

        // Disabled -> untouched.
        assert!(!patch_data_rpf_compressed_region_inner(
            INPUT_ICONS_RPF_OFFSET - lead as u64,
            &mut chunk,
            || false
        ));
        assert_eq!(chunk, before);

        // Wrong offset -> untouched.
        assert!(!patch_data_rpf_compressed_region_inner(0, &mut chunk, || true));
        assert_eq!(chunk, before);

        // Matching offset + enabled -> patched cleanly, surrounding bytes untouched.
        assert!(patch_data_rpf_compressed_region_inner(
            INPUT_ICONS_RPF_OFFSET - lead as u64,
            &mut chunk,
            || true
        ));
        assert!(chunk[..lead].iter().all(|&b| b == 0xCC));
        assert!(chunk[lead + INPUT_ICONS_COMPRESSED_LEN..].iter().all(|&b| b == 0xCC));
        let mut expected = orig;
        assert!(patch_input_icons_gfx(&mut expected));
        let reinflated = inflate_zlib(&chunk[lead..lead + INPUT_ICONS_COMPRESSED_LEN], 65_536).unwrap();
        assert_eq!(reinflated, expected);
    }

    #[test]
    fn non_matching_buffers_are_untouched() {
        let mut bogus = b"GFX\x2a\x10\x00\x00\x00not_input_icons_at_all_123456".to_vec();
        let copy = bogus.clone();
        assert!(!patch_input_icons_gfx(&mut bogus));
        assert_eq!(bogus, copy);
    }
}
