//! The frame rate, drawn into the top left corner of every frame shown.
//!
//! The window's caption has the frame rate, but a capture of the window's
//! picture (a stream) does not carry the caption. Here the number is part
//! of the picture: a small box of text copied onto the swap chain's back
//! buffer right after the title's frame, before it is presented. The
//! title's own frame is not touched, so saved frames and everything read
//! back from it are as before.
//!
//! The text is drawn on the CPU with a font of its own (5 by 7 dots a
//! glyph) into a buffer the card copies from; a buffer is made once for a
//! text and never written again, so a present still in the air reads what
//! it was recorded with. On by default; `VICTORY_FPS_COUNTER=0` leaves the
//! frame as the title drew it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The frame rate in tenths of a frame a second, as last measured (once a
/// second, by the presenter's meter); `u32::MAX` before the first.
static TENTHS: AtomicU32 = AtomicU32::new(u32::MAX);

/// Whether the counter is drawn.
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FPS_COUNTER").is_none_or(|v| v != "0"))
}

/// The frame rate just measured.
pub(crate) fn set(fps: f64) {
    if fps.is_finite() && fps >= 0.0 {
        TENTHS.store((fps * 10.0).round().min(99_999.0) as u32, Ordering::Relaxed);
    }
}

/// A glyph's seven rows, five dots each (bit 4 is the left one).
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        '%' => [0b11001, 0b11010, 0b00100, 0b01000, 0b10011, 0b00011, 0],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        _ => [0; 7],
    }
}

/// The text for a frame rate in tenths.
fn text(tenths: u32) -> String {
    format!("{}.{} FPS", tenths / 10, tenths % 10)
}

/// Seamless in-frame shader warmup badge text while pipelines are compiling.
fn shader_text(pct: u32, pre_world: bool) -> String {
    if pre_world {
        format!("LOADING SHADERS {pct}%")
    } else {
        format!("SHADERS {pct}%")
    }
}

/// Bytes a pixel of a swap chain format this layer presents in, or none.
fn pixel_bytes(format: DXGI_FORMAT) -> Option<usize> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

/// One pixel of `format`: of the text (`lit`) or of the box behind it.
#[cfg(test)]
fn pixel(format: DXGI_FORMAT, lit: bool, out: &mut [u8]) {
    pixel_rgb(format, (1.0, 0.92, 0.16), (0.03, 0.03, 0.03), lit, out);
}

fn pixel_rgb(format: DXGI_FORMAT, fg: (f32, f32, f32), bg: (f32, f32, f32), lit: bool, out: &mut [u8]) {
    let (r, g, b) = if lit { fg } else { bg };
    let byte = |v: f32| (v * 255.0).round() as u8;
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => out.copy_from_slice(&[byte(r), byte(g), byte(b), 255]),
        DXGI_FORMAT_B8G8R8A8_UNORM => out.copy_from_slice(&[byte(b), byte(g), byte(r), 255]),
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let ten = |v: f32| (v * 1023.0).round() as u32;
            out.copy_from_slice(&(ten(r) | ten(g) << 10 | ten(b) << 20 | 3 << 30).to_le_bytes());
        }
        _ => {
            for (i, v) in [r, g, b, 1.0].into_iter().enumerate() {
                out[i * 2..i * 2 + 2].copy_from_slice(&half(v).to_le_bytes());
            }
        }
    }
}

/// A value from 0 to 1 as a 16-bit float.
fn half(v: f32) -> u16 {
    let bits = v.clamp(0.0, 1.0).to_bits();
    if v <= 0.0 {
        return 0;
    }
    // Sign 0; the exponent moves from a bias of 127 to one of 15; the top
    // ten bits of the fraction stay. Values under 2^-14 (none of ours) are 0.
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    if exponent <= 0 {
        return 0;
    }
    ((exponent as u32) << 10 | (bits >> 13) & 0x3FF) as u16
}

/// The box for `text` at `scale` dots a pixel: its width and height in
/// pixels and, row by row from the top, whether each pixel is text.
fn bitmap(text: &str, scale: u32) -> (u32, u32, Vec<bool>) {
    let chars: Vec<char> = text.chars().collect();
    let pad = 2 * scale;
    let width = 2 * pad + chars.len() as u32 * 6 * scale - scale;
    let height = 2 * pad + 7 * scale;
    let mut lit = vec![false; (width * height) as usize];
    for (i, &c) in chars.iter().enumerate() {
        let rows = glyph(c);
        for (row, bits) in rows.into_iter().enumerate() {
            for column in 0..5u32 {
                if bits >> (4 - column) & 1 == 0 {
                    continue;
                }
                let (x0, y0) = (pad + (i as u32 * 6 + column) * scale, pad + row as u32 * scale);
                for y in y0..y0 + scale {
                    for x in x0..x0 + scale {
                        lit[(y * width + x) as usize] = true;
                    }
                }
            }
        }
    }
    (width, height, lit)
}

/// A buffer the card copies a text's box from, with its footprint.
struct Made {
    buffer: ID3D12Resource,
    width: u32,
    height: u32,
    pitch: u32,
}

// SAFETY: a free-threaded D3D12 object and plain numbers.
unsafe impl Send for Made {}

/// The boxes made so far, by text, format and scale.
static MADE: Mutex<Option<HashMap<(u32, i32, u32), Made>>> = Mutex::new(None);
/// Seamless shader-warmup badges cached by `(pct, pre_world, format, scale)`.
static SHADER_MADE: Mutex<Option<HashMap<(u32, bool, i32, u32), Made>>> = Mutex::new(None);

fn build_box(
    host: &crate::host::Host,
    label: &str,
    format: DXGI_FORMAT,
    bytes: usize,
    scale: u32,
    fg: (f32, f32, f32),
    bg: (f32, f32, f32),
) -> Option<Made> {
    let (w, h, lit) = bitmap(label, scale);
    let pitch = (w as usize * bytes).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize);
    let (buffer, data) = host.shared_buffer((pitch * h as usize) as u64)?;
    // SAFETY: the buffer's own mapped bytes, `pitch * h` of them.
    let out = unsafe { std::slice::from_raw_parts_mut(data, pitch * h as usize) };
    for y in 0..h as usize {
        for x in 0..w as usize {
            let at = y * pitch + x * bytes;
            pixel_rgb(format, fg, bg, lit[y * w as usize + x], &mut out[at..at + bytes]);
        }
    }
    Some(Made { buffer, width: w, height: h, pitch: pitch as u32 })
}

fn blit_box(list: &ID3D12GraphicsCommandList, back: &ID3D12Resource, format: DXGI_FORMAT, m: &Made, dst_x: u32, dst_y: u32) {
    // SAFETY: an open list; the back buffer is a copy's destination in it
    // and the box lies inside it; our buffer holds `pitch * height` bytes
    // and is kept for as long as the process runs.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&m.buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: m.width, Height: m.height, Depth: 1, RowPitch: m.pitch },
                },
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(back),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.CopyTextureRegion(&to, dst_x, dst_y, 0, &from, None);
    }
}

/// Copies the frame rate's box (and, while shaders are compiling, a seamless
/// warmup badge right beneath it) onto `back`, a back buffer of `width` by
/// `height` pixels of `format` that `list` holds as a copy's destination.
pub(crate) fn draw(host: &crate::host::Host, list: &ID3D12GraphicsCommandList, back: &ID3D12Resource, (width, height, format): (u32, u32, DXGI_FORMAT)) {
    if !on() {
        return;
    }
    let Some(bytes) = pixel_bytes(format) else {
        return;
    };
    let tenths = TENTHS.load(Ordering::Relaxed);
    let shader_progress = crate::pipeline::loading_status();
    if tenths == u32::MAX && shader_progress.is_none() {
        return;
    }
    // About a fortieth of the frame's height a glyph: 6 dots a pixel at
    // 2160 lines, 3 at 1080.
    let scale = (height / 360).clamp(2, 12);
    let margin = 4 * scale;
    let mut next_y = margin;

    if tenths != u32::MAX {
        let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
        let made = made.get_or_insert_with(HashMap::new);
        if made.len() > 2048 {
            made.clear();
        }
        let key = (tenths, format.0, scale);
        if let std::collections::hash_map::Entry::Vacant(e) = made.entry(key)
            && let Some(m) = build_box(host, &text(tenths), format, bytes, scale, (1.0, 0.92, 0.16), (0.03, 0.03, 0.03))
        {
            e.insert(m);
        }
        if let Some(m) = made.get(&key)
            && m.width + 2 * margin <= width
            && m.height + 2 * margin <= height
        {
            blit_box(list, back, format, m, margin, margin);
            next_y = margin + m.height + 2 * scale;
        }
    }

    if let Some((done, total, pre_world)) = shader_progress
        && total > 0
    {
        let pct = ((done.saturating_mul(100)) / total).min(99) as u32;
        let shader_scale = (scale.saturating_sub(1)).max(2);
        let mut made = SHADER_MADE.lock().unwrap_or_else(|e| e.into_inner());
        let made = made.get_or_insert_with(HashMap::new);
        if made.len() > 256 {
            made.clear();
        }
        let key = (pct, pre_world, format.0, shader_scale);
        if let std::collections::hash_map::Entry::Vacant(e) = made.entry(key)
            && let Some(m) = build_box(
                host,
                &shader_text(pct, pre_world),
                format,
                bytes,
                shader_scale,
                (0.35, 0.88, 1.0),
                (0.03, 0.05, 0.09),
            )
        {
            e.insert(m);
        }
        if let Some(m) = made.get(&key)
            && m.width + 2 * margin <= width
            && next_y + m.height + margin <= height
        {
            blit_box(list, back, format, m, margin, next_y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_is_the_rate_to_a_tenth() {
        assert_eq!(text(103), "10.3 FPS");
        assert_eq!(text(7), "0.7 FPS");
        assert_eq!(text(300), "30.0 FPS");
    }

    #[test]
    fn the_box_holds_every_glyph_with_a_margin() {
        let (w, h, lit) = bitmap("10.3 FPS", 3);
        // Eight glyphs of five dots and seven gaps of one, a pad of two dots all round.
        assert_eq!((w, h), ((8 * 5 + 7 + 4) * 3, (7 + 4) * 3));
        assert_eq!(lit.len(), (w * h) as usize);
        // The pad is box, not text; the "1" has its stem lit.
        assert!(lit[..(w * 6) as usize].iter().all(|&l| !l), "the top pad");
        assert!((0..h).all(|y| !lit[(y * w) as usize] && !lit[(y * w + w - 1) as usize]), "the side pads");
        let at = |x: u32, y: u32| lit[(y * w + x) as usize];
        // Glyph 0 ("1"): dot column 2 of row 0 is lit, column 0 is not.
        assert!(at(6 + 2 * 3, 6) && !at(6, 6));
        // A space lights nothing: glyph 4.
        assert!((0..7 * 3).all(|y| (0..5 * 3).all(|x| !at(6 + 4 * 18 + x, 6 + y))));
        // Every glyph used has at least one dot, but the space.
        for c in "0123456789.%/ADEFGHILNOPRS".chars() {
            assert!(glyph(c).iter().any(|&r| r != 0), "{c}");
        }
        assert_eq!(shader_text(69, true), "LOADING SHADERS 69%");
        assert_eq!(shader_text(42, false), "SHADERS 42%");
    }

    #[test]
    fn pixels_are_in_the_formats_the_layer_presents_in() {
        let mut four = [0u8; 4];
        pixel(DXGI_FORMAT_R8G8B8A8_UNORM, true, &mut four);
        assert_eq!(four, [255, 235, 41, 255]);
        pixel(DXGI_FORMAT_B8G8R8A8_UNORM, true, &mut four);
        assert_eq!(four, [41, 235, 255, 255]);
        pixel(DXGI_FORMAT_R10G10B10A2_UNORM, true, &mut four);
        let v = u32::from_le_bytes(four);
        assert_eq!((v & 1023, v >> 10 & 1023, v >> 20 & 1023, v >> 30), (1023, 941, 164, 3));
        let mut eight = [0u8; 8];
        pixel(DXGI_FORMAT_R16G16B16A16_FLOAT, true, &mut eight);
        assert_eq!(u16::from_le_bytes([eight[0], eight[1]]), 0x3C00, "1.0 as a half");
        assert_eq!(u16::from_le_bytes([eight[6], eight[7]]), 0x3C00);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(0.0), 0);
        assert_eq!(pixel_bytes(DXGI_FORMAT_R16G16B16A16_FLOAT), Some(8));
        assert_eq!(pixel_bytes(DXGI_FORMAT_D32_FLOAT), None);
    }
}
