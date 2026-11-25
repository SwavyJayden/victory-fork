//! The DXGI side of the device: one adapter with one output, reached from
//! the device by interface query. Tables follow the PC interfaces' order
//! (IUnknown, the object methods, then each interface's methods).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{
    self, ADAPTER, DXGI_DEVICE, E_INVALIDARG, FACTORY, HRESULT, Iid, MORE_DATA, NOT_FOUND, OUTPUT,
    Object, S_OK, Table, This, guid,
};

/// The DXGI device interfaces, base first (as on PC).
pub(crate) const DEVICE_IIDS: [Iid; 3] = [
    guid(
        0x54ec77fa,
        0x1377,
        0x44e6,
        [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c],
    ),
    guid(
        0x77db970f,
        0x6276,
        0x48ba,
        [0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c],
    ),
    guid(
        0x05008617,
        0xfbfd,
        0x4051,
        [0xa7, 0x90, 0x14, 0x48, 0x84, 0xb4, 0xf6, 0xa9],
    ),
];
const ADAPTER_IIDS: [Iid; 3] = [
    guid(
        0x2411e7e1,
        0x12ac,
        0x4ccf,
        [0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0],
    ),
    guid(
        0x29038f61,
        0x3839,
        0x4626,
        [0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05],
    ),
    guid(
        0x0aa1ae0a,
        0xfa0e,
        0x4b84,
        [0x86, 0x44, 0xe0, 0x5f, 0xf8, 0xe5, 0xac, 0xb5],
    ),
];
const FACTORY_IIDS: [Iid; 3] = [
    guid(
        0x7b7166ec,
        0x21c7,
        0x44ae,
        [0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69],
    ),
    guid(
        0x770aae78,
        0xf26f,
        0x4dba,
        [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    ),
    guid(
        0x50c83a1c,
        0xe072,
        0x4c48,
        [0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0],
    ),
];
const OUTPUT_IIDS: [Iid; 1] = [guid(
    0xae02eedb,
    0xc735,
    0x4690,
    [0x8d, 0x52, 0x5a, 0x8d, 0xc2, 0x02, 0x13, 0xaa],
)];

static DEVICE: Object = Object::new(&DEVICE_FUNCTIONS);
static ADAPTER_OBJECT: Object = Object::new(&ADAPTER_FUNCTIONS);
static FACTORY_OBJECT: Object = Object::new(&FACTORY_FUNCTIONS);
static OUTPUT_OBJECT: Object = Object::new(&OUTPUT_FUNCTIONS);

pub(crate) fn device() -> This {
    &DEVICE
}

/// `found` if `iid` is one of `known`.
fn query(kind: usize, iid: *const Iid, known: &[Iid], found: This, out: *mut This) -> HRESULT {
    object::answer(kind, iid, object::is(iid, known).then_some(found), out)
}

/// The DXGI device also answers for the D3D12 device.
extern "system" fn device_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    if crate::device::is_device(iid) {
        return object::answer(DXGI_DEVICE, iid, Some(crate::device::device()), out);
    }
    query(DXGI_DEVICE, iid, &DEVICE_IIDS, this, out)
}
extern "system" fn device_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(DXGI_DEVICE, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}
extern "system" fn device_adapter(_: This, out: *mut This) -> HRESULT {
    object::answer(DXGI_DEVICE, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn adapter_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &ADAPTER_IIDS, this, out)
}
extern "system" fn adapter_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(ADAPTER, iid, &FACTORY_IIDS, &FACTORY_OBJECT, out)
}
extern "system" fn adapter_outputs(_: This, index: u32, out: *mut This) -> HRESULT {
    if index != 0 {
        return NOT_FOUND;
    }
    object::answer(ADAPTER, &OUTPUT_IIDS[0], Some(&OUTPUT_OBJECT), out)
}

#[repr(C)]
struct AdapterDescription {
    name: [u16; 128],
    vendor: u32,
    device: u32,
    subsystem: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    luid: [u32; 2],
}
const _: () = assert!(size_of::<AdapterDescription>() == 0x130);

const AMD: u32 = 0x1002;

extern "system" fn adapter_description(_: This, out: *mut AdapterDescription) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let mut name = [0; 128];
    for (slot, unit) in name.iter_mut().zip("VIctory console GPU".encode_utf16()) {
        *slot = unit;
    }
    // shortcut: memory sizes are the title's split (GPU-optimal, standard),
    // not measured; revisit once memory is emulated.
    let (gpu_optimal, standard) = crate::console::current().title_memory();
    let description = AdapterDescription {
        name,
        vendor: AMD,
        device: 0,
        subsystem: 0,
        revision: 0,
        dedicated_video_memory: gpu_optimal,
        dedicated_system_memory: 0,
        shared_system_memory: standard,
        luid: [1, 0],
    };
    // SAFETY: checked non-null; the title passes room for the description.
    unsafe { out.write(description) };
    S_OK
}

extern "system" fn factory_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(FACTORY, iid, &FACTORY_IIDS, this, out)
}
extern "system" fn factory_adapters(_: This, index: u32, out: *mut This) -> HRESULT {
    if index != 0 {
        return NOT_FOUND;
    }
    object::answer(FACTORY, &ADAPTER_IIDS[0], Some(&ADAPTER_OBJECT), out)
}

extern "system" fn output_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &OUTPUT_IIDS, this, out)
}
extern "system" fn output_parent(_: This, iid: *const Iid, out: *mut This) -> HRESULT {
    query(OUTPUT, iid, &ADAPTER_IIDS, &ADAPTER_OBJECT, out)
}

/// IUnknown 3, object 4, device 5 + 2 + 3.
static DEVICE_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<DXGI_DEVICE, 32>();
    f[0] = device_query as *const ();
    f[6] = device_parent as *const (); // 0x30
    f[7] = device_adapter as *const (); // 0x38
    Table(f)
};

/// IUnknown 3, object 4, adapter 3 + 1 + 1.
static ADAPTER_FUNCTIONS: Table<24> = {
    let mut f = object::missing_table::<ADAPTER, 24>();
    f[0] = adapter_query as *const ();
    f[6] = adapter_parent as *const (); // 0x30
    f[7] = adapter_outputs as *const (); // 0x38
    f[8] = adapter_description as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, factory 5 + 2 + 11.
static FACTORY_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<FACTORY, 32>();
    f[0] = factory_query as *const ();
    f[7] = factory_adapters as *const (); // 0x38
    f[12] = factory_adapters as *const (); // 0x60, the same for version 1
    Table(f)
};

/// A display mode as the console lists it: size, refresh rate (numerator,
/// denominator), format, scanline order, scaling, flags.
#[repr(C)]
struct Mode {
    width: u32,
    height: u32,
    refresh: [u32; 2],
    format: u32,
    scanlines: u32,
    scaling: u32,
    flags: u32,
}
const _: () = assert!(size_of::<Mode>() == 32);

/// Display mode configuration resolved from `VICTORY_DISPLAY_MODE` (and
/// `VICTORY_FPS_CAP` for 120 Hz reporting) or the active console default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DisplayMode {
    width: u32,
    height: u32,
    allow_120hz: bool,
}

/// Parses a resolution preset (`1080p`, `1440p`, `900p`, `720p`, `4k`/`2160p`/
/// `native`) or `WxH` within supported bounds (`640..=7680` x `360..=4320`).
fn parse_display_size(raw: &str) -> Option<(u32, u32)> {
    let trimmed = raw.trim();
    let preset = match trimmed.to_ascii_lowercase().as_str() {
        "720p" => Some((1280, 720)),
        "900p" => Some((1600, 900)),
        "1080p" => Some((1920, 1080)),
        "1440p" => Some((2560, 1440)),
        "2160p" | "4k" | "native" => Some((3840, 2160)),
        _ => None,
    };
    if let Some(size) = preset {
        return Some(size);
    }
    let (w, h) = trimmed.split_once(['x', 'X'])?;
    let w = w.trim().parse::<u32>().ok()?;
    let h = h.trim().parse::<u32>().ok()?;
    ((640..=7680).contains(&w) && (360..=4320).contains(&h)).then_some((w, h))
}

fn parse_refresh_hz(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    let digits = trimmed
        .strip_suffix("hz")
        .or_else(|| trimmed.strip_suffix("HZ"))
        .or_else(|| trimmed.strip_suffix("Hz"))
        .unwrap_or(trimmed)
        .trim();
    let hz = digits.parse::<u32>().ok()?;
    matches!(hz, 60 | 120).then_some(hz)
}

fn fps_cap_allows_120hz(cap: Option<&str>) -> bool {
    let Some(raw) = cap.map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };
    if raw.eq_ignore_ascii_case("unlimited") || raw == "0" {
        return true;
    }
    raw.parse::<u32>().is_ok_and(|fps| fps > 60)
}

/// Resolves the console display mode from `VICTORY_DISPLAY_MODE` (e.g.
/// `1920x1080`, `2560x1440`, `1600x900`, `1280x720`, `1080p`, or `WxH`,
/// optionally suffixed with `@60` / `@120`), falling back cleanly to
/// 1920x1080 on Series S (`--console series-s`) and Native 4K (3840x2160) on
/// Series X.
fn resolve_display_mode(
    mode_var: Option<&str>,
    fps_cap_var: Option<&str>,
    console: crate::console::Console,
) -> DisplayMode {
    let default_size = match console {
        // A Series S on a 1080p set (its common case) lists 1080p; the
        // title sizes its targets from the mode and the device type.
        crate::console::Console::SeriesS => (1920, 1080),
        crate::console::Console::SeriesX => (3840, 2160),
    };
    let parsed = mode_var
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|v| match v.split_once('@') {
            Some((size_part, hz_part)) => Some((parse_display_size(size_part)?, Some(parse_refresh_hz(hz_part)?))),
            None => Some((parse_display_size(v)?, None)),
        });
    let ((width, height), explicit_hz) = match parsed {
        Some(res) => res,
        None => (default_size, None),
    };
    let allow_120hz = match explicit_hz {
        Some(120) => true,
        Some(_) => false,
        None => fps_cap_allows_120hz(fps_cap_var),
    };
    DisplayMode {
        width,
        height,
        allow_120hz,
    }
}

/// The display mode the console reports: `VICTORY_DISPLAY_MODE` if set (as a
/// console reports the television it is plugged into: a Series X on a 1080p
/// set lists a 1080p mode), else 1920x1080 for a Series S
/// (`--console series-s`) and 3840x2160 (Native 4K) for a Series X.
/// shortcut: not yet the host display's own size by default; make it so
/// once a title is seen to render for the mode it is given.
fn display_mode() -> DisplayMode {
    static MODE: std::sync::OnceLock<DisplayMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode_var = std::env::var("VICTORY_DISPLAY_MODE").ok();
        let fps_cap_var = std::env::var("VICTORY_FPS_CAP").ok();
        let mode = resolve_display_mode(
            mode_var.as_deref(),
            fps_cap_var.as_deref(),
            crate::console::current(),
        );
        tracing::info!(
            width = mode.width,
            height = mode.height,
            refresh_120hz = mode.allow_120hz,
            "display mode reported to the title"
        );
        mode
    })
}

fn modes_into(mode: DisplayMode, count: *mut u32, out: *mut Mode) -> HRESULT {
    // SAFETY: the title passes its count (null is tolerated).
    let Some(count) = (unsafe { count.as_mut() }) else {
        return E_INVALIDARG;
    };
    let rates: &[u32] = if mode.allow_120hz { &[60, 120] } else { &[60] };
    let available = rates.len() as u32;
    if out.is_null() {
        *count = available;
        return S_OK;
    }
    let requested = *count;
    let to_write = requested.min(available);
    for (i, &hz) in rates.iter().take(to_write as usize).enumerate() {
        let entry = Mode {
            width: mode.width,
            height: mode.height,
            refresh: [hz, 1],
            format: 0,
            scanlines: 1, // progressive
            scaling: 0,
            flags: 0,
        };
        // SAFETY: `out` is non-null and the caller provided room for `requested >= to_write` modes.
        unsafe { out.add(i).write(entry) };
    }
    *count = to_write;
    if requested < available {
        MORE_DATA
    } else {
        S_OK
    }
}

/// The console's mode list: 60 Hz (and 120 Hz when requested via
/// `VICTORY_DISPLAY_MODE=...@120` or `VICTORY_FPS_CAP` > 60 / `0`).
/// shortcut: fixed mode, not the host display's; report the host's rates if
/// a title paces by them.
extern "system" fn modes(
    _: This,
    _format: u32,
    _flags: u32,
    count: *mut u32,
    out: *mut Mode,
) -> HRESULT {
    modes_into(display_mode(), count, out)
}

/// IUnknown 3, object 4, output 15 (12 as on PC, then 3 console ones).
static OUTPUT_FUNCTIONS: Table<32> = {
    let mut f = object::missing_table::<OUTPUT, 32>();
    f[0] = output_query as *const ();
    f[6] = output_parent as *const (); // 0x30
    f[21] = modes as *const (); // 0xa8
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, slot};

    type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
    type Get = extern "system" fn(This, *mut This) -> HRESULT;
    type Indexed = extern "system" fn(This, u32, *mut This) -> HRESULT;
    type Describe = extern "system" fn(This, *mut AdapterDescription) -> HRESULT;

    /// Minecraft's startup: device → DXGI device (newest first) → adapter →
    /// description, factory, first output.
    #[test]
    fn startup_walks_from_device_to_output() {
        let d3d = crate::device::device();
        let newest_pc = guid(
            0x6007896c,
            0x3244,
            0x4afd,
            [0xbf, 0x18, 0xa6, 0xd3, 0xbe, 0xda, 0x50, 0x23],
        );
        let mut dxgi = std::ptr::null();
        let qi = slot::<Query>(d3d, 0);
        assert_eq!(qi(d3d, &newest_pc, &mut dxgi), E_NOINTERFACE);
        assert!(dxgi.is_null());
        assert_eq!(qi(d3d, &DEVICE_IIDS[2], &mut dxgi), S_OK);
        assert_eq!(dxgi, device());

        let mut adapter = std::ptr::null();
        assert_eq!(slot::<Get>(dxgi, 0x38)(dxgi, &mut adapter), S_OK);
        let mut description = std::mem::MaybeUninit::<AdapterDescription>::uninit();
        assert_eq!(
            slot::<Describe>(adapter, 0x40)(adapter, description.as_mut_ptr()),
            S_OK
        );
        // SAFETY: written above.
        assert_eq!(unsafe { description.assume_init() }.vendor, AMD);

        let mut factory = std::ptr::null();
        let parent = slot::<Query>(adapter, 0x30);
        assert_eq!(parent(adapter, &FACTORY_IIDS[2], &mut factory), S_OK);
        let mut again = std::ptr::null();
        assert_eq!(slot::<Indexed>(factory, 0x38)(factory, 0, &mut again), S_OK);
        assert_eq!(again, adapter);

        let mut output = std::ptr::null();
        let outputs = slot::<Indexed>(adapter, 0x38);
        assert_eq!(outputs(adapter, 0, &mut output), S_OK);
        assert!(!output.is_null());
        assert_eq!(outputs(adapter, 1, &mut output), NOT_FOUND);
    }

    #[test]
    fn display_mode_parses_presets_wxh_and_console_defaults() {
        use crate::console::Console::{SeriesS, SeriesX};

        assert_eq!(
            resolve_display_mode(None, None, SeriesX),
            DisplayMode {
                width: 3840,
                height: 2160,
                allow_120hz: false,
            }
        );
        assert_eq!(
            resolve_display_mode(None, None, SeriesS),
            DisplayMode {
                width: 1920,
                height: 1080,
                allow_120hz: false,
            }
        );
        for (input, expected) in [
            ("1920x1080", (1920, 1080)),
            ("2560x1440", (2560, 1440)),
            ("1600x900", (1600, 900)),
            ("1280x720", (1280, 720)),
            ("3840X2160", (3840, 2160)),
            (" 1080p ", (1920, 1080)),
            ("1440P", (2560, 1440)),
            ("900p", (1600, 900)),
            ("720p", (1280, 720)),
            ("4k", (3840, 2160)),
            ("native", (3840, 2160)),
        ] {
            let resolved = resolve_display_mode(Some(input), None, SeriesX);
            assert_eq!((resolved.width, resolved.height), expected, "input={input}");
            assert!(!resolved.allow_120hz);
        }

        // Invalid or out-of-range values fall back cleanly to Native 4K on Series X.
        for bad in ["", "invalid", "320x240", "8000x4000", "1920x1080@999", "1920x"] {
            assert_eq!(
                resolve_display_mode(Some(bad), None, SeriesX),
                DisplayMode {
                    width: 3840,
                    height: 2160,
                    allow_120hz: false,
                },
                "bad={bad}"
            );
        }

        // Refresh rate overrides via @Hz or VICTORY_FPS_CAP.
        assert!(resolve_display_mode(Some("1920x1080@120"), None, SeriesX).allow_120hz);
        assert!(resolve_display_mode(Some("1080p@120Hz"), None, SeriesX).allow_120hz);
        assert!(!resolve_display_mode(Some("1920x1080@60"), Some("0"), SeriesX).allow_120hz);
        assert!(resolve_display_mode(None, Some("0"), SeriesX).allow_120hz);
        assert!(resolve_display_mode(None, Some("unlimited"), SeriesX).allow_120hz);
        assert!(resolve_display_mode(None, Some("120"), SeriesX).allow_120hz);
        assert!(!resolve_display_mode(None, Some("60"), SeriesX).allow_120hz);
    }

    #[test]
    fn modes_enumeration_reports_60hz_and_optional_120hz() {
        assert_eq!(
            modes_into(
                DisplayMode {
                    width: 1920,
                    height: 1080,
                    allow_120hz: false,
                },
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
            E_INVALIDARG
        );

        let mode_60 = DisplayMode {
            width: 1920,
            height: 1080,
            allow_120hz: false,
        };
        let mut count = 0u32;
        assert_eq!(modes_into(mode_60, &mut count, std::ptr::null_mut()), S_OK);
        assert_eq!(count, 1);

        let mut single = [Mode {
            width: 0,
            height: 0,
            refresh: [0, 0],
            format: 0,
            scanlines: 0,
            scaling: 0,
            flags: 0,
        }];
        count = 0;
        assert_eq!(modes_into(mode_60, &mut count, single.as_mut_ptr()), MORE_DATA);
        assert_eq!(count, 0);

        count = 1;
        assert_eq!(modes_into(mode_60, &mut count, single.as_mut_ptr()), S_OK);
        assert_eq!(count, 1);
        assert_eq!((single[0].width, single[0].height, single[0].refresh), (1920, 1080, [60, 1]));

        let mode_120 = DisplayMode {
            width: 2560,
            height: 1440,
            allow_120hz: true,
        };
        assert_eq!(modes_into(mode_120, &mut count, std::ptr::null_mut()), S_OK);
        assert_eq!(count, 2);

        // Requesting fewer slots than available writes what fits and returns MORE_DATA.
        count = 1;
        assert_eq!(modes_into(mode_120, &mut count, single.as_mut_ptr()), MORE_DATA);
        assert_eq!(count, 1);
        assert_eq!((single[0].width, single[0].height, single[0].refresh), (2560, 1440, [60, 1]));

        let mut both = [
            Mode {
                width: 0,
                height: 0,
                refresh: [0, 0],
                format: 0,
                scanlines: 0,
                scaling: 0,
                flags: 0,
            },
            Mode {
                width: 0,
                height: 0,
                refresh: [0, 0],
                format: 0,
                scanlines: 0,
                scaling: 0,
                flags: 0,
            },
        ];
        count = 2;
        assert_eq!(modes_into(mode_120, &mut count, both.as_mut_ptr()), S_OK);
        assert_eq!(count, 2);
        assert_eq!((both[0].width, both[0].height, both[0].refresh), (2560, 1440, [60, 1]));
        assert_eq!((both[1].width, both[1].height, both[1].refresh), (2560, 1440, [120, 1]));
    }
}
