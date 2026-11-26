//! Native Windows HID support for Sony PlayStation gamepads (PS5 DualSense,
//! DualSense Edge, and DualShock 4) over USB and Bluetooth, using only
//! built-in Windows DLLs (`user32.dll`, `kernel32.dll`, `hid.dll`).

use super::gameinput::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

/// Sony Interactive Entertainment USB / Bluetooth HID vendor ID.
pub(super) const SONY_VENDOR_ID: u32 = 0x054C;

/// Supported Sony PlayStation gamepad product IDs.
pub(super) const PID_DUALSHOCK4_V1: u32 = 0x05C4;
pub(super) const PID_DUALSHOCK4_V2: u32 = 0x09CC;
pub(super) const PID_DUALSHOCK4_DONGLE: u32 = 0x0BA0;
pub(super) const PID_DUALSENSE: u32 = 0x0CE6;
pub(super) const PID_DUALSENSE_EDGE: u32 = 0x0DF2;

/// Family of Sony controller determining the HID report layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ControllerKind {
    /// PS5 DualSense (`0x0CE6`) and DualSense Edge (`0x0DF2`).
    DualSense,
    /// PS4 DualShock 4 v1 (`0x05C4`), v2 (`0x09CC`), and USB wireless dongle (`0x0BA0`).
    DualShock4,
}

impl ControllerKind {
    /// Maps a HID `(vendor_id, product_id)` pair to a supported Sony controller.
    pub(super) const fn from_vid_pid(vendor_id: u32, product_id: u32) -> Option<Self> {
        if vendor_id != SONY_VENDOR_ID {
            return None;
        }
        match product_id {
            PID_DUALSENSE | PID_DUALSENSE_EDGE => Some(Self::DualSense),
            PID_DUALSHOCK4_V1 | PID_DUALSHOCK4_V2 | PID_DUALSHOCK4_DONGLE => Some(Self::DualShock4),
            _ => None,
        }
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::DualSense => "DualSense",
            Self::DualShock4 => "DualShock 4",
        }
    }

    #[cfg(windows)]
    const fn usb_output_report_len(self) -> usize {
        match self {
            Self::DualSense => 48,
            Self::DualShock4 => 32,
        }
    }
}

/// Transport link used by an opened Sony HID device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Transport {
    Usb,
    Bluetooth,
}

/// Parses a `VICTORY_HID_GAMEPAD` environment variable value (`"0"` disables HID).
pub(super) fn hid_enabled_env(value: Option<&str>) -> bool {
    !matches!(value.map(str::trim), Some("0"))
}

/// Detects Bluetooth HID device paths on Windows (`BTHENUM` or Bluetooth HID service GUID).
pub(super) fn is_bluetooth_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.contains("bthenum") || lower.contains("{00001124-0000-1000-8000-00805f9b34fb}")
}

/// Normalizes an unsigned 8-bit stick axis (`0..=255`, center `128`) to `-1.0..=1.0`,
/// absorbing tiny `+-2` raw-byte resting ADC chatter around center.
fn stick_x(raw: u8) -> f32 {
    if raw.abs_diff(128) <= 2 {
        return 0.0;
    }
    ((f32::from(raw) - 128.0) / 127.0).clamp(-1.0, 1.0)
}

/// Normalizes an unsigned 8-bit stick Y axis, inverted so `0 -> +1.0` (up) and `255 -> -1.0` (down),
/// absorbing tiny `+-2` raw-byte resting ADC chatter around center.
fn stick_y(raw: u8) -> f32 {
    if raw.abs_diff(128) <= 2 {
        return 0.0;
    }
    ((128.0 - f32::from(raw)) / 127.0).clamp(-1.0, 1.0)
}

fn trigger(raw: u8) -> f32 {
    if raw <= 2 {
        0.0
    } else {
        f32::from(raw) / 255.0
    }
}

fn decode_dpad(hat: u8) -> u32 {
    match hat & 0x0F {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => DOWN | RIGHT,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => UP | LEFT,
        _ => 0,
    }
}

fn decode_buttons(btn0: u8, btn1: u8, btn2: u8) -> u32 {
    let mut buttons = decode_dpad(btn0);
    // PlayStation face buttons mapped positionally to Xbox GameInput buttons.
    if btn0 & 0x10 != 0 {
        buttons |= X; // Square
    }
    if btn0 & 0x20 != 0 {
        buttons |= A; // Cross
    }
    if btn0 & 0x40 != 0 {
        buttons |= B; // Circle
    }
    if btn0 & 0x80 != 0 {
        buttons |= Y; // Triangle
    }
    if btn1 & 0x01 != 0 {
        buttons |= LEFT_SHOULDER; // L1
    }
    if btn1 & 0x02 != 0 {
        buttons |= RIGHT_SHOULDER; // R1
    }
    if btn1 & 0x10 != 0 {
        buttons |= VIEW; // Create / Share
    }
    if btn1 & 0x20 != 0 {
        buttons |= MENU; // Options
    }
    if btn1 & 0x40 != 0 {
        buttons |= LEFT_STICK; // L3
    }
    if btn1 & 0x80 != 0 {
        buttons |= RIGHT_STICK; // R3
    }
    if btn2 & 0x02 != 0 {
        buttons |= VIEW; // Touchpad click also acts as View
    }
    buttons
}

/// Parses a Sony HID input report into a console `PadState`.
///
/// Returns `None` for unrecognized report IDs or truncated packets so callers
/// preserve the previous state.
pub(super) fn parse_report(
    kind: ControllerKind,
    transport: Transport,
    buf: &[u8],
) -> Option<PadState> {
    let &report_id = buf.first()?;
    match kind {
        ControllerKind::DualSense => match report_id {
            // DualSense Bluetooth full report (78 bytes; shifted +1 vs USB 0x01).
            0x31 if buf.len() >= 12 => Some(PadState {
                buttons: decode_buttons(buf[9], buf[10], buf[11]),
                left_trigger: trigger(buf[6]),
                right_trigger: trigger(buf[7]),
                left_x: stick_x(buf[2]),
                left_y: stick_y(buf[3]),
                right_x: stick_x(buf[4]),
                right_y: stick_y(buf[5]),
            }),
            // DualSense Bluetooth simple report (10 bytes on wire, may be zero-padded by HID class
            // driver to 78 bytes over BT): sticks at 1..5, buttons at 5..8, triggers at 8..10.
            0x01 if (transport == Transport::Bluetooth || buf.len() == 10) && buf.len() >= 10 => {
                Some(PadState {
                    buttons: decode_buttons(buf[5], buf[6], buf[7]),
                    left_trigger: trigger(buf[8]),
                    right_trigger: trigger(buf[9]),
                    left_x: stick_x(buf[1]),
                    left_y: stick_y(buf[2]),
                    right_x: stick_x(buf[3]),
                    right_y: stick_y(buf[4]),
                })
            }
            // DualSense USB report (64 bytes): sticks 1..5, triggers 5..7, seq 7, buttons 8..11.
            0x01 if buf.len() >= 11 => Some(PadState {
                buttons: decode_buttons(buf[8], buf[9], buf[10]),
                left_trigger: trigger(buf[5]),
                right_trigger: trigger(buf[6]),
                left_x: stick_x(buf[1]),
                left_y: stick_y(buf[2]),
                right_x: stick_x(buf[3]),
                right_y: stick_y(buf[4]),
            }),
            _ => None,
        },
        ControllerKind::DualShock4 => match report_id {
            // DualShock 4 Bluetooth full report (78 bytes; shifted +2 vs USB 0x01).
            0x11 if buf.len() >= 12 => Some(PadState {
                buttons: decode_buttons(buf[7], buf[8], buf[9]),
                left_trigger: trigger(buf[10]),
                right_trigger: trigger(buf[11]),
                left_x: stick_x(buf[3]),
                left_y: stick_y(buf[4]),
                right_x: stick_x(buf[5]),
                right_y: stick_y(buf[6]),
            }),
            // DualShock 4 USB (64 bytes), dongle (64 bytes), and BT simple (10 bytes).
            0x01 if buf.len() >= 10 => Some(PadState {
                buttons: decode_buttons(buf[5], buf[6], buf[7]),
                left_trigger: trigger(buf[8]),
                right_trigger: trigger(buf[9]),
                left_x: stick_x(buf[1]),
                left_y: stick_y(buf[2]),
                right_x: stick_x(buf[3]),
                right_y: stick_y(buf[4]),
            }),
            _ => None,
        },
    }
}

/// Builds a USB HID rumble output report for DualSense or DualShock 4.
///
/// Returns `None` for Bluetooth devices (where output reports require CRC-32 and
/// firmware-specific flags; rumble stays best-effort).
pub(super) fn build_usb_rumble_report(
    kind: ControllerKind,
    transport: Transport,
    low: u8,
    high: u8,
    output_len: usize,
) -> Option<Vec<u8>> {
    if transport != Transport::Usb {
        return None;
    }
    match kind {
        ControllerKind::DualSense => {
            let len = output_len.max(48);
            let mut buf = vec![0u8; len];
            buf[0] = 0x02; // USB output report ID
            buf[1] = 0x03; // valid_flag0: compatible vibration + haptics select
            buf[2] = 0x00; // valid_flag1: preserve LEDs / mute / player lights
            buf[3] = high; // right / high-frequency motor
            buf[4] = low; // left / low-frequency motor
            Some(buf)
        }
        ControllerKind::DualShock4 => {
            let len = output_len.max(32);
            let mut buf = vec![0u8; len];
            buf[0] = 0x05; // USB output report ID
            buf[1] = 0x01; // enable rumble only; preserve lightbar
            buf[4] = high; // right / weak motor
            buf[5] = low; // left / strong motor
            Some(buf)
        }
    }
}

#[cfg(windows)]
mod win {
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, LazyLock, Mutex, PoisonError};
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GENERIC_WRITE, GetLastError, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

    use super::{
        ControllerKind, PadState, Transport, build_usb_rumble_report, hid_enabled_env,
        is_bluetooth_path, parse_report,
    };
    use crate::modules::gameinput::PADS;

    const RIM_TYPEHID: u32 = 2;
    const RIDI_DEVICENAME: u32 = 0x2000_0007;
    const RIDI_DEVICEINFO: u32 = 0x2000_000B;

    /// Generic Desktop Controls usage page (`0x0001`).
    const HID_USAGE_PAGE_GENERIC: u16 = 0x0001;
    /// Joystick (`0x0004`) and Gamepad (`0x0005`) top-level collections.
    const HID_USAGE_JOYSTICK: u16 = 0x0004;
    const HID_USAGE_GAMEPAD: u16 = 0x0005;

    const HIDP_STATUS_SUCCESS: i32 = 0x0011_0000;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawInputDeviceList {
        h_device: *mut c_void,
        dw_type: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RidDeviceInfoHid {
        vendor_id: u32,
        product_id: u32,
        version_number: u32,
        usage_page: u16,
        usage: u16,
    }

    #[repr(C)]
    union RidDeviceInfoUnion {
        hid: RidDeviceInfoHid,
        /// Matches `RID_DEVICE_INFO_KEYBOARD` (largest union arm, 24 bytes) so
        /// `size_of::<RidDeviceInfo>() == 32` as required by `GetRawInputDeviceInfoW`.
        _keyboard_pad: [u32; 6],
    }

    #[repr(C)]
    struct RidDeviceInfo {
        cb_size: u32,
        dw_type: u32,
        data: RidDeviceInfoUnion,
    }

    const _: () = assert!(size_of::<RidDeviceInfo>() == 32);

    #[repr(C)]
    #[derive(Default)]
    struct HidpCaps {
        usage: u16,
        usage_page: u16,
        input_report_byte_length: u16,
        output_report_byte_length: u16,
        feature_report_byte_length: u16,
        reserved: [u16; 17],
        number_link_collection_nodes: u16,
        number_input_button_caps: u16,
        number_input_value_caps: u16,
        number_input_data_indices: u16,
        number_output_button_caps: u16,
        number_output_value_caps: u16,
        number_output_data_indices: u16,
        number_feature_button_caps: u16,
        number_feature_value_caps: u16,
        number_feature_data_indices: u16,
    }

    const _: () = assert!(size_of::<HidpCaps>() == 64);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn ReadFile(
            file: HANDLE,
            buffer: *mut u8,
            bytes_to_read: u32,
            bytes_read: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
        fn WriteFile(
            file: HANDLE,
            buffer: *const u8,
            bytes_to_write: u32,
            bytes_written: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn GetRawInputDeviceList(
            raw_input_device_list: *mut RawInputDeviceList,
            num_devices: *mut u32,
            size: u32,
        ) -> u32;
        fn GetRawInputDeviceInfoW(
            device: *mut c_void,
            command: u32,
            data: *mut c_void,
            size: *mut u32,
        ) -> u32;
    }

    type HidDGetFeatureFn = unsafe extern "system" fn(HANDLE, *mut c_void, u32) -> u8;
    type HidDGetPreparsedDataFn = unsafe extern "system" fn(HANDLE, *mut *mut c_void) -> u8;
    type HidDFreePreparsedDataFn = unsafe extern "system" fn(*mut c_void) -> u8;
    type HidPGetCapsFn = unsafe extern "system" fn(*mut c_void, *mut HidpCaps) -> i32;

    struct HidApi {
        get_feature: Option<HidDGetFeatureFn>,
        get_preparsed_data: Option<HidDGetPreparsedDataFn>,
        free_preparsed_data: Option<HidDFreePreparsedDataFn>,
        get_caps: Option<HidPGetCapsFn>,
    }

    impl HidApi {
        fn load() -> Self {
            // SAFETY: loading standard system DLL `hid.dll` and resolving known exports.
            unsafe {
                let module = LoadLibraryA(c"hid.dll".as_ptr().cast());
                if module.is_null() {
                    return Self {
                        get_feature: None,
                        get_preparsed_data: None,
                        free_preparsed_data: None,
                        get_caps: None,
                    };
                }
                let get_feature = GetProcAddress(module, c"HidD_GetFeature".as_ptr().cast())
                    .map(|f| std::mem::transmute::<_, HidDGetFeatureFn>(f));
                let get_preparsed_data =
                    GetProcAddress(module, c"HidD_GetPreparsedData".as_ptr().cast())
                        .map(|f| std::mem::transmute::<_, HidDGetPreparsedDataFn>(f));
                let free_preparsed_data =
                    GetProcAddress(module, c"HidD_FreePreparsedData".as_ptr().cast())
                        .map(|f| std::mem::transmute::<_, HidDFreePreparsedDataFn>(f));
                let get_caps = GetProcAddress(module, c"HidP_GetCaps".as_ptr().cast())
                    .map(|f| std::mem::transmute::<_, HidPGetCapsFn>(f));
                Self {
                    get_feature,
                    get_preparsed_data,
                    free_preparsed_data,
                    get_caps,
                }
            }
        }

        fn query_caps(&self, handle: HANDLE) -> Option<HidpCaps> {
            let get_preparsed = self.get_preparsed_data?;
            let free_preparsed = self.free_preparsed_data?;
            let get_caps = self.get_caps?;
            let mut preparsed = std::ptr::null_mut();
            // SAFETY: `handle` is an open synchronous HID handle.
            if unsafe { get_preparsed(handle, &mut preparsed) } == 0 || preparsed.is_null() {
                return None;
            }
            let mut caps = HidpCaps::default();
            // SAFETY: `preparsed` was returned by `HidD_GetPreparsedData` and freed immediately after.
            let status = unsafe { get_caps(preparsed, &mut caps) };
            // SAFETY: paired with `HidD_GetPreparsedData`.
            unsafe { free_preparsed(preparsed) };
            (status == HIDP_STATUS_SUCCESS).then_some(caps)
        }

        fn get_feature(&self, handle: HANDLE, buf: &mut [u8]) -> bool {
            let Some(func) = self.get_feature else {
                return false;
            };
            // SAFETY: `handle` is an open synchronous HID handle and `buf` is writable for `buf.len()`.
            unsafe { func(handle, buf.as_mut_ptr().cast(), buf.len() as u32) != 0 }
        }
    }

    static HID_API: LazyLock<HidApi> = LazyLock::new(HidApi::load);

    /// Owned Win32 `HANDLE` closed automatically on drop.
    struct HidHandle(HANDLE);

    // SAFETY: Win32 file/device handles are process-wide kernel references safe to send across threads.
    unsafe impl Send for HidHandle {}

    impl Drop for HidHandle {
        fn drop(&mut self) {
            if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
                // SAFETY: `self.0` is an owned handle opened via `CreateFileW`.
                unsafe { CloseHandle(self.0) };
            }
        }
    }

    /// A discovered Sony HID gamepad from Raw Input enumeration.
    #[derive(Clone, Debug)]
    pub(super) struct DiscoveredDevice {
        pub(super) kind: ControllerKind,
        pub(super) product_id: u32,
        pub(super) path_wide: Vec<u16>,
        pub(super) path_key: String,
    }

    /// Enumerates all currently connected Sony HID gamepads via Windows Raw Input.
    pub(super) fn enumerate_sony_gamepads() -> Vec<DiscoveredDevice> {
        let entry_size = size_of::<RawInputDeviceList>() as u32;
        let mut count = 0u32;
        // SAFETY: querying device count with null buffer.
        if unsafe { GetRawInputDeviceList(std::ptr::null_mut(), &mut count, entry_size) }
            == u32::MAX
            || count == 0
        {
            return Vec::new();
        }

        let mut list = Vec::new();
        for _ in 0..3 {
            list.resize(
                count as usize,
                RawInputDeviceList {
                    h_device: std::ptr::null_mut(),
                    dw_type: 0,
                },
            );
            // SAFETY: `list` has room for `count` entries of `entry_size` bytes.
            let got = unsafe { GetRawInputDeviceList(list.as_mut_ptr(), &mut count, entry_size) };
            if got != u32::MAX {
                list.truncate(got as usize);
                break;
            }
            if count == 0 {
                return Vec::new();
            }
        }

        let mut found = Vec::new();
        for item in list {
            if item.dw_type != RIM_TYPEHID {
                continue;
            }
            let mut info = RidDeviceInfo {
                cb_size: size_of::<RidDeviceInfo>() as u32,
                dw_type: 0,
                data: RidDeviceInfoUnion {
                    _keyboard_pad: [0; 6],
                },
            };
            let mut info_size = size_of::<RidDeviceInfo>() as u32;
            // SAFETY: `info` is initialized with `cb_size == 32`.
            let ret = unsafe {
                GetRawInputDeviceInfoW(
                    item.h_device,
                    RIDI_DEVICEINFO,
                    (&raw mut info).cast(),
                    &mut info_size,
                )
            };
            if ret == u32::MAX || info.dw_type != RIM_TYPEHID {
                continue;
            }
            // SAFETY: `info.dw_type == RIM_TYPEHID`, so the `hid` union arm is active.
            let hid = unsafe { info.data.hid };
            let Some(kind) = ControllerKind::from_vid_pid(hid.vendor_id, hid.product_id) else {
                continue;
            };
            if hid.usage_page != HID_USAGE_PAGE_GENERIC
                || (hid.usage != HID_USAGE_GAMEPAD && hid.usage != HID_USAGE_JOYSTICK)
            {
                continue;
            }

            let mut name_chars = 0u32;
            // SAFETY: querying device path character count.
            if unsafe {
                GetRawInputDeviceInfoW(
                    item.h_device,
                    RIDI_DEVICENAME,
                    std::ptr::null_mut(),
                    &mut name_chars,
                )
            } == u32::MAX
                || name_chars == 0
            {
                continue;
            }
            let mut path_wide = vec![0u16; name_chars as usize + 1];
            let mut capacity = name_chars;
            // SAFETY: `path_wide` has room for `capacity` UTF-16 code units plus a trailing NUL.
            let written = unsafe {
                GetRawInputDeviceInfoW(
                    item.h_device,
                    RIDI_DEVICENAME,
                    path_wide.as_mut_ptr().cast(),
                    &mut capacity,
                )
            };
            if written == u32::MAX || written == 0 {
                continue;
            }
            let len = path_wide
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(written as usize);
            path_wide.truncate(len);
            if path_wide.is_empty() {
                continue;
            }
            let path_key = String::from_utf16_lossy(&path_wide).to_ascii_lowercase();
            path_wide.push(0); // NUL terminator for `CreateFileW`.

            if !found.iter().any(|d: &DiscoveredDevice| d.path_key == path_key) {
                found.push(DiscoveredDevice {
                    kind,
                    product_id: hid.product_id,
                    path_wide,
                    path_key,
                });
            }
        }
        found
    }

    fn open_hid_handle(path_wide: &[u16], access: u32) -> Option<HidHandle> {
        // SAFETY: `path_wide` is NUL-terminated UTF-16; sharing read+write avoids blocking other apps.
        let handle = unsafe {
            CreateFileW(
                path_wide.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            None
        } else {
            Some(HidHandle(handle))
        }
    }

    struct HidPadState {
        connected: bool,
        pad: PadState,
        prev_buttons: u32,
        latched_buttons: u32,
        #[cfg(test)]
        reports_read: u64,
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    struct HidDevice {
        path_key: String,
        kind: ControllerKind,
        transport: Transport,
        output_report_len: usize,
        slot: Option<usize>,
        shared: Arc<Mutex<HidPadState>>,
        stop: Arc<AtomicBool>,
        write_handle: Option<HidHandle>,
        last_rumble: (u8, u8),
    }

    impl Drop for HidDevice {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    impl HidDevice {
        fn open(dev: &DiscoveredDevice) -> Option<Self> {
            // Try read+write first (confirming write access), falling back to read-only so
            // apps like Steam do not prevent us from reading controller input.
            let (read_handle, can_write) =
                match open_hid_handle(&dev.path_wide, GENERIC_READ | GENERIC_WRITE) {
                    Some(h) => (h, true),
                    None => (open_hid_handle(&dev.path_wide, GENERIC_READ)?, false),
                };

            let caps = HID_API.query_caps(read_handle.0);
            let input_cap = caps.as_ref().map_or(0, |c| usize::from(c.input_report_byte_length));
            let output_cap = caps.as_ref().map_or(0, |c| usize::from(c.output_report_byte_length));
            let feature_cap = caps.as_ref().map_or(0, |c| usize::from(c.feature_report_byte_length));

            let transport = if is_bluetooth_path(&dev.path_key) || input_cap > 64 {
                Transport::Bluetooth
            } else {
                Transport::Usb
            };

            // On Bluetooth, requesting the calibration feature report switches DualSense
            // (0x05) to full 0x31 reports and DualShock 4 (0x02) to full 0x11 reports.
            if transport == Transport::Bluetooth {
                let feature_id = match dev.kind {
                    ControllerKind::DualSense => 0x05u8,
                    ControllerKind::DualShock4 => 0x02u8,
                };
                let mut feat = vec![0u8; feature_cap.max(64)];
                feat[0] = feature_id;
                let ok = HID_API.get_feature(read_handle.0, &mut feat);
                tracing::debug!(
                    device = dev.kind.name(),
                    feature_id = format_args!("{feature_id:#04x}"),
                    ok,
                    "requested Bluetooth HID feature report"
                );
            }

            // Use a dedicated write handle so synchronous `WriteFile` never serializes
            // behind the reader thread's blocking `ReadFile` on `read_handle`.
            let write_handle = if can_write && transport == Transport::Usb {
                open_hid_handle(&dev.path_wide, GENERIC_WRITE)
            } else {
                None
            };

            let read_len = if input_cap > 0 {
                input_cap.max(64)
            } else if transport == Transport::Bluetooth {
                547
            } else {
                64
            };
            let output_report_len = if output_cap > 0 {
                output_cap
            } else {
                dev.kind.usb_output_report_len()
            };

            let shared = Arc::new(Mutex::new(HidPadState {
                connected: true,
                pad: PadState::default(),
                prev_buttons: 0,
                latched_buttons: 0,
                #[cfg(test)]
                reports_read: 0,
            }));
            let stop = Arc::new(AtomicBool::new(false));

            tracing::info!(
                device = dev.kind.name(),
                pid = format_args!("{:#06x}", dev.product_id),
                transport = ?transport,
                input_len = read_len,
                rumble = write_handle.is_some(),
                "Sony HID gamepad connected"
            );

            let thread_shared = Arc::clone(&shared);
            let thread_stop = Arc::clone(&stop);
            let kind = dev.kind;
            let _ = std::thread::Builder::new()
                .name(format!("victory-hid-{}", kind.name()))
                .spawn(move || {
                    reader_loop(read_handle, kind, transport, read_len, thread_shared, thread_stop);
                });

            Some(Self {
                path_key: dev.path_key.clone(),
                kind: dev.kind,
                transport,
                output_report_len,
                slot: None,
                shared,
                stop,
                write_handle,
                last_rumble: (0, 0),
            })
        }

        fn is_connected(&self) -> bool {
            lock(&self.shared).connected
        }

        fn pad(&self) -> Option<PadState> {
            let mut guard = lock(&self.shared);
            if !guard.connected {
                return None;
            }
            let mut pad = guard.pad;
            let latched = std::mem::take(&mut guard.latched_buttons);
            pad.buttons |= latched & !pad.buttons;
            Some(pad)
        }

        fn rumble(&mut self, low: u16, high: u16) {
            let low8 = (low >> 8) as u8;
            let high8 = (high >> 8) as u8;
            if (low8, high8) == self.last_rumble {
                return;
            }
            let Some(ref handle) = self.write_handle else {
                return;
            };
            let Some(report) = build_usb_rumble_report(
                self.kind,
                self.transport,
                low8,
                high8,
                self.output_report_len,
            ) else {
                return;
            };
            let mut written = 0u32;
            // SAFETY: `handle.0` is an open writable HID handle and `report` is readable for `report.len()`.
            let ok = unsafe {
                WriteFile(
                    handle.0,
                    report.as_ptr(),
                    report.len() as u32,
                    &mut written,
                    std::ptr::null_mut(),
                )
            };
            if ok != 0 {
                self.last_rumble = (low8, high8);
            } else {
                // SAFETY: plain thread-local error query.
                let err = unsafe { GetLastError() };
                tracing::debug!(
                    device = self.kind.name(),
                    error = err,
                    "Sony HID rumble write failed; disabling rumble on handle"
                );
                self.write_handle = None;
            }
        }
    }

    fn reader_loop(
        read_handle: HidHandle,
        kind: ControllerKind,
        transport: Transport,
        read_len: usize,
        shared: Arc<Mutex<HidPadState>>,
        stop: Arc<AtomicBool>,
    ) {
        let mut buf = vec![0u8; read_len];
        while !stop.load(Ordering::Relaxed) {
            let mut bytes_read = 0u32;
            // SAFETY: `read_handle.0` is owned by this thread and `buf` has `read_len` bytes.
            let ok = unsafe {
                ReadFile(
                    read_handle.0,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                    &mut bytes_read,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                // SAFETY: plain thread-local error query.
                let err = unsafe { GetLastError() };
                tracing::info!(
                    device = kind.name(),
                    error = err,
                    "Sony HID gamepad disconnected"
                );
                lock(&shared).connected = false;
                return;
            }
            let n = (bytes_read as usize).min(buf.len());
            if let Some(pad) = parse_report(kind, transport, &buf[..n]) {
                let mut guard = lock(&shared);
                let rising = pad.buttons & !guard.prev_buttons;
                guard.latched_buttons |= rising;
                guard.prev_buttons = pad.buttons;
                guard.pad = pad;
                #[cfg(test)]
                {
                    guard.reports_read = guard.reports_read.saturating_add(1);
                }
            }
        }
    }

    /// Manages hot-plugged Sony HID gamepads mapped into free host slots (`0..PADS`).
    pub(crate) struct HidManager {
        enabled: bool,
        next_scan: Option<Instant>,
        devices: Vec<HidDevice>,
        failed_paths: HashSet<String>,
    }

    impl Default for HidManager {
        fn default() -> Self {
            let env_val = std::env::var("VICTORY_HID_GAMEPAD").ok();
            let enabled = hid_enabled_env(env_val.as_deref());
            if !enabled {
                tracing::info!("Sony HID gamepad support disabled via VICTORY_HID_GAMEPAD=0");
            }
            Self {
                enabled,
                next_scan: None,
                devices: Vec::new(),
                failed_paths: HashSet::new(),
            }
        }
    }

    impl HidManager {
        /// Moves any HID pad occupying `slot` to another free slot (used when XInput claims `slot`).
        pub(crate) fn vacate_slot(&mut self, slot: usize, xinput_occupied: &[bool; PADS]) {
            if let Some(dev) = self.devices.iter_mut().find(|d| d.slot == Some(slot)) {
                dev.slot = None;
                self.assign_slots(xinput_occupied);
            }
        }

        /// Reaps disconnected HID devices and rescans Raw Input at most once per `rescan_interval`.
        pub(crate) fn poll(&mut self, xinput_occupied: &[bool; PADS], rescan_interval: Duration) {
            if !self.enabled {
                return;
            }
            self.devices.retain(HidDevice::is_connected);

            let now = Instant::now();
            if self.next_scan.is_none_or(|until| now >= until) {
                self.next_scan = Some(now + rescan_interval);
                self.scan(xinput_occupied);
            } else {
                self.assign_slots(xinput_occupied);
            }
        }

        fn scan(&mut self, xinput_occupied: &[bool; PADS]) {
            let discovered = enumerate_sony_gamepads();
            // Clear remembered open failures for paths that were unplugged.
            self.failed_paths
                .retain(|p| discovered.iter().any(|d| d.path_key == *p));

            for dev in &discovered {
                if self.devices.len() >= PADS {
                    break;
                }
                if self.devices.iter().any(|open| open.path_key == dev.path_key) {
                    continue;
                }
                if let Some(opened) = HidDevice::open(dev) {
                    self.failed_paths.remove(&dev.path_key);
                    self.devices.push(opened);
                } else if self.failed_paths.insert(dev.path_key.clone()) {
                    // SAFETY: plain thread-local error query.
                    let err = unsafe { GetLastError() };
                    tracing::warn!(
                        device = dev.kind.name(),
                        pid = format_args!("{:#06x}", dev.product_id),
                        error = err,
                        "failed to open Sony HID gamepad"
                    );
                }
            }
            self.assign_slots(xinput_occupied);
        }

        fn assign_slots(&mut self, xinput_occupied: &[bool; PADS]) {
            let mut taken = *xinput_occupied;
            for dev in &mut self.devices {
                if let Some(s) = dev.slot {
                    if s < PADS && !taken[s] {
                        taken[s] = true;
                    } else {
                        dev.slot = None;
                    }
                }
            }
            for dev in &mut self.devices {
                if dev.slot.is_none()
                    && let Some(free) = taken.iter().position(|&busy| !busy)
                {
                    dev.slot = Some(free);
                    taken[free] = true;
                }
            }
        }

        /// Returns the latest non-blocking `PadState` for `slot`, if a Sony HID pad occupies it.
        pub(crate) fn pad(&self, slot: usize) -> Option<PadState> {
            self.devices
                .iter()
                .find(|d| d.slot == Some(slot))
                .and_then(HidDevice::pad)
        }

        /// Sends best-effort rumble to the Sony HID pad occupying `slot`, if any.
        pub(crate) fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            if let Some(dev) = self.devices.iter_mut().find(|d| d.slot == Some(slot)) {
                dev.rumble(low, high);
            }
        }

        #[cfg(test)]
        pub(super) fn device_reports(&self) -> Vec<(ControllerKind, Transport, Option<usize>, u64)> {
            self.devices
                .iter()
                .map(|d| (d.kind, d.transport, d.slot, lock(&d.shared).reports_read))
                .collect()
        }
    }
}

#[cfg(windows)]
pub(super) use win::HidManager;

/// Returns `true` if at least one Sony PlayStation HID gamepad (DualSense,
/// DualSense Edge, or DualShock 4) is currently connected to the host (unless
/// disabled via `VICTORY_HID_GAMEPAD=0`).
#[allow(dead_code)]
pub(crate) fn has_sony_gamepad() -> bool {
    #[cfg(windows)]
    {
        let env_val = std::env::var("VICTORY_HID_GAMEPAD").ok();
        if !hid_enabled_env(env_val.as_deref()) {
            return false;
        }
        !win::enumerate_sony_gamepads().is_empty()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn vid_pid_classification() {
        assert_eq!(
            ControllerKind::from_vid_pid(SONY_VENDOR_ID, PID_DUALSENSE),
            Some(ControllerKind::DualSense)
        );
        assert_eq!(
            ControllerKind::from_vid_pid(SONY_VENDOR_ID, PID_DUALSENSE_EDGE),
            Some(ControllerKind::DualSense)
        );
        assert_eq!(
            ControllerKind::from_vid_pid(SONY_VENDOR_ID, PID_DUALSHOCK4_V1),
            Some(ControllerKind::DualShock4)
        );
        assert_eq!(
            ControllerKind::from_vid_pid(SONY_VENDOR_ID, PID_DUALSHOCK4_V2),
            Some(ControllerKind::DualShock4)
        );
        assert_eq!(
            ControllerKind::from_vid_pid(SONY_VENDOR_ID, PID_DUALSHOCK4_DONGLE),
            Some(ControllerKind::DualShock4)
        );
        assert_eq!(ControllerKind::from_vid_pid(0x045E, 0x028E), None);
        assert_eq!(ControllerKind::from_vid_pid(SONY_VENDOR_ID, 0x1234), None);
    }

    #[test]
    fn env_opt_out_and_bt_path_detection() {
        assert!(hid_enabled_env(None));
        assert!(hid_enabled_env(Some("1")));
        assert!(hid_enabled_env(Some("")));
        assert!(!hid_enabled_env(Some("0")));
        assert!(!hid_enabled_env(Some(" 0 ")));

        assert!(is_bluetooth_path(
            r"\\?\HID#{00001124-0000-1000-8000-00805f9b34fb}_VID&0002054c_PID&0ce6#8&1234"
        ));
        assert!(is_bluetooth_path(r"\\?\BTHENUM\Dev_1234"));
        assert!(!is_bluetooth_path(
            r"\\?\HID#VID_054C&PID_0CE6&MI_03#7&1234&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}"
        ));
    }

    #[test]
    fn dualsense_usb_neutral_sticks_triggers_and_all_buttons() {
        // Resting 64-byte USB report: sticks centered at 0x80, neutral D-pad 0x08.
        let mut buf = [0u8; 64];
        buf[0] = 0x01;
        buf[1] = 0x80;
        buf[2] = 0x80;
        buf[3] = 0x80;
        buf[4] = 0x80;
        buf[8] = 0x08;
        let neutral = parse_report(ControllerKind::DualSense, Transport::Usb, &buf).unwrap();
        assert_eq!(neutral, PadState::default());

        // Full sticks (with Y inversion: 0 -> +1.0 up, 255 -> -1.0 down), triggers, and all buttons.
        buf[1] = 0x00; // LX full left -> -1.0
        buf[2] = 0x00; // LY full up -> +1.0
        buf[3] = 0xFF; // RX full right -> +1.0
        buf[4] = 0xFF; // RY full down -> -1.0
        buf[5] = 0xFF; // L2 = 1.0
        buf[6] = 0x7F; // R2 ~ 0.498
        buf[8] = 0x08 | 0x10 | 0x20 | 0x40 | 0x80; // Square, Cross, Circle, Triangle
        buf[9] = 0x01 | 0x02 | 0x10 | 0x20 | 0x40 | 0x80; // L1, R1, Create, Options, L3, R3
        buf[10] = 0x00;
        let state = parse_report(ControllerKind::DualSense, Transport::Usb, &buf).unwrap();
        assert_eq!(
            state.buttons,
            X | A | B | Y | LEFT_SHOULDER | RIGHT_SHOULDER | VIEW | MENU | LEFT_STICK | RIGHT_STICK
        );
        assert_eq!(state.left_x, -1.0);
        assert_eq!(state.left_y, 1.0);
        assert_eq!(state.right_x, 1.0);
        assert_eq!(state.right_y, -1.0);
        assert_eq!(state.left_trigger, 1.0);
        assert!(approx(state.right_trigger, 127.0 / 255.0));

        // Touchpad click alone maps to VIEW.
        let mut touch = [0u8; 64];
        touch[0] = 0x01;
        touch[1..5].fill(0x80);
        touch[8] = 0x08;
        touch[10] = 0x02;
        let s = parse_report(ControllerKind::DualSense, Transport::Usb, &touch).unwrap();
        assert_eq!(s.buttons, VIEW);
    }

    #[test]
    fn dualsense_dpad_all_eight_directions_and_neutral() {
        let expected = [
            (0, UP),
            (1, UP | RIGHT),
            (2, RIGHT),
            (3, DOWN | RIGHT),
            (4, DOWN),
            (5, DOWN | LEFT),
            (6, LEFT),
            (7, UP | LEFT),
            (8, 0),
            (15, 0),
        ];
        for (hat, want) in expected {
            let mut buf = [0u8; 64];
            buf[0] = 0x01;
            buf[1..5].fill(0x80);
            buf[8] = hat;
            let state = parse_report(ControllerKind::DualSense, Transport::Usb, &buf).unwrap();
            assert_eq!(state.buttons, want, "hat={hat}");
        }
    }

    #[test]
    fn dualsense_bluetooth_full_0x31_report() {
        let mut buf = [0u8; 78];
        buf[0] = 0x31;
        buf[1] = 0x02; // BT header
        buf[2] = 0xFF; // LX +1.0
        buf[3] = 0x00; // LY +1.0 (up)
        buf[4] = 0x00; // RX -1.0
        buf[5] = 0xFF; // RY -1.0 (down)
        buf[6] = 0x80; // L2
        buf[7] = 0xFF; // R2
        buf[9] = 0x02 | 0x20 | 0x80; // D-pad East + Cross (A) + Triangle (Y)
        buf[10] = 0x01 | 0x20 | 0x40; // L1 + Options (MENU) + L3
        buf[11] = 0x02; // Touchpad click -> VIEW

        let state = parse_report(ControllerKind::DualSense, Transport::Bluetooth, &buf).unwrap();
        assert_eq!(
            state.buttons,
            RIGHT | A | Y | LEFT_SHOULDER | MENU | LEFT_STICK | VIEW
        );
        assert_eq!(state.left_x, 1.0);
        assert_eq!(state.left_y, 1.0);
        assert_eq!(state.right_x, -1.0);
        assert_eq!(state.right_y, -1.0);
        assert!(approx(state.left_trigger, 128.0 / 255.0));
        assert_eq!(state.right_trigger, 1.0);
    }

    #[test]
    fn dualsense_bluetooth_simple_0x01_report_unpadded_and_padded() {
        // 10-byte simple BT report.
        let simple = [
            0x01,
            0x00, // LX -1.0
            0xFF, // LY -1.0 (down)
            0x80, // RX 0.0
            0x00, // RY +1.0 (up)
            0x06 | 0x40, // D-pad West + Circle (B)
            0x02 | 0x10 | 0x80, // R1 + Create (VIEW) + R3
            0x00,
            0xFF, // L2 1.0
            0x40, // R2
        ];
        let state = parse_report(ControllerKind::DualSense, Transport::Bluetooth, &simple).unwrap();
        assert_eq!(state.buttons, LEFT | B | RIGHT_SHOULDER | VIEW | RIGHT_STICK);
        assert_eq!(state.left_x, -1.0);
        assert_eq!(state.left_y, -1.0);
        assert_eq!(state.right_x, 0.0);
        assert_eq!(state.right_y, 1.0);
        assert_eq!(state.left_trigger, 1.0);
        assert!(approx(state.right_trigger, 64.0 / 255.0));

        // Same report zero-padded by Windows HID class driver to 78 bytes over Bluetooth.
        let mut padded = [0u8; 78];
        padded[..simple.len()].copy_from_slice(&simple);
        let state_padded =
            parse_report(ControllerKind::DualSense, Transport::Bluetooth, &padded).unwrap();
        assert_eq!(state_padded, state);

        // Even if transport is Usb, an exact 10-byte 0x01 packet unambiguously uses simple layout.
        let state_len = parse_report(ControllerKind::DualSense, Transport::Usb, &simple).unwrap();
        assert_eq!(state_len, state);
    }

    #[test]
    fn dualshock4_usb_0x01_and_bt_0x11_reports() {
        // DS4 USB 0x01 (64 bytes).
        let mut usb = [0u8; 64];
        usb[0] = 0x01;
        usb[1] = 0xFF; // LX +1.0
        usb[2] = 0x00; // LY +1.0
        usb[3] = 0x80; // RX 0.0
        usb[4] = 0x80; // RY 0.0
        usb[5] = 0x04 | 0x10; // D-pad South + Square (X)
        usb[6] = 0x01 | 0x20; // L1 + Options (MENU)
        usb[7] = 0x02; // Touchpad click -> VIEW
        usb[8] = 0xFF; // L2
        usb[9] = 0x00; // R2
        let s_usb = parse_report(ControllerKind::DualShock4, Transport::Usb, &usb).unwrap();
        assert_eq!(s_usb.buttons, DOWN | X | LEFT_SHOULDER | MENU | VIEW);
        assert_eq!(s_usb.left_x, 1.0);
        assert_eq!(s_usb.left_y, 1.0);
        assert_eq!(s_usb.left_trigger, 1.0);
        assert_eq!(s_usb.right_trigger, 0.0);

        // DS4 Bluetooth full 0x11 (78 bytes; shifted +2 bytes vs USB 0x01).
        let mut bt = [0u8; 78];
        bt[0] = 0x11;
        bt[1] = 0xC0;
        bt[2] = 0x00;
        bt[3] = 0x00; // LX -1.0
        bt[4] = 0xFF; // LY -1.0
        bt[5] = 0xFF; // RX +1.0
        bt[6] = 0x00; // RY +1.0
        bt[7] = 0x07 | 0x20 | 0x40; // NW + Cross (A) + Circle (B)
        bt[8] = 0x02 | 0x10 | 0x40 | 0x80; // R1 + Share (VIEW) + L3 + R3
        bt[9] = 0x00;
        bt[10] = 0x40; // L2
        bt[11] = 0xFF; // R2
        let s_bt = parse_report(ControllerKind::DualShock4, Transport::Bluetooth, &bt).unwrap();
        assert_eq!(
            s_bt.buttons,
            UP | LEFT | A | B | RIGHT_SHOULDER | VIEW | LEFT_STICK | RIGHT_STICK
        );
        assert_eq!(s_bt.left_x, -1.0);
        assert_eq!(s_bt.left_y, -1.0);
        assert_eq!(s_bt.right_x, 1.0);
        assert_eq!(s_bt.right_y, 1.0);
        assert!(approx(s_bt.left_trigger, 64.0 / 255.0));
        assert_eq!(s_bt.right_trigger, 1.0);
    }

    #[test]
    fn unrecognized_or_truncated_reports_return_none() {
        assert_eq!(parse_report(ControllerKind::DualSense, Transport::Usb, &[]), None);
        assert_eq!(
            parse_report(ControllerKind::DualSense, Transport::Usb, &[0x01; 9]),
            None
        );
        assert_eq!(
            parse_report(ControllerKind::DualSense, Transport::Bluetooth, &[0x31; 11]),
            None
        );
        assert_eq!(
            parse_report(ControllerKind::DualSense, Transport::Usb, &[0x05; 64]),
            None
        );
        assert_eq!(
            parse_report(ControllerKind::DualShock4, Transport::Bluetooth, &[0x11; 11]),
            None
        );
        assert_eq!(
            parse_report(ControllerKind::DualShock4, Transport::Usb, &[0x02; 64]),
            None
        );
    }

    #[test]
    fn usb_rumble_report_layout() {
        let ds5 = build_usb_rumble_report(ControllerKind::DualSense, Transport::Usb, 0xAA, 0x55, 48)
            .unwrap();
        assert_eq!(ds5.len(), 48);
        assert_eq!(&ds5[..5], &[0x02, 0x03, 0x00, 0x55, 0xAA]);

        let ds4 = build_usb_rumble_report(ControllerKind::DualShock4, Transport::Usb, 0xCC, 0x33, 32)
            .unwrap();
        assert_eq!(ds4.len(), 32);
        assert_eq!(&ds4[..6], &[0x05, 0x01, 0x00, 0x00, 0x33, 0xCC]);

        assert!(
            build_usb_rumble_report(ControllerKind::DualSense, Transport::Bluetooth, 0xFF, 0xFF, 78)
                .is_none()
        );
    }

    #[cfg(windows)]
    #[test]
    fn live_windows_sony_hid_enumeration_smoke_test() {
        let found = win::enumerate_sony_gamepads();
        for d in &found {
            eprintln!(
                "enumerated Sony HID gamepad: kind={:?} pid={:#06x} path={}",
                d.kind, d.product_id, d.path_key
            );
        }

        let mut mgr = HidManager::default();
        let xinput_occupied = [false; crate::modules::gameinput::PADS];
        mgr.poll(&xinput_occupied, std::time::Duration::from_secs(1));

        if !found.is_empty() {
            // Give the background USB/BT reader thread a brief window to receive live reports.
            std::thread::sleep(std::time::Duration::from_millis(50));
            let info = mgr.device_reports();
            eprintln!("opened Sony HID device(s): {info:?}");
            assert!(!info.is_empty(), "detected Sony HID pad should open cleanly");
            let state = mgr
                .pad(0)
                .expect("slot 0 should expose the connected Sony HID pad");
            eprintln!("live slot 0 PadState: {state:?}");
            assert!((-1.0..=1.0).contains(&state.left_x));
            assert!((-1.0..=1.0).contains(&state.left_y));
            assert!((-1.0..=1.0).contains(&state.right_x));
            assert!((-1.0..=1.0).contains(&state.right_y));
            assert!((0.0..=1.0).contains(&state.left_trigger));
            assert!((0.0..=1.0).contains(&state.right_trigger));
        }
    }
}
