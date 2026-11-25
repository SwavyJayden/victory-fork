//! The graphics driver's errors, seen as the system records them.
//!
//! A card is not lost out of nothing. Every loss of runs 201, 204, 208 and
//! 224 has entries of the card's driver in the system's event log (NVIDIA:
//! provider `nvlddmkm`, event 153, "Error occurred on GPUID") from 2.7 to
//! 6.3 seconds BEFORE the device was removed, several of them; run 210 has
//! three and kept its device; the runs with no such entry lost nothing.
//! Nothing in the layer saw them: the first it knew was the device gone.
//!
//! Here the layer subscribes to that log. Each entry of a display driver
//! is written to our log when the system records it, with our own clock,
//! so it can be laid beside what the layer did in that moment; it is
//! counted; and the first one trips a switch ([`tripped`]) that the paths
//! under suspicion ask before they run (the draw windows of `crate::draw`
//! turn themselves off for the rest of the run). So a driver error costs a
//! slower frame from then on, and not the card.
//!
//! On by default; `VICTORY_DRIVER_WATCH=0` subscribes to nothing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError};
use windows_sys::Win32::System::EventLog::{
    EVT_HANDLE, EVT_SUBSCRIBE_NOTIFY_ACTION, EvtRender, EvtRenderEventXml, EvtSubscribe,
    EvtSubscribeActionDeliver, EvtSubscribeActionError, EvtSubscribeToFutureEvents,
};

/// XPath query for display-driver errors, warnings, and critical events in
/// the System event log. Explicitly includes NVIDIA graphics fault IDs
/// (`13` FECS / graphics exception, `14` recovery action / PF FLR, `153`
/// GPUID / GpuRcReset TDR) and Windows Display TDR (`4101`) so they match
/// regardless of event level, while filtering out benign info/verbose
/// provider chatter.
const QUERY: &str = "*[System[Provider[@Name='nvlddmkm' or @Name='amdkmdag' or @Name='amdwddmg' \
    or @Name='igfx' or @Name='igfxn' or @Name='Display'] and \
    ((Level=0 or Level<=3) or EventID=13 or EventID=14 or EventID=153 or EventID=4101)]]";

/// Entries of a display driver seen since the start.
static ERRORS: AtomicU64 = AtomicU64::new(0);
/// Whether one was seen ([`tripped`]).
static TRIPPED: AtomicBool = AtomicBool::new(false);

/// Parsed summary of a delivered Windows Event Log record.
#[derive(Debug, Default, PartialEq, Eq)]
struct EventInfo {
    provider: String,
    event_id: Option<u32>,
    level: Option<u32>,
    detail: String,
}

/// Whether the card's driver has reported an error in this run: what could
/// have caused it stops for the rest of the run.
pub(crate) fn tripped() -> bool {
    TRIPPED.load(Relaxed)
}

/// Driver errors seen since the start.
pub(crate) fn errors() -> u64 {
    ERRORS.load(Relaxed)
}

/// Extracts an attribute value (`attr='...'` or `attr="..."`) from `xml`.
fn xml_attr<'a>(xml: &'a str, tag: &str, attr: &str) -> Option<&'a str> {
    let start = xml.find(tag)?;
    let rest = &xml[start + tag.len()..];
    let tag_end = rest.find('>')?;
    let head = &rest[..tag_end];
    for quote in ['\'', '"'] {
        let needle = format!("{attr}={quote}");
        if let Some(pos) = head.find(&needle) {
            let val = &head[pos + needle.len()..];
            if let Some(end) = val.find(quote) {
                return Some(&val[..end]);
            }
        }
    }
    None
}

/// Extracts the inner text of the first `<tag ...>text</tag>` element.
fn xml_tag_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open_prefix = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut rest = xml;
    while let Some(pos) = rest.find(&open_prefix) {
        let after = &rest[pos + open_prefix.len()..];
        let next = after.as_bytes().first().copied();
        if matches!(next, Some(b'>' | b' ' | b'\t' | b'\r' | b'\n')) {
            let gt = after.find('>')?;
            let inner = &after[gt + 1..];
            let end = inner.find(&close)?;
            return Some(inner[..end].trim());
        }
        rest = after;
    }
    None
}

/// Parses provider, event ID, level, and `<Data>` payload strings from
/// `EvtRenderEventXml` output.
fn parse_event_xml(xml: &str) -> EventInfo {
    let provider = xml_attr(xml, "<Provider", "Name").unwrap_or("").to_owned();
    let event_id = xml_tag_text(xml, "EventID").and_then(|s| s.parse::<u32>().ok());
    let level = xml_tag_text(xml, "Level").and_then(|s| s.parse::<u32>().ok());
    let mut parts = Vec::new();
    let mut rest = xml;
    while let Some(pos) = rest.find("<Data") {
        let after = &rest[pos + 5..];
        let Some(gt) = after.find('>') else {
            break;
        };
        let head = &after[..gt];
        if head.ends_with('/') {
            rest = &after[gt + 1..];
            continue;
        }
        let body = &after[gt + 1..];
        let Some(end) = body.find("</Data>") else {
            break;
        };
        let value = body[..end].trim();
        if !value.is_empty() {
            parts.push(value);
        }
        rest = &body[end + 7..];
    }
    let mut detail = parts.join(" | ");
    if detail.len() > 240 {
        let mut cut = 240;
        while cut > 0 && !detail.is_char_boundary(cut) {
            cut -= 1;
        }
        detail.truncate(cut);
        detail.push_str("...");
    }
    EventInfo {
        provider,
        event_id,
        level,
        detail,
    }
}

/// Whether `info` describes a severe GPU fault / reset escalation (NVIDIA
/// FECS exception 13, recovery action / PF FLR 14, Display TDR 4101, or
/// fatal detail markers).
fn is_fatal_event(info: &EventInfo) -> bool {
    if matches!(info.event_id, Some(13 | 14 | 4101)) {
        return true;
    }
    info.detail.contains("PF FLR")
        || info.detail.contains("UCODE Fatal")
        || info.detail.contains("FECS Exception")
        || info.detail.contains("GpuRcReset")
}

/// Renders a delivered event handle to XML using `EvtRender`, if available.
fn render_event_xml(event: EVT_HANDLE) -> Option<String> {
    if event == 0 {
        return None;
    }
    const MAX_BYTES: u32 = 32 * 1024;
    let mut buf = vec![0u16; 2048];
    let mut used = 0u32;
    let mut count = 0u32;
    // SAFETY: `event` is a valid handle during the `EvtSubscribe` callback;
    // `buf` is writable for `buf.len() * 2` bytes.
    let mut ok = unsafe {
        EvtRender(
            0,
            event,
            EvtRenderEventXml,
            (buf.len() * 2) as u32,
            buf.as_mut_ptr().cast(),
            &raw mut used,
            &raw mut count,
        )
    };
    if ok == 0 {
        // SAFETY: reads thread-local Win32 error code set by `EvtRender`.
        let err = unsafe { GetLastError() };
        if err == ERROR_INSUFFICIENT_BUFFER && used > 0 && used <= MAX_BYTES {
            buf.resize((used as usize).div_ceil(2), 0);
            // SAFETY: resized buffer holds at least `used` bytes.
            ok = unsafe {
                EvtRender(
                    0,
                    event,
                    EvtRenderEventXml,
                    (buf.len() * 2) as u32,
                    buf.as_mut_ptr().cast(),
                    &raw mut used,
                    &raw mut count,
                )
            };
        }
    }
    if ok == 0 || used == 0 {
        return None;
    }
    let units = ((used as usize) / 2).min(buf.len());
    let slice = &buf[..units];
    let len = slice.iter().position(|&c| c == 0).unwrap_or(slice.len());
    Some(String::from_utf16_lossy(&slice[..len]))
}

/// An entry of a display driver was recorded.
fn seen_info(info: Option<&EventInfo>) {
    let n = ERRORS.fetch_add(1, Relaxed) + 1;
    let first = !TRIPPED.swap(true, Relaxed);
    let fatal = info.is_some_and(is_fatal_event);
    let provider = info.map_or("unknown", |i| i.provider.as_str());
    let event_id = info.and_then(|i| i.event_id);
    let level = info.and_then(|i| i.level);
    let detail = info.map_or("", |i| i.detail.as_str());
    tracing::error!(
        n,
        provider,
        event_id,
        level,
        fatal,
        detail,
        "the graphics driver reported an error to the system's event log{}",
        if first {
            ": from now on the paths under suspicion are off for this run (draw windows clamp each record again; BVH walks held)"
        } else {
            ""
        }
    );
}

#[cfg(test)]
fn seen() {
    seen_info(None);
}

unsafe extern "system" fn on_event(action: EVT_SUBSCRIBE_NOTIFY_ACTION, _context: *const core::ffi::c_void, event: EVT_HANDLE) -> u32 {
    if action == EvtSubscribeActionDeliver {
        let parsed = render_event_xml(event).map(|xml| parse_event_xml(&xml));
        seen_info(parsed.as_ref());
    } else if action == EvtSubscribeActionError {
        tracing::warn!(
            error_code = event as usize,
            "system event log driver watch subscription reported a callback error"
        );
    }
    0
}

/// Subscribes to the display drivers' entries in the system's event log,
/// once. Entries from before the call are not delivered.
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("VICTORY_DRIVER_WATCH").is_some_and(|v| v == "0") {
            return;
        }
        let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let channel = wide("System");
        // The kernel drivers of the three vendors, and the system's own
        // "display driver stopped responding" source.
        let query = wide(QUERY);
        // SAFETY: valid strings for the call's length; the callback is a
        // plain function that touches only statics; the subscription is
        // kept for the life of the process.
        let subscription = unsafe {
            EvtSubscribe(
                0,
                std::ptr::null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                std::ptr::null(),
                Some(on_event),
                EvtSubscribeToFutureEvents,
            )
        };
        if subscription == 0 {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "no subscription to the system's event log: the graphics driver's errors are not seen as they happen"
            );
        } else {
            tracing::info!(
                "the graphics driver's errors in the system's event log are watched: each is logged here as it is recorded, and \
                 the first turns the paths under suspicion off (VICTORY_DRIVER_WATCH=0: not watched)"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_error_trips_the_switch_and_every_one_is_counted() {
        // (Other tests do not touch these statics.)
        assert!(!tripped());
        assert_eq!(errors(), 0);
        seen();
        assert!(tripped());
        seen();
        assert_eq!(errors(), 2);
        assert!(tripped());
    }

    /// End to end, by hand (`--ignored`): an entry written to the
    /// Application log under a name of our own reaches a subscription made
    /// the same way as [`start`]'s, within a second. It leaves one
    /// information entry in that log.
    #[test]
    #[ignore = "writes one entry to the Application event log: run it by name with --ignored"]
    fn an_entry_written_reaches_a_subscription() {
        use windows_sys::Win32::System::EventLog::{
            DeregisterEventSource, EVENTLOG_INFORMATION_TYPE, EvtClose, RegisterEventSourceW, ReportEventW,
        };
        static GOT: AtomicU64 = AtomicU64::new(0);
        unsafe extern "system" fn got(action: EVT_SUBSCRIBE_NOTIFY_ACTION, _: *const core::ffi::c_void, _: EVT_HANDLE) -> u32 {
            if action == EvtSubscribeActionDeliver {
                GOT.fetch_add(1, Relaxed);
            }
            0
        }
        let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let (channel, query) = (wide("Application"), wide("*[System[Provider[@Name='victory-drvwatch-test']]]"));
        // SAFETY: as in `start`.
        let subscription = unsafe {
            EvtSubscribe(0, std::ptr::null_mut(), channel.as_ptr(), query.as_ptr(), 0, std::ptr::null(), Some(got), EvtSubscribeToFutureEvents)
        };
        assert_ne!(subscription, 0, "{}", std::io::Error::last_os_error());
        let (name, text) = (wide("victory-drvwatch-test"), wide("a test entry of the VIctory layer's driver watch"));
        // SAFETY: a source of our own name, one string, no data.
        unsafe {
            let source = RegisterEventSourceW(std::ptr::null(), name.as_ptr());
            assert!(!source.is_null(), "{}", std::io::Error::last_os_error());
            let strings = [text.as_ptr()];
            assert_ne!(
                ReportEventW(source, EVENTLOG_INFORMATION_TYPE, 0, 1, std::ptr::null_mut(), 1, 0, strings.as_ptr(), std::ptr::null()),
                0,
                "{}",
                std::io::Error::last_os_error()
            );
            DeregisterEventSource(source);
        }
        let started = std::time::Instant::now();
        while GOT.load(Relaxed) == 0 && started.elapsed() < std::time::Duration::from_secs(5) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let after = started.elapsed();
        // SAFETY: our own subscription.
        unsafe { EvtClose(subscription) };
        assert_eq!(GOT.load(Relaxed), 1, "the entry was not delivered in 5 s");
        eprintln!("delivered after {} ms", after.as_millis());
    }

    /// The subscription itself: the system accepts the channel and the
    /// query (a query it cannot read is refused at once).
    #[test]
    fn the_system_takes_the_subscription() {
        let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let (channel, query) = (wide("System"), wide(QUERY));
        // SAFETY: as in `start`.
        let subscription = unsafe {
            EvtSubscribe(0, std::ptr::null_mut(), channel.as_ptr(), query.as_ptr(), 0, std::ptr::null(), Some(on_event), EvtSubscribeToFutureEvents)
        };
        assert_ne!(subscription, 0, "{}", std::io::Error::last_os_error());
        // SAFETY: our own subscription.
        unsafe { windows_sys::Win32::System::EventLog::EvtClose(subscription) };
    }

    #[test]
    fn query_includes_nvidia_and_display_fault_ids() {
        for needle in ["nvlddmkm", "Display", "EventID=13", "EventID=14", "EventID=153", "EventID=4101"] {
            assert!(QUERY.contains(needle), "QUERY missing {needle}");
        }
    }

    #[test]
    fn parses_nvlddmkm_153_13_14_and_display_4101_xml() {
        let e153 = "<Event><System><Provider Name='nvlddmkm'/><EventID Qualifiers='0'>153</EventID><Level>2</Level></System>\
            <EventData><Data>\\Device\\Video3</Data><Data>Error occurred on GPUID: 700</Data></EventData></Event>";
        let info = parse_event_xml(e153);
        assert_eq!(info.provider, "nvlddmkm");
        assert_eq!(info.event_id, Some(153));
        assert_eq!(info.level, Some(2));
        assert_eq!(info.detail, "\\Device\\Video3 | Error occurred on GPUID: 700");
        assert!(!is_fatal_event(&info));

        let e153_tdr = "<Event><System><Provider Name=\"nvlddmkm\"/><EventID>153</EventID><Level>2</Level></System>\
            <EventData><Data>\\Device\\Video3</Data><Data>GpuRcReset TDR occurred on GPUID:700</Data></EventData></Event>";
        let info = parse_event_xml(e153_tdr);
        assert_eq!(info.event_id, Some(153));
        assert!(is_fatal_event(&info));

        let e13 = "<Event><System><Provider Name='nvlddmkm'/><EventID>13</EventID><Level>2</Level></System>\
            <EventData><Data>\\Device\\Video3</Data><Data>Graphics FECS Exception: UCODE Fatal Error</Data></EventData></Event>";
        let info = parse_event_xml(e13);
        assert_eq!(info.event_id, Some(13));
        assert!(is_fatal_event(&info));

        let e14 = "<Event><System><Provider Name='nvlddmkm'/><EventID>14</EventID><Level>2</Level></System>\
            <EventData><Data>\\Device\\Video3</Data><Data>GPU recovery action changed from 0x0 (None) to 0x1 (PF FLR)</Data></EventData></Event>";
        let info = parse_event_xml(e14);
        assert_eq!(info.event_id, Some(14));
        assert!(is_fatal_event(&info));

        let e4101 = "<Event><System><Provider Name='Display'/><EventID>4101</EventID><Level>3</Level></System>\
            <EventData><Data>nvlddmkm</Data><Data/></EventData></Event>";
        let info = parse_event_xml(e4101);
        assert_eq!(info.provider, "Display");
        assert_eq!(info.event_id, Some(4101));
        assert_eq!(info.level, Some(3));
        assert_eq!(info.detail, "nvlddmkm");
        assert!(is_fatal_event(&info));
    }
}
