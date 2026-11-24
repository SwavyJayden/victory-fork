//! Explorer launcher menu and drag-and-drop game target resolution.
//!
//! Keeps `comctl32.dll` and `comdlg32.dll` out of `victory.exe`'s static
//! import table by loading them dynamically only when the interactive menu
//! runs, and tears down its activation context before any title PE loads.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const CLI_VERBS: &[&str] = &[
    "inspect",
    "census",
    "lib-exports",
    "shader-info",
    "translate",
    "run",
    "config",
    "help",
];

/// Returns true when a single argument is a dropped file/folder path rather
/// than a CLI subcommand or flag.
pub(crate) fn should_treat_as_drop_target(arg: &OsStr) -> bool {
    let Some(s) = arg.to_str() else {
        return Path::new(arg).exists();
    };
    if s.starts_with('-') || CLI_VERBS.iter().any(|v| v.eq_ignore_ascii_case(s)) {
        return false;
    }
    Path::new(arg).exists()
}

fn strip_xml_comments(xml: &str) -> String {
    let mut out = String::with_capacity(xml.len());
    let mut rest = xml;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 4..];
        if let Some(end) = after.find("-->") {
            rest = &after[end + 3..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

/// Extracts `Name="..."` from an `<Executable ...>` element in a
/// `MicrosoftGame.config` manifest, ignoring XML comments.
fn parse_manifest_executable(xml: &str) -> Option<String> {
    let clean = strip_xml_comments(xml);
    let lower = clean.to_ascii_lowercase();
    let mut search = 0usize;
    while let Some(rel) = lower[search..].find("<executable") {
        let idx = search + rel;
        let next_ch = clean.as_bytes().get(idx + "<executable".len()).copied();
        let is_boundary = matches!(next_ch, Some(b' ' | b'\t' | b'\r' | b'\n' | b'>' | b'/'));
        if is_boundary && let Some(tag_end) = clean[idx..].find('>') {
            let tag = &clean[idx..idx + tag_end];
            if let Some(name) = extract_xml_attr(tag, "Name") {
                return Some(name);
            }
        }
        search = idx + 1;
    }
    None
}

fn extract_xml_attr(tag: &str, attr_name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = format!("{}=", attr_name.to_ascii_lowercase());
    let mut pos = 0usize;
    while let Some(rel) = lower[pos..].find(&needle) {
        let idx = pos + rel;
        let prev = idx
            .checked_sub(1)
            .and_then(|i| tag.as_bytes().get(i).copied());
        if matches!(prev, Some(b' ' | b'\t' | b'\r' | b'\n')) {
            let val_part = tag[idx + needle.len()..].trim_start();
            let quote = val_part.chars().next()?;
            if quote == '"' || quote == '\'' {
                let inner = &val_part[quote.len_utf8()..];
                if let Some(end_q) = inner.find(quote) {
                    let candidate = inner[..end_q].trim();
                    if !candidate.is_empty() {
                        return Some(candidate.to_owned());
                    }
                }
            }
        }
        pos = idx + needle.len();
    }
    None
}

fn find_largest_exe_in_dir(dir: &Path) -> Result<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_exe = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("exe"));
        if !is_exe {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if stem.starts_with("unitycrashhandler") || stem.starts_with("unins") {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if best.as_ref().is_none_or(|(best_len, _)| size > *best_len) {
            best = Some((size, path));
        }
    }
    best.map(|(_, p)| p)
        .with_context(|| format!("no game executable found inside {}", dir.display()))
}

fn resolve_from_manifest(manifest: &Path) -> Result<PathBuf> {
    let dir = manifest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if let Ok(xml) = std::fs::read_to_string(manifest)
        && let Some(exe_rel) = parse_manifest_executable(&xml)
    {
        let candidate = dir.join(exe_rel.replace('\\', "/"));
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    find_largest_exe_in_dir(dir)
}

fn is_runner_executable(candidate: &Path) -> bool {
    if let Ok(current) = std::env::current_exe()
        && let (Ok(c_canon), Ok(e_canon)) = (
            std::fs::canonicalize(candidate),
            std::fs::canonicalize(current),
        )
        && c_canon == e_canon
    {
        return true;
    }
    candidate
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| {
            n.eq_ignore_ascii_case("victory.exe")
                || n.eq_ignore_ascii_case("gta6rt.exe")
                || n.eq_ignore_ascii_case("xcl.exe")
        })
}

/// Resolves a user-picked `.exe`, `MicrosoftGame.config`, or folder into the
/// executable path to run.
pub fn resolve_game_exe(target: &Path) -> Result<PathBuf> {
    let resolved = if target.is_file() {
        let is_manifest = target
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.eq_ignore_ascii_case("MicrosoftGame.config"));
        if is_manifest {
            resolve_from_manifest(target)?
        } else {
            target.to_path_buf()
        }
    } else if target.is_dir() {
        let manifest = target.join("MicrosoftGame.config");
        if manifest.is_file() {
            resolve_from_manifest(&manifest)?
        } else {
            find_largest_exe_in_dir(target)?
        }
    } else {
        anyhow::bail!("target does not exist: {}", target.display())
    };
    if is_runner_executable(&resolved) {
        anyhow::bail!(
            "{} is a compatibility runner, not a game executable. Pick your unpacked Xbox Series game's .exe or MicrosoftGame.config instead.",
            resolved.display()
        );
    }
    Ok(resolved)
}

#[cfg(windows)]
mod win {
    use std::path::{Path, PathBuf};

    use anyhow::Result;

    use super::resolve_game_exe;

    const DEFAULT_PERF_ENV: &[(&str, &str)] = &[
        ("VICTORY_LATE_WAIT", "3"),
        ("VICTORY_WIDE_BACK", "1"),
        ("VICTORY_CARD_BUFFERS_MIN_MB", "1"),
        ("VICTORY_CARD_BUFFERS_MB", "1024"),
        ("VICTORY_PIPELINE_WAIT_MS", "24"),
    ];

    fn wide_nul(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub(super) fn is_standalone_console() -> bool {
        let mut pids = [0u32; 4];
        // SAFETY: `pids` is a valid mutable buffer of 4 `u32` elements.
        unsafe {
            windows_sys::Win32::System::Console::GetConsoleProcessList(
                pids.as_mut_ptr(),
                pids.len() as u32,
            ) == 1
        }
    }

    /// Dynamically resolves a procedure from a Windows system DLL so GUI dialog
    /// libraries stay absent from `victory.exe`'s static imports.
    unsafe fn load_system_proc(dll: &str, symbol: &str) -> Option<*const core::ffi::c_void> {
        use windows_sys::Win32::System::LibraryLoader::{
            GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
        };
        let w_dll = wide_nul(dll);
        // SAFETY: `w_dll` is NUL-terminated; restricting search to System32 avoids DLL planting.
        let module = unsafe {
            LoadLibraryExW(
                w_dll.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if module.is_null() {
            return None;
        }
        let c_sym = format!("{symbol}\0");
        // SAFETY: `module` is non-null and `c_sym` is NUL-terminated ASCII.
        unsafe { GetProcAddress(module, c_sym.as_ptr()) }.map(|fp| fp as *const core::ffi::c_void)
    }

    struct DialogContext {
        act_ctx: Option<(windows_sys::Win32::Foundation::HANDLE, usize)>,
        com_initialized: bool,
    }

    impl DialogContext {
        fn enter() -> Self {
            use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows_sys::Win32::System::ApplicationInstallationAndServicing::{
                ACTCTXW, ActivateActCtx, CreateActCtxW,
            };
            use windows_sys::Win32::System::Com::{COINIT_APARTMENTTHREADED, CoInitializeEx};

            const V6_MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency><dependentAssembly>
    <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
  </dependentAssembly></dependency>
</assembly>
"#;
            // SAFETY: Initializes STA COM on the main UI thread prior to shell dialogs.
            let hr = unsafe { CoInitializeEx(std::ptr::null(), COINIT_APARTMENTTHREADED as u32) };
            let com_initialized = hr >= 0;

            let manifest_path = std::env::temp_dir().join("victory-ui-v6.manifest");
            let act_ctx = std::fs::write(&manifest_path, V6_MANIFEST)
                .ok()
                .and_then(|()| {
                    let w_path = wide_nul(&manifest_path.to_string_lossy());
                    let mut ctx: ACTCTXW = unsafe { std::mem::zeroed() };
                    ctx.cbSize = std::mem::size_of::<ACTCTXW>() as u32;
                    ctx.lpSource = w_path.as_ptr();
                    // SAFETY: `ctx` is initialized with `cbSize` and NUL-terminated `lpSource`.
                    let handle = unsafe { CreateActCtxW(&ctx) };
                    if handle == INVALID_HANDLE_VALUE {
                        return None;
                    }
                    let mut cookie = 0usize;
                    // SAFETY: `handle` is valid and `cookie` points to stack storage.
                    if unsafe { ActivateActCtx(handle, &mut cookie) } != 0 {
                        Some((handle, cookie))
                    } else {
                        unsafe {
                            windows_sys::Win32::System::ApplicationInstallationAndServicing::ReleaseActCtx(handle);
                        }
                        None
                    }
                });

            Self {
                act_ctx,
                com_initialized,
            }
        }
    }

    impl Drop for DialogContext {
        fn drop(&mut self) {
            use windows_sys::Win32::System::ApplicationInstallationAndServicing::{
                DeactivateActCtx, ReleaseActCtx,
            };
            use windows_sys::Win32::System::Com::CoUninitialize;

            if let Some((handle, cookie)) = self.act_ctx.take() {
                // SAFETY: Matches the successful `ActivateActCtx`/`CreateActCtxW` pair.
                unsafe {
                    let _ = DeactivateActCtx(0, cookie);
                    ReleaseActCtx(handle);
                }
            }
            if self.com_initialized {
                // SAFETY: Paired with `CoInitializeEx` on the same thread.
                unsafe { CoUninitialize() };
            }
        }
    }

    enum MenuChoice {
        Again,
        Choose,
        ShowConfig,
        Cancel,
        DirectFilePicker,
    }

    fn show_task_menu(last_or_configured: Option<&Path>) -> MenuChoice {
        use windows_sys::Win32::System::Console::GetConsoleWindow;
        use windows_sys::Win32::UI::Controls::{
            TASKDIALOG_BUTTON, TASKDIALOGCONFIG, TDCBF_CANCEL_BUTTON,
            TDF_ALLOW_DIALOG_CANCELLATION, TDF_POSITION_RELATIVE_TO_WINDOW, TDF_USE_COMMAND_LINKS,
        };

        const BTN_AGAIN: i32 = 1001;
        const BTN_CHOOSE: i32 = 1002;
        const BTN_CONFIG: i32 = 1003;

        type FnTaskDialogIndirect =
            unsafe extern "system" fn(*const TASKDIALOGCONFIG, *mut i32, *mut i32, *mut i32) -> i32;

        // SAFETY: Lookup resolves `comctl32.dll!TaskDialogIndirect` from System32.
        let Some(proc) = (unsafe { load_system_proc("comctl32.dll", "TaskDialogIndirect") }) else {
            return MenuChoice::DirectFilePicker;
        };

        let title = wide_nul("VIctory");
        let heading = wide_nul("Run an Xbox Series title");
        let content = wide_nul(
            "Pick the unpacked game's executable or MicrosoftGame.config.\nYour game folder is never modified.",
        );
        let again_text =
            last_or_configured.map(|p| wide_nul(&format!("Run again\n{}", p.display())));
        let choose_text =
            wide_nul("Choose a game...\nPick the game's .exe or MicrosoftGame.config");
        let config_text = wide_nul("Configuration...\nShow cache, save data and config paths");

        let mut buttons = Vec::new();
        if let Some(ref txt) = again_text {
            buttons.push(TASKDIALOG_BUTTON {
                nButtonID: BTN_AGAIN,
                pszButtonText: txt.as_ptr(),
            });
        }
        buttons.push(TASKDIALOG_BUTTON {
            nButtonID: BTN_CHOOSE,
            pszButtonText: choose_text.as_ptr(),
        });
        buttons.push(TASKDIALOG_BUTTON {
            nButtonID: BTN_CONFIG,
            pszButtonText: config_text.as_ptr(),
        });

        // SAFETY: POD Win32 struct zero-initialized before setting all required fields.
        let mut cfg: TASKDIALOGCONFIG = unsafe { std::mem::zeroed() };
        cfg.cbSize = std::mem::size_of::<TASKDIALOGCONFIG>() as u32;
        cfg.hwndParent = unsafe { GetConsoleWindow() };
        cfg.dwFlags =
            TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
        cfg.dwCommonButtons = TDCBF_CANCEL_BUTTON;
        cfg.pszWindowTitle = title.as_ptr();
        cfg.pszMainInstruction = heading.as_ptr();
        cfg.pszContent = content.as_ptr();
        cfg.cButtons = buttons.len() as u32;
        cfg.pButtons = buttons.as_ptr();
        cfg.nDefaultButton = if again_text.is_some() {
            BTN_AGAIN
        } else {
            BTN_CHOOSE
        };

        let mut pressed = 0i32;
        // SAFETY: Signature matches Win32 `TaskDialogIndirect`; all wide strings stay alive for the call.
        let hr = unsafe {
            let func: FnTaskDialogIndirect = std::mem::transmute(proc);
            func(
                &cfg,
                &mut pressed,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };

        match (hr, pressed) {
            (h, _) if h < 0 => MenuChoice::DirectFilePicker,
            (_, BTN_AGAIN) => MenuChoice::Again,
            (_, BTN_CHOOSE) => MenuChoice::Choose,
            (_, BTN_CONFIG) => MenuChoice::ShowConfig,
            _ => MenuChoice::Cancel,
        }
    }

    fn pick_game_file(initial_dir: Option<&Path>) -> Option<PathBuf> {
        use windows_sys::Win32::System::Console::GetConsoleWindow;
        use windows_sys::Win32::UI::Controls::Dialogs::{
            OFN_EXPLORER, OFN_FILEMUSTEXIST, OFN_NOCHANGEDIR, OFN_PATHMUSTEXIST, OPENFILENAMEW,
        };

        type FnGetOpenFileNameW = unsafe extern "system" fn(*mut OPENFILENAMEW) -> i32;

        // SAFETY: Lookup resolves `comdlg32.dll!GetOpenFileNameW` from System32.
        let proc = unsafe { load_system_proc("comdlg32.dll", "GetOpenFileNameW") }?;

        let filter: Vec<u16> =
            "Xbox Series title (*.exe; MicrosoftGame.config)\0*.exe;MicrosoftGame.config\0All files (*.*)\0*.*\0\0"
                .encode_utf16()
                .collect();
        let title = wide_nul("VIctory - Choose a game");
        let init_dir = initial_dir
            .filter(|d| d.is_dir())
            .map(|d| wide_nul(&d.to_string_lossy()));
        let mut path_buf = vec![0u16; 32_768];

        // SAFETY: POD Win32 struct zero-initialized before populating valid pointers.
        let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
        ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
        ofn.hwndOwner = unsafe { GetConsoleWindow() };
        ofn.lpstrFilter = filter.as_ptr();
        ofn.lpstrFile = path_buf.as_mut_ptr();
        ofn.nMaxFile = path_buf.len() as u32;
        ofn.lpstrInitialDir = init_dir.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        ofn.lpstrTitle = title.as_ptr();
        ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

        // SAFETY: Signature matches Win32 `GetOpenFileNameW`; buffers outlive the modal call.
        let ok = unsafe {
            let func: FnGetOpenFileNameW = std::mem::transmute(proc);
            func(&mut ofn)
        };
        if ok == 0 {
            return None;
        }
        let len = path_buf
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(path_buf.len());
        Some(PathBuf::from(String::from_utf16_lossy(&path_buf[..len])))
    }

    fn show_config_box(cfg: &victory_core::Config) {
        use windows_sys::Win32::System::Console::GetConsoleWindow;
        use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONINFORMATION, MB_OK, MessageBoxW};

        let body = format!(
            "config file : {}\ngame_dir    : {}\ncache_dir   : {}\ndata_dir    : {}\n\nDrop unpacked game files into game_dir or choose an executable from the menu.",
            cfg.source
                .as_ref()
                .map_or("(none)".into(), |p| p.display().to_string()),
            cfg.game_dir
                .as_ref()
                .map_or("(not set)".into(), |p| p.display().to_string()),
            cfg.cache_dir.display(),
            cfg.data_dir.display(),
        );
        let w_body = wide_nul(&body);
        let w_title = wide_nul("VIctory Configuration");
        // SAFETY: Both strings are NUL-terminated UTF-16 buffers.
        unsafe {
            let _ = MessageBoxW(
                GetConsoleWindow(),
                w_body.as_ptr(),
                w_title.as_ptr(),
                MB_OK | MB_ICONINFORMATION,
            );
        }
    }

    fn apply_default_perf_env() {
        for &(key, val) in DEFAULT_PERF_ENV {
            if std::env::var_os(key).is_none() {
                // SAFETY: Called on the single main thread before any worker threads or title code start.
                unsafe { std::env::set_var(key, val) };
            }
        }
    }

    fn load_last_or_configured_exe(cfg: &victory_core::Config) -> Option<PathBuf> {
        let last_file = cfg.data_dir.join("last-game.txt");
        if let Ok(saved) = std::fs::read_to_string(&last_file) {
            let trimmed = PathBuf::from(saved.trim());
            if let Ok(resolved) = resolve_game_exe(&trimmed)
                && resolved.is_file()
            {
                return Some(resolved);
            }
        }
        if let Some(ref game_dir) = cfg.game_dir
            && let Ok(resolved) = resolve_game_exe(game_dir)
            && resolved.is_file()
        {
            return Some(resolved);
        }
        None
    }

    fn pause_if_standalone(always_pause: bool) {
        if always_pause || is_standalone_console() {
            println!("\nPress Enter to close.");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
    }

    fn execute_target(target: &Path, pause_on_exit: bool) -> i32 {
        let exe = match resolve_game_exe(target) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("\nVIctory could not resolve {}: {err:#}", target.display());
                pause_if_standalone(pause_on_exit);
                return 1;
            }
        };

        if let Ok(cfg) = victory_core::Config::load() {
            let _ = std::fs::create_dir_all(&cfg.data_dir);
            let _ = std::fs::write(
                cfg.data_dir.join("last-game.txt"),
                exe.to_string_lossy().as_bytes(),
            );
        }

        apply_default_perf_env();
        println!("victory run \"{}\" --execute", exe.display());
        let outcome: Result<()> = crate::run::run(
            &exe,
            crate::Unimplemented::Trap,
            false,
            true,
            crate::Console::SeriesX,
        );
        match outcome {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("\nVIctory stopped: {err:#}");
                pause_if_standalone(pause_on_exit);
                1
            }
        }
    }

    pub(crate) fn run_interactive() -> i32 {
        let cfg = match victory_core::Config::load() {
            Ok(c) => c,
            Err(err) => {
                eprintln!("VIctory config error: {err:#}");
                pause_if_standalone(true);
                return 1;
            }
        };
        let last_exe = load_last_or_configured_exe(&cfg);

        let selected: Option<PathBuf> = {
            let _ui = DialogContext::enter();
            loop {
                match show_task_menu(last_exe.as_deref()) {
                    MenuChoice::Again => break last_exe.clone(),
                    MenuChoice::Choose | MenuChoice::DirectFilePicker => {
                        let initial = last_exe
                            .as_deref()
                            .and_then(|p| p.parent())
                            .or(cfg.game_dir.as_deref());
                        if let Some(picked) = pick_game_file(initial) {
                            break Some(picked);
                        }
                        if last_exe.is_none() {
                            // Keep looping on cancel? No: file-picker cancel exits cleanly if user cancelledpicker directly.
                        }
                        break None;
                    }
                    MenuChoice::ShowConfig => show_config_box(&cfg),
                    MenuChoice::Cancel => break None,
                }
            }
        };

        let Some(target) = selected else {
            return 0;
        };
        execute_target(&target, true)
    }

    pub(crate) fn run_path_target(target: &Path) -> i32 {
        execute_target(target, false)
    }
}

#[cfg(windows)]
pub(crate) use win::{run_interactive, run_path_target};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_direct_exe_keeps_path() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("game_scarlett_master.exe");
        std::fs::write(&exe, b"MZ").unwrap();
        assert_eq!(resolve_game_exe(&exe).unwrap(), exe);
    }

    #[test]
    fn resolve_manifest_extracts_executable_name() {
        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("game_scarlett_master.exe");
        std::fs::write(&exe, b"MZ").unwrap();
        let manifest = tmp.path().join("MicrosoftGame.config");
        std::fs::write(
            &manifest,
            r#"<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <!-- <Executable Name="ignored_comment.exe" /> -->
  <ExecutableList>
    <Executable Name="game_scarlett_master.exe" TargetDeviceFamily="Scarlett" />
  </ExecutableList>
</Game>
"#,
        )
        .unwrap();
        assert_eq!(resolve_game_exe(&manifest).unwrap(), exe);
        assert_eq!(resolve_game_exe(tmp.path()).unwrap(), exe);
    }

    #[test]
    fn resolve_directory_falls_back_to_largest_exe() {
        let tmp = tempfile::tempdir().unwrap();
        let crash = tmp.path().join("UnityCrashHandler64.exe");
        let small = tmp.path().join("helper.exe");
        let main_exe = tmp.path().join("title.exe");
        std::fs::write(&crash, vec![0u8; 4096]).unwrap();
        std::fs::write(&small, vec![0u8; 64]).unwrap();
        std::fs::write(&main_exe, vec![0u8; 512]).unwrap();
        assert_eq!(resolve_game_exe(tmp.path()).unwrap(), main_exe);
    }

    #[test]
    fn resolve_empty_directory_fails_cleanly() {
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_game_exe(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("no game executable found"));
    }

    #[test]
    fn drop_target_ignores_subcommands_and_flags() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!should_treat_as_drop_target(OsStr::new("run")));
        assert!(!should_treat_as_drop_target(OsStr::new("--help")));
        assert!(!should_treat_as_drop_target(OsStr::new("-V")));
        assert!(should_treat_as_drop_target(tmp.path().as_os_str()));
    }

    #[test]
    fn resolve_rejects_runner_executables() {
        let tmp = tempfile::tempdir().unwrap();
        let runner = tmp.path().join("victory.exe");
        std::fs::write(&runner, b"MZ").unwrap();
        let err = resolve_game_exe(&runner).unwrap_err();
        assert!(err.to_string().contains("compatibility runner"));
    }
}
