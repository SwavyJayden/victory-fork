//! Steps of a submission that read title memory flights in the air may
//! still write, done on the card in the flights' order instead of on the
//! CPU after a wait for them (`VICTORY_CARD_STEPS`, see [`stage`]).

use std::sync::Arc;

use super::*;
use crate::shadow::Shadow;

/// The stage (`VICTORY_CARD_STEPS`; off by default): `1` constant buffers
/// of descriptor tables (`Step::Fill` with a slot of its own), `2` the
/// title's buffer copies (`Step::Copy`) too, with the steps of the same
/// submission that depend on them. Only with the relay's second stage
/// (`relay_exact`).
///
/// Why: eleven dispatches a frame (the skinning passes, in two early
/// submissions) write through a view of the title's whole 174 MB heap, at
/// places they read out of job records in buffers: no bound of what they
/// write is known before they ran (frame capture replay2: 17af15c4015cb4b6,
/// 55679a0547649d0b, 89d08e561897d4e5, fdd56d97a806bcca and three more,
/// all through one view at offset 5920 of the title's bindless table). The
/// title's ring of constants lies in the same heap. One constant buffer of
/// 480 bytes and one copy of 20 KiB a frame, at a new place of the ring
/// each time, waited for these passes to land: 25-30 ms of a 105 ms frame,
/// the submitting thread standing (run record-1; with `VICTORY_FILL_TITLE=1`
/// in run relaycopy-1 the copy alone took over the wait).
///
/// What is done: a list of ours goes to the card in front of the
/// submission's lists, in the same call. It copies the bytes read from the
/// host copy that the flights write into a buffer of ours, and from there
/// to the destination's host copy (a copy), or the fill's descriptor slot
/// is pointed at that buffer (a fill; the slot is the fill's alone, the
/// root signatures say such descriptors are volatile, and the list that
/// names it has not gone to the card).
///
/// Why that is the value the CPU would have read after the wait. After the
/// wait title memory holds, for every byte read, what the title wrote, or
/// what the last of the flights that wrote it wrote (the landings put it
/// there, in order). The host copy on the card holds the same when our list
/// runs: every flight that may write one of these bytes writes this same
/// host copy ([`holder`]: a write through another host copy, a read-back
/// or a value written behind a flight there, and the old wait is taken),
/// they are all ahead of our list on the queue, and the title's own bytes
/// are brought up to date by a refresh step of our list, which is prepared
/// like any other list's (also under flights' declared writes:
/// `stale_under_flights`, which settles if it cannot). Nothing of this
/// submission writes these bytes earlier than our list: the steps are done
/// in front of all its lists in either case, and among themselves in the
/// order they stand in wherever one depends on another: [`plan`] takes
/// every copy and fill of the submission that writes what a taken one
/// reads or writes, or reads what a taken one writes. Every draw that
/// reads a buffer view of the heap relies on the same refresh and the same
/// order; what is new is only that a constant buffer and a copy do too.
/// Where a page is not mapped the host copy may hold other bytes than the
/// CPU's copy would have left: refused.
///
/// For a copy's destination our list declares a shader write, exactly the
/// bytes copied, after a refresh of them: from there on it is what any
/// shader's write is. The relay compares, and puts what changed into title
/// memory before the title is told anything of this flight (its
/// bottom-of-pipe writes and fences come after, in order); CPU steps of
/// later submissions that read it find it declared and wait, or are done
/// here. Refused: a step of the submission that cannot be taken (a texture,
/// vertices expanded, a fill of memory of another kind) reads the
/// destination behind the copy, or a refresh brings another host copy of it
/// up to date ([`plan`], [`elsewhere`]). While such a copy is in the air a
/// submission that refreshes another host copy of its destination, or
/// reads it in a step that is not checked otherwise, waits for it
/// ([`copied_read`]); with the copy done on the CPU it was there at once.
///
/// What stage 2 does not see: code of ours that reads title memory away
/// from a submission's steps (the pass over textures picked by index, while
/// a list is recorded). With the CPU's copy such a read of the destination
/// found the bytes at once; now it finds them when the flight has landed.
/// The landing tells the guards (`crate::watch::landing`), so a texture
/// made from older bytes is made again. This is why stage 2 is its own.
pub(super) fn stage() -> u32 {
    static STAGE: OnceLock<u32> = OnceLock::new();
    *STAGE.get_or_init(|| match std::env::var("VICTORY_CARD_STEPS").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

/// Whether a copy at its place in a list ([`place`]) is only in the way of
/// what is done on the card in front of the lists if that stands behind it in
/// them (`VICTORY_PLACED_ORDER=0` disables; on by default), and a submission
/// where it does is cut behind the copy's list ([`cut_for`]).
///
/// The fills of constant buffers read the title's ring (4,800–7,500 a frame),
/// and one copy of a few hundred bytes a frame writes into it at its place in
/// a list. With this off, the whole submission was refused and stood 40 to
/// 60 ms until the flights had landed (run n19 and Wave 19A session 112814).
fn placed_order() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PLACED_ORDER") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLACED_ORDER").is_none_or(|v| v != "0"))
}

/// Why [`plan`] refuses for [`place`].
const BEHIND_PLACED: &str = "a copy at its place in a list writes what it reads, or reads or writes what it writes";

/// The bytes of our buffer between reading and writing: what one submission
/// may do this way (each place read once, however many fills take it).
/// Sized at 8 MiB so 8-part joined submissions (`3–4+ MiB` of constant fills)
/// never refuse and force a synchronous flight drain.
const BETWEEN: u64 = 8 << 20;

/// The most steps of a submission done this way.
const MOST: usize = 16384;

/// How a fill's bytes get where its list reads them, if the card brings
/// them.
#[derive(Clone)]
pub(super) enum FillTo {
    /// A descriptor slot that shows the fill's memory, and nothing else does
    /// (title heap, slot): it is made to show our buffer instead.
    Slot(usize, u32),
    /// A buffer the card can write, filled from its beginning.
    Buffer(ID3D12Resource),
}

/// Title memory something in the air writes: the host copy it goes through
/// (`None`: none, the bytes reach title memory when the flight lands), from,
/// to, and whether it is the destination of a copy done here.
pub(super) struct Writer {
    shadow: Option<Arc<Shadow>>,
    from: usize,
    to: usize,
    copy: bool,
}

/// What `items` of the line write.
pub(super) fn writers(items: &[Late]) -> Vec<Writer> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Late::Run(flight) => {
                for (shadow, ranges) in &flight.back {
                    out.extend(ranges.iter().map(|&(from, to)| Writer {
                        shadow: Some(shadow.clone()),
                        from,
                        to,
                        copy: false,
                    }));
                }
                for &i in &flight.order {
                    for &(_, at, [row, rows, pitch]) in &flight.lists[i].host().deferred().backs {
                        out.push(Writer {
                            shadow: None,
                            from: at,
                            to: at + rows.saturating_sub(1) * pitch + row,
                            copy: false,
                        });
                    }
                }
                out.extend(flight.own.iter().flat_map(Taken::copied));
            }
            Late::Write(at, _, bytes) => out.push(Writer {
                shadow: None,
                from: *at,
                to: *at + *bytes,
                copy: false,
            }),
            Late::Signal(..) => {}
        }
    }
    out
}

/// The destinations of copies done here that have not landed.
pub(super) fn copies(line: &Line) -> Vec<Writer> {
    if stage() < 2 {
        return Vec::new();
    }
    let flying = line.line.iter().filter_map(|l| match l {
        Late::Run(flight) => flight.own.as_ref(),
        _ => None,
    });
    let landing = line.landing.iter().filter(|w| w.copy).map(|w| Writer {
        shadow: w.shadow.clone(),
        ..*w
    });
    flying.flat_map(Taken::copied).chain(landing).collect()
}

/// Whether `runs` refresh another host copy of a destination in `copies`,
/// or read it in a step that flights are not waited for otherwise
/// (`Step::Host`): that reads title memory, where the copy is not yet.
pub(super) fn copied_read(copies: &[Writer], runs: &[&HostList]) -> bool {
    if copies.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| match step {
            Step::Refresh(shadow, at, len) => copies
                .iter()
                .any(|w| w.from < *at + *len && *at < w.to && w.shadow.as_ref().is_none_or(|s| !Arc::ptr_eq(s, shadow))),
            Step::Host(_, (from, to)) => copies.iter().any(|w| w.from < *to && *from < w.to),
            Step::Plane((at, len), _) => copies.iter().any(|w| w.from < at.saturating_add(*len) && *at < w.to),
            _ => false,
        })
    })
}

/// What a step of a submission does to title memory, as the plan sees it.
/// `asked`: flights in the air stop it from being done on the CPU now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Does {
    Copy { dst: usize, src: usize, len: usize, asked: bool },
    /// `to`: its memory can be given the bytes on the card.
    Fill { src: usize, len: usize, asked: bool, to: bool },
    /// Another read on the CPU (a texture, vertices expanded).
    Read { from: usize, to: usize, asked: bool },
    /// A refresh of a host copy (named by its address).
    Refresh { shadow: usize, from: usize, to: usize },
    /// A copy that its list does on the card ([`place`]).
    Placed { dst: usize, src: usize, len: usize },
    /// Anything, anywhere.
    Run,
}

fn meet(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The bytes `d` reads and writes as a copy or a fill (empty ranges for
/// what it does not).
fn reads_writes(d: &Does) -> Option<((usize, usize), (usize, usize))> {
    match *d {
        Does::Copy { dst, src, len, .. } => Some(((src, src + len), (dst, dst + len))),
        Does::Fill { src, len, .. } => Some(((src, src + len), (0, 0))),
        _ => None,
    }
}

fn touched(ranges: &[(usize, usize)], r: (usize, usize)) -> bool {
    touches(ranges, r.0, r.1)
}

/// The steps of `does` (by place) to be done on the card, in order, each
/// with its level; or why not. All that are asked for, and every copy and
/// fill that is in the way of one that is done there: it writes what that
/// one reads or writes, or reads what it writes, whichever comes first. So
/// what is left on the CPU writes none of their bytes, and reads of them
/// only what a taken copy has not been told to write yet: bytes a step
/// ahead of it reads, in title memory as it is.
///
/// `behind`: for [`BEHIND_PLACED`], the copy at its place and the first step
/// behind it that is in its way.
///
/// Steps of a level are done together, every read before every write; a
/// step that writes what an earlier one reads or writes, or reads what it
/// writes, is of a higher level. Our list does them level by level.
fn plan(does: &[Does], stage: u32, by_order: bool, behind: &mut Option<(usize, usize)>) -> Result<Vec<(usize, u32)>, &'static str> {
    let takes = |d: &Does| match *d {
        Does::Copy { .. } if stage < 2 => Err("a buffer copy (VICTORY_CARD_STEPS=2 does it on the card)"),
        Does::Fill { to: false, .. } => Err("a fill of memory the card cannot be given the bytes of"),
        _ => Ok(()),
    };
    let mut more = Vec::new();
    let mut has_copy = false;
    let mut has_placed = false;
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Read { asked: true, .. } => return Err("a texture reads it"),
            Does::Copy { asked, .. } => {
                has_copy = true;
                if asked {
                    more.push(i);
                }
            }
            Does::Fill { asked: true, .. } => more.push(i),
            Does::Placed { .. } => has_placed = true,
            _ => {}
        }
    }
    if more.is_empty() {
        return Ok(Vec::new());
    }
    // Common fills-only fast path (thousands of constant-buffer fills per frame
    // with no buffer copies or placed copies): fills never write, so nothing
    // can depend on them and every taken fill runs at level 0.
    if !has_copy && !has_placed {
        if more.len() > MOST {
            return Err("too many steps depend on it");
        }
        let mut out = Vec::with_capacity(more.len());
        for i in more {
            takes(&does[i])?;
            out.push((i, 0));
        }
        return Ok(out);
    }
    let mut taken = vec![false; does.len()];
    let mut taken_count = 0usize;
    let (mut read, mut written) = (Vec::new(), Vec::new());
    let (mut reads, mut writes) = (Vec::new(), Vec::new());
    while !more.is_empty() {
        // (Joined once for the round: one by one the 2,800 fills of a frame
        // sorted the ranges each time, 16 ms of the submitting thread.)
        reads.clear();
        writes.clear();
        for i in more.drain(..) {
            takes(&does[i])?;
            taken[i] = true;
            taken_count += 1;
            if let Some((r, w)) = reads_writes(&does[i]) {
                if w.0 < w.1 {
                    if meet(r, w) {
                        return Err("a copy onto itself");
                    }
                    writes.push(w);
                }
                reads.push(r);
            }
        }
        join_ranges(&mut read, reads.drain(..));
        if !writes.is_empty() {
            join_ranges(&mut written, writes.drain(..));
        }
        if taken_count > MOST {
            return Err("too many steps depend on it");
        }
        if !has_copy {
            break;
        }
        let has_written = !written.is_empty();
        for (i, d) in does.iter().enumerate() {
            // A fill that cannot be taken is looked at below.
            if taken[i] || matches!(d, Does::Fill { to: false, .. }) {
                continue;
            }
            if !has_written && !matches!(d, Does::Copy { .. }) {
                continue;
            }
            if let Some((r, w)) = reads_writes(d)
                && ((has_written && (touched(&written, r) || touched(&written, w))) || touched(&read, w))
            {
                more.push(i);
            }
        }
    }
    // A copy at its place in a list comes after what is done in front of all
    // of them: as it should after the steps ahead of it, but not after those
    // behind it that read or write its bytes.
    if has_placed {
        for (p, d) in does.iter().enumerate() {
            if let Does::Placed { dst, src, len } = *d
                && (touched(&read, (dst, dst + len)) || touched(&written, (dst, dst + len)) || touched(&written, (src, src + len)))
            {
                let (to, from) = ((dst, dst + len), (src, src + len));
                let in_the_way = |i: usize| {
                    reads_writes(&does[i]).is_some_and(|(r, w)| meet(r, to) || w.0 < w.1 && (meet(w, to) || meet(w, from)))
                };
                let found = (p + 1..does.len()).find(|&i| taken[i] && in_the_way(i));
                *behind = found.map(|i| (p, i));
                if !by_order || found.is_some() {
                    return Err(BEHIND_PLACED);
                }
            }
        }
    }
    let order: Vec<usize> = (0..does.len()).filter(|&i| taken[i]).collect();
    if written.is_empty() {
        return Ok(order.into_iter().map(|i| (i, 0)).collect());
    }
    // What stays on the CPU and reads what an earlier taken copy writes
    // would read it too early.
    let mut behind_ranges: Vec<(usize, usize)> = Vec::new();
    let mut behind_dirty = false;
    for (i, d) in does.iter().enumerate() {
        if taken[i] {
            if let Some((_, w)) = reads_writes(d)
                && w.0 < w.1
            {
                behind_ranges.push(w);
                behind_dirty = true;
            }
            continue;
        }
        if behind_ranges.is_empty() {
            continue;
        }
        let query = match *d {
            Does::Read { from, to, .. } => Some((from, to, "a texture or expanded vertices read what it writes")),
            Does::Fill { src, len, .. } => {
                Some((src, src + len, "a fill of memory the card cannot be given the bytes of reads what it writes"))
            }
            _ => None,
        };
        if let Some((from, to, err)) = query {
            if behind_dirty {
                join_ranges(&mut behind_ranges, []);
                behind_dirty = false;
            }
            if touched(&behind_ranges, (from, to)) {
                return Err(err);
            }
        }
    }
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(order.len());
    // The copies among them, which alone put others behind them or have to
    // stand behind others.
    let mut copies: Vec<usize> = Vec::new();
    for &i in &order {
        let Some((r, w)) = reads_writes(&does[i]) else {
            continue;
        };
        let writes = w.0 < w.1;
        let mut level = 0;
        // Behind what writes what this reads or writes, and, if this
        // writes, behind what reads it.
        let mut check = |k: usize| {
            let (j, at): (usize, u32) = out[k];
            if let Some((jr, jw)) = reads_writes(&does[j])
                && at >= level
                && (meet(jw, r) || meet(jw, w) || meet(jr, w))
            {
                level = at + 1;
            }
        };
        if writes {
            for k in 0..out.len() {
                check(k);
            }
        } else {
            for &k in &copies {
                check(k);
            }
        }
        if writes {
            copies.push(out.len());
        }
        out.push((i, level));
    }
    Ok(out)
}

/// Whether a refresh of `does` brings a host copy other than `shadow` up to
/// date in `[from, to)`.
fn elsewhere(does: &[Does], shadow: usize, from: usize, to: usize) -> bool {
    does.iter().any(|d| match *d {
        Does::Refresh { shadow: s, from: a, to: b } => s != shadow && meet((a, b), (from, to)),
        _ => false,
    })
}

/// Coalesced interval entry for O(log N) overlapping-writer queries without
/// scanning thousands of in-flight write ranges per fill.
struct WriterEntry {
    shadow: Option<u32>,
    from: usize,
    to: usize,
    max_to: usize,
    /// True iff no other entry in `entries` overlaps `[from, to)`, allowing an
    /// O(1) hot-interval cache hit without binary search or conflict scanning.
    isolated: bool,
}

/// Deduplicated, sorted index of in-flight writers plus any `Does::Copy`
/// writers appended during this submission.
struct WriterIndex {
    shadows: Vec<Arc<Shadow>>,
    entries: Vec<WriterEntry>,
    extra: Vec<(u32, usize, usize)>,
    /// Verified `(from, to, shadow_idx)` from the last isolated hit whose
    /// shadow is non-in-place and covers `[from, to)`.
    last_isolated: Option<(usize, usize, u32)>,
}

impl WriterIndex {
    fn intern(shadows: &mut Vec<Arc<Shadow>>, shadow: &Arc<Shadow>) -> u32 {
        if let Some(pos) = shadows.iter().position(|s| Arc::ptr_eq(s, shadow)) {
            return pos as u32;
        }
        shadows.push(shadow.clone());
        (shadows.len() - 1) as u32
    }

    /// Snapshots in-flight and landing writers under `line`'s lock with one
    /// `Arc::clone` per unique shadow, returning raw intervals to index outside
    /// the lock.
    fn collect_line(line: &Line) -> (Vec<Arc<Shadow>>, Vec<(Option<u32>, usize, usize)>) {
        let mut shadows = Vec::new();
        let mut raw = Vec::new();
        for item in &line.line {
            match item {
                Late::Run(flight) => {
                    for (shadow, ranges) in &flight.back {
                        let id = Some(Self::intern(&mut shadows, shadow));
                        raw.extend(ranges.iter().map(|&(from, to)| (id, from, to)));
                    }
                    for &i in &flight.order {
                        for &(_, at, [row, rows, pitch]) in &flight.lists[i].host().deferred().backs {
                            raw.push((None, at, at + rows.saturating_sub(1) * pitch + row));
                        }
                    }
                    if let Some(own) = &flight.own {
                        for (shadow, from, to) in &own.copied {
                            let id = Some(Self::intern(&mut shadows, shadow));
                            raw.push((id, *from, *to));
                        }
                    }
                }
                Late::Write(at, _, bytes) => raw.push((None, *at, *at + *bytes)),
                Late::Signal(..) => {}
            }
        }
        for w in &line.landing {
            let id = w.shadow.as_ref().map(|s| Self::intern(&mut shadows, s));
            raw.push((id, w.from, w.to));
        }
        (shadows, raw)
    }

    /// Coalesces overlapping/touching intervals with the same shadow and builds
    /// a `from`-sorted prefix-max index with per-entry isolation flags.
    fn build(shadows: Vec<Arc<Shadow>>, mut raw: Vec<(Option<u32>, usize, usize)>) -> Self {
        raw.sort_unstable();
        let mut merged: Vec<(Option<u32>, usize, usize)> = Vec::with_capacity(raw.len());
        for (shadow, from, to) in raw {
            match merged.last_mut() {
                Some(last) if last.0 == shadow && from <= last.2 && last.1 < last.2 && from < to => {
                    last.2 = last.2.max(to);
                }
                Some(last) if *last == (shadow, from, to) => {}
                _ => merged.push((shadow, from, to)),
            }
        }
        merged.sort_unstable_by_key(|e| e.1);
        let mut max_to = 0usize;
        let mut entries: Vec<WriterEntry> = merged
            .into_iter()
            .map(|(shadow, from, to)| {
                let isolated = from >= max_to;
                max_to = max_to.max(to);
                WriterEntry {
                    shadow,
                    from,
                    to,
                    max_to,
                    isolated,
                }
            })
            .collect();
        let mut next_min_from = usize::MAX;
        for entry in entries.iter_mut().rev() {
            if entry.to > next_min_from {
                entry.isolated = false;
            }
            next_min_from = next_min_from.min(entry.from);
        }
        Self {
            shadows,
            entries,
            extra: Vec::new(),
            last_isolated: None,
        }
    }

    #[cfg(test)]
    fn from_writers(writers: &[Writer]) -> Self {
        let mut shadows = Vec::new();
        let mut raw = Vec::with_capacity(writers.len());
        for w in writers {
            let id = w.shadow.as_ref().map(|s| Self::intern(&mut shadows, s));
            raw.push((id, w.from, w.to));
        }
        Self::build(shadows, raw)
    }

    /// Queries overlapping writers for `[from, to)`, returning
    /// `Ok(Some((shadow_idx, isolated_bounds)))` when matched.
    fn query_writers_detail(&self, from: usize, to: usize) -> Result<Option<(u32, Option<(usize, usize)>)>, &'static str> {
        let mut found: Option<u32> = None;
        let mut isolated_bounds: Option<(usize, usize)> = None;
        let hi = self.entries.partition_point(|e| e.from < to);
        for e in self.entries[..hi].iter().rev() {
            if e.max_to <= from {
                break;
            }
            if e.to > from {
                match (e.shadow, found) {
                    (None, _) => return Err("a read-back or a value behind a flight is written there"),
                    (Some(s), Some(f)) if s != f => return Err("flights write it through two host copies"),
                    (Some(s), _) => {
                        if found.is_none() && e.isolated && self.extra.is_empty() && from >= e.from && to <= e.to {
                            isolated_bounds = Some((e.from, e.to));
                        } else {
                            isolated_bounds = None;
                        }
                        found = Some(s);
                    }
                }
            }
        }
        for &(s, w_from, w_to) in &self.extra {
            if meet((w_from, w_to), (from, to)) {
                isolated_bounds = None;
                match found {
                    Some(f) if f != s => return Err("flights write it through two host copies"),
                    _ => found = Some(s),
                }
            }
        }
        Ok(found.map(|s| (s, isolated_bounds)))
    }

    #[cfg(test)]
    fn query_writers(&self, from: usize, to: usize) -> Result<Option<u32>, &'static str> {
        self.query_writers_detail(from, to).map(|hit| hit.map(|(s, _)| s))
    }

    fn holder(&mut self, from: usize, to: usize, fallback_cache: &mut HashMap<(usize, usize), u32>) -> Result<u32, &'static str> {
        if let Some((c_from, c_to, idx)) = self.last_isolated
            && from >= c_from
            && to <= c_to
        {
            return Ok(idx);
        }
        let (idx, isolated_bounds) = match self.query_writers_detail(from, to)? {
            Some(hit) => hit,
            None => match fallback_cache.get(&(from, to - from)) {
                Some(&idx) => return Ok(idx),
                None => {
                    let shadow = crate::shadow::find(from as u64, (to - from) as u64).ok_or("no host copy of it")?;
                    let (start, len) = shadow.span();
                    if shadow.in_place() || (from as u64) < start || to as u64 > start + len {
                        return Err("no host copy holds all of it");
                    }
                    let idx = Self::intern(&mut self.shadows, &shadow);
                    fallback_cache.insert((from, to - from), idx);
                    return Ok(idx);
                }
            },
        };
        let shadow = &self.shadows[idx as usize];
        let (start, len) = shadow.span();
        if shadow.in_place() || (from as u64) < start || to as u64 > start + len {
            return Err("no host copy holds all of it");
        }
        if let Some((e_from, e_to)) = isolated_bounds {
            let c_from = e_from.max(start as usize);
            let c_to = e_to.min((start + len) as usize);
            if c_from < c_to {
                self.last_isolated = Some((c_from, c_to, idx));
            }
        }
        Ok(idx)
    }

    fn push_extra(&mut self, shadow: u32, from: usize, to: usize) {
        self.last_isolated = None;
        self.extra.push((shadow, from, to));
    }
}

/// The host copy that holds `[from, to)` on the card once what `writers`
/// write is there: the one they all write it through, or without any the
/// one shaders would be given.
#[cfg(test)]
fn holder(writers: &[Writer], from: usize, to: usize) -> Result<Arc<Shadow>, &'static str> {
    let mut index = WriterIndex::from_writers(writers);
    let mut cache = HashMap::default();
    let idx = index.holder(from, to, &mut cache)?;
    Ok(index.shadows[idx as usize].clone())
}

/// Populates `does`, `steps`, and (when `with_fills`) `fills` from `runs`
/// without over-reserving for skipped `Step::Refresh` entries.
fn does_of_into(
    written: &[(usize, usize)],
    runs: &[&HostList],
    with_fills: bool,
    does: &mut Vec<Does>,
    steps: &mut Vec<(usize, usize)>,
    fills: &mut Vec<Option<FillTo>>,
) {
    does.clear();
    steps.clear();
    fills.clear();
    // `Does::Refresh` is only inspected by `elsewhere()` for `Does::Copy`
    // steps during `take_steps` (`with_fills == true`); skip pushing them when
    // no list in `runs` contains a `Step::Copy`.
    let keep_refresh = with_fills && runs.iter().any(|list| list.deferred().has_copy);
    for list in runs {
        let ptr = std::ptr::from_ref::<HostList>(list) as usize;
        let deferred = list.deferred();
        for (i, step) in deferred.copies.iter().enumerate() {
            let mut fill_target = None;
            does.push(match step {
                &Step::Copy(dst, src, len) => Does::Copy {
                    dst,
                    src,
                    len,
                    asked: copy_blocked(written, dst, src, len).is_some(),
                },
                Step::Fill(_, src, len, to) => {
                    if with_fills {
                        fill_target = to.clone();
                    }
                    Does::Fill {
                        src: *src,
                        len: *len,
                        asked: read_blocked(written, *src, *len),
                        to: to.is_some(),
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    Does::Read {
                        from: at,
                        to: at + len,
                        asked: touches(written, at, at + len),
                    }
                }
                &Step::Plane((at, len), _) => Does::Read {
                    from: at,
                    to: at.saturating_add(len),
                    asked: touches(written, at, at.saturating_add(len)),
                },
                &Step::Host(_, (from, to)) if from < to => Does::Read { from, to, asked: false },
                Step::Refresh(shadow, at, len) if keep_refresh => Does::Refresh {
                    shadow: Arc::as_ptr(shadow) as usize,
                    from: *at,
                    to: *at + *len,
                },
                Step::Run(_) => Does::Run,
                Step::Placed(p) => Does::Placed { dst: p.dst, src: p.src, len: p.len },
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) => continue,
            });
            steps.push((ptr, i));
            if with_fills {
                fills.push(fill_target);
            }
        }
    }
}

fn does_of(
    written: &[(usize, usize)],
    runs: &[&HostList],
    with_fills: bool,
) -> (Vec<Does>, Vec<(usize, usize)>, Vec<Option<FillTo>>) {
    let mut does = Vec::new();
    let mut steps = Vec::new();
    let mut fills = Vec::new();
    does_of_into(written, runs, with_fills, &mut does, &mut steps, &mut fills);
    (does, steps, fills)
}

/// With flights in the air: before which of `runs` the submission is cut so
/// that what reads or writes the bytes of a copy at its place in an earlier
/// list ([`BEHIND_PLACED`]) is done on the card in front of a submission of
/// its own, behind that copy.
pub(super) fn cut_for(runs: &[&HostList]) -> Option<usize> {
    if stage() == 0 || !relay_exact() || !placed_order() || runs.len() < 2 {
        return None;
    }
    // (Most submissions have none.)
    let first = runs.iter().position(|l| l.deferred().has_placed)?;
    if first + 1 == runs.len() {
        return None;
    }
    let written = line().written.clone();
    blocked_by(&written, runs, false)?;
    let (does, steps, _) = does_of(&written, runs, false);
    let mut behind = None;
    if plan(&does, stage(), true, &mut behind) != Err(BEHIND_PLACED) {
        return None;
    }
    let (placed, reader) = behind?;
    let place = |i: usize| runs.iter().position(|l| std::ptr::from_ref::<HostList>(l) as usize == steps[i].0);
    let cut = place(placed)? + 1;
    (place(reader)? >= cut).then_some(cut)
}

/// Per-submission cache of readable 4 KiB title pages packed into 16-page
/// (64 KiB) bitmasks plus a 1-entry hot-page register so consecutive constant
/// fills in the same 4 KiB page avoid hashing altogether.
struct MappedCache {
    last_page: usize,
    blocks: HashMap<usize, u16>,
}

impl Default for MappedCache {
    fn default() -> Self {
        Self { last_page: usize::MAX, blocks: HashMap::default() }
    }
}

impl MappedCache {
    fn clear(&mut self) {
        self.last_page = usize::MAX;
        self.blocks.clear();
    }

    /// Whether every page of `[from, to)` can be read (through `safecopy` or
    /// the kernel: asking Windows walks the page tables, milliseconds in the
    /// title's heap).
    fn check(&mut self, from: usize, to: usize) -> bool {
        const PAGE: usize = 4096;
        for page in from / PAGE..to.div_ceil(PAGE) {
            if page == self.last_page {
                continue;
            }
            let block = page >> 4;
            let bit = 1u16 << (page & 15);
            if self.blocks.get(&block).is_some_and(|&mask| mask & bit != 0) {
                self.last_page = page;
                continue;
            }
            if !read_into((page * PAGE).max(from) as *const u8, &mut [0]) {
                return false;
            }
            *self.blocks.entry(block).or_insert(0) |= bit;
            self.last_page = page;
        }
        true
    }
}

/// A table by addresses and numbers of ours.
type HashMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<AddressHash>>;

/// A coalesced `CopyBufferRegion` read from one host copy into `own.between`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadSpan {
    shadow: u32,
    level: u32,
    src: usize,
    len: usize,
    at: u64,
}

/// Read-layout request for one `Op` before coalescing into `own.between`.
#[derive(Clone, Copy, Debug)]
struct OpRead {
    shadow: u32,
    src: usize,
    len: usize,
    level: u32,
    /// `To::Slot` constant buffer views require 256-byte alignment in `between`.
    aligned_256: bool,
}

/// Sorts op indices once into `order` for [`layout_reads_sorted`].
fn sort_op_reads(ops: &[OpRead], order: &mut Vec<usize>) {
    order.clear();
    order.extend(0..ops.len());
    order.sort_unstable_by_key(|&i| {
        let o = ops[i];
        (o.level, o.shadow, o.aligned_256 && !o.src.is_multiple_of(256), o.src, o.len)
    });
}

/// Coalesces adjacent, overlapping, and near-gap (`<= max_gap`) reads from a
/// pre-sorted `order` into reusable `spans` / `op_ats` buffers while preserving
/// 256-byte CBV alignment for every `aligned_256` op. Returns `None` if the
/// packed layout exceeds `budget`.
fn layout_reads_sorted(
    ops: &[OpRead],
    order: &[usize],
    budget: u64,
    max_gap: usize,
    spans: &mut Vec<ReadSpan>,
    op_ats: &mut Vec<u64>,
) -> Option<u64> {
    spans.clear();
    op_ats.clear();
    if ops.is_empty() {
        return Some(0);
    }
    op_ats.resize(ops.len(), 0);
    let mut end = 0u64;
    let mut k = 0usize;
    while k < order.len() {
        let first = ops[order[k]];
        let unaligned_slot = first.aligned_256 && !first.src.is_multiple_of(256);
        if unaligned_slot {
            // Rare unaligned CBV source: place at its own 256-byte-aligned
            // offset in `between`, deduplicating exact matches.
            let at = match spans.last() {
                Some(prev)
                    if prev.level == first.level
                        && prev.shadow == first.shadow
                        && prev.src == first.src
                        && prev.len == first.len =>
                {
                    prev.at
                }
                _ => {
                    if end + first.len as u64 > budget {
                        return None;
                    }
                    let at = end;
                    end = (end + first.len as u64).next_multiple_of(256);
                    spans.push(ReadSpan {
                        shadow: first.shadow,
                        level: first.level,
                        src: first.src,
                        len: first.len,
                        at,
                    });
                    at
                }
            };
            op_ats[order[k]] = at;
            k += 1;
            continue;
        }
        // Coalesce a run of normal ops within one `(level, shadow)`.
        let group_start = k;
        let span_src = first.src;
        let mut span_end = first.src.saturating_add(first.len);
        k += 1;
        while k < order.len() {
            let next = ops[order[k]];
            let next_unaligned = next.aligned_256 && !next.src.is_multiple_of(256);
            if next_unaligned
                || next.level != first.level
                || next.shadow != first.shadow
                || next.src > span_end.saturating_add(max_gap)
                || (next.aligned_256 && !span_src.is_multiple_of(256))
            {
                break;
            }
            span_end = span_end.max(next.src.saturating_add(next.len));
            k += 1;
        }
        let span_len = span_end - span_src;
        if end + span_len as u64 > budget {
            return None;
        }
        let at = end;
        end = (end + span_len as u64).next_multiple_of(256);
        spans.push(ReadSpan {
            shadow: first.shadow,
            level: first.level,
            src: span_src,
            len: span_len,
            at,
        });
        for &idx in &order[group_start..k] {
            op_ats[idx] = at + (ops[idx].src - span_src) as u64;
        }
    }
    Some(end)
}

/// Coalesces `ops` trying widening gap thresholds from largest to 0 while
/// sorting `order` only once.
fn layout_reads_cascade(
    ops: &[OpRead],
    budget: u64,
    gaps: &[usize],
    order: &mut Vec<usize>,
    spans: &mut Vec<ReadSpan>,
    op_ats: &mut Vec<u64>,
) -> Option<u64> {
    sort_op_reads(ops, order);
    for &gap in gaps {
        if let Some(end) = layout_reads_sorted(ops, order, budget, gap, spans, op_ats) {
            return Some(end);
        }
    }
    None
}

#[cfg(test)]
fn layout_reads(ops: &[OpRead], budget: u64, max_gap: usize) -> Option<(Vec<ReadSpan>, Vec<u64>, u64)> {
    let mut order = Vec::new();
    let mut spans = Vec::new();
    let mut op_ats = Vec::new();
    sort_op_reads(ops, &mut order);
    let end = layout_reads_sorted(ops, &order, budget, max_gap, &mut spans, &mut op_ats)?;
    Some((spans, op_ats, end))
}

/// A step done on the card.
struct Op {
    from_shadow: u32,
    src: usize,
    len: usize,
    /// Its place in the buffer between.
    at: u64,
    level: u32,
    to: To,
}

enum To {
    /// A host copy (index in `WriterIndex::shadows`), and the title address.
    Copy(u32, usize),
    /// A descriptor slot (title heap, slot).
    Slot(usize, u32),
    /// A buffer's beginning.
    Buffer(ID3D12Resource),
}

/// Records a level of what [`stage`] describes into `list` (open): `reads`
/// are copied into `between` (source, offset in it, offset in `between`,
/// bytes), then `writes` out of it (destination, offset in it, offset in
/// `between`, bytes). `between` is left readable as constants (and must be
/// so, unless this is the `first` level), the others in the common state,
/// as the submission's lists expect them.
fn record(
    first: bool,
    list: &ID3D12GraphicsCommandList,
    between: &ID3D12Resource,
    reads: &[(&ID3D12Resource, u64, u64, u64)],
    writes: &[(&ID3D12Resource, u64, u64, u64)],
) {
    let common = D3D12_RESOURCE_STATE_COMMON;
    let (source, dest) = (D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut sources: Vec<&ID3D12Resource> = Vec::new();
    let mut dests: Vec<&ID3D12Resource> = Vec::new();
    for (all, list) in [(reads, &mut sources), (writes, &mut dests)] {
        for &(buffer, ..) in all {
            if !list.iter().any(|b| b.as_raw() == buffer.as_raw()) {
                list.push(buffer);
            }
        }
    }
    let both = |b: &ID3D12Resource| dests.iter().any(|d| d.as_raw() == b.as_raw());
    let constants = source | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    let mut barriers = vec![transition(between, dest, constants)];
    barriers.extend(sources.iter().map(|b| transition(b, source, if both(b) { dest } else { common })));
    let done: Vec<D3D12_RESOURCE_BARRIER> = dests.iter().map(|b| transition(b, dest, common)).collect();
    // SAFETY: an open list; live buffers, and the callers' ranges lie in
    // them. The buffers start in the common state (the list is the first of
    // its call) and are promoted by their first use.
    unsafe {
        if first {
            // What earlier lists' shaders wrote, first.
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            }]);
        } else {
            list.ResourceBarrier(&[transition(between, constants, dest)]);
        }
        for &(buffer, offset, at, len) in reads {
            list.CopyBufferRegion(between, at, buffer, offset, len);
        }
        list.ResourceBarrier(&barriers);
        for &(buffer, offset, at, len) in writes {
            list.CopyBufferRegion(buffer, offset, between, at, len);
        }
        if !done.is_empty() {
            list.ResourceBarrier(&done);
        }
    }
}

/// The list in front of a submission's, and its buffer ([`BETWEEN`] bytes).
struct Own {
    list: HostList,
    between: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded, and the list is touched by one
// thread at a time (the submitting one until it is sent, then the one that
// lands it, then nobody until it is taken out of `POOL`).
unsafe impl Send for Own {}

/// Those that have landed: closed, their steps dropped.
static POOL: Mutex<Vec<Own>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills and copies done on the card, their
/// bytes, and submissions that waited instead, refused by a check.
static DONE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Per-`HostList` bitset builder for [`Taken::steps`] (replaces a 9,000-entry
/// `HashSet<(usize, usize)>` with O(1) bit tests and zero hashing).
fn build_taken_steps(chosen: &[(usize, u32)], steps: &[(usize, usize)]) -> Vec<(usize, Vec<u64>)> {
    let mut out: Vec<(usize, Vec<u64>)> = Vec::new();
    for &(idx, _) in chosen {
        let (list_ptr, step_i) = steps[idx];
        let word = step_i >> 6;
        let bit = 1u64 << (step_i & 63);
        let bits = match out.last_mut() {
            Some((ptr, bits)) if *ptr == list_ptr => bits,
            _ => match out.iter_mut().find(|(ptr, _)| *ptr == list_ptr) {
                Some((_, bits)) => bits,
                None => {
                    out.push((list_ptr, Vec::new()));
                    &mut out.last_mut().expect("pushed above").1
                }
            },
        };
        if bits.len() <= word {
            bits.resize(word + 1, 0);
        }
        bits[word] |= bit;
    }
    out
}

/// What [`Host::card_steps`] took off a submission: the list that does it,
/// for the submission to run first, and the steps the CPU leaves out (per
/// list address, bitset of deferred step indices).
pub(super) struct Taken {
    own: Own,
    steps: Vec<(usize, Vec<u64>)>,
    copied: Vec<(Arc<Shadow>, usize, usize)>,
}

impl Taken {
    pub(super) fn list(&self) -> &HostList {
        &self.own.list
    }

    /// Returns the bitset slice for `list` once per `HostList` so
    /// `run_deferred` checks each step in 2 instructions with zero hashing.
    #[inline]
    pub(super) fn takes_bits(&self, list: &HostList) -> Option<&[u64]> {
        let ptr = std::ptr::from_ref(list) as usize;
        self.steps.iter().find(|(p, _)| *p == ptr).map(|(_, bits)| bits.as_slice())
    }

    #[inline]
    pub(super) fn bit_taken(bits: Option<&[u64]>, i: usize) -> bool {
        bits.and_then(|b| b.get(i >> 6)).is_some_and(|&w| (w >> (i & 63)) & 1 != 0)
    }

    /// Whether step `i` of `list` is done by this.
    #[inline]
    pub(super) fn takes(&self, list: &HostList, i: usize) -> bool {
        Self::bit_taken(self.takes_bits(list), i)
    }

    fn copied(&self) -> impl Iterator<Item = Writer> {
        self.copied.iter().map(|(shadow, from, to)| Writer {
            shadow: Some(shadow.clone()),
            from: *from,
            to: *to,
            copy: true,
        })
    }

    /// The submission has run and landed: the list and its buffer are free.
    pub(super) fn landed(self) {
        // The host copies its steps held go; its commands stay until the
        // next use resets it.
        drop(std::mem::take(&mut *self.own.list.deferred()));
        POOL.lock().unwrap_or_else(|e| e.into_inner()).push(self.own);
    }
}

/// Reusable thread-local scratch for [`Host::take_steps`] (eliminates ~2.4 MB
/// of per-submission heap allocations on `victory-submit`).
struct CardStepsScratch {
    does: Vec<Does>,
    steps: Vec<(usize, usize)>,
    fills: Vec<Option<FillTo>>,
    ops: Vec<Op>,
    op_reads: Vec<OpRead>,
    order: Vec<usize>,
    read_spans: Vec<ReadSpan>,
    op_ats: Vec<u64>,
    shadow_spans: Vec<Option<(usize, usize)>>,
    buckets: [Vec<(usize, u32, u64, u32)>; 8],
    held_fallback: HashMap<(usize, usize), u32>,
    room: MappedCache,
}

impl Default for CardStepsScratch {
    fn default() -> Self {
        Self {
            does: Vec::new(),
            steps: Vec::new(),
            fills: Vec::new(),
            ops: Vec::new(),
            op_reads: Vec::new(),
            order: Vec::new(),
            read_spans: Vec::new(),
            op_ats: Vec::new(),
            shadow_spans: Vec::new(),
            buckets: std::array::from_fn(|_| Vec::new()),
            held_fallback: HashMap::default(),
            room: MappedCache::default(),
        }
    }
}

/// Bucket count for `constant_slots_batch`, matching `land::each_under`'s
/// active thread count (`workers + caller`) so one balanced wave finishes all
/// constant-slot writes without a 2-wave tail imbalance on 6C/12T CPUs.
fn slot_batch_parts(op_count: usize) -> usize {
    if op_count < 64 {
        return 1;
    }
    static PARTS: OnceLock<usize> = OnceLock::new();
    *PARTS.get_or_init(|| {
        let logical = std::thread::available_parallelism().map_or(12, |n| n.get());
        let workers = std::env::var("VICTORY_UNDER_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| (logical / 3).clamp(2, 6));
        (workers.clamp(1, 7) + 1).min(8)
    })
}

impl Host {
    /// With flights in the air: if what keeps `runs` from being prepared
    /// now is reads of title memory that can all be done on the card
    /// ([`stage`]), the list that does them. `None` if nothing is in the
    /// way or the old wait is due. Under the submission lock.
    pub(super) fn card_steps(&self, runs: &[&HostList]) -> Option<Taken> {
        if stage() == 0 || !relay_exact() {
            return None;
        }
        let (written, copies) = {
            let line = line();
            if line.written.is_empty() {
                return None;
            }
            (line.written.clone(), copies(&line))
        };
        blocked_by(&written, runs, false)?;
        let taken = if copied_read(&copies, runs) {
            Err("a copy done on the card is read through another host copy")
        } else {
            self.take_steps(&written, runs)
        };
        match taken {
            Ok(taken) => taken,
            Err(why) => {
                DONE[3].fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                refused(why);
                #[cfg(not(feature = "diagnostics"))]
                {
                    static LOGGED: Mutex<Option<std::collections::HashSet<&'static str>>> = Mutex::new(None);
                    let mut guard = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
                    if guard.get_or_insert_with(std::collections::HashSet::new).insert(why) {
                        tracing::info!(why, "card_steps refused a submission and fell back to waiting for flights");
                    }
                }
                None
            }
        }
    }

    fn take_steps(&self, written: &[(usize, usize)], runs: &[&HostList]) -> Result<Option<Taken>, &'static str> {
        thread_local! {
            static SCRATCH: std::cell::RefCell<CardStepsScratch> = std::cell::RefCell::new(CardStepsScratch::default());
        }
        SCRATCH.with_borrow_mut(|s| {
            let result = (|| -> Result<Option<Taken>, &'static str> {
                let mut spent = crate::timeline::span("steps on the card: chosen");
                does_of_into(written, runs, true, &mut s.does, &mut s.steps, &mut s.fills);
                let chosen = plan(&s.does, stage(), placed_order(), &mut None)?;
                if chosen.is_empty() {
                    return Ok(None);
                }
                spent.next("steps on the card: what holds their bytes");
                let (shadows, raw_writers) = {
                    let line = line();
                    WriterIndex::collect_line(&line)
                };
                let mut writer_index = WriterIndex::build(shadows, raw_writers);
                s.ops.clear();
                s.op_reads.clear();
                s.held_fallback.clear();
                s.room.clear();
                let mut has_writes = false;
                let mut copy_count = 0usize;
                let mut total_bytes = 0u64;
                let mut max_level = 0u32;
                for &(i, level) in &chosen {
                    let Some(((src, _), (dst, _))) = reads_writes(&s.does[i]) else {
                        continue;
                    };
                    let (Does::Copy { len, .. } | Does::Fill { len, .. }) = s.does[i] else {
                        continue;
                    };
                    if len == 0 {
                        return Err("nothing to read");
                    }
                    let from_shadow = writer_index.holder(src, src + len, &mut s.held_fallback)?;
                    if !s.room.check(src, src + len) {
                        return Err("not all of what it reads is mapped");
                    }
                    let to = match s.does[i] {
                        Does::Copy { .. } => {
                            let to_shadow = writer_index.holder(dst, dst + len, &mut s.held_fallback)?;
                            let to_ptr = Arc::as_ptr(&writer_index.shadows[to_shadow as usize]) as usize;
                            if elsewhere(&s.does, to_ptr, dst, dst + len) {
                                return Err("another host copy of what it writes is read");
                            }
                            if !s.room.check(dst, dst + len) {
                                return Err("not all of what it writes is mapped");
                            }
                            // What the steps behind it read of it is in that host
                            // copy, or nowhere.
                            writer_index.push_extra(to_shadow, dst, dst + len);
                            s.held_fallback.clear();
                            has_writes = true;
                            copy_count += 1;
                            To::Copy(to_shadow, dst)
                        }
                        _ => match s.fills[i].take() {
                            Some(FillTo::Slot(heap, index)) => To::Slot(heap, index),
                            Some(FillTo::Buffer(buffer)) => {
                                has_writes = true;
                                To::Buffer(buffer)
                            }
                            None => continue,
                        },
                    };
                    let aligned_256 = matches!(to, To::Slot(..));
                    max_level = max_level.max(level);
                    total_bytes += len as u64;
                    s.op_reads.push(OpRead {
                        shadow: from_shadow,
                        src,
                        len,
                        level,
                        aligned_256,
                    });
                    s.ops.push(Op {
                        from_shadow,
                        src,
                        len,
                        at: 0,
                        level,
                        to,
                    });
                }
                // Coalesce adjacent/overlapping/near-gap reads per `(level, shadow)`
                // with a single sort pass to collapse thousands of small
                // `CopyBufferRegion` calls into a handful of bulk GDDR7 copies.
                let end = layout_reads_cascade(
                    &s.op_reads,
                    BETWEEN,
                    &[65536, 16384, 2048, 256, 0],
                    &mut s.order,
                    &mut s.read_spans,
                    &mut s.op_ats,
                )
                .ok_or("too much at once")?;
                for (op, &at) in s.ops.iter_mut().zip(s.op_ats.iter()) {
                    op.at = at;
                }
                spent.next("steps on the card: the list");
                let Some(own) = self.own_list() else {
                    return Err("no host list or buffer for it");
                };
                // First every refresh, then the writes: the bytes compared with are
                // the title's. (One step for each host copy, over all that is read of
                // it: a flush brings a copy up to date from the first range asked for
                // to the last, whatever the steps.)
                s.shadow_spans.clear();
                s.shadow_spans.resize(writer_index.shadows.len(), None);
                for op in &s.ops {
                    let slot = &mut s.shadow_spans[op.from_shadow as usize];
                    match slot {
                        Some((lo, hi)) => {
                            *lo = (*lo).min(op.src);
                            *hi = (*hi).max(op.src + op.len);
                        }
                        None => *slot = Some((op.src, op.src + op.len)),
                    }
                    if let To::Copy(to_shadow, dst) = op.to {
                        let slot = &mut s.shadow_spans[to_shadow as usize];
                        match slot {
                            Some((lo, hi)) => {
                                *lo = (*lo).min(dst);
                                *hi = (*hi).max(dst + op.len);
                            }
                            None => *slot = Some((dst, dst + op.len)),
                        }
                    }
                }
                for (shadow, &span) in writer_index.shadows.iter().zip(s.shadow_spans.iter()) {
                    if let Some((from, to)) = span {
                        shadow.refresh(&own.list, from as u64, (to - from) as u64);
                    }
                }
                let mut copied = Vec::with_capacity(copy_count);
                if copy_count > 0 {
                    for op in &s.ops {
                        if let To::Copy(to_shadow, dst) = op.to {
                            let to = &writer_index.shadows[to_shadow as usize];
                            to.gpu_writes(&own.list, dst as u64, op.len as u64);
                            copied.push((to.clone(), dst, dst + op.len));
                        }
                    }
                }
                #[cfg(feature = "diagnostics")]
                own.list.note_op(|| {
                    format!(
                        "{} copies of title memory done on the card ({} coalesced reads, {end} bytes, VICTORY_CARD_STEPS)",
                        s.ops.len(),
                        s.read_spans.len(),
                    )
                });
                #[cfg(not(feature = "diagnostics"))]
                let _ = end;
                let mut span_cursor = 0usize;
                let mut reads = Vec::with_capacity(s.read_spans.len());
                let mut writes = Vec::new();
                for level in 0..=max_level {
                    reads.clear();
                    writes.clear();
                    while span_cursor < s.read_spans.len() && s.read_spans[span_cursor].level == level {
                        let rs = s.read_spans[span_cursor];
                        let shadow = &writer_index.shadows[rs.shadow as usize];
                        reads.push((&shadow.buffer, shadow.offset(rs.src as u64), rs.at, rs.len as u64));
                        span_cursor += 1;
                    }
                    if has_writes {
                        for op in s.ops.iter().filter(|op| op.level == level) {
                            let len = op.len as u64;
                            match &op.to {
                                To::Copy(to_shadow, dst) => {
                                    let to = &writer_index.shadows[*to_shadow as usize];
                                    writes.push((&to.buffer, to.offset(*dst as u64), op.at, len));
                                }
                                To::Buffer(buffer) => writes.push((buffer, 0, op.at, len)),
                                To::Slot(..) => {}
                            }
                        }
                    }
                    record(level == 0, &own.list.list, &own.between, &reads, &writes);
                }
                own.list.close();
                // SAFETY: a live buffer.
                let gpu = unsafe { own.between.GetGPUVirtualAddress() };
                spent.next("steps on the card: the slots");
                // Pre-bucket once into `land::each_under`'s active worker count
                // so every worker acquires `SHADER_HEAPS.read()` once and
                // finishes in a single balanced wave, preserving per-slot order.
                let parts = slot_batch_parts(s.ops.len());
                for b in &mut s.buckets[..parts] {
                    b.clear();
                }
                for op in &s.ops {
                    if let To::Slot(heap, index) = &op.to {
                        let part = (*index as usize / 4) % parts;
                        s.buckets[part].push((*heap, *index, gpu + op.at, op.len as u32));
                    }
                }
                let buckets = &s.buckets[..parts];
                crate::land::each_under(parts, &|part| {
                    if !buckets[part].is_empty() {
                        crate::descriptor::constant_slots_batch(&buckets[part], &own.between);
                    }
                });
                DONE[0].fetch_add((s.ops.len() - copy_count) as u64, Ordering::Relaxed);
                DONE[1].fetch_add(copy_count as u64, Ordering::Relaxed);
                DONE[2].fetch_add(total_bytes, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                noted(&s.ops, &writer_index.shadows);
                spent.end();
                let taken_steps = build_taken_steps(&chosen, &s.steps);
                Ok(Some(Taken {
                    own,
                    steps: taken_steps,
                    copied,
                }))
            })();
            // Always drop any `ID3D12Resource` COM references before returning.
            s.fills.clear();
            s.ops.clear();
            result
        })
    }

    /// A list of ours open for recording, and its buffer.
    fn own_list(&self) -> Option<Own> {
        let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(own) = pooled {
            own.list.reset();
            return Some(own);
        }
        let between = self.uav_buffer(BETWEEN)?;
        name(&between, || "copies of title memory on the card".into());
        let list = self.new_list().ok()?;
        Some(Own { list, between })
    }
}

/// Diagnostics: the copies and fills done on the card since the last
/// frame-rate line, by bytes, host copy read and host copy written (start
/// and length; zeros for a fill): how many, and the last title addresses.
#[cfg(feature = "diagnostics")]
type Noted = std::collections::HashMap<(usize, (u64, u64), (u64, u64)), (u64, usize, usize)>;
#[cfg(feature = "diagnostics")]
static NOTED: Mutex<Option<Noted>> = Mutex::new(None);

#[cfg(feature = "diagnostics")]
fn noted(ops: &[Op], shadows: &[Arc<Shadow>]) {
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    let noted = noted.get_or_insert_with(Default::default);
    for op in ops {
        let (to, dst) = match &op.to {
            To::Copy(to_shadow, dst) => (shadows[*to_shadow as usize].span(), *dst),
            To::Slot(..) | To::Buffer(_) => ((0, 0), 0),
        };
        let e = noted.entry((op.len, shadows[op.from_shadow as usize].span(), to)).or_default();
        *e = (e.0 + 1, op.src, dst);
    }
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    if stage() == 0 {
        return;
    }
    let d: Vec<f64> = DONE.iter().map(|d| d.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "steps done on the card instead of waiting for flights, per frame (VICTORY_CARD_STEPS={}): {:.2} fills, {:.2} buffer \
         copies, {:.1} KB; {:.2} submissions refused and waited",
        stage(),
        d[0],
        d[1],
        d[2] / 1024.0,
        d[3],
    );
    #[cfg(feature = "diagnostics")]
    {
        let noted = NOTED.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        let mut noted: Vec<_> = noted.into_iter().collect();
        noted.sort_unstable_by_key(|(_, e)| std::cmp::Reverse(e.0));
        for ((len, from, to), (n, src, dst)) in noted.into_iter().take(5) {
            let n = n as f64 / frames;
            if to.1 == 0 {
                tracing::info!("done on the card, per frame: {n:.2} fills of {len:#x} bytes from the host copy of {:#x}+{:#x} (last {src:#x})", from.0, from.1);
            } else {
                tracing::info!(
                    "done on the card, per frame: {n:.2} copies of {len:#x} bytes from the host copy of {:#x}+{:#x} (last {src:#x}) \
                     to that of {:#x}+{:#x} (last {dst:#x})",
                    from.0,
                    from.1,
                    to.0,
                    to.1
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(dst: usize, src: usize, len: usize, asked: bool) -> Does {
        Does::Copy { dst, src, len, asked }
    }

    fn fill(src: usize, len: usize, asked: bool) -> Does {
        Does::Fill { src, len, asked, to: true }
    }

    fn stuck(src: usize, len: usize) -> Does {
        Does::Fill { src, len, asked: false, to: false }
    }

    /// What is done on the card, as (place, level), in stage 2.
    fn planned(does: &[Does]) -> Vec<(usize, u32)> {
        plan(does, 2, true, &mut None).unwrap()
    }

    /// Only what is asked for is taken, all of it, and a step that stands
    /// in none of their ways stays where it is.
    #[test]
    fn steps_apart_from_one_another_are_taken() {
        let does = [
            copy(1000, 5000, 100, false),
            fill(2000, 64, true),
            Does::Refresh { shadow: 1, from: 0, to: 9000 },
            copy(3000, 6000, 100, true),
            fill(1000, 64, false),
            Does::Read { from: 3100, to: 4000, asked: false },
            stuck(3100, 5),
        ];
        assert_eq!(planned(&does), [(1, 0), (3, 0)]);
        // Two fills of the same bytes.
        assert_eq!(plan(&[fill(0, 64, true), fill(32, 64, true)], 1, true, &mut None), Ok(vec![(0, 0), (1, 0)]));
        assert_eq!(plan(&[fill(0, 64, false)], 1, true, &mut None), Ok(vec![]));
        // Ends that touch are apart.
        let does = [
            copy(1000, 5000, 100, true),
            copy(900, 9000, 100, false),
            copy(1100, 9000, 5, false),
            copy(4900, 9000, 100, false),
        ];
        assert_eq!(planned(&does), [(0, 0)]);
    }

    /// What depends on a taken step, or it on it, is taken too, on the level
    /// the order of the steps asks for.
    #[test]
    fn steps_that_depend_on_one_another_are_taken_in_order() {
        let taken = copy(1000, 5000, 100, true);
        // Reads what it writes, behind and ahead of it.
        assert_eq!(planned(&[taken, fill(1099, 64, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[fill(1099, 64, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(7000, 1050, 10, false)]), [(0, 0), (1, 1)]);
        // Writes what it reads, and what it writes.
        assert_eq!(planned(&[copy(5099, 9000, 1, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(5099, 9000, 1, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(999, 9000, 2, false)]), [(0, 0), (1, 1)]);
        // The same list twice.
        assert_eq!(planned(&[taken, taken]), [(0, 0), (1, 1)]);
        // A chain, with steps apart from it in between, and two fills of one
        // copy on one level.
        let does = [
            copy(8000, 7000, 50, false),
            taken,
            copy(2000, 1010, 20, false),
            fill(90_000, 10, false),
            fill(2000, 20, false),
            fill(2010, 20, true),
            copy(1000, 2005, 5, false),
            fill(1000, 100, false),
        ];
        assert_eq!(planned(&does), [(1, 0), (2, 1), (4, 2), (5, 2), (6, 2), (7, 3)]);
        // A fill is in nobody's way.
        assert_eq!(planned(&[taken, fill(5000, 100, false)]), [(0, 0)]);
        // A copy ahead that a taken one only shares its source with.
        assert_eq!(planned(&[copy(7000, 5000, 10, false), taken]), [(1, 0)]);
    }

    /// A copy at its place in a list is in the way of what reads or writes its
    /// bytes behind it, not of what did ahead of it.
    #[test]
    fn copies_at_their_place_stand_in_the_way_of_what_is_behind_them() {
        let placed = Does::Placed { dst: 2000, src: 3000, len: 64 };
        let behind = |does: &[Does]| {
            let mut found = None;
            let refused = plan(does, 2, true, &mut found);
            assert!(refused.is_ok() || found.is_some() && refused == Err(BEHIND_PLACED));
            assert_eq!(plan(does, 2, false, &mut None).is_err(), does.len() > 1 && !matches!(does, [_, Does::Fill { src: 5000, .. }]));
            found
        };
        assert_eq!(behind(&[fill(2032, 64, true), placed]), None);
        assert_eq!(behind(&[placed, fill(2032, 64, true)]), Some((0, 1)));
        assert_eq!(behind(&[placed, fill(5000, 64, true)]), None);
        assert_eq!(behind(&[fill(2000, 8, true), placed, fill(2000, 8, true)]), Some((1, 2)));
        // One that writes what it copies from, and one that writes over it.
        assert_eq!(behind(&[copy(3000, 5000, 8, true), placed]), None);
        assert_eq!(behind(&[placed, copy(3000, 5000, 8, true)]), Some((0, 1)));
        assert_eq!(behind(&[copy(2060, 5000, 8, true), placed]), None);
        assert_eq!(behind(&[placed, copy(2060, 5000, 8, true)]), Some((0, 1)));
    }

    /// Whatever the steps, done level by level (in a level every read
    /// before every write) they read and leave what they read and leave one
    /// after the other, the CPU's steps ahead of all of them.
    #[test]
    fn levels_do_what_the_order_does() {
        let mut seed = 12345u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        let (mut done, mut refused) = (0, 0);
        for _ in 0..4000 {
            let does: Vec<Does> = (0..2 + random(9))
                .map(|_| {
                    let (at, other, len) = (random(60) as usize, random(60) as usize, 1 + random(8) as usize);
                    let asked = random(4) == 0;
                    match random(8) {
                        0..4 => copy(at, other, len, asked),
                        4..6 => fill(at, len, asked),
                        6 => stuck(at, len),
                        _ => Does::Read { from: at, to: at + len, asked: false },
                    }
                })
                .collect();
            let Ok(chosen) = plan(&does, 2, true, &mut None) else {
                refused += 1;
                continue;
            };
            done += 1;
            let start: Vec<u32> = (0..80).collect();
            // One after the other: what each reads, and what is left.
            let mut memory = start.clone();
            let mut wanted = Vec::new();
            for d in &does {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                wanted.push(memory[from..to].to_vec());
                if let Does::Copy { dst, src, len, .. } = *d {
                    memory.copy_within(src..src + len, dst);
                }
            }
            // The CPU's, then the card's.
            let mut cpu = start.clone();
            let taken = |i: usize| chosen.iter().any(|c| c.0 == i);
            for (i, d) in does.iter().enumerate().filter(|&(i, _)| !taken(i)) {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                assert_eq!(cpu[from..to], wanted[i][..], "step {i} of {does:?}");
                if let Does::Copy { dst, src, len, .. } = *d {
                    cpu.copy_within(src..src + len, dst);
                }
            }
            let mut card = cpu.clone();
            for level in 0..=chosen.iter().map(|c| c.1).max().unwrap_or(0) {
                let of: Vec<usize> = chosen.iter().filter(|c| c.1 == level).map(|c| c.0).collect();
                let before = card.clone();
                for i in of {
                    let ((from, to), (dst, _)) = reads_writes(&does[i]).unwrap();
                    assert_eq!(before[from..to], wanted[i][..], "step {i} of {does:?}: {chosen:?}");
                    if matches!(does[i], Does::Copy { .. }) {
                        card[dst..dst + to - from].copy_from_slice(&before[from..to]);
                    }
                }
            }
            assert_eq!(card, memory, "{does:?}: {chosen:?}");
        }
        assert!(done > 500 && refused > 100, "{done} {refused}");
    }

    /// What cannot be done on the card refuses the lot, where it stands in
    /// the way.
    #[test]
    fn steps_in_the_way_are_refused() {
        let taken = copy(1000, 5000, 100, true);
        assert!(plan(&[taken, Does::Run], 2, true, &mut None).is_err());
        // A texture that itself waits for flights; a fill that cannot be
        // taken but waits.
        assert!(plan(&[taken, Does::Read { from: 1, to: 2, asked: true }], 2, true, &mut None).is_err());
        assert!(plan(&[Does::Fill { src: 1, len: 1, asked: true, to: false }], 2, true, &mut None).is_err());
        // Reads on the CPU of what it writes: behind it, not ahead of it.
        for read in [Does::Read { from: 900, to: 1001, asked: false }, stuck(1050, 3)] {
            assert!(plan(&[taken, read], 2, true, &mut None).is_err(), "{read:?} behind");
            assert_eq!(planned(&[read, taken]), [(1, 0)], "{read:?} ahead");
        }
        // Also of what is taken along.
        assert!(plan(&[taken, copy(3000, 1000, 10, false), stuck(3005, 1)], 2, true, &mut None).is_err());
        // Into itself, and copies without stage 2.
        assert!(plan(&[copy(1000, 1050, 100, true)], 2, true, &mut None).is_err());
        assert!(plan(&[taken], 1, true, &mut None).is_err());
        assert!(plan(&[fill(1000, 4, true), copy(1000, 5000, 100, false)], 1, true, &mut None).is_err());
    }

    #[test]
    fn a_refresh_of_another_host_copy_is_seen() {
        let does = [Does::Refresh { shadow: 7, from: 100, to: 200 }];
        assert!(!elsewhere(&does, 7, 100, 200));
        assert!(elsewhere(&does, 8, 199, 300));
        assert!(!elsewhere(&does, 8, 200, 300));
    }

    /// A value written behind a flight, or a read-back, is in no host copy
    /// before it lands.
    #[test]
    fn what_no_host_copy_holds_is_refused() {
        let writers = [Writer { shadow: None, from: 100, to: 108, copy: false }];
        assert!(holder(&writers, 104, 120).is_err());
    }

    /// `WriterIndex` coalesces redundant intervals and produces identical
    /// conflict/match results to scanning raw intervals.
    #[test]
    fn writer_index_coalesces_and_detects_conflicts() {
        let idx = WriterIndex::build(
            Vec::new(),
            vec![
                (Some(0), 1000, 2000),
                (Some(0), 1500, 3000),
                (Some(0), 3000, 4000),
                (Some(1), 3500, 5000),
                (None, 8000, 8064),
            ],
        );
        // Three overlapping/touching shadow-0 spans merged into one [1000, 4000).
        assert_eq!(idx.entries.len(), 3);
        assert_eq!(idx.query_writers(1200, 1400), Ok(Some(0)));
        assert_eq!(idx.query_writers(3400, 3500), Ok(Some(0)));
        assert_eq!(idx.query_writers(4000, 4200), Ok(Some(1)));
        assert_eq!(idx.query_writers(5000, 6000), Ok(None));
        // Overlap between shadow 0 and shadow 1 -> error.
        assert!(idx.query_writers(3400, 3600).is_err());
        // Overlap with shadowless writer -> error.
        assert!(idx.query_writers(7990, 8010).is_err());
    }

    #[test]
    fn writer_index_isolated_flag_and_bitset_helpers_are_exact() {
        let idx = WriterIndex::build(
            Vec::new(),
            vec![
                (Some(0), 0x1000, 0x3000), // isolated
                (Some(1), 0x4000, 0x6000), // overlaps shadow 2
                (Some(2), 0x5000, 0x7000), // overlaps shadow 1
                (Some(3), 0x7000, 0x9000), // touching [0x5000, 0x7000) at boundary -> isolated!
            ],
        );
        assert_eq!(idx.entries.len(), 4);
        assert!(idx.entries[0].isolated);
        assert!(!idx.entries[1].isolated);
        assert!(!idx.entries[2].isolated);
        assert!(idx.entries[3].isolated);
        assert_eq!(idx.query_writers_detail(0x1200, 0x1400), Ok(Some((0, Some((0x1000, 0x3000))))));
        assert_eq!(idx.query_writers_detail(0x4100, 0x4200), Ok(Some((1, None))));
        assert_eq!(idx.query_writers_detail(0x7000, 0x7200), Ok(Some((3, Some((0x7000, 0x9000))))));

        let steps = vec![(0xAAA, 0), (0xAAA, 63), (0xAAA, 64), (0xBBB, 5), (0xAAA, 130)];
        let chosen = vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)];
        let built = build_taken_steps(&chosen, &steps);
        assert_eq!(built.len(), 2);
        let a_bits = built.iter().find(|(p, _)| *p == 0xAAA).map(|(_, b)| b.as_slice());
        let b_bits = built.iter().find(|(p, _)| *p == 0xBBB).map(|(_, b)| b.as_slice());
        for &i in &[0, 63, 64, 130] {
            assert!(Taken::bit_taken(a_bits, i));
        }
        assert!(!Taken::bit_taken(a_bits, 1));
        assert!(!Taken::bit_taken(a_bits, 129));
        assert!(Taken::bit_taken(b_bits, 5));
        assert!(!Taken::bit_taken(b_bits, 4));
    }

    #[test]
    fn layout_reads_cascade_coalesces_64k_gaps_and_falls_back_cleanly() {
        let ops = [
            OpRead { shadow: 0, src: 0x10000, len: 256, level: 0, aligned_256: true },
            OpRead { shadow: 0, src: 0x18000, len: 256, level: 0, aligned_256: true }, // 32 KiB gap
            OpRead { shadow: 0, src: 0x20000, len: 256, level: 0, aligned_256: true }, // 32 KiB gap
        ];
        let (mut order, mut spans, mut ats) = (Vec::new(), Vec::new(), Vec::new());
        // Plenty of budget: 64 KiB gap collapses all 3 CBVs into 1 span while keeping 256B alignment.
        let end = layout_reads_cascade(&ops, BETWEEN, &[65536, 16384, 2048, 256, 0], &mut order, &mut spans, &mut ats)
            .expect("coalesced fits");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].len, 0x10100);
        assert!(ats.iter().all(|&a| a.is_multiple_of(256)));
        assert!(end <= BETWEEN);

        // Tight budget (1024 B): cascades down to gap=0 -> 3 separate 256B spans (768 B total).
        let end_tight = layout_reads_cascade(&ops, 1024, &[65536, 16384, 2048, 256, 0], &mut order, &mut spans, &mut ats)
            .expect("tight fallback fits");
        assert_eq!(spans.len(), 3);
        assert_eq!(end_tight, 768);
        assert_eq!(ats, vec![0, 256, 512]);
    }

    /// `layout_reads` merges adjacent, overlapping, and near-gap reads into
    /// bulk spans while keeping every CBV (`aligned_256`) 256-byte aligned and
    /// falling back cleanly when a gap budget would overflow `BETWEEN`.
    #[test]
    fn layout_reads_coalesces_and_preserves_cbv_alignment() {
        let ops = [
            OpRead { shadow: 0, src: 0x10000, len: 192, level: 0, aligned_256: true },
            OpRead { shadow: 0, src: 0x10100, len: 480, level: 0, aligned_256: true },
            OpRead { shadow: 0, src: 0x10400, len: 256, level: 0, aligned_256: true },
            // Unaligned copy inside the same 256B-aligned span merges too.
            OpRead { shadow: 0, src: 0x10040, len: 32, level: 0, aligned_256: false },
            // Rare unaligned CBV stays isolated at its own 256B-aligned offset.
            OpRead { shadow: 0, src: 0x10080, len: 64, level: 0, aligned_256: true },
            // Different level never merges with level 0.
            OpRead { shadow: 0, src: 0x10000, len: 256, level: 1, aligned_256: true },
        ];
        let (spans, ats, end) = layout_reads(&ops, BETWEEN, 2048).expect("fits in budget");
        // Level 0 normal ops collapse into 1 span [0x10000..0x10500), unaligned slot gets 1, level 1 gets 1.
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0], ReadSpan { shadow: 0, level: 0, src: 0x10000, len: 0x500, at: 0 });
        assert_eq!(ats[0], 0);
        assert_eq!(ats[1], 0x100);
        assert_eq!(ats[2], 0x400);
        assert_eq!(ats[3], 0x40);
        assert_eq!(ats[4] % 256, 0);
        assert_eq!(ats[5] % 256, 0);
        for (op, &at) in ops.iter().zip(&ats) {
            if op.aligned_256 {
                assert_eq!(at % 256, 0, "CBV must be 256B aligned");
                assert!(at + op.len.next_multiple_of(256) as u64 <= end);
            }
        }
        // Tight budget rejects 2048B gap bridging but succeeds with 0B gap.
        let gapped = [
            OpRead { shadow: 0, src: 0x10000, len: 256, level: 0, aligned_256: true },
            OpRead { shadow: 0, src: 0x10800, len: 256, level: 0, aligned_256: true },
        ];
        assert!(layout_reads(&gapped, 512, 2048).is_none());
        let (tight_spans, tight_ats, tight_end) = layout_reads(&gapped, 512, 0).expect("fits with gap=0");
        assert_eq!(tight_spans.len(), 2);
        assert_eq!(tight_ats, vec![0, 256]);
        assert_eq!(tight_end, 512);
    }

    /// Reads are all before writes, between two buffers and within one, and
    /// the buffers are usable in the lists that follow in the same call.
    #[test]
    fn the_card_copies_what_is_recorded() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let (a, a_data) = host.shared_buffer(1 << 16).unwrap();
        let (b, b_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: mappings of 64 KiB.
        let (a_bytes, b_bytes) =
            unsafe { (std::slice::from_raw_parts_mut(a_data, 1 << 16), std::slice::from_raw_parts_mut(b_data, 1 << 16)) };
        for (i, (x, y)) in a_bytes.iter_mut().zip(b_bytes.iter_mut()).enumerate() {
            (*x, *y) = (i as u8 ^ (i >> 8) as u8, 0xee);
        }
        let was = a_bytes.to_vec();
        let own = host.own_list().unwrap();
        // Within `a` (there and back), from `a` to `b`, and a read alone.
        record(
            true,
            &own.list.list,
            &own.between,
            &[(&a, 33, 0, 100), (&a, 2000, 256, 100), (&a, 5001, 512, 7), (&a, 9, 1024, 4)],
            &[(&a, 2000, 0, 100), (&a, 33, 256, 100), (&b, 77, 512, 7)],
        );
        // A level behind it, of what it wrote.
        record(false, &own.list.list, &own.between, &[(&b, 77, 2048, 7)], &[(&b, 1077, 2048, 7)]);
        own.list.close();
        // A list behind it that uses the buffers in other ways.
        let next = host.new_list().unwrap();
        let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: an open list; live buffers.
        unsafe {
            next.list.CopyBufferRegion(&out, 0, &own.between, 1024, 4);
            next.list.CopyBufferRegion(&b, 4096, &a, 33, 100);
        }
        next.close();
        host.execute(&[&own.list, &next]);
        // With the debug layer (`VICTORY_D3D12_DEBUG=1`, diagnostics).
        assert_eq!(host.log_messages(), None);
        assert_eq!(a_bytes[2000..2100], was[33..133]);
        assert_eq!(a_bytes[33..133], was[2000..2100]);
        assert_eq!(b_bytes[77..84], was[5001..5008]);
        assert_eq!(b_bytes[1077..1084], was[5001..5008]);
        assert_eq!(b_bytes[4096..4196], was[2000..2100]);
        // SAFETY: a mapping of 64 KiB.
        assert_eq!(unsafe { std::slice::from_raw_parts(out_data, 4) }, &was[9..13]);
        assert!(b_bytes[..77].iter().all(|&x| x == 0xee));
        a_bytes[2000..2100].copy_from_slice(&was[2000..2100]);
        a_bytes[33..133].copy_from_slice(&was[33..133]);
        assert_eq!(a_bytes, &was[..]);
        // As the landing does: its next use records anew.
        POOL.lock().unwrap().push(own);
        let own = host.own_list().unwrap();
        own.list.close();
    }

    /// The whole way, in a process of its own with the relay on:
    /// `VICTORY_RELAY=2 VICTORY_CARD_STEPS=2 cargo test -p victory-gpu --lib
    /// card_steps -- --ignored --test-threads=1`. A flight's list writes
    /// part of what the next submission's copy reads, the title wrote the
    /// rest; the copy is done on the card behind the flight, and a second
    /// one reads what the first wrote. Title memory ends as on the CPU.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_CARD_STEPS=2 and a process of its own"]
    fn a_copy_behind_a_flight_reads_what_the_flight_wrote() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const LEN: usize = 4 << 20;
        // Without it (`TEST_UNWATCHED=1`) the title's bytes cannot be put in
        // under a flight's writes: the flights that declared them are waited
        // for.
        let write_watch = if std::env::var_os("TEST_UNWATCHED").is_some() { 0 } else { 0x20_0000 };
        let Some(host) = host() else {
            return;
        };
        assert!(relay_exact() && stage() == 2, "set VICTORY_RELAY=2 VICTORY_CARD_STEPS=2");
        // Two title buffers: in the first a copy within it, from it into the
        // second, and back.
        let (title, start) = {
            // SAFETY: memory of our own, never freed.
            let base = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | write_watch, PAGE_READWRITE) }.cast::<u8>();
            assert!(!base.is_null());
            // SAFETY: `LEN` bytes just committed.
            (unsafe { std::slice::from_raw_parts_mut(base, LEN) }, base as usize)
        };
        const HALF: usize = LEN / 2;
        crate::shadow::register(0x1111, start as u64, HALF as u64);
        crate::shadow::register(0x2222, (start + HALF) as u64, HALF as u64);
        let first = crate::shadow::find(start as u64, 64).unwrap();
        let (slow_from, slow_to) = (host.uav_buffer(64 << 20).unwrap(), host.uav_buffer(64 << 20).unwrap());
        let before = DONE[1].load(Ordering::Relaxed);
        let mut lists = Vec::new();
        for round in 0..8usize {
            // The title writes; then a "shader" of a flight writes 300 bytes
            // of what the copy reads, behind some work.
            let seed = |i: usize| (i * 7 + round * 13 + i / 251) as u8;
            for (i, b) in title[4000..6000].iter_mut().enumerate() {
                *b = seed(i);
            }
            let (upload, data) = host.upload_buffer(4096).unwrap();
            for i in 0..300 {
                // SAFETY: inside the mapping.
                unsafe { data.add(i).write(0x80 | (round * 31 + i) as u8) };
            }
            let a = host.new_list().unwrap();
            first.refresh(&a, start as u64, HALF as u64);
            first.gpu_writes(&a, start as u64, HALF as u64);
            // SAFETY: an open list; live buffers.
            unsafe {
                for _ in 0..24 {
                    a.list.CopyBufferRegion(&slow_to, 0, &slow_from, 0, 64 << 20);
                }
                a.list.CopyBufferRegion(&first.buffer, 4500, &upload, 0, 300);
                a.list
                    .ResourceBarrier(&[transition(&first.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
            }
            a.close();
            host.execute_own(&a);
            // With the flight in the air, beside what it writes.
            for (i, b) in title[5300..5400].iter_mut().enumerate() {
                *b = seed(1300 + i) ^ 0x5c;
            }
            // Thousands of runs of bytes of its own.
            for b in title[1_700_000..1_708_000].iter_mut().step_by(2) {
                *b = 1 + round as u8;
            }
            // Into the second buffer, and within the first.
            let b = host.new_list().unwrap();
            b.defer_copy((start + HALF + 100) as *mut u8, (start + 4001) as *const u8, 1500);
            b.defer_copy((start + 900_001) as *mut u8, (start + 4400) as *const u8, 777);
            b.defer_copy((start + HALF + 10_000) as *mut u8, (start + 1_700_000) as *const u8, 8000);
            // What depends on them: a copy of the first copy, upload memory
            // filled from that, and the first one's bytes copied over.
            b.defer_copy((start + 1_500_000) as *mut u8, (start + HALF + 600) as *const u8, 700);
            let (staging, staged) = b.upload(host, 4096).unwrap();
            b.defer_fill_buffer(&staging, staged, (start + 1_500_100) as *const u8, 500);
            b.defer_copy((start + HALF + 300) as *mut u8, (start + 8000) as *const u8, 64);
            b.close();
            host.execute_own(&b);
            // What the first wrote, read by the next submission.
            let c = host.new_list().unwrap();
            c.defer_copy((start + 1_200_000) as *mut u8, (start + HALF + 100) as *const u8, 1500);
            c.close();
            let number = host.execute_own(&c);
            settle_now(WHY_SYNC);
            assert!(FINISHED.load(Ordering::Acquire) >= number);
            let wanted: Vec<u8> = (0..2000)
                .map(|i| match i {
                    500..800 => 0x80 | (round * 31 + i - 500) as u8,
                    1300..1400 => seed(i) ^ 0x5c,
                    _ => seed(i),
                })
                .collect();
            assert!(title[4000..6000] == wanted[..], "what the flight wrote, round {round}");
            // (The copy over the first one's bytes came last.)
            let mut over = wanted[1..1501].to_vec();
            over[200..264].fill(0);
            assert!(title[HALF + 100..HALF + 1600] == over[..], "the copy to the other buffer, round {round}");
            assert!(title[1_500_000..1_500_700] == wanted[501..1201], "the copy behind a copy, round {round}");
            // SAFETY: the mapping, which the flight wrote and has landed.
            let staged = unsafe { std::slice::from_raw_parts(staged, 500) };
            assert!(staged == &wanted[601..1101], "the fill behind it, round {round}");
            assert!(title[900_001..900_778] == wanted[400..1177], "the copy within the buffer, round {round}");
            assert!(title[1_200_000..1_201_500] == over[..], "the copy of the copy, round {round}");
            assert!(title[HALF..HALF + 100].iter().all(|&x| x == 0));
            assert!(title[HALF + 1600..HALF + 10_000].iter().all(|&x| x == 0));
            assert!(title[HALF + 18_000..].iter().all(|&x| x == 0));
            let runs = &title[HALF + 10_000..HALF + 18_000];
            assert!(runs.chunks(2).all(|b| b == [1 + round as u8, 0]), "the runs, round {round}");
            a.keep(&upload);
            lists.extend([a, b, c]);
        }
        assert_eq!(host.log_messages(), None);
        let done = DONE[1].load(Ordering::Relaxed) - before;
        eprintln!("{done} of 48 copies were done on the card, {} submissions waited", DONE[3].load(Ordering::Relaxed));
        eprintln!("settles for title memory under shader writes: {}", LATE_STATS[WHY_STALE].load(Ordering::Relaxed));
        assert!(done >= 20, "only {done} copies were done on the card");
    }
}
