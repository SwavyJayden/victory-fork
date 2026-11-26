//! Ranges cut from a large block and given back: the free-space allocator
//! behind pooled GPU memory ([`crate::pool`], serving both 64 MiB texture
//! heaps and 64 MiB upload staging blocks).
//!
//! Adapted from `myokex/victory-amd` (`c3723f3`, after `shadowdog00`'s
//! `vi-pc-combo` `0781f61`, GPL-3.0): maintains exact O(1) total free bytes
//! (`room`), a high-water mark (`fresh`) so untouched tails skip zeroing, and
//! a per-alignment refused-length cache so full or fragmented 64 MiB blocks
//! are skipped in O(1) without linearly scanning their free-range `BTreeMap`.
//! Unlike single-alignment heaps, `pool.rs` places both 4 KiB small textures
//! and 64 KiB textures into the same 64 MiB heap, so refusals are tracked per
//! alignment and only reject requests whose alignment is a multiple of a
//! refused alignment (guaranteeing zero false refusals across mixed alignments).

use std::collections::BTreeMap;

/// Maximum distinct alignments cached per block (pools use at most 2:
/// 512 B for staging, or 4 KiB + 64 KiB for textures).
const REFUSED_SLOTS: usize = 4;

/// The free ranges of one block, by offset (coalesced: no two entries touch).
pub(crate) struct Ranges {
    free: BTreeMap<u64, u64>,
    /// Free bytes in all (exact O(1) upper bound for [`take`](Self::take)).
    room: u64,
    /// Per-alignment shortest length refused since the last [`give`](Self::give):
    /// `(align, refused_len)`, with `align == 0` marking an unused slot.
    refused: [(u64, u64); REFUSED_SLOTS],
    /// The first byte no range handed out has ever held.
    fresh: u64,
}

impl Ranges {
    pub(crate) fn new(size: u64) -> Ranges {
        let mut free = BTreeMap::new();
        if size > 0 {
            free.insert(0, size);
        }
        Ranges {
            free,
            room: size,
            refused: [(0, u64::MAX); REFUSED_SLOTS],
            fresh: 0,
        }
    }

    /// Free bytes in all (exact O(1) upper bound for what [`take`](Self::take)
    /// can hand out).
    pub(crate) fn room(&self) -> u64 {
        self.room
    }

    /// Whether the entire block of `size` bytes is free.
    pub(crate) fn whole(&self, size: u64) -> bool {
        self.room == size
    }

    /// The lowest offset for `len` bytes at a multiple of `align` (`align > 0`),
    /// and whether any byte of it was handed out before (`start < fresh`);
    /// `None` if nothing fits.
    pub(crate) fn take(&mut self, len: u64, align: u64) -> Option<(u64, bool)> {
        if len == 0 || align == 0 || len > self.room {
            return None;
        }
        // Any multiple of a previously refused alignment is strictly at least
        // as restrictive; since no ranges were returned since the refusal, a
        // request of length `>= r` cannot fit.
        if self
            .refused
            .iter()
            .any(|&(a, r)| a != 0 && align.is_multiple_of(a) && len >= r)
        {
            return None;
        }
        let found = self.free.iter().find_map(|(&at, &size)| {
            let start = at.checked_next_multiple_of(align)?;
            let end = start.checked_add(len)?;
            let range_end = at.checked_add(size)?;
            (end <= range_end).then_some((at, size, start))
        });
        let Some((at, size, start)) = found else {
            self.note_refused(align, len);
            return None;
        };
        self.free.remove(&at);
        if start > at {
            self.free.insert(at, start - at);
        }
        let end = start + len;
        let range_end = at + size;
        if end < range_end {
            self.free.insert(end, range_end - end);
        }
        self.room -= len;
        let used_before = start < self.fresh;
        self.fresh = self.fresh.max(end);
        Some((start, used_before))
    }

    fn note_refused(&mut self, align: u64, len: u64) {
        if let Some(slot) = self.refused.iter_mut().find(|(a, _)| *a == align) {
            slot.1 = slot.1.min(len);
        } else if let Some(slot) = self.refused.iter_mut().find(|(a, _)| *a == 0) {
            *slot = (align, len);
        }
    }

    /// Gives back `len` bytes at `offset` previously returned by
    /// [`take`](Self::take), coalescing with adjacent free ranges.
    pub(crate) fn give(&mut self, offset: u64, len: u64) {
        if len == 0 {
            return;
        }
        let (mut at, mut size) = (offset, len);
        if let Some((&before, &its)) = self.free.range(..offset).next_back()
            && before + its == offset
        {
            self.free.remove(&before);
            (at, size) = (before, its + size);
        }
        if let Some(after) = self.free.remove(&(offset + len)) {
            size += after;
        }
        self.free.insert(at, size);
        self.room += len;
        self.refused = [(0, u64::MAX); REFUSED_SLOTS];
    }

    #[cfg(test)]
    fn pieces(&self) -> Vec<(u64, u64)> {
        self.free.iter().map(|(&a, &s)| (a, s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Ranges;

    #[test]
    fn takes_first_fit_and_joins_what_is_given_back() {
        let mut r = Ranges::new(1 << 20);
        assert!(r.whole(1 << 20));
        let a = r.take(4096, 4096).unwrap().0;
        let b = r.take(8192, 4096).unwrap().0;
        let c = r.take(4096, 4096).unwrap().0;
        assert_eq!((a, b, c), (0, 4096, 12288));
        assert_eq!(r.room(), (1 << 20) - 16384);
        assert!(!r.whole(1 << 20));
        r.give(b, 8192);
        assert_eq!(r.pieces(), vec![(4096, 8192), (16384, (1 << 20) - 16384)]);
        assert_eq!(r.take(12288, 4096), Some((16384, false)));
        assert_eq!(r.take(8192, 4096), Some((4096, true)));
        r.give(16384, 12288);
        r.give(a, 4096);
        r.give(c, 4096);
        r.give(4096, 8192);
        assert_eq!(r.pieces(), vec![(0, 1 << 20)]);
        assert_eq!(r.room(), 1 << 20);
        assert!(r.whole(1 << 20));
    }

    #[test]
    fn says_which_ranges_held_something_before() {
        let mut r = Ranges::new(65536);
        assert_eq!(r.take(8192, 4096), Some((0, false)));
        assert_eq!(r.take(8192, 4096), Some((8192, false)));
        r.give(8192, 8192);
        // Half of it was handed out before, half of it never.
        assert_eq!(r.take(16384, 4096), Some((8192, true)));
        assert_eq!(r.take(4096, 4096), Some((24576, false)));
    }

    #[test]
    fn remembers_refused_length_until_give_and_respects_multiple_alignments() {
        // 128 KiB block: take 4 KiB at 0, then 60 KiB at 68 KiB..128 KiB so the
        // only free range is [4 KiB, 68 KiB) (length 64 KiB, 4 KiB aligned, NOT
        // 64 KiB aligned).
        let mut r = Ranges::new(128 << 10);
        assert_eq!(r.take(4 << 10, 4 << 10), Some((0, false)));
        assert_eq!(r.take(64 << 10, 4 << 10), Some((4 << 10, false)));
        assert_eq!(r.take(60 << 10, 4 << 10), Some((68 << 10, false)));
        r.give(4 << 10, 64 << 10);
        assert_eq!(r.pieces(), vec![(4 << 10, 64 << 10)]);

        // 64 KiB at 64 KiB alignment cannot fit (next multiple of 64 KiB is 64 KiB,
        // leaving only 4 KiB before 68 KiB).
        assert_eq!(r.take(64 << 10, 64 << 10), None);
        // A 64 KiB refusal at 64 KiB alignment MUST NOT falsely block 64 KiB at
        // 4 KiB alignment!
        assert_eq!(r.take(64 << 10, 4 << 10), Some((4 << 10, true)));
        assert_eq!(r.room(), 0);

        // Now test refusal caching and clearing on `give`:
        let mut r2 = Ranges::new(65536);
        let a = r2.take(32768, 4096).unwrap().0;
        r2.take(16384, 4096).unwrap();
        assert_eq!(r2.take(32768, 4096), None);
        // Larger length at same or stricter alignment (64 KiB is a multiple of 4 KiB)
        // is fast-rejected.
        assert_eq!(r2.take(40960, 4096), None);
        assert_eq!(r2.take(32768, 65536), None);
        // Shorter ones still find room.
        assert_eq!(r2.take(8192, 4096), Some((49152, false)));
        // Returning `a` clears the refusal cache.
        r2.give(a, 32768);
        assert_eq!(r2.take(32768, 4096), Some((0, true)));
    }

    #[test]
    fn keeps_alignment_and_refuses_invalid_or_overflowing_requests() {
        const MIB_4: u64 = 4 << 20;
        let mut r = Ranges::new(16 << 20);
        // Small 4 KiB allocation at 0 leaves [4 KiB, 16 MiB).
        assert_eq!(r.take(4096, 4096), Some((0, false)));
        // 64 KiB aligned allocation jumps to 64 KiB, preserving [4 KiB, 64 KiB) hole.
        assert_eq!(r.take(65536, 65536), Some((65536, false)));
        assert_eq!(r.pieces()[0], (4096, 61440));
        // 4 MiB aligned allocation jumps to 4 MiB.
        assert_eq!(r.take(MIB_4, MIB_4), Some((MIB_4, false)));
        // Smaller 4 KiB allocation reuses the padding hole at 4 KiB.
        assert_eq!(r.take(4096, 4096), Some((4096, true)));
        // Invalid / zero / oversized requests.
        assert_eq!(r.take(0, 65536), None);
        assert_eq!(r.take(4096, 0), None);
        assert_eq!(r.take((16 << 20) + 1, 65536), None);
        assert_eq!(r.take(u64::MAX, 65536), None);
        assert_eq!(r.take(4096, u64::MAX), None);
    }

    #[test]
    fn fragmentation_stress_and_full_coalesce() {
        const BLOCK: u64 = 64 << 20;
        let mut r = Ranges::new(BLOCK);
        let mut live: Vec<(u64, u64)> = Vec::new();
        let specs = [
            (4 << 10, 4 << 10),
            (16 << 10, 4 << 10),
            (64 << 10, 64 << 10),
            (256 << 10, 64 << 10),
            (1 << 20, 64 << 10),
        ];
        // 40 cycles x (~1.375 MiB with 64 KiB alignment) = ~55 MiB < 64 MiB.
        for i in 0..200usize {
            let (len, align) = specs[i % specs.len()];
            let (off, _) = r.take(len, align).expect("must fit in 64 MiB");
            assert_eq!(off % align, 0);
            assert!(
                live.iter().all(|&(o, l)| off + len <= o || o + l <= off),
                "allocated range overlaps an existing live range"
            );
            live.push((off, len));
        }
        // Free every odd entry to create heavy fragmentation.
        let mut kept = Vec::new();
        for (i, (off, len)) in live.into_iter().enumerate() {
            if i % 2 == 1 {
                r.give(off, len);
            } else {
                kept.push((off, len));
            }
        }
        // Reallocate mixed sizes into the fragmented holes.
        for i in 0..64usize {
            let (len, align) = specs[i % specs.len()];
            let (off, _) = r.take(len, align).expect("reallocation into holes must fit");
            assert_eq!(off % align, 0);
            assert!(kept.iter().all(|&(o, l)| off + len <= o || o + l <= off));
            kept.push((off, len));
        }
        // Free everything in scrambled order and verify exact full coalesce.
        while let Some((off, len)) = kept.pop() {
            r.give(off, len);
        }
        assert!(r.whole(BLOCK));
        assert_eq!(r.room(), BLOCK);
        assert_eq!(r.pieces(), vec![(0, BLOCK)]);
    }
}
