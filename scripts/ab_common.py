"""Shared timing-order logic for the A/B scripts.

Two systematic biases have been found in this project's interleaved A/B
measurements. Both looked exactly like a small real speedup, and neither was
noise. Both were caught only by a **self-control** run -- the same code timed
against itself, where the true ratio is 1.000x by construction.

**Asymmetric sampling.** Timing one arm twice per round and another once, then
taking min of each, compares min-of-2 against min-of-1. The arm sampled twice
gets a free ~2.5%.

**Asymmetric ordering.** Fixing the first left rounds looking like
`off, on, off, on`, which hands the off arm slots 1 and 3 and the on arm slots
2 and 4 every single round. On this machine the later slots are systematically
faster -- graph-pool warm-up -- so the on arm got a free head start. A
self-control that had to read 1.000x instead read **1.115x** at the smallest
shape, and 1.012x over the whole seq-128 group. Symmetric sampling does not
imply symmetric ordering.

`balanced_order` fixes both at once: every arm is timed the same number of
times, the slot pattern is symmetric about the middle of the round, and which
arm leads alternates between rounds so no arm keeps the good slots.

    for rnd in range(rounds):
        for arm in balanced_order(("off", "on"), rnd):
            times[arm].append(time_it(arm))

For two arms that gives `off on on off` then `on off off on`, so across any two
rounds each arm has occupied slots 1, 2, 3 and 4 exactly once.

csrc/TUNING.md, "Two ways an interleaved A/B lies", has the measurements.
"""

from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def balanced_order(arms: Sequence[T], rnd: int) -> List[T]:
    """One round's timing order: every arm twice, positions symmetric.

    `arms` is the arms in any fixed order; `rnd` is the round index. Forward
    then reversed puts each arm's two slots symmetrically about the middle, and
    flipping the base order on odd rounds stops the first-listed arm always
    taking the extremes.
    """
    base = list(arms) if rnd % 2 == 0 else list(arms)[::-1]
    return base + base[::-1]
