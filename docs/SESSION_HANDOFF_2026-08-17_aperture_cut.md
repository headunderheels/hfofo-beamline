# Session handoff: milestone D, eigen-emittance -- live blocker resolved

**Written by:** Claude, continuing the eigen-emittance thread from the prior
handoff (`SESSION_HANDOFF_2026-08-17_eigen_emittance.md`, if that file is
still present -- this document supersedes its §2.4 "live blocker" section;
everything else in it is still accurate and not repeated here).

**Repos, at the time of THIS handoff:**
- `beamline` fork: HEAD `2013daa` (unchanged from the prior handoff)
- `hfofo-beamline` app repo: HEAD `873da38` plus this session's uncommitted
  work (see §3) -- **not yet committed or pushed**, since Claude can clone
  but not push; the user applies patches themselves.

---

## 0. Load-bearing fix made before anything else this session

`pyproject.toml`/`uv.lock` were still pinned to `beamline@f1497b7` -- the
**old, broken** commit, predating the `ThickSolenoid` shell-placement fix
and the stepsize/stochastic-solve fixes. This means every claim in the
prior handoff about milestone C's fidelity (and by extension anything built
on `track_with_drag` this session) would have silently been running on
reverted physics if left as-is. Fixed: pin updated to `2013daa`,
`uv lock --upgrade-package beamline` regenerated the lockfile, verified
`ThickSolenoid.B_shells`'s source actually reflects the fix post-sync, and
confirmed all 14 pre-existing tests still pass. **If you're picking this up
fresh, verify this pin is still correct before trusting anything else** --
if the user has re-cloned or reset since this handoff, check it again.

---

## 1. What this session actually did (§2.4 of the prior handoff)

The prior handoff's live blocker: vmapping a 24-particle ensemble through
`track_with_drag` hit the sandbox's tool-timeout, framed as possibly a
"compile cost scales badly with N" problem. **That framing was wrong.**

### 1.1 What it actually was

Compile time is flat (~15-18s) from N=4 through N=24 -- confirmed by timing
compile and run separately (`.lower(...).compile()` vs the call itself,
exactly as the prior handoff's step 1 asked for). The real mechanism:
`jax.vmap` forces every particle in a batch through the same number of
adaptive solver steps. A realistic, diverse beam sample -- drawn from the
actual `initial.dat`, not a synthetic toy -- will generically include at
least one particle whose own local dynamics are genuinely unstable, and
that one particle can single-handedly exhaust `max_steps` for the *entire*
batch regardless of how well-behaved the other N-1 are. This is why N=4/8
"worked" in earlier ad hoc tests (lucky draws that happened not to include
such a particle) while N=16/24 failed.

**Confirmed directly, not inferred:** tested every one of the 24 particles
in the reproducible seed=0 sample individually (non-vmapped) against the
*full, unmodified* `track_with_drag` at the safe default tolerance
(`rtol=1e-3`). Exactly one (index 20, `x=77.8mm y=-43.4mm pz=274.9 MeV/c` --
nothing visually unusual about its initial conditions) fails. Bisecting the
z-range it fails over shows its transverse radius genuinely diverging:
65mm -> 130mm -> 309mm over just 400mm of travel (roughly doubling every
150-200mm) -- a real instability, not a numerical-resolution artifact
(confirmed: making `dz` *finer* made the failure *worse*, the opposite of
what a resolution problem would show).

### 1.2 The wrong fix, tried and rejected

First instinct: loosen tolerance (`rtol=1e-2`) to stop the crash. This
"worked" in the sense that N=24 completed without raising -- but inspecting
the actual output showed particle 20 ending up **100+ kilometers off-axis
with negative kinetic energy** (`x=138105mm y=-42109mm pz=-1.43 MeV`).
Every other particle's output was physically sane. Loosening tolerance
doesn't fix the instability; it just lets the solver push through into
nonphysical territory instead of correctly refusing (diffrax's `max_steps`
safety check firing here is the *same* safety net the earlier handoff
documented as "already caught a real bug once" -- suppressing/evading it
broadly is the same mistake in a new place). Computing an ensemble
covariance/emittance from a batch containing that one point produced a
garbage result (output eigen-emittance product ~4.4 billion vs an input of
~1.6 million -- obviously not real physics). **Do not loosen tolerance as
the fix for this class of problem** -- confirmed, don't re-try it.

### 1.3 The actual fix: an aperture cut

The real channel has aperture/collimation physics this simplified model
never had at all: iris apertures (200-300mm depending on cavity variant),
the `abtube` kill volume (r>500mm), wedge physical extents. A particle this
unstable would almost certainly be physically lost in the real G4BL
simulation, cleanly removed by a collimator -- not tracked forever through
vacuum with nothing to stop it. Added `aperture_radius` to
`track_with_drag` (`hfofo/background.py`): once a particle's transverse
radius exceeds it, freeze the ENTIRE state (x, y, ct, px, py, pz) from that
point on -- critically, feed the *frozen* state into subsequent
`diffrax_solve` calls (not the particle's own possibly-still-diverging
state), so a lost particle can never re-trigger the same `max_steps` issue
later. `z` still advances normally to track the requested grid.

Default aperture: 200mm, the tightest iris radius in the deck (`RFC2`'s).
This is a simplification (one uniform global value, not the real
per-element aperture map that varies 200-300mm by cavity variant, plus the
wedges' own physical extents) -- deliberately conservative, erring toward
excluding a few more particles than the true structure would rather than
under-cutting and letting a lost particle back in. A real refinement would
track which element is nearest and use its actual aperture.

Verified: particle 20 alone now completes cleanly (freezes at `r≈200.7mm`,
reached almost immediately -- by `z=-100mm`, well before the numerical
blowup region found in §1.1's bisection). The full N=24 ensemble now
completes at the SAFE tolerance (`rtol=1e-3`, no loosening needed) in
~94s total (17.5s compile + 76.5s run).

Added regression tests (`tests/test_aperture_cut.py`): a particle that
never approaches the aperture tracks identically with/without
`aperture_radius` set (freezing logic doesn't perturb the ordinary case); a
particle that starts beyond the aperture freezes immediately and stays
exactly frozen; and the specific particle-20 case is locked in directly
(loads the real `initial.dat` sample, confirms it no longer raises and ends
up correctly flagged as lost). All pass, along with the full existing suite
(21/21 total including the emittance tests below).

---

## 2. The actual physics result -- the first real answer this whole thread was building toward

With the aperture cut in place, ran the full pipeline: 24-particle
`initial.dat` sample -> INPUT eigen-emittances -> track 1 period -> exclude
frozen/lost particles -> OUTPUT eigen-emittances from survivors.

```
INPUT:  eps1=2929.06  eps2=556.77  product=1,630,806
        (naive product 3,196,377 -- confirms eigen-emittances matter here,
        ~2x difference from real x-y coupling, same as the prior handoff)

8/24 particles (33%) lost to the aperture within just 1 period.

OUTPUT (16 survivors): eps1=3839.40  eps2=327.51  product=1,257,436
        (naive product 4,072,628 -- naive/eigen ratio grew to ~3.2x,
        i.e. coupling matters MORE after 1 period than at input, not less)
```

**The eigen-emittance product decreased ~23%** (1,630,806 -> 1,257,436) for
the surviving particles -- real cooling, in the eigen-emittance sense, for
this actual initial beam distribution over one period. This is the first
genuine physics answer this whole differentiability/optimization thread has
produced.

**But the 33% loss fraction in a single period is large, and I have not
determined whether it's realistic or an artifact of the aperture choice.**
Two genuinely open possibilities, not distinguished yet:
1. It's realistic: `initial.dat` is a raw pre-acceptance capture beam (recall
   the prior handoff's note: pz spans 26.5-3781.7 MeV/c before the cut, and
   even after the `|pz-247.5|<75` cut the transverse spread is still large,
   `sigmaX=sigmaY=80mm` in the deck's own beam definition) -- real cooling
   channels often do lose a large fraction of an unmatched beam in the first
   few periods while it's captured into the channel's acceptance. A 33%
   first-period loss could be entirely physically expected.
2. It's an artifact of the deliberately conservative, uniform 200mm
   aperture (§1.3) -- the real per-element structure has apertures up to
   300mm in some regions, and this model applies the tightest value
   everywhere. A less conservative (or properly z-dependent) aperture might
   show meaningfully less loss.

**This is exactly where the next session should start** -- not by treating
either the 23% cooling number or the 33% loss number as final, but by
distinguishing which of (1)/(2) is actually going on. Concretely:
- Try a larger sample (N=48 or more) to see if the survival/cooling
  numbers stabilize or drift with sample size (24 is small for a 33% loss
  rate -- the true fraction could plausibly be anywhere from ~20% to ~45%
  at this N).
- Try a less conservative aperture (e.g. 300mm, or a genuinely z-dependent
  aperture map keyed to the nearest cavity's actual `irisRadius`) and see
  how much the loss fraction and the surviving-particle cooling number
  move.
- Track more than 1 period (2-3) to see whether the loss fraction keeps
  climbing at a similar rate (would support interpretation 1, ongoing
  capture-into-acceptance) or drops off sharply after the first period
  (would be consistent with either interpretation, less diagnostic alone).

---

## 3. What's actually in the working tree vs. what's committed

**Nothing from this session is committed yet** (matching the "Claude clones,
doesn't push" constraint -- this will need to go out as a patch, same
pattern as milestone C's deliveries). In the working tree at
`/home/claude/hfofo-beamline-fresh`:

- `pyproject.toml` / `uv.lock`: the stale-pin fix (§0). **Package this as
  its own first commit if generating patches** -- it's a correctness fix
  independent of everything else and shouldn't be bundled invisibly inside
  the emittance work.
- `src/hfofo/emittance.py`: unchanged from the prior handoff's pasted
  code -- independently re-verified this session (uncoupled analytic case,
  coupled-vs-naive divergence, the real `initial.dat` ~2x ratio, and
  `jax.jvp` differentiability all re-checked directly, not just trusted
  from the handoff). One correction made to the *documentation* (not the
  code): the prior handoff's claim that FD checking `eigen_emittances`
  needs "an appropriately-sized eps" turns out to describe differentiating
  through the *full tracking pipeline*, not this function in isolation --
  direct testing shows `eigen_emittances` alone matches FD to near machine
  precision even at `eps=1e-6`. Fixed in `test_emittance.py`'s docstring so
  a future session doesn't inherit an overcautious claim about a
  well-conditioned function.
- `tests/test_emittance.py`: the two checks already done ad hoc in the
  prior handoff, turned into real pytest tests, plus one more (the real
  `initial.dat` ~2x ratio, parameterized to skip gracefully if the file
  isn't present in a given environment) and the differentiability check.
  4/4 pass.
- `src/hfofo/background.py`: `track_with_drag` gained `aperture_radius`
  (§1.3). Backward compatible -- default `None`, existing callers
  (`track_full_channel.py`, `track_ensemble.py`) unaffected; verified via
  `test_particle_within_aperture_unaffected`.
- `tests/test_aperture_cut.py`: new, 3/3 pass (§1.3).
- `scripts/emittance_sandbox.py`: the working scaffold from the prior
  handoff's description, completed -- loads `initial.dat`, samples,
  computes INPUT emittances (`main()`, no flags), and now also completes
  the ensemble-tracking + OUTPUT-emittance comparison end to end
  (`--track`), with the aperture cut wired in and lost particles correctly
  excluded from the output covariance. Also supports `--bisect` (times
  compile/run separately across N=4/8/16/20/24) if the N-scaling
  investigation needs re-running.
- `scripts/track_full_channel.py`: one-line fix, the stale "~6 MeV" RMS
  figure the prior handoff flagged (never actually fixed in code, just
  noted) -> corrected to "~7.9 MeV".
- This document.

Once the user confirms this is the right direction, next step is
generating clean patches (one per logical change, matching the project's
established pattern -- see the pin fix note above) and delivering them, not
pushing directly.

---

## 4. Instructions for whoever picks this up next

1. Read this document, then the prior handoff for full milestone A-D
   context (only its §2.4 is superseded).
2. If working from a fresh clone rather than continuing in this exact
   sandbox: **check the `beamline` pin first** (§0) before trusting
   anything -- don't assume it's still correct.
3. Run the full test suite (`uv run pytest`) to confirm the 21/21 baseline
   before changing anything further.
4. Pick up at §2's open question: is the 33% first-period loss realistic
   or an aperture-choice artifact? Don't present either the 23% cooling
   number or the 33% loss number as final until this is resolved -- both
   are currently honest-but-provisional, not verified conclusions.
5. Once that's resolved, the differentiability step from the prior
   handoff's §2.4 (step 4: `jax.jvp` of the *output* eigen-emittance
   product with respect to a design parameter, using the `eqx.tree_at`
   override pattern already proven in the single-particle smoke test) is
   still the right next milestone-D step after this -- unblocked now that
   ensemble tracking actually works, but not yet attempted this session.
6. Same working rhythm as always: verify every claim directly rather than
   trusting a prior write-up (this session found the "compile scales with
   N" framing was wrong, the "loosen tolerance" fix was actively harmful,
   and a stale doc figure that was flagged-but-never-fixed -- all by
   checking rather than assuming); check in before expensive/scope-changing
   work; report when a hypothesis doesn't pan out rather than smoothing it
   into a tidier story than what actually happened.
