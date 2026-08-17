# Session handoff: milestone D (differentiability), eigen-emittance optimization

**Written by:** Claude, mid-task, handing off because the previous session ran
out of usage credits. This is a literal continuation point, not a summary for
its own sake -- read it as "what would the previous session tell itself."

**Repos:**
- `beamline` fork: `github.com/headunderheels/beamline`, HEAD `2013daa` (the
  physics library)
- `hfofo-beamline` app repo: `github.com/headunderheels/hfofo-beamline`, HEAD
  `873da38` at the time of this handoff (the application built on it)

Both belong to the user. Claude can clone but not push -- the user applies
patches/commits themselves.

---

## 1. Full project arc so far (milestones A-D)

- **A (lattice tracking): done.** Reference muon tracks through 187 solenoids
  + 379 cavities.
- **B (RF phasing): done.** `phase = -2*pi*f*timeOffset` in `build.py`.
- **C (absorbers): substantially done.** 171 LiH wedges, GH2 background gas,
  presswall, Be windows all modeled. Full 31-period single-particle run
  validated against the real G4BL reference trace
  (`hfofo-latest/g4bl-output/ReferenceParticle_247pt5.txt`) at **~7.9 MeV
  energy RMS / ~9.6mm radius RMS** against ~250 MeV / ~60-100mm typical
  scales. (Note: an earlier docstring claimed ~6 MeV -- that number was from a
  since-superseded commit; ~7.9 MeV is the correct current figure. If you see
  ~6 MeV anywhere, it's stale.)
- **D (differentiation/optimization): in progress -- this is the live
  thread this doc exists to hand off.**

### Upstream fixes already landed in `beamline` -- don't redo these
`SumField` summing fix; `WedgeVolume`/`AbsorberWedge` geometry;
on-axis-NaN root-fixed in `Tangent._change_basis` (this superseded an earlier,
narrower per-field fix); `PillboxCavity.mode/m/n/p` marked
`eqx.field(static=True)`; `BoundaryAwareStepSizeController.init()` was missing
`abs(sdf)`; a `stochastic_solve` retry-loop bug; `ThickSolenoid.B_shells`
shell-placement fixed to match G4BL's actual midpoint-of-bins algorithm (now
reproduces G4BL to <=0.03% with the **exact** Amp->e/ns unit conversion, no
fitting -- this superseded an earlier wrong conclusion that "a thin shell is
more physically correct than a thick coil"; the real bug was a mismatched
reference-coil geometry, not a thick-vs-thin physics difference).

### App-repo history already landed -- don't redo these
- `stack_components` (in `hfofo/stacked.py`) had a real bug: it used
  `eqx.is_array` instead of `eqx.is_array_like`, which silently routed every
  ordinary Python float (i.e. every physical parameter derived from
  `hepunits` scaling, like `current * AMP_TO_JPHI`) into the "shared/static"
  partition instead of the "per-instance" partition -- meaning **every
  solenoid was silently using solenoid #0's current, every cavity solenoid
  #0's phase and gradient.** Fixed.
- **Multiple Coulomb scattering (MCS)** was implemented
  (`hfofo/background.py`: `highland_theta0_squared`, `apply_scattering_kick`,
  Highland formula, momentum-magnitude-exact kick), tested via a 3-seed
  vmapped ensemble, and found to **not help** the ~8 MeV residual -- it made
  the ensemble-mean residual larger, not smaller. Left in the code but **off
  by default**; opt in via `HFOFO_MCS=1` on `track_full_channel.py`.
- **Windowed (locality-limited) field/material builders** added for a ~5-6x
  speedup: `build_channel_batched_windowed` / `build_wedges_windowed` in
  `build.py`, selecting the K nearest elements (defaults
  `K_SOLENOIDS_LOCAL=24`, `K_CAVITIES_LOCAL=44`, `K_WEDGES_LOCAL=22`) instead
  of vmapping over all 566 EM elements + 171 wedges regardless of location.
  **On by default**; `HFOFO_NO_WINDOW=1` disables it.
  **Important limitation for the differentiability work below:** the
  K-nearest selection is a plain Python `sorted(...)` call -- it happens
  before JAX ever traces anything, which is what makes it fast, but it also
  means it is **not differentiable**. Gradients with respect to an already-
  selected element's *value* (current, phase, wedge width) are fine.
  Gradients with respect to *position* (which would change which window an
  element falls into) are **not correctly captured** by the windowed
  builders -- use the unwindowed `build_channel_batched`/`build_wedges` for
  that specific class of parameter if it's ever needed.
- A persistent JAX compilation cache was attempted and **deliberately not
  pursued**: diffrax's default `throw=True` runtime check (which raises via
  an `equinox.error_if` host callback on max-steps-exceeded / non-finite
  results) makes the compiled graph fundamentally uncacheable, confirmed via
  `jax_explain_cache_misses`. Disabling that check to gain caching would trade
  away a safety net that already caught a real bug (the beta->0/max_steps
  degeneracy behind `apply_energy_loss`'s mass clamp). Documented, not fixed.
  The practical workaround with no such tradeoff: run as one long-lived
  process rather than many short resumed invocations -- the ordinary
  in-memory JAX cache already reuses the compiled function across periods
  within a single process.
- Two clean git patches for the MCS + windowing work were generated,
  verified (applied to a fresh clone, byte-identical result, 14/14 tests
  pass), and already delivered to the user in a prior turn. **Don't
  regenerate these** -- they're the commits already at HEAD `873da38`
  (`33daca6` MCS, `873da38` windowing).
- **The ~6->~7.9 MeV residual was investigated at length.** Findings: it's a
  **tune-mismatch beat** (peaks in the *middle* third of the channel, periods
  6-17, and partially resolves by period 30 -- **not** a monotonic drift, as
  an earlier commit's docstring claimed). Two single-variable hypotheses were
  tested directly and **ruled out**: RF phase drift (measured against G4BL's
  actual `t` column -- bounded, weakly correlated, r=-0.14) and
  transverse-position-driven wedge loss (r=0.035, essentially uncorrelated).
  MCS was also ruled out as a fix (see above). Best current understanding: a
  small aggregate lattice-tune mismatch that shows up as a beat in *both* the
  transverse envelope and the longitudinal energy simultaneously (they're
  coupled through path-length: larger transverse amplitude -> longer helical
  path -> different time-of-flight -> different effective RF phase ->
  different energy -> different focusing strength, since focusing scales
  roughly as 1/p^2 -> different tune -> closes the loop). Root cause not
  fully identified. **The user explicitly decided to stop chasing this and
  accept current fidelity for milestone C.** Don't reopen it unless asked.

---

## 2. Milestone D -- exactly where this session left off

The user's stated goal: **reduce the phase-space of a muon beam**, using
**eigenmode (normal-mode) emittances** rather than naive projected x/y
emittances -- the right call, because this is a helically-coupled solenoidal
channel (the tilted/rotating solenoids couple the x and y planes), and naive
projected emittances ignore that coupling.

### 2.1 Established, verified facts about differentiability -- don't redo these checks

1. **`jax.grad` (reverse-mode AD) fails on this pipeline outright.**
   Confirmed by direct error: `ValueError: Reverse-mode differentiation does
   not work for lax.while_loop or lax.fori_loop with dynamic start/stop
   values.` Diffrax's adaptive step-size control uses a `lax.while_loop`
   whose length isn't known ahead of time; reverse-mode AD fundamentally
   can't differentiate through that.
2. **`jax.jvp` (forward-mode AD) works correctly.** Verified against
   finite-difference to ~machine precision (0.0000% relative difference),
   both through the bare `diffrax_solve` call and through the full
   `track_with_drag` (including wedges + GH2 background), differentiating
   with respect to one solenoid's current, using a final transverse radius
   as the merit. **Conclusion, load-bearing for everything downstream: all
   gradient work on this pipeline must use forward-mode AD (`jax.jvp` /
   `jax.jacfwd`), never `jax.grad`.**
3. **`jnp.linalg.eig` (general, non-symmetric eigendecomposition) is not
   differentiable in JAX at all** -- raises
   `NotImplementedError: Derivatives of non-symmetric eigenvectors are only
   valid under assumptions...`. Confirmed directly with a minimal repro.
   This matters because it's the "obvious" way to compute eigen-emittances,
   and it's a dead end.

### 2.2 The differentiable eigen-emittance formula -- built, verified, ready to use

Derivation (Williamson's theorem applied to a 4D transverse phase space):
for a positive-definite 4x4 covariance Sigma and the canonical symplectic
form S (order x, px, y, py), the eigenvalues of M = Sigma.S are purely
imaginary pairs +-i*eps1, +-i*eps2. Since tr(M) = 0 always (S is
antisymmetric, Sigma is symmetric -- a short cyclic-trace argument shows
tr(Sigma.S) = -tr(Sigma.S) => tr(Sigma.S) = 0), the characteristic
polynomial reduces to lambda^4 + c2*lambda^2 + det(M) = 0, giving:

```
eps1^2 + eps2^2 = c2 = -trace(M @ M) / 2
eps1 * eps2      = sqrt(det(M))
```

Both use only `trace`, `matmul`, `det` -- all differentiable, no
eigendecomposition needed. See `src/hfofo/emittance.py` for the
implementation (added this session -- verified independently, see the
follow-up handoff `SESSION_HANDOFF_2026-08-17_aperture_cut.md`).

Verification already done (don't redo, but re-verified independently in the
follow-up session, see the aperture-cut handoff):
- **Exact match** against the known analytic uncoupled case (block-diagonal
  Sigma reduces to ordinary epsx = sqrt(det(Sigmax)), epsy =
  sqrt(det(Sigmay))).
- **Correctly diverges from naive projected emittances under real coupling**
  -- this is the entire point of using eigen-emittances here, and it's
  numerically real, not just theoretical (see S2.3).
- **Differentiates correctly via `jax.jvp`**, matching finite-difference at
  an appropriately-sized step. NOTE (corrected in the follow-up handoff):
  the FD-conditioning sensitivity described here is specific to
  differentiating through the *full tracking pipeline* -- the
  `eigen_emittances` function alone, tested in isolation, matches FD to near
  machine precision even at very small step sizes. Don't apply the "pick an
  appropriately-sized eps" caution to that function specifically; it does
  apply to gradients through the ODE solve.

### 2.3 Real initial beam data -- located, loaded, sampled

File: `/home/claude/muon-cooling/hfofo-frozen/g4bl-input/initial.dat` (also
identical copy at `input-beam-files/initial.dat`). Format:
`x y z Px Py Pz t PDGid EventID TrackID ParentID Weight` (mm, mm, mm, MeV/c,
MeV/c, MeV/c, ns, ...). 12471 total rows; PDGid in {-13 (mu+), 211 (pi+)};
11755 mu+ rows, all at a fixed z=101750mm plane (a beam snapshot, not
distributed in z -- G4BL's `beam ascii file=initial.dat beamZ=$beamstart`
directive relocates it to the tracking start, which is why the z column
should be discarded and every particle placed at `BEAM_START = -700mm` in
our tracking convention).

**`pz` spread is enormous**: 26.5 to 3781.7 MeV/c (mean 401.5, std 297.6) --
this is a raw pre-acceptance capture beam; a real G4BL run loses the extreme
tails to apertures/kill-volumes we haven't modeled at all. Feeding raw
outliers into our simplified channel risks numerical nonsense unrelated to
the actual physics question. **Decision made (my judgment call, stated to
the user but not separately re-confirmed -- worth flagging if you want to
revisit it):** applied a cut `|pz - 247.5| < 75 MeV/c` before sampling (7200
of 11755 mu+ rows pass).

Drew a **reproducible 24-particle sample**:
```python
rng = np.random.default_rng(0)
idx = rng.choice(len(mu_cut), size=24, replace=False)
```
This was saved to `/tmp/initial_sample.npy`, which is **not guaranteed to
persist** across sessions/sandboxes -- regenerate with the exact code above
(load `initial.dat`, filter `PDGid==-13`, apply the pz cut, sample with
`seed=0`, `size=24`) if it's gone. The full loading/filtering/sampling code
is reproduced in S5 for convenience.

**Computed INPUT eigen-emittances from this real 24-particle sample:**
eps1=2929.06, eps2=556.77, product=1,630,806 mm*MeV/c -- versus the naive
projected product of 3,196,377. **The naive number is ~2x the true one** --
this confirms, on this actual channel's actual input distribution (not a
toy example), that eigen-emittances are the right tool, not just in
principle but by a large margin numerically.

### 2.4 [SUPERSEDED -- see SESSION_HANDOFF_2026-08-17_aperture_cut.md]

The live blocker described in this section at the time of the original
handoff (ensemble-tracking compile cost, apparently scaling badly with N)
has been resolved -- and the diagnosis in this section turned out to be
wrong in an important way: it was never a compile-time scaling problem.
See the follow-up handoff document for the full story, including a genuine
physics result (input vs. output eigen-emittances on the real 24-particle
sample) and a still-open question about aperture-loss realism. Read that
document instead of continuing from this section.

---

## 5. Reference code (already verified working, for direct reuse)

### Loading, filtering, and sampling the initial beam

```python
import numpy as np
d = np.genfromtxt('/home/claude/muon-cooling/hfofo-frozen/g4bl-input/initial.dat', comments='#')
mu = d[d[:, 7] == -13]  # PDGid == -13 is mu+
x, y, z, px, py, pz, t = mu[:,0], mu[:,1], mu[:,2], mu[:,3], mu[:,4], mu[:,5], mu[:,6]
cut = np.abs(pz - 247.5) < 75.0
mu_cut = mu[cut]
rng = np.random.default_rng(0)
idx = rng.choice(len(mu_cut), size=24, replace=False)
sample = mu_cut[idx]
```

### Building the JAX ensemble state from the sample

```python
import jax, jax.numpy as jnp, hepunits as u
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.kinematics import MuonStateDz

BEAM_START = -700.0 * u.mm
sx, sy, spx, spy, spz, st = sample[:,0], sample[:,1], sample[:,3], sample[:,4], sample[:,5], sample[:,6]
ct0 = st - st.mean()  # center so mean ct=0 matches the reference-particle convention

def make_state(x, y, px, py, pz, ct):
    return MuonStateDz.make(
        position=Cartesian4.make(x=x*u.mm, y=y*u.mm, z=BEAM_START, ct=ct*u.c_light),
        momentum=Cartesian3.make(x=px*u.MeV, y=py*u.MeV, z=pz*u.MeV), q=1,
    )

state0 = jax.vmap(make_state)(
    jnp.array(sx), jnp.array(sy), jnp.array(spx), jnp.array(spy), jnp.array(spz), jnp.array(ct0)
)
```

### The one-parameter differentiability smoke-test pattern (proven to work; reuse for the output-emittance gradient)

```python
import equinox as eqx
from hfofo.stacked import StackedField, BatchedChannel
from hfofo.build import build_channel_batched_windowed, AMP_TO_JPHI

def merit(current0):
    ch = build_channel_batched_windowed(lattice, z_center=1750.0)
    sol_group = ch.groups[0]
    new_jphi = sol_group.stack.field.jphi.at[0].set(current0 * AMP_TO_JPHI)
    new_stack = eqx.tree_at(lambda s: s.field.jphi, sol_group.stack, new_jphi)
    new_channel = BatchedChannel(groups=[StackedField(stack=new_stack), ch.groups[1]])
    # ... track, compute merit from the result ...
    return merit_value

val, tangent = jax.jvp(merit, (current_nominal,), (1.0,))
# compare against a central finite-difference at a few step sizes, NOT just one
```
