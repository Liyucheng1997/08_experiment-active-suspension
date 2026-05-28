# Experiment Plan: Risk-Aware Active Suspension Control with Super-Twisting Normal-Load Estimation

> Implementation plan for the experiments of the paper "Risk-Aware Active Suspension Control for Tire Friction Margin Protection Using Super-Twisting Normal Load Estimation" (IEEE Access target).

---

## 0. Design Philosophy

**Three orthogonal complexity dimensions** — never advance more than one at a time:

| Dimension | Progression |
|---|---|
| Plant model | quarter-car → half-car → full-car → CarSim |
| Algorithm | passive → STO alone → comfort QP → risk-aware QP → risk-aware MPC |
| Scenario | bump / sine → random road → cornering → cornering + road → braking + cornering + road |

**Core principles**

- Each step advances one feature on the simplest sufficient model. Other two dimensions retreat to their simplest. Any failure has a single cause.
- Modules built early (e.g., quarter-car STO) are reused unchanged later.
- Every algorithm step is validated against a known truth (analytical or injected) before being chained.
- Every step produces: code + unit test + diagnostic figure + one metric number.
- Each step states what to check first if it fails — making downstream debugging O(1).

---

## 1. Repository Conventions

```
repo/
├── src/
│   ├── plants/          # quarter_car.py, half_car.py, full_car.py
│   ├── observers/       # sto.py
│   ├── controllers/     # passive.py, skyhook.py, lqr.py,
│   │                    # comfort_qp.py, risk_qp.py, risk_mpc.py
│   ├── inputs/          # road.py, maneuver.py
│   ├── metrics/         # signals.py, tire.py
│   ├── solvers/         # qp_harness.py
│   ├── utils/           # logger.py, config.py, plot.py
│   └── scenarios/       # scenario_lib.py
├── tests/               # pytest, mirrors src/
├── configs/             # YAML for vehicle / controller / scenario specs
├── results/             # per-run output, gitignored
└── notebooks/           # exploration only, not in CI
```

Per-run output layout:

```
results/<phase>_<step>_<descriptor>_<timestamp>/
├── log.txt
├── config.yaml         # snapshot of parameters used
├── metrics.csv         # tabulated metrics
├── figures/*.pdf
└── raw.npz             # full state, input, observer traces
```

---

## 2. Phase Summary

| Phase | Name | # Steps |
|---|---|---|
| 0 | Engineering Scaffolding | 4 |
| 1 | Plant Open-Loop Verification | 6 |
| 2 | STO Development | 7 |
| 3 | Baseline Controllers | 5 |
| 4 | Risk-Aware One-Step QP | 8 |
| 5 | Risk-Aware MPC | 5 |
| 6 | Full-Car Scenario Sweep | 8 |
| 7 | CarSim Co-Simulation | 6 |
| **Total** | | **49** |

**Cross-phase dependencies**

```
0  ──► 1  ──► 2  ──┐
                   ├──► 4 ──► 5 ──► 6 ──► 7
       1  ──► 3  ──┘
```

Any failure in a later phase that does *not* reproduce on the earlier phase points to an integration bug, not a component bug.

---

## 3. Detailed Steps

Each step has four fields:

- **Purpose** — what question this step answers.
- **Implementation** — the concrete coding action.
- **Acceptance** — the pass condition.
- **Fallback** — the first place to look if it fails.

---

### Phase 0 — Engineering Scaffolding

**0.1 Repository skeleton, dependencies, and test framework**

- *Purpose*: Establish directory layout, pinned dependencies, and pytest infrastructure.
- *Implementation*: Create directories per Section 1. Pin `numpy`, `scipy`, `cvxpy`, `osqp`, `matplotlib`, `pytest`, `pyyaml`, `dataclasses`. Add `conftest.py` with a `default_vehicle` fixture loading `configs/vehicle_default.yaml`.
- *Acceptance*: `pip install -e .` succeeds; `pytest tests/` returns green (0 tests fine).
- *Fallback*: cvxpy ↔ osqp version conflicts — pin `cvxpy>=1.4` and let it resolve osqp.

**0.2 Parameter system**

- *Purpose*: Centralize all parameters in YAML loaded into typed dataclasses, eliminating silent typos.
- *Implementation*: `src/utils/config.py` with `VehicleParams`, `ControllerParams`, `ObserverParams` dataclasses; `from_yaml(path)` constructor. Provide one default mid-size sedan (m≈1500 kg, l_f=1.2, l_r=1.4, h_g=0.55, t=1.55, k_s=30 kN/m, c_s=2 kN·s/m, m_u=50, k_t=200 kN/m, I_x=600, I_y=2400 kg·m²).
- *Acceptance*: Unit test loads YAML, asserts each field type and reasonable bounds.
- *Fallback*: YAML int-vs-float mismatch is the most common loader failure.

**0.3 Logger, metrics, and plotting templates**

- *Purpose*: Standardize result recording across all steps.
- *Implementation*:
  - `utils/logger.py`: per-run logger writing `log.txt` + `metrics.csv`; structured `log_kv(key, value)`.
  - `metrics/signals.py`: RMSE, peak abs, IAE, ITAE, 95-percentile.
  - `utils/plot.py`: `plot_timeseries`, `plot_freq_response`, `plot_side_by_side` with consistent fonts/colors via shared `rcParams`.
- *Acceptance*: Unit tests against hand-computed metric values; one demo figure produced.
- *Fallback*: Inconsistent styling between runs — `rcParams` must be set inside `plot.py` import, not in user code.

**0.4 Road and maneuver generators**

- *Purpose*: Reproducible road profiles and driver inputs for all scenarios.
- *Implementation*:
  - `inputs/road.py`: `sine`, `rounded_bump`, `iso8608(class_, v_x, t, seed)` per ISO 8608 PSD, `single_wheel_bump` (3 channels zero-padded).
  - `inputs/maneuver.py`: `straight`, `single_lane_change`, `double_lane_change`, `step_steer`, `j_turn`, `emergency_brake`; returns `(t, a_x, delta_f)`.
- *Acceptance*: Class-B road PSD matches ISO 8608 in 0.5–20 Hz; single-wheel bump is exactly zero on the other 3 channels.
- *Fallback*: Flat PSD ⇒ missing `1/Ω²` spatial-frequency scaling.

---

### Phase 1 — Plant Open-Loop Verification

**1.1 Quarter-car class and integrator**

- *Purpose*: A validated quarter-car model, the simplest test bed for downstream STO and controller work.
- *Implementation*: `QuarterCar(params)` with state `[z_s, dot z_s, z_u, dot z_u]`, matrices `A, B, E`, `step(x, u, w, dt)` using fixed-step RK4; `simulate(t, u_seq, w_seq)` batch helper.
- *Acceptance*: Hand-derived `A` matches numerical `A` element-wise; unforced energy decays monotonically; eigenvalues of `A` give correct sprung/unsprung modes.
- *Fallback*: Sign error in damping/stiffness is the typical bug — compare row-by-row.

**1.2 Quarter-car validation suite**

- *Purpose*: Confirm quarter-car physics under free vibration, frequency sweep, and standard bump.
- *Implementation*: (a) free vibration from `z_s(0)=0.05`; (b) chirp 0.5–25 Hz on `z_r`; (c) ISO step bump.
- *Acceptance*: ω_n within 2% of `sqrt(k_s/m_s)`; ζ within 5%; transmissibility shows sprung + unsprung peaks; tire force ≥ 0 throughout bump.
- *Fallback*: Negative tire force on small bump ⇒ k_t wrong or RK4 step too coarse.

**1.3 Half-car class and decoupling tests**

- *Purpose*: Validate roll dynamics and heave/roll decoupling.
- *Implementation*: `HalfCar(params)` with 8 states. Apply symmetric road both sides (heave-only excitation), then anti-symmetric (roll-only).
- *Acceptance*: Symmetric: `max|phi(t)| < 1e-6`. Anti-symmetric: `max|z_s(t)| < 1e-6`. Roll natural frequency matches analytical.
- *Fallback*: Coupling leakage ⇒ asymmetry in geometry or `k_s`; assert symmetry in unit tests.

**1.4 Full-car class, static load, and decoupling**

- *Purpose*: Validate the 14-state full-car model.
- *Implementation*: `FullCar(params)` with state `[z_s, dot z_s, phi, dot phi, theta, dot theta, {z_u_ij, dot z_u_ij}_4]`. At rest: compute per-corner static F_z. Cross-decoupling test via symmetric / anti-symmetric road combinations.
- *Acceptance*: `Σ F_z = mg` to 1e-3 N; front/rear split matches `mg·l_r/L`, `mg·l_f/L`; heave/roll/pitch cross-coupling < 1% in linear regime.
- *Fallback*: Unequal left/right static loads ⇒ track-width sign convention in `y_ij`.

**1.5 Full-car load transfer verification**

- *Purpose*: Confirm quasi-static load transfer formulas match the dynamic model.
- *Implementation*: Apply pure `a_x = 5 m/s²` step; separately pure `a_y = 5 m/s²`. Measure steady-state ΔF_z.
- *Acceptance*: Longitudinal `ΔF_z_front ≈ -mh_g·a_x/L` within 5%. Lateral `ΔF_z_left/right ≈ ±mh_g·a_y/t` within 5%.
- *Fallback*: Factor-of-2 error ⇒ per-axle vs per-wheel split.

**1.6 Full-car long run on rough road**

- *Purpose*: Confirm realistic ride dynamics over a representative driving scenario.
- *Implementation*: Straight 80 km/h on ISO class-B road for 60 s. Record vertical accel, suspension strokes, tire forces.
- *Acceptance*: Vertical accel RMS within typical passenger-car range (0.3–0.6 m/s² on class B); no stroke violations.
- *Fallback*: High RMS ⇒ damping ratio off; check `c_s`.

---

### Phase 2 — Super-Twisting Observer Development

**2.1 STO class skeleton and ideal (sgn) implementation**

- *Purpose*: Implement the canonical super-twisting observer per paper eqs. (sto_v)–(sto_chi).
- *Implementation*: `STO(params)` with state `[v_u_hat, chi_hat]`, method `step(v_u_meas, phi_known, dt) → (d_z_hat, F_z_hat)` using discrete Euler. Gains `λ_1, λ_2` configurable.
- *Acceptance*: At zero injected disturbance, `|chi_hat| < 1e-6` after transient. Interface plugs cleanly into Phase 1 plant.
- *Fallback*: Drift at zero disturbance ⇒ `phi` known-term computation wrong (typically `F_z_bar` inconsistency).

**2.2 Quarter-car STO with synthetic disturbance**

- *Purpose*: Validate finite-time convergence under known ground-truth `d_z`.
- *Implementation*: Inject `d_z = 200 N` (constant) and `d_z = 200 sin(2π·3 t) N` (sinusoidal). Compare `d_z_hat` against truth.
- *Acceptance*: Constant: settling < 0.2 s, steady-state error < 1%. Sinusoidal: tracking RMSE < 5% of amplitude after transient.
- *Fallback*: Slow convergence ⇒ λ_1, λ_2 too small. Oscillation ⇒ λ_2 too large relative to `|dot chi|` bound.

**2.3 STO gain tuning sweep**

- *Purpose*: Identify a robust default `(λ_1, λ_2)`.
- *Implementation*: Grid sweep `λ_1, λ_2 ∈ [1, 100]` on the sinusoidal test from 2.2. Record settling time and steady-state RMSE.
- *Acceptance*: Heatmap produced; default operating point selected (RMSE < 5%, settling < 0.1 s).
- *Fallback*: All-unstable region ⇒ discretization too coarse for chosen gains; ensure `λ_1·dt` is small.

**2.4 STO under real road excitation**

- *Purpose*: Validate STO against d_z generated by the plant under road input (no injection).
- *Implementation*: Use ISO class-B road; truth is `k_t(z_u - z_r) - (terms in F_z_bar)`.
- *Acceptance*: `d_z_hat` tracks the truth residual; cross-correlation > 0.9.
- *Fallback*: Poor correlation ⇒ `F_z_bar` definition includes/excludes terms inconsistently — re-check eq. (fz_decomposition).

**2.5 Measurement noise and saturation implementation**

- *Purpose*: Make the STO practical under sensor noise.
- *Implementation*: Add Gaussian noise on `v_u` at SNR = 20 dB. Show ideal-sgn STO chatters. Replace sgn with `sat(e_v/ε)`; sweep ε.
- *Acceptance*: Saturation eliminates chatter; `|e_chi|` converges to a quantifiable neighborhood whose size scales with ε.
- *Fallback*: Loss of speed under saturation ⇒ ε too large; tune until `e_chi` settles within one suspension natural period.

**2.6 STO error bound calibration (ΔF_z^err)**

- *Purpose*: Provide the controller with the empirical bound used for constraint tightening.
- *Implementation*: For default `(λ_1, λ_2, ε)` and a small sweep of noise/sampling conditions, measure `ē_chi`; fit `ΔF_z^err = m_u · ē_chi` lookup.
- *Acceptance*: Predicted vs measured bound within 30% across the sweep.
- *Fallback*: Bound too loose ⇒ over-conservative controller. Too tight ⇒ ρ violations later. Recalibrate on worst-case row.

**2.7 Multi-wheel STO on half/full-car**

- *Purpose*: Confirm per-wheel independence and joint correctness under coupled road + load transfer.
- *Implementation*: 4 STO instances. Test (a) half-car single-wheel bump → only that wheel responds; (b) full-car class-B road + lateral acc step → STO does *not* absorb the quasi-static lateral transfer.
- *Acceptance*: (a) off-wheel `d_z_hat` < 5% of on-wheel. (b) `d_z_hat` near-zero mean over the maneuver.
- *Fallback*: Drift under lateral accel ⇒ missing `ΔF_z_lat` in `F_z_bar`.

---

### Phase 3 — Baseline Controllers

**3.1 Passive baseline**

- *Purpose*: Trivial `F_e ≡ 0` reference under the same interface as actives.
- *Implementation*: `PassiveController(params)` with `compute(state, ...) → zeros(4)`.
- *Acceptance*: Bit-identical to plant-only Phase 1 run.
- *Fallback*: Mismatch ⇒ controller-plant interface leaking state or feedthrough.

**3.2 Skyhook controller (quarter + full-car)**

- *Purpose*: Simplest active comfort baseline.
- *Implementation*: Quarter: `F_e = -c_sky · dot z_s`. Full-car: per-corner using sprung-corner velocity `dot z_s + x_ij · dot theta + y_ij · dot phi`. Tune `c_sky` on quarter-car.
- *Acceptance*: Vertical accel RMS on class B reduced ≥ 20% vs passive (quarter); heave + roll + pitch RMS all reduced (full-car).
- *Fallback*: Excessive stroke increase ⇒ add `|F_e| ≤ F_max` saturation.

**3.3 LQR comfort controller (quarter + full-car)**

- *Purpose*: Stronger linear comfort baseline; the unconstrained optimum for the QP to recover.
- *Implementation*: Discrete-time DARE via `scipy.linalg.solve_discrete_are`. `Q` on `[ddot z_s, phi, dot phi, theta, dot theta]`, `R` on `F_e`. Document weights in YAML.
- *Acceptance*: Closed-loop poles inside unit circle; comfort metric beats Skyhook by ≥ 10% on class B.
- *Fallback*: DARE non-convergence ⇒ system uncontrollable from `F_e` in chosen state representation; recheck `B_d`.

**3.4 QP solver harness and comfort-only QP**

- *Purpose*: Stand up the QP infrastructure that risk-aware QP will extend.
- *Implementation*:
  - `solvers/qp_harness.py`: wrap OSQP via `cvxpy` Parameter pattern (re-solve, not rebuild); warm start; timing report.
  - `controllers/comfort_qp.py`: `min ‖G_a u_s + a_0‖²_{Q_a} + u_s^T R_u u_s` subject to actuator bounds.
- *Acceptance*: Single solve < 1 ms on dev hardware; unconstrained limit matches LQR within numerical tolerance.
- *Fallback*: Solve time > 10 ms ⇒ QP being rebuilt each call; switch to `cvxpy` Parameters or direct OSQP.

**3.5 Baseline comparison table**

- *Purpose*: Canonical reference table for paper Section V.
- *Implementation*: Run passive / Skyhook / LQR / comfort-QP on three scenarios: class B straight, class D straight, single lane change on class B.
- *Acceptance*: CSV output; ordering passive > Skyhook > LQR ≈ comfort-QP on the comfort metric (worst→best).
- *Fallback*: LQR loses on some metric ⇒ re-tune `Q` to match Skyhook's implicit weighting.

---

### Phase 4 — Risk-Aware One-Step QP

**4.1 Tire utilization and required-normal-load metrics**

- *Purpose*: Implement the friction-margin metrics.
- *Implementation*: `metrics/tire.py`:
  - `rho(F_x, F_y, F_z, mu)`
  - `rho_max(F_x_arr, F_y_arr, F_z_arr, mu_arr)`
  - `F_z_required(F_c, mu, rho_safe)`
- *Acceptance*: Hand-computed unit tests; `F_z_required(F_c=0)=0`; `rho` clamps `F_z ≥ F_z_min` to avoid NaN at low speed.
- *Fallback*: NaN propagation ⇒ missing F_z floor.

**4.2 Risk weighting functions**

- *Purpose*: Implement the sigmoid scheduler and global/local weights.
- *Implementation*: `controllers/risk_weights.py`:
  - `sigma_rho(rho_max, rho_th, k_rho)` with input clipping.
  - `q_c(sigma), q_p(sigma)` linear interpolation `[min, max]`.
  - `q_p_local(sigma, rho_ij, rho_th, kappa_rho)`.
- *Acceptance*: Monotonicity, smoothness, limit behavior verified. `sigma(rho_th)=0.5`. With `q_p ≡ 0`, cost reduces to comfort-only.
- *Fallback*: `exp` overflow for large negative arg ⇒ clip input before `exp`.

**4.3 Risk-aware QP cost re-parameterization with slack**

- *Purpose*: Extend the comfort QP to accept time-varying weights and a slack variable.
- *Implementation*: `controllers/risk_qp.py`:
  - Decision `[u_s; xi]`, `xi ∈ R^4`.
  - Cost `q_c · ‖G_a u_s + a_0‖² + Σ q_{p,ij} xi_ij² + u_s^T R_u u_s + (u_s − u_s^-)^T R_du (u_s − u_s^-)`.
  - Constraints: `xi ≥ 0`, actuator bounds, rate bounds.
- *Acceptance*: With `q_p ≡ 0` and no margin constraint, solution matches Phase 3.4. With no margin constraint active, `xi* = 0`.
- *Fallback*: Startup infeasibility ⇒ `u_s_prev` uninitialized; default to zeros.

**4.4 Margin constraint with ground-truth F_z (quarter-car)**

- *Purpose*: Validate the margin-protection mechanism in isolation (no STO yet).
- *Implementation*: Add `F_z_true + γ F_e + xi ≥ F_z_required` per wheel. Test scenario: lateral demand + bump on a quarter-car-equivalent.
- *Acceptance*: With sufficient actuator authority, `rho_actual ≤ rho_safe` throughout; without it, `xi > 0` quantifies violation.
- *Fallback*: Constraint never binds ⇒ scenario insufficiently aggressive; raise `a_y` or lower `μ`.

**4.5 Full-car closed-loop with ground-truth F_z**

- *Purpose*: Validate risk-aware QP on full-car under cornering + outer-wheel bump.
- *Implementation*: 80 km/h, J-turn at `a_y = 6 m/s²`, 60 mm × 30 ms bump on outer-front wheel mid-turn.
- *Acceptance*: Peak `rho_max` reduced ≥ 10% vs comfort-QP (3.4); comfort degradation in the same window < 30%.
- *Fallback*: Reduction < 5% ⇒ local weight not amplifying on the critical wheel. Comfort drop > 50% ⇒ `q_p^max` too high.

**4.6 Connect STO output to the QP**

- *Purpose*: First end-to-end risk-aware run with STO estimate.
- *Implementation*: Pipe `F_z_hat` from Phase 2 STO into the risk-aware QP. Recompute `rho_hat`, `rho_max_hat`, `q_p`, `q_{p,ij}` each step.
- *Acceptance*: Result on 4.5's scenario within 5% of the ground-truth run on peak ρ and comfort metric.
- *Fallback*: Divergence ⇒ log `F_z_true` vs `F_z_hat`; STO may be lagging, raise λ_1, λ_2 or shorten controller dt.

**4.7 Add observer-error tightening**

- *Purpose*: Make the controller robust to STO estimation error.
- *Implementation*: Use `F_z_safe = F_z_hat − ΔF_z^err` (from 2.6) in the margin constraint. Re-run 4.6's scenario with intentionally degraded STO (smaller λ values).
- *Acceptance*: `rho_actual ≤ rho_safe` throughout, even with degraded STO; or violations bounded by the tightening.
- *Fallback*: Excessive conservatism ⇒ `ΔF_z^err` overestimated; recalibrate 2.6 with the actually deployed STO config.

**4.8 Risk-aware QP parameter sweep + saturation stress test**

- *Purpose*: Characterize design space and stress-test against actuator saturation.
- *Implementation*: (a) small grid sweep of `k_rho, rho_th, rho_safe, kappa_rho`; produce Pareto plot comfort vs `peak rho_max`. (b) Push scenario beyond actuator limits.
- *Acceptance*: Pareto plot has a clean knee; saturation test does not destabilize the system; `xi*` non-zero and scales with severity.
- *Fallback*: Monotone sweep (no knee) ⇒ threshold outside explored range; extend the sweep.

---

### Phase 5 — Risk-Aware MPC

**5.1 Discretization and prediction matrix stack**

- *Purpose*: Lift the suspension dynamics to a finite-horizon prediction.
- *Implementation*: `controllers/risk_mpc.py`:
  - ZOH discretization `(A_s, B_s, E_s) → (A_d, B_d, E_d)` at sampling `T_s`.
  - Stack `bar_A, bar_B, bar_E` over horizon `N_p` so that `X = bar_A x_0 + bar_B U + bar_E W`.
- *Acceptance*: At `T_s = 1 ms`, discrete vs continuous trajectories match within numerical tolerance; stacked rollout matches step-by-step rollout exactly.
- *Fallback*: Shape errors in `bar_B` ⇒ off-by-one in horizon indexing.

**5.2 Stacked cost and horizon constraints**

- *Purpose*: Build the QP form of the MPC.
- *Implementation*: `J = U^T H U + 2 f^T U` with time-varying `q_c, q_{p,ij}`; horizon constraints (actuator, rate, stroke, F_z > 0); margin constraint with slack per `(ij, ℓ)`.
- *Acceptance*: At `N_p = 1`, the MPC reproduces Phase 4.7's QP exactly (within solver tolerance).
- *Fallback*: `N_p = 1` mismatch ⇒ weight or rate-bound shift between steps is wrong.

**5.3 Residual prediction modes**

- *Purpose*: Provide both random-walk and decay residual prediction.
- *Implementation*: Two modes — (a) freeze `d_z_hat` over horizon; (b) decay `d_z_{k+ℓ+1} = α_d · d_z_{k+ℓ}` with configurable `α_d ∈ [0.9, 1.0]`.
- *Acceptance*: Both modes produce stable closed-loop runs on 4.5's scenario; comparison plot generated.
- *Fallback*: Decay instability ⇒ `α_d` too small relative to actual residual time constant.

**5.4 Warm start and solver timing benchmark**

- *Purpose*: Make the MPC real-time capable.
- *Implementation*: Warm start each solve with previous solution shifted by one step. Benchmark per-solve time for `N_p ∈ {1, 5, 10, 20}`.
- *Acceptance*: Warm start reduces solver iterations ≥ 30%; `N_p = 10` solves within `T_s = 5 ms` (or document the largest feasible `N_p`).
- *Fallback*: Non-linear timing growth with `N_p` ⇒ QP being re-factorized; ensure OSQP reuses KKT factorization.

**5.5 MPC vs one-step QP comparison**

- *Purpose*: Confirm MPC matches or improves on QP on the canonical scenario.
- *Implementation*: Re-run 4.7's scenario with MPC at `N_p ∈ {5, 10, 20}`; same plant, STO, weights.
- *Acceptance*: MPC at `N_p = 10` matches QP on comfort and reduces peak `rho_max` by an additional ≥ 5% in the bump-during-cornering case.
- *Fallback*: MPC underperforms QP ⇒ prediction model mis-aligned; re-verify 5.1's discrete-vs-continuous match.

---

### Phase 6 — Full-Car Scenario Sweep

**6.1 Low-risk scenarios (S1–S3)**

- *Purpose*: Confirm comfort performance in non-critical maneuvers.
- *Implementation*: S1 = class B + straight + 80 km/h; S2 = smooth + single lane change; S3 = smooth + J-turn at moderate `a_y`. Run all controllers.
- *Acceptance*: `σ_ρ ≈ 0` throughout; risk-aware controllers match comfort-QP (no unnecessary protection activation).
- *Fallback*: Spurious risk activation ⇒ `rho_th` set too low.

**6.2 Rough-road cornering (S4, S5)**

- *Purpose*: Test the STO and the controller under sustained excitation combined with cornering.
- *Implementation*: S4 = class C + steady-state cornering (`a_y = 4 m/s²`); S5 = class C + lane change.
- *Acceptance*: STO produces non-zero `d_z_hat` consistent with class-C PSD; risk-aware controllers handle combined excitation without exceeding `rho_safe`.
- *Fallback*: Persistent `xi > 0` in S4 ⇒ actuator authority insufficient — document as limitation.

**6.3 Emergency braking scenarios (S6, S7)**

- *Purpose*: Friction-margin protection under longitudinal load transfer.
- *Implementation*: S6 = smooth + emergency brake (`a_x = -8 m/s²`); S7 = smooth + emergency brake during cornering (`a_x = -6, a_y = 4 m/s²`).
- *Acceptance*: S6: rear F_z drop captured, no instability. S7: `rho_max > rho_th` triggers risk mode; risk-MPC keeps `rho ≤ rho_safe` where comfort-QP exceeds it.
- *Fallback*: Negative rear F_z ⇒ load-transfer mis-signed.

**6.4 Worst-case scenario (S8)**

- *Purpose*: The headline paper scenario — class D + emergency brake + cornering.
- *Implementation*: Class D + brake (`a_x = -6 m/s²`) + cornering (`a_y = 5 m/s²`), 4 s window.
- *Acceptance*: Comfort-QP shows at least one wheel with `rho > rho_lim` at some time; risk-MPC keeps `rho ≤ rho_safe` (or maintains a measurable margin), with documented comfort trade-off.
- *Fallback*: Risk-MPC fails to bound `rho` ⇒ verify `F_e^max` realism or relax `rho_safe`.

**6.5 Transient road impact during cornering (S9)**

- *Purpose*: Showcase the STO as the dynamic risk channel.
- *Implementation*: Smooth + cornering (`a_y = 5 m/s²`) + 50 mm × 20 ms bump on outer-front wheel mid-turn.
- *Acceptance*: `d_z_hat` shows a transient negative spike; risk-MPC redistributes load within ≤ 50 ms; transient `rho` peak reduced ≥ 15% vs comfort-QP.
- *Fallback*: STO misses the bump ⇒ verify `m_u, k_t` and STO bandwidth.

**6.6 Component ablations (A1–A3)**

- *Purpose*: Quantify the contribution of each risk-aware mechanism. Each ablation maps to one of the paper's three contributions.
- *Implementation*: On S8 and S9, run — A1: STO disabled (use only `F_z_bar`); A2: `σ_ρ ≡ 0`; A3: `κ_ρ = 0`.
- *Acceptance*: All three ablations measurably degrade vs full risk-MPC.
- *Fallback*: No degradation ⇒ that mechanism isn't contributing; revisit the design.

**6.7 Parameter sensitivity (Σ1–Σ3)**

- *Purpose*: Demonstrate robustness to common uncertainties.
- *Implementation*: On S8, perturb (one at a time): `μ ± 20%`, `h_g ± 10%`, `m ± 10%`. Use nominal controller settings.
- *Acceptance*: Under `μ −20%`, `ΔF_z^err` tightening keeps `rho_actual ≤ rho_safe`. Under `h_g, m` perturbations, performance degrades gracefully, no instability.
- *Fallback*: `μ` underestimate causes ρ violation ⇒ tightening margin too small; recalibrate 2.6 with a μ-mismatch model.

**6.8 Paper figure pack**

- *Purpose*: Produce all Section V figures and tables in publication-ready form.
- *Implementation*: Aggregate metrics across S1–S9 × {passive, LQR, comfort-QP, risk-QP, risk-MPC}. Time-series plots for S7/S8/S9; ablation and sensitivity bar charts.
- *Acceptance*: All figures as `.pdf` + `.png`, tables as `.csv`, one Markdown summary linking figure → scenario × controller.
- *Fallback*: Style drift across figures ⇒ confirm 0.3's plot styling applied uniformly.

---

### Phase 7 — CarSim Co-Simulation

**7.1 CarSim interface setup and parameter matching**

- *Purpose*: Bridge controller and observer to a high-fidelity vehicle simulator.
- *Implementation*: CarSim ↔ Python (or Simulink) interface. Expose `F_e_ij` inputs, vehicle states + tire forces outputs at controller sampling rate. Match CarSim vehicle parameters to the analytical full-car.
- *Acceptance*: Dummy passthrough (`F_e = 0`) completes; parameter spec sheet side-by-side shows agreement.
- *Fallback*: Sampling-rate mismatch ⇒ align step sizes or interpolate.

**7.2 Open-loop validation against analytical model**

- *Purpose*: Confirm CarSim and analytical full-car agree on basics before closing the loop.
- *Implementation*: Both models on (a) straight + smooth 80 km/h, (b) `a_x` step, (c) `a_y` step. Compare heave PSD and ΔF_z magnitudes.
- *Acceptance*: Heave PSD agrees within 10% in 0.1–10 Hz; ΔF_z within 5% of analytical at steady state.
- *Fallback*: Systematic offset ⇒ unsprung mass or tire stiffness; confirm CarSim's tire defaults.

**7.3 STO on CarSim plant**

- *Purpose*: Verify the observer works against a high-fidelity, nonlinear plant.
- *Implementation*: Run Phase 2 STO against CarSim. Use CarSim's reported tire normal force as truth.
- *Acceptance*: STO does not diverge; tracking error bounded; correlation with CarSim's F_z > 0.85 on a class-B segment.
- *Fallback*: Effects not in the analytical model (e.g., wheel-hop modes) ⇒ absorb into `ΔF_z^err`.

**7.4 Risk-aware MPC on CarSim: key scenarios**

- *Purpose*: Reproduce the paper's main results on the high-fidelity plant.
- *Implementation*: Close the loop with risk-MPC. Run S4 (class C cornering) and S8 (worst case).
- *Acceptance*: ρ behavior in CarSim qualitatively matches analytical predictions; risk-MPC keeps ρ bounded where comfort-QP fails.
- *Fallback*: Qualitative disagreement ⇒ verify linear-bicycle assumption is not violated; log slip angles.

**7.5 Robustness on CarSim (Pacejka tire + μ mismatch)**

- *Purpose*: Stress-test against realistic tire and friction uncertainty.
- *Implementation*: Switch CarSim to Pacejka magic-formula tires (controller still assumes linear). Separately, set CarSim `μ = 0.6` while controller assumes `μ = 0.8`.
- *Acceptance*: Controller remains effective; graceful performance degradation vs matched case, quantified.
- *Fallback*: Instability with Pacejka ⇒ linear-tire estimate severely wrong near saturation; flag as future work.

**7.6 CarSim final figure pack**

- *Purpose*: Publication-ready figures from high-fidelity validation.
- *Implementation*: Time-series and metric tables for all CarSim runs. Match Phase 6.8 style.
- *Acceptance*: All `.pdf`/`.png` generated; comparison table CarSim vs analytical.
- *Fallback*: Style drift from 6.8 ⇒ re-use the same plotting utility.

---

## 4. Execution Discipline

1. **Never skip a step's Acceptance check**, even if it looks trivial. The acceptance criteria are the contract that lets later steps assume their dependencies work.
2. **Never advance a step until its Fallback path is unused** (i.e., until the step passes acceptance cleanly).
3. **Every diagnostic figure goes into the per-run `figures/` folder** with a descriptive filename. Don't overwrite.
4. **Commit code after every step that passes acceptance**, with a commit message `phase-X.Y: <step name>`.
5. **If a step fails repeatedly**, isolate by binary-searching back to the most recent step that fully passed.

---

## 5. Mapping to Paper Sections

| Phase | Paper section |
|---|---|
| 0–1 | Section II (vehicle and suspension model) |
| 2 | Section III (super-twisting observer) |
| 3 | Section V baselines |
| 4 | Section IV (risk-aware controller, one-step QP) |
| 5 | Section IV (risk-aware MPC) |
| 6 | Section V main results, ablations, sensitivity |
| 7 | Section V high-fidelity validation |
