# Forward Pass Physics Model

_Phase 6 · Task 6.1 · Decision Criteria Definition_

This document defines the physics-based decision rules that the Phase 6 decision engine uses to determine whether a rugby pass is forward. It covers the current Phase 1 displacement model, a simplified momentum adjustment, borderline handling, and the hooks required for more advanced models (e.g., metric tensor-based reasoning). The implementation in `ml/decision_engine.py` must treat this specification as the single source of truth.

---

## 1. Field Frame, Inputs, and Notation

- **Field coordinate system**: Right-handed XYZ frame anchored to the center of the halfway line.
  - **X-axis (`forward`)**: Positive toward the opponents' goal line.
  - **Y-axis (`lateral`)**: Positive toward the right touch line when facing forward.
  - **Z-axis (`vertical`)**: Positive upward.
- **Forward axis vector**: `f̂ = normalize(field_axis_forward)` and must match the global frame used in `ml/spatial_model.py`.
- **Ball trajectory**: `B(t_i) ∈ ℝ³`, reconstructed 3D ball positions sampled at timestamps `t_i` (uniform cadence, typically video frame rate).
- **Passer trajectory**: `P(t_i) ∈ ℝ³` for the identified passer, aligned to the same timestamps when available.
- **Pass window**: `[t_start, t_end]` indices returned by pass detection (Task 6.2).

We assume at least two synchronized cameras, calibrated intrinsics/extrinsics, and already-synchronized timestamps (Phase 3-5 deliverables).

---

## 2. Rugby Law Summary and Modeling Assumptions

- Law 12: A pass is forward if the ball travels toward the opponents' dead-ball line relative to the ball carrier's momentum at the moment of release.
- **Assumptions for Phase 6**:
  1. Air resistance and spin are negligible over the short time window of a pass.
  2. Player momentum is approximated by instantaneous velocity at ball release.
  3. Passer identification is reliable for the release frame (Phase 4 tracking output).
  4. Small reconstruction noise (<0.15 m) is expected and handled through thresholds.

---

## 3. Phase 1 Displacement Model (Baseline Required Now)

1. **Extract pass window**:
   - Use `(start_idx, end_idx)` from `detect_pass_events`.
   - Define `B_start = B(t_start)` and `B_end = B(t_end)`.
2. **Compute net forward displacement**:
   ```math
   Δx_ball = (B_end - B_start) · f̂
   ```
3. **Apply tolerance band**:
   - `δ_noise = 0.20 m` (≈ ball length) accounts for slight reconstruction drift.
4. **Classification** (no passer data):
   - `Δx_ball > δ_noise` → **Forward pass**.
   - `Δx_ball < -δ_noise` → **Not forward** (ball clearly traveled backward).
   - `|Δx_ball| ≤ δ_noise` → **Borderline / indeterminate** (flag for low confidence).

This simple displacement test is the fallback whenever passer trajectory data is missing or flagged unreliable.

---

## 4. Passer Momentum Adjustment (Preferred When Data Exists)

When the passer trajectory is available for frames around release, incorporate their momentum:

1. **Estimate passer velocity at release**:
   ```math
   V_passer = \frac{P(t_start + Δt) - P(t_start - Δt)}{2Δt}
   ```
   - Use `Δt = 2 / fps` (two-frame window) to smooth noise.
2. **Forward component of momentum**:
   ```math
   v_forward = V_passer · f̂
   ```
3. **Effective release plane advance**:
   ```math
   Δx_passer = v_forward × (t_end - t_start)
   ```
   - Represents how far the passer's momentum would carry them along `f̂` during the ball's flight.
4. **Relative displacement**:
   ```math
   Δx_rel = Δx_ball - Δx_passer
   ```
5. **Decision with momentum**:
   - `Δx_rel > δ_forward` → **Forward pass**.
   - `Δx_rel < -δ_backward` → **Not forward**.
   - Otherwise → **Borderline**.
   - Recommended values: `δ_forward = 0.15 m`, `δ_backward = 0.05 m`.

The asymmetry reflects rugby interpretations: only a clear forward travel after removing the passer’s momentum should trigger a forward call, whereas even small backward travel is acceptable.

---

## 5. Confidence Heuristics and Borderline Handling

- **Confidence tie-in (Task 6.5)**:
  - Base confidence derived from detection/tracking quality.
  - Reduce confidence linearly within the borderline band (`|Δx_rel| ≤ 0.20 m`).
  - Cap confidence at `0.45` if passer data is missing (since only the simple model is used).
- **Edge cases**:
  - **Missing passer data**: Use `Δx_ball` only and flag `metadata["momentum_used"] = False`.
  - **High reconstruction variance**: Inflate `δ_noise` by the 95th percentile positional error.
  - **Multiple touches**: Re-slice pass windows to isolate the dominant transfer; apply criteria per segment.

---

## 6. Algorithm Summary (for `ml/decision_engine.py`)

```
1. Validate inputs (ball trajectory, timestamps, field axis).
2. Smooth ball (and passer) trajectories over ≤5 frames to reduce noise.
3. Detect pass window indices (Task 6.2 output).
4. Compute Δx_ball along f̂.
5. If passer data reliable:
     a. Estimate passer forward velocity at release.
     b. Compute Δx_rel = Δx_ball - Δx_passer.
     c. Classify using δ_forward / δ_backward thresholds.
   Else:
     a. Use Δx_ball with δ_noise threshold.
6. Populate DecisionResult:
     - `is_forward`
     - `confidence` (per Section 5 heuristic)
     - `explanation` summarizing Δx values, thresholds, and data availability.
     - `metadata` with raw metrics (Δx_ball, Δx_passer, Δx_rel, thresholds).
```

---

## 7. Worked Examples

| Scenario | Inputs | Computation | Decision |
| --- | --- | --- | --- |
| **Clearly forward** | `Δx_ball = +5.0 m`, `Δx_passer = +1.0 m` | `Δx_rel = +4.0 m` > `δ_forward` | **Forward**, high confidence |
| **Clearly backward** | `Δx_ball = -2.0 m`, no passer data | `Δx_ball < -δ_noise` | **Not forward** |
| **Borderline forward** | `Δx_ball = +1.5 m`, `Δx_passer = +1.0 m` | `Δx_rel = +0.5 m` just above `δ_forward` | Forward, low confidence |
| **Borderline backward** | `Δx_ball = +0.8 m`, `Δx_passer = +1.0 m` | `Δx_rel = -0.2 m` | Not forward, low confidence |
| **Passer missing** | `Δx_ball = +0.6 m`, no passer | `Δx_ball < δ_noise` | Borderline (indeterminate) |

All implementations must unit test at least these five scenarios.

---

## 8. Extension Hooks (Phase 6+ Roadmap)

1. **Metric tensor model**:
   - Treat the field as a manifold with local transformations to account for moving reference frames (e.g., accelerating passers).
   - Store a `G` tensor describing scaling between coordinate bases; compute displacement as `Δx_rel = (B_end - B_start)^T G f̂`.
2. **Aerodynamic modeling**:
   - Incorporate drag coefficients for long-distance kicks acting like passes.
3. **Player rotation handling**:
   - Account for the passer turning during release by rotating `f̂` using the passer’s yaw rate.
4. **Probabilistic thresholds**:
   - Replace deterministic `δ_*` with distributions learned from historical labeled data.

Document any upgrades here before implementing them to keep the physics model auditable.

---

## 9. Implementation Checklist

- [ ] Use guard clauses when validating inputs in `ml/decision_engine.py`.
- [ ] Keep functions under two nesting levels.
- [ ] Emit `metadata` for downstream explainability.
- [ ] Update unit tests whenever thresholds change.
- [ ] Reference this document in relevant docstrings and module headers.

Once this document is approved, Phase 6 Tasks 6.2–6.5 should rely on it for shared constants and reasoning.
