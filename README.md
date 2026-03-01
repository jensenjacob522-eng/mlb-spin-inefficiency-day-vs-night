# MLB Spin Inefficiency: Day vs Night (June–July 2024)

## Overview

Do MLB pitchers show differences in spin efficiency between day and night games?

This project analyzes 2024 Statcast data to evaluate whether 4-seam fastball spin efficiency differs between early-day (pre-2pm ET) and night (post-6pm ET) games.

---

## Data

- **2024 MLB Statcast**
- **June–July only**
- **4-Seam Fastballs (FF)**
- **Wind ≤ 5 mph**
- **26 pitchers** (≥75 pitches per time bucket)
- Time buckets defined in **Eastern Time (ET)**

---

## Methods

Spin efficiency was **physically inferred from pitch movement and velocity**, not directly measured.

- Lift force estimated from observed pitch movement  
- Air density adjusted per pitch using game temperature (ideal gas law)  
- Within-pitcher comparison design  
- Pitch-level mixed-effects regression

 ## Results
Pitcher-Mean Comparison
Paired t-test: p = 0.451
No statistically significant difference

Velocity Relationship
Δvelo vs Δspin inefficiency
r = 0.026 (no meaningful association)

Mixed-Effects Model (Pitch-Level)
β(post-6pm) = −0.00679
p = 0.005
95% CI [−0.0115, −0.0020]

 ## Interpretation
- At the pitcher-mean level, there is no statistically significant difference between day and night games. However, pitch-level modeling detects a very small reduction (~0.7 percentage points) in inferred spin inefficiency during post-6pm games after controlling for velocity and pitcher-specific baselines.
Because:
- Spin efficiency is inferred from movement (not directly measured)
- Time buckets are ET-based (not stadium-local)
- Effect size is small

This should be interpreted as a modest statistical association, not a definitive physiological claim.

### Model Specification

```python
spin_ineff ~ is_post6 + velo_z + (1 | pitcher)
