# MLB Spin Inefficiency: Day vs Night (June–July 2024)

## Overview

Do MLB pitchers show differences in inferred spin efficiency in day vs night games?

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

Spin efficiency was **inferred**, not directly measured.

- Magnus-based model (movement + velocity)
- Air density adjusted per pitch using game temperature (ideal gas law)
- Within-pitcher comparison
- Pitch-level mixed-effects regression

### Model Specification

```python
spin_ineff ~ is_post6 + velo_z + (1 | pitcher)


