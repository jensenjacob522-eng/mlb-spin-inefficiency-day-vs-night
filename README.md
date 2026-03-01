MLB Spin Inefficiency: Day vs Night (June–July 2024)
Overview

Do MLB pitchers show differences in inferred spin efficiency in day vs night games?

This project analyzes 2024 Statcast data to evaluate whether 4-seam fastball spin efficiency differs between early-day (pre-2pm ET) and night (post-6pm ET) games.

Data

2024 MLB Statcast

June–July only

4-Seam Fastballs (FF)

Wind ≤ 5 mph

26 pitchers (≥75 pitches per time bucket)

Time buckets defined in Eastern Time (ET)

Methods

Spin efficiency was inferred, not directly measured.

Magnus-based Option A model (movement + velocity)

Air density adjusted per pitch using game temperature (ideal gas law)

Within-pitcher comparison

Pitch-level mixed-effects regression

Model specification:

spin_ineff ~ is_post6 + velo_z + (1 | pitcher)

Where:

is_post6 = night game indicator

velo_z = standardized velocity

Random intercept per pitcher

Results

Pitcher-mean comparison

Paired t-test: p = 0.451

No significant difference

Velocity relationship

Δvelo vs Δspin inefficiency

r = 0.026 (no association)

Mixed-effects model (pitch-level)

β(post-6pm) = −0.00679

p = 0.005

95% CI [−0.0115, −0.0020]

Interpretation

At the pitcher-mean level, there is no statistically significant difference between day and night games.

However, pitch-level modeling detects a very small reduction (~0.7 percentage points) in inferred spin inefficiency during post-6pm games after controlling for velocity and pitcher-specific baselines.

Because:

Spin efficiency is inferred (not directly measured)

Time buckets are ET-based (not stadium-local)

Effect size is small

This should be interpreted as a modest statistical association rather than a definitive physiological conclusion.

Reproducibility

Core script:

src/spin_study.py

Dataset built from Statcast using pybaseball.

Author

Jacob Jensen
Performance Science
