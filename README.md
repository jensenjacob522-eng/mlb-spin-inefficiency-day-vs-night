MLB Spin Inefficiency: Day vs Night (June–July 2024)
Question

Do MLB pitchers show differences in inferred spin efficiency in day vs night games?

Dataset

2024 Statcast

June–July only

4-Seam Fastballs (FF)

Wind ≤ 5 mph

26 pitchers (≥75 pitches per time bucket)

Time buckets: Pre-2pm vs Post-6pm (ET)

Methods

Spin efficiency inferred using a Magnus-based Option A model (movement + velocity).

Air density adjusted using game temperature (ideal gas law).

Within-pitcher comparison.

Pitch-level mixed-effects regression controlling for standardized velocity.

Model:

spin_ineff ~ is_post6 + velo_z + (1 | pitcher)
Results

Paired t-test (pitcher means): p = 0.451

Correlation (Δvelo vs Δspin ineff): r = 0.026

Mixed-effects (pitch-level):
β(post-6pm) = -0.00679
p = 0.005
95% CI [-0.0115, -0.0020]

Interpretation

No significant difference at the pitcher-mean level.
Pitch-level modeling suggests a very small reduction (~0.7 percentage points) in inferred spin inefficiency in night games after controlling for velocity.

Effect size is small. Spin efficiency is inferred (not directly measured), so findings represent an association rather than a causal physiological claim.
