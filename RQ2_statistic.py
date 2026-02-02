
"""
Two-way Repeated-Measures ANOVA (Model × Encoding) with post-hoc tests
- Subject = seed/run (within-subject for both factors)
- Factors:
    * model: 4 levels (e.g., CodeBERT, CodeT5, UniX-512, UniX-1024)
    * encoding: 5 levels (e.g., after-only, after+markers, before+after, diff w/ tags, added->deleted)
- Input expected shape:  models × encodings × 10 seeds = 4 × 5 × 10 (total 200 numbers)

What this script does:
1) Validates input shapes
2) Builds a long-format DataFrame: columns = [subject, model, encoding, F1]
3) 2-way repeated-measures ANOVA via statsmodels.AnovaRM (within=['model','encoding'])
4) Effect sizes (partial eta^2 and Cohen's f) for each effect (model, encoding, interaction)
5) Post-hoc tests:
   5.1) Encoding main effect (collapsed over models): per-subject marginal means, paired t-tests + Holm correction
   5.2) Simple effects: within each model, all encoding pairwise paired t-tests + Holm correction
6) Saves CSVs for ANOVA table and post-hoc tables; prints concise summaries

Dependencies: numpy, pandas, scipy, statsmodels
Run: python twoway_rm_anova.py
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# ========== 1) INPUT (EDIT HERE) ==========
# Replace the placeholder lists below with your actual numbers.
# Each list under a given encoding must contain exactly 10 values (10 seeds).
# There must be exactly 5 encodings and 4 models.

INPUT = {
    # CodeBERT
    "ModelA": {
        'after-only': [0.3556, 0.3896, 0.4108, 0.39, 0.3634, 0.3781, 0.3641, 0.3465, 0.3685, 0.3645],
        'after+markers': [0.3213, 0.3532, 0.3814, 0.3876, 0.351, 0.3767, 0.3463, 0.3342, 0.3528, 0.3194],
        'before+after': [0.3615, 0.3094, 0.3693, 0.3724, 0.3002, 0.2866, 0.3914, 0.3725, 0.3929, 0.3644],
        'diff w\\ tags': [0.3827, 0.3985, 0.3956, 0.406, 0.3751, 0.3963, 0.3941, 0.396, 0.3996, 0.4093],
        'added -> deleted': [0.3793, 0.4047, 0.4107, 0.392, 0.3806, 0.3919, 0.3934, 0.4042, 0.4027, 0.4111]
    },
    # CodeT5+
    "ModelB": {
        "after-only": [0.3853, 0.3445, 0.3663, 0.3675, 0.3881, 0.3488, 0.3731, 0.3609, 0.3333, 0.362],
        "after+markers": [0.3156, 0.3372, 0.3552, 0.3304, 0.3723, 0.3141, 0.3143, 0.337, 0.3939, 0.311],
        "before+after": [0.3515, 0.2979, 0.3753, 0.2862, 0.2783, 0.2692, 0.3019, 0.2925, 0.3607, 0.3052],
        "diff w\\ tags": [0.3921, 0.39, 0.393, 0.3881, 0.3487, 0.3704, 0.3741, 0.3974, 0.3821, 0.3279],
        "added -> deleted": [0.3921, 0.3773, 0.4033, 0.384, 0.3426, 0.3812, 0.3464, 0.3908, 0.3857, 0.368]
    },
    # UniXcoder-512
    "ModelC": {
        "after-only": [0.3346, 0.367, 0.3215, 0.2907, 0.3918, 0.3757, 0.3792, 0.3333, 0.3595, 0.3418],
        "after+markers": [0.3052, 0.3095, 0.2718, 0.3774, 0.3065, 0.284, 0.3506, 0.3491, 0.3159, 0.3396],
        "before+after": [0.3948, 0.2496, 0.3148, 0.2513, 0.3785, 0.3406, 0.2853, 0.3825, 0.2968, 0.3021],
        "diff w\\ tags": [0.4006, 0.3648, 0.3291, 0.4119, 0.3846, 0.355, 0.4041, 0.3777, 0.3903, 0.3392],
        "added -> deleted": [0.4006, 0.3936, 0.2872, 0.4016, 0.4215, 0.347, 0.4038, 0.3646, 0.3894, 0.3063]
    },
    # UniXcoder-1024
    "ModelD": {
        "after-only": [0.3235, 0.3377, 0.3261, 0.3231, 0.2771, 0.3355, 0.3377, 0.3103, 0.1952, 0.2517],
        "after+markers": [0.2072, 0.3095, 0.2828, 0.2782, 0.3759, 0.2192, 0.3055, 0.3166, 0.3316, 0.3124],
        "before+after": [0.3996, 0.2375, 0.293, 0.2579, 0.3737, 0.3884, 0.2574, 0.3127, 0.2468, 0.3562],
        "diff w\\ tags": [0.398, 0.2505, 0.3524, 0.3135, 0.4054, 0.4062, 0.2572, 0.3401, 0.3319, 0.3438],
        "added -> deleted": [0.4007, 0.2544, 0.3329, 0.3237, 0.4147, 0.4103, 0.2399, 0.3542, 0.2455, 0.376]
    },
    # Qwen2.5-512
    "ModelE": {
        "after-only":        [0.3561, 0.3324, 0.3449, 0.3289, 0.3708, 0.3344, 0.3319, 0.3433, 0.3394, 0.3066],
        "after+markers":     [0.3575, 0.3369, 0.3257, 0.3415, 0.3469, 0.3319, 0.3598, 0.3333, 0.3450, 0.3457],
        "before+after":      [0.3653, 0.3387, 0.3455, 0.3286, 0.3452, 0.3491, 0.3245, 0.3279, 0.3308, 0.3267],
        "diff w\\ tags":      [0.3724, 0.3560, 0.3598, 0.3504, 0.3549, 0.3633, 0.3510, 0.3372, 0.3484, 0.3431],
        "added -> deleted":  [0.3576, 0.3374, 0.3541, 0.3754, 0.3363, 0.3625, 0.3442, 0.3302, 0.3431, 0.3567]
    },
    # Qwen2.5-1024
    "ModelF": {
        "after-only":        [0.3382, 0.3035, 0.3177, 0.3122, 0.3366, 0.3639, 0.3483, 0.3233, 0.3648, 0.3218],
        "after+markers":     [0.3312, 0.3333, 0.3228, 0.3666, 0.3366, 0.3424, 0.3319, 0.3426, 0.3609, 0.3383],
        "before+after":      [0.3186, 0.3622, 0.3319, 0.3294, 0.3438, 0.3517, 0.3466, 0.3569, 0.3346, 0.3667],
        "diff w\\ tags":     [0.3686, 0.3658, 0.3495, 0.3419, 0.3519, 0.3705, 0.3354, 0.3449, 0.3303, 0.3465],
        "added -> deleted":  [0.3297, 0.3598, 0.3391, 0.3128, 0.3418, 0.3214, 0.3652, 0.3393, 0.3254, 0.3613]
    }
}

for model, encodings in INPUT.items():
    print(f"--- {model} Results ---")
    for enc, values in encodings.items():
        mean = np.mean(values)
        # ddof=1을 추가하여 N-1로 계산
        std = np.std(values, ddof=1)
        print(f"{enc:18}: {mean:.4f} ± {std:.4f}")
    print()

# ========== 2) Validation & reshape to long-format ==========

def validate_input(inp):
    models = list(inp.keys())
    enc_ref = None
    for m in models:
        encs = list(inp[m].keys())
        if sorted(encs) != sorted(["after-only","after+markers","before+after","diff w\\ tags","added -> deleted"]):
            raise ValueError(f"Model {m}: encodings must be exactly the 5 predefined names. Got: {encs}")
        for e in encs:
            vals = inp[m][e]
            if len(vals) != 10:
                raise ValueError(f"Model {m}, encoding {e}: expected 10 values (seeds); got {len(vals)}")
    return models

models = validate_input(INPUT)
encodings = ["after-only","after+markers","before+after","diff w\\ tags","added -> deleted"]
subjects = np.arange(1, 11)  # 10 seeds -> subjects 1..10

rows = []
for s_idx, s in enumerate(subjects):
    for m in models:
        for e in encodings:
            rows.append({
                "subject": s,
                "model": m,
                "encoding": e,
                "F1": float(INPUT[m][e][s_idx])
            })
df = pd.DataFrame(rows)

# Quick summaries
means_table = df.groupby(["model","encoding"])["F1"].agg(["mean","std"])

# ========== 3) Two-way repeated-measures ANOVA ==========
aov = AnovaRM(df, depvar="F1", subject="subject", within=["model","encoding"]).fit()
aov_table = aov.anova_table.reset_index().rename(columns={"index":"Effect"})

# Compute partial eta^2 and Cohen's f for each effect
def partial_eta2(F, df_num, df_den):
    return (F * df_num) / (F * df_num + df_den)

def cohen_f_from_eta2p(eta2p):
    return np.sqrt(eta2p/(1-eta2p)) if eta2p < 1.0 else np.inf

eff_rows = []
for i, r in aov_table.iterrows():
    Fv = float(r["F Value"]); dfn = float(r["Num DF"]); dfd = float(r["Den DF"])
    eta2p = partial_eta2(Fv, dfn, dfd)
    cf = cohen_f_from_eta2p(eta2p)
    eff_rows.append({"Effect": r["Effect"], "F": Fv, "df_num": dfn, "df_den": dfd,
                     "p_value": float(r["Pr > F"]), "partial_eta2": eta2p, "cohen_f": cf})
effects_df = pd.DataFrame(eff_rows)

# ========== 4) Post-hoc tests ==========
def holm_correction(pvals, labels, alpha=0.05):
    """Return DataFrame with raw p, Holm-adjusted p, and rejection flags."""
    dfp = pd.DataFrame({"label": labels, "p_raw": pvals})
    dfp = dfp.sort_values("p_raw").reset_index(drop=True)
    m = len(dfp)
    adj = []
    max_so_far = 0.0
    for i in range(m):
        a = min(1.0, (m - i) * dfp.loc[i, "p_raw"])
        max_so_far = max(max_so_far, a)  # enforce monotonicity
        adj.append(max_so_far)
    dfp["p_holm"] = adj
    dfp["reject@0.05"] = dfp["p_holm"] <= alpha
    return dfp

# 4.1 Encoding main effect post-hoc (collapsed over models)
# For each subject, compute marginal mean across models for each encoding
collapsed = df.groupby(["subject","encoding"])["F1"].mean().reset_index()
enc_pairs = list(itertools.combinations(encodings, 2))
pvals, labels, tstats = [], [], []
for a, b in enc_pairs:
    a_vals = collapsed.loc[collapsed["encoding"]==a, "F1"].values
    b_vals = collapsed.loc[collapsed["encoding"]==b, "F1"].values
    t, p = stats.ttest_rel(a_vals, b_vals)
    pvals.append(p); labels.append(f"{a} vs {b}"); tstats.append(t)
enc_posthoc = holm_correction(pvals, labels)
enc_posthoc["t"] = tstats

# 4.2 Simple effects: within each model, all encoding pairwise tests
simple_effects = []
for m in models:
    dsub = df[df["model"]==m]
    pvals_m, labels_m, tstats_m = [], [], []
    for a, b in enc_pairs:
        aa = dsub[dsub["encoding"]==a].sort_values("subject")["F1"].values
        bb = dsub[dsub["encoding"]==b].sort_values("subject")["F1"].values
        t, p = stats.ttest_rel(aa, bb)
        pvals_m.append(p); labels_m.append(f"{m}: {a} vs {b}"); tstats_m.append(t)
    dfm = holm_correction(pvals_m, labels_m)
    dfm["t"] = tstats_m
    simple_effects.append(dfm)
simple_effects_df = pd.concat(simple_effects, ignore_index=True)

# ========== 5) Output ==========
pd.set_option("display.width", 140)
pd.set_option("display.precision", 6)

print("=== Means and STDs by Model × Encoding ===")
print(means_table, "\n")

print("=== Two-way Repeated-Measures ANOVA (within: model, encoding) ===")
print(aov.summary(), "\n")

print("=== Effect Sizes (partial eta^2 and Cohen's f) ===")
print(effects_df, "\n")

print("=== Post-hoc: Encoding Main Effect (collapsed over models), paired t-tests + Holm ===")
print(enc_posthoc[["label","t","p_raw","p_holm","reject@0.05"]], "\n")

print("=== Post-hoc: Simple Effects within each Model, paired t-tests + Holm ===")
print(simple_effects_df[["label","t","p_raw","p_holm","reject@0.05"]])
