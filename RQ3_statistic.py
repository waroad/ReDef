# rq3_f1_analysis.py
# Usage: python rq3_f1_analysis.py
import numpy as np
import pandas as pd
from math import sqrt

# Optional SciPy (recommended). Falls back to normal approx if unavailable.
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ==== Your data (as provided) ====
conditions_data = {
    "CodeBERT":{
        'Spurious change markers': {
            'original': [0.3213, 0.3532, 0.3814, 0.3876, 0.351, 0.3767, 0.3463, 0.3342, 0.3528, 0.3194],
            'perturbed': [0.3887, 0.3415, 0.3819, 0.3218, 0.2905, 0.3549, 0.3658, 0.3636, 0.3564, 0.3667]
        },
        'Swapped snapshots': {
            'original': [0.3615, 0.3094, 0.3693, 0.3724, 0.3002, 0.2866, 0.3914, 0.3725, 0.3929, 0.3644],
            'perturbed': [0.361, 0.3142, 0.3741, 0.3707, 0.3053, 0.2857, 0.3927, 0.37, 0.3958, 0.3687]
        },
        'Reversed diff tags': {
            'original': [0.3827, 0.3985, 0.3956, 0.406, 0.3751, 0.3963, 0.3941, 0.396, 0.3996, 0.4093],
            'perturbed': [0.3773, 0.3941, 0.3943, 0.4061, 0.3723, 0.3886, 0.3913, 0.389, 0.3981, 0.4078]
        },
        'Swapped added/deleted blocks': {
            'original': [0.3793, 0.4047, 0.4107, 0.392, 0.3806, 0.3919, 0.3934, 0.4042, 0.4027, 0.4111],
            'perturbed': [0.3893, 0.3966, 0.4098, 0.4062, 0.4041, 0.3804, 0.3861, 0.3957, 0.3872, 0.3976]
        }
    },
    "Qwen2.5-512":{
        'Spurious change markers': {
            'original': [0.3575, 0.3369, 0.3257, 0.3415, 0.3469, 0.3319, 0.3598, 0.3333, 0.3450, 0.3457],
            'perturbed': [0.3599, 0.3450, 0.3142, 0.3560, 0.2999, 0.3452, 0.3462, 0.3369, 0.3241, 0.3011]
        },
        'Swapped snapshots': {
            'original': [0.3653, 0.3387, 0.3455, 0.3286, 0.3452, 0.3491, 0.3245, 0.3279, 0.3308, 0.3267],
            'perturbed': [0.3391, 0.3170, 0.3227, 0.3265, 0.3483, 0.3508, 0.3511, 0.3438, 0.3542, 0.3388]
        },
        'Reversed diff tags': {
            'original': [0.3724, 0.3560, 0.3598, 0.3504, 0.3549, 0.3633, 0.3510, 0.3372, 0.3484, 0.3431],
            'perturbed': [0.3486, 0.3444, 0.3259, 0.3478, 0.3633, 0.3377, 0.3383, 0.3229, 0.3513, 0.3300]
        },
        'Swapped added/deleted blocks': {
            'original': [0.3576, 0.3374, 0.3541, 0.3754, 0.3363, 0.3625, 0.3442, 0.3302, 0.3431, 0.3567],
            'perturbed': [0.3591, 0.3524, 0.3279, 0.3590, 0.3453, 0.3413, 0.3666, 0.3395, 0.3571, 0.3558]
        }
    }
}

# ==== stats helpers ====
def paired_t_one_sided(orig, pert):
    """H1: mean(orig - pert) > 0 (degradation under perturbation)"""
    d = np.array(orig) - np.array(pert)
    n = d.size
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        t = np.inf if mean > 0 else -np.inf if mean < 0 else 0.0
        p = 0.0 if mean > 0 else 1.0
        return t, p, d
    t = mean / (sd / sqrt(n))
    if SCIPY_AVAILABLE:
        p = 1 - stats.t.cdf(t, df=n-1)
    else:
        # normal approx
        from math import erf
        p = 0.5 * (1 - erf(t / sqrt(2)))
    return t, p, d

def paired_t_two_sided(orig, pert):
    d = np.array(orig) - np.array(pert)
    n = d.size
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0, 1.0
    t = mean / (sd / sqrt(n))
    if SCIPY_AVAILABLE:
        p = 2 * (1 - stats.t.cdf(abs(t), df=n-1))
    else:
        from math import erf
        # normal approx
        p = 2 * (1 - 0.5*(1+erf(abs(t)/sqrt(2))))
    return t, p

def wilcoxon_one_sided(orig, pert):
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan
    try:
        res = stats.wilcoxon(orig, pert, zero_method='wilcox', alternative='greater', correction=False, mode='auto')
        return res.statistic, res.pvalue
    except Exception:
        return np.nan, np.nan

def cohens_dz(diffs):
    sd = diffs.std(ddof=1)
    return diffs.mean() / sd if sd != 0 else np.inf

def bootstrap_ci_mean(diffs, n_boot=10000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    low = float(np.percentile(means, 100*alpha/2))
    high = float(np.percentile(means, 100*(1 - alpha/2)))
    return low, high

def holm_correction(pvals, labels):
    """Step-down Holm over the family (one-sided t-tests here)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    max_so_far = 0.0
    for i, idx in enumerate(order):
        adj_p = min((m - i) * pvals[idx], 1.0)
        max_so_far = adj_p if i == 0 else max(max_so_far, adj_p)
        adj[idx] = max_so_far
    return {labels[i]: adj[i] for i in range(m)}

# ==== run analyses ====
for model in conditions_data:
    rows = []
    pvals = []
    labels = []
    print("\n","-"*20,model,"-"*20,"\n")
    for cond, vals in conditions_data[model].items():
        x = np.array(vals['original'], dtype=float)
        y = np.array(vals['perturbed'], dtype=float)

        t1, p1, d = paired_t_one_sided(x, y)  # H1: orig > pert
        w_stat, w_p = wilcoxon_one_sided(x, y)
        dz = cohens_dz(d)
        ci_low, ci_high = bootstrap_ci_mean(d, n_boot=10000, alpha=0.05, random_state=123)

        rows.append({
            'Condition': cond,
            'Mean F1 (orig)': float(x.mean()),
            'Mean F1 (pert)': float(y.mean()),
            'Mean Δ (orig-pert)': float(d.mean()),
            "Cohen's d_z": float(dz),
            't (one-sided)': float(t1),
            'p_t (one-sided)': float(p1),
            'Wilcoxon W': float('nan') if np.isnan(w_stat) else float(w_stat),
            'p_wilcoxon (one-sided)': float('nan') if np.isnan(w_p) else float(w_p),
            'Bootstrap 95% CI for Δ (low)': ci_low,
            'Bootstrap 95% CI for Δ (high)': ci_high,
        })
        pvals.append(p1)
        labels.append(cond)

    df = pd.DataFrame(rows)

    # Holm over the five one-sided t-tests
    holm_adj = holm_correction(pvals, labels)
    df['Holm-adjusted p (t one-sided)'] = df['Condition'].map(holm_adj)

    # Pretty print
    with pd.option_context('display.max_columns', None, 'display.width', 120):
        print(df.round(4))
