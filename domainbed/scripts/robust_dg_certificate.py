"""
robust_dg_certificate.py
========================
Experiments validating the Route A robust DG certificate:

    |R̄(h, C_test)  −  R̄(h, C_source)|
        ≤  B · (1−ε) · MMD_k(μ_source, μ_test)          [Theorem: Credal DG Certificate]

Route A: credal ellipsoid C(x) ⊆ B_ε(μ(x))  (UAI Proposition)
         → TV outer approximation is 2-monotone
         → Theorem 1 (kernel IIPM = Hausdorff) applies to the ball
         → certificate inherits validity without assuming 2-monotonicity of ellipsoid

THREE EXPERIMENTS:
  Exp 1 — Prediction:   ρ((1−ε)·MMD, accuracy_gap) > ρ(MMD, accuracy_gap)
           = IIPM discounting by ε makes shift prediction strictly better

  Exp 2 — Non-vacuousness:  B·(1−ε)·MMD  <  actual accuracy gap  for each domain
           = the certificate is informative, not just vacuously true

  Exp 3 — Adaptive beats fixed:
           gap(adaptive) = (ε_max − E[ε(x)])·(M − E[ℓ])  > 0
           = per-instance ε tighter than global ε_max

USAGE:
  # Run all three experiments (requires features from credal_dg_pacs.py):
  python robust_dg_certificate.py \\
      --pacs_root /path/to/PACS \\
      --output_dir ./robust_dg_results

  # If you already ran credal_dg_pacs.py and saved features:
  python robust_dg_certificate.py \\
      --feats_npz ./credal_dg_results/domain_feats.npz \\
      --output_dir ./robust_dg_results

DEPENDENCIES:
  pip install torch torchvision scipy matplotlib tqdm Pillow scikit-learn numpy
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────

DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
DOMAIN_SHORT = {
    'art_painting': 'Art',
    'cartoon':      'Cartoon',
    'photo':        'Photo',
    'sketch':       'Sketch',
}
DOMAIN_COLORS = {
    'art_painting': '#4C72B0',
    'cartoon':      '#DD8452',
    'photo':        '#55A868',
    'sketch':       '#C44E52',
}

# Published ERM accuracy — the y-axis ground truth
# Source: Gulrajani & Lopez-Paz (2021), training-domain validation
ERM_ACC = {
    'art_painting': 77.2,
    'cartoon':      75.8,
    'photo':        96.0,
    'sketch':       69.2,
}

# Published GroupDRO accuracy (same backbone, same split)
# Source: Sagawa et al. (2020) / DomainBed README
GROUPDRO_ACC = {
    'art_painting': 76.7,
    'cartoon':      76.4,
    'photo':        95.8,
    'sketch':       70.3,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def convert_numpy_types(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# =============================================================================
# PART 0 — Feature extraction (reuses credal_dg_pacs logic)
# =============================================================================

def load_or_extract_features(
    pacs_root: Optional[str],
    feats_npz: Optional[str],
    arch:        str   = 'resnet18',
    n_heads:     int   = 5,
    dropout_p:   float = 0.15,
    max_samples: int   = 400,
    device:      str   = 'cpu',
) -> Dict[str, Dict]:
    """
    Either load pre-extracted features from .npz or extract fresh from PACS.
    If feats_npz exists, always prefer it (faster).
    """
    if feats_npz and Path(feats_npz).exists():

        data = np.load(feats_npz, allow_pickle=True)
        return data['domain_feats'].item()

    if pacs_root is None:
        raise ValueError(
            "Provide either --pacs_root or --feats_npz.\n"
            "To extract fresh features, run credal_dg_pacs.py first:\n"
            "  python credal_dg_pacs.py --pacs_root /path/to/PACS"
        )

    # Import from credal_dg (must be in same directory or PYTHONPATH)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from credal_dg import (
            parse_label_file, extract_credal_features, load_backbone
        )
    except ImportError:
        raise ImportError(
            "credal_dg.py must be in the same directory.\n"
            "Make sure credal_dg.py is in the scripts directory."
        )

    pacs_root  = Path(pacs_root)
    img_root   = pacs_root / 'pacs_data'
    label_root = pacs_root / 'pacs_label'

    backbone, feat_dim = load_backbone(arch, device)
    domain_feats = {}

    for domain in DOMAINS:
        label_file = label_root / f'{domain}_test_kfold.txt'

        samples = parse_label_file(str(label_file), str(img_root))
        domain_feats[domain] = extract_credal_features(
            backbone, samples, device,
            n_heads=n_heads, dropout_p=dropout_p, max_samples=max_samples
        )
        n = domain_feats[domain]['n']


    return domain_feats


# =============================================================================
# PART 1 — Core measures
# =============================================================================

def mmd_linear(mu_a: np.ndarray, mu_b: np.ndarray) -> float:
    """MMD with linear kernel = ||mean_a - mean_b||_2"""
    return float(np.linalg.norm(mu_a.mean(0) - mu_b.mean(0)))


def eps_domain(feats: Dict) -> Tuple[float, float, np.ndarray]:
    """
    ε statistics for one domain.

    Returns:
      eps_mean  : E[ε(x)]  = mean credal width
      eps_max   : max ε(x) = fixed-ε equivalent
      eps_vec   : (N,) per-instance ε(x) = √Tr(Σ_epi(x))
    """
    eps_vec  = feats['epsilon']             # (N,) already computed
    return float(eps_vec.mean()), float(eps_vec.max()), eps_vec


def route_a_iipm(
    eps:    float,        # ε = max(ε_source, ε_test)
    mmd:    float,        # MMD_k(μ_source, μ_test)
) -> float:
    """
    Route A: kernel IIPM via TV outer approximation.
    IIPM(C_source, C_test) ≤ (1−ε) · MMD_k(μ_source, μ_test)

    This is the computable upper bound from Theorem (Credal DG Certificate).
    Uses the fact that C(x) ⊆ B_ε(μ(x)) and B_ε is 2-monotone,
    so Theorem 1 applies to the Huber balls exactly.
    """
    return (1.0 - eps) * mmd


def robust_risk_bound(
    r_bar_source: float,  # R̄(h, C_source) = ERM acc gap proxy
    B:            float,  # ‖W‖_F or RKHS norm of loss
    eps:          float,  # max(ε_source, ε_test)
    mmd:          float,  # MMD_k(μ_source, μ_test)
) -> float:
    """
    R̄(h, C_test) ≤ R̄(h, C_source) + B · (1−ε) · MMD_k(μ_source, μ_test)

    Route A certificate: ellipsoid → TV ball → Theorem 1 → closed form.
    """
    return r_bar_source + B * route_a_iipm(eps, mmd)


def adaptive_vs_fixed_gap(
    eps_vec: np.ndarray,     # per-instance ε(x), shape (N,)
    M:       float = 1.0,    # sup loss (cross-entropy ∈ [0,∞), use 1.0 as normalised)
    ell_mean: float = 0.3,   # E[ℓ(h,x)] ≈ 1 - accuracy
) -> Dict:
    """
    Theorem (Adaptive beats Fixed):
      gap = (ε_max − E[ε(x)]) · (M − E[ℓ])

    Returns both the theoretical gap and the empirical distribution info.
    """
    eps_mean = float(eps_vec.mean())
    eps_max  = float(eps_vec.max())
    eps_std  = float(eps_vec.std())

    gap = (eps_max - eps_mean) * (M - ell_mean)

    return {
        'eps_mean': eps_mean,
        'eps_max':  eps_max,
        'eps_std':  eps_std,
        'gap':      gap,
        'gap_pct':  gap / (eps_max * (M - ell_mean) + 1e-8) * 100,  # % tightening
        'heterogeneous': eps_std > 0.01,  # flag if ε is meaningfully non-constant
    }


# =============================================================================
# PART 2 — Three experiments
# =============================================================================

def experiment_1_prediction(
    domain_feats: Dict[str, Dict],
    held_out_acc: Dict[str, float],
) -> Dict:
    """
    Exp 1 — Does (1−ε)·MMD predict accuracy gaps better than raw MMD?

    For each held-out domain:
      - mmd   = MMD_k(μ_train, μ_test)
      - iipm  = (1−ε)·MMD   [Route A]
      - accuracy_gap = max_acc − held_out_acc

    Prediction: ρ(iipm, gap) > ρ(mmd, gap)
    because the credal (1−ε) discount captures whether the model
    is already uncertain in the target regime.
    """


    max_acc   = max(held_out_acc.values())
    records   = []

    for held_out in DOMAINS:
        train_doms  = [d for d in DOMAINS if d != held_out]
        train_feats = [domain_feats[d] for d in train_doms]
        test_feats  = domain_feats[held_out]

        # Pooled training mean
        mu_train = np.concatenate([f['mu'] for f in train_feats]).mean(0)
        mu_test  = test_feats['mu'].mean(0)

        mmd  = float(np.linalg.norm(mu_train - mu_test))

        # ε = max(ε_source, ε_test)
        eps_train = np.concatenate([f['epsilon'] for f in train_feats]).mean()
        eps_test  = test_feats['epsilon'].mean()
        eps       = float(max(eps_train, eps_test))

        iipm      = route_a_iipm(eps, mmd)
        acc       = held_out_acc[held_out]
        gap       = max_acc - acc   # accuracy gap: how much worse than best domain



        records.append({
            'domain': held_out,
            'mmd':    mmd,
            'iipm':   iipm,
            'eps':    eps,
            'acc':    acc,
            'gap':    gap,
        })

    gaps  = [r['gap']  for r in records]
    mmds  = [r['mmd']  for r in records]
    iipms = [r['iipm'] for r in records]

    rho_mmd,  p_mmd  = spearmanr(mmds,  gaps)
    rho_iipm, p_iipm = spearmanr(iipms, gaps)

    confirmed = abs(rho_iipm) > abs(rho_mmd)



    return {
        'records':    records,
        'rho_mmd':    float(rho_mmd),  'p_mmd':  float(p_mmd),
        'rho_iipm':   float(rho_iipm), 'p_iipm': float(p_iipm),
        'confirmed':  confirmed,
    }


def experiment_2_nonvacuous(
    domain_feats: Dict[str, Dict],
    held_out_acc: Dict[str, float],
    B:            float = 1.0,    # ‖W‖_F / sqrt(K) normalised to 1.0
    M:            float = 1.0,    # sup normalised cross-entropy
) -> Dict:
    """
    Exp 2 — Is the Route A bound non-vacuous?

    For each held-out domain, compute:
      certificate = B · (1−ε) · MMD_k(μ_source, μ_test)
      actual_gap  = |acc_source_mean − acc_test|

    Non-vacuous: certificate < actual_gap  (bound is tighter than trivial)
    Non-trivial: certificate > 0           (bound says something)

    Also computes the Choquet bound = B · ε · diam_k for comparison:
    Route A is tighter than raw Huber/TV-ball whenever (1−ε)·MMD < ε·diam_k.
    """
    print("\n" + "="*60)
    print("EXP 2: Non-vacuousness of Route A robust DG certificate")
    print("="*60)
    print(f"  B = {B:.3f}  (normalised classifier weight norm)")
    print(f"  M = {M:.3f}  (normalised sup loss)")

    records  = []

    for held_out in DOMAINS:
        train_doms  = [d for d in DOMAINS if d != held_out]
        train_feats = [domain_feats[d] for d in train_doms]
        test_feats  = domain_feats[held_out]

        mu_train = np.concatenate([f['mu'] for f in train_feats]).mean(0)
        mu_test  = test_feats['mu'].mean(0)
        mmd      = float(np.linalg.norm(mu_train - mu_test))

        eps_train = np.concatenate([f['epsilon'] for f in train_feats]).mean()
        eps_test  = test_feats['epsilon'].mean()
        eps       = float(max(eps_train, eps_test))

        # Route A certificate
        certificate = B * route_a_iipm(eps, mmd)

        # Choquet/TV bound (what you'd get without the ε discount)
        D_k       = float(np.linalg.norm(mu_train))  # approx diam_k(X)
        tv_bound  = B * eps * D_k

        # Actual accuracy gap (normalised to [0,1])
        acc_source = np.mean([held_out_acc[d] for d in train_doms]) / 100.0
        acc_test   = held_out_acc[held_out] / 100.0
        actual_gap = abs(acc_source - acc_test)

        # Robust source risk proxy: 1 − acc_source (misclassification rate)
        r_bar_source = 1.0 - acc_source

        # Bound on test robust risk
        r_bar_test_bound = robust_risk_bound(r_bar_source, B, eps, mmd)

        vacuous    = certificate >= actual_gap
        tighter    = certificate < tv_bound



        records.append({
            'domain':         held_out,
            'mmd':            mmd,
            'eps':            eps,
            'certificate':    certificate,
            'tv_bound':       tv_bound,
            'actual_gap':     actual_gap,
            'r_bar_source':   r_bar_source,
            'r_bar_test_bnd': r_bar_test_bound,
            'r_bar_test_act': 1.0 - acc_test,
            'non_vacuous':    not vacuous,
            'tighter_than_tv': tighter,
        })

    n_nonvac = sum(r['non_vacuous'] for r in records)
    n_tight  = sum(r['tighter_than_tv'] for r in records)


    return {'records': records, 'n_nonvacuous': n_nonvac, 'n_tighter': n_tight}


def experiment_3_adaptive(
    domain_feats: Dict[str, Dict],
    held_out_acc: Dict[str, float],
    M:            float = 1.0,
) -> Dict:
    """
    Exp 3 — Adaptive ε(x) is provably tighter than fixed ε_max.

    For each held-out domain:
      gap = (ε_max − E[ε(x)]) · (M − E[ℓ])

    where:
      ε(x)   = per-instance credal width from ensemble disagreement
      E[ℓ]   = 1 − accuracy (proxy for mean loss)
      gap > 0 iff ε(x) varies across instances AND M > E[ℓ]

    Key connection to UAI paper:
      ε(x) = √Tr(Σ_epi(x)) is already trained to be heterogeneous
      (Mode C joint training). The UAI result (ρ(ε,flip) ≈ 0.64)
      means ε(x) is calibrated — high ε → high flip probability.
      This calibration is what makes adaptive ε tighter than fixed ε_max.
    """
    print("\n" + "="*60)
    print("EXP 3: Adaptive ε(x) vs fixed ε_max — tightening gap")
    print("="*60)

    records = []

    for domain in DOMAINS:
        feats     = domain_feats[domain]
        eps_vec   = feats['epsilon']       # (N,) per-instance ε(x)
        acc       = held_out_acc[domain]
        ell_mean  = 1.0 - acc / 100.0     # proxy for E[ℓ(h,x)]

        stats = adaptive_vs_fixed_gap(eps_vec, M=M, ell_mean=ell_mean)



        records.append({'domain': domain, 'acc': acc, **stats})

    total_gap = sum(r['gap'] for r in records)
    mean_pct  = np.mean([r['gap_pct'] for r in records])


    return {'records': records, 'mean_tightening_pct': mean_pct}


# =============================================================================
# PART 3 — Figures
# =============================================================================

def make_all_figures(
    res1: Dict,
    res2: Dict,
    res3: Dict,
    output_dir: str,
):
    """
    Three-panel figure for the paper:
      Panel A: Route A IIPM vs MMD — shift prediction (Exp 1)
      Panel B: Non-vacuousness — certificate vs actual gap (Exp 2)
      Panel C: Adaptive ε distribution — tightening gap (Exp 3)
    """
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel A: Exp 1 — IIPM vs MMD prediction ──────────────────────────
    ax1 = fig.add_subplot(gs[0])
    records1 = res1['records']
    gaps  = [r['gap']  for r in records1]
    mmds  = [r['mmd']  for r in records1]
    iipms = [r['iipm'] for r in records1]

    def norm01(v):
        v = np.array(v, dtype=float)
        return (v - v.min()) / (v.max() - v.min() + 1e-8)

    ax1.scatter(norm01(mmds),  gaps, s=110, color='#C44E52', alpha=0.9,
                label=f"MMD  (ρ={res1['rho_mmd']:+.2f})", zorder=5)
    ax1.scatter(norm01(iipms), gaps, s=110, marker='D', color='#4C72B0',
                alpha=0.9,
                label=f"(1−ε)·MMD  (ρ={res1['rho_iipm']:+.2f})", zorder=5)

    for v, key, col in [(mmds,'mmd','#C44E52'), (iipms,'iipm','#4C72B0')]:
        xn = norm01(v)
        if len(xn) > 1:
            m, c = np.polyfit(xn, gaps, 1)
            xs = np.linspace(0, 1, 50)
            ax1.plot(xs, m*xs + c, '--', color=col, alpha=0.5, lw=1.5)

    for i, r in enumerate(records1):
        ax1.annotate(DOMAIN_SHORT[r['domain']][:4],
                     (norm01(iipms)[i], gaps[i]),
                     textcoords='offset points', xytext=(5,2),
                     fontsize=8, color='#4C72B0')

    ax1.set_xlabel('Normalised distance  (↑ = more shift)', fontsize=10)
    ax1.set_ylabel('Accuracy gap (pp)', fontsize=10)
    status = "confirmed" if res1['confirmed'] else "not confirmed"
    ax1.set_title(f'A: IIPM=(1−ε)·MMD vs MMD\n'
                  f'$|\\rho_{{\\mathrm{{IIPM}}}}| > |\\rho_{{\\mathrm{{MMD}}}}|$: {status}',
                  fontsize=10)
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    # ── Panel B: Exp 2 — Non-vacuousness ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    records2 = res2['records']
    domains  = [DOMAIN_SHORT[r['domain']] for r in records2]
    x        = np.arange(len(domains))
    w        = 0.28

    certs  = [r['certificate']    for r in records2]
    tvs    = [r['tv_bound']       for r in records2]
    actuals= [r['actual_gap']     for r in records2]

    # Normalise all to same scale
    scale  = max(max(actuals), max(tvs)) + 1e-8

    ax2.bar(x - w, [c/scale for c in certs],   w, color='#4C72B0',
            alpha=0.85, label='Route A cert')
    ax2.bar(x,     [t/scale for t in tvs],      w, color='#8172B2',
            alpha=0.85, label='TV/Huber cert')
    ax2.bar(x + w, [a/scale for a in actuals],  w, color='#55A868',
            alpha=0.85, label='Actual gap')

    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, fontsize=9)
    ax2.set_ylabel('Normalised value', fontsize=10)
    n_nv = res2['n_nonvacuous']
    ax2.set_title(f'B: Certificate non-vacuousness\n'
                  f'Route A < actual gap: {n_nv}/{len(DOMAINS)} domains',
                  fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.25)

    # ── Panel C: Exp 3 — Adaptive ε distributions ────────────────────────
    ax3 = fig.add_subplot(gs[2])
    records3 = res3['records']

    eps_means = [r['eps_mean'] for r in records3]
    eps_maxes = [r['eps_max']  for r in records3]
    gaps3     = [r['gap']      for r in records3]
    dom_labels= [DOMAIN_SHORT[r['domain']] for r in records3]

    x3 = np.arange(len(records3))
    w3 = 0.35

    bars_mean = ax3.bar(x3 - w3/2, eps_means, w3, color='#4C72B0',
                        alpha=0.85, label='E[ε(x)] adaptive')
    bars_max  = ax3.bar(x3 + w3/2, eps_maxes, w3, color='#C44E52',
                        alpha=0.85, label='ε_max fixed')

    # Annotate tightening gap
    for i, (em, ex, g) in enumerate(zip(eps_means, eps_maxes, gaps3)):
        ax3.annotate(f'gap\n={g:.3f}',
                     xy=(x3[i], max(em, ex) + 0.005),
                     ha='center', fontsize=7, color='#333333')

    ax3.set_xticks(x3)
    ax3.set_xticklabels(dom_labels, fontsize=9)
    ax3.set_ylabel('ε value', fontsize=10)
    mean_pct = res3['mean_tightening_pct']
    ax3.set_title(f'C: Adaptive vs fixed ε\n'
                  f'Mean tightening = {mean_pct:.1f}%',
                  fontsize=10)
    ax3.legend(fontsize=8, framealpha=0.9)
    ax3.grid(True, axis='y', alpha=0.25)

    fig.suptitle(
        'Route A Robust DG Certificate: '
        '$|\\bar{{R}}(h,\\mathcal{{C}}_\\mathrm{{test}}) - '
        '\\bar{{R}}(h,\\mathcal{{C}}_\\mathrm{{source}})| '
        '\\leq B\\cdot(1-\\varepsilon)\\cdot\\mathrm{{MMD}}$',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    out_path = Path(output_dir) / 'fig_robust_dg_certificate.pdf'
    plt.savefig(str(out_path), bbox_inches='tight', dpi=150)
    plt.close()

    return str(out_path)


def make_latex_table(res1: Dict, res2: Dict, res3: Dict, output_dir: str):
    """
    Drop-in LaTeX table for the paper.
    One row per domain, columns: MMD, IIPM=(1-ε)·MMD, cert, actual_gap, ε_mean, gap%.
    Last rows: Spearman ρ.
    """
    r1 = {r['domain']: r for r in res1['records']}
    r2 = {r['domain']: r for r in res2['records']}
    r3 = {r['domain']: r for r in res3['records']}

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Route A robust DG certificate on PACS.',
        r'IIPM~$= (1-\varepsilon)\cdot\mathrm{MMD}$;',
        r'certificate~$= B\cdot(1-\varepsilon)\cdot\mathrm{MMD}$;',
        r'actual gap~$= |\bar{\mathrm{acc}}_{\mathrm{train}} - \mathrm{acc}_{\mathrm{test}}|$;',
        r'adaptive gap~$= (\varepsilon_{\max} - \mathbb{E}[\varepsilon(x)])\cdot(M - \mathbb{E}[\ell])$.}',
        r'\label{tab:route_a_certificate}',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{lccccccc}',
        r'\toprule',
        r'Domain & Acc (\%) & MMD & $(1-\varepsilon)\cdot$MMD & Cert & Actual gap '
        r'& $\mathbb{E}[\varepsilon]$ & Adapt. gap \\',
        r'\midrule',
    ]

    for domain in DOMAINS:
        a = ERM_ACC[domain]
        d1, d2, d3 = r1[domain], r2[domain], r3[domain]
        nv = '\\checkmark' if d2['non_vacuous'] else '$\\times$'
        lines.append(
            f"{domain.replace('_',' ')} & {a:.1f} "
            f"& {d1['mmd']:.3f} "
            f"& {d1['iipm']:.3f} "
            f"& {d2['certificate']:.3f} "
            f"& {d2['actual_gap']:.3f}\\,{nv} "
            f"& {d3['eps_mean']:.3f} "
            f"& {d3['gap']:.4f} \\\\"
        )

    lines += [
        r'\midrule',
        f"Spearman $\\rho$ & — & {res1['rho_mmd']:+.2f} "
        f"& \\textbf{{{res1['rho_iipm']:+.2f}}} & — & — & — & — \\\\",
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}',
    ]

    out = Path(output_dir) / 'table_route_a.tex'
    out.write_text('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Route A Robust DG Certificate — three experiments on PACS'
    )
    parser.add_argument('--pacs_root',  type=str, default=None,
                        help='Path to PACS root (for fresh feature extraction)')
    parser.add_argument('--feats_npz',  type=str, default=None,
                        help='Pre-extracted features .npz from credal_dg_pacs.py')
    parser.add_argument('--output_dir', type=str, default='robust_dg_results')
    parser.add_argument('--arch',       type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'])
    parser.add_argument('--n_heads',    type=int,  default=5)
    parser.add_argument('--dropout_p', type=float, default=0.15)
    parser.add_argument('--max_samples',type=int,  default=400)
    parser.add_argument('--device',     type=str,  default='cpu')
    parser.add_argument('--B',          type=float, default=1.0,
                        help='RKHS norm of loss B = ||W||_F (normalised)')
    parser.add_argument('--M',          type=float, default=1.0,
                        help='Supremum of normalised loss')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load or extract features ──────────────────────────────────────────
    domain_feats = load_or_extract_features(
        pacs_root=args.pacs_root,
        feats_npz=args.feats_npz,
        arch=args.arch,
        n_heads=args.n_heads,
        dropout_p=args.dropout_p,
        max_samples=args.max_samples,
        device=args.device,
    )

    # Save features for reuse
    feats_path = out / 'domain_feats.npz'
    np.savez(str(feats_path), domain_feats=domain_feats)


    # ── Run three experiments ─────────────────────────────────────────────
    res1 = experiment_1_prediction(domain_feats, ERM_ACC)
    res2 = experiment_2_nonvacuous(domain_feats, ERM_ACC, B=args.B, M=args.M)
    res3 = experiment_3_adaptive(domain_feats, ERM_ACC, M=args.M)

    # ── Figures + table ───────────────────────────────────────────────────
    make_all_figures(res1, res2, res3, str(out))
    make_latex_table(res1, res2, res3, str(out))

    # ── JSON dump ─────────────────────────────────────────────────────────
    results = {
        'exp1': convert_numpy_types(res1), 
        'exp2': convert_numpy_types(res2), 
        'exp3': convert_numpy_types(res3)
    }
    (out / 'results.json').write_text(json.dumps(results, indent=2))

    # ── Final summary ─────────────────────────────────────────────────────



if __name__ == '__main__':
    main()