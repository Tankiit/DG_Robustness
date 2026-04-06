"""
credal_dg_pacs.py
=================
Credal IIPM vs MMD for Domain Generalisation — PACS dataset.

PRIMARY METHOD: Credal ellipsoid from MC Dropout (H stochastic passes).
  - ε(x)   = sqrt(mean_d σ²_d(x))  — TV outer approximation radius
  - ε_dom  = mean_x ε(x)            — domain-level contamination
  - MMI    = 2·max_d σ_d            — support function at widest direction
  - All quantities live in feature space (D=512), same space as MMD.
  - Direct theoretical connection via C(x) ⊆ B_ε(μ(x)) [Mukherjee et al.]
  - No training required. No sensitivity to epoch count.

ABLATION (kept for later): Laplace approximation on linear head.
  - fit_and_query_laplace() is preserved but NOT called in run_experiment().
  - ε lives in logit space (C=7); requires bridging argument for paper.
  - Sensitive to head_epochs — use ≤15 if you call it.

Architecture:
  - Frozen ResNet-18 backbone (ImageNet pretrained)
  - Dropout(p=0.15) on penultimate 512-dim features
  - H=5 stochastic passes per target domain → Σ_epi(x)

USAGE:
  python credal_dg_pacs.py --pacs_root /path/to/PACS
  python credal_dg_pacs.py --pacs_root /path/to/PACS --H 10
  python credal_dg_pacs.py --pacs_root /path/to/PACS --max_samples 100

OUTPUTS (in ./credal_dg_results/):
  results.json
  fig_iipm_vs_mmd.pdf
  fig_dro_mmi_triangle.pdf
  table_latex.tex

RUNTIME (MPS, ResNet-18, PACS, H=5, 400 samples/domain):
  Feature extraction (H passes):  ~3-5 min total
  Total:                          ~5-8 min  (no Laplace fitting)
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from laplace import Laplace   # used only in fit_and_query_laplace (ablation)

warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

# Published ERM accuracy (ResNet-50, training-domain validation)
# Source: Gulrajani & Lopez-Paz "In Search of Lost Domain Generalization" (2021)
ERM_ACC_PACS = {
    'art_painting': 84.7,
    'cartoon':      80.8,
    'photo':        96.0,
    'sketch':       79.3,
}

DOMAINS    = ['art_painting', 'cartoon', 'photo', 'sketch']
N_CLASSES  = 7
FEAT_DIM   = 512
COLORS     = {
    'iipm': '#4C72B0',
    'mmd':  '#C44E52',
    'mmi':  '#55A868',
    'eps':  '#8172B2',
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def parse_label_file(label_file: str, img_root: str) -> List[Tuple[str, int]]:
    """
    Parse a PACS kfold label file.
    Each line: <relative_path> <class_index>
    Returns list of (absolute_image_path, class_index).
    """
    samples = []
    label_path = Path(label_file)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) != 2:
                continue
            rel_path, label = parts
            label = int(label)
            full_path = Path(img_root) / rel_path
            if not full_path.exists():
                # Some label files omit the leading domain prefix
                alt_path = Path(img_root) / rel_path.split('/', 1)[-1]
                if alt_path.exists():
                    full_path = alt_path
            samples.append((str(full_path), label))
    return samples


def load_image(path: str, img_size: int = 224) -> np.ndarray:
    """Load one image → (3, H, W) float32, ImageNet normalised."""
    from PIL import Image
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)
    except Exception:
        return np.zeros((3, img_size, img_size), dtype=np.float32)


class PACSDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], img_size: int = 224):
        self.samples  = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image(path, self.img_size)
        return torch.from_numpy(img), label


# =============================================================================
# 2. BACKBONE  (frozen ResNet-18 + dropout + trainable linear head)
# =============================================================================

class FrozenResNet18WithDropout(nn.Module):
    """
    Frozen ImageNet ResNet-18 backbone with:
      - Dropout(p) on penultimate features (for stochastic passes)
      - Trainable linear head (for Laplace approximation)

    Call model.train() to activate dropout during inference.
    Backbone parameters are always frozen (requires_grad=False).
    """
    def __init__(self, num_classes: int = 7, p_drop: float = 0.15):
        super().__init__()
        base = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # Remove classification head — keep up to avgpool
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.drop     = nn.Dropout(p=p_drop)
        self.head     = nn.Linear(FEAT_DIM, num_classes)

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).flatten(1)   # (B, 512)
        feats = self.drop(feats)              # stochastic if .train()
        return self.head(feats)              # (B, C)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return post-dropout features without the head."""
        feats = self.backbone(x).flatten(1)
        return self.drop(feats)


def extract_features_single(
    loader: DataLoader,
    model: FrozenResNet18WithDropout,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single deterministic pass (dropout OFF) for MMD feature means.
    Returns (feats: N×512, labels: N).
    """
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            feats = model.backbone(imgs.to(device)).flatten(1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def extract_credal_ellipsoid(
    loader: DataLoader,
    model: FrozenResNet18WithDropout,
    device: str,
    H: int = 5,
) -> Dict:
    """
    Run H stochastic forward passes (dropout ON) through frozen backbone.
    Compute the credal ellipsoid diagonal Σ_epi(x) for each instance.

    This is the PRIMARY ε estimator. It lives in feature space (D=512),
    exactly where MMD lives, so no bridging argument is needed.

    Theory connection (Route A, §4 of paper):
      C(x) = {h : (h−μ)ᵀ Σ_epi⁻¹ (h−μ) ≤ 1}
      C(x) ⊆ B_ε(μ(x))  with  ε(x) = sqrt(mean_d σ²_d(x))
      [Mukherjee et al. 2026, Prop. 1]

    Returns:
      mu        : (N, 512)  mean feature per instance
      sigma_sq  : (N, 512)  per-dimension variance  ← Σ_epi diagonal
      eps_per   : (N,)      per-instance ε = sqrt(mean_d σ²_d)
      eps_domain: scalar    mean ε over target domain
      mmi       : scalar    2 · max_d sqrt(mean_n σ²_d(x))
    """
    # Activate dropout, freeze BN at running stats
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    # Collect all images first (we need to re-pass H times)
    all_imgs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_imgs.append(imgs)
            all_labels.append(labels)
    all_imgs   = torch.cat(all_imgs)    # (N, 3, 224, 224)
    all_labels = torch.cat(all_labels)  # (N,)
    N = len(all_imgs)

    # H stochastic passes — each gives a different dropout mask
    passes = []
    batch_size = loader.batch_size or 32
    with torch.no_grad():
        for h in range(H):
            h_feats = []
            for start in range(0, N, batch_size):
                batch = all_imgs[start:start + batch_size].to(device)
                feats = model.get_features(batch).cpu()
                h_feats.append(feats)
            passes.append(torch.cat(h_feats))   # (N, 512)

    passes   = torch.stack(passes)               # (H, N, 512)
    mu       = passes.mean(dim=0)                # (N, 512)
    sigma_sq = passes.var(dim=0, unbiased=True)  # (N, 512)  Σ_epi diagonal

    # ε per instance: TV ball outer approximation radius
    eps_per   = sigma_sq.max(dim=-1).values.sqrt()     # (N,)
    eps_domain = eps_per.mean().item()

    # MMI: support function of credal ellipsoid at widest feature direction
    # σ_d = sqrt(mean_n σ²_d(x))  — per-dimension std averaged over instances
    sigma_d = sigma_sq.mean(dim=0).sqrt()        # (512,)
    mmi     = (2.0 * sigma_d.max()).item()

    return {
        'mu':         mu,
        'sigma_sq':   sigma_sq,
        'eps_per':    eps_per,
        'eps_domain': eps_domain,
        'mmi':        mmi,
        'labels':     all_labels,
    }


# =============================================================================
# 3. TRAIN LINEAR HEAD (MAP estimate for Laplace)
# =============================================================================

def train_head(
    head: nn.Linear,
    src_feats: torch.Tensor,
    src_labels: torch.Tensor,
    device: str = 'cpu',
    epochs: int = 15,
    lr: float = 1e-3,
) -> None:
    """
    Train the linear head on source features.
    Laplace needs a MAP estimate to expand around.
    """
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(src_feats.to(device), src_labels.to(device))
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(loader):.4f}")

    head.eval()


# =============================================================================
# 4. LAPLACE APPROXIMATION  →  ε per domain
# =============================================================================

def fit_and_query_laplace(
    head: nn.Linear,
    src_feats: torch.Tensor,
    src_labels: torch.Tensor,
    tgt_feats: torch.Tensor,
    device: str = 'cpu',
) -> Dict:
    """
    Fit diagonal Laplace on the linear head, query on target features.

    API notes:
    - weight_subset='all': the model IS a single Linear layer;
      'last_layer' requires laplace-torch's feature extractor hook which
      fails on bare Linear modules. 'all' is identical here.
    - hessian_structure='diag': 'kron' requires multi-layer structure
      and crashes on single-layer models.
    - dummy wrapper: laplace-torch requires len(list(model.modules())) > 2.

    ε normalisation:
    - Raw GLM predictive variance is in logit units (unbounded).
    - We normalise via sigmoid(logit_std − median(logit_std)).
    - This keeps ε ∈ (0,1) and is ROBUST to the overfitting level of the
      MAP estimate: the median domain always maps to ε≈0.5 regardless of
      absolute variance scale. The relative ordering — which is what drives
      the (1−ε)·MMD discount — is preserved.
    - WARNING: if head_epochs is too high (>20), the MAP estimate overfits
      and logit variances inflate uniformly, compressing the spread. Use
      head_epochs=15 (default) for best differential signal.
    """
    head = head.to(device)
    dummy = nn.Sequential(nn.Identity(), nn.Identity(), head)

    la = Laplace(
        dummy,
        likelihood='classification',
        subset_of_weights='all',
        hessian_structure='diag',
    )

    src_dl = DataLoader(
        TensorDataset(src_feats.to(device), src_labels.to(device)),
        batch_size=256, shuffle=True,
    )
    la.fit(src_dl)
    la.optimize_prior_precision(method='marglik')

    with torch.no_grad():
        tgt_dev = tgt_feats.to(device)
        f_mean, f_var = la._glm_predictive_distribution(tgt_dev)
        if f_var.dim() == 3:
            f_var = f_var.diagonal(dim1=-2, dim2=-1)   # (N, C)

    # Per-instance logit std: mean over C classes, then sqrt
    logit_std = f_var.mean(dim=-1).sqrt().cpu()         # (N,)

    # Median-normalised sigmoid: robust to absolute variance scale
    # σ > median → ε > 0.5 (more uncertain); σ < median → ε < 0.5
    median_std   = logit_std.median()
    eps_per_inst = torch.sigmoid(logit_std - median_std)  # (N,) ∈ (0,1)
    eps_domain   = eps_per_inst.mean().item()

    # MMI proxy: 2 * max_c sqrt(mean_n f_var[n,c])
    per_class_std = f_var.cpu().mean(dim=0).sqrt()
    mmi = (2.0 * per_class_std.max()).item()

    return {
        'eps':          eps_domain,
        'eps_per_inst': eps_per_inst,
        'f_var':        f_var.cpu(),
        'mmi':          mmi,
    }


# =============================================================================
# 5. DISTANCE MEASURES
# =============================================================================

def compute_mmd_linear(
    mu_sources: List[torch.Tensor],
    mu_target: torch.Tensor,
) -> float:
    """
    Linear kernel MMD: ||mean(source features) - mean(target features)||_2.
    mu_sources: list of (N_m, 512) feature tensors, one per source domain.
    mu_target:  (N_t, 512) feature tensor.
    """
    mu_s = torch.cat(mu_sources, dim=0).mean(dim=0)  # (512,)
    mu_t = mu_target.mean(dim=0)                      # (512,)
    return (mu_s - mu_t).norm().item()


def compute_mmi_from_laplace(f_var: torch.Tensor) -> float:
    """
    MMI = 2 * max_c sqrt(mean_n f_var[n, c]).
    Analogous to 2 * max_d sigma_d from feature-space dropout,
    but over class dimensions (logit space) from Laplace.

    This is a proxy MMI. For exact feature-space MMI you need
    per-dimension dropout variance (see architecture discussion).
    """
    per_class_std = f_var.mean(dim=0).sqrt()          # (C,)
    return (2.0 * per_class_std.max()).item()


# =============================================================================
# 6. LEAVE-ONE-DOMAIN-OUT EXPERIMENT
# =============================================================================

def run_experiment(
    domain_loaders: Dict[str, DataLoader],
    model: FrozenResNet18WithDropout,
    held_out_acc: Dict[str, float],
    device: str,
    H: int = 5,
) -> Dict:
    """
    Leave-one-domain-out experiment using credal ellipsoid ε.

    For each held-out domain:
      1. Extract deterministic features for all domains → MMD
      2. Run H stochastic passes on target domain → Σ_epi, ε, MMI
      3. Compute (1−ε)·MMD certificate

    No head training required. No Laplace. Pure feature-space geometry.
    """
    records = []

    # Step 1: extract deterministic features for all domains (for MMD)
    print("  Extracting deterministic features for MMD...")
    det_feats = {}
    for d in DOMAINS:
        feats, labels = extract_features_single(domain_loaders[d], model, device)
        det_feats[d] = (feats, labels)
        print(f"    [{d}] {feats.shape}")

    # Step 2: LODO loop
    for held_out in DOMAINS:
        print(f"\n--- Held-out: {held_out} ---")
        source_domains = [d for d in DOMAINS if d != held_out]

        # MMD: deterministic source mean vs deterministic target mean
        mmd = compute_mmd_linear(
            [det_feats[d][0] for d in source_domains],
            det_feats[held_out][0],
        )

        # Credal ellipsoid: H stochastic passes on target only
        print(f"  Computing credal ellipsoid (H={H} passes)...")
        credal = extract_credal_ellipsoid(
            domain_loaders[held_out], model, device, H=H
        )
        eps  = credal['eps_domain']
        mmi  = credal['mmi']
        cert = (1.0 - eps) * mmd

        true_acc = held_out_acc.get(held_out, 0.0)
        print(
            f"  acc={true_acc:.1f}%  "
            f"ε={eps:.3f}  mmd={mmd:.3f}  "
            f"(1−ε)·MMD={cert:.3f}  MMI={mmi:.3f}"
        )

        records.append({
            'domain':   held_out,
            'accuracy': true_acc,
            'mmd':      mmd,
            'cert':     cert,
            'mmi':      mmi,
            'eps':      eps,
        })

    # Spearman ρ for each measure vs held-out accuracy
    accs  = [r['accuracy'] for r in records]
    mmds  = [r['mmd']      for r in records]
    certs = [r['cert']     for r in records]
    mmis  = [r['mmi']      for r in records]
    epss  = [r['eps']      for r in records]

    rho_mmd,  p_mmd  = spearmanr(mmds,  accs)
    rho_cert, p_cert = spearmanr(certs, accs)
    rho_mmi,  p_mmi  = spearmanr(mmis,  accs)
    rho_eps,  p_eps  = spearmanr(epss,  accs)

    rho_iipm, p_iipm = rho_cert, p_cert

    print(f"\n{'─'*55}")
    print(f"  {'Measure':<22}  {'ρ':>6}   {'p':>6}")
    print(f"{'─'*55}")
    for label, rho, p in [
        ('MMD (baseline)',      rho_mmd,  p_mmd),
        ('(1−ε)·MMD  [cert]',  rho_cert, p_cert),
        ('MMI',                rho_mmi,  p_mmi),
        ('ε (credal)',         rho_eps,  p_eps),
    ]:
        print(f"  {label:<22}  {rho:+.3f}   {p:.3f}")
    print(f"{'─'*55}")

    confirmed = abs(rho_cert) > abs(rho_mmd)
    print(f"\n  Hypothesis confirmed: {confirmed}")
    print(f"  (|ρ_cert|={abs(rho_cert):.3f}  >  |ρ_mmd|={abs(rho_mmd):.3f})")

    return {
        'records':   records,
        'rho_mmd':   float(rho_mmd),   'p_mmd':   float(p_mmd),
        'rho_iipm':  float(rho_iipm),  'p_iipm':  float(p_iipm),
        'rho_mmi':   float(rho_mmi),   'p_mmi':   float(p_mmi),
        'rho_eps':   float(rho_eps),   'p_eps':   float(p_eps),
        'confirmed': confirmed,
        'H':         H,
    }


# =============================================================================
# 7. FIGURES
# =============================================================================

def plot_scatter(results: Dict, output_path: str) -> None:
    """
    Scatter: (1-ε)·MMD vs MMD as predictors of held-out accuracy.
    Steeper negative slope for cert = hypothesis confirmed.
    """
    records = results['records']
    accs    = np.array([r['accuracy'] for r in records])

    def norm(key: str) -> np.ndarray:
        v = np.array([r[key] for r in records], dtype=float)
        rng = v.max() - v.min()
        return (v - v.min()) / (rng + 1e-8)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.scatter(
        norm('mmd'), accs, s=100,
        color=COLORS['mmd'], zorder=5, alpha=0.9,
        label=f"MMD  (ρ={results['rho_mmd']:+.2f})",
    )
    ax.scatter(
        norm('cert'), accs, s=100, marker='D',
        color=COLORS['iipm'], zorder=5, alpha=0.9,
        label=f"(1−ε)·MMD  (ρ={results['rho_iipm']:+.2f})",
    )

    # Trend lines
    for key, col in [('mmd', COLORS['mmd']), ('cert', COLORS['iipm'])]:
        xn = norm(key)
        m, c = np.polyfit(xn, accs, 1)
        xs = np.linspace(-0.05, 1.05, 60)
        ax.plot(xs, m * xs + c, '--', color=col, alpha=0.55, lw=1.6)

    # Domain labels on cert points
    domains = [r['domain'].replace('_', '\n') for r in records]
    for i, d in enumerate(domains):
        ax.annotate(
            d, (norm('cert')[i], accs[i]),
            textcoords='offset points', xytext=(6, 2),
            fontsize=8, color=COLORS['iipm'],
        )

    ax.set_xlim(-0.1, 1.3)
    ax.set_xlabel("Normalised distance  (↑ = more shift)", fontsize=11)
    ax.set_ylabel("Held-out domain accuracy (%)", fontsize=11)
    status = "confirmed ✓" if results['confirmed'] else "not confirmed ✗"
    ax.set_title(
        f"PACS: (1−ε)·MMD vs raw MMD  —  {status}",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_triangle(results: Dict, output_path: str) -> None:
    """
    Bar chart: MMD / (1-ε)·MMD / MMI per domain, sorted by accuracy.
    Overlaid with ERM accuracy as a line.
    All three measures should rank domains consistently.
    """
    records = sorted(results['records'], key=lambda r: r['accuracy'])
    labels  = [r['domain'].replace('_', '\n') for r in records]
    accs    = [r['accuracy'] for r in records]

    def norm(key: str) -> np.ndarray:
        v = np.array([r[key] for r in records], dtype=float)
        return v / (v.max() + 1e-8)

    x = np.arange(len(records))
    w = 0.22

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(x - w, norm('cert'), w, color=COLORS['iipm'],
            alpha=0.85, label='(1−ε)·MMD  [certificate]')
    ax1.bar(x,     norm('mmd'),  w, color=COLORS['mmd'],
            alpha=0.85, label='MMD  [baseline]')
    ax1.bar(x + w, norm('mmi'),  w, color=COLORS['mmi'],
            alpha=0.85, label='MMI  [= 2 max σ]')

    ax2.plot(x, accs, 'ko--', ms=7, lw=2,
             label='ERM accuracy (%)', zorder=10)
    ax2.set_ylabel("Held-out accuracy (%)", fontsize=10)
    ax2.set_ylim(50, 115)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Normalised measure  (↑ = harder)", fontsize=10)
    ax1.set_ylim(0, 1.3)
    ax1.set_title(
        "PACS leave-one-domain-out  ·  credal ellipsoid ε  ·  frozen ResNet-18",
        fontsize=10, pad=8,
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8,
               loc='upper left', framealpha=0.9)
    ax1.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# 8. LATEX TABLE
# =============================================================================

def write_latex_table(results: Dict, output_path: str) -> None:
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{',
        r'  Credal DG diagnostic on PACS leave-one-domain-out.',
        r'  \emph{Accuracy}: published ERM baseline (ResNet-50, \citealt{gulrajani2021search}).',
        r'  \emph{ε}: credal ellipsoid radius (MC dropout, feature space).',
        r'  $(1{-}\varepsilon)\cdot$MMD: credal certificate.',
        r'  MMI: maximum epistemic spread.',
        r'  $\rho$: Spearman with held-out accuracy.',
        r'}',
        r'\label{tab:pacs}',
        r'\small',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Domain & Acc (\%) & ε & MMD & $(1{-}ε)\cdot$MMD & MMI \\',
        r'\midrule',
    ]

    for r in sorted(results['records'], key=lambda x: -x['accuracy']):
        lines.append(
            f"  {r['domain'].replace('_', ' ')} & {r['accuracy']:.1f} "
            f"& {r['eps']:.3f} & {r['mmd']:.2f} "
            f"& {r['cert']:.2f} & {r['mmi']:.3f} \\\\"
        )

    lines += [
        r'\midrule',
        f"  Spearman $\\rho$ & --- & {results['rho_eps']:+.2f} "
        f"& {results['rho_mmd']:+.2f} "
        f"& \\textbf{{{results['rho_iipm']:+.2f}}} "
        f"& {results['rho_mmi']:+.2f} \\\\",
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    Path(output_path).write_text('\n'.join(lines))
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def pick_device(requested: str) -> str:
    """Resolve device. 'auto' prefers MPS > CUDA > CPU."""
    if requested != 'auto':
        return requested
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description='Credal IIPM vs MMD on PACS — credal ellipsoid ε'
    )
    parser.add_argument('--pacs_root',   type=str, required=True,
                        help='Path to PACS/ (contains pacs_data/ pacs_label/)')
    parser.add_argument('--output_dir',  type=str, default='credal_dg_results')
    parser.add_argument('--max_samples', type=int, default=400)
    parser.add_argument('--split',       type=str, default='test',
                        choices=['test', 'crossval'])
    parser.add_argument('--device',      type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--img_size',    type=int, default=224)
    parser.add_argument('--H',           type=int, default=5,
                        help='Number of stochastic dropout passes for '
                             'credal ellipsoid estimation')
    parser.add_argument('--dropout_p',   type=float, default=0.15)
    args = parser.parse_args()

    pacs_root  = Path(args.pacs_root)
    img_root   = pacs_root / 'pacs_data'
    label_root = pacs_root / 'pacs_label'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not img_root.exists():
        raise FileNotFoundError(f"pacs_data/ not found at {img_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"pacs_label/ not found at {label_root}")

    device = pick_device(args.device)
    print(f"\nDevice: {device}  |  H={args.H}  |  dropout_p={args.dropout_p}"
          f"  |  batch={args.batch_size}  |  img_size={args.img_size}")

    # ── Load samples ──────────────────────────────────────────────────────
    domain_samples = {}
    for domain in DOMAINS:
        label_file = label_root / f'{domain}_{args.split}_kfold.txt'
        print(f"\n[{domain}] {label_file.name}")
        samples = parse_label_file(str(label_file), str(img_root))
        if args.max_samples and len(samples) > args.max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(samples), size=args.max_samples, replace=False)
            samples = [samples[i] for i in idx]
        domain_samples[domain] = samples
        print(f"  {len(samples)} instances.")

    # ── Build loaders (images stay on disk, loaded on demand) ─────────────
    # We keep loaders rather than cached tensors so extract_credal_ellipsoid
    # can re-pass images H times with fresh dropout masks.
    domain_loaders = {
        d: DataLoader(
            PACSDataset(domain_samples[d], img_size=args.img_size),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        for d in DOMAINS
    }

    # ── Build backbone ────────────────────────────────────────────────────
    model = FrozenResNet18WithDropout(
        num_classes=N_CLASSES, p_drop=args.dropout_p
    ).to(device)

    # ── Run LODO experiment ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Leave-One-Domain-Out  (credal ellipsoid, H={args.H})")
    print(f"{'='*55}")
    results = run_experiment(
        domain_loaders, model, ERM_ACC_PACS, device, H=args.H
    )

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_scatter(results,  str(output_dir / 'fig_iipm_vs_mmd.pdf'))
    plot_triangle(results, str(output_dir / 'fig_dro_mmi_triangle.pdf'))
    write_latex_table(results, str(output_dir / 'table_latex.tex'))

    # ── JSON ──────────────────────────────────────────────────────────────
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(i) for i in obj]
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    (output_dir / 'results.json').write_text(
        json.dumps(to_python(results), indent=2)
    )
    print(f"\nAll outputs in: {output_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()
