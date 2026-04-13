from __future__ import annotations

"""Credal DG / epistemic-uncertainty experiments (PACS-focused CLI).

For the NeurIPS-style local and MPS experiment suite (Tables 1–3, OfficeHome,
CIFAR-10-C, H and backbone ablations, certificate outputs), see
``credal_dg_local.py`` in this directory. DomainNet-scale runs use
``credal_dg_modal.py``.
"""

import argparse
import json
import math
import warnings
from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.models as tv_models
import torchvision.transforms as TVT
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
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
        feats = self.backbone(x).flatten(1)
        return self.drop(feats)


def extract_features_single(
    loader: DataLoader,
    model: FrozenResNet18WithDropout,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(src_feats.to(device), src_labels.to(device))
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(head(x), y)
            loss.backward()
            optimizer.step()

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
    mu_s = torch.cat(mu_sources, dim=0).mean(dim=0)  # (512,)
    mu_t = mu_target.mean(dim=0)                      # (512,)
    return (mu_s - mu_t).norm().item()


def compute_mmi_from_laplace(f_var: torch.Tensor) -> float:
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
    records = []

    # Step 1: extract deterministic features for all domains (for MMD)
    det_feats = {}
    for d in DOMAINS:
        feats, labels = extract_features_single(domain_loaders[d], model, device)
        det_feats[d] = (feats, labels)

    # Step 2: LODO loop
    for held_out in DOMAINS:
        source_domains = [d for d in DOMAINS if d != held_out]

        # MMD: deterministic source mean vs deterministic target mean
        mmd = compute_mmd_linear(
            [det_feats[d][0] for d in source_domains],
            det_feats[held_out][0],
        )

        # Credal ellipsoid: H stochastic passes on target only
        credal = extract_credal_ellipsoid(
            domain_loaders[held_out], model, device, H=H
        )
        eps  = credal['eps_domain']
        mmi  = credal['mmi']
        cert = (1.0 - eps) * mmd

        true_acc = held_out_acc.get(held_out, 0.0)

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

    confirmed = abs(rho_cert) > abs(rho_mmd)

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
    status = "confirmed" if results['confirmed'] else "not confirmed"
    ax.set_title(
        f"PACS: (1−ε)·MMD vs raw MMD  —  {status}",
        fontsize=10, pad=8,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_triangle(results: Dict, output_path: str) -> None:
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


# =============================================================================
# UNIFIED MPS PIPELINE  (credal_dg_mps — NeurIPS 2026 Kernel IIPM paper)
#   Domain-structured folders: data_root /<domain>/<class>/*.{jpg,png,...}
#   Run:  python credal_dg.py mps --exp e1 --dataset pacs --data_root ...
# =============================================================================

_IMG_EXTS_MPS = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG'}

GT_ACC_MPS = {
    'pacs': {
        'art_painting': 84.7,
        'cartoon':      80.8,
        'photo':        97.2,
        'sketch':       79.3,
    },
    'officehome': {
        'Art':        61.3,
        'Clipart':    52.4,
        'Product':    75.8,
        'Real_World': 76.6,
    },
    'cifar10c': {},
}

N_CLASSES_MPS = {
    'pacs':       7,
    'officehome': 65,
    'domainnet':  345,
    'cifar10c':   10,
    'iwildcam':   182,
}

E3_HEAD_EPOCHS = [1, 3, 5, 10, 15, 20]
E3_DROPOUT_P = [0.05, 0.10, 0.15, 0.20, 0.30]
ENSEMBLE_SEEDS = [0, 1, 2, 3]

_MPS_MEAN = [0.485, 0.456, 0.406]
_MPS_STD = [0.229, 0.224, 0.225]
transform_eval_mps = TVT.Compose([
    TVT.Resize(256),
    TVT.CenterCrop(224),
    TVT.ToTensor(),
    TVT.Normalize(_MPS_MEAN, _MPS_STD),
])


class DomainDatasetMPS(Dataset):
    def __init__(self, domain_dir: Path, n: Optional[int] = None, seed: int = 42):
        paths = [p for p in sorted(domain_dir.rglob('*')) if p.suffix in _IMG_EXTS_MPS]
        if n is not None and n < len(paths):
            rng = np.random.default_rng(seed)
            paths = rng.choice(paths, n, replace=False).tolist()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = PILImage.open(self.paths[i]).convert('RGB')
        return transform_eval_mps(img)


def build_frozen_resnet18_mps() -> Tuple[nn.Module, nn.Linear]:
    base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    features = nn.Sequential(*list(base.children())[:-1])
    head = base.fc
    for p in features.parameters():
        p.requires_grad_(False)
    return features, head


def _enable_dropout_only_mps(model: nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


class EpsEstimator(ABC):
    @abstractmethod
    def estimate(self, loader: DataLoader, device: str) -> dict:
        ...

    @staticmethod
    def _pack(mu: torch.Tensor, sigma: torch.Tensor, D: int) -> dict:
        eps = (sigma.pow(2).mean(dim=1).sqrt() / math.sqrt(D)).clamp(1e-6, 1 - 1e-6)
        mmi = 2.0 * sigma.max(dim=1).values
        return {'mu': mu, 'sigma': sigma, 'eps': eps, 'mmi': mmi}


class MCDropoutEstimator(EpsEstimator):
    def __init__(self, H: int = 5, p: float = 0.15):
        self.H = H
        self.p = p

    def estimate(self, loader: DataLoader, device: str) -> dict:
        features, _ = build_frozen_resnet18_mps()
        dropout = nn.Dropout(p=self.p)
        backbone = nn.Sequential(features, nn.Flatten(), dropout).to(device)
        _enable_dropout_only_mps(backbone)

        all_mu, all_sigma = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device)
                passes = torch.stack([backbone(x) for _ in range(self.H)], dim=1)
                all_mu.append(passes.mean(1).cpu())
                all_sigma.append(passes.std(1).cpu())

        mu = torch.cat(all_mu, dim=0)
        sigma = torch.cat(all_sigma, dim=0)
        D = mu.shape[1]
        return self._pack(mu, sigma, D)


class LaplaceEstimatorMPS(EpsEstimator):
    def estimate(self, loader: DataLoader, device: str) -> dict:
        from laplace import Laplace

        base = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
        ).to(device).eval()
        for name, p in base.named_parameters():
            if 'fc' not in name:
                p.requires_grad_(False)

        la = Laplace(
            base,
            likelihood='classification',
            subset_of_weights='last_layer',
            hessian_structure='diag',
        )

        pseudo_data = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device)
                out = base(x)
                y = out.argmax(1)
                pseudo_data.append((x.cpu(), y.cpu()))

        class PseudoDS(Dataset):
            def __init__(self, data):
                self.x = torch.cat([d[0] for d in data])
                self.y = torch.cat([d[1] for d in data])

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return self.x[i], self.y[i]

        pseudo_ldr = DataLoader(PseudoDS(pseudo_data), batch_size=64, shuffle=False)
        la.fit(pseudo_ldr)
        la.optimize_prior_precision(method='marglik')

        post_var = la.posterior_variance
        D_feat = base.fc.in_features
        K_out = base.fc.out_features
        var_W = post_var.reshape(K_out, D_feat)
        var_per_feat = var_W.mean(0)

        feat_extractor = nn.Sequential(
            *list(base.children())[:-1], nn.Flatten()
        ).to(device).eval()

        all_mu = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(device)
                all_mu.append(feat_extractor(x).cpu())

        mu = torch.cat(all_mu, dim=0)
        sigma = var_per_feat.sqrt().unsqueeze(0).expand_as(mu)

        return self._pack(mu, sigma, D_feat)


class DeepEnsembleEstimator(EpsEstimator):
    def __init__(self, seeds: Optional[List[int]] = None):
        self.seeds = seeds or ENSEMBLE_SEEDS

    def estimate(self, loader: DataLoader, device: str) -> dict:
        all_passes = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            base = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
            )
            nn.init.kaiming_normal_(base.fc.weight, nonlinearity='relu')
            nn.init.zeros_(base.fc.bias)
            extractor = nn.Sequential(
                *list(base.children())[:-1], nn.Flatten()
            ).to(device).eval()

            member_feats = []
            with torch.no_grad():
                for batch in loader:
                    x = batch.to(device)
                    member_feats.append(extractor(x).cpu())
            all_passes.append(torch.cat(member_feats, dim=0))

        passes = torch.stack(all_passes, dim=1)
        mu = passes.mean(1)
        sigma = passes.std(1)
        D = mu.shape[1]
        return self._pack(mu, sigma, D)


def compute_certificate_mps(
    source_feats: dict,
    target_feats: dict,
    K: int,
    head: Optional[nn.Linear] = None,
) -> dict:
    mu_S = source_feats['mu'].mean(0)
    mu_T = target_feats['mu'].mean(0)

    mmd = (mu_S - mu_T).norm().item()
    eps_S = source_feats['eps'].mean().item()
    eps_T = target_feats['eps'].mean().item()
    eps = max(eps_S, eps_T)
    cert = (1.0 - eps) * mmd
    mmi = target_feats['mmi'].mean().item()

    out = {
        'mmd': mmd, 'eps_S': eps_S, 'eps_T': eps_T,
        'eps': eps, 'cert': cert, 'mmi': mmi,
    }

    if head is not None:
        W = head.weight.detach().float().cpu()
        B = torch.linalg.matrix_norm(W, ord=2).item()
        full_cert = B * cert
        non_vacuous = full_cert < math.log(K)
        out.update({
            'B': B, 'full_cert': full_cert,
            'log_K': math.log(K), 'non_vacuous': non_vacuous,
        })

    return out


def run_e1_mps(
    dataset: str,
    data_root: Path,
    estimators: Dict[str, EpsEstimator],
    device: str,
    n_per_domain: int = 400,
    batch: int = 64,
    seed: int = 42,
) -> dict:
    domains = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    K = N_CLASSES_MPS.get(dataset, 10)
    gt = GT_ACC_MPS.get(dataset, {})
    _, head = build_frozen_resnet18_mps()

    results = {name: {} for name in estimators}
    cache = {}

    for est_name, estimator in estimators.items():
        cache[est_name] = {}

        for dom in domains:
            ds = DomainDatasetMPS(data_root / dom, n=n_per_domain, seed=seed)
            ldr = _mps_loader(ds, batch)
            feats = estimator.estimate(ldr, device)
            cache[est_name][dom] = feats

        for target in domains:
            sources = [d for d in domains if d != target]
            pool = {
                'mu': torch.cat([cache[est_name][d]['mu'] for d in sources]),
                'eps': torch.cat([cache[est_name][d]['eps'] for d in sources]),
            }
            cert = compute_certificate_mps(pool, cache[est_name][target], K=K, head=head)
            cert['acc'] = gt.get(target, None)
            results[est_name][target] = cert

    return results


def _mps_loader(ds: Dataset, batch: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)


def run_e3_mps(
    data_root: Path,
    target_domain: str,
    device: str,
    n: int = 400,
    batch: int = 64,
    seed: int = 42,
) -> dict:
    domains = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    sources = [d for d in domains if d != target_domain]
    gt = GT_ACC_MPS.get('pacs', {})
    K = N_CLASSES_MPS['pacs']

    class CombinedDS(Dataset):
        def __init__(self, dirs: List[Path]):
            self.samples = []
            classes = set()
            for d in dirs:
                for cls_dir in sorted(d.iterdir()):
                    if cls_dir.is_dir():
                        classes.add(cls_dir.name)
            self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
            for d in dirs:
                for cls_dir in sorted(d.iterdir()):
                    if not cls_dir.is_dir():
                        continue
                    label = self.class_to_idx[cls_dir.name]
                    for img_path in cls_dir.rglob('*'):
                        if img_path.suffix in _IMG_EXTS_MPS:
                            self.samples.append((img_path, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            path, label = self.samples[i]
            return transform_eval_mps(PILImage.open(path).convert('RGB')), label

    source_ds = CombinedDS([data_root / d for d in sources])
    source_ldr = DataLoader(
        source_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False,
    )

    target_ds = DomainDatasetMPS(data_root / target_domain, n=n, seed=seed)
    target_ldr = _mps_loader(target_ds, batch)

    grid = {}

    for epoch, p in product(E3_HEAD_EPOCHS, E3_DROPOUT_P):
        base = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1
        ).to(device)
        for name, param in base.named_parameters():
            if 'fc' not in name:
                param.requires_grad_(False)

        n_classes = len(source_ds.class_to_idx)
        base.fc = nn.Linear(512, n_classes).to(device)

        opt = torch.optim.SGD(base.fc.parameters(), lr=1e-3, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        base.train()
        for _ in range(epoch):
            for xb, yb in source_ldr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(base(xb), yb).backward()
                opt.step()

        extractor = nn.Sequential(
            *list(base.children())[:-1], nn.Flatten(),
            nn.Dropout(p=p),
        ).to(device)

        domain_feats = {}
        for dom in domains:
            ds = DomainDatasetMPS(data_root / dom, n=n, seed=seed)
            ldr = _mps_loader(ds, batch)
            _enable_dropout_only_mps(extractor)

            all_mu, all_sigma = [], []
            with torch.no_grad():
                for xb in ldr:
                    xb = xb.to(device)
                    passes = torch.stack([extractor(xb) for _ in range(5)], dim=1)
                    all_mu.append(passes.mean(1).cpu())
                    all_sigma.append(passes.std(1).cpu())

            mu = torch.cat(all_mu)
            sigma = torch.cat(all_sigma)
            D = mu.shape[1]
            eps = (sigma.pow(2).mean(1).sqrt() / math.sqrt(D)).clamp(1e-6, 1 - 1e-6)
            mmi = 2.0 * sigma.max(1).values
            domain_feats[dom] = {'mu': mu, 'sigma': sigma, 'eps': eps, 'mmi': mmi}

        pool = {
            'mu': torch.cat([domain_feats[d]['mu'] for d in sources]),
            'eps': torch.cat([domain_feats[d]['eps'] for d in sources]),
        }
        cert = compute_certificate_mps(
            pool, domain_feats[target_domain],
            K=K, head=base.fc,
        )

        grid.setdefault(epoch, {})[p] = cert

    return grid


def run_e4_mps(
    dataset: str,
    data_root: Path,
    estimator: EpsEstimator,
    device: str,
    n_per_domain: int = 400,
    batch: int = 64,
    seed: int = 42,
) -> dict:
    if dataset == 'domainnet':
        return run_e1_mps(
            dataset=dataset,
            data_root=data_root,
            estimators={'dropout': estimator},
            device=device,
            n_per_domain=n_per_domain,
            batch=batch,
            seed=seed,
        )

    if dataset == 'iwildcam':
        try:
            from wilds import get_dataset
            get_dataset('iwildcam', root_dir=str(data_root), download=False)
        except ImportError as e:
            raise ImportError('pip install wilds') from e
        raise NotImplementedError(
            'Use WILDS pipeline: python credal_dg.py wilds --dataset iwildcam --data_root ...'
        )

    raise ValueError(f'Unknown dataset for E4: {dataset}')


DEFAULT_MPS_ROOTS = {
    'pacs':       Path('./data/pacs_data'),
    'officehome': Path('./data/OfficeHome'),
    'cifar10c':   Path('./data/CIFAR-10-C'),
    'domainnet':  Path('./data/DomainNet'),
    'iwildcam':   Path('./data/wilds'),
}


def build_estimators_mps(choice: str) -> Dict[str, EpsEstimator]:
    if choice == 'dropout':
        return {'dropout': MCDropoutEstimator(H=5, p=0.15)}
    if choice == 'laplace':
        return {'laplace': LaplaceEstimatorMPS()}
    if choice == 'ensemble':
        return {'ensemble': DeepEnsembleEstimator(seeds=ENSEMBLE_SEEDS)}
    if choice == 'all':
        return {
            'dropout': MCDropoutEstimator(H=5, p=0.15),
            'laplace': LaplaceEstimatorMPS(),
            'ensemble': DeepEnsembleEstimator(seeds=ENSEMBLE_SEEDS),
        }
    raise ValueError(choice)


def parse_args_mps():
    p = argparse.ArgumentParser(description='Unified credal DG / Kernel IIPM experiments')
    p.add_argument('--exp', choices=['e1', 'e3', 'e4'], default='e1')
    p.add_argument(
        '--dataset', default='pacs',
        choices=['pacs', 'officehome', 'cifar10c', 'domainnet', 'iwildcam'],
    )
    p.add_argument('--data_root', type=Path, default=None,
                   help='Root with one subdir per domain (ImageFolder layout)')
    p.add_argument('--estimator', default='dropout',
                   choices=['dropout', 'laplace', 'ensemble', 'all'])
    p.add_argument('--device', default='auto',
                   choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--n', type=int, default=400, help='Instances per domain')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', default='sketch', help='Target domain for E3')
    p.add_argument('--out_dir', type=Path, default=Path('results'))
    return p.parse_args()


def _jsonify_mps(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify_mps(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_mps(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


def main_mps():
    args = parse_args_mps()
    data_root = args.data_root or DEFAULT_MPS_ROOTS[args.dataset]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    if args.exp == 'e1':
        estimators = build_estimators_mps(args.estimator)
        results = run_e1_mps(
            dataset=args.dataset,
            data_root=data_root,
            estimators=estimators,
            device=device,
            n_per_domain=args.n,
            batch=args.batch,
            seed=args.seed,
        )
        tag = f'e1_{args.dataset}_{args.estimator}'

    elif args.exp == 'e3':
        results = run_e3_mps(
            data_root=data_root,
            target_domain=args.target,
            device=device,
            n=args.n,
            batch=args.batch,
            seed=args.seed,
        )
        tag = f'e3_{args.dataset}_target-{args.target}'

    elif args.exp == 'e4':
        est = build_estimators_mps(args.estimator)
        estimator = list(est.values())[0]
        results = run_e4_mps(
            dataset=args.dataset,
            data_root=data_root,
            estimator=estimator,
            device=device,
            n_per_domain=args.n,
            batch=args.batch,
            seed=args.seed,
        )
        tag = f'e4_{args.dataset}_{args.estimator}'

    else:
        raise ValueError(args.exp)

    out_path = args.out_dir / f'{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(_jsonify_mps(results), f, indent=2)


# =============================================================================
# WILDS pipeline  (wilds_credal_dg — NeurIPS 2026)
#   python credal_dg.py wilds --dataset iwildcam --data_root /Users/.../data/wilds
# =============================================================================

GT_ACC_WILDS = {
    'camelyon17': {
        0: 93.2,
        1: 85.6,
        2: 91.4,
        3: 70.3,
        4: 88.9,
    },
    'fmow': {
        'Africa':   32.3,
        'Americas': 48.7,
        'Oceania':  51.2,
        'Asia':     55.6,
        'Europe':   59.1,
    },
    'iwildcam': {},
}

N_CLASSES_WILDS = {
    'iwildcam':   182,
    'camelyon17': 2,
    'fmow':       62,
}

DOMAIN_COL_WILDS = {
    'iwildcam':   0,
    'camelyon17': 0,
    'fmow':       0,
}

FMOW_REGION_NAMES = {
    0: 'Africa', 1: 'Americas', 2: 'Oceania', 3: 'Asia', 4: 'Europe',
}

transform_eval_wilds = TVT.Compose([
    TVT.Resize(256),
    TVT.CenterCrop(224),
    TVT.ToTensor(),
    TVT.Normalize(_MPS_MEAN, _MPS_STD),
])


def build_backbone_wilds(n_classes: int, device: str) -> Tuple[nn.Module, nn.Linear]:
    base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    head = nn.Linear(512, n_classes)
    nn.init.kaiming_normal_(head.weight)
    nn.init.zeros_(head.bias)
    for p in extractor.parameters():
        p.requires_grad_(False)
    extractor = extractor.to(device).eval()
    head = head.to(device).eval()
    return extractor, head


def _enable_dropout_wilds(model: nn.Module, p: float) -> nn.Module:
    net = nn.Sequential(model, nn.Dropout(p=p))
    net.train()
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    return net


def wilds_collate_fn(batch):
    xs = torch.stack([item[0] for item in batch])
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return xs, ys


class WILDSDomainSplitter:
    def __init__(
        self,
        wilds_ds,
        domain_col: int,
        split: str = 'test',
        n_per_domain: int = 200,
        batch: int = 64,
        seed: int = 42,
    ):
        self.wilds_ds = wilds_ds
        self.domain_col = domain_col
        self.n_per_domain = n_per_domain
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        self.split_ds = wilds_ds.get_subset(split, transform=transform_eval_wilds)
        meta = self.split_ds.metadata_array
        if hasattr(meta, 'numpy'):
            self.domain_ids = meta[:, domain_col].numpy()
        else:
            self.domain_ids = np.asarray(meta[:, domain_col])
        self.unique_domains = np.unique(self.domain_ids)

    def get_domain_loader(self, domain_id) -> DataLoader:
        did = int(domain_id)
        idx = np.where(self.domain_ids == did)[0]
        if len(idx) > self.n_per_domain:
            idx = self.rng.choice(idx, self.n_per_domain, replace=False)
        subset = Subset(self.split_ds, idx.tolist())
        return DataLoader(
            subset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=wilds_collate_fn,
        )

    def n_instances(self, domain_id) -> int:
        did = int(domain_id)
        return int((self.domain_ids == did).sum())


@torch.no_grad()
def extract_domain_feats_wilds(
    loader: DataLoader,
    extractor: nn.Module,
    dropout_p: float = 0.15,
    H: int = 5,
    device: str = 'cpu',
) -> dict:
    net = _enable_dropout_wilds(extractor, p=dropout_p)
    D = 512
    all_mu, all_sigma = [], []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        passes = torch.stack([net(x) for _ in range(H)], dim=1)
        all_mu.append(passes.mean(1).cpu())
        all_sigma.append(passes.std(1).cpu())
    mu = torch.cat(all_mu)
    sigma = torch.cat(all_sigma)
    eps = (sigma.pow(2).mean(1).sqrt() / math.sqrt(D)).clamp(1e-6, 1 - 1e-6)
    mmi = 2.0 * sigma.max(1).values
    return {'mu': mu, 'sigma': sigma, 'eps': eps, 'mmi': mmi}


def compute_certificate_wilds(
    source_pool: dict,
    target_feats: dict,
    K: int,
    head: nn.Linear,
) -> dict:
    mu_S = source_pool['mu'].mean(0)
    mu_T = target_feats['mu'].mean(0)
    mmd = (mu_S - mu_T).norm().item()
    eps_S = source_pool['eps'].mean().item()
    eps_T = target_feats['eps'].mean().item()
    eps = max(eps_S, eps_T)
    cert = (1.0 - eps) * mmd
    mmi = target_feats['mmi'].mean().item()
    W = head.weight.detach().float().cpu()
    B = torch.linalg.matrix_norm(W, ord=2).item()
    full = B * cert
    return {
        'mmd': mmd, 'eps_S': eps_S, 'eps_T': eps_T, 'eps': eps,
        'cert': cert, 'mmi': mmi,
        'B': B, 'full_cert': full,
        'log_K': math.log(K),
        'non_vacuous': full < math.log(K),
    }


def compute_gt_from_val_wilds(
    wilds_ds,
    extractor: nn.Module,
    head: nn.Linear,
    domain_col: int,
    n_per_domain: int,
    batch: int,
    device: str,
    top_k_domains: int = 20,
) -> dict:
    splitter = WILDSDomainSplitter(
        wilds_ds, domain_col, split='val',
        n_per_domain=n_per_domain, batch=batch,
    )
    domain_sizes = {d: splitter.n_instances(d) for d in splitter.unique_domains}
    top_domains = sorted(domain_sizes, key=domain_sizes.get, reverse=True)[:top_k_domains]

    full_head = nn.Sequential(extractor, head).to(device).eval()
    gt = {}
    for dom in top_domains:
        ldr = splitter.get_domain_loader(dom)
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in ldr:
                xb = xb.to(device)
                pred = full_head(xb).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += len(yb)
        if total > 0:
            gt[int(dom)] = 100.0 * correct / total
        else:
            gt[int(dom)] = 0.0
    return gt


def run_wilds(
    dataset: str,
    data_root: Path,
    device: str = 'cpu',
    H: int = 5,
    dropout_p: float = 0.15,
    n_per_domain: int = 200,
    batch: int = 64,
    seed: int = 42,
    max_domains: int = 30,
    out_dir: Path = Path('results'),
) -> dict:
    from wilds import get_dataset

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    K = N_CLASSES_WILDS[dataset]
    domain_col = DOMAIN_COL_WILDS[dataset]

    wilds_ds = get_dataset(dataset, root_dir=str(data_root), download=False)
    extractor, head = build_backbone_wilds(K, device)

    gt = {**GT_ACC_WILDS}
    if dataset == 'iwildcam':
        gt['iwildcam'] = compute_gt_from_val_wilds(
            wilds_ds, extractor, head, domain_col,
            n_per_domain=n_per_domain, batch=batch, device=device,
            top_k_domains=max_domains,
        )
    gt_acc = gt[dataset]

    test_splitter = WILDSDomainSplitter(
        wilds_ds, domain_col, split='test',
        n_per_domain=n_per_domain, batch=batch, seed=seed,
    )
    train_splitter = WILDSDomainSplitter(
        wilds_ds, domain_col, split='train',
        n_per_domain=n_per_domain, batch=batch, seed=seed,
    )

    if dataset == 'iwildcam':
        eval_domains = [
            d for d in test_splitter.unique_domains
            if int(d) in gt_acc
        ][:max_domains]
    elif dataset == 'fmow':
        eval_domains = list(range(5))
    else:
        eval_domains = list(range(5))

    source_mus, source_eps = [], []
    train_dom_list = train_splitter.unique_domains
    if len(train_dom_list) > max_domains:
        train_dom_list = train_dom_list[:max_domains]
    for dom in train_dom_list:
        ldr = train_splitter.get_domain_loader(dom)
        feats = extract_domain_feats_wilds(ldr, extractor, dropout_p, H, device)
        source_mus.append(feats['mu'])
        source_eps.append(feats['eps'])

    source_pool = {
        'mu': torch.cat(source_mus),
        'eps': torch.cat(source_eps),
    }

    results = {}
    all_accs, all_mmds, all_certs, all_mmis = [], [], [], []
    domain_feats_pt = {'source_pool': source_pool, 'targets': {}}

    for dom in eval_domains:
        dom_key = FMOW_REGION_NAMES.get(dom, int(dom)) if dataset == 'fmow' else int(dom)
        acc = gt_acc.get(dom_key, gt_acc.get(int(dom), None))
        if acc is None:
            continue

        ldr = test_splitter.get_domain_loader(dom)
        feats = extract_domain_feats_wilds(ldr, extractor, dropout_p, H, device)
        domain_feats_pt['targets'][str(dom_key)] = feats

        cert = compute_certificate_wilds(source_pool, feats, K=K, head=head)
        cert['acc'] = acc

        results[str(dom_key)] = cert
        all_accs.append(acc)
        all_mmds.append(cert['mmd'])
        all_certs.append(cert['cert'])
        all_mmis.append(cert['mmi'])

    if len(all_accs) >= 3:
        rho_mmd, _ = spearmanr(all_accs, all_mmds)
        rho_cert, _ = spearmanr(all_accs, all_certs)
        rho_mmi, _ = spearmanr(all_accs, all_mmis)
        results['__rho__'] = {
            'mmd': float(rho_mmd),
            'cert': float(rho_cert),
            'mmi': float(rho_mmi),
            'n_domains': len(all_accs),
        }

    def _jsonify_wilds(obj):
        if isinstance(obj, torch.Tensor):
            return float(obj) if obj.ndim == 0 else obj.tolist()
        if isinstance(obj, np.ndarray):
            return float(obj) if obj.ndim == 0 else obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify_wilds(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify_wilds(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (bool, str, int, float)) or obj is None:
            return obj
        return str(obj)

    out_json = out_dir / f'{dataset}_wilds.json'
    with open(out_json, 'w') as f:
        json.dump(_jsonify_wilds(results), f, indent=2)

    pt_path = out_dir / f'domain_feats_{dataset}.pt'
    torch.save(domain_feats_pt, pt_path)

    return results


def parse_args_wilds():
    p = argparse.ArgumentParser(description='WILDS credal DG (MC dropout ε, certificates)')
    p.add_argument(
        '--dataset', default='camelyon17',
        choices=['iwildcam', 'camelyon17', 'fmow'],
    )
    p.add_argument(
        '--data_root', type=Path, default=Path('./data/wilds'),
        help='WILDS root_dir (download target). Example: /Users/you/research/data/wilds',
    )
    p.add_argument(
        '--device', default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
    )
    p.add_argument('--H', type=int, default=5)
    p.add_argument('--dropout_p', type=float, default=0.15)
    p.add_argument('--n_per_domain', type=int, default=200)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--max_domains', type=int, default=30,
        help='Cap train domains pooled; iWildCam test domains use GT overlap',
    )
    p.add_argument('--out_dir', type=Path, default=Path('results'))
    return p.parse_args()


def main_wilds():
    args = parse_args_wilds()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    device = pick_device(args.device)
    run_wilds(
        dataset=args.dataset,
        data_root=data_root,
        device=device,
        H=args.H,
        dropout_p=args.dropout_p,
        n_per_domain=args.n_per_domain,
        batch=args.batch,
        seed=args.seed,
        max_domains=args.max_domains,
        out_dir=out_dir,
    )


def pick_device(requested: str) -> str:
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

    # ── Load samples ──────────────────────────────────────────────────────
    domain_samples = {}
    for domain in DOMAINS:
        label_file = label_root / f'{domain}_{args.split}_kfold.txt'
        samples = parse_label_file(str(label_file), str(img_root))
        if args.max_samples and len(samples) > args.max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(samples), size=args.max_samples, replace=False)
            samples = [samples[i] for i in idx]
        domain_samples[domain] = samples

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
    results = run_experiment(
        domain_loaders, model, ERM_ACC_PACS, device, H=args.H
    )

    # ── Figures ───────────────────────────────────────────────────────────
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


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'mps':
        sys.argv.pop(1)
        main_mps()
    elif len(sys.argv) > 1 and sys.argv[1] == 'wilds':
        sys.argv.pop(1)
        main_wilds()
    else:
        main()
