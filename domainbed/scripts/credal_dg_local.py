"""
credal_dg_local.py
==================
NeurIPS 2026 — "Epistemic Uncertainty as a Domain Generalisation Asset"

Mac Mini / MPS version of the full experiment suite.
Produces Tables 1–4 of the paper + appendix ablation tables.

Datasets handled locally (no HuggingFace streaming needed):
  Table 1  PACS        /PACS/pacs_data/          (already on disk)
  Table 2  OfficeHome  ./data/OfficeHome/         (already on disk)
  Table 3  CIFAR-10-C  ./data/CIFAR-10-C/         (already on disk)

For DomainNet → use credal_dg_modal.py::run_gpu1 (too large for Mac Mini).

Outputs (all in ./results/):
  table1_pacs.csv / .tex
  table2_officehome.csv / .tex
  table3_cifar10c.csv / .tex
  ablation_H.csv / .tex        (H ∈ {1,5,10,20} on OfficeHome)
  ablation_backbone.csv / .tex (ResNet-18 vs ViT-B/16 on PACS+OfficeHome)
  certificate_pacs.csv         (B·(1-ε)·MMD vs accuracy gap, non-vacuousness)

Usage:
  python credal_dg_local.py --smoke            # tiny check, 2 domains
  python credal_dg_local.py --tables 1 2 3     # specific tables only
  python credal_dg_local.py --ablation H       # H sweep only
  python credal_dg_local.py                    # everything

  # With MPS acceleration:
  python credal_dg_local.py --mps

Critical invariants (DO NOT CHANGE):
  ε = sqrt(mean_d σ²_d)    — MEAN not SUM over feature dimensions
  H = 5 dropout passes      — primary configuration
  p = 0.15 dropout rate
  head_epochs ≤ 15          — overfitting destroys differential ε signal
  BN layers always in eval mode during MC extraction
"""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mps",        action="store_true")
parser.add_argument("--smoke",      action="store_true",
                    help="Tiny run: 2 domains, 50 imgs, 2 dropout passes")
parser.add_argument("--tables",     nargs="+", type=int, default=[1, 2, 3],
                    choices=[1, 2, 3],
                    help="Which tables to produce (1=PACS, 2=OfficeHome, 3=CIFAR-10-C)")
parser.add_argument("--ablation",   nargs="+", default=[],
                    choices=["H", "backbone"],
                    help="Ablation tables to run")
parser.add_argument("--pacs_root",  type=str, default="/PACS/pacs_data")
parser.add_argument("--oh_root",    type=str, default="./data/OfficeHome")
parser.add_argument("--c10c_root",  type=str, default="./data/CIFAR-10-C")
parser.add_argument("--out",        type=str, default="./results")
parser.add_argument("--H",          type=int, default=5)
parser.add_argument("--drop",       type=float, default=0.15)
parser.add_argument("--head_epochs",type=int, default=10)
parser.add_argument("--n_per_domain",type=int, default=400)
parser.add_argument("--batch",      type=int, default=64)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

if args.smoke:
    args.H            = 2
    args.n_per_domain = 50
    args.head_epochs  = 2

DEVICE = (torch.device("mps")
          if args.mps and torch.backends.mps.is_available()
          else torch.device("cpu"))
OUT    = Path(args.out)
OUT.mkdir(exist_ok=True)
torch.manual_seed(args.seed)

D = 512   # ResNet-18 penultimate dim

print(f"\n{'='*60}")
print(f"  credal_dg_local.py — NeurIPS 2026 Credal Discount")
print(f"  device={DEVICE}  H={args.H}  drop={args.drop}  "
      f"head_epochs={args.head_epochs}")
print(f"  tables={args.tables}  ablations={args.ablation}")
print(f"{'='*60}\n")

# Ground truth accuracies (DomainBed ERM ResNet-50, training-domain val)
GT_ACC = {
    # PACS
    "art_painting": 84.7, "cartoon": 80.8,
    "photo": 97.2, "sketch": 79.3,
    # OfficeHome
    "Art": 61.3, "Clipart": 52.4,
    "Product": 75.8, "Real_World": 76.6,
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATASETS
# ──────────────────────────────────────────────────────────────────────────────

TF_TRAIN = T.Compose([
    T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
TF_EVAL = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class FolderDomainDataset(Dataset):
    """
    ImageFolder-style dataset where each domain is a subdirectory.
    Structure expected:
      root/
        domain_name/
          class_name/
            img.jpg ...

    Works for both PACS and OfficeHome layouts.
    """
    def __init__(self, root, domain, transform, n=None, seed=42):
        domain_dir = Path(root) / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Domain dir not found: {domain_dir}")
        base_ds = torchvision.datasets.ImageFolder(str(domain_dir),
                                                    transform=transform)
        if n is not None and len(base_ds) > n:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(base_ds), n, replace=False)
            self.samples = [base_ds.samples[i] for i in idx]
        else:
            self.samples = base_ds.samples
        self.transform = transform
        self.loader    = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(self.loader(path)), label


def get_domain_loader(root, domain, transform, n, batch, seed):
    ds = FolderDomainDataset(root, domain, transform, n=n, seed=seed)
    return DataLoader(ds, batch_size=batch, shuffle=False,
                      num_workers=0, pin_memory=False)


def discover_domains(root):
    """Return sorted list of subdirectory names (= domain names)."""
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir())


# ──────────────────────────────────────────────────────────────────────────────
# 2. BACKBONE
# ──────────────────────────────────────────────────────────────────────────────

def build_backbone(arch="resnet18", K=7, head_epochs=10,
                   source_loaders=None):
    """
    Frozen backbone + K-class head, optionally fine-tuned on source domains.

    arch: "resnet18" | "vitb16"
    K: number of classes in this dataset
    source_loaders: list of DataLoader for source domains (for fine-tuning)
    """
    if arch == "resnet18":
        base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        for name, p in base.named_parameters():
            if "fc" not in name:
                p.requires_grad_(False)
        base.fc = nn.Linear(D, K)

    elif arch == "vitb16":
        base = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        )
        for name, p in base.named_parameters():
            if "heads" not in name:
                p.requires_grad_(False)
        in_feat = base.heads.head.in_features
        base.heads.head = nn.Linear(in_feat, K)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    base = base.to(DEVICE)

    if source_loaders and head_epochs > 0:
        _finetune(base, source_loaders, head_epochs, arch)

    return base


def _finetune(base, source_loaders, epochs, arch):
    """Fine-tune only the classification head on pooled source domains."""
    params = (base.fc.parameters() if arch == "resnet18"
              else base.heads.parameters())
    opt  = torch.optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=1e-4)
    base.train()

    for ep in range(epochs):
        total_loss = 0.0; n_batch = 0
        for loader in source_loaders:
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(base(xb), yb)
                loss.backward()
                opt.step()
                total_loss += loss.item(); n_batch += 1
        print(f"    ft epoch {ep+1}/{epochs}  "
              f"loss={total_loss/max(n_batch,1):.3f}")


def build_mc_extractor(base, arch, p_drop):
    """
    MC dropout extractor. Returns (N, D) features with stochastic dropout.
    BN layers stay in eval mode to prevent batch statistics drifting.
    """
    if arch == "resnet18":
        extractor = nn.Sequential(
            *list(base.children())[:-1],
            nn.Flatten(),
            nn.Dropout(p=p_drop),
        ).to(DEVICE)
    else:
        # ViT: extract CLS token from encoder output
        extractor = _ViTExtractor(base, p_drop).to(DEVICE)

    extractor.train()
    for m in extractor.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    return extractor


class _ViTExtractor(nn.Module):
    """ViT-B/16 feature extractor returning CLS token + Dropout."""
    def __init__(self, vit, p_drop):
        super().__init__()
        self.vit    = vit
        self.drop   = nn.Dropout(p=p_drop)
        self.d_out  = vit.heads.head.in_features

    def forward(self, x):
        # ViT forward up to pre-logits
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_cls = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_cls, x], dim=1)
        x = self.vit.encoder(x)
        h = x[:, 0]                 # CLS token
        return self.drop(h)


# ──────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION + ε CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_dropout_feats(loader, extractor, H):
    """
    H stochastic forward passes → credal ellipsoid parameters.

    Returns dict:
      mu    (N, D)  ensemble mean
      sigma (N, D)  per-dim std  (diagonal sqrt(Σ_epi))
      eps   (N,)    ε(x) = sqrt(mean_d σ²_d)  [MEAN not SUM]
      mmi   (N,)    MMI(x) = 2·max_d σ_d  [Corollary 5]
    """
    mus, sigs = [], []
    for xb, _ in loader:
        xb     = xb.to(DEVICE)
        passes = torch.stack([extractor(xb) for _ in range(H)], dim=1)
        mus.append(passes.mean(1).cpu())
        sigs.append(passes.std(1).cpu())

    mu    = torch.cat(mus)
    sigma = torch.cat(sigs)

    # CRITICAL: mean over feature dimensions, not sum
    # sum gives ε >> 1 for D=512 — breaks the certificate entirely
    eps = sigma.pow(2).mean(1).sqrt().clamp(1e-6, 1 - 1e-6)
    mmi = 2.0 * sigma.max(1).values

    return {"mu": mu, "sigma": sigma, "eps": eps, "mmi": mmi}


def compute_entropy_baseline(feats):
    """
    Predictive entropy of the ensemble mean as a baseline.
    This is what the paper claims MMI outperforms.
    H(BetP) = -sum_c p_c log p_c  where p = softmax(mu).
    """
    mu = feats["mu"].float()
    p  = torch.softmax(mu, dim=-1)
    return float(-(p * (p + 1e-8).log()).sum(-1).mean())


# ──────────────────────────────────────────────────────────────────────────────
# 4. CERTIFICATE QUANTITIES
# ──────────────────────────────────────────────────────────────────────────────

def compute_B(base, arch):
    """B = σ_max(W) — spectral norm of classification head weight matrix."""
    if arch == "resnet18":
        W = base.fc.weight.detach().float().cpu()
    else:
        W = base.heads.head.weight.detach().float().cpu()
    return torch.linalg.matrix_norm(W, ord=2).item()


def full_certificate(src_feats, tgt_feats, B, K):
    """
    Compute all Q1/Q2/Q3 certificate quantities for one source→target pair.

    Returns dict with:
      mmd         ‖μ_S − μ_T‖
      eps         max(ε_S, ε_T)  — conservative upper bound
      route_a     (1−ε)·MMD      — Route A certificate value
      mmi         2·max_d σ_d on target
      entropy     H(softmax(μ)) on target
      cert        B·(1−ε)·MMD   — full certificate (Theorem 2)
      non_vac     cert < log(K) — non-vacuous?
      log_K       log(K)
    """
    mu_S    = src_feats["mu"].mean(0)
    mu_T    = tgt_feats["mu"].mean(0)
    sigma_T = tgt_feats["sigma"].mean(0)

    eps_S   = src_feats["eps"].mean().item()
    eps_T   = tgt_feats["eps"].mean().item()
    eps     = max(eps_S, eps_T)

    mmd     = (mu_S - mu_T).norm().item()
    route_a = (1.0 - eps) * mmd
    mmi     = 2.0 * sigma_T.max().item()
    entropy = compute_entropy_baseline(tgt_feats)

    log_K   = math.log(K)
    cert    = B * route_a

    return {
        "mmd":      mmd,
        "eps":      eps,
        "route_a":  route_a,
        "mmi":      mmi,
        "entropy":  entropy,
        "B":        B,
        "cert":     cert,
        "log_K":    log_K,
        "non_vac":  cert < log_K,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. LEAVE-ONE-DOMAIN-OUT RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_lodo(root, dataset_name, K, arch="resnet18",
             H=None, drop=None, head_epochs=None,
             n_per_domain=None, smoke=False):
    """
    Leave-One-Domain-Out experiment for PACS or OfficeHome.
    Returns list of per-domain result dicts.
    """
    H            = H            or args.H
    drop         = drop         or args.drop
    head_epochs  = head_epochs  or args.head_epochs
    n_per_domain = n_per_domain or args.n_per_domain

    domains = discover_domains(root)
    if smoke:
        domains = domains[:2]
    print(f"\n  {dataset_name} ({arch})  domains={domains}")

    results = []

    for target in domains:
        print(f"\n  ── target: {target} ──")
        sources = [d for d in domains if d != target]

        # Source loaders (for fine-tuning)
        src_loaders = [
            get_domain_loader(root, d, TF_TRAIN,
                              n_per_domain, args.batch, args.seed)
            for d in sources
        ]

        # Build and fine-tune backbone
        base = build_backbone(arch, K, head_epochs, src_loaders)
        B    = compute_B(base, arch)

        # MC extractor
        ext = build_mc_extractor(base, arch, drop)

        # Extract source pool features (pooled across all sources)
        src_mus, src_sigs, src_eps, src_mmi = [], [], [], []
        for d in sources:
            loader = get_domain_loader(root, d, TF_EVAL,
                                       n_per_domain, args.batch, args.seed)
            f = mc_dropout_feats(loader, ext, H)
            src_mus.append(f["mu"])
            src_sigs.append(f["sigma"])
            src_eps.append(f["eps"])
            src_mmi.append(f["mmi"])

        src_feats = {
            "mu":    torch.cat(src_mus),
            "sigma": torch.cat(src_sigs),
            "eps":   torch.cat(src_eps),
            "mmi":   torch.cat(src_mmi),
        }

        # Extract target features
        tgt_loader = get_domain_loader(root, target, TF_EVAL,
                                       n_per_domain, args.batch, args.seed)
        tgt_feats  = mc_dropout_feats(tgt_loader, ext, H)

        # Compute certificate
        cert = full_certificate(src_feats, tgt_feats, B, K)
        cert["domain"]  = target
        cert["gt_acc"]  = GT_ACC.get(target)
        cert["dataset"] = dataset_name
        cert["arch"]    = arch
        cert["H"]       = H

        _print_domain(target, cert)
        results.append(cert)

    # Spearman ρ
    rhos = _spearman(results)
    for r in results:
        r.update(rhos)

    return results


def _print_domain(name, c):
    nv = "✓" if c["non_vac"] else "✗"
    print(f"    MMD={c['mmd']:.3f}  ε={c['eps']:.3f}  "
          f"(1-ε)·MMD={c['route_a']:.3f}  MMI={c['mmi']:.3f}  "
          f"H(p)={c['entropy']:.3f}")
    print(f"    B={c['B']:.3f}  cert={c['cert']:.3f} {nv}  "
          f"log(K)={c['log_K']:.3f}")


def _spearman(results):
    rows = [r for r in results if r.get("gt_acc") is not None]
    if len(rows) < 3:
        print("  ρ: <3 domains with GT accuracy — skipping")
        return {}

    accs    = [r["gt_acc"]  for r in rows]
    mmds    = [r["mmd"]     for r in rows]
    routes  = [r["route_a"] for r in rows]
    mmis    = [r["mmi"]     for r in rows]
    entrs   = [r["entropy"] for r in rows]

    rho_mmd,   p_mmd   = spearmanr(accs, mmds)
    rho_route, p_route = spearmanr(accs, routes)
    rho_mmi,   p_mmi   = spearmanr(accs, mmis)
    rho_entr,  p_entr  = spearmanr(accs, entrs)

    print(f"\n  ── Spearman ρ (N={len(rows)}) ──")
    print(f"  MMD:       ρ={rho_mmd:+.2f}  p={p_mmd:.3f}")
    print(f"  (1-ε)·MMD: ρ={rho_route:+.2f}  p={p_route:.3f}")
    print(f"  MMI:       ρ={rho_mmi:+.2f}  p={p_mmi:.3f}")
    print(f"  Entropy:   ρ={rho_entr:+.2f}  p={p_entr:.3f}")

    return dict(
        rho_mmd=round(rho_mmd,3), p_mmd=round(p_mmd,3),
        rho_route=round(rho_route,3), p_route=round(p_route,3),
        rho_mmi=round(rho_mmi,3), p_mmi=round(p_mmi,3),
        rho_entr=round(rho_entr,3), p_entr=round(p_entr,3),
        n_domains=len(rows),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. CIFAR-10-C (Table 3) — corruption severity as domain axis
# ──────────────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

ALL_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "shot_noise", "zoom_blur",
]

# Published ResNet-18 mean accuracy over severity 1–5 (RobustBench)
GT_ACC_C10C = {
    "brightness": 91.6, "fog": 86.2, "elastic_transform": 88.4,
    "defocus_blur": 82.5, "snow": 83.0, "jpeg_compression": 84.5,
    "zoom_blur": 79.8, "motion_blur": 80.0, "frost": 81.6,
    "contrast": 76.9, "pixelate": 82.0, "shot_noise": 64.5,
    "impulse_noise": 55.3, "glass_blur": 58.5, "gaussian_noise": 48.7,
}


class CIFAR10CDataset(Dataset):
    """Single corruption + severity slice from CIFAR-10-C numpy files."""
    def __init__(self, root, corruption, severity, transform):
        data   = np.load(Path(root) / f"{corruption}.npy")
        labels = np.load(Path(root) / "labels.npy")
        idx    = slice((severity - 1) * 10000, severity * 10000)
        data   = data[idx].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        self.data      = torch.from_numpy(data)
        self.labels    = torch.from_numpy(labels[idx].astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.data[i]
        # Normalise inline (transform already applied channel-wise)
        mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
        std  = torch.tensor(CIFAR10_STD).view(3,1,1)
        x    = (x - mean) / std
        return F.interpolate(x.unsqueeze(0), size=224,
                             mode="bilinear", align_corners=False).squeeze(0), \
               self.labels[i]


def run_cifar10c(root, arch="resnet18", smoke=False):
    """
    For each corruption type, treat severity 3 as the 'domain'.
    Source = clean CIFAR-10 test set.
    Compute full certificate; ρ with published accuracy across corruptions.

    Returns list of per-corruption result dicts.
    """
    print(f"\n  CIFAR-10-C ({arch})")

    # Clean CIFAR-10 test as source
    tf_c10 = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    clean_ds  = torchvision.datasets.CIFAR10(
        root=str(Path(root).parent), train=False, download=True,
        transform=tf_c10
    )
    if args.smoke:
        from torch.utils.data import Subset
        clean_ds = Subset(clean_ds, list(range(200)))

    clean_loader = DataLoader(clean_ds, batch_size=args.batch,
                              shuffle=False, num_workers=0)

    # Build backbone (no fine-tuning — using ImageNet pretrained directly)
    base = build_backbone(arch, K=10, head_epochs=0)
    B    = compute_B(base, arch)
    ext  = build_mc_extractor(base, arch, args.drop)

    # Source features from clean CIFAR-10
    src_feats = mc_dropout_feats(clean_loader, ext, args.H)

    corruptions = ALL_CORRUPTIONS[:2] if smoke else ALL_CORRUPTIONS
    results = []

    for corr in corruptions:
        try:
            c10c_ds = CIFAR10CDataset(root, corr, severity=3, transform=None)
            if args.smoke:
                from torch.utils.data import Subset
                c10c_ds = Subset(c10c_ds, list(range(200)))

            tgt_loader = DataLoader(c10c_ds, batch_size=args.batch,
                                    shuffle=False, num_workers=0)
            tgt_feats  = mc_dropout_feats(tgt_loader, ext, args.H)

            cert = full_certificate(src_feats, tgt_feats, B, K=10)
            cert["domain"]  = corr
            cert["gt_acc"]  = GT_ACC_C10C.get(corr)
            cert["dataset"] = "cifar10c"
            cert["arch"]    = arch

            _print_domain(corr, cert)
            results.append(cert)

        except FileNotFoundError:
            print(f"  [SKIP] {corr}.npy not found in {root}")

    rhos = _spearman(results)
    for r in results:
        r.update(rhos)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 7. H ABLATION
# ──────────────────────────────────────────────────────────────────────────────

def run_H_ablation(root="officehome", H_values=None):
    """
    H ∈ {1, 5, 10, 20} on OfficeHome.
    Shows H=5 is sufficient; diminishing returns beyond.
    Returns list of dicts with H, domain, rho values.
    """
    H_values = H_values or [1, 5, 10, 20]
    root     = args.oh_root
    print(f"\n  H ablation on OfficeHome: {H_values}")

    all_rows = []
    for H in H_values:
        print(f"\n  H={H}")
        res = run_lodo(root, "officehome", K=65, arch="resnet18",
                       H=H, smoke=args.smoke)
        for r in res:
            all_rows.append({
                "H":        H,
                "domain":   r["domain"],
                "mmi":      r["mmi"],
                "route_a":  r["route_a"],
                "rho_mmi":  r.get("rho_mmi", ""),
                "rho_route":r.get("rho_route",""),
            })
    return all_rows


# ──────────────────────────────────────────────────────────────────────────────
# 8. CSV + LATEX WRITERS
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(rows, name):
    path = OUT / f"{name}.csv"
    if not rows:
        print(f"  [SKIP] {name} — no rows"); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


def save_latex_lodo(rows, name, caption):
    """
    LaTeX table for PACS / OfficeHome / CIFAR-10-C leave-one-domain-out.
    Columns: Domain | GT Acc | MMD | Entropy | (1-ε)·MMD | MMI
    Bold the best predictor (highest |ρ|) column header.
    """
    if not rows: return

    rhos = {
        "MMD":      abs(rows[0].get("rho_mmd",   0) or 0),
        "Entropy":  abs(rows[0].get("rho_entr",  0) or 0),
        "(1-ε)·MMD":abs(rows[0].get("rho_route", 0) or 0),
        "MMI":      abs(rows[0].get("rho_mmi",   0) or 0),
    }
    best = max(rhos, key=rhos.get)

    def hdr(col):
        return rf"\textbf{{{col}}}" if col == best else col

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{name}}}",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        rf"Domain & GT Acc & {hdr('MMD')} & {hdr('Entropy')} "
        rf"& {hdr('(1-ε)·MMD')} & {hdr('MMI')} \\",
        r"\midrule",
    ]

    for r in rows:
        acc = f"{r['gt_acc']:.1f}" if r.get("gt_acc") is not None else "—"
        lines.append(
            f"{r['domain']} & {acc} & {r['mmd']:.3f} & "
            f"{r['entropy']:.3f} & {r['route_a']:.3f} & {r['mmi']:.3f} \\\\"
        )

    # ρ row
    rho_vals = [
        rows[0].get("rho_mmd",   ""),
        rows[0].get("rho_entr",  ""),
        rows[0].get("rho_route", ""),
        rows[0].get("rho_mmi",   ""),
    ]
    numeric_rhos = [x for x in rho_vals if isinstance(x, (int, float))]
    max_abs = max((abs(x) for x in numeric_rhos), default=0.0)

    def fmt_rho(v):
        if v == "" or v is None:
            return "—"
        if isinstance(v, (int, float)) and numeric_rhos and abs(v) == max_abs:
            return rf"\textbf{{{v:+.2f}}}"
        if isinstance(v, (int, float)):
            return f"{v:+.2f}"
        return str(v)

    lines += [
        r"\midrule",
        rf"Spearman $\rho$ & — & {fmt_rho(rho_vals[0])} & "
        rf"{fmt_rho(rho_vals[1])} & {fmt_rho(rho_vals[2])} & "
        rf"{fmt_rho(rho_vals[3])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = OUT / f"{name}.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    all_results = {}

    # ── Table 1: PACS ────────────────────────────────────────────
    if 1 in args.tables:
        print("\n" + "─"*50)
        print("TABLE 1 — PACS")
        rows = run_lodo(args.pacs_root, "pacs", K=7,
                        arch="resnet18", smoke=args.smoke)
        save_csv(rows, "table1_pacs")
        save_latex_lodo(rows, "table1_pacs",
            "PACS leave-one-domain-out. Spearman $\\rho$ with held-out accuracy "
            "(DomainBed ERM ResNet-50). ResNet-18 backbone, $H=5$ dropout passes, "
            "no DG training. \\textbf{Bold} = best predictor.")
        all_results["pacs"] = rows

    # ── Table 2: OfficeHome ───────────────────────────────────────
    if 2 in args.tables:
        print("\n" + "─"*50)
        print("TABLE 2 — OfficeHome")
        rows = run_lodo(args.oh_root, "officehome", K=65,
                        arch="resnet18", smoke=args.smoke)
        save_csv(rows, "table2_officehome")
        save_latex_lodo(rows, "table2_officehome",
            "OfficeHome leave-one-domain-out. Same protocol as Table~\\ref{tab:table1_pacs}.")
        all_results["officehome"] = rows

    # ── Table 3: CIFAR-10-C ───────────────────────────────────────
    if 3 in args.tables:
        print("\n" + "─"*50)
        print("TABLE 3 — CIFAR-10-C (severity 3, all corruptions)")
        rows = run_cifar10c(args.c10c_root, arch="resnet18",
                            smoke=args.smoke)
        save_csv(rows, "table3_cifar10c")
        save_latex_lodo(rows, "table3_cifar10c",
            "CIFAR-10-C: clean CIFAR-10 as source, each corruption type "
            "(severity 3) as target. $\\rho$ with published ResNet-18 accuracy "
            "across corruption types.")
        all_results["cifar10c"] = rows

    # ── Ablation: H sweep ─────────────────────────────────────────
    if "H" in args.ablation:
        print("\n" + "─"*50)
        print("ABLATION — H sweep on OfficeHome")
        rows = run_H_ablation()
        save_csv(rows, "ablation_H")

    # ── Ablation: backbone ────────────────────────────────────────
    if "backbone" in args.ablation:
        print("\n" + "─"*50)
        print("ABLATION — ViT-B/16 on PACS + OfficeHome")
        for dset, root, K in [("pacs", args.pacs_root, 7),
                               ("officehome", args.oh_root, 65)]:
            rows = run_lodo(root, dset, K=K, arch="vitb16",
                            smoke=args.smoke)
            save_csv(rows, f"ablation_vitb16_{dset}")
            save_latex_lodo(rows, f"ablation_vitb16_{dset}",
                f"{dset} ViT-B/16 backbone ablation.")

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPLETE")
    for dset, rows in all_results.items():
        if rows and rows[0].get("rho_mmi") is not None:
            r = rows[0]
            print(f"  {dset:15s}  "
                  f"ρ_MMD={r.get('rho_mmd',0):+.2f}  "
                  f"ρ_(1-ε)·MMD={r.get('rho_route',0):+.2f}  "
                  f"ρ_MMI={r.get('rho_mmi',0):+.2f}  "
                  f"ρ_entropy={r.get('rho_entr',0):+.2f}")
    print(f"  Results in: {OUT}")
    print(f"{'='*60}\n")
