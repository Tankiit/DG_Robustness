import modal
import math
from itertools import product
import torch

# ─────────────────────────────────────────────────────────────────
# MODAL IMAGE
# ─────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "datasets",
        "huggingface_hub",
        "scipy",
        "numpy",
        "Pillow",
    )
)

hf_secret   = modal.Secret.from_name("huggingface")
app         = modal.App("credal-dg-neurips2026", image=image)
results_vol = modal.Volume.from_name("credal-results", create_if_missing=True)
RESULTS     = "/results"


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

H             = 5        # MC dropout passes
DROP          = 0.15     # dropout probability
N_DOM         = 400      # images per domain
BATCH         = 64
SEED          = 42
D             = 512      # ResNet-18 penultimate dim
FINETUNE_EP   = 5        # epochs to fine-tune K-class head on source domains
N_DIRECTIONS  = 1024     # Monte Carlo directions for Hausdorff IIPM

E3_EPOCHS     = [1, 3, 5, 10, 15, 20]
E3_DROPS      = [0.05, 0.10, 0.15, 0.20, 0.30]


# ─────────────────────────────────────────────────────────────────
# DATASET CONFIG
# Confirmed column names from inspect_hf_datasets.py
# ─────────────────────────────────────────────────────────────────

HF = {
    "pacs": {
        "hf_id":        "flwrlabs/pacs",
        "splits":       ["train"],
        "domain_col":   "domain",       # str: art_painting / cartoon / photo / sketch
        "label_col":    "label",
        "n_classes":    7,
        "domain_type":  "str",
        "domain_map":   None,           # no int→str mapping needed
        "gt_acc": {
            "art_painting": 84.7,
            "cartoon":      80.8,
            "photo":        97.2,
            "sketch":       79.3,
        },
    },
    "domainnet": {
        "hf_id":        "wltjr1007/DomainNet",
        "splits":       ["train"],
        "domain_col":   "domain",       # int 0-5
        "label_col":    "label",
        "n_classes":    345,
        "domain_type":  "int",
        "domain_map": {                 # confirmed from image_path inspection
            0: "clipart",
            1: "infograph",
            2: "painting",
            3: "quickdraw",
            4: "real",
            5: "sketch",
        },
        "gt_acc": {                     # DomainBed ERM ResNet-50
            "clipart":   53.3,
            "infograph": 18.8,
            "painting":  45.1,
            "quickdraw": 42.3,
            "real":      58.9,
            "sketch":    47.6,
        },
    },
    "camelyon17": {
        "hf_id":        "wltjr1007/Camelyon17-WILDS",
        # Centers are spread across splits:
        #   train      → centers 0, 3, 4
        #   validation → center  1
        #   test       → center  2
        # Must pool all three to get all 5 centers.
        "splits":       ["train", "validation", "test"],
        "domain_col":   "center",       # confirmed: NOT "hospital"
        "label_col":    "label",
        "n_classes":    2,
        "domain_type":  "int",
        "domain_map": {
            0: "center_0",
            1: "center_1",
            2: "center_2",
            3: "center_3",
            4: "center_4",
        },
        "gt_acc":       {},             # no published per-center numbers
    },
    "officehome": {
        "hf_id":        "GATE-engine/office-home",
        "splits":       ["train"],
        "domain_col":   "domain",       # verify with inspect_hf_datasets.py
        "label_col":    "label",
        "n_classes":    65,
        "domain_type":  "str",
        "domain_map":   None,
        "gt_acc": {                     # DomainBed ERM ResNet-50
            "Art":        61.3,
            "Clipart":    52.4,
            "Product":    75.8,
            "Real_World": 76.6,
        },
    },
}


# ─────────────────────────────────────────────────────────────────
# SECTION 1: DATA HELPERS
# ─────────────────────────────────────────────────────────────────

def get_all_domains(cfg: dict) -> list:
    from datasets import load_dataset
    seen = set()
    for split_name in cfg["splits"]:
        ds = load_dataset(cfg["hf_id"], split=split_name)
        seen.update(ds[cfg["domain_col"]])
    return sorted(seen)


def get_domain_loader(cfg: dict, domain_val, transform,
                      n: int, seed: int, batch: int):
    from datasets import load_dataset, concatenate_datasets
    from torch.utils.data import DataLoader, Dataset

    parts = []
    for split_name in cfg["splits"]:
        ds = load_dataset(cfg["hf_id"], split=split_name)
        if cfg["domain_type"] == "int":
            ds = ds.filter(lambda x: x[cfg["domain_col"]] == int(domain_val))
        else:
            ds = ds.filter(lambda x: x[cfg["domain_col"]] == domain_val)
        if len(ds) > 0:
            parts.append(ds)

    if not parts:
        return None

    combined = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if len(combined) > n:
        combined = combined.shuffle(seed=seed).select(range(n))

    class HFDomainDataset(Dataset):
        def __len__(self):
            return len(combined)
        def __getitem__(self, i):
            r = combined[i]
            return transform(r["image"].convert("RGB")), r[cfg["label_col"]]

    return DataLoader(
        HFDomainDataset(),
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────
# SECTION 2: BACKBONE + FINE-TUNING
# ─────────────────────────────────────────────────────────────────

def build_resnet18_kclass(K: int, device: str):
    import torch.nn as nn
    import torchvision.models as models  # pyright: ignore[reportMissingImports]

    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, p in base.named_parameters():
        if "fc" not in name:
            p.requires_grad_(False)
    base.fc = nn.Linear(D, K)
    return base.to(device)


def finetune_head(cfg: dict, source_domains: list,
                  base, device: str, transform_train, epochs: int):
    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    ds_full = load_dataset(cfg["hf_id"], split=cfg["splits"][0])
    if cfg["domain_type"] == "int":
        sub = ds_full.filter(lambda x: x[cfg["domain_col"]] in source_domains)
    else:
        sub = ds_full.filter(lambda x: x[cfg["domain_col"]] in source_domains)
    sub = sub.shuffle(seed=SEED).select(
        range(min(4000 * len(source_domains), len(sub)))
    )

    class SourceDataset(Dataset):
        def __len__(self):
            return len(sub)
        def __getitem__(self, i):
            r = sub[i]
            return transform_train(r["image"].convert("RGB")), r[cfg["label_col"]]

    loader  = DataLoader(SourceDataset(), batch_size=BATCH,
                         shuffle=True, num_workers=4, pin_memory=True)
    opt     = torch.optim.SGD(base.fc.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    base.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(base(xb), yb)
            loss.backward()
            opt.step()

    return base


def build_mc_extractor(base, device: str, p_drop: float = DROP):
    import torch.nn as nn

    extractor = nn.Sequential(
        *list(base.children())[:-1],  # conv layers + avgpool
        nn.Flatten(),                  # (B, 512)
        nn.Dropout(p=p_drop),
    ).to(device)

    extractor.train()
    for m in extractor.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()          # BN in eval: uses running stats, not batch stats

    return extractor


# ─────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

def mc_dropout_feats(loader, extractor, H: int, device: str) -> dict:
    import torch

    mus, sigs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb     = xb.to(device)
            passes = torch.stack(
                [extractor(xb) for _ in range(H)], dim=1
            )                           # (B, H, D)
            mus.append(passes.mean(1).cpu())
            sigs.append(passes.std(1).cpu())

    mu    = torch.cat(mus)              # (N, D)
    sigma = torch.cat(sigs)            # (N, D)

    # ε(x) = sqrt(mean_d σ²_d(x)) / sqrt(D)
    eps   = (sigma.pow(2).mean(1).sqrt() / math.sqrt(D)).clamp(1e-6, 1 - 1e-6)
    # MMI(x) = 2·max_d σ_d(x)   [Corollary 5, linear kernel]
    mmi   = 2.0 * sigma.max(1).values

    return {"mu": mu, "sigma": sigma, "eps": eps, "mmi": mmi}


def compute_per_instance_loss(loader, base, device: str) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    base.eval()
    losses = []
    with torch.no_grad():
        for xb, _ in loader:
            xb       = xb.to(device)
            logits   = base(xb)                     # (B, K)
            pseudo_y = logits.argmax(1)              # argmax = pseudo-label
            ce       = F.cross_entropy(logits, pseudo_y, reduction="none")
            losses.append(ce.cpu())
    return torch.cat(losses)                         # (N,)


# ─────────────────────────────────────────────────────────────────
# SECTION 4: IIPM + SUPPORT FUNCTION
# Implements the paper's theoretical objects exactly.
# ─────────────────────────────────────────────────────────────────

def support_function(f: "torch.Tensor",
                     mu: "torch.Tensor",
                     sigma: "torch.Tensor") -> float:
    linear_term    = (f * mu).sum()                     # f^T μ
    quadratic_term = (f * sigma).pow(2).sum().sqrt()    # sqrt(f^T Σ f)
    return (linear_term + quadratic_term).item()


def hausdorff_iipm(mu_S: "torch.Tensor", sigma_S: "torch.Tensor",
                   mu_T: "torch.Tensor", sigma_T: "torch.Tensor",
                   n_directions: int = N_DIRECTIONS,
                   seed: int = SEED) -> dict:
    import torch

    D_   = mu_S.shape[0]
    rng  = torch.Generator()
    rng.manual_seed(seed)

    # Sample n_directions random unit vectors on the D-sphere
    F    = torch.randn(n_directions, D_, generator=rng)
    F    = F / F.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (n_dir, D)

    # h_{K_S}(f) = f^T μ_S + sqrt(f^T Σ_S f)   for each direction
    h_S  = (F * mu_S).sum(1) + (F * sigma_S).pow(2).sum(1).sqrt()
    # h_{K_T}(f) analogously
    h_T  = (F * mu_T).sum(1) + (F * sigma_T).pow(2).sum(1).sqrt()

    # Hausdorff = sup |h_S(f) − h_T(f)|
    diffs        = (h_S - h_T).abs()               # (n_dir,)
    iipm_mc      = diffs.max().item()
    best_dir     = F[diffs.argmax()]                # (D,) — widest direction

    # Closed-form upper bound via triangle inequality
    iipm_ub      = (mu_S - mu_T).norm().item() + (sigma_S - sigma_T).norm().item()

    return {
        "iipm_mc":   iipm_mc,     # Monte Carlo lower bound on true IIPM
        "iipm_ub":   iipm_ub,     # closed-form upper bound (triangle ineq.)
        "best_dir":  best_dir,    # direction achieving the MC sup
    }


def mmi_from_sigma(sigma: "torch.Tensor") -> float:
    return 2.0 * sigma.max().item()


def adaptive_gap(eps_per_instance: "torch.Tensor",
                 loss_per_instance: "torch.Tensor",
                 M: float) -> dict:
    eps_max  = eps_per_instance.max().item()
    eps_mean = eps_per_instance.mean().item()
    ell_mean = loss_per_instance.mean().item()
    gap      = max((eps_max - eps_mean) * (M - ell_mean), 0.0)

    return {
        "adaptive_gap": gap,
        "eps_max":      eps_max,
        "eps_mean":     eps_mean,
        "ell_mean":     ell_mean,
    }


def q1_tv_check(src_pool: dict, tgt_feats: dict, eps_hat: float) -> dict:
    import torch

    mu_S  = src_pool["mu"].mean(0)
    mu_T  = tgt_feats["mu"].mean(0)
    sig_S = src_pool["sigma"].mean(0)
    sig_T = tgt_feats["sigma"].mean(0)

    var_S = sig_S.pow(2).clamp(min=1e-8)
    var_T = sig_T.pow(2).clamp(min=1e-8)

    # KL(N_S ‖ N_T) for diagonal Gaussians
    kl    = 0.5 * (
        (var_S / var_T).sum()
        + ((mu_T - mu_S).pow(2) / var_T).sum()
        - D
        + (var_T.log().sum() - var_S.log().sum())
    )
    tv_ub = (kl.clamp(min=0) / 2).sqrt().item()

    return {
        "tv_upper_bound": tv_ub,
        "eps_hat":        eps_hat,
        "q1_covered":     tv_ub <= eps_hat,
    }


# ─────────────────────────────────────────────────────────────────
# SECTION 5: FULL CERTIFICATE
# Combines all theoretical objects into one call.
# ─────────────────────────────────────────────────────────────────

def full_certificate(src_pool: dict,
                     tgt_feats: dict,
                     K: int,
                     head,
                     loader_tgt,
                     base,
                     device: str) -> dict:
    import torch

    # ── domain-level means and stds ───────────────────────────
    mu_S    = src_pool["mu"].mean(0)        # (D,)
    mu_T    = tgt_feats["mu"].mean(0)       # (D,)
    sigma_S = src_pool["sigma"].mean(0)     # (D,)
    sigma_T = tgt_feats["sigma"].mean(0)    # (D,)

    # ── ε scalars ─────────────────────────────────────────────
    eps_S   = src_pool["eps"].mean().item()
    eps_T   = tgt_feats["eps"].mean().item()
    eps     = max(eps_S, eps_T)             # conservative upper bound

    # ── Route A: (1−ε)·MMD  [Corollary 4] ────────────────────
    mmd     = (mu_S - mu_T).norm().item()
    route_a = (1.0 - eps) * mmd

    # ── Exact IIPM via Hausdorff  [Theorem 1] ─────────────────
    hauss      = hausdorff_iipm(mu_S, sigma_S, mu_T, sigma_T)
    iipm_mc    = hauss["iipm_mc"]
    iipm_ub    = hauss["iipm_ub"]

    # Sanity: exact IIPM ≤ Route A  (by construction; flag if violated)
    route_a_dominates = iipm_mc <= route_a + 1e-4

    # ── MMI  [Corollary 5] ────────────────────────────────────
    mmi     = mmi_from_sigma(tgt_feats["sigma"].mean(0))

    # ── B = spectral norm of K-class head ─────────────────────
    B       = torch.linalg.matrix_norm(
                  head.weight.detach().float().cpu(), ord=2
              ).item()
    log_K   = math.log(K)
    M       = log_K                         # sup CE loss for K classes

    # ── Full certificates  [Theorem 2] ────────────────────────
    cert_route_a = B * route_a
    cert_iipm    = B * iipm_mc

    # ── Adaptive gap  [Proposition 3] ─────────────────────────
    loss_inst = compute_per_instance_loss(loader_tgt, base, device)
    gap_info  = adaptive_gap(tgt_feats["eps"], loss_inst, M)

    # ── Q1: TV check ──────────────────────────────────────────
    q1        = q1_tv_check(src_pool, tgt_feats, eps)

    return {
        # --- Q2: discrepancy measures ---
        "mmd":               mmd,
        "eps_S":             eps_S,
        "eps_T":             eps_T,
        "eps":               eps,
        "route_a":           route_a,           # (1−ε)·MMD
        "mmi":               mmi,               # 2·max_d σ_d
        # --- Exact IIPM ---
        "iipm_mc":           iipm_mc,           # Monte Carlo lower bound
        "iipm_ub":           iipm_ub,           # triangle ineq. upper bound
        "route_a_dominates": route_a_dominates, # sanity flag
        # --- Q3: certificates ---
        "B":                 B,
        "log_K":             log_K,
        "cert_route_a":      cert_route_a,      # B·(1−ε)·MMD  (conservative)
        "cert_iipm":         cert_iipm,         # B·IIPM_mc    (tighter)
        "non_vac_route_a":   cert_route_a < log_K,
        "non_vac_iipm":      cert_iipm    < log_K,
        # --- Proposition 3 ---
        "adaptive_gap":      gap_info["adaptive_gap"],
        "eps_max":           gap_info["eps_max"],
        "eps_mean":          gap_info["eps_mean"],
        # --- Q1 ---
        "tv_upper_bound":    q1["tv_upper_bound"],
        "q1_covered":        q1["q1_covered"],
    }


# ─────────────────────────────────────────────────────────────────
# SECTION 6: REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────

def spearman_summary(results: dict, domains: list) -> dict:
    from scipy.stats import spearmanr  # pyright: ignore[reportMissingImports]

    rows = [results[d] for d in domains
            if isinstance(results.get(d), dict)
            and results[d].get("acc") is not None]
    if len(rows) < 3:
        return {}

    accs   = [r["acc"]      for r in rows]
    mmds   = [r["mmd"]      for r in rows]
    routes = [r["route_a"]  for r in rows]
    mmis   = [r["mmi"]      for r in rows]
    iipms  = [r["iipm_mc"]  for r in rows]

    rho_mmd,   p_mmd   = spearmanr(accs, mmds)
    rho_route, p_route = spearmanr(accs, routes)
    rho_mmi,   p_mmi   = spearmanr(accs, mmis)
    rho_iipm,  p_iipm  = spearmanr(accs, iipms)

    return dict(
        rho_mmd=rho_mmd,     p_mmd=p_mmd,
        rho_route=rho_route, p_route=p_route,
        rho_mmi=rho_mmi,     p_mmi=p_mmi,
        rho_iipm=rho_iipm,   p_iipm=p_iipm,
        n_domains=len(rows),
    )


def save(tag: str, data: dict):
    import json
    import os

    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/{tag}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    results_vol.commit()


# ─────────────────────────────────────────────────────────────────
# SECTION 7: MAIN EXPERIMENT FUNCTION
# One function handles any HF-hosted dataset.
# Answers Q1 + Q2 + Q3 in a single pass.
# ─────────────────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=60 * 180,
    volumes={RESULTS: results_vol},
    secrets=[hf_secret],
)
def run_hf(dataset: str):
    import torch
    import torchvision.transforms as T  # pyright: ignore[reportMissingImports]

    cfg    = HF[dataset]
    device = "cuda"
    K      = cfg["n_classes"]
    gt     = cfg.get("gt_acc", {})

    transform_train = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_eval = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_domains = get_all_domains(cfg)

    results = {}

    for target in all_domains:
        sources  = [d for d in all_domains if d != target]
        name     = cfg["domain_map"][target] if cfg["domain_map"] else str(target)

        # ── 1. Fresh K-class backbone for this split ───────────
        base = build_resnet18_kclass(K, device)

        # ── 2. Fine-tune head on source domains ────────────────
        base = finetune_head(cfg, sources, base, device,
                             transform_train, FINETUNE_EP)

        import torch as _torch

        # ── 3. MC dropout extractor ────────────────────────────
        extractor = build_mc_extractor(base, device, DROP)

        # ── 4. Extract features for all domains ────────────────
        feats = {}
        for dom in all_domains:
            ldr = get_domain_loader(cfg, dom, transform_eval,
                                    N_DOM, SEED, BATCH)
            if ldr is None:
                continue
            feats[dom] = mc_dropout_feats(ldr, extractor, H, device)

        if target not in feats:
            continue

        # ── 5. Pool source features ────────────────────────────
        src_pool = {
            "mu":    _torch.cat([feats[d]["mu"]    for d in sources if d in feats]),
            "sigma": _torch.cat([feats[d]["sigma"] for d in sources if d in feats]),
            "eps":   _torch.cat([feats[d]["eps"]   for d in sources if d in feats]),
        }

        # ── 6. Full certificate (Q1 + Q2 + Q3 + IIPM + gap) ───
        ldr_tgt = get_domain_loader(cfg, target, transform_eval,
                                    N_DOM, SEED, BATCH)
        cert    = full_certificate(
            src_pool   = src_pool,
            tgt_feats  = feats[target],
            K          = K,
            head       = base.fc,
            loader_tgt = ldr_tgt,
            base       = base,
            device     = device,
        )

        # ── 7. Attach GT accuracy ──────────────────────────────
        cert["acc"]  = gt.get(name)
        cert["name"] = name
        results[target] = cert

    # ── Q2: Spearman ρ across domains ─────────────────────────
    rho = spearman_summary(results, all_domains)
    results["__rho__"] = rho

    save(f"full_{dataset}", results)
    return results


# ─────────────────────────────────────────────────────────────────
# SECTION 8: E3 STABILITY GRID
# head_epochs × dropout_p — 30 cells, sequential on same GPU
# ─────────────────────────────────────────────────────────────────

@app.function(
    gpu="T4",
    timeout=60 * 25,
    volumes={RESULTS: results_vol},
    secrets=[hf_secret],
)
def e3_cell(epoch: int, p_drop: float, target: str = "sketch"):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T  # pyright: ignore[reportMissingImports]
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    device = "cuda"
    cfg    = HF["pacs"]
    K      = cfg["n_classes"]

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds      = load_dataset(cfg["hf_id"], split=cfg["splits"][0])
    domains = sorted(set(ds[cfg["domain_col"]]))
    sources = [d for d in domains if d != target]

    # Build and fine-tune backbone
    base = build_resnet18_kclass(K, device)
    base = finetune_head(cfg, sources, base, device, transform, epoch)

    extractor = build_mc_extractor(base, device, p_drop)

    # Extract all domains
    feats = {}
    for dom in domains:
        ldr        = get_domain_loader(cfg, dom, transform, N_DOM, SEED, BATCH)
        feats[dom] = mc_dropout_feats(ldr, extractor, H, device)

    src_pool = {
        "mu":    torch.cat([feats[d]["mu"]    for d in sources]),
        "sigma": torch.cat([feats[d]["sigma"] for d in sources]),
        "eps":   torch.cat([feats[d]["eps"]   for d in sources]),
    }

    ldr_tgt = get_domain_loader(cfg, target, transform, N_DOM, SEED, BATCH)
    cert    = full_certificate(
        src_pool   = src_pool,
        tgt_feats  = feats[target],
        K          = K,
        head       = base.fc,
        loader_tgt = ldr_tgt,
        base       = base,
        device     = device,
    )

    result = dict(epoch=epoch, dropout_p=p_drop, target=target, **cert)
    return result


@app.function(
    timeout=60 * 10,
    volumes={RESULTS: results_vol},
    secrets=[hf_secret],
)
def e3(target: str = "sketch"):
    grid = {}
    for epoch, p_drop in product(E3_EPOCHS, E3_DROPS):
        r = e3_cell.local(epoch, p_drop, target)
        grid.setdefault(r["epoch"], {})[r["dropout_p"]] = r

    save(f"e3_pacs_{target}", grid)
    return grid


# ─────────────────────────────────────────────────────────────────
# SECTION 9: GPU GROUPINGS  (3 GPUs max)
# ─────────────────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=60 * 180,
    volumes={RESULTS: results_vol},
    secrets=[hf_secret],
)
def run_gpu1():
    return run_hf.local("domainnet")


@app.function(
    gpu="A100",
    timeout=60 * 150,
    volumes={RESULTS: results_vol},
    secrets=[hf_secret],
)
def run_gpu2():
    r1 = run_hf.local("officehome")
    r2 = run_hf.local("pacs")
    return {"officehome": r1, "pacs": r2}


@app.function(
    gpu="T4",
    timeout=60 * 150,
    volumes={RESULTS: results_vol},
    secrets=[hf_secret],
)
def run_gpu3():
    r1 = run_hf.local("camelyon17")
    r2 = e3.local("sketch")
    return {"camelyon17": r1, "e3": r2}


# ─────────────────────────────────────────────────────────────────
# SECTION 10: ENTRYPOINT
# ─────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    jobs = {
        "gpu1 — domainnet":         run_gpu1.spawn(),
        "gpu2 — officehome + pacs": run_gpu2.spawn(),
        "gpu3 — camelyon17 + e3":   run_gpu3.spawn(),
    }

    for job in jobs.values():
        job.get()