"""
plot_credal_pacs.py
===================
Generate Figure 2 — PACS credal diagnostic panel — from actual
numbers produced by credal_dg_pacs.py (results.json).

USAGE:
    python plot_credal_pacs.py --results credal_dg_results/results.json
    python plot_credal_pacs.py --results credal_dg_results/results.json \
                               --pacs_root /PACS \
                               --output credal_dg_results/fig_pacs_credal.pdf

OUTPUTS:
    fig_pacs_credal.pdf   — publication-ready (vector, no raster)
    fig_pacs_credal.png   — 300 dpi raster for slides/preview

DEPENDENCIES:
    pip install matplotlib numpy pillow
    (torchvision only needed if --pacs_root is given for thumbnails)

LAYOUT (4 rows):
    Row 1  Domain thumbnails (loaded from PACS if --pacs_root given,
           otherwise schematic placeholders)
    Row 2  Credal ellipses (radius ∝ ε from results.json)
    Row 3  Two bar rows: raw MMD then (1−ε)·MMD
    Row 4  Mechanism callout: photo vs sketch
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib import gridspec


# ── aesthetics (matches NeurIPS style) ───────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         8,
    'axes.linewidth':    0.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  2,
    'ytick.major.size':  2,
    'pdf.fonttype':      42,   # embeds fonts for ACM/IEEE
    'ps.fonttype':       42,
})

TEAL   = '#1D9E75'
TEAL_L = '#E1F5EE'
TEAL_D = '#085041'
RED    = '#E24B4A'
RED_L  = '#FCEBEB'
GRAY   = '#888780'
GRAY_L = '#F1EFE8'
GRAY_M = '#D3D1C7'

DOMAIN_ORDER   = ['photo', 'art_painting', 'cartoon', 'sketch']
DOMAIN_LABELS  = ['Photo', 'Art', 'Cartoon', 'Sketch']
DOMAIN_COLORS  = [TEAL, TEAL, TEAL, RED]    # sketch = red (tight, exposed)
DOMAIN_FILL    = [TEAL_L, TEAL_L, TEAL_L, RED_L]


# =============================================================================
# 1. LOAD RESULTS
# =============================================================================

def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Index by domain name for easy lookup
    by_domain = {r['domain']: r for r in data['records']}
    return data, by_domain


# =============================================================================
# 2. DOMAIN THUMBNAILS
# =============================================================================

def load_pacs_thumbnail(pacs_root: str, domain: str, size: int = 96) -> np.ndarray | None:
    """
    Load one representative image from the PACS domain.
    Returns (H, W, 3) uint8 array or None if unavailable.
    """
    try:
        from PIL import Image
        domain_dir = Path(pacs_root) / 'pacs_data' / domain
        # Take the first image found (sorted for reproducibility)
        imgs = sorted(domain_dir.rglob('*.jpg')) + sorted(domain_dir.rglob('*.png'))
        if not imgs:
            return None
        img = Image.open(imgs[0]).convert('RGB')
        # Centre-crop to square then resize
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


def draw_schematic_thumbnail(ax, domain: str) -> None:
    """
    Draw a schematic domain icon when real images are unavailable.
    Uses simple matplotlib patches to suggest the domain style.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    if domain == 'photo':
        # Photorealistic: filled shapes, tonal
        sky = mpatches.Rectangle((0, 0.5), 1, 0.5, fc='#B5D4F4', ec='none')
        ground = mpatches.Rectangle((0, 0), 1, 0.5, fc='#97C459', ec='none')
        body = mpatches.Ellipse((0.42, 0.38), 0.44, 0.28, fc='#EF9F27', ec='#633806', lw=0.6)
        head = mpatches.Circle((0.65, 0.52), 0.18, fc='#FAC775', ec='#633806', lw=0.6)
        ear = mpatches.Ellipse((0.57, 0.64), 0.10, 0.16, fc='#EF9F27', ec='#633806', lw=0.6)
        for p in [sky, ground, body, head, ear]:
            ax.add_patch(p)
        ax.plot(0.69, 0.52, 'o', ms=3, color='#2C2C2A')

    elif domain == 'art_painting':
        # Painterly: loose colour washes
        ax.set_facecolor('#FAEEDA')
        for cx, cy, rx, ry, c, a in [
            (0.35, 0.65, 0.30, 0.22, '#EF9F27', 0.30),
            (0.55, 0.52, 0.28, 0.18, '#9FE1CB', 0.25),
            (0.45, 0.45, 0.22, 0.16, '#BA7517', 0.75),
            (0.58, 0.57, 0.15, 0.12, '#EF9F27', 0.80),
        ]:
            ax.add_patch(mpatches.Ellipse((cx, cy), rx*2, ry*2, fc=c, ec='none', alpha=a))
        ax.plot([0.30, 0.42], [0.35, 0.33], color='#633806', lw=1.5, solid_capstyle='round')

    elif domain == 'cartoon':
        # Cartoon: bold outlines, flat fill
        sky = mpatches.Rectangle((0, 0.55), 1, 0.45, fc='#B5D4F4', ec='none')
        ground = mpatches.Rectangle((0, 0), 1, 0.45, fc='#C0DD97', ec='none')
        body = mpatches.Ellipse((0.42, 0.42), 0.44, 0.28, fc='#EF9F27', ec='#2C2C2A', lw=1.8)
        head = mpatches.Circle((0.66, 0.56), 0.18, fc='#FAC775', ec='#2C2C2A', lw=1.8)
        ear = mpatches.Polygon([[0.58, 0.68], [0.53, 0.80], [0.48, 0.68]],
                                fc='#EF9F27', ec='#2C2C2A', lw=1.8)
        eye_w = mpatches.Circle((0.71, 0.56), 0.055, fc='white', ec='#2C2C2A', lw=1.2)
        eye_p = mpatches.Circle((0.71, 0.56), 0.028, fc='#2C2C2A')
        tongue = mpatches.Ellipse((0.74, 0.46), 0.07, 0.09, fc='#D4537E', ec='#2C2C2A', lw=1.2)
        for p in [sky, ground, body, head, ear, eye_w, eye_p, tongue]:
            ax.add_patch(p)

    elif domain == 'sketch':
        # Sketch: pure line art, light background
        ax.set_facecolor('#F8F6F0')
        lw = 1.1
        c  = '#2C2C2A'
        # Body
        ax.add_patch(mpatches.Ellipse((0.40, 0.38), 0.44, 0.26, fc='none', ec=c, lw=lw))
        # Head
        ax.add_patch(mpatches.Circle((0.62, 0.54), 0.18, fc='none', ec=c, lw=lw))
        # Ear
        ax.add_patch(mpatches.Arc((0.54, 0.66), 0.10, 0.20, angle=0,
                                   theta1=170, theta2=360, ec=c, lw=lw))
        # Eye
        ax.plot(0.66, 0.54, 'o', ms=2.5, color=c)
        # Nose
        ax.add_patch(mpatches.Ellipse((0.72, 0.48), 0.04, 0.025, fc=c, ec='none'))
        # Legs
        for x in [0.28, 0.38]:
            ax.plot([x, x - 0.02], [0.25, 0.10], color=c, lw=lw, solid_capstyle='round')
        # Tail
        ax.add_patch(mpatches.Arc((0.20, 0.38), 0.18, 0.22, angle=0,
                                   theta1=200, theta2=360, ec=c, lw=lw))
        # Hatching
        for dx in [0.38, 0.43, 0.48]:
            ax.plot([dx, dx + 0.04], [0.42, 0.36], color=c, lw=0.4, alpha=0.4)


# =============================================================================
# 3. MAIN FIGURE
# =============================================================================

def make_figure(
    results_path: str,
    pacs_root: str | None,
    output_stem: str,
) -> None:
    data, by_domain = load_results(results_path)

    rho_mmd  = data['rho_mmd']
    rho_iipm = data['rho_iipm']   # (1-ε)·MMD

    # Per-domain values in display order
    domains = DOMAIN_ORDER
    labels  = DOMAIN_LABELS
    accs    = [by_domain[d]['accuracy'] for d in domains]
    mmds    = [by_domain[d]['mmd']      for d in domains]
    certs   = [by_domain[d]['cert']     for d in domains]  # (1-ε)·MMD
    epss    = [by_domain[d]['eps']      for d in domains]
    mmis    = [by_domain[d]['mmi']      for d in domains]

    # Normalise bars to [0,1] for visual clarity
    mmd_max  = max(mmds)
    cert_max = max(certs)
    mmds_n   = [v / mmd_max  for v in mmds]
    certs_n  = [v / cert_max for v in certs]

    # Ellipse radii proportional to ε (scale to reasonable plot units)
    eps_max   = max(epss)
    eps_scale = 0.38   # max half-width in axes units [0,1]
    radii     = [e / eps_max * eps_scale for e in epss]

    # ── figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 6.8))
    gs  = gridspec.GridSpec(
        4, 4,
        figure=fig,
        height_ratios=[1.2, 1.0, 0.9, 0.7],
        hspace=0.55,
        wspace=0.18,
    )

    # ── Row 0: thumbnails ───────────────────────────────────────────────
    thumb_axes = []
    for col, (domain, label) in enumerate(zip(domains, labels)):
        ax = fig.add_subplot(gs[0, col])
        thumb_axes.append(ax)

        if pacs_root:
            img = load_pacs_thumbnail(pacs_root, domain, size=96)
            if img is not None:
                ax.imshow(img, interpolation='lanczos')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color(GRAY_M)
            else:
                draw_schematic_thumbnail(ax, domain)
        else:
            draw_schematic_thumbnail(ax, domain)

        col_c = RED if domain == 'sketch' else TEAL_D
        ax.set_title(label, fontsize=8, fontweight='bold', color=col_c, pad=3)

    # ── Row 1: credal ellipses ──────────────────────────────────────────
    ellipse_axes = []
    for col, (domain, eps, r) in enumerate(zip(domains, epss, radii)):
        ax = fig.add_subplot(gs[1, col])
        ellipse_axes.append(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Ellipse fill + border
        fc = TEAL_L if domain != 'sketch' else RED_L
        ec = TEAL   if domain != 'sketch' else RED
        el = Ellipse(
            xy=(0.5, 0.5), width=r * 2, height=r * 1.3,
            fc=fc, ec=ec, lw=0.9,
            linestyle=(0, (4, 3)),   # dashed
        )
        ax.add_patch(el)

        # Centre dot
        dot_c = TEAL if domain != 'sketch' else RED
        ax.plot(0.5, 0.5, 'o', ms=5, color=dot_c, zorder=3)

        # ε label inside
        txt_c = TEAL_D if domain != 'sketch' else '#791F1F'
        ax.text(0.5, 0.22, f'ε = {eps:.3f}',
                ha='center', va='center', fontsize=7, color=txt_c)

        # Absorption annotation
        if domain != 'sketch':
            ax.text(0.5, 0.10, 'absorbs shift',
                    ha='center', va='center', fontsize=6.5, color=TEAL)
        else:
            ax.text(0.5, 0.10, 'fully exposed',
                    ha='center', va='center', fontsize=6.5, color=RED)

        # ε radius indicator (first domain only)
        if col == 0:
            ax.annotate(
                '', xy=(0.5, 0.5 - r * 0.65), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=0.7),
            )
            ax.text(0.56, 0.5 - r * 0.32, 'ε',
                    ha='left', va='center', fontsize=7,
                    color=TEAL, style='italic')

    # ── Row 2: two bar sub-rows ─────────────────────────────────────────
    bar_ax = fig.add_subplot(gs[2, :])
    bar_ax.axis('off')

    n  = len(domains)
    xs = np.arange(n)
    bw = 0.28   # bar width in domain-unit space

    # We draw manually inside bar_ax using transforms
    # Use a helper axis for each bar sub-row
    ax_mmd  = fig.add_axes([0.08, 0.395, 0.88, 0.075])
    ax_cert = fig.add_axes([0.08, 0.300, 0.88, 0.075])

    for ax, vals, rho, label, color in [
        (ax_mmd,  mmds_n,  rho_mmd,  'MMD',          RED),
        (ax_cert, certs_n, rho_iipm, '(1−ε)·MMD',    TEAL),
    ]:
        bar_colors = [
            RED if (d == 'sketch' and color == RED) else
            TEAL if (color == TEAL) else
            '#F09595'
            for d in domains
        ]
        if color == TEAL:
            bar_colors = [TEAL if d != 'sketch' else RED for d in domains]
        else:
            bar_colors = ['#F09595' if d != 'sketch' else RED for d in domains]

        bars = ax.barh(xs, vals, height=0.55, color=bar_colors, alpha=0.85)

        # Value labels inside bars
        raw_vals = mmds if color == RED else certs
        for i, (bar, v) in enumerate(zip(bars, raw_vals)):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{v:.2f}', va='center', ha='left', fontsize=6.5,
                    color=GRAY if domains[i] != 'sketch' else RED)

        ax.set_yticks(xs)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, 1.22)
        ax.set_ylim(-0.5, n - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([])
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)

        rho_c = TEAL if color == TEAL else RED
        ax.text(1.20, -0.3,
                f'{label}   ρ = {rho:+.2f}',
                ha='right', va='top', fontsize=7,
                color=rho_c, style='italic',
                transform=ax.transData)

    # Accuracy overlay dots on cert row
    ax2 = ax_cert.twiny()
    ax2.set_xlim(70, 105)
    ax2.invert_yaxis()
    ax2.plot(accs, xs, 'D', ms=4, color='#2C2C2A', zorder=5, label='ERM acc (%)')
    ax2.set_xticks([75, 80, 85, 90, 95, 100])
    ax2.tick_params(axis='x', labelsize=6, colors='#444441')
    ax2.spines['top'].set_linewidth(0.4)
    ax2.spines['top'].set_color(GRAY_M)
    ax2.set_xlabel('ERM accuracy (%)', fontsize=6.5, color='#444441', labelpad=2)

    # Add accuracy values next to dots
    for acc, xi, domain in zip(accs, xs, domains):
        ax2.text(acc + 0.6, xi, f'{acc:.1f}%',
                 va='center', ha='left', fontsize=6,
                 color=RED if domain == 'sketch' else TEAL_D)

    # ── Row 3: mechanism callout ────────────────────────────────────────
    ax_mech = fig.add_axes([0.08, 0.04, 0.88, 0.11])
    ax_mech.axis('off')
    ax_mech.set_xlim(0, 1)
    ax_mech.set_ylim(0, 1)

    # Photo box
    photo_box = mpatches.FancyBboxPatch(
        (0.01, 0.05), 0.28, 0.88,
        boxstyle='round,pad=0.02',
        fc=TEAL_L, ec=TEAL, lw=0.6,
    )
    ax_mech.add_patch(photo_box)
    r = by_domain['photo']
    ax_mech.text(0.15, 0.78, 'Photo', ha='center', fontsize=8,
                 fontweight='bold', color=TEAL_D)
    ax_mech.text(0.15, 0.55,
                 f'MMD = {r["mmd"]:.2f}   ε = {r["eps"]:.3f}',
                 ha='center', fontsize=6.5, color=TEAL_D)
    ax_mech.text(0.15, 0.33, 'wide set → absorbs shift',
                 ha='center', fontsize=6.5, color=TEAL)
    ax_mech.text(0.15, 0.12, f'{r["accuracy"]:.1f}% acc  ✓',
                 ha='center', fontsize=7, fontweight='bold', color=TEAL_D)

    # Central insight box
    insight_box = mpatches.FancyBboxPatch(
        (0.34, 0.05), 0.32, 0.88,
        boxstyle='round,pad=0.02',
        fc=TEAL_D, ec=TEAL, lw=0.6,
    )
    ax_mech.add_patch(insight_box)
    ax_mech.text(0.50, 0.68, 'epistemic uncertainty',
                 ha='center', fontsize=7.5, fontweight='bold', color='#9FE1CB')
    ax_mech.text(0.50, 0.45, 'is a geometric asset',
                 ha='center', fontsize=7, color='#E1F5EE')
    ax_mech.text(0.50, 0.22, 'wider ε → smaller (1−ε)·MMD',
                 ha='center', fontsize=6.5, color='#5DCAA5')

    # Sketch box
    sketch_box = mpatches.FancyBboxPatch(
        (0.71, 0.05), 0.28, 0.88,
        boxstyle='round,pad=0.02',
        fc=RED_L, ec=RED, lw=0.6,
    )
    ax_mech.add_patch(sketch_box)
    r = by_domain['sketch']
    ax_mech.text(0.85, 0.78, 'Sketch', ha='center', fontsize=8,
                 fontweight='bold', color='#791F1F')
    ax_mech.text(0.85, 0.55,
                 f'MMD = {r["mmd"]:.2f}   ε = {r["eps"]:.3f}',
                 ha='center', fontsize=6.5, color='#791F1F')
    ax_mech.text(0.85, 0.33, 'tight set → fully exposed',
                 ha='center', fontsize=6.5, color=RED)
    ax_mech.text(0.85, 0.12, f'{r["accuracy"]:.1f}% acc  ✗',
                 ha='center', fontsize=7, fontweight='bold', color='#791F1F')

    # ── global title and row labels ─────────────────────────────────────
    fig.text(0.50, 0.995, 'PACS credal diagnostic', ha='center', va='top',
             fontsize=9, fontweight='bold', color='#2C2C2A')
    fig.text(0.50, 0.978,
             f'frozen ResNet-18  ·  Laplace ε  ·  no DG training  ·  '
             f'ρ(MMD) = {rho_mmd:+.2f}  →  ρ((1−ε)·MMD) = {rho_iipm:+.2f}',
             ha='center', va='top', fontsize=7, color=GRAY)

    # Row labels (left margin)
    for y, txt in [
        (0.855, 'domain'),
        (0.680, 'credal set\nC_ε(·)'),
        (0.430, 'distance\nmeasures'),
        (0.095, 'mechanism'),
    ]:
        fig.text(0.005, y, txt, ha='left', va='center',
                 fontsize=6.5, color=GRAY, rotation=0,
                 style='italic')

    # ── save ────────────────────────────────────────────────────────────
    for ext in ['pdf', 'png']:
        out = f'{output_stem}.{ext}'
        dpi = 72 if ext == 'pdf' else 300
        fig.savefig(out, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'  Saved: {out}')

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot PACS credal figure from results.json'
    )
    parser.add_argument(
        '--results', type=str,
        default='credal_dg_results/results.json',
        help='Path to results.json produced by credal_dg_pacs.py',
    )
    parser.add_argument(
        '--pacs_root', type=str, default=None,
        help='Path to PACS/ directory (optional — enables real thumbnails)',
    )
    parser.add_argument(
        '--output', type=str,
        default='credal_dg_results/fig_pacs_credal',
        help='Output path stem (without extension — .pdf and .png produced)',
    )
    args = parser.parse_args()

    if not Path(args.results).exists():
        raise FileNotFoundError(
            f"results.json not found at {args.results}\n"
            "Run credal_dg_pacs.py first."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading results: {args.results}")
    print(f"PACS root: {args.pacs_root or 'not provided — using schematic thumbnails'}")
    print(f"Output stem: {args.output}")

    make_figure(args.results, args.pacs_root, args.output)
    print("Done.")


if __name__ == '__main__':
    main()