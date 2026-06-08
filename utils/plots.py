import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fuzzy_logic.core import (
    mf_income, mf_household, mf_workexp, mf_age, mf_dependents, mf_output,
    INCOME_CAP,
)
from fuzzy_logic.inference import UNIVERSE, SUGENO_CONST

CLR = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']


def _eval_mf(mf_func, x_vals, labels):
    """Evaluate a membership function over an array, returning a dict of {label: [values]}."""
    return {lbl: [mf_func(v)[lbl] for v in x_vals] for lbl in labels}


def plot_membership_functions():
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Fungsi Keanggotaan - Eligibilitas Rumah Subsidi', fontsize=14, fontweight='bold')

    # --- Income ---
    ax = axes[0, 0]
    x = np.linspace(0, INCOME_CAP, 1000)
    for i, lbl in enumerate(['Rendah', 'Menengah_Bawah', 'Menengah', 'Tinggi']):
        ys = [mf_income(v)[lbl] for v in x]
        ax.plot(x / 1000, ys, color=CLR[i], lw=2, label=lbl)
    ax.set_title('Income (ribu)'); ax.set_xlabel('Income (ribu)'); ax.set_ylabel('\u00b5')
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    # --- Household Size ---
    ax = axes[0, 1]
    x = np.linspace(1, 7, 200)
    for i, lbl in enumerate(['Kecil', 'Sedang', 'Besar']):
        ys = [mf_household(v)[lbl] for v in x]
        ax.plot(x, ys, color=CLR[i], lw=2, label=lbl)
    ax.set_title('Household Size'); ax.set_xlabel('Jumlah Anggota'); ax.set_ylabel('\u00b5')
    ax.legend(); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    # --- Work Experience ---
    ax = axes[1, 0]
    x = np.linspace(0, 50, 500)
    for i, lbl in enumerate(['Baru', 'Cukup', 'Berpengalaman']):
        ys = [mf_workexp(v)[lbl] for v in x]
        ax.plot(x, ys, color=CLR[i], lw=2, label=lbl)
    ax.set_title('Work Experience (tahun)'); ax.set_xlabel('Tahun'); ax.set_ylabel('\u00b5')
    ax.legend(); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    # --- Age ---
    ax = axes[1, 1]
    x = np.linspace(18, 70, 500)
    for i, lbl in enumerate(['Muda', 'Produktif', 'Senior']):
        ys = [mf_age(v)[lbl] for v in x]
        ax.plot(x, ys, color=CLR[i], lw=2, label=lbl)
    ax.set_title('Age (tahun)'); ax.set_xlabel('Usia'); ax.set_ylabel('\u00b5')
    ax.legend(); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    # --- Dependents ---
    ax = axes[2, 0]
    x = np.linspace(0, 5, 100)
    for i, lbl in enumerate(['Sedikit', 'Sedang', 'Banyak']):
        ys = [mf_dependents(v)[lbl] for v in x]
        ax.plot(x, ys, color=CLR[i], lw=2, label=lbl)
    ax.set_title('Number of Dependents'); ax.set_xlabel('Jumlah Tanggungan'); ax.set_ylabel('\u00b5')
    ax.legend(); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    # --- Output ---
    ax = axes[2, 1]
    x = np.linspace(0, 100, 500)
    for i, lbl in enumerate(['Tidak_Layak', 'Kurang_Layak', 'Cukup_Layak', 'Layak']):
        ys = [mf_output(v)[lbl] for v in x]
        ax.plot(x, ys, color=CLR[i], lw=2, label=lbl)
    ax.set_title('Output: Skor Eligibilitas'); ax.set_xlabel('Skor (0-100)'); ax.set_ylabel('\u00b5')
    ax.legend(); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_mamdani_defuzz(agg, score, label, title=''):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(UNIVERSE, 0, agg, alpha=0.35, color='steelblue')
    ax.plot(UNIVERSE, agg, color='steelblue', lw=2)
    ax.axvline(score, color='red', lw=2.5, ls='--', label=f'Centroid = {score:.2f}')
    ax.set_title(
        f'{title}\nSkor = {score:.2f}: {label}' if title else f'Skor = {score:.2f}: {label}',
        fontweight='bold',
    )
    ax.set_xlabel('Skor Eligibilitas'); ax.set_ylabel('\u00b5')
    ax.legend(); ax.set_xlim(0, 100); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sugeno_defuzz(firings, score, label, title=''):
    fig, ax = plt.subplots(figsize=(7, 4))
    const_labels = list(SUGENO_CONST.keys())
    firing_dict = {}
    for strength, val in firings:
        for k, v in SUGENO_CONST.items():
            if abs(v - val) < 0.01:
                firing_dict[k] = max(firing_dict.get(k, 0), strength)
    bar_h = [firing_dict.get(k, 0) for k in const_labels]
    clr2 = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']
    ax.bar(const_labels, bar_h, color=clr2, alpha=0.75)
    ax.axhline(score / 100, color='red', lw=2, ls='--', label=f'WA = {score:.2f}')
    ax.set_title(
        f'{title}\nSkor = {score:.2f} \u2192 {label}' if title else f'Skor = {score:.2f} \u2192 {label}',
        fontweight='bold',
    )
    ax.set_xlabel('Himpunan Output'); ax.set_ylabel('Firing Strength')
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


COLOR_PALETTE = [
    '#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
    '#00BCD4', '#FF5722', '#607D8B', '#795548', '#3F51B5',
]


def plot_score_distribution(scores_dict, threshold=55):
    """Plot histogram overlays for multiple methods.

    scores_dict: ordered dict of {name: np.array} — first two (Mamdani, Sugeno)
                 use their original colors for readability; remaining use palette.
    """
    names = list(scores_dict.keys())
    fig, ax = plt.subplots(figsize=(12, 5))

    base_styles = {
        names[0]: {'color': 'steelblue', 'alpha': 0.5},
        names[1]: {'color': 'seagreen', 'alpha': 0.5},
    } if len(names) >= 2 else {}

    for i, name in enumerate(names):
        style = base_styles.get(name, {})
        c = style.get('color', COLOR_PALETTE[i % len(COLOR_PALETTE)])
        a = style.get('alpha', 0.35)
        ax.hist(scores_dict[name], bins=40, alpha=a, color=c,
                label=name, edgecolor='white')

    ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Threshold={threshold}')
    ax.set_title('Distribusi Skor Output — Semua Metode')
    ax.set_xlabel('Skor'); ax.set_ylabel('Frekuensi')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_scatter_comparison(scores_dict, reference_name=None):
    """Scatter plots for multiple methods.

    If reference_name is given (default: first key in dict), every other method
    is scattered against that reference in a grid. The first subplot always
    shows the first two methods (Mamdani vs Sugeno) for the core fuzzy comparison.
    """
    names = list(scores_dict.keys())
    other = names[2:] if len(names) > 2 else []

    n_plots = 1 + len(other)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    ref = reference_name or names[0]

    idx = 0
    # first subplot: Mamdani vs Sugeno (or first two methods)
    if len(names) >= 2:
        ax = axes[idx]
        ax.scatter(scores_dict[names[0]], scores_dict[names[1]],
                   alpha=0.3, s=8, color='purple')
        ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Identitas')
        ax.set_title(f'{names[0]} vs {names[1]}')
        ax.set_xlabel(f'Skor {names[0]}'); ax.set_ylabel(f'Skor {names[1]}')
        ax.legend(); ax.grid(alpha=0.3)
        idx += 1

    # remaining subplots: each other method vs reference
    pair_colors = ['#E91E63', '#FF9800', '#4CAF50', '#00BCD4', '#9C27B0', '#FF5722', '#607D8B']
    for j, name in enumerate(other):
        ax = axes[idx]
        ax.scatter(scores_dict[ref], scores_dict[name],
                   alpha=0.3, s=8, color=pair_colors[j % len(pair_colors)])
        ax.plot([0, 100], [0, 100], 'r--', lw=2, label='Identitas')
        ax.set_title(f'{ref} vs {name}')
        ax.set_xlabel(f'Skor {ref}'); ax.set_ylabel(f'Skor {name}')
        ax.legend(); ax.grid(alpha=0.3)
        idx += 1

    for k in range(idx, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    return fig


def plot_segment_comparison(scores_dict, incomes):
    """Grouped bar chart: mean score per income segment for all methods.

    scores_dict: ordered dict of {name: np.array}
    """
    data = pd.DataFrame({'Income': incomes})
    for name, scores in scores_dict.items():
        data[f'{name}_Score'] = scores

    data['Income_Seg'] = pd.cut(
        data['Income'].clip(upper=INCOME_CAP),
        bins=[0, 70000, 150000, 400000, INCOME_CAP],
        labels=['Rendah', 'Menengah Bawah', 'Menengah', 'Tinggi'],
    )

    methods = [f'{n}_Score' for n in scores_dict.keys()]
    base_colors = {'Mamdani_Score': 'steelblue', 'Sugeno_Score': 'seagreen'}
    colors = []
    display_names = []
    for name in scores_dict.keys():
        colors.append(base_colors.get(f'{name}_Score',
                       COLOR_PALETTE[len(colors) % len(COLOR_PALETTE)]))
        display_names.append(name)

    grp = data.groupby('Income_Seg', observed=False)[methods].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(methods)
    w = 0.6 / n
    x_pos = np.arange(len(grp))

    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - (n - 1) / 2) * w
        ax.bar(x_pos + offset, grp[method], width=w, color=color,
               label=display_names[i], alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(grp.index, fontsize=9)
    ax.set_title('Rata-rata Skor per Segmen Income — Semua Metode')
    ax.set_ylabel('Rata-rata Skor')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig
