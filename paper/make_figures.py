"""Figures for the PimaLens IEEE paper.

Run from the project root with the project venv:
    .venv\\Scripts\\python.exe paper\\make_figures.py
Most figures read the tables written by phase1.ipynb (results/). The embedding grid, the
metaheuristic traces, the subset landscape and the test ROC are recomputed with the same
seeds, folds and code as the notebook; the script asserts that the recomputed numbers match.
"""
import itertools, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
import umap

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

# ---------------------------------------------------------------- style
INK, TEAL, RED, BLUE, GREY, AMBER, PALE = '#11222a', '#0a6970', '#b3412c', '#4f63a8', '#8a9aa0', '#c98a1b', '#e6eef0'
NEG, POS = '#5aa9a3', '#d9693f'           # Outcome 0 / 1
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': '#5d6b70',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.dpi': 200, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'axes.grid': True, 'grid.color': '#e3e9ea', 'grid.linewidth': 0.5,
})
COL1, COL2 = 3.45, 7.16   # IEEE column / page width (in)

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)
    print('  wrote', name)

# ---------------------------------------------------------------- data + notebook pipeline
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
SHORT = {'Pregnancies': 'Preg', 'Glucose': 'Glu', 'BloodPressure': 'BP', 'SkinThickness': 'Skin', 'Insulin': 'Ins',
         'BMI': 'BMI', 'DiabetesPedigreeFunction': 'DPF', 'Age': 'Age'}
ZERO_INVALID = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
raw = pd.read_csv(ROOT / 'data' / 'pima_indians_diabetes.csv').rename(columns=dict(zip(
    ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class'], FEATURES + ['Outcome'])))
raw['Outcome'] = raw['Outcome'].map({'tested_negative': 0, 'tested_positive': 1})
X_all, y_all = raw[FEATURES], raw['Outcome']
X_dev, X_test, y_dev, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
OUTER = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_dev, y_dev))

class ColumnSubset(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None): self.columns = columns
    def fit(self, X, y=None): self.columns_ = list(X.columns) if self.columns is None else list(self.columns); return self
    def transform(self, X): return X[self.columns_].copy()
class ZeroToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for c in ZERO_INVALID:
            if c in X.columns: X[c] = X[c].replace(0, np.nan)
        return X
def make_pre(cols=None):
    return Pipeline([('s', ColumnSubset(cols)), ('z', ZeroToNaN()), ('sc', StandardScaler()), ('im', KNNImputer(n_neighbors=5))])

def subset_auc(cols):
    s = []
    for tr, va in OUTER:
        p = Pipeline([('pre', make_pre(cols)), ('clf', LogisticRegression(max_iter=2000))]).fit(X_dev.iloc[tr], y_dev.iloc[tr])
        s.append(roc_auc_score(y_dev.iloc[va], p.predict_proba(X_dev.iloc[va])[:, 1]))
    return float(np.mean(s))

ALL_MASKS = [m for m in itertools.product((0, 1), repeat=8) if sum(m)]

# ================================================================= F: missingness
def fig_missingness():
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.0), gridspec_kw={'width_ratios': [1.3, 1]})
    share0 = raw[raw.Outcome == 0][ZERO_INVALID].eq(0).mean()
    share1 = raw[raw.Outcome == 1][ZERO_INVALID].eq(0).mean()
    order = raw[ZERO_INVALID].eq(0).mean().sort_values().index
    y = np.arange(len(order))
    ax[0].barh(y + 0.19, share0[order] * 100, 0.36, color=NEG, label='No diabetes (n=500)')
    ax[0].barh(y - 0.19, share1[order] * 100, 0.36, color=POS, label='Diabetes (n=268)')
    for i, c in enumerate(order):
        ax[0].text(max(share0[c], share1[c]) * 100 + 1.5, i,f'{raw[c].eq(0).sum()} zeros', va='center', fontsize=6.5, color=INK)
    ax[0].set_yticks(y, [{'SkinThickness': 'Skin fold', 'BloodPressure': 'Blood pressure'}.get(c, c) for c in order])
    ax[0].set_xlabel('Patients recorded as 0 (%)'); ax[0].set_xlim(0, 66); ax[0].legend(loc='lower right', frameon=False)
    ax[0].set_title('(a) Invalid zeros by measurement and class', loc='left')
    k = raw[ZERO_INVALID].eq(0).sum(1).value_counts().sort_index()
    ax[1].bar(k.index, k.values, color=[TEAL] + [GREY] * (len(k) - 1), width=0.7)
    for xi, v in zip(k.index, k.values): ax[1].text(xi, v + 8, str(v), ha='center', fontsize=7)
    ax[1].set_xlabel('Missing measurements per patient'); ax[1].set_ylabel('Patients'); ax[1].set_ylim(0, 450)
    ax[1].set_title('(b) Only 392 of 768 patients are complete', loc='left'); ax[1].grid(axis='x', visible=False)
    save(fig, 'fig_missingness')

# ================================================================= F: leakage evidence
def fig_leak():
    clean = raw.copy()
    for c in ZERO_INVALID:
        for k in (0, 1):
            m = raw.loc[(raw.Outcome == k) & (raw[c] != 0), c].median()
            clean.loc[(clean.Outcome == k) & (clean[c] == 0), c] = m
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.1), gridspec_kw={'width_ratios': [1.45, 1]})
    bins = np.arange(0, 600, 12)
    obs = raw[raw.Insulin > 0]
    ax[0].hist([obs[obs.Outcome == 0].Insulin, obs[obs.Outcome == 1].Insulin], bins=bins, stacked=True, color=[NEG, POS], alpha=0.55,
               label=['observed, no diabetes', 'observed, diabetes'])
    miss = raw.Insulin == 0
    for k, col, v in ((0, NEG, 102.5), (1, POS, 169.5)):
        n = int((miss & (raw.Outcome == k)).sum())
        ax[0].bar(v, n, width=7, color=col, edgecolor=INK, linewidth=0.6)
        ax[0].annotate(f'{n} patients filled\nwith {v} (class {k} median)', (v, n), xytext=(v + 60, n - 25 if k == 0 else n - 45),
                       fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.6, color=INK))
    ax[0].set_xlim(0, 600); ax[0].set_xlabel('2-hour serum insulin in the "clean" file (µU/mL)'); ax[0].set_ylabel('Patients')
    ax[0].legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 0.78)); ax[0].set_title('(a) The filled value encodes the diagnosis', loc='left')
    vals = [('Filled insulin,\n374 missing rows', roc_auc_score(raw.Outcome[miss], clean.Insulin[miss]), RED),
            ('Clean-file insulin,\nall 768', roc_auc_score(raw.Outcome, clean.Insulin), AMBER),
            ('Genuine insulin,\n394 observed', roc_auc_score(raw.Outcome[~miss], raw.Insulin[~miss]), TEAL),
            ('Glucose alone\n(raw, observed)', roc_auc_score(raw.Outcome[raw.Glucose > 0], raw.Glucose[raw.Glucose > 0]), BLUE)]
    yy = np.arange(len(vals))[::-1]
    for yi, (lab, v, col) in zip(yy, vals):
        ax[1].plot([0.52, v], [yi, yi], color=col, lw=2.2, solid_capstyle='round'); ax[1].plot(v, yi, 'o', color=col, ms=5)
        ax[1].text(v + 0.02, yi + 0.22, f'{v:.3f}', va='center', fontsize=7)
    ax[1].set_yticks(yy, [v[0] for v in vals], fontsize=6.5); ax[1].set_xlim(0.5, 1.09); ax[1].set_ylim(-0.5, 3.6); ax[1].axvline(0.5, color=GREY, lw=0.6)
    ax[1].set_xlabel('Single-feature ROC-AUC'); ax[1].set_title('(b) Leak strength', loc='left'); ax[1].grid(axis='y', visible=False)
    save(fig, 'fig_leak')

# ================================================================= F: embedding grid
def sne_joint_P(X, perplexity=30.0):
    n = X.shape[0]; D = squareform(pdist(X, 'sqeuclidean')); P = np.zeros((n, n)); target = np.log(perplexity)
    for i in range(n):
        beta, bmin, bmax = 1.0, -np.inf, np.inf; Di = np.delete(D[i], i)
        for _ in range(50):
            Pi = np.exp(-Di * beta); s = Pi.sum() + 1e-12; H = np.log(s) + beta * np.sum(Di * Pi) / s
            if abs(H - target) < 1e-5: break
            if H > target: bmin = beta; beta = beta * 2 if bmax == np.inf else (beta + bmax) / 2
            else: bmax = beta; beta = beta / 2 if bmin == -np.inf else (beta + bmin) / 2
        Pi = np.exp(-Di * beta); Pi /= Pi.sum() + 1e-12; P[i, np.arange(n) != i] = Pi
    P = (P + P.T) / (2 * n); P = np.maximum(P, 1e-12); np.fill_diagonal(P, 0); return P / P.sum()
def fit_sne(X, seed=42, n_iter=1000, lr=200.0):
    rng = np.random.default_rng(seed); P = sne_joint_P(X); Y = rng.normal(0, 1e-4, (len(X), 2)); V = np.zeros_like(Y)
    for it in range(n_iter):
        sq = (Y ** 2).sum(1); D = np.maximum(sq[:, None] + sq[None] - 2 * Y @ Y.T, 0); Q = np.exp(-D); np.fill_diagonal(Q, 0); Q /= Q.sum() + 1e-12
        PQ = P - Q; g = 4 * (PQ.sum(1)[:, None] * Y - PQ @ Y); gn = np.linalg.norm(g)
        if gn > 100: g *= 100 / gn
        V = (0.5 if it < 250 else 0.8) * V - lr * g; Y += V; Y -= Y.mean(0)
    return Y

def fig_embeddings():
    Z = make_pre().fit(X_dev).transform(X_dev); y = y_dev.to_numpy()
    tw = pd.read_csv(R / 'embedding_quality.csv').set_index('method')['trustworthiness_k10']
    methods = [('PCA', 'PCA', lambda: PCA(2, random_state=SEED).fit_transform(Z)),
               ('LDA (supervised, 1-D)', 'LDA (1-D, supervised)', lambda: LinearDiscriminantAnalysis(n_components=1).fit_transform(Z, y)),
               ('Kernel PCA (RBF, gamma=0.1)', 'Kernel PCA (RBF)', lambda: KernelPCA(2, kernel='rbf', gamma=0.1, random_state=SEED).fit_transform(Z)),
               ('SNE (custom symmetric)', 'Symmetric SNE (ours)', lambda: fit_sne(Z)),
               ('t-SNE', 't-SNE', lambda: TSNE(2, perplexity=30, init='pca', learning_rate='auto', max_iter=1000, random_state=SEED).fit_transform(Z)),
               ('UMAP', 'UMAP', lambda: umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(Z)),
               ('Metric MDS', 'Metric MDS', lambda: MDS(2, metric_mds=True, init='classical_mds', n_init=1, max_iter=300, random_state=SEED).fit_transform(Z)),
               ('Non-metric MDS', 'Non-metric MDS', lambda: MDS(2, metric_mds=False, init='classical_mds', n_init=1, max_iter=300, random_state=SEED).fit_transform(Z)),
               ('Isomap (k=15)', 'Isomap', lambda: Isomap(n_components=2, n_neighbors=15).fit_transform(Z)),
               ('LLE (k=15)', 'LLE', lambda: LocallyLinearEmbedding(n_components=2, n_neighbors=15, reg=1e-3, eigen_solver='dense', random_state=SEED).fit_transform(Z))]
    fig, axes = plt.subplots(2, 5, figsize=(COL2, 3.15))
    for ax, (key, title, f) in zip(axes.ravel(), methods):
        E = np.asarray(f(), float)
        if E.shape[1] == 1:
            bins = np.linspace(E.min(), E.max(), 30)
            ax.hist(E[y == 0, 0], bins, color=NEG, alpha=0.7, density=True); ax.hist(E[y == 1, 0], bins, color=POS, alpha=0.7, density=True)
        else:
            for k, col in ((0, NEG), (1, POS)):
                ax.scatter(E[y == k, 0], E[y == k, 1], s=1.6, color=col, alpha=0.75, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(True); s.set_color('#c7d3d5')
        ax.text(0.03, 0.04, f'T@10 = {tw[key]:.3f}', transform=ax.transAxes, fontsize=6.5, color=INK,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    fig.legend(handles=[plt.Line2D([], [], ls='', marker='o', color=NEG, ms=4, label='No diabetes'),
                        plt.Line2D([], [], ls='', marker='o', color=POS, ms=4, label='Diabetes')],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(wspace=0.06, hspace=0.22)
    save(fig, 'fig_embeddings')

# ================================================================= F: trustworthiness vs separation
def fig_quality():
    e = pd.read_csv(R / 'embedding_quality.csv')
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.25), gridspec_kw={'width_ratios': [1.25, 1]})
    names = {'Kernel PCA (RBF, gamma=0.1)': 'Kernel PCA', 'LDA (supervised, 1-D)': 'LDA (sup.)', 'SNE (custom symmetric)': 'SNE (ours)',
             'Isomap (k=15)': 'Isomap', 'LLE (k=15)': 'LLE', 'Truncated SVD': 'SVD'}
    for _, r in e.iterrows():
        col = RED if r.method.startswith(('t-SNE', 'UMAP')) else (AMBER if r.method.startswith('LDA') else (TEAL if r.method.startswith('SNE') else BLUE))
        ax[0].plot([r.trustworthiness_k5, r.trustworthiness_k30], [r.label_agreement_10nn] * 2, color=col, lw=0.8, alpha=0.6)
        ax[0].scatter(r.trustworthiness_k10, r.label_agreement_10nn, s=14 + 60 * np.sqrt(r.seconds / 14), color=col, edgecolor='white', lw=0.5, zorder=3)
        if r.method != 'Truncated SVD':
            dx, dy = {'PCA': (-0.006, 0.0022), 'Metric MDS': (0.006, 0.0032), 'Non-metric MDS': (0.012, -0.0042), 'SNE (custom symmetric)': (0.012, -0.0005),
                      'Isomap (k=15)': (-0.004, -0.0048), 'Kernel PCA (RBF, gamma=0.1)': (0.006, -0.0040), 'LLE (k=15)': (-0.012, 0.0022),
                      't-SNE': (-0.01, 0.0042), 'UMAP': (-0.012, -0.0045), 'LDA (supervised, 1-D)': (0.008, -0.0006)}.get(r.method, (0.006, 0.0007))
            ax[0].text(r.trustworthiness_k10 + dx, r.label_agreement_10nn + dy, names.get(r.method, r.method), fontsize=6.3,
                       ha='right' if dx < 0 else 'left')
    ax[0].axhline(0.6956, color=INK, lw=0.8, ls='--'); ax[0].text(0.645, 0.6975, 'original 8-D space', fontsize=6.3)
    ax[0].axhline(0.6515, color=GREY, lw=0.8, ls=':'); ax[0].text(0.645, 0.6535, 'majority class', fontsize=6.3, color=GREY)
    ax[0].set_xlabel('Trustworthiness at k=10 (whiskers: k=5 to k=30)'); ax[0].set_ylabel('10-NN label agreement')
    ax[0].set_xlim(0.64, 1.0); ax[0].set_ylim(0.645, 0.716)
    ax[0].set_title('(a) Neighbourhood quality does not create class separation', loc='left')
    e2 = e.sort_values('seconds')
    ax[1].barh(range(len(e2)), e2.seconds, color=[TEAL if s.startswith('SNE') else BLUE for s in e2.method], height=0.65)
    ax[1].set_xscale('log'); ax[1].set_yticks(range(len(e2)), [names.get(m, m) for m in e2.method]); ax[1].set_xlabel('Seconds on 614 patients (log scale)')
    for i, v in enumerate(e2.seconds): ax[1].text(v * 1.25, i, f'{v:.3g}', va='center', fontsize=6)
    ax[1].set_xlim(4e-4, 80); ax[1].grid(axis='y', visible=False); ax[1].set_title('(b) Runtime', loc='left')
    save(fig, 'fig_quality')

# ================================================================= F: forest plot of delta AUC (LR reference)
FAMC = {'All 8': INK, 'Linear': BLUE, 'Nonlinear': RED, 'Filter': GREY, 'Wrapper': GREY, 'Embedded': GREY, 'Metaheuristic / exhaustive (raw)': TEAL}
def pretty(r):
    return (r.replace('Selected raw: ', 'Raw subset: ').replace(' (LR objective)', '').replace(' (inner CV)', '').replace('Nonlinear: ', '')
             .replace('Linear: ', '').replace('Wrapper: ', '').replace('Embedded: ', '').replace('Filter: ', 'MI filter: ')
             .replace('(GB objective, tree-GA)', '(GB objective)').replace('mutual information ', '').replace('Kernel PCA RBF', 'Kernel PCA'))
def fig_forest():
    f = pd.read_csv(R / 'representation_x_classifier_folds.csv'); f = f[f.classifier == 'Logistic Regression']
    base = f[f.representation == 'All 8 original features'].set_index('fold')['roc_auc']
    rows = []
    for r, g in f.groupby('representation', sort=False):
        d = g.set_index('fold')['roc_auc'] - base
        rows.append((r, d.mean(), d.std() / np.sqrt(5), d.values, g['roc_auc'].mean()))
    ref = pd.read_csv(R / 'reference_lr_comparison.csv').set_index('representation')
    rows = sorted(rows, key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(COL1, 4.3))
    for i, (r, m, se, d, auc) in enumerate(rows):
        col = FAMC[ref.loc[r, 'family']]
        ax.scatter(d, [i] * 5, s=4, color=col, alpha=0.35, linewidths=0, zorder=2)
        ax.plot([m - 1.96 * se, m + 1.96 * se], [i, i], color=col, lw=1.1, zorder=3)
        ax.plot(m, i, 'D' if r == 'All 8 original features' else 'o', color=col, ms=3.6, zorder=4, mec='white', mew=0.4)
        ax.text(0.052, i, f'{auc:.3f}', va='center', fontsize=6, color=INK)
    ax.axvline(0, color=INK, lw=0.7); ax.axvspan(-0.005, 0.005, color=PALE, zorder=0)
    ax.set_yticks(range(len(rows)), [pretty(r[0]) for r in rows], fontsize=6.3)
    ax.set_xlim(-0.2, 0.068); ax.set_xlabel(r'$\Delta$ ROC-AUC vs all 8 features (mean $\pm$1.96 SE; dots = folds)')
    ax.text(0.052, len(rows) - 0.2, 'AUC', fontsize=6, weight='bold')
    hand = [plt.Line2D([], [], color=c, marker='o', ls='-', ms=3, lw=1, label=l) for l, c in
            [('Raw-column search', TEAL), ('Filter / wrapper / embedded', GREY), ('Linear reduction', BLUE), ('Nonlinear reduction', RED)]]
    ax.legend(handles=hand, loc='upper left', frameon=True, framealpha=0.95, edgecolor='none', fontsize=6)
    ax.grid(axis='y', visible=False)
    save(fig, 'fig_forest')

# ================================================================= F: delta heatmap representation x classifier
def fig_heat():
    s = pd.read_csv(R / 'representation_x_classifier_cv.csv')
    piv = s.pivot(index='representation', columns='classifier', values='cv_roc_auc_mean')
    order_c = ['Logistic Regression', 'Gaussian Naive Bayes', 'SVM (RBF)', 'Random Forest', 'Gradient Boosting', 'KNN (k=7)', 'Decision Tree (depth 5)']
    piv = piv[order_c]; delta = piv - piv.loc['All 8 original features']
    order_r = s.drop_duplicates('representation').set_index('representation')['family']
    fam_rank = {'All 8': 0, 'Metaheuristic / exhaustive (raw)': 1, 'Filter': 2, 'Wrapper': 2, 'Embedded': 2, 'Linear': 3, 'Nonlinear': 4}
    rows = sorted(piv.index, key=lambda r: (fam_rank[order_r[r]], -delta.loc[r, 'Logistic Regression']))
    delta = delta.loc[rows]; piv = piv.loc[rows]
    cmap = LinearSegmentedColormap.from_list('d', [RED, '#f2c9bc', '#f7f9f9', '#a9d3d0', TEAL])
    fig, ax = plt.subplots(figsize=(COL2, 4.1))
    im = ax.imshow(delta.values, cmap=cmap, norm=TwoSlopeNorm(0, -0.08, 0.04), aspect='auto')
    for i in range(len(rows)):
        for j in range(len(order_c)):
            v = piv.values[i, j]; d = delta.values[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=5.8, color='white' if d < -0.05 or d > 0.03 else INK,
                    weight='bold' if i and d > 0.002 else 'normal')
    ax.set_xticks(range(len(order_c)), ['Logistic\nregression', 'Naive\nBayes', 'SVM\n(RBF)', 'Random\nforest', 'Gradient\nboosting', 'KNN\n(k=7)', 'Decision\ntree'])
    ax.set_yticks(range(len(rows)), [pretty(r) for r in rows], fontsize=6.3); ax.grid(False)
    ax.xaxis.tick_top()
    for b in (0.5, 4.5, 11.5, 16.5):
        ax.axhline(b, color='white', lw=2.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label(r'$\Delta$ ROC-AUC vs all 8 features (same classifier)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, 'fig_heat')

# ================================================================= metaheuristics (recomputed with the notebook's code)
def build_table():
    t0 = time.perf_counter()
    tab = {m: subset_auc([f for f, b in zip(FEATURES, m) if b]) for m in ALL_MASKS}
    print(f'  exhaustive LR table in {time.perf_counter() - t0:.1f}s')
    return tab
class Obj:
    def __init__(self, tab, pen=0.002):
        self.tab, self.pen, self.cache, self.calls = tab, pen, set(), 0
        self.best, self.trace, self.utrace = -np.inf, [], []
    def __call__(self, m):
        k = tuple(int(b) for b in m); self.calls += 1; self.cache.add(k)
        f = self.tab[k] - self.pen * sum(k); self.best = max(self.best, f)
        self.trace.append(self.best); self.utrace.append(len(self.cache)); return f
def _rep(m, rng):
    if m.sum() == 0: m[rng.integers(len(m))] = 1
    return m
def run_ga(o, rng, P=10, G=8, cx=0.9, mu=1 / 8):
    pop = np.array([_rep(rng.integers(0, 2, 8), rng) for _ in range(P)]); fit = np.array([o(c) for c in pop])
    for _ in range(G - 1):
        def t():
            i, j = rng.choice(P, 2, replace=False); return pop[i] if fit[i] >= fit[j] else pop[j]
        ch = [pop[np.argmax(fit)].copy()]
        while len(ch) < P:
            a, b = t(), t()
            c = np.concatenate([a[:(k := rng.integers(1, 8))], b[k:]]) if rng.random() < cx else a.copy()
            ch.append(_rep(np.where(rng.random(8) < mu, 1 - c, c), rng))
        pop = np.array(ch); fit = np.array([o(c) for c in pop])
def run_pso(o, rng, P=10, G=8, w=0.729, c1=1.49445, c2=1.49445, vmax=4.0):
    X = np.array([_rep(rng.integers(0, 2, 8), rng) for _ in range(P)]); V = rng.uniform(-1, 1, (P, 8))
    f = np.array([o(x) for x in X]); pb, pf = X.copy(), f.copy(); g = pb[np.argmax(pf)].copy()
    for _ in range(G - 1):
        r1, r2 = rng.random((P, 8)), rng.random((P, 8))
        V = np.clip(w * V + c1 * r1 * (pb - X) + c2 * r2 * (g - X), -vmax, vmax)
        X = np.array([_rep(x, rng) for x in (rng.random((P, 8)) < 1 / (1 + np.exp(-V))).astype(int)])
        f = np.array([o(x) for x in X]); bt = f > pf; pb[bt], pf[bt] = X[bt], f[bt]; g = pb[np.argmax(pf)].copy()
def run_rs(o, rng, B=80):
    for _ in range(B): o(ALL_MASKS[rng.integers(len(ALL_MASKS))])

def fig_meta(tab):
    opt = max(v - 0.002 * sum(k) for k, v in tab.items())
    seeds = pd.read_csv(R / 'metaheuristics_lr_seeds.csv')
    algs = {'Binary PSO': (run_pso, TEAL), 'GA': (run_ga, BLUE), 'Random search': (run_rs, GREY)}
    traces = {}
    for a, (fn, _) in algs.items():
        traces[a] = []
        for s in range(10):
            o = Obj(tab); fn(o, np.random.default_rng(s)); traces[a].append((np.array(o.trace), np.array(o.utrace)))
            ref = seeds[(seeds.algorithm == a) & (seeds.seed == s)].iloc[0]
            assert abs(o.best - ref.fitness) < 1e-9 and len(o.cache) == ref.unique_subsets, (a, s)
    print('  metaheuristic traces reproduce the notebook (fitness and unique counts per seed)')
    gb = pd.read_csv(R / 'metaheuristics_gb_seeds.csv')
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.25), gridspec_kw={'width_ratios': [1.25, 1.05, 0.9]})
    x = np.arange(1, 81)
    for a, (_, col) in algs.items():
        G = np.array([opt - t for t, _ in traces[a]]) + 1e-5
        ax[0].fill_between(x, np.percentile(G, 10, 0), np.percentile(G, 90, 0), color=col, alpha=0.13, lw=0)
        ax[0].plot(x, np.median(G, 0), color=col, lw=1.4, label=a)
    ax[0].set_yscale('log'); ax[0].set_xlabel('Objective calls (budget 80)'); ax[0].set_ylabel('Gap to exhaustive optimum + $10^{-5}$')
    ax[0].legend(frameon=False); ax[0].set_title('(a) Convergence (10 seeds)', loc='left')
    for a, (_, col) in algs.items():
        sd = seeds[seeds.algorithm == a]
        ax[1].scatter(sd.unique_subsets + np.random.default_rng(1).normal(0, 0.6, len(sd)), sd.gap_to_exact + 1e-5, s=16, color=col, alpha=0.85,
                      edgecolor='white', lw=0.4, label=a)
    ax[1].scatter([255], [1e-5], marker='*', s=70, color=INK, label='Exhaustive', zorder=4)
    ax[1].set_yscale('log'); ax[1].set_xlabel('Unique subsets evaluated (cost)'); ax[1].set_ylabel('Final gap + $10^{-5}$')
    ax[1].set_xlim(20, 270); ax[1].set_title('(b) Quality vs cost', loc='left')
    ax[1].legend(frameon=False, fontsize=6, loc='center right')
    hits = pd.DataFrame({'LR objective': seeds.groupby('algorithm')['hit_exact_optimum'].sum(),
                         'GB objective': gb.groupby('algorithm')['hit_exact_optimum'].sum()}).loc[['Binary PSO', 'GA', 'Random search']]
    yy = np.arange(3)
    ax[2].barh(yy + 0.2, hits['LR objective'], 0.38, color=[TEAL, BLUE, GREY], label='LR objective')
    ax[2].barh(yy - 0.2, hits['GB objective'], 0.38, color=[TEAL, BLUE, GREY], alpha=0.45, hatch='////', edgecolor='white', label='GB objective')
    for i, (a, b) in enumerate(hits.values):
        ax[2].text(a + 0.2, i + 0.2, f'{a}/10', va='center', fontsize=6.5); ax[2].text(b + 0.2, i - 0.2, f'{b}/10', va='center', fontsize=6.5)
    ax[2].set_yticks(yy, hits.index); ax[2].invert_yaxis(); ax[2].set_xlim(0, 11.8); ax[2].set_xlabel('Seeds reaching the exact optimum')
    ax[2].legend(frameon=False, fontsize=6, loc='lower right'); ax[2].grid(axis='y', visible=False); ax[2].set_title('(c) Hit rate', loc='left')
    save(fig, 'fig_meta')

def fig_landscape(tab):
    df = pd.DataFrame([{'mask': k, 'size': sum(k), 'auc': v, 'F': v - 0.002 * sum(k)} for k, v in tab.items()])
    opt = df.loc[df.F.idxmax()]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.2), gridspec_kw={'width_ratios': [1.2, 1]})
    rng = np.random.default_rng(0)
    has_g = df['mask'].apply(lambda m: m[1] == 1)
    ax[0].scatter(df['size'][~has_g] + rng.uniform(-0.22, 0.22, (~has_g).sum()), df.F[~has_g], s=6, color=GREY, alpha=0.7, lw=0, label='without Glucose')
    ax[0].scatter(df['size'][has_g] + rng.uniform(-0.22, 0.22, has_g.sum()), df.F[has_g], s=6, color=TEAL, alpha=0.8, lw=0, label='with Glucose')
    ax[0].scatter(opt['size'], opt.F, marker='*', s=90, color=RED, zorder=4, label='exact optimum')
    ax[0].annotate('Preg + Glu + BMI + DPF\nF = %.4f (AUC %.4f)' % (opt.F, opt.auc), (opt['size'], opt.F), xytext=(5.2, 0.70), fontsize=6.5,
                   arrowprops=dict(arrowstyle='-', lw=0.6))
    ax[0].set_xlabel('Subset size |S|'); ax[0].set_ylabel(r'Objective $F(S)$ = CV AUC $-$ 0.002|S|'); ax[0].legend(frameon=False, loc='lower right')
    ax[0].set_title('(a) All 255 subsets of the raw columns', loc='left')
    top = df.nlargest(20, 'F')
    M = np.array(top['mask'].tolist())
    ax[1].imshow(M.T, cmap=LinearSegmentedColormap.from_list('b', ['#f3f6f6', TEAL]), aspect='auto')
    ax[1].set_yticks(range(8), [SHORT[f] for f in FEATURES]); ax[1].set_xticks(range(0, 20, 4), [f'#{i + 1}' for i in range(0, 20, 4)])
    ax[1].set_xlabel(f'Top-20 subsets ranked by F ({top.F.iloc[0]:.4f} down to {top.F.iloc[-1]:.4f})'); ax[1].grid(False)
    ax[1].set_title('(b) Which measurements the best subsets use', loc='left')
    save(fig, 'fig_landscape')

# ================================================================= F: selection matrix
def fig_selection():
    S = {'Mutual information': FEATURES,
         'Forward SFS': ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'],
         'Backward SFS': ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'],
         'RFECV': ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
         'L1 (lasso)': ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
         'L2 (|w| >= mean)': ['Pregnancies', 'Glucose', 'BMI'],
         'Elastic Net': ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
         'Exhaustive (LR)': ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'],
         'Exhaustive (GB)': ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']}
    seeds = pd.read_csv(R / 'metaheuristics_lr_seeds.csv')
    freq = {a: [seeds[seeds.algorithm == a].selected.str.contains(f).mean() for f in FEATURES] for a in ['GA', 'Binary PSO']}
    names = list(S) + ['GA, 10 seeds', 'Binary PSO, 10 seeds']
    M = np.array([[float(f in S[k]) for f in FEATURES] for k in S] + [freq['GA'], freq['Binary PSO']])
    fam = ['Filter'] + ['Wrapper'] * 3 + ['Embedded'] * 3 + ['Exact'] * 2 + ['Metaheuristic'] * 2
    fig, ax = plt.subplots(figsize=(COL1, 2.75))
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list('s', ['#f3f6f6', TEAL]), vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(8):
            if 0 < M[i, j] < 1: ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=6, color=INK if M[i, j] < 0.6 else 'white')
    ax.set_xticks(range(8), [SHORT[f] for f in FEATURES]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(names)), names); ax.grid(False)
    for b in (0.5, 3.5, 6.5, 8.5): ax.axhline(b, color='white', lw=2)
    for i, f in enumerate(fam):
        if i == 0 or fam[i - 1] != f: ax.text(8.1, i - 0.35, f, fontsize=5.8, color=GREY, va='top', style='italic')
    ax.set_xlim(-0.5, 9.6)
    save(fig, 'fig_selection')

# ================================================================= F: over/underfitting
def fig_fit():
    s = pd.read_csv(R / 'leaderboard.csv')
    pick = s[s.representation.isin(['All 8 original features', 'Raw columns (preprocessing inside tuned pipeline)'])].copy()
    pick['label'] = pick.classifier.str.replace(' (depth 5)', '').str.replace(' (k=7)', '').str.replace('Gaussian ', '')
    pick = pick.sort_values('cv_roc_auc_mean')
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    for i, (_, r) in enumerate(pick.iterrows()):
        tuned = 'tuned' in r.classifier
        ax.plot([r.cv_roc_auc_mean, r.train_roc_auc_mean], [i, i], color=TEAL if tuned else BLUE, lw=1.3, alpha=0.6)
        ax.plot(r.cv_roc_auc_mean, i, 'o', color=TEAL if tuned else BLUE, ms=4); ax.plot(r.train_roc_auc_mean, i, 's', mfc='white', mec=TEAL if tuned else BLUE, ms=3.6)
        ax.text(1.008, i, f'+{r.auc_gap_train_minus_cv:.3f}', va='center', fontsize=6, color=RED if r.auc_gap_train_minus_cv > 0.06 else INK)
    ax.set_yticks(range(len(pick)), pick.label, fontsize=6.4); ax.set_xlim(0.77, 1.045)
    ax.set_xlabel('ROC-AUC  (● outer-CV validation, □ training folds)'); ax.grid(axis='y', visible=False)
    ax.text(1.008, len(pick) - 0.3, 'gap', fontsize=6, weight='bold')
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker='o', label='default settings'), plt.Line2D([], [], color=TEAL, marker='o', label='nested-tuned')],
              frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6)
    save(fig, 'fig_fit')

# ================================================================= F: final test (CM + ROC)
def fig_final():
    cols = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
    pipe = Pipeline([('pre', make_pre(cols)), ('clf', LogisticRegression(max_iter=2000))]).fit(X_dev, y_dev)
    p = pipe.predict_proba(X_test)[:, 1]; y = y_test.to_numpy()
    assert abs(roc_auc_score(y, p) - 0.8105555555555556) < 1e-9, roc_auc_score(y, p)
    print('  final-model test probabilities reproduce the notebook (AUC 0.8106)')
    fig = plt.figure(figsize=(COL2, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.45)
    cmap = LinearSegmentedColormap.from_list('c', ['#f4f8f8', TEAL])
    for k, (t, title) in enumerate([(0.5, '(a) Threshold 0.5'), (0.2349, '(b) Screening threshold 0.235')]):
        ax = fig.add_subplot(gs[0, k]); pr = (p >= t).astype(int)
        cm = np.array([[((y == a) & (pr == b)).sum() for b in (0, 1)] for a in (0, 1)])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=100); ax.grid(False)
        lab = [['TN', 'FP'], ['FN', 'TP']]
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f'{lab[a][b]}\n{cm[a, b]}', ha='center', va='center', fontsize=8.5, color='white' if cm[a, b] > 55 else INK,
                        weight='bold' if lab[a][b] == 'FN' else 'normal')
        ax.set_xticks([0, 1], ['No diabetes', 'Diabetes']); ax.set_yticks([0, 1], ['No diabetes', 'Diabetes'], rotation=90, va='center')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title, loc='left')
        rec = cm[1, 1] / cm[1].sum(); spec = cm[0, 0] / cm[0].sum()
        ax.text(0.5, -0.42, f'recall {rec:.3f} · specificity {spec:.3f}', transform=ax.transAxes, ha='center', fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(SEED); grid = np.linspace(0, 1, 101); tprs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        f, t, _ = roc_curve(y[i], p[i]); tprs.append(np.interp(grid, f, t))
    tprs = np.array(tprs)
    ax.fill_between(grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), color=TEAL, alpha=0.15, lw=0, label='bootstrap 95% band')
    f, t, thr = roc_curve(y, p); ax.plot(f, t, color=TEAL, lw=1.5, label=f'test AUC {roc_auc_score(y, p):.3f}')
    for tt, mk in ((0.5, 'o'), (0.2349, 's')):
        pr = p >= tt; ax.plot(((y == 0) & pr).sum() / (y == 0).sum(), ((y == 1) & pr).sum() / (y == 1).sum(), mk, color=RED, ms=4.5)
    ax.annotate('0.5', (0.20, 0.50), xytext=(0.32, 0.36), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.annotate('0.235', (0.37, 0.889), xytext=(0.48, 0.75), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls='--'); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(frameon=False, loc='lower right', fontsize=6); ax.set_title('(c) ROC on the locked test set', loc='left')
    save(fig, 'fig_final')

# ================================================================= F: CV vs test, literature
def fig_compare():
    ph = pd.read_csv(R / 'posthoc_test_diagnostic.csv')
    lab = ['Finalist (LR, 4 raw)', 'LR, all 8', 'SVM, tuned', 'Random forest', 'Historical GB (nested)']
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.15), gridspec_kw={'width_ratios': [1, 1.15]})
    for i, (_, r) in enumerate(ph.iterrows()):
        yy = len(ph) - 1 - i
        ax[0].annotate('', (r.test_auc, yy), (r.cv_auc, yy), arrowprops=dict(arrowstyle='->', color=RED if i == 0 else GREY, lw=1.1))
        ax[0].plot(r.cv_auc, yy, 'o', color=TEAL, ms=4.5); ax[0].plot(r.test_auc, yy, 'o', mfc='white', mec=RED, ms=4.5)
        ax[0].text(0.8545, yy, f'−{r.auc_drop:.3f}', va='center', fontsize=6.3)
    ax[0].set_yticks(range(len(ph))[::-1], lab); ax[0].set_xlim(0.80, 0.868); ax[0].set_xticks([0.80, 0.82, 0.84]); ax[0].grid(axis='y', visible=False)
    ax[0].set_xlabel('ROC-AUC  (● 5-fold CV → ○ locked test)'); ax[0].set_title('(a) Every pipeline drops on this test split', loc='left')
    lit = [('Smith 1988, ADAP (sens = spec)', 0.76, None, GREY), ('Zou 2018, RF + mRMR', 0.7721, None, GREY),
           ('Chang 2022, RF (70/30)', 0.7957, 0.8624, GREY), ('Hasan 2020, ensemble', None, 0.950, GREY),
           ('Old notebook (leaked)', 0.878, 0.942, RED), ('This work, 5-fold CV', 0.7915, 0.8458, TEAL), ('This work, locked test', 0.6948, 0.8106, TEAL)]
    yy = np.arange(len(lit))[::-1]
    for y_, (n, acc, auc, col) in zip(yy, lit):
        if acc is not None:
            ax[1].plot(acc, y_ + 0.15, 'o', color=col, ms=4.5); ax[1].text(acc + 0.008, y_ + 0.15, f'{acc:.3f}', va='center', fontsize=5.8)
        if auc is not None:
            ax[1].plot(auc, y_ - 0.15, 'D', mfc='white', mec=col, ms=4, mew=1.1); ax[1].text(auc + 0.008, y_ - 0.15, f'{auc:.3f}', va='center', fontsize=5.8)
    ax[1].axvspan(0.77, 0.80, color=PALE, zorder=0); ax[1].text(0.785, -0.75, 'published accuracy', ha='center', fontsize=5.6, color=GREY)
    ax[1].set_yticks(yy, [l[0] for l in lit], fontsize=6.2); ax[1].set_xlim(0.66, 0.99); ax[1].set_ylim(-1.0, len(lit) - 0.4); ax[1].grid(axis='y', visible=False)
    ax[1].legend(handles=[plt.Line2D([], [], ls='', marker='o', color=INK, label='accuracy'),
                          plt.Line2D([], [], ls='', marker='D', mfc='white', mec=INK, label='ROC-AUC')], frameon=False, loc='lower right', fontsize=6)
    ax[1].set_title('(b) Against published Pima results', loc='left')
    fig.subplots_adjust(wspace=0.95)
    save(fig, 'fig_compare')

# ================================================================= F: clustering
def fig_cluster():
    pre = make_pre().fit(X_dev); Z = pre.transform(X_dev); y = y_dev.to_numpy()
    km = KMeans(2, n_init=20, random_state=SEED).fit(Z); c = km.labels_
    E = PCA(2, random_state=SEED).fit_transform(Z)
    units = pd.DataFrame(pre.named_steps['sc'].inverse_transform(Z), columns=FEATURES)
    hi = int(pd.Series(y).groupby(c).mean().idxmax())
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.15), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    for k, col in ((0, NEG), (1, POS)): ax[0].scatter(E[y == k, 0], E[y == k, 1], s=3, color=col, lw=0, alpha=0.8, rasterized=True)
    ax[0].set_title('(a) Known diagnosis on PCA axes', loc='left')
    for k, col, nm in ((hi, '#334e8c', 'cluster A (older, heavier)'), (1 - hi, '#9cb3d9', 'cluster B (younger, leaner)')):
        ax[1].scatter(E[c == k, 0], E[c == k, 1], s=3, color=col, lw=0, alpha=0.85, label=nm, rasterized=True)
    ax[1].legend(frameon=False, fontsize=5.8, loc='lower left', markerscale=2.5); ax[1].set_title('(b) K-means (k=2), no labels', loc='left')
    for a in ax[:2]: a.set_xticks([]); a.set_yticks([]); a.set_xlabel('PC1'); a.set_ylabel('PC2')
    feats = ['Age', 'Pregnancies', 'Glucose', 'BMI', 'Insulin', 'SkinThickness']
    med = units.groupby(c)[feats].median(); rel = (med.loc[hi] / med.loc[1 - hi] - 1) * 100
    ax[2].barh(range(len(feats)), rel.values, color='#334e8c', height=0.6)
    for i, (f, v) in enumerate(rel.items()):
        ax[2].text(v + 3, i, f'{med.loc[hi, f]:.0f} vs {med.loc[1 - hi, f]:.0f}', va='center', fontsize=6)
    ax[2].set_yticks(range(len(feats)), [SHORT.get(f, f) for f in feats]); ax[2].set_xlim(0, 260); ax[2].grid(axis='y', visible=False)
    rate = pd.Series(y).groupby(c).mean()
    ax[2].set_xlabel('Median in A relative to B (%)')
    ax[2].set_title(f'(c) Diabetes rate A {rate[hi]:.0%} vs B {rate[1 - hi]:.0%}', loc='left')
    save(fig, 'fig_cluster')

if __name__ == '__main__':
    t0 = time.perf_counter()
    fig_missingness(); fig_leak(); fig_quality(); fig_forest(); fig_heat(); fig_selection(); fig_fit(); fig_compare(); fig_cluster()
    fig_final()
    tab = build_table(); fig_meta(tab); fig_landscape(tab)
    fig_embeddings()
    print(f'done in {time.perf_counter() - t0:.0f}s')
