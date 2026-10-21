import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import hdbscan

# =============================================================================
# ▶ НАСТРОЙКИ
# =============================================================================

INPUT_DIR    = r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\CASA METRICS\link_g 30, link_sp_diff 0.85, link_m_d 100, thresh 160, col_dist 30, min_dur 20\CASA TRACKS v2"
FILE_PATTERN = "*.xlsx"
OUTPUT_DIR   = r"D:\Lab\20 videos\Without grid\UNDERSTANDING\HDBSCAN CASA 2 PARAMETERS\PURE\VAP STR, EOM MCS 0.03, MS 10, IM MERGE LARGEST SLOW + NOISE P95, A PROB 0.5, C Cluster"

X_COL = 'CASA_VAP'   # ось «скорость»
Y_COL = 'CASA_STR'   # ось «прямолинейность»
Z_COL = None         # третий признак для HDBSCAN, например 'CASA_LIN' или 'CASA_ALH_mean'; None — только два.
                     # IM и A по-прежнему определяются по X_COL и Y_COL, графики — тоже X×Y.

MODE = 'per_video'   # 'per_video' — каждое видео отдельно; 'pooled' — все треки всех видео вместе

SCALER   = 'standard'   # 'standard' | 'robust' | 'yeo-johnson'
WEIGHT_X = 1.0           # вес признака после стандартизации
WEIGHT_Y = 1.0
WEIGHT_Z = 1.0
# IM_FROM_2D — только при Z_COL: неподвижные (IM) берутся из отдельного прогона HDBSCAN на X_COL + Y_COL,
# а третий признак используется только для остальных кластеров (A и др.). У неподвижных третий признак
# (LIN, STR) случаен и размывает их кластер, а для них важна только скорость.
IM_FROM_2D = True

HDBSCAN_PARAMS = dict(
    min_cluster_size=0.03,           # int — число треков; float < 1 — доля от треков
    min_samples=10,
    cluster_selection_method='eom',  # 'eom' | 'leaf'
    cluster_selection_epsilon=0.0,
    metric='euclidean',
)

# IM_MERGE: опорный кластер (IM_REF) = неподвижные (IM). Граница = 95-й перцентиль его скорости.
# Другие кластеры с медианой скорости ниже границы сливаются с ним, шумовые треки медленнее границы входят в IM.
IM_MERGE    = True
IM_QUANTILE = 0.95
# IM_REF — какой кластер взять за неподвижный:
#   'slowest'      — самый медленный кластер C1;
#   'largest_slow' — самый большой из «медленных» кластеров, т.е. с медианой скорости ниже медианы скорости
#                    всех треков этого видео (без заданных чисел; если таких нет — C1).
IM_REF      = 'largest_slow'
# IM_LIMIT_BY_A — граница IM не дальше «середины» между IM и A: среднего геометрического медиан скорости
# кластеров IM и A (берётся из самих кластеров, заданных чисел нет). Треки IM быстрее середины → шум.
IM_LIMIT_BY_A = True
# A_MODE — добор шума в быстрый кластер A (быстрые прогрессивные). A = самый большой из «быстрых» кластеров,
# т.е. с медианой скорости выше медианы скорости всех треков видео (если таких нет — самый быстрый кластер).
#   None       — не добирать (A всё равно выделяется и подписывается);
#   'prob'     — шум → A, если по вероятностям HDBSCAN он относится к A с долей ≥ A_PROB;
#   'quantile' — шум → A, если его скорость и прямолинейность не ниже A_QUANTILE-перцентиля (5%) самого A.
A_MODE     = 'prob'
A_PROB     = 0.5
A_QUANTILE = 0.05
# NP — непрогрессивные (класс c): треки вне IM и A, которые не медленнее A (скорость ≥ NP_QUANTILE-перцентиля
# скорости A), но кривее его (STR < NP_QUANTILE-перцентиля STR A). False — не выделять.
# A_TRIM — из A убирается его медленная часть: треки медленнее A_TRIM-перцентиля скорости самого A
# (граница своя в каждом видео) → 'A_slow'. None — не убирать.
A_TRIM      = 0.33
NP_MODE     = True
NP_QUANTILE = 0.05
SOFT_ASSIGN = None   # None — шум остаётся шумом; например 0.5 — шум → самый вероятный кластер, если вероятность ≥ 0.5
MIN_TRACKS  = 20

SAVE_PLOTS        = True
SAVE_TRACK_LABELS = True

# =============================================================================
# 1. ЗАГРУЗКА
# =============================================================================

def load_videos() -> dict:
    files = sorted(f for f in Path(INPUT_DIR).glob(FILE_PATTERN) if not f.name.startswith('~$'))
    if not files:
        raise FileNotFoundError(f"Нет файлов {FILE_PATTERN} в {INPUT_DIR}")
    videos = {}
    for f in files:
        df = pd.read_excel(f)
        df['x'] = pd.to_numeric(df[X_COL], errors='coerce')
        df['y'] = pd.to_numeric(df[Y_COL], errors='coerce')
        if Z_COL:
            df['z'] = pd.to_numeric(df[Z_COL], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'] + (['z'] if Z_COL else [])).reset_index(drop=True)
        if len(df) >= MIN_TRACKS:
            videos[f.stem] = df
    print(f"Загружено видео: {len(videos)}")
    return videos

# =============================================================================
# 2. HDBSCAN
# =============================================================================

def standardize(X: np.ndarray) -> np.ndarray:
    scaler = {'standard': StandardScaler(), 'robust': RobustScaler(),
              'yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True)}[SCALER]
    Xs = scaler.fit_transform(X.astype(float))
    return Xs * np.array([WEIGHT_X, WEIGHT_Y, WEIGHT_Z][:Xs.shape[1]])


def cluster(df: pd.DataFrame, use_z: bool = True):
    """Метки: -1 = шум, 1..k = кластеры по возрастанию средней скорости."""
    p = dict(HDBSCAN_PARAMS)
    if isinstance(p['min_cluster_size'], float) and p['min_cluster_size'] < 1:
        p['min_cluster_size'] = max(2, int(round(p['min_cluster_size'] * len(df))))
    Xs = standardize(df[['x', 'y'] + (['z'] if Z_COL and use_z else [])].values)
    cl = hdbscan.HDBSCAN(**p, prediction_data=SOFT_ASSIGN is not None or A_MODE == 'prob').fit(Xs)
    raw = cl.labels_.copy()
    ids = sorted(set(raw) - {-1})
    share = None
    if (SOFT_ASSIGN is not None or A_MODE == 'prob') and ids:
        mv = np.asarray(hdbscan.all_points_membership_vectors(cl)).reshape(len(raw), -1)
        share = mv / np.where(mv.sum(1, keepdims=True) > 0, mv.sum(1, keepdims=True), 1)

    if SOFT_ASSIGN is not None and ids:
        take = (raw == -1) & (share.max(1) >= SOFT_ASSIGN)
        raw[take] = share[take].argmax(1)

    order = sorted(ids, key=lambda c: df['x'].values[raw == c].mean())
    remap = {old: new for new, old in enumerate(order, start=1)}
    labels = np.array([remap.get(c, -1) for c in raw])
    # вероятности в той же нумерации, что и метки: prob[:, k] — доля принадлежности к кластеру Ck
    prob = None
    if share is not None:
        prob = np.zeros((len(raw), len(ids) + 1))
        for old, new in remap.items():
            prob[:, new] = share[:, old]
    return labels, p, prob


A = 100   # метка быстрого кластера A
NP = 102  # метка непрогрессивных (c)
A_SLOW = 103  # медленная часть A (A_TRIM)


def trim_fast(df: pd.DataFrame, labels: np.ndarray):
    """Треки A медленнее A_TRIM-перцентиля скорости A → A_SLOW. Возвращает метки и границу."""
    in_a = labels == A
    if A_TRIM is None or not in_a.any():
        return labels, np.nan
    cut = float(np.quantile(df['x'].values[in_a], A_TRIM))
    new = labels.copy()
    new[in_a & (df['x'].values < cut)] = A_SLOW
    return new, round(cut, 3)


def assign_np(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Треки вне IM и A со скоростью на уровне A, но прямолинейностью ниже A → NP."""
    in_a = labels == A
    if not NP_MODE or not in_a.any():
        return labels
    sp, st = df['x'].values, df['y'].values
    new = labels.copy()
    new[~np.isin(labels, [0, A]) & (sp >= np.quantile(sp[in_a], NP_QUANTILE))
        & (st < np.quantile(st[in_a], NP_QUANTILE))] = NP
    return new


def assign_fast(df: pd.DataFrame, labels: np.ndarray, prob):
    """Выделяет быстрый кластер A (метка 100) и, по A_MODE, добирает в него шум."""
    sp, st = df['x'].values, df['y'].values
    ids = [k for k in set(labels) if k > 0]
    if not ids:
        return labels
    fast = [k for k in ids if np.median(sp[labels == k]) > np.median(sp)]
    ref = max(fast, key=lambda k: (labels == k).sum()) if fast else max(ids, key=lambda k: np.median(sp[labels == k]))
    new = labels.copy()
    noise = labels == -1
    if A_MODE == 'prob' and prob is not None:
        take = noise & (prob[:, ref] >= A_PROB)
    elif A_MODE == 'quantile':
        in_a = labels == ref
        take = noise & (sp >= np.quantile(sp[in_a], A_QUANTILE)) & (st >= np.quantile(st[in_a], A_QUANTILE))
    else:
        take = np.zeros(len(labels), bool)
    new[(labels == ref) | take] = A
    return new

# =============================================================================
# 3. СТАТИСТИКА
# =============================================================================

def merge_immotile(df: pd.DataFrame, labels: np.ndarray):
    """C1 + кластеры с медианой скорости ниже 95-го перцентиля C1 + шум медленнее этой границы → IM (метка 0)."""
    if not IM_MERGE or not (set(labels) - {-1, A}):
        return labels, np.nan
    sp = df['x'].values
    ref = min(set(labels) - {-1, A}, key=lambda k: np.median(sp[labels == k]))
    if IM_REF == 'largest_slow':
        slow_ids = [k for k in set(labels) - {-1, A} if np.median(sp[labels == k]) < np.median(sp)]
        if slow_ids:
            ref = max(slow_ids, key=lambda k: (labels == k).sum())
    cut = float(np.quantile(sp[labels == ref], IM_QUANTILE))
    slow = [k for k in set(labels) - {-1, A} if k == ref or np.median(sp[labels == k]) < cut]
    new = labels.copy()
    new[np.isin(labels, slow)] = 0
    new[(labels == -1) & (sp < cut)] = 0
    if IM_LIMIT_BY_A and (new == A).any() and (new == 0).any():
        mid = float(np.sqrt(np.median(sp[new == 0]) * np.median(sp[new == A])))
        new[(new == 0) & (sp >= mid)] = -1
        cut = min(cut, mid)
    rest = sorted(set(new) - {-1, 0, A}, key=lambda k: sp[new == k].mean())     # оставшиеся кластеры → C1..Ck
    remap = {old: i for i, old in enumerate(rest, start=1)}
    return np.array([remap.get(k, k) for k in new]), round(cut, 3)


def name_of(k) -> str:
    return {-1: 'noise', 0: 'IM', A: 'A', NP: 'NP', A_SLOW: 'A_slow'}.get(k, f'C{k}')


def summarize(video: str, df: pd.DataFrame, labels: np.ndarray, params: dict):
    n = len(labels)
    wide = {'video': video, 'n_tracks': n, 'n_clusters': len(set(labels) - {-1}),
            'min_cluster_size_used': params['min_cluster_size'],
            'noise_pct': round((labels == -1).mean() * 100, 1), 'im_cut': params.get('im_cut'), 'a_cut': params.get('a_cut')}
    long = []
    for k in sorted(set(labels)):
        m = labels == k
        sp, lf = df['x'].values[m], df['y'].values[m]
        nm = name_of(k)
        wide[f'{nm}_pct'] = round(m.mean() * 100, 1)
        long.append({'video': video, 'cluster': nm, 'n': int(m.sum()), 'pct': round(m.mean() * 100, 1),
                     f'{X_COL}_median': round(np.median(sp), 2), f'{X_COL}_q10': round(np.quantile(sp, 0.1), 2),
                     f'{X_COL}_q90': round(np.quantile(sp, 0.9), 2),
                     f'{Y_COL}_median': round(np.median(lf), 3), f'{Y_COL}_q10': round(np.quantile(lf, 0.1), 3),
                     f'{Y_COL}_q90': round(np.quantile(lf, 0.9), 3)})
    return wide, long

# =============================================================================
# 4. ГРАФИК
# =============================================================================

def plot(video: str, df: pd.DataFrame, labels: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    m = labels == -1
    ax.scatter(df['x'][m], df['y'][m], s=6, c='#bbbbbb', alpha=0.5, label=f'noise (n={m.sum()})')
    m = labels == 0
    if m.any():
        ax.scatter(df['x'][m], df['y'][m], s=8, c='#2c3e50', alpha=0.8, label=f'IM (n={m.sum()})')
    m = labels == A
    if m.any():
        ax.scatter(df['x'][m], df['y'][m], s=8, c='#c0392b', alpha=0.8, label=f'A (n={m.sum()})')
    cmap = plt.get_cmap('tab10')
    m = labels == NP
    if m.any():
        ax.scatter(df['x'][m], df['y'][m], s=8, c='#8e44ad', alpha=0.8, label=f'NP (n={m.sum()})')
    m = labels == A_SLOW
    if m.any():
        ax.scatter(df['x'][m], df['y'][m], s=8, c='#e67e22', alpha=0.8, label=f'A_slow (n={m.sum()})')
    for k in sorted(set(labels) - {-1, 0, A, NP, A_SLOW}):
        m = labels == k
        ax.scatter(df['x'][m], df['y'][m], s=8, color=cmap((k - 1) % 10), alpha=0.75, label=f'C{k} (n={m.sum()})')
        ax.annotate(f'C{k}', (df['x'][m].median(), df['y'][m].median()), fontsize=12, fontweight='bold',
                    ha='center', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    ax.set_xlabel(f'{X_COL} (px/frame)'); ax.set_ylabel(Y_COL)
    ax.set_title(video, fontsize=10); ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    safe = ''.join(ch if ch.isalnum() or ch in ' ._-' else '_' for ch in video)
    fig.savefig(out_dir / f'{safe}_scatter.png', dpi=150)
    plt.close(fig)

# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = load_videos()

    results = {}
    if MODE == 'pooled':
        names = list(videos)
        pooled = pd.concat([videos[v] for v in names], ignore_index=True)
        lab, params, prob = cluster(pooled)
        lab = assign_fast(pooled, lab, prob)
        lab, cut = merge_immotile(pooled, lab)
        lab = assign_np(pooled, lab)
        lab, a_cut = trim_fast(pooled, lab)
        params = {**params, 'a_cut': a_cut}
        params = {**params, 'im_cut': cut}
        start = 0
        for v in names:
            results[v] = (lab[start:start + len(videos[v])], params)
            start += len(videos[v])
    else:
        for v, d in videos.items():
            lab, params, prob = cluster(d)
            lab = assign_fast(d, lab, prob)
            lab, cut = merge_immotile(d, lab)
            if Z_COL and IM_FROM_2D:
                lab2, _, prob2 = cluster(d, use_z=False)
                lab2, cut = merge_immotile(d, assign_fast(d, lab2, prob2))
                im2 = lab2 == 0
                lab[(lab == 0) & ~im2] = -1          # не неподвижные по 2D-прогону → шум
                lab[im2 & (lab != A)] = 0            # неподвижные по 2D-прогону → IM
            lab = assign_np(d, lab)
            lab, a_cut = trim_fast(d, lab)
            results[v] = (lab, {**params, 'im_cut': cut, 'a_cut': a_cut})

    wide_rows, long_rows, tracks = [], [], []
    for v, (labels, params) in results.items():
        w, l = summarize(v, videos[v], labels, params)
        wide_rows.append(w); long_rows.extend(l)
        if SAVE_PLOTS:
            plot(v, videos[v], labels, out_dir)
        if SAVE_TRACK_LABELS:
            tracks.append(videos[v].assign(video=v, cluster=[name_of(k) for k in labels]))

    wide = pd.DataFrame(wide_rows)
    cl_cols = sorted([c for c in wide.columns if c.startswith('C') and c.endswith('_pct')], key=lambda c: int(c[1:-4]))
    wide = wide[['video', 'n_tracks', 'n_clusters', 'min_cluster_size_used', 'noise_pct', 'im_cut', 'a_cut']
                + [c for c in ['IM_pct', 'A_pct', 'A_slow_pct', 'NP_pct'] if c in wide] + cl_cols]
    params_df = pd.DataFrame([('X_COL', X_COL), ('Y_COL', Y_COL), ('Z_COL', Z_COL), ('WEIGHT_Z', WEIGHT_Z), ('IM_FROM_2D', IM_FROM_2D), ('MODE', MODE), ('SCALER', SCALER),
                              ('WEIGHT_X', WEIGHT_X), ('WEIGHT_Y', WEIGHT_Y), ('IM_MERGE', IM_MERGE), ('IM_QUANTILE', IM_QUANTILE), ('IM_REF', IM_REF), ('IM_LIMIT_BY_A', IM_LIMIT_BY_A),
                              ('A_MODE', A_MODE), ('A_PROB', A_PROB), ('A_QUANTILE', A_QUANTILE),
                              ('A_TRIM', A_TRIM), ('NP_MODE', NP_MODE), ('NP_QUANTILE', NP_QUANTILE), ('SOFT_ASSIGN', SOFT_ASSIGN)]
                             + [(f'hdbscan.{k}', v) for k, v in HDBSCAN_PARAMS.items()], columns=['parameter', 'value'])
    with pd.ExcelWriter(out_dir / 'summary_all_videos.xlsx') as xw:
        wide.to_excel(xw, sheet_name='per_video', index=False)
        pd.DataFrame(long_rows).to_excel(xw, sheet_name='per_cluster', index=False)
        params_df.to_excel(xw, sheet_name='params', index=False)
    if SAVE_TRACK_LABELS:
        pd.concat(tracks, ignore_index=True).to_excel(out_dir / 'tracks_labeled.xlsx', index=False)
    pd.set_option('display.width', 250)
    print(wide.to_string(index=False))
    print(f"\n[✓] {out_dir}")


if __name__ == '__main__':
    main()
