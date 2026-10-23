import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import NearestNeighbors
import hdbscan
import umap
from pathlib import Path

# =============================================================================
# ▶ НАСТРОЙКИ
# =============================================================================

FILE_LIST = [r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\15v link_g 30, link_sp_diff 0.85, link_m_d 100, thresh 160, col_dist 30, min_dur 20.xlsx",
    #r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\FIRST TERTILE 5 8 2 15 10 1 6.xlsx",
    #r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\SECOND TERTILE 14 18 4 13 20 12.xlsx",
    #r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\THIRD TERTILE 11 9 7 3 17 16 19.xlsx"
] #импорт файлов с данными

# ── HDBSCAN ───────────────────────────────────────────────────────────────────
MCS_FACTOR  = 0.5
MCS_MIN     = 10 #минимальное число точек в кластере
MCS_MAX     = 100 #максимальное число точек в кластере
MIN_SAMPLES = 5 #предположительно, что это

MCS_RETRY_SHRINK = 0.6 #фактор уменьшения mcs  циклах кластеризации
MCS_RETRY_MAX    = 6 #количество повторов найти 4 кластера в методе eom
MCS_HARD_FLOOR   = 5 #минимальный порог

# ── Веса признаков ────────────────────────────────────────────────────────────
WEIGHT_SPEED  = 2.0
WEIGHT_LFP    = 4.0
WEIGHT_LOG_DR = 0.5

MIN_SPEED_BOUNDARY = 4.0    # px/frame — граница speed не может быть ниже
MIN_LFP_BOUNDARY   = 0.30   # граница LFP не может быть ниже
MAX_SPEED_BOUNDARY = 12.0   # граница speed не выше (для редких быстрых видео)
MAX_LFP_BOUNDARY   = 0.65


POSTPROC_ENABLED      = True
SPEED_FLOOR_FACTOR    = 0.5   # доля от border_speed для отсечки
LFP_FLOOR_FACTOR      = 0.5

# ── kNN для шума ──────────────────────────────────────────
KNN_NEIGHBORS = 15

# ── UMAP ──────────────────────────────────────────────────────────────────────
UMAP_MIN_DIST       = 0.10
UMAP_MAX_POINTS     = 5000 #
UMAP_STRATIFIED     = True

# ── Визуализация 3D ──────────────────────────────────────────────────────────
PLOT_3D_LOG_Z         = True
PLOT_3D_CLIP_PERCENT  = 99.5
PLOT_3D_LOG_FLOOR     = 1.0

# ── Вывод графиков ────────────────────────────────────────────────────────────
SAVE_PLOTS    = True
SHOW_PLOTS    = False
OUTPUT_DIR    = r'D:\Lab\20 videos\Without grid\UNDERSTANDING\UMAP+HDBSCAN'

# Цвета
COLOR_RP = '#27ae60'
COLOR_SP = '#2980b9'
COLOR_NP = '#e67e22'
COLOR_IM = '#c0392b'
COLOR_N  = '#aaaaaa'

# =============================================================================
# 1. ЗАГРУЗКА
# =============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    df = df.copy()
    df['LFP']        = df['LINEARITY_OF_FORWARD_PROGRESSION']
    df['DIST_RATIO'] = (df['MAX_DISTANCE_TRAVELED']
                        / (df['TRACK_DISPLACEMENT'].replace(0, np.nan) + eps))
    df['LOG_DR']     = np.log1p(df['DIST_RATIO'])
    return df


def load_video(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=0)
    return compute_features(df)

# =============================================================================
# 2. ПОИСК ГРАНИЦЫ ЧЕРЕЗ KDE-ДОЛИНУ ПО ЦЕНТРОИДАМ
# =============================================================================

def find_valley(values: np.ndarray, # получаем значения центроидов
                search_lo: float, search_hi: float, # нижние и верхние значения квантилей каждой метрики
                bw=0.4, # параметр для kde
                n_grid: int = 300): #
    if len(values) < 3: #
        return None #
    grid = np.linspace(values.min() * 0.9, values.max() * 1.1, n_grid) # создаем сетку, с минимальными/максимальными нашими значениями, делим ее на 300 штук
    try: #
        kde_vals = gaussian_kde(values, bw_method=bw)(grid) # метод для получения красивых вероятностной кривой. Если значения встречаются редко в одной области, то спуск
    except Exception: #
        return None # ничего не возвращаем
    mins_idx = argrelmin(kde_vals, order=5)[0] # найти локальный минимум (то есть где вероятность меньше всего), причем соседей справа и слева по 5 штук
    valid = mins_idx[(grid[mins_idx] >= search_lo) & (grid[mins_idx] <= search_hi)] # если все ок по границам, у нас больше 0
    if len(valid) > 0: #
        return float(grid[valid[0]]) # возвращаем значения сетки, где меньше всего вероятность
    return None # ничего не возвращаем


def find_boundary(centroid_values: np.ndarray,
                  all_values: np.ndarray,
                  name: str,
                  min_bound: float = None,
                  max_bound: float = None) -> float:


    lo = np.percentile(all_values, 10) # находим 10 квантиль у всех параметров
    hi = np.percentile(all_values, 90) #

    found = None #
    source = '' #

    if len(centroid_values) >= 3: # если у нас есть больше = 3 центроидов (то есть кластеров)
        v = find_valley(centroid_values, lo, hi, bw=0.4) # находим среди центроидов минимальные значения (это странно, потому что центроиды это несколько пар значений по линейности и скорости)
        if v is not None: # если нашли минимальные значения
            found = v #
            source = 'KDE центроидов' #

    if found is None: # логика здесь есть, потому что находить минимумы вероятностей по всем точкам нужно, так как минимумы по вероятностям это и есть наши границы между кластерами
        sample = all_values # значения скоростей или линейности всех треков
        if len(sample) > 3000: #
            sample = np.random.default_rng(0).choice(sample, 3000, replace=False) #
        v = find_valley(sample, lo, hi, bw='scott') #
        if v is not None: #
            found = v #
            source = 'KDE всех точек' # переменная для print, чтобы понимать, по какому kde мы все нашли

    if found is None:
        found = float(np.median(centroid_values))
        source = 'медиана центроидов'

    original = found
    if min_bound is not None and found < min_bound: # если есть минимальная границы, и из KDE нашли более низкие значения минимума, то теперь минимум берем из нашего параметра
        found = min_bound #
        print(f"    Граница {name} ({source}): {original:.4f} → клип до MIN={min_bound:.3f}") #
    elif max_bound is not None and found > max_bound: #
        found = max_bound # но зачем идет проверка elif, если нам нужна проверка максимального значения через KDE всегда
        print(f"    Граница {name} ({source}): {original:.4f} → клип до MAX={max_bound:.3f}") #
    else: #
        print(f"    Граница {name} ({source}): {found:.4f}") #

    return float(found) # выводим значения

# =============================================================================
# 3. МАППИНГ КЛАСТЕРОВ → RP/SP/NP/IM
# =============================================================================

def map_clusters_to_abcd(df: pd.DataFrame,
                         raw_labels: np.ndarray):
    speed = df['TRACK_MEAN_SPEED'].values #получая значения скоростей
    lfp   = df['LFP'].values #значения линейности

    unique_cl = sorted(set(raw_labels) - {-1}) #получение количества кластеров без шума
    n_noise   = (raw_labels == -1).sum() #шум
    print(f"    HDBSCAN: {len(unique_cl)} кластеров, шум={n_noise}")

    if not unique_cl:
        return np.full(len(df), 'IM', dtype=object), {} #если нет кластеров, то просто называем immotile, это именно для кластеризации

    centroids = {}
    for c in unique_cl: #пробегаем по каждому кластеру, перед этим создаем словарь
        mask = raw_labels == c
        centroids[c] = {
            'speed': speed[mask].mean(),
            'lfp':   lfp[mask].mean(),
            'n':     mask.sum()
        } #нахождение центроидов (только две оси (линейность и скорость) кластеров с помощью среднего значения скорости и линейности

    c_speeds = np.array([centroids[c]['speed'] for c in unique_cl]) #извлекаем скорость и линейность
    c_lfps   = np.array([centroids[c]['lfp']   for c in unique_cl])

    print(f"    Центроиды speed: {np.round(c_speeds, 2)}")
    print(f"    Центроиды LFP:   {np.round(c_lfps, 3)}")

    # Границы с защитой от слишком низких/высоких значений
    b_speed = find_boundary(c_speeds, speed, 'speed',
                            min_bound=MIN_SPEED_BOUNDARY,
                            max_bound=MAX_SPEED_BOUNDARY)
    b_lfp   = find_boundary(c_lfps,   lfp,   'LFP',
                            min_bound=MIN_LFP_BOUNDARY,
                            max_bound=MAX_LFP_BOUNDARY)

    quadrants = {
        'RP': (+1, +1),
        'SP': (-1, +1),
        'NP': (+1, -1),
        'IM': (-1, -1),
    }

    def norm(s, l): #нормировка значений с учетом границ скорости и линейности
        ns = (s - b_speed) / (b_speed + 1e-9)
        nl = (l - b_lfp)   / (b_lfp   + 1e-9)
        return ns, nl

    available     = list(quadrants.keys()) #берем названия RP SP NP IM
    cluster_order = sorted(unique_cl, key=lambda c: centroids[c]['n'], reverse=True) #сортируем в порядке убывания размеров кластеров
    mapping = {}

    for cl in cluster_order:
        ns, nl = norm(centroids[cl]['speed'], centroids[cl]['lfp']) #нормировка моих значений
        if available:
            best = min(available,
                       key=lambda q: (ns - quadrants[q][0])**2 +
                                     (nl - quadrants[q][1])**2) #мини-функция внутри, которая находит евклидово расстояние, чтобы определить принадлежность к кластеру
            mapping[cl] = best #получается так, что q дает key, а 0 1 дают конкретные числа квадрантов. Минимальное расстояние до кластера значит наш кластер
            available.remove(best) #убираем название кластера с наилучшим совпадением
        else: #если кластеров больше 4, то оставшиеся кластеры становятся каким-либо из 4 кластеров
            best = min(quadrants,
                       key=lambda q: (ns - quadrants[q][0])**2 +
                                     (nl - quadrants[q][1])**2)
            mapping[cl] = best

    mapping_str = {c: f'{mapping[c]} (speed={centroids[c]["speed"]:.2f}, '
                      f'LFP={centroids[c]["lfp"]:.3f}, n={centroids[c]["n"]})'
                   for c in mapping}
    print(f"    Маппинг: {mapping_str}")

    result = np.full(len(df), 'NOISE', dtype=object) #создаем массив только со значеями NOISE, потом этот Шум становится нормальным кластером
    for cl, cat in mapping.items(): #cl - это сам кластер
        result[raw_labels == cl] = cat

    info = { #здесь просто дается словарь
        'boundary_speed': b_speed,
        'boundary_lfp':   b_lfp,
        'n_clusters':     len(unique_cl),
        'mapping':        mapping,
        'centroids':      centroids,
    }
    return result, info #возвращаем не только названные кластеры, но и информацию в словаре (краевые значения, количества кластеров, маппинг? и значения центроидов

# =============================================================================
# 4. POST-PROCESSING SANITY CHECK (НОВОЕ в v11)
# =============================================================================

def postprocess_sanity_check(df: pd.DataFrame, labels: np.ndarray,
                              b_speed: float, b_lfp: float) -> np.ndarray:

    if not POSTPROC_ENABLED:
        return labels

    speed = df['TRACK_MEAN_SPEED'].values
    lfp   = df['LFP'].values

    speed_floor = b_speed * SPEED_FLOOR_FACTOR
    lfp_floor   = b_lfp   * LFP_FLOOR_FACTOR

    print(f"  Post-processing: speed_floor={speed_floor:.2f}, lfp_floor={lfp_floor:.3f}")

    new_labels = labels.copy()
    changes = {'NP→IM': 0, 'RP→SP': 0, 'SP→IM': 0, 'RP→NP': 0}

    for i in range(len(labels)):
        cat = labels[i]
        s, l = speed[i], lfp[i]

        # Категории "быстрых" (NP, RP) с реально низкой скоростью → понижаем
        if cat == 'NP' and s < speed_floor:
            new_labels[i] = 'IM'
            changes['NP→IM'] += 1
        elif cat == 'RP' and s < speed_floor:
            new_labels[i] = 'SP'  # сохраняем высокий LFP
            changes['RP→SP'] += 1

        # Категории "высокого LFP" (SP, RP) с реально низким LFP → понижаем
        if cat == 'SP' and l < lfp_floor:
            new_labels[i] = 'IM'
            changes['SP→IM'] += 1
        elif cat == 'RP' and l < lfp_floor:
            new_labels[i] = 'NP'  # сохраняем высокий speed
            changes['RP→NP'] += 1

    total_changed = sum(changes.values())
    if total_changed > 0:
        print(f"  Sanity check: переброшено {total_changed} треков: {changes}")
    else:
        print(f"  Sanity check: нарушений не найдено")

    return new_labels

# =============================================================================
# 5. КЛАСТЕРИЗАЦИЯ ОДНОГО ВИДЕО
# =============================================================================

def absorb_noise(Xs: np.ndarray, labels: np.ndarray,
                 k: int = KNN_NEIGHBORS) -> np.ndarray:
    noise = labels == 'NOISE' #создаем массив, если там есть NOISE, то говорим что TRUE
    labeled = ~noise #названные кластеры - это не NOISE
    if not noise.any() or labeled.sum() < 2: #если нет в noise хоть одного TRUE или количество кластеров меньше 2
        if noise.any(): #
            labels[noise] = 'IM' # называем все что NOISE как IM
        return labels # возвращаем массив без шума
    k_eff = min(k, labeled.sum()) #задаем параметр минимального количества точек вокруг образца для отнесения образца к кластеру
    nbrs  = NearestNeighbors(n_neighbors=k_eff).fit(Xs[labeled]) # он упорядочивает данные, запоминает их
    _, idxs = nbrs.kneighbors(Xs[noise]) # учитывая расстояния, он дает индексы ближайщих соседей, но внутри данных без самих NOISE
    li = np.where(labeled)[0] # он дает глобальные индексы, зная локальный индекс
    new = labels.copy() #создаем копию названных кластеров
    for i, ni in enumerate(np.where(noise)[0]): #дает локальный номер шума, а потом
        nl = labels[li[idxs[i]]] #зная позицию шумовой точки глобально, дает категорию
        cats, counts = np.unique(nl, return_counts=True) #дает сами категории RP SP NP IM и их количество
        new[ni] = cats[np.argmax(counts)] # выводит то название кластера, которое встретилось больше всего
    return new # возвращаем массив без шума


def compute_adaptive_mcs(n: int) -> int: #посчитать минимальное количество точек в кластере
    mcs = int(np.sqrt(n) * MCS_FACTOR)
    return int(np.clip(mcs, MCS_MIN, MCS_MAX))


def run_hdbscan_with_retry(Xs: np.ndarray, n_total: int):
    mcs = compute_adaptive_mcs(n_total) #вычисляем с помощью функции количество точек в кластере (исправлю этот момент)
    print(f"    Стартовый min_cluster_size: {mcs} (n={n_total})")

    best_labels = None
    best_mcs    = mcs
    best_count  = 0
    best_method = 'eom'

    for attempt in range(MCS_RETRY_MAX): #проверяю hdbscan с помощью первого метода mcs retry max раз
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=MIN_SAMPLES,
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(Xs) #получается вектор numpy, где у каждого кластера свой номер, у шума -1
        n_found = len(set(labels) - {-1}) #убираем шум как кластер
        print(f"    [eom попытка {attempt+1}] mcs={mcs} → {n_found} кластеров")

        if n_found > best_count: #я нашел хоть какое-то количество кластеров, запоминаем настройки кластеризации
            best_labels = labels
            best_mcs    = mcs
            best_count  = n_found
            best_method = 'eom'

        if n_found >= 4:
            return labels, mcs, n_found, 'eom' #нашел отличные настройки для кластерации, сразу их выдаю.

        new_mcs = max(MCS_HARD_FLOOR, int(mcs * MCS_RETRY_SHRINK)) #нахожу заново mcs внутри цикла, чтобы изменить значение mcs, уменьшить его
        if new_mcs == mcs: #
            break
        mcs = new_mcs

    print(f"    [!] 'eom' не дал 4 кластеров, пробуем 'leaf'...")  #использую другой метод, но проверяю только 3 раза
    mcs = compute_adaptive_mcs(n_total) #заново считаем минимальное число точек в кластере
    for attempt in range(3):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=MIN_SAMPLES,
            cluster_selection_method='leaf',
            prediction_data=True
        )
        labels = clusterer.fit_predict(Xs) #кластеризую с помощью другого метода
        n_found = len(set(labels) - {-1}) #количество кластеров без шума
        print(f"    [leaf попытка {attempt+1}] mcs={mcs} → {n_found} кластеров")

        if n_found > best_count:  #нашел хоть какие-то настройки, поэтому я буду в конце их выводить
            best_labels = labels
            best_mcs    = mcs
            best_count  = n_found
            best_method = 'leaf'

        if n_found >= 4: #нашел идеальное количество кластеров, я их сразу запоминаю
            return labels, mcs, n_found, 'leaf'

        new_mcs = max(MCS_HARD_FLOOR, int(mcs * MCS_RETRY_SHRINK))
        if new_mcs == mcs:
            break
        mcs = new_mcs

    print(f"    [!] Лучший результат: {best_count} кластеров "
          f"(mcs={best_mcs}, method={best_method})")
    return best_labels, best_mcs, best_count, best_method #вывод настроек, которые дали хотя бы 1 кластер


def cluster_one_video(df: pd.DataFrame,
                      video_name: str):
    print(f"\n  [{video_name}] n={len(df)} треков")

    X = df[['TRACK_MEAN_SPEED', 'LFP', 'LOG_DR']].values.astype(float) # получение значений из определенных столюцов

    for col in range(X.shape[1]): # пробегаем все колонки (shape[1] дает количество колонок
        X[:, col] = np.clip(X[:, col], None, np.nanpercentile(X[:, col], 99.5)) # ограничиваем значения колонок до 99.5 персентиля (чтобы совсем большие значения были равными 99.5 персентилю)

    Xs = PowerTransformer(method='yeo-johnson').fit_transform(X) # стандартизуем данные с помощью библиотеки
    Xs[:, 0] *= WEIGHT_SPEED # умножаем каждую скорость на вес
    Xs[:, 1] *= WEIGHT_LFP # и тд
    Xs[:, 2] *= WEIGHT_LOG_DR #

    raw_labels, used_mcs, n_found, method = run_hdbscan_with_retry(Xs, len(df)) # получаю данные кластеризации. Знаю трек и его кластер,
    #настройки количества точек в кластере, количество всего кластеров, использованный метод кластеризации
    final, info = map_clusters_to_abcd(df, raw_labels) #знаю циферку каждого кластера, теперь я должен назначить каждому кластера название RP SP NP IM
    info['Xs']             = Xs # сам датасет с измененными значениями
    info['used_mcs']       = used_mcs # использованные настройки
    info['n_clusters_raw'] = n_found # количество кластеров
    info['method']         = method # информарция о методе

    # Поглощение шума
    final = absorb_noise(Xs, final) #убираем шум (идут датасет с измененными значениями и названиями кластеров для каждого трека

    # Sanity check (НОВОЕ в v11)
    final = postprocess_sanity_check(df, final, # проверяем правильность данных
                                     info['boundary_speed'],
                                     info['boundary_lfp'])

    return final, info #возвращаем саму таблицу с названиями кластеров и информацию о кластеризации

# =============================================================================
# 6. СТАТИСТИКА
# =============================================================================

def video_stats(df: pd.DataFrame, final: np.ndarray,
                video_name: str) -> dict:
    total = len(final)
    row   = {'video': video_name, 'n_total': total}
    print(f"\n  Результаты [{video_name}]:")
    print(f"  {'Кл':>3} | {'N':>5} | {'%':>5} | {'Speed':>6} | {'LFP':>6} | {'DIST_R':>7}")
    print(f"  {'-'*45}")
    for cat in ['RP', 'SP', 'NP', 'IM']:
        m   = final == cat
        cnt = m.sum()
        pct = cnt / total * 100
        row[f'n_{cat}']   = cnt
        row[f'pct_{cat}'] = round(pct, 1)
        if cnt > 0:
            sub = df[m]
            row[f'speed_{cat}'] = round(sub['TRACK_MEAN_SPEED'].mean(), 2)
            row[f'lfp_{cat}']   = round(sub['LFP'].mean(), 3)
            row[f'dr_{cat}']    = round(sub['DIST_RATIO'].mean(), 2)
            print(f"  {cat:>3} | {cnt:>5} | {pct:>4.1f}% | "
                  f"{row[f'speed_{cat}']:>6.2f} | "
                  f"{row[f'lfp_{cat}']:>6.3f} | "
                  f"{row[f'dr_{cat}']:>7.2f}")
        else:
            row[f'speed_{cat}'] = None
            row[f'lfp_{cat}']   = None
            row[f'dr_{cat}']    = None
            print(f"  {cat:>3} | {'0':>5} | {'0.0%':>5} | {'—':>6} | {'—':>6} | {'—':>7}")
    return row

# =============================================================================
# 7. СТРАТИФИЦИРОВАННЫЙ СУБСЭМПЛИНГ ДЛЯ UMAP
# =============================================================================

def stratified_subsample(labels: np.ndarray, max_total: int, # labels — названия кластеров каждого трека, max_total — бюджет точек для UMAP
                         seed: int = 42) -> np.ndarray: # seed фиксирует случайность — при каждом запуске выборка будет одинаковой
    rng = np.random.default_rng(seed) # генератор случайных чисел
    unique = np.unique(labels) # массив неповторяющихся названий кластеров: ['IM', 'NP', 'RP', 'SP']
    n_clusters = len(unique) # количество кластеров
    if n_clusters == 0: # если кластеров нет
        return np.array([], dtype=int) # возвращаем пустой массив

    quota_per = max_total // n_clusters # 5000 точек делим поровну — квота мест на один кластер
    selected  = [] # список массивов индексов отобранных треков (по одному массиву на кластер)
    deficit   = 0 # накопленный остаток: сколько мест не использовали маленькие кластеры

    big_clusters = [] # кластеры, которые больше квоты — им достанется бонус из deficit
    for cat in unique: # пробегаем по названиям кластеров
        idx = np.where(labels == cat)[0] # массив позиций (номеров строк) всех треков этого кластера, например [3, 7, 12, 25, ...]
        if len(idx) <= quota_per: # если кластер маленький (треков меньше квоты) — берём его целиком
            selected.append(idx) # добавляем все индексы этого кластера
            deficit += quota_per - len(idx) # неиспользованные места копим — отдадим большим кластерам
        else:
            big_clusters.append((cat, idx)) # большой кластер: запоминаем его название и индексы треков

    if big_clusters:  # если есть большие кластеры — раздаём им бонус из накопленного остатка
        extra_per = deficit // len(big_clusters) # бонус на каждый большой кластер поровну
        for cat, idx in big_clusters:
            take = min(len(idx), quota_per + extra_per) # берём оставшиеся места, но не больше размера кластера
            chosen = rng.choice(idx, take, replace=False) # случайно выбираем take индексов без повторений
            selected.append(chosen) # добавляем в итоговый список

    return np.concatenate(selected) # selected — список из нескольких массивов; concatenate склеивает их в один плоский массив индексов

# =============================================================================
# 8. ВИЗУАЛИЗАЦИЯ
# =============================================================================

def _scatter_abcd(ax, px, py, labels, xl, yl, title, s=8):
    for cat, col in [('NOISE', COLOR_N), ('IM', COLOR_IM),
                     ('NP', COLOR_NP), ('SP', COLOR_SP), ('RP', COLOR_RP)]:
        m = labels == cat
        if m.sum() == 0:
            continue
        lbl = 'Шум' if cat == 'NOISE' else f'CLUSTER {cat}'
        ax.scatter(px[m], py[m], c=col, s=s, alpha=0.75,
                   label=f'{lbl} (n={m.sum()})', rasterized=True)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(fontsize=7, markerscale=1.5)


def adaptive_umap_params(n: int) -> dict:
    if n < 3000:
        nn = 15
    elif n < 10000:
        nn = 30
    else:
        nn = 50
    return {
        'n_neighbors': nn,
        'min_dist':    UMAP_MIN_DIST,
    }


def prepare_z_for_3d(z_raw: np.ndarray) -> tuple[np.ndarray, str, str]:
    z = z_raw.copy()
    clip_val = np.nanpercentile(z, PLOT_3D_CLIP_PERCENT)
    z = np.clip(z, None, clip_val)
    if PLOT_3D_LOG_Z:
        z = np.maximum(z, PLOT_3D_LOG_FLOOR)
        return z, 'log', 'DIST_RATIO (log scale)'
    else:
        return z, 'linear', 'DIST_RATIO'


def plot_video(df: pd.DataFrame, final: np.ndarray, #
               info: dict, video_name: str, out_dir: Path): #датасет с измененными значениями
    x = df['TRACK_MEAN_SPEED'].values #
    y = df['LFP'].values #
    z_raw = df['DIST_RATIO'].values #

    # ── UMAP ──────────────────────────────────────────────────────────────────
    Xs = info['Xs'] # измененный датасет с названиями кластеров
    n  = len(df) #количество строк = количество треков

    if n > UMAP_MAX_POINTS: #если треков больше, чем порог в 5000
        if UMAP_STRATIFIED: #всегда запускается еще один цикл
            idx_sub = stratified_subsample(final, UMAP_MAX_POINTS, seed=42) # стратифицированный субсэмплинг
            print(f"  UMAP: стратифицированная подвыборка {len(idx_sub)} из {n}") # вывод коилчества точек для umap из всего количества треков
        else: #
            rng     = np.random.default_rng(42) # случайные числа (будущие индексы)
            idx_sub = rng.choice(n, UMAP_MAX_POINTS, replace=False) # случано отбираем индексы из всего количества треков
            print(f"  UMAP: случайная подвыборка {len(idx_sub)} из {n}") #
    else: # если количество треков меньше 5000, то мы берем весь массив
        idx_sub = np.arange(n) #
        print(f"  UMAP: все {n} точек") #

    Xs_sub    = Xs[idx_sub] # массив измененных значений
    final_sub = final[idx_sub] # названия выбранных треков

    umap_p = adaptive_umap_params(len(idx_sub)) # подбираем параметры umap в зависимости от количества точек (возможно максимум 5000)

    reducer = umap.UMAP( #запускаем
        n_neighbors=min(umap_p['n_neighbors'], len(idx_sub) - 1), # смотрим либо на глобальную структуру, либо на локальную
        min_dist=umap_p['min_dist'], # нужно для расстояния между точками на 2d внутри кластеров
        n_components=2, # количество получаемых компонентов
        random_state=42, # случайное число для воспроизводимости
        verbose=False, # не загружает результаты обучения в консоль
        low_memory=True # анализ файла целиком будет медленным
    )
    emb = reducer.fit_transform(Xs_sub) # обучение UMAP и возвращение координат в 2d модели

    # ── График 1: UMAP 2D ─────────────────────────────────────────────────────
    fig_umap, ax_umap = plt.subplots(figsize=(7, 6)) #
    _scatter_abcd(ax_umap, emb[:, 0], emb[:, 1], final_sub, # дает само место строить график, значения осей umap, названия кластера каждого трека
                  'UMAP-1', 'UMAP-2', # оси
                  f'{video_name}\nUMAP  (n={len(idx_sub)})', #
                  s=10) # размер точек на UMAP
    ax_umap.legend(fontsize=14, markerscale=2, framealpha=0.85, # настройки изображения
                   loc='best', title='CLUSTER', title_fontsize=9) #
    fig_umap.tight_layout() # корректирует и правильно ставит части изображения в окне
    if SAVE_PLOTS: # всегда запускаем цикл, чтобы сохранить график в виде файла
        p = out_dir / f'{video_name}_umap.png' # директорий
        fig_umap.savefig(p, dpi=200, bbox_inches='tight') # сохраняем в директории
        print(f"  [✓] UMAP: {p}") #
    if SHOW_PLOTS: # не запускаем цикл по показу umap
        plt.show() # показ изображения
    plt.close(fig_umap) #

    # ── График 2: 3D scatter с лог-осью Z ─────────────────────────────────────
    z_plot, z_scale, z_label = prepare_z_for_3d(z_raw)

    fig_3d = plt.figure(figsize=(7, 6))
    ax3d = fig_3d.add_subplot(111, projection='3d')

    for cat, col in [('NOISE', COLOR_N), ('IM', COLOR_IM), ('NP', COLOR_NP),
                     ('SP', COLOR_SP), ('RP', COLOR_RP)]:
        m = final == cat
        if m.sum() == 0:
            continue
        ax3d.scatter(x[m], y[m], z_plot[m], c=col, s=10, alpha=0.75,
                     label=f'{"Шум" if cat == "NOISE" else "CLUSTER " + cat} (n={m.sum()})',
                     rasterized=True)

    ax3d.set_xlabel('SPEED (px/frame)', fontsize=9, labelpad=6)
    ax3d.set_ylabel('LFP', fontsize=9, labelpad=6)
    ax3d.set_zlabel(z_label, fontsize=9, labelpad=6)

    if z_scale == 'log':
        ax3d.set_zscale('log')

    ax3d.tick_params(labelsize=7)
    ax3d.legend(fontsize=8, markerscale=2, framealpha=0.85,
                loc='upper left', title='CLUSTER', title_fontsize=8)
    ax3d.set_title(f'{video_name}\nSpeed × LFP × DIST_RATIO', fontsize=10)
    fig_3d.tight_layout()
    if SAVE_PLOTS:
        p = out_dir / f'{video_name}_3d.png'
        fig_3d.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  [✓] 3D: {p}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_3d)

    # ── График 3: таблица ─────────────────────────────────────────────────────
    col_map = {'RP': COLOR_RP, 'SP': COLOR_SP, 'NP': COLOR_NP, 'IM': COLOR_IM}
    hdr  = ['CLUSTER', 'N', '%', 'MEAN SPEED\n(px/fr)', 'LFP', 'DIST_R']
    rows = []
    total   = len(final)
    noise_n = (final == 'NOISE').sum()
    if noise_n:
        rows.append(['Шум', str(noise_n), f'{noise_n/total*100:.1f}', '—', '—', '—'])
    for cat in ['RP', 'SP', 'NP', 'IM']:
        m   = final == cat
        cnt = m.sum()
        if cnt == 0:
            rows.append([cat, '0', '0.0', '—', '—', '—'])
            continue
        sub = df[m]
        rows.append([cat,
                     str(cnt),
                     f'{cnt/total*100:.1f}',
                     f'{sub["TRACK_MEAN_SPEED"].mean():.2f}',
                     f'{sub["LFP"].mean():.3f}',
                     f'{sub["DIST_RATIO"].mean():.2f}'])

    n_rows = len(rows)
    fig_tbl, ax_t = plt.subplots(figsize=(7, 0.55 * (n_rows + 1.8)))
    ax_t.axis('off')

    tbl = ax_t.table(cellText=rows, colLabels=hdr,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for j in range(len(hdr)):
        cell = tbl[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.18)

    for i, row in enumerate(rows):
        clr = col_map.get(row[0], '#888888')
        for j in range(len(hdr)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(clr + '40')
            cell.set_height(0.14)

    col_widths = [0.22, 0.13, 0.13, 0.18, 0.13, 0.15]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            tbl[i, j].set_width(w)

    ax_t.set_title(f'{video_name} — статистика кластеров',
                   fontsize=11, fontweight='bold', pad=14)
    fig_tbl.tight_layout()
    if SAVE_PLOTS:
        p = out_dir / f'{video_name}_table.png'
        fig_tbl.savefig(p, dpi=200, bbox_inches='tight')
        print(f"  [✓] Таблица: {p}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig_tbl)

# =============================================================================
# 9. ЭКСПОРТ
# =============================================================================

def export_results(df: pd.DataFrame, final: np.ndarray,
                   video_name: str, out_dir: Path):
    out = df.copy()
    out['CLUSTER_ABCD'] = final
    path = out_dir / f'{video_name}_clustered.xlsx'
    out.to_excel(path, index=False)
    print(f"  [✓] Таблица: {path}")


def export_summary(summary_rows: list, out_dir: Path):
    smry = pd.DataFrame(summary_rows)
    path = out_dir / 'summary_all_videos.xlsx'
    smry.to_excel(path, index=False)
    print(f"\n[✓] Сводная таблица: {path}")

    print(f"\n{'='*75}")
    print(f"  СВОДКА ПО ВСЕМ ВИДЕО")
    print(f"{'='*75}")
    print(f"  {'Видео':<30} | {'RP%':>5} | {'SP%':>5} | {'NP%':>5} | {'IM%':>5} | {'N':>5}")
    print(f"  {'-'*75}")
    for r in summary_rows:
        print(f"  {r['video']:<30} | "
              f"{r.get('pct_RP', 0):>5.1f} | "
              f"{r.get('pct_SP', 0):>5.1f} | "
              f"{r.get('pct_NP', 0):>5.1f} | "
              f"{r.get('pct_IM', 0):>5.1f} | "
              f"{r.get('n_total', 0):>5}")
    print(f"{'='*75}")

# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for filepath in FILE_LIST:
        video_name = Path(filepath).stem
        print(f"\n{'='*60}")
        print(f"  Обработка: {video_name}")
        print(f"{'='*60}")

        try:
            df = load_video(filepath)
            df = df.dropna(subset=['TRACK_MEAN_SPEED', 'LFP', 'DIST_RATIO']).reset_index(drop=True)

            if len(df) < 20: #проверка, что треков достаточно много
                print(f"  [!] Слишком мало треков ({len(df)}), пропускаем")
                continue

            final, info = cluster_one_video(df, video_name)
            row = video_stats(df, final, video_name)
            summary_rows.append(row)

            if SAVE_PLOTS:
                plot_video(df, final, info, video_name, out_dir)

            export_results(df, final, video_name, out_dir)

        except Exception as e:
            print(f"  [ОШИБКА] {video_name}: {e}")
            import traceback
            traceback.print_exc()

    if summary_rows:
        export_summary(summary_rows, out_dir)

    print("\nГотово!")


if __name__ == '__main__':
    main()
