import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# ==============================================================================
#  ПУТИ К ПАПКАМ
# ==============================================================================

SPOTS_DIR  = r"D:\Lab\20 videos\Without grid\BEFORE FILTER SPOTS\2 video new"
TRACKS_DIR = r"D:\Lab\20 videos\Without grid\BEFORE FILTER\2 video new"
OUTPUT_DIR = r"D:\Lab\20 videos\Without grid\FILTERED DATA\3 4 5 new video\2v link_g 30, link_sp_diff 0.85, link_m_d 100, thresh 160, col_dist 30, min_dur 20"

SPOTS_SUFFIX  = "_allspots"
TRACKS_SUFFIX = "_tracks"
OUTPUT_SUFFIX = "proc"

# ==============================================================================
#  НАСТРОЙКИ — ЭТАП 1: СШИВКА (NOISE LINKING)
# ==============================================================================

LINK_MAX_GAP       = 30   # Макс. пропуск кадров между фрагментами
LINK_SPEED_DIFF    = 0.85 # Макс. относительная разница средних скоростей (60%)
BASE_SEARCH_RADIUS = 25   # Радиус поиска при gap=0 (пикселей)
MAX_SPEED_PX       = 25   # Физический предел скорости (px/кадр) — задаёт динамический радиус
LINK_MAX_DIST_ABS  = 100   # Жёсткий потолок расстояния (px) — никогда не сшиваем дальше,
                          # даже если gap большой и dynamic_max_dist это позволяет
MIN_LINK_SPEED     = 1.0  # Мин. средняя скорость фрагмента (px/кадр) — фрагменты
                          # медленнее этого считаются стоячими объектами и не сшиваются
MIN_LINK_SPOTS     = 2    # Мин. число спотов в каждом фрагменте — короткие огрызки
                          # (2-3 спота) имеют ненадёжную скорость и не сшиваются

# ==============================================================================
#  НАСТРОЙКИ — ЭТАП 2: РАЗРЕЗ У ГРАНИЦ
# ==============================================================================

GAP_SCALE_PAR = 2
BORDER_X_MIN  = 20
BORDER_X_MAX  = 700
BORDER_Y_MIN  = 15
BORDER_Y_MAX  = 560
THRESHOLD     = 160
BORDER_VELOCITY_TAIL = 3

# ==============================================================================
#  НАСТРОЙКИ — ЭТАП 3: ДЕТЕКЦИЯ И СВАП КОЛЛИЗИЙ
# ==============================================================================

COLLISION_DISTANCE_THRESHOLD = 30
MIN_FRAGMENT_SIZE   = 2
SHORT_FRAGMENT_SIZE = 4
SHORT_POST_SPOTS    = 7
LONG_POST_SPOTS     = 20
VELOCITY_TAIL       = 3
CROSS_SPEED_THRESHOLD = 0.5

'''# ==============================================================================
#  НАСТРОЙКИ — ЭТАП 3.5: ТЕЛЕПОРТ-РАЗРЕЗЫ
# ==============================================================================

TELEPORT_SPEED_THRESHOLD  = 35
TELEPORT_MAX_FRAME_STEP   = 1000
TELEPORT_MEAN_MULTIPLIER  = 3.5'''

# ==============================================================================
#  НАСТРОЙКИ — ЭТАП 4: ФИЛЬТРАЦИЯ ПО ДЛИНЕ
# ==============================================================================

MIN_TRACK_DURATION = 20
MIN_SPOTS = 20

# ==============================================================================
#  ПОИСК ПАР ФАЙЛОВ
# ==============================================================================

def extract_prefix(filename: str, suffix: str) -> str:
    stem = Path(filename).stem
    if suffix in stem:
        stem = stem[:stem.index(suffix)]
    if '(' in stem:
        stem = stem[:stem.index('(')]
    return stem.strip()


def find_video_pairs(spots_dir: str, tracks_dir: str):
    spots_files  = {extract_prefix(f.name, SPOTS_SUFFIX):  f
                    for f in Path(spots_dir).glob("*.xlsx")
                    if SPOTS_SUFFIX in f.name and not f.name.startswith('~$')}
    tracks_files = {extract_prefix(f.name, TRACKS_SUFFIX): f
                    for f in Path(tracks_dir).glob("*.xlsx")
                    if TRACKS_SUFFIX in f.name and not f.name.startswith('~$')}

    all_prefixes = set(spots_files) | set(tracks_files)
    pairs, missing = [], []

    for prefix in sorted(all_prefixes):
        if prefix not in spots_files:
            missing.append(f"  нет файла спотов для префикса '{prefix}'")
            continue
        if prefix not in tracks_files:
            missing.append(f"  нет файла треков для префикса '{prefix}'")
            continue
        pairs.append({'prefix': prefix,
                      'spots':  spots_files[prefix],
                      'tracks': tracks_files[prefix]})

    if missing:
        raise FileNotFoundError("Не удалось сопоставить все файлы:\n" + "\n".join(missing))

    return pairs


# ==============================================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def force_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace(",", ".", regex=False)
                     .str.replace(" ", "", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fmt(v):
    return f"{v:.3f}" if v is not None else "~"


def point_distance(x1, y1, x2, y2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def calc_velocity_vector(df_frag, tail=VELOCITY_TAIL, use_tail=True):
    df_frag = df_frag.sort_values('FRAME').reset_index(drop=True)
    if len(df_frag) < 2:
        return np.array([0.0, 0.0])
    sub = df_frag.tail(tail) if use_tail else df_frag.head(tail)
    dx = sub['POSITION_X'].diff().dropna()
    dy = sub['POSITION_Y'].diff().dropna()
    return np.array([dx.mean(), dy.mean()])


def calc_mean_speed(df_frag, frame1=None, frame2=None):
    df = df_frag.copy()
    if frame1 is not None:
        df = df[df['FRAME'] >= frame1]
    if frame2 is not None:
        df = df[df['FRAME'] <= frame2]
    df = df.sort_values('FRAME').reset_index(drop=True)
    if len(df) < 2:
        return None
    coords = df[['POSITION_X', 'POSITION_Y']].values
    diffs = np.diff(coords, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    delta_frames = np.diff(df['FRAME'].values).astype(float)
    valid = delta_frames == 1
    if valid.any():
        return float((distances[valid] / delta_frames[valid]).mean())
    return float((distances / delta_frames).mean())


def calc_straight_line_speed(df_frag, frame1=None, frame2=None):
    df = df_frag.copy()
    if frame1 is not None:
        df = df[df['FRAME'] >= frame1]
    if frame2 is not None:
        df = df[df['FRAME'] <= frame2]
    df = df.sort_values('FRAME').reset_index(drop=True)
    if len(df) < 2:
        return None
    x1, y1 = df.iloc[0]['POSITION_X'], df.iloc[0]['POSITION_Y']
    x2, y2 = df.iloc[-1]['POSITION_X'], df.iloc[-1]['POSITION_Y']
    displacement = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    duration = int(df.iloc[-1]['FRAME'] - df.iloc[0]['FRAME'])
    if duration == 0:
        return None
    return displacement / duration


def calc_straightness(df_frag, frame1=None, frame2=None):
    df = df_frag.copy()
    if frame1 is not None:
        df = df[df['FRAME'] >= frame1]
    if frame2 is not None:
        df = df[df['FRAME'] <= frame2]
    df = df.sort_values('FRAME').reset_index(drop=True)
    if len(df) < 2:
        return None
    coords = df[['POSITION_X', 'POSITION_Y']].values
    diffs = np.diff(coords, axis=0)
    total_dist = float(np.sqrt((diffs ** 2).sum(axis=1)).sum())
    if total_dist == 0:
        return None
    x1, y1 = coords[0, 0], coords[0, 1]
    x2, y2 = coords[-1, 0], coords[-1, 1]
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / total_dist


def calc_displacement(df_frag, frame1=None, frame2=None):
    df = df_frag.copy()
    if frame1 is not None:
        df = df[df['FRAME'] >= frame1]
    if frame2 is not None:
        df = df[df['FRAME'] <= frame2]
    df = df.sort_values('FRAME').reset_index(drop=True)
    if len(df) < 2:
        return None
    x1, y1 = df.iloc[0]['POSITION_X'], df.iloc[0]['POSITION_Y']
    x2, y2 = df.iloc[-1]['POSITION_X'], df.iloc[-1]['POSITION_Y']
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def speed_diff(spd1, spd2):
    if spd1 is None or spd2 is None:
        return 0.0
    return abs(spd2 - spd1)


def calc_directional_change_rate(group):
    group = group.sort_values('FRAME').reset_index(drop=True)
    frames = group['FRAME'].values
    coords = group[['POSITION_X', 'POSITION_Y']].values
    delta_frames = np.diff(frames)
    diffs = np.diff(coords, axis=0)

    if len(diffs) < 2:
        return None

    valid = delta_frames == 1
    angles = []
    for i in range(1, len(diffs)):
        if valid[i - 1] and valid[i]:
            angles.append(np.radians(angle_between(diffs[i - 1], diffs[i])))

    return float(np.mean(angles)) if angles else None


def calc_track_stats(group):
    group = group.sort_values('FRAME').reset_index(drop=True)

    # Убираем дубли кадров — два спота на одном кадре дают delta_frames=0 → inf скорость.
    # Дубли появляются когда сшивка соединяет треки с совпадающим крайним кадром.
    # Оставляем первый спот из каждого кадра.
    dupes = group.duplicated(subset='FRAME', keep='first')
    if dupes.any():
        print(f"  [calc_track_stats] трек {group['TRACK_ID'].iloc[0]}: "
              f"удалено {dupes.sum()} дублей кадров {sorted(group.loc[dupes,'FRAME'].tolist())}")
        group = group[~dupes].reset_index(drop=True)

    track_id    = int(group['TRACK_ID'].iloc[0])
    track_start = int(group['FRAME'].min())
    track_stop  = int(group['FRAME'].max())
    track_dur   = track_stop - track_start

    coords = group[['POSITION_X', 'POSITION_Y']].values
    diffs  = np.diff(coords, axis=0)
    distances  = np.sqrt((diffs ** 2).sum(axis=1))
    total_dist = float(distances.sum()) if len(distances) > 0 else 0.0

    frames = group['FRAME'].values
    delta_frames = np.diff(frames).astype(float)

    if len(distances) > 0:
        # delta_frames==0 (дубль кадра) даёт inf — заменяем на nan и исключаем из статистики
        with np.errstate(divide='ignore', invalid='ignore'):
            step_speeds = distances / delta_frames
        step_speeds = np.where(delta_frames == 0, np.nan, step_speeds)
        valid_steps = step_speeds[np.isfinite(step_speeds)]
        if len(valid_steps) > 0:
            mean_speed   = float(np.mean(valid_steps))
            max_speed    = float(np.max(valid_steps))
            min_speed    = float(np.min(valid_steps))
            median_speed = float(np.median(valid_steps))
            std_speed    = float(np.std(valid_steps))
        else:
            mean_speed = max_speed = min_speed = median_speed = std_speed = np.nan
    else:
        mean_speed = max_speed = min_speed = median_speed = std_speed = np.nan

    displacement = float(np.sqrt(
        (coords[-1, 0] - coords[0, 0]) ** 2 +
        (coords[-1, 1] - coords[0, 1]) ** 2
    )) if len(coords) > 1 else 0.0

    confinement_ratio        = round(displacement / total_dist, 4) if total_dist > 0 else None
    mean_straight_line_speed = round(displacement / track_dur, 6)  if track_dur > 0  else None
    linearity = (round(displacement / total_dist, 4) if total_dist != 0 else None)

    mean_dcr = calc_directional_change_rate(group)

    gaps        = delta_frames[delta_frames > 1]
    number_gaps = int(len(gaps))
    longest_gap = int(gaps.max() - 1) if number_gaps > 0 else 0

    mean_quality = (round(float(group['QUALITY'].mean()), 4)
                    if 'QUALITY' in group.columns else None)
    track_x = round(float(group['POSITION_X'].mean()), 4)
    track_y = round(float(group['POSITION_Y'].mean()), 4)
    track_z = (round(float(group['POSITION_Z'].mean()), 4)
               if 'POSITION_Z' in group.columns else None)

    if len(coords) > 1:
        from_start        = coords - coords[0]
        dists_start       = np.sqrt((from_start ** 2).sum(axis=1))
        max_dist_traveled = round(float(dists_start.max()), 4)
    else:
        max_dist_traveled = 0.0

    subject_id   = group['SubjectID'].iloc[0]   if 'SubjectID'    in group.columns else None
    id_ejaculate = group['ID_Ejaculate'].iloc[0] if 'ID_Ejaculate' in group.columns else None
    frame_range  = group['Frame_range'].iloc[0]  if 'Frame_range'  in group.columns else f"{track_start}-{track_stop}"

    if number_gaps > 0:
        old_valid  = delta_frames == 1
        old_speeds = distances[old_valid] / delta_frames[old_valid] if old_valid.any() else np.array([np.nan])
        print(f"Track {track_id}: gaps={number_gaps}, "
              f"old_mean={np.nanmean(old_speeds):.4f}, "
              f"new_mean={mean_speed:.4f}, "
              f"diff={abs(mean_speed - np.nanmean(old_speeds)):.6f}")

    return {
        'LABEL': f"Track_{track_id}",
        'TRACK_INDEX': track_id,
        'TRACK_ID': track_id,
        'SubjectID': subject_id,
        'ID_Ejaculate': id_ejaculate,
        'Frame_range': frame_range,
        'NUMBER_SPOTS': len(group),
        'NUMBER_GAPS': number_gaps,
        'NUMBER_SPLITS': 0,
        'NUMBER_MERGES': 0,
        'NUMBER_COMPLEX': 0,
        'LONGEST_GAP': longest_gap,
        'TRACK_DURATION': track_dur,
        'TRACK_START': track_start,
        'TRACK_STOP': track_stop,
        'TRACK_DISPLACEMENT': round(displacement, 4),
        'TRACK_X_LOCATION': track_x,
        'TRACK_Y_LOCATION': track_y,
        'TRACK_Z_LOCATION': track_z,
        'TRACK_MEAN_SPEED': round(mean_speed, 6),
        'TRACK_MAX_SPEED': round(max_speed, 6),
        'TRACK_MIN_SPEED': round(min_speed, 6),
        'TRACK_MEDIAN_SPEED': round(median_speed, 6),
        'TRACK_STD_SPEED': round(std_speed, 6),
        'TRACK_MEAN_QUALITY': mean_quality,
        'TOTAL_DISTANCE_TRAVELED': round(total_dist, 4),
        'MAX_DISTANCE_TRAVELED': max_dist_traveled,
        'CONFINEMENT_RATIO': confinement_ratio,
        'MEAN_STRAIGHT_LINE_SPEED': mean_straight_line_speed,
        'LINEARITY_OF_FORWARD_PROGRESSION': linearity,
        'MEAN_DIRECTIONAL_CHANGE_RATE': round(mean_dcr, 4) if mean_dcr is not None else None,
    }


TRACKS_COLUMNS = [
    'LABEL', 'TRACK_INDEX', 'TRACK_ID', 'SubjectID', 'ID_Ejaculate', 'Frame_range',
    'NUMBER_SPOTS', 'NUMBER_GAPS', 'NUMBER_SPLITS', 'NUMBER_MERGES', 'NUMBER_COMPLEX',
    'LONGEST_GAP', 'TRACK_DURATION', 'TRACK_START', 'TRACK_STOP', 'TRACK_DISPLACEMENT',
    'TRACK_X_LOCATION', 'TRACK_Y_LOCATION', 'TRACK_Z_LOCATION',
    'TRACK_MEAN_SPEED', 'TRACK_MAX_SPEED', 'TRACK_MIN_SPEED',
    'TRACK_MEDIAN_SPEED', 'TRACK_STD_SPEED', 'TRACK_MEAN_QUALITY',
    'TOTAL_DISTANCE_TRAVELED', 'MAX_DISTANCE_TRAVELED', 'CONFINEMENT_RATIO',
    'MEAN_STRAIGHT_LINE_SPEED', 'LINEARITY_OF_FORWARD_PROGRESSION',
    'MEAN_DIRECTIONAL_CHANGE_RATE',
]


# ==============================================================================
#  ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ОДНОГО ВИДЕО
# ==============================================================================

def process_video(spots_path: Path, tracks_path: Path, output_path: Path, prefix: str):

    print(f"\n{'=' * 80}")
    print(f"  ОБРАБОТКА: {prefix}")
    print(f"  Споты:  {spots_path.name}")
    print(f"  Треки:  {tracks_path.name}")
    print(f"  Выход:  {output_path.name}")
    print(f"{'=' * 80}")

    # --------------------------------------------------------------------------
    #  ЗАГРУЗКА ДАННЫХ
    # --------------------------------------------------------------------------

    print("\n--- ЗАГРУЗКА ДАННЫХ ---")

    df_spots  = pd.read_excel(spots_path)
    df_tracks = pd.read_excel(tracks_path)

    num_spots_cols  = ['POSITION_X', 'POSITION_Y', 'TRACK_ID', 'FRAME', 'QUALITY', 'POSITION_Z']
    num_tracks_cols = ['TRACK_ID', 'TRACK_START', 'TRACK_STOP', 'TRACK_DURATION', 'TRACK_MEAN_SPEED']
    df_spots  = force_numeric(df_spots,  num_spots_cols)
    df_tracks = force_numeric(df_tracks, num_tracks_cols)

    df_spots = df_spots.dropna(subset=['TRACK_ID', 'FRAME', 'POSITION_X', 'POSITION_Y'])
    df_spots['TRACK_ID'] = df_spots['TRACK_ID'].astype(int)
    df_spots['FRAME']    = df_spots['FRAME'].astype(int)

    print(f"Загружено: {len(df_spots)} спотов, {df_spots['TRACK_ID'].nunique()} треков")

    # --------------------------------------------------------------------------
    #  ЭТАП 1: СШИВКА ТРЕКОВ (NOISE LINKING) — ИСПРАВЛЕННЫЙ
    # --------------------------------------------------------------------------

    print("\n--- ЭТАП 1: СШИВКА ТРЕКОВ ---")

    def _track_mean_speed(group):
        g = group.sort_values('FRAME')
        dx = g['POSITION_X'].diff()
        dy = g['POSITION_Y'].diff()
        dt = g['FRAME'].diff()
        dist = np.sqrt(dx ** 2 + dy ** 2)
        spd = dist / dt
        spd = spd.mask(dt == 0, np.nan)
        spd = spd.replace([np.inf, -np.inf], np.nan)
        return spd.mean(skipna=True)

    # --- ПРЕДВЫЧИСЛЕНИЯ (O(N) - делается ОДИН раз) ---
    link_speed_map = df_spots.groupby('TRACK_ID').apply(_track_mean_speed, include_groups=False)
    track_lengths = df_spots.groupby('TRACK_ID').size()  # Быстрый подсчет спотов в каждом треке

    track_ends = df_spots.sort_values('FRAME').groupby('TRACK_ID').tail(1).set_index('TRACK_ID')
    track_starts = df_spots.sort_values('FRAME').groupby('TRACK_ID').head(1).set_index('TRACK_ID')

    candidates = []
    for end_id, end_row in track_ends.iterrows():
        spd_end = link_speed_map.get(end_id, 0)
        if pd.isna(spd_end): spd_end = 0
        len_end = track_lengths.get(end_id, 0)  # Достаем из словаря за O(1)

        end_frame = end_row['FRAME']

        for start_id, start_row in track_starts.iterrows():
            if start_id == end_id:
                continue

            start_frame = start_row['FRAME']
            if start_frame < end_frame:
                continue

            missing_frames = max(0, int(start_frame - end_frame - 1))
            if missing_frames > LINK_MAX_GAP:
                continue

            d = point_distance(end_row['POSITION_X'], end_row['POSITION_Y'],
                               start_row['POSITION_X'], start_row['POSITION_Y'])

            # Вычисляем всё, что нужно для проверок, ДО самих проверок
            spd_start = link_speed_map.get(start_id, 0)
            if pd.isna(spd_start): spd_start = 0
            len_start = track_lengths.get(start_id, 0)  # Достаем из словаря за O(1)

            # --- ДЕБАГ ПРОФИЛИРОВАНИЯ ---
            if missing_frames <= 30 and d < 200:
                reason = "PASSED"

                if d > BASE_SEARCH_RADIUS + (MAX_SPEED_PX * missing_frames):
                    reason = "DYNAMIC_DIST_FAIL"
                elif d > LINK_MAX_DIST_ABS:
                    reason = "ABS_DIST_FAIL"
                elif d / (missing_frames + 1) > MAX_SPEED_PX:
                    reason = "STITCH_SPEED_FAIL"
                elif spd_end < MIN_LINK_SPEED or spd_start < MIN_LINK_SPEED:
                    reason = "MIN_SPEED_FAIL"
                elif spd_end > 0 and spd_start > 0 and abs(spd_start - spd_end) / max(spd_end,
                                                                                      spd_start) > LINK_SPEED_DIFF:
                    reason = "SPEED_DIFF_FAIL"
                elif len_end < MIN_LINK_SPOTS or len_start < MIN_LINK_SPOTS:
                    reason = "MIN_SPOTS_FAIL"

                if reason != "PASSED":
                    spd_diff = abs(spd_start - spd_end) / max(spd_end, spd_start) if max(spd_end, spd_start) > 0 else 0
                    print(
                        f"От {end_id} к {start_id}: ОТБРАКОВКА ({reason}) | dist={d:.1f}, gap={missing_frames}, spd_diff={spd_diff:.2f}")

            # --- ЖЕСТКИЕ ПРОВЕРКИ ДЛЯ АЛГОРИТМА ---
            dynamic_max_dist = BASE_SEARCH_RADIUS + (MAX_SPEED_PX * missing_frames)
            if d > dynamic_max_dist:
                continue

            if d > LINK_MAX_DIST_ABS:
                continue

            stitch_speed = d / (missing_frames + 1)
            if stitch_speed > MAX_SPEED_PX:
                continue

            if spd_end < MIN_LINK_SPEED or spd_start < MIN_LINK_SPEED:
                continue

            if len_end < MIN_LINK_SPOTS or len_start < MIN_LINK_SPOTS:
                continue

            if spd_end > 0 and spd_start > 0:
                max_spd = max(spd_end, spd_start)
                if abs(spd_start - spd_end) / max_spd > LINK_SPEED_DIFF:
                    continue

            # --- СКОРИНГ ---
            cost = d + (missing_frames * MAX_SPEED_PX * 0.5)
            candidates.append({
                'from': end_id,
                'to': start_id,
                'distance': d,
                'gap': missing_frames,
                'cost': cost,
            })

    print(f"Найдено кандидатов для сшивки: {len(candidates)}")

    best_links = []
    used_from = set()
    used_to = set()

    if candidates:
        # Сортируем по cost, а не по distance
        cand_df = pd.DataFrame(candidates).sort_values('cost')
        for _, row in cand_df.iterrows():
            f, t = row['from'], row['to']
            if f not in used_from and t not in used_to:
                best_links.append({'from': f, 'to': t})
                used_from.add(f)
                used_to.add(t)

    print(f"Отобрано связей: {len(best_links)}")

    # ------------------------------------------------------------------
    #  ТАБЛИЦА СШИВОК — подробный отчёт по каждой склейке
    # ------------------------------------------------------------------
    if best_links:
        cand_lookup = {(c['from'], c['to']): c for c in candidates}

        link_report_rows = []
        for lnk in best_links:
            f, t = lnk['from'], lnk['to']
            c = cand_lookup.get((f, t), {})
            d = c.get('distance', float('nan'))
            gap = c.get('gap', float('nan'))
            cost = c.get('cost', float('nan'))

            end_frame = int(track_ends.loc[f, 'FRAME']) if f in track_ends.index else -1
            start_frame = int(track_starts.loc[t, 'FRAME']) if t in track_starts.index else -1
            end_x = track_ends.loc[f, 'POSITION_X'] if f in track_ends.index else float('nan')
            end_y = track_ends.loc[f, 'POSITION_Y'] if f in track_ends.index else float('nan')
            start_x = track_starts.loc[t, 'POSITION_X'] if t in track_starts.index else float('nan')
            start_y = track_starts.loc[t, 'POSITION_Y'] if t in track_starts.index else float('nan')

            spd_f = link_speed_map.get(f, float('nan'))
            spd_t = link_speed_map.get(t, float('nan'))
            len_f = int((df_spots['TRACK_ID'] == f).sum())
            len_t = int((df_spots['TRACK_ID'] == t).sum())
            stitch_speed = d / (gap + 1) if (not pd.isna(d) and gap >= 0) else float('nan')

            link_report_rows.append({
                'ОТ_ТРЕК': f,
                'К_ТРЕКУ': t,
                'КАДР_КОНЦА': end_frame,
                'КАДР_НАЧАЛА': start_frame,
                'GAP_КАДРОВ': gap,
                'РАССТ_px': round(d, 1),
                'СКОР_СТЫКА': round(stitch_speed, 2),
                'СРД_СКР_ОТ': round(spd_f, 3) if not pd.isna(spd_f) else '~',
                'СРД_СКР_К': round(spd_t, 3) if not pd.isna(spd_t) else '~',
                'СПОТОВ_ОТ': len_f,
                'СПОТОВ_К': len_t,
                'X_КОНЦА': round(end_x, 1),
                'Y_КОНЦА': round(end_y, 1),
                'X_НАЧАЛА': round(start_x, 1),
                'Y_НАЧАЛА': round(start_y, 1),
                'COST': round(cost, 2),
            })

        df_link_report = pd.DataFrame(link_report_rows).sort_values('КАДР_КОНЦА')
        print("\nТАБЛИЦА СШИВОК:")
        print(df_link_report.to_string(index=False))
    # ------------------------------------------------------------------

    all_ids = df_spots['TRACK_ID'].unique()
    merge_map = {tid: tid for tid in all_ids}

    if best_links:
        G = nx.DiGraph()
        G.add_nodes_from(all_ids)
        for lnk in best_links:
            G.add_edge(lnk['from'], lnk['to'])
        for component in nx.weakly_connected_components(G):
            root_id = min(component)
            for node in component:
                merge_map[node] = root_id

    df_spots['TRACK_ID'] = df_spots['TRACK_ID'].map(merge_map)
    n_merged = sum(1 for k, v in merge_map.items() if k != v)
    print(f"\nОбъединено фрагментов треков: {n_merged}")
    print(f"Треков после сшивки: {df_spots['TRACK_ID'].nunique()}")

    # Цепочки сшивок — показываем треки собранные из 2+ фрагментов
    if best_links:
        chains = {}
        for component in nx.weakly_connected_components(G):
            if len(component) > 1:
                root_id = min(component)
                sub = G.subgraph(component)
                # Начало цепочки — узел без входящих рёбер
                chain_starts = [n for n in sub.nodes if sub.in_degree(n) == 0]
                for s in chain_starts:
                    path = [s]
                    cur = s
                    while True:
                        succs = list(sub.successors(cur))
                        if not succs:
                            break
                        cur = succs[0]
                        path.append(cur)
                    parts = []
                    for node in path:
                        fs = int(track_starts.loc[node, 'FRAME']) if node in track_starts.index else -1
                        fe = int(track_ends.loc[node, 'FRAME']) if node in track_ends.index else -1
                        parts.append(f"ID{node}[{fs}-{fe}]")
                    chains[root_id] = " → ".join(parts)

        if chains:
            print(f"\nЦЕПОЧКИ СШИВОК ({len(chains)} треков из 2+ фрагментов):")
            for root_id, chain in sorted(chains.items()):
                print(f"  Новый ID {root_id}: {chain}")

    # --------------------------------------------------------------------------
    #  ЭТАП 2: РАЗРЕЗ У ГРАНИЦ
    # --------------------------------------------------------------------------

    print("\n--- ЭТАП 2: РАЗРЕЗ У ГРАНИЦ (исправленный) ---")

    df_spots = df_spots.sort_values(['TRACK_ID', 'FRAME']).reset_index(drop=True)
    df_spots['GAP_SCALE'] = 0

    for tid, table in df_spots.groupby('TRACK_ID'):
        idx = table.index.tolist()
        frames = table['FRAME'].values
        for j in range(1, len(idx)):
            df_spots.loc[idx[j - 1], 'GAP_SCALE'] = int(frames[j] - frames[j - 1] - 1)

    # Границы + большой разрыв
    border_mask = (
            (df_spots['POSITION_X'] <= BORDER_X_MIN) |
            (df_spots['POSITION_X'] >= BORDER_X_MAX) |
            (df_spots['POSITION_Y'] <= BORDER_Y_MIN) |
            (df_spots['POSITION_Y'] >= BORDER_Y_MAX)
    )
    df_gaps = df_spots[border_mask & (df_spots['GAP_SCALE'] > GAP_SCALE_PAR)].copy()

    print(f"Спотов у границы с разрывами (> {GAP_SCALE_PAR} кадров): {len(df_gaps)}")

    next_new_id = int(df_spots['TRACK_ID'].max()) + 1
    last_cut = {}

    for _, row in df_gaps.sort_values(['TRACK_ID', 'FRAME']).iterrows():
        spot_mask = (
                (df_spots['FRAME'] == row['FRAME']) &
                (df_spots['POSITION_X'] == row['POSITION_X']) &
                (df_spots['POSITION_Y'] == row['POSITION_Y'])
        )
        if not spot_mask.any(): continue
        track = int(df_spots.loc[spot_mask, 'TRACK_ID'].iloc[0])
        frame = int(row['FRAME'])

        spots_data = df_spots[df_spots['TRACK_ID'] == track].sort_values('FRAME')
        if spots_data.empty: continue

        if track not in last_cut:
            last_cut[track] = int(spots_data['FRAME'].min())

        before = spots_data[(spots_data['FRAME'] >= last_cut[track]) & (spots_data['FRAME'] <= frame)]
        after = spots_data[spots_data['FRAME'] > frame]

        if len(before) < 3 or len(after) < 3:
            last_cut[track] = frame + 1
            continue

        vec_before = calc_velocity_vector(before, tail=BORDER_VELOCITY_TAIL, use_tail=True)
        vec_after = calc_velocity_vector(after, tail=BORDER_VELOCITY_TAIL, use_tail=False)

        angle = angle_between(vec_before, vec_after)

        if angle < THRESHOLD:  # ← теперь строже
            print(f'   РЕЖЕМ трек {track} на кадре {frame} (угол {angle:.1f}° < {THRESHOLD}°)')
            cut_mask = (df_spots['TRACK_ID'] == track) & (df_spots['FRAME'] > frame)
            df_spots.loc[cut_mask, 'TRACK_ID'] = next_new_id
            next_new_id += 1
            last_cut[track] = frame + 1
        else:
            last_cut[track] = frame + 1  # продолжаем трек

    # --------------------------------------------------------------------------
    #  ЭТАП 3: ДЕТЕКЦИЯ И СВАП КОЛЛИЗИЙ
    # --------------------------------------------------------------------------

    print("\n--- ЭТАП 3: ДЕТЕКЦИЯ И СВАП КОЛЛИЗИЙ ---")

    df_spots['ORIG_TRACK_ID'] = df_spots['TRACK_ID'].copy()
    df_spots['NEW_TRACK_ID']  = df_spots['TRACK_ID'].copy()

    frame_index: dict = {}
    for _, row in df_spots.iterrows():
        f = int(row['FRAME'])
        frame_index.setdefault(f, {})[int(row['ORIG_TRACK_ID'])] = (
            row['POSITION_X'], row['POSITION_Y']
        )

    collision_events = []
    seen_pairs: set  = set()

    for tid, table in df_spots.sort_values('FRAME').groupby('ORIG_TRACK_ID'):
        table = table.sort_values('FRAME').reset_index(drop=True)
        for j in range(1, len(table)):
            gap_start_frame = int(table.loc[j - 1, 'FRAME'])
            gap_end_frame   = int(table.loc[j,     'FRAME'])
            gap = gap_end_frame - gap_start_frame - 1
            if gap <= 0:
                continue

            ax = table.loc[j - 1, 'POSITION_X']
            ay = table.loc[j - 1, 'POSITION_Y']

            search_frame = gap_start_frame + 1

            best_dist, best_tid_b = None, None
            for tid_b, (bx, by) in frame_index.get(search_frame, {}).items():
                if tid_b == tid:
                    continue
                d = point_distance(ax, ay, bx, by)
                if d < COLLISION_DISTANCE_THRESHOLD:
                    if best_dist is None or d < best_dist:
                        best_dist, best_tid_b = d, tid_b

            if best_tid_b is not None:
                pair_key = (min(tid, best_tid_b), max(tid, best_tid_b), search_frame)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                collision_events.append({
                    'track_a':       tid,
                    'track_b':       best_tid_b,
                    'frame_gap_end': search_frame,
                    'distance':      best_dist,
                    'gap_frames':    gap,
                })

    print(f"Найдено потенциальных коллизий: {len(collision_events)}")

    last_event_frame: dict = {}
    report_rows = []

    def get_last_fragment(orig_tid, f_split):
        prev       = last_event_frame.get(orig_tid)
        track_data = df_spots[df_spots['NEW_TRACK_ID'] == orig_tid]
        if prev is None:
            return track_data[track_data['FRAME'] < f_split]
        return track_data[(track_data['FRAME'] > prev) & (track_data['FRAME'] < f_split)]

    def get_next_fragment(orig_tid, f_split):
        track_data = df_spots[df_spots['NEW_TRACK_ID'] == orig_tid]
        next_event_frames = sorted([
            ev['frame_gap_end'] for ev in collision_events
            if (ev['track_a'] == orig_tid or ev['track_b'] == orig_tid)
               and ev['frame_gap_end'] > f_split
        ])
        if not next_event_frames:
            return None
        f_next = next_event_frames[0]
        if len(next_event_frames) > 1:
            return track_data[(track_data['FRAME'] >= f_next) & (track_data['FRAME'] < next_event_frames[1])]
        return track_data[track_data['FRAME'] >= f_next]

    for i, ev in enumerate(collision_events):
        t_a, t_b, f_split = ev['track_a'], ev['track_b'], ev['frame_gap_end']

        last_frag_a = get_last_fragment(t_a, f_split)
        last_frag_b = get_last_fragment(t_b, f_split)
        frag_a_post = df_spots[(df_spots['NEW_TRACK_ID'] == t_a) & (df_spots['FRAME'] >= f_split)]
        frag_b_post = df_spots[(df_spots['NEW_TRACK_ID'] == t_b) & (df_spots['FRAME'] >= f_split)]

        a_post_extended = False
        b_post_extended = False

        if 0 < len(frag_a_post) <= SHORT_FRAGMENT_SIZE:
            next_a = get_next_fragment(t_a, f_split)
            if next_a is not None and len(next_a) > SHORT_FRAGMENT_SIZE:
                frag_a_post = next_a
                a_post_extended = True
            else:
                frag_a_post = df_spots.iloc[0:0]

        if 0 < len(frag_b_post) <= SHORT_FRAGMENT_SIZE:
            next_b = get_next_fragment(t_b, f_split)
            if next_b is not None and len(next_b) > SHORT_FRAGMENT_SIZE:
                frag_b_post = next_b
                b_post_extended = True
            else:
                frag_b_post = df_spots.iloc[0:0]

        a_full_start = int(df_spots[df_spots['ORIG_TRACK_ID'] == t_a]['FRAME'].min())
        a_full_stop  = int(df_spots[df_spots['ORIG_TRACK_ID'] == t_a]['FRAME'].max())
        b_full_start = int(df_spots[df_spots['ORIG_TRACK_ID'] == t_b]['FRAME'].min())
        b_full_stop  = int(df_spots[df_spots['ORIG_TRACK_ID'] == t_b]['FRAME'].max())
        a_last_start = int(last_frag_a['FRAME'].min()) if not last_frag_a.empty else a_full_start
        b_last_start = int(last_frag_b['FRAME'].min()) if not last_frag_b.empty else b_full_start

        skip_reason = None
        if last_frag_a.empty or frag_b_post.empty:
            skip_reason = "нет last_A или B_post"
        elif len(last_frag_a) < MIN_FRAGMENT_SIZE:
            skip_reason = f"last_frag_A < {MIN_FRAGMENT_SIZE} спотов"
        elif len(frag_b_post) < MIN_FRAGMENT_SIZE:
            skip_reason = f"B_post < {MIN_FRAGMENT_SIZE} спотов"

        post_len = min(
            len(frag_a_post) if not frag_a_post.empty else 0,
            len(frag_b_post) if not frag_b_post.empty else 0,
        )
        mode = 'FULL' if post_len > LONG_POST_SPOTS else 'SPEED_VECT'

        spd_last_a = calc_mean_speed(last_frag_a)
        spd_last_b = calc_mean_speed(last_frag_b) if not last_frag_b.empty else None
        spd_a_post = calc_mean_speed(frag_a_post) if not frag_a_post.empty else None
        spd_b_post = calc_mean_speed(frag_b_post) if not frag_b_post.empty else None

        sls_a_post = calc_straight_line_speed(frag_a_post) if (mode == 'FULL' and not frag_a_post.empty) else None
        sls_b_post = calc_straight_line_speed(frag_b_post) if (mode == 'FULL' and not frag_b_post.empty) else None
        sls_last_a = calc_straight_line_speed(last_frag_a) if (mode == 'FULL' and not last_frag_a.empty) else None
        sls_last_b = calc_straight_line_speed(last_frag_b) if (mode == 'FULL' and not last_frag_b.empty) else None

        v_last_a = calc_velocity_vector(last_frag_a, use_tail=True)
        v_last_b = calc_velocity_vector(last_frag_b, use_tail=True) if not last_frag_b.empty else np.array([0., 0.])
        v_a_post = calc_velocity_vector(frag_a_post, use_tail=False) if not frag_a_post.empty else np.array([0., 0.])
        v_b_post = calc_velocity_vector(frag_b_post, use_tail=False) if not frag_b_post.empty else np.array([0., 0.])
        angle_keep = angle_between(v_last_a, v_a_post) + angle_between(v_last_b, v_b_post)
        angle_swap = angle_between(v_last_a, v_b_post) + angle_between(v_last_b, v_a_post)

        decision      = None
        spd_keep_diff = 0.0
        spd_swap_diff = 0.0

        if skip_reason:
            decision = f'ПРОПУСК ({skip_reason})'

        if decision is None:
            spd_keep_diff = speed_diff(spd_last_a, spd_a_post) + speed_diff(spd_last_b, spd_b_post)
            spd_swap_diff = speed_diff(spd_last_a, spd_b_post) + speed_diff(spd_last_b, spd_a_post)

            angle_favors_swap = (angle_swap < angle_keep - 1e-9)

            if mode == 'SPEED_VECT':
                if angle_favors_swap and spd_swap_diff < spd_keep_diff:
                    decision = 'СВАП (векторы + скорость)'
                elif not angle_favors_swap and spd_keep_diff < spd_swap_diff:
                    decision = 'БЕЗ СВАПА (векторы + скорость)'
                elif spd_swap_diff < spd_keep_diff:
                    decision = 'СВАП (скорость)'
                elif spd_keep_diff < spd_swap_diff:
                    decision = 'БЕЗ СВАПА (скорость)'
                elif angle_favors_swap:
                    decision = 'СВАП (только векторы)'
                else:
                    decision = 'БЕЗ СВАПА'

            elif mode == 'FULL':
                angle_favors_swap = (angle_swap < angle_keep - 1e-9
                                     if angle_keep is not None else False)

                sls_keep_diff = speed_diff(sls_last_a, sls_a_post) + speed_diff(sls_last_b, sls_b_post)
                sls_swap_diff = speed_diff(sls_last_a, sls_b_post) + speed_diff(sls_last_b, sls_a_post)

                if angle_favors_swap and spd_swap_diff < spd_keep_diff and sls_swap_diff < sls_keep_diff:
                    decision = 'СВАП (углы + скорость + SLS)'
                elif not angle_favors_swap and spd_keep_diff < spd_swap_diff and sls_keep_diff < sls_swap_diff:
                    decision = 'БЕЗ СВАПА (углы + скорость + SLS)'
                elif spd_swap_diff < spd_keep_diff and sls_swap_diff < sls_keep_diff:
                    decision = 'СВАП (скорость + SLS)'
                elif spd_keep_diff < spd_swap_diff and sls_keep_diff < sls_swap_diff:
                    decision = 'БЕЗ СВАПА (скорость + SLS)'
                elif angle_favors_swap and spd_swap_diff < spd_keep_diff:
                    decision = 'СВАП (углы + скорость)'
                elif not angle_favors_swap and spd_keep_diff < spd_swap_diff:
                    decision = 'БЕЗ СВАПА (углы + скорость)'
                elif spd_swap_diff < spd_keep_diff:
                    decision = 'СВАП (скорость)'
                elif spd_keep_diff < spd_swap_diff:
                    decision = 'БЕЗ СВАПА (скорость)'
                elif angle_favors_swap:
                    decision = 'СВАП (только углы)'
                else:
                    decision = 'БЕЗ СВАПА'

        final_skip = decision.startswith('ПРОПУСК') if decision else False

        if not final_skip:
            last_event_frame[t_a] = f_split
            last_event_frame[t_b] = f_split

        base_row = {
            'N': i + 1,
            'КАДР': f_split,
            'РАССТ': round(ev['distance'], 1),
            'GAP': ev['gap_frames'],
            'РЕЖИМ': mode,
            'POST_LEN': post_len,
            'A_ID': t_a,
            'A_КАДРЫ': f"{a_full_start}-{a_full_stop}",
            'A_LAST_START': a_last_start,
            'A_post_EXT': 'ДА' if a_post_extended else '-',
            'B_ID': t_b,
            'B_КАДРЫ': f"{b_full_start}-{b_full_stop}",
            'B_LAST_START': b_last_start,
            'B_post_EXT': 'ДА' if b_post_extended else '-',
            'SPD_last_A': fmt(spd_last_a),
            'SPD_last_B': fmt(spd_last_b),
            'SPD_A_post': fmt(spd_a_post),
            'SPD_B_post': fmt(spd_b_post),
            'SPD_KEEP_DIFF': round(spd_keep_diff, 3) if not final_skip else '-',
            'SPD_SWAP_DIFF': round(spd_swap_diff, 3) if not final_skip else '-',
            'SLS_last_A': fmt(sls_last_a),
            'SLS_last_B': fmt(sls_last_b),
            'SLS_A_post': fmt(sls_a_post),
            'SLS_B_post': fmt(sls_b_post),
            'УГ_KEEP': round(angle_keep, 1) if angle_keep is not None else '-',
            'УГ_SWAP': round(angle_swap, 1) if angle_swap is not None else '-',
        }

        if final_skip:
            report_rows.append({**base_row, 'РЕШЕНИЕ': decision,
                                'A_НОВЫЕ_КАДРЫ': '-', 'B_НОВЫЕ_КАДРЫ': '-'})
            continue

        if decision.startswith('СВАП'):
            max_join_dist = COLLISION_DISTANCE_THRESHOLD * 4
            try:
                ax0, ay0 = float(last_frag_a.iloc[-1]['POSITION_X']), float(last_frag_a.iloc[-1]['POSITION_Y'])
                bx0, by0 = float(last_frag_b.iloc[-1]['POSITION_X']), float(last_frag_b.iloc[-1]['POSITION_Y'])
                a_keep = frag_a_post.sort_values('FRAME').iloc[0] if not frag_a_post.empty else None
                b_keep = frag_b_post.sort_values('FRAME').iloc[0] if not frag_b_post.empty else None
                a_swap = frag_b_post.sort_values('FRAME').iloc[0] if not frag_b_post.empty else None
                b_swap = frag_a_post.sort_values('FRAME').iloc[0] if not frag_a_post.empty else None

                keep_join = (point_distance(ax0, ay0, float(a_keep['POSITION_X']), float(a_keep['POSITION_Y'])) +
                             point_distance(bx0, by0, float(b_keep['POSITION_X']), float(b_keep['POSITION_Y']))
                             ) if (a_keep is not None and b_keep is not None) else None
                swap_join = (point_distance(ax0, ay0, float(a_swap['POSITION_X']), float(a_swap['POSITION_Y'])) +
                             point_distance(bx0, by0, float(b_swap['POSITION_X']), float(b_swap['POSITION_Y']))
                             ) if (a_swap is not None and b_swap is not None) else None

                if swap_join is None or (keep_join is not None and swap_join > max_join_dist and swap_join > keep_join):
                    decision = f"ПРОПУСК (swap_join={fmt(swap_join)} > {max_join_dist})"
            except Exception:
                pass

        if decision.startswith('СВАП'):
            idx_a = df_spots[(df_spots['NEW_TRACK_ID'] == t_a) & (df_spots['FRAME'] >= f_split)].index
            idx_b = df_spots[(df_spots['NEW_TRACK_ID'] == t_b) & (df_spots['FRAME'] >= f_split)].index
            df_spots.loc[idx_a, 'NEW_TRACK_ID'] = t_b
            df_spots.loc[idx_b, 'NEW_TRACK_ID'] = t_a
            a_new_range = (f"{a_full_start}-{a_full_stop} "
                           f"(ORIG {t_a}[<{f_split}] + ORIG {t_b}[>={f_split}])")
            b_new_range = (f"{b_full_start}-{b_full_stop} "
                           f"(ORIG {t_b}[<{f_split}] + ORIG {t_a}[>={f_split}])")
        else:
            a_new_range = f"{a_full_start}-{a_full_stop} (не изменён)"
            b_new_range = f"{b_full_start}-{b_full_stop} (не изменён)"

        report_rows.append({**base_row, 'РЕШЕНИЕ': decision,
                            'A_НОВЫЕ_КАДРЫ': a_new_range,
                            'B_НОВЫЕ_КАДРЫ': b_new_range})

    number_of_swaps = sum(
        1 for row in report_rows
        if isinstance(row.get('РЕШЕНИЕ', ''), str) and row['РЕШЕНИЕ'].startswith('СВАП')
    )

    df_spots['TRACK_ID'] = df_spots['NEW_TRACK_ID']
    if report_rows:
        df_report = pd.DataFrame(report_rows)
        print("\nТАБЛИЦА СВАПОВ:")
        print(df_report.to_string(index=False))
        print(f'Количество СВАПОВ: {number_of_swaps}')
    else:
        print("Коллизий не обнаружено.")

    '''# --------------------------------------------------------------------------
    #  ЭТАП 3.5: РАЗРЕЗ ПО "ТЕЛЕПОРТАМ"
    # --------------------------------------------------------------------------

    print("\n--- ЭТАП 3.5: ТЕЛЕПОРТ-РАЗРЕЗЫ ---")

    TELEPORT_DIST = TELEPORT_SPEED_THRESHOLD
    next_new_id   = int(df_spots['TRACK_ID'].max()) + 1
    n_teleport_cuts = 0

    tracks_to_check = list(df_spots['TRACK_ID'].unique())

    while tracks_to_check:
        tid = tracks_to_check.pop(0)

        mask = df_spots['TRACK_ID'] == tid
        g    = df_spots[mask].sort_values('FRAME')

        frames = g['FRAME'].to_numpy()
        x      = g['POSITION_X'].to_numpy(dtype=float)
        y      = g['POSITION_Y'].to_numpy(dtype=float)

        if len(frames) < 2:
            continue

        dfr  = np.diff(frames)
        dx   = np.diff(x)
        dy   = np.diff(y)
        dist = np.sqrt(dx * dx + dy * dy)

        with np.errstate(divide='ignore', invalid='ignore'):
            speed_per_frame = dist / dfr

        valid_mean        = (dfr >= 1) & np.isfinite(speed_per_frame)
        track_mean_speed  = float(np.median(speed_per_frame[valid_mean])) if valid_mean.any() else 0.0
        dynamic_threshold = max(TELEPORT_DIST, track_mean_speed * TELEPORT_MEAN_MULTIPLIER)

        teleport_steps = np.where(
            (dfr >= 1) &
            (dfr <= TELEPORT_MAX_FRAME_STEP) &
            np.isfinite(speed_per_frame) &
            (speed_per_frame > dynamic_threshold)
        )[0]

        if len(teleport_steps) == 0:
            continue

        first_teleport_idx = teleport_steps[0]
        frame_start = int(frames[first_teleport_idx])
        frame_end   = int(frames[first_teleport_idx + 1])
        spike_speed = speed_per_frame[first_teleport_idx]

        print(
            f"Трек {tid}: скачок скорости {spike_speed:.1f} px/кадр "
            f"(порог {dynamic_threshold:.1f}). Кадры: {frame_start} -> {frame_end}"
        )

        mask_tail = (df_spots['TRACK_ID'] == tid) & (df_spots['FRAME'] >= frame_end)
        df_spots.loc[mask_tail, 'TRACK_ID'] = next_new_id
        tracks_to_check.append(next_new_id)
        next_new_id     += 1
        n_teleport_cuts += 1

    if n_teleport_cuts:
        print(f"Итого телепорт-разрезов: {n_teleport_cuts}")
    else:
        print("Телепортов не найдено.")'''

    # --------------------------------------------------------------------------
    #  ЭТАП 4: ФИЛЬТРАЦИЯ ПО ДЛИНЕ
    # --------------------------------------------------------------------------

    print("\n--- ЭТАП 4: ФИЛЬТРАЦИЯ ПО ДЛИНЕ ---")

    track_durations   = df_spots.groupby('TRACK_ID').apply(
        lambda g: int(g['FRAME'].max() - g['FRAME'].min()),
        include_groups=False
    )
    track_spot_counts = df_spots.groupby('TRACK_ID').size()
    valid_by_duration = track_durations[
        (track_durations >= MIN_TRACK_DURATION) &
        (track_spot_counts >= MIN_SPOTS)
    ].index

    df_spots_final = df_spots[df_spots['TRACK_ID'].isin(valid_by_duration)].copy().reset_index(drop=True)
    print(f"Треков после фильтрации (длительность >= {MIN_TRACK_DURATION} кадров, спотов >= 10): "
          f"{df_spots_final['TRACK_ID'].nunique()}")

    # --------------------------------------------------------------------------
    #  ПЕРЕСЧЁТ ФИНАЛЬНОЙ СТАТИСТИКИ ТРЕКОВ
    # --------------------------------------------------------------------------

    print("\n--- ПЕРЕСЧЁТ СТАТИСТИКИ ТРЕКОВ ---")

    new_tracks_list = []
    for tid, group in df_spots_final.groupby('TRACK_ID'):
        new_tracks_list.append(calc_track_stats(group))

    df_tracks_final = (pd.DataFrame(new_tracks_list)
                       .sort_values('TRACK_ID')
                       .reset_index(drop=True)
                       [TRACKS_COLUMNS])

    new_ids = list(range(1, len(df_tracks_final) + 1))
    df_tracks_final['TRACK_ID']    = new_ids
    df_tracks_final['TRACK_INDEX'] = new_ids
    df_tracks_final['LABEL']       = [f"Track_{i}" for i in new_ids]

    print(f"Финальных треков: {len(df_tracks_final)}")
    print(df_tracks_final.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_tracks_final.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\nТреки сохранены -> {output_path}")


# ==============================================================================
#  ТОЧКА ВХОДА: ПАКЕТНАЯ ОБРАБОТКА
# ==============================================================================

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    print("=" * 80)
    print("  ПОИСК ФАЙЛОВ")
    print("=" * 80)

    pairs = find_video_pairs(SPOTS_DIR, TRACKS_DIR)
    print(f"Найдено видео для обработки: {len(pairs)}")
    for p in pairs:
        print(f"  {p['prefix']}")
        print(f"    споты:  {p['spots'].name}")
        print(f"    треки:  {p['tracks'].name}")

    print(f"\nВыходная папка: {OUTPUT_DIR}")
    print("=" * 80)

    results = []

    for p in pairs:
        output_path = Path(OUTPUT_DIR) / f"{p['prefix']}{OUTPUT_SUFFIX}.xlsx"
        process_video(
            spots_path=p['spots'],
            tracks_path=p['tracks'],
            output_path=output_path,
            prefix=p['prefix'],
        )
        results.append({'prefix': p['prefix'], 'output': output_path})

    print("\n" + "=" * 80)
    print("  ИТОГ ПАКЕТНОЙ ОБРАБОТКИ")
    print("=" * 80)
    print(f"Обработано видео: {len(results)}")
    for r in results:
        print(f"  OK {r['prefix']} -> {r['output'].name}")
    print("\nГотово.")