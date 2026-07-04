import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
#  НАСТРОЙКИ
# ==============================================================================
INPUT_DIR = r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\THIRD TERTILE 11 9 7 3 17 16 19.xlsx"
OUTPUT_PATH = r"D:\Lab\20 videos\Without grid\CONCATENATED TABLES\THIRD TERTILE (p) 11 9 7 3 17 16 19.xlsx"

# Названия колонок из TrackMate
COL_SPEED = 'TRACK_MEAN_SPEED'
COL_LFP = 'LINEARITY_OF_FORWARD_PROGRESSION'
COL_STD = 'TRACK_STD_SPEED'

FEATURES = [COL_SPEED, COL_LFP, COL_STD]

# ПОРОГИ ВОЗ 2021
SPEED_IMMOTILE = 3.525
SPEED_PROGRESSIVE = 7.245
LFP_PROGRESSIVE = 0.122

# Цвета категорий
COLORS = {
    'RP': '#2ecc71',  # зелёный  — прогрессивная
    'SP': '#3498db',  # синий    — непрогрессивная быстрая
    'NP': '#e67e22',  # оранжевый — непрогрессивная медленная
    'IM': '#e74c3c',  # красный  — неподвижная
}


# ==============================================================================
#  ФУНКЦИИ
# ==============================================================================

def force_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Очистка строк от запятых/пробелов и конвертация в числа."""
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace(",", ".", regex=False)
                     .str.replace(" ", "", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def classify_who(df: pd.DataFrame) -> pd.DataFrame:
    """Классификация по ВОЗ 2021."""
    conds = [
        (df[COL_SPEED] < SPEED_IMMOTILE),                                              # D — неподвижные
        (df[COL_LFP] < LFP_PROGRESSIVE),                                               # C — непрогрессивные (медленные)
        (df[COL_SPEED] < SPEED_PROGRESSIVE),                                           # B — непрогрессивные (быстрые)
        (df[COL_SPEED] >= SPEED_PROGRESSIVE) & (df[COL_LFP] >= LFP_PROGRESSIVE),      # A — прогрессивные
    ]
    choices = ['IM', 'NP', 'SP', 'RP']
    df['WHO_CATEGORY'] = np.select(conds, choices, default='Unknown')
    return df


def print_percentages(df: pd.DataFrame) -> None:
    """Вывод процентов по каждой категории."""
    total = len(df)
    counts = df['WHO_CATEGORY'].value_counts().reindex(['RP', 'SP', 'NP', 'IM'], fill_value=0)

    labels = {
        'RP': 'Прогрессивная (RP)',
        'SP': 'Непрогрессивная быстрая (SP)',
        'NP': 'Непрогрессивная медленная (NP)',
        'IM': 'Неподвижная (IM)',
    }

    print("\n" + "=" * 45)
    print(f"  Всего треков: {total}")
    print("=" * 45)
    for cat in ['RP', 'SP', 'NP', 'IM']:
        n = counts[cat]
        pct = 100 * n / total if total > 0 else 0
        print(f"  {labels[cat]:<35} {n:>5}  ({pct:5.1f}%)")
    print("=" * 45 + "\n")

x_max=10


def plot_2d_scatter(df: pd.DataFrame) -> None:
    """Оптимизированный 2D scatter-plot для научного постера."""
    # Увеличиваем размер графика, чтобы на постере всё было крупно
    fig, ax = plt.subplots(figsize=(10, 8))

    total = len(df)

    # 1. Рисуем точки (в легенду пойдут ТОЛЬКО категории и их %/n)
    for cat in ['RP', 'SP', 'NP', 'IM']:
        mask = df['WHO_CATEGORY'] == cat
        subset = df[mask]
        n = mask.sum()
        pct = n * 100 / total if total > 0 else 0

        ax.scatter(
            subset[COL_SPEED],
            subset[COL_LFP],
            c=COLORS[cat],
            # Лаконичная и крупная подпись для легенды:
            label=f'{cat}: {pct:.1f}% (n={n})',
            alpha=0.6,
            s=25,  # Чуть увеличили размер точек для видимости
            linewidths=0,
            zorder=3,
        )

    # 2. Линии порогов (УБРАЛИ параметр label, чтобы они не лезли в легенду!)
    ax.axvline(x=SPEED_IMMOTILE, color='#c0392b', linestyle='--', linewidth=1.6, zorder=4)
    ax.axvline(x=SPEED_PROGRESSIVE, color='#8e44ad', linestyle='--', linewidth=1.6, zorder=4)
    ax.axhline(y=LFP_PROGRESSIVE, color='#2980b9', linestyle='--', linewidth=1.6, zorder=4)

    # 3. Значения порогов на осях (Шрифты крупнее, размерность исправлена на px/frame)
    ax.annotate(f'{SPEED_IMMOTILE} px/fr', xy=(SPEED_IMMOTILE, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -18), textcoords='offset points',
                color='#c0392b', fontsize=10, fontweight='bold', ha='center', va='top',
                annotation_clip=False)

    ax.annotate(f'{SPEED_PROGRESSIVE} px/fr', xy=(SPEED_PROGRESSIVE, 0), xycoords=('data', 'axes fraction'),
                xytext=(0, -18), textcoords='offset points',
                color='#8e44ad', fontsize=10, fontweight='bold', ha='center', va='top',
                annotation_clip=False)

    ax.annotate(f'{LFP_PROGRESSIVE}', xy=(0, LFP_PROGRESSIVE), xycoords=('axes fraction', 'data'),
                xytext=(-10, 0), textcoords='offset points',
                color='#2980b9', fontsize=10, fontweight='bold', ha='right', va='center',
                annotation_clip=False)

    # 4. Крупные подписи зон внутри графика (fontsize=60 для постера)
    zone_style = dict(fontsize=60, alpha=0.25, fontweight='bold', ha='center', va='center')

    # Статичные координаты для текста, чтобы они не прыгали от квантилей
    ax.text(16, 0.6, 'RP', color=COLORS['RP'], **zone_style)
    ax.text(5.4, 0.6, 'SP', color=COLORS['SP'], **zone_style)
    ax.text(16, 0.06, 'NP', color=COLORS['NP'], **zone_style)
    ax.text(1.7, 0.5, 'IM', color=COLORS['IM'], **zone_style)

    # 5. Оформление осей (Размеры шрифтов увеличены для чтения с расстояния)
    ax.set_xlabel('MEAN SPEED (pixel/frame)', fontsize=14, fontweight='bold', labelpad=20)
    ax.set_ylabel('LINEARITY OF FORWARD PROGRESSION', fontsize=14, fontweight='bold')
    ax.set_title('SPERM MOTILITY CLUSTERISATION', fontsize=16, fontweight='bold', pad=15)

    # Легенда: теперь тут всего 4 чистые строчки, шрифт увеличен до 11
    ax.legend(fontsize=14, loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#bdc3c7')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    # Вставляй это перед plt.show()
    plt.savefig(r"D:\Lab\20 videos\PHOTOS\mCARM 2026 poster\MANUAL\THIRD_TERTILEM",
                dpi=300,
                bbox_inches='tight')  # 'tight' убирает лишние белые поля вокруг
    plt.show()


# ==============================================================================
#  MAIN
# ==============================================================================
if __name__ == '__main__':
    print("🚀 Загрузка данных...")

    try:
        df_all = pd.read_excel(INPUT_DIR)
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        exit()

    df_all = force_numeric(df_all, FEATURES)

    print("📊 Классификация по ВОЗ 2021...")
    df_all = classify_who(df_all)

    # Перемещаем WHO_CATEGORY в начало
    cols = list(df_all.columns)
    cols.remove('WHO_CATEGORY')
    cols.insert(1, 'WHO_CATEGORY')
    df_all = df_all[cols]

    # --- СОХРАНЕНИЕ ---
    '''Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df_all.to_excel(writer, sheet_name='All_Tracks', index=False)'''

    #print(f"✅ Файл сохранен: {OUTPUT_PATH}")

    # --- ПРОЦЕНТЫ ---
    print_percentages(df_all)

    # --- ВИЗУАЛИЗАЦИЯ 2D ---
    plot_2d_scatter(df_all)