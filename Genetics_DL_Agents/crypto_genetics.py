from __future__ import annotations
import os, sys, io, json, time, calendar, pickle, tempfile, warnings
import builtins as _builtins
import numpy as np
import multiprocessing as _mp
from collections import deque
from typing import List, Tuple, Dict, Optional


def _bootstrap_project_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = base_dir
    if not os.path.isdir(os.path.join(project_root, "Retrodate_cryptotrade")):
        project_root = os.path.dirname(base_dir)
    for extra_dir in (
        os.path.join(project_root, "Retrodate_cryptotrade"),
        os.path.join(project_root, "Genetics_DL_Agents"),
    ):
        if os.path.isdir(extra_dir) and extra_dir not in sys.path:
            sys.path.insert(0, extra_dir)


_bootstrap_project_paths()

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore', over='ignore')


def _safe_print(*args, **kwargs):
    """Печатает безопасно для Windows-консолей без UTF-8, не роняя импорт genetics."""
    try:
        _builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        file = kwargs.get('file', sys.stdout)
        flush = kwargs.get('flush', False)
        text = sep.join(str(a) for a in args)
        encoding = getattr(file, 'encoding', None) or 'utf-8'
        safe_text = text.encode(encoding, errors='replace').decode(encoding, errors='replace')
        _builtins.print(safe_text, end=end, file=file, flush=flush)


print = _safe_print

# ── Флаг подпроцесса ─────────────────────────────────────────────────────────
# На Windows spawn создаёт новый процесс. _is_worker() проверяет:
# 1. Переменную окружения _GENETICS_WORKER (устанавливается перед p.start())
# 2. Имя процесса (spawn-воркеры называются не '__main__' / 'MainProcess')
# Функция (не константа) — проверяется в момент print, а не при импорте.
def _is_worker() -> bool:
    if os.environ.get('_GENETICS_WORKER') == '1':
        return True
    # Дополнительная проверка через multiprocessing — spawn-воркеры не являются MainProcess
    try:
        import multiprocessing as _mp2
        return _mp2.current_process().name != 'MainProcess'
    except Exception:
        return False

_IS_WORKER: bool = False   # legacy alias

# ── Numba JIT (критическая оптимизация simulate_batch: 50-100x ускорение) ──
try:
    import numba
    from numba import njit, prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    print("  [WARNING] numba НЕ установлен! pip install numba  (без него 1 ген = 30+ мин)")
    # Заглушки чтобы декораторы не ломали код
    def njit(*a, **kw):
        def dec(f): return f
        return dec
    def prange(*a): return range(*a)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  значения по умолчанию; переопределяются из settings_genetic.txt
# ══════════════════════════════════════════════════════════════════════════════

def _load_genetic_settings() -> dict:
    """Загружает settings_genetic.txt из папки скрипта или текущей директории."""
    import os as _os
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    candidates = [
        _os.path.join(script_dir, "settings_genetic.txt"),
        "settings_genetic.txt",
    ]
    path = next((p for p in candidates if _os.path.exists(p)), None)
    if path is None:
        return {}
    cfg = {}
    try:
        with open(path, encoding="utf-8") as _f:
            for line in _f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip().lower()
                val = val.strip()
                if "#" in val:
                    val = val[:val.index("#")].strip()
                if key and val:
                    cfg[key] = val
        if not _is_worker():
            print(f"  [genetic_settings] Загружено {len(cfg)} параметров из {path}")
    except Exception as _e:
        if not _is_worker():
            print(f"  [genetic_settings] Ошибка чтения {path}: {_e}")
    return cfg

def _gscfg_f(cfg, key, default):
    try:    return float(cfg.get(key, default))
    except: return float(default)
def _gscfg_i(cfg, key, default):
    try:    return int(cfg.get(key, default))
    except: return int(default)
def _gscfg_b(cfg, key, default_on=True):
    v = str(cfg.get(key, "on" if default_on else "off")).lower()
    return v in ("on", "yes", "true", "1")

_GS = _load_genetic_settings()   # глобальный словарь настроек

# ── Импорт параметров биржи для синхронизации train == live ───────────────────
# Все общие параметры симуляции (комиссии, плечо, стоп, fraction и т.д.)
# берутся из crypto_exchange, чтобы гарантировать идентичность обучения и торговли.
try:
    import crypto_exchange as _cx
    _CX_OK = True
except Exception as _cx_err:
    _cx = None
    _CX_OK = False
    print(f"  [genetics] ⚠ crypto_exchange недоступен ({_cx_err}) — используются fallback-значения")

# ── [1] Популяция и поколения ─────────────────────────────────────────────────
POP_SIZE       = _gscfg_i(_GS, 'pop_size',       600)
N_GENERATIONS  = _gscfg_i(_GS, 'n_generations',  100)
ELITE_SIZE     = _gscfg_i(_GS, 'elite_size',     8)
TOURNAMENT_K   = _gscfg_i(_GS, 'tournament_k',   5)
TOURNAMENT_K_MAX         = _gscfg_i(_GS, 'tournament_k_max',  10)
TOURNAMENT_PRESSURE_STEP = 1
CROSSOVER_RATE = _gscfg_f(_GS, 'crossover_rate', 0.70)
MUTATION_RATE  = _gscfg_f(_GS, 'mutation_rate',  0.20)
MUTATION_SIGMA = _gscfg_f(_GS, 'mutation_sigma', 0.25)
SIGMA_DECAY    = _gscfg_f(_GS, 'sigma_decay',    0.997)
SIGMA_MIN      = _gscfg_f(_GS, 'sigma_min',      0.100)
SEED           = 42

# ── [2] Архитектура нейронной сети ────────────────────────────────────────────
N_INPUT   = _gscfg_i(_GS, 'n_input',   28)   # +2: cross-sectional rank фичи (26=mom_rank, 27=vol_rank)
N_HIDDEN1 = _gscfg_i(_GS, 'n_hidden1', 128)   # 28→128→64→32→9 = 14,345 params, coverage 6.3x
N_HIDDEN2 = _gscfg_i(_GS, 'n_hidden2',  64)
N_HIDDEN3 = _gscfg_i(_GS, 'n_hidden3',  32)
N_ACTIONS = _gscfg_i(_GS, 'n_actions',   9)   # 9-action: half/full sizing + close

# ── [3] Острова ───────────────────────────────────────────────────────────────
N_ISLANDS          = _gscfg_i(_GS, 'n_islands',          3)   # 3 острова: crash/bear/bull
MIGRATION_INTERVAL = _gscfg_i(_GS, 'migration_interval', 10)
MIGRATION_SIZE     = _gscfg_i(_GS, 'migration_size',     3)
PHASE1_ENABLED     = _gscfg_b(_GS, 'phase1_enabled',     True)
PHASE1_GENS_FRAC   = _gscfg_f(_GS, 'phase1_gens_frac',   0.30)
# Phase 1 теперь использует ПОЛНЫЙ датасет, но с островными весами для каждого периода.
# Каждый остров получает высокий вес для своих режимов и малый бонус для чужих.
# Это устраняет обрыв фитнеса при переходе Phase1→Phase2.
PHASE1_OWN_WEIGHT    = _gscfg_f(_GS, 'phase1_own_weight',    1.0)   # вес своего режима
PHASE1_BONUS_WEIGHT  = _gscfg_f(_GS, 'phase1_bonus_weight',  0.25)  # бонус соседнего режима
PHASE1_MIN_WEIGHT    = _gscfg_f(_GS, 'phase1_min_weight',    0.08)  # мин. вес чужого режима

# ── [4] Алгоритмы оптимизации ─────────────────────────────────────────────────
DE_FRACTION            = _gscfg_f(_GS, 'de_fraction',             0.20)
DE_F                   = _gscfg_f(_GS, 'de_f',                    0.45)
NES_FRAC               = _gscfg_f(_GS, 'nes_frac',                0.15)
NES_LR                 = _gscfg_f(_GS, 'nes_lr',                  0.04)
NES_MIN_SAMPLES        = 8
CMA_MEMORY_LR          = _gscfg_f(_GS, 'cma_memory_lr',           0.15)
CMA_BLEND              = _gscfg_f(_GS, 'cma_blend',               0.08)
STAGNATION_GENS        = _gscfg_i(_GS, 'stagnation_gens',         5)
EXPLOIT_SIGMA_MULT     = _gscfg_f(_GS, 'exploit_sigma_mult',      0.50)
ESCAPE_SIGMA_MULT      = _gscfg_f(_GS, 'escape_sigma_mult',       4.5)
EXPLOIT_SIGMA_MIN      = SIGMA_MIN * 3
MAX_RESTARTS_NO_IMPROVE= _gscfg_i(_GS, 'max_restarts_no_improve', 2)
LOCAL_FRAC             = _gscfg_f(_GS, 'local_frac',              0.15)
LOCAL_SIGMA_MULT       = _gscfg_f(_GS, 'local_sigma_mult',        0.08)
DIVERSITY_INTERVAL     = _gscfg_i(_GS, 'diversity_interval',      8)
DIVERSITY_FRAC         = _gscfg_f(_GS, 'diversity_frac',          0.22)
DIVERSITY_THRESHOLD    = _gscfg_f(_GS, 'diversity_threshold',     0.08)

# ── Phase 2 (Stabilization) / Phase 3 (Dynamic) ──────────────────────────────
PHASE2_ENABLED     = _gscfg_b(_GS, 'phase2_enabled',      True)
PHASE2_GENS_FRAC   = _gscfg_f(_GS, 'phase2_gens_frac',    0.12)
PHASE2_SIGMA       = _gscfg_f(_GS, 'phase2_sigma',         0.030)
PHASE3_SIGMA_RESTART = _gscfg_f(_GS, 'phase3_sigma_restart', 0.135)

# ── NES-коллапс детект ────────────────────────────────────────────────────────
NES_COLLAPSE_GENS      = _gscfg_i(_GS, 'nes_collapse_gens',      8)
NES_COLLAPSE_THRESHOLD = _gscfg_f(_GS, 'nes_collapse_threshold', 0.18)

# ── Послойная инъекция ────────────────────────────────────────────────────────
LAYER_INJECTION_ENABLED = _gscfg_b(_GS, 'layer_injection_enabled', True)

# ── Lévy-flight мутации ───────────────────────────────────────────────────────
LEVY_ENABLED      = _gscfg_b(_GS, 'levy_enabled',      True)
LEVY_FRAC         = _gscfg_f(_GS, 'levy_frac',         0.10)  # доля потомков через Lévy
LEVY_ALPHA        = _gscfg_f(_GS, 'levy_alpha',        1.50)  # хвостовой параметр Lévy (1<α<2)
LEVY_SCALE        = _gscfg_f(_GS, 'levy_scale',        0.08)  # масштаб Lévy-скачка

# ── Sigma reduction on new best (exploit mode) ─────────────────────────────────
# Корень стагнации gen51-57: sigma остаётся высокой (0.25) после нового лучшего
# (gen53: best=26.49) → огромные мутации разрушают решение (gen54: best=17.22).
# Немедленное снижение sigma при новом лучшем предотвращает вылет из хорошего региона.
EXPLOIT_SIGMA_ON_BEST       = _gscfg_b(_GS, 'exploit_sigma_on_best',       True)
EXPLOIT_SIGMA_ON_BEST_MULT  = _gscfg_f(_GS, 'exploit_sigma_on_best_mult',  0.65)
EXPLOIT_SIGMA_ON_BEST_FLOOR = _gscfg_f(_GS, 'exploit_sigma_on_best_floor', 0.06)

# ── Cosine annealing sigma ─────────────────────────────────────────────────────
COSINE_SIGMA_ENABLED = _gscfg_b(_GS, 'cosine_sigma_enabled', True)
COSINE_T0            = _gscfg_i(_GS, 'cosine_t0',            20)  # период cosine цикла
COSINE_TMULT         = _gscfg_f(_GS, 'cosine_tmult',         1.5) # мультипликатор удлинения

_lsm_raw = _GS.get('layer_sigma_mult', '1.2,0.9,0.8,1.5')
try:    LAYER_SIGMA_MULT = [float(x.strip()) for x in _lsm_raw.split(',')]
except: LAYER_SIGMA_MULT = [1.2, 0.9, 0.8, 1.5]

# ── [5] Функция фитнеса ───────────────────────────────────────────────────────
FITNESS_W_MEAN          = _gscfg_f(_GS, 'fitness_w_mean',          0.20)
FITNESS_W_CALMAR        = _gscfg_f(_GS, 'fitness_w_calmar',        0.25)
FITNESS_W_SHARPE        = _gscfg_f(_GS, 'fitness_w_sharpe',        0.10)
FITNESS_W_DAILY_SHARPE  = _gscfg_f(_GS, 'fitness_w_daily_sharpe',  0.15)
FITNESS_W_DAILY_WINRATE = _gscfg_f(_GS, 'fitness_w_daily_winrate', 0.10)
FITNESS_W_CONSISTENCY   = _gscfg_f(_GS, 'fitness_w_consistency',   0.08)
FITNESS_ALPHA           = _gscfg_f(_GS, 'fitness_alpha',            0.20)
FITNESS_BETA            = _gscfg_f(_GS, 'fitness_beta',             0.10)
FITNESS_GAMMA           = _gscfg_f(_GS, 'fitness_gamma',            1.20)
FITNESS_DELTA           = _gscfg_f(_GS, 'fitness_delta',            0.08)
FITNESS_EPSILON         = _gscfg_f(_GS, 'fitness_epsilon',          0.60)
FITNESS_ZETA            = _gscfg_f(_GS, 'fitness_zeta',             0.10)
FITNESS_ETA             = _gscfg_f(_GS, 'fitness_eta',              0.08)
FITNESS_THETA           = _gscfg_f(_GS, 'fitness_theta',            0.18)
TRADE_REWARD_W          = _gscfg_f(_GS, 'trade_reward_w',           0.30)
TRADE_REWARD_REF        = _gscfg_f(_GS, 'trade_reward_ref',         0.01)

# ── [6] Пороги вознаграждения за период ───────────────────────────────────────
WIN1_PROFIT_THRESH  = _gscfg_f(_GS, 'win1_profit_thresh',  1.0)   # мини-победа ≥1%
WIN1_FITNESS_WEIGHT = _gscfg_f(_GS, 'win1_fitness_weight', 0.05)
WIN_PROFIT_THRESH   = _gscfg_f(_GS, 'win5_profit_thresh',  5.0)   # победа ≥5%
FITNESS_W_WIN_5PCT  = _gscfg_f(_GS, 'win5_fitness_weight', 0.15)
BONUS_PROFIT_THRESH = _gscfg_f(_GS, 'win20_profit_thresh', 20.0)  # бонус ≥20%
FITNESS_W_BONUS_20  = _gscfg_f(_GS, 'win20_fitness_weight',0.10)

# ── [7] Параметры симуляции — берутся из crypto_exchange для идентичности ──────
# Только уникальные для тренера параметры (нет аналога в exchange) берутся из _GS.
# Всё остальное синхронизируется с биржей автоматически.
SNAPSHOT_BARS  = _gscfg_i(_GS, 'snapshot_bars',  24)
GPU_CHUNK_BARS = _gscfg_i(_GS, 'gpu_chunk_bars', 1024)

if _CX_OK:
    # ── Комиссии ────────────────────────────────────────────────────────────────
    TRAIN_FEE         = _cx.SPOT_FEE             # спот-комиссия (taker)
    TRAIN_SLIPPAGE    = _cx.SLIPPAGE             # проскальзывание
    TRAIN_FUTURES_FEE = _cx.FUTURES_FEE          # фьючерсная комиссия
    TRAIN_LIQ_FEE     = _cx.LIQUIDATION_FEE      # штраф при ликвидации
    # ── Торговые параметры ───────────────────────────────────────────────────────
    TRAIN_LEVERAGE    = _cx.LEVERAGE             # плечо фьючерсов
    TRAIN_FRACTION    = _cx.TRADE_FRACTION       # доля капитала на сделку
    TRAIN_MAX_POS     = _gscfg_i(_GS, 'train_max_pos', 10)  # нет прямого аналога в exchange
    # ── Стоп-параметры ───────────────────────────────────────────────────────────
    # FIX: если stop_trade = off в settings — тренер тоже НЕ применяет стоп (TRAIN_STOP_PCT=0)
    TRAIN_STOP_PCT    = (_cx.STOP_TRADE_PCT / 100.0) if _cx.STOP_TRADE_ENABLED else 0.0
    TRAIN_STOP_MODE   = _cx.STOP_TRADE_MODE           # 'from_peak' | 'from_start'
    # FIX: APY применяется ТОЛЬКО при stop_trade_next_period = hold_APY (как в бирже).
    # При режиме 'reset' или 'hold' — после стопа APY не начисляется (0.0).
    _stop_next        = str(getattr(_cx, 'STOP_TRADE_NEXT_PERIOD', 'hold_apy')).lower()
    TRAIN_APY_RATE    = _cx.APY_RATE if _stop_next == 'hold_apy' else 0.0
    # ── Funding rate (бессрочные фьючерсы) ───────────────────────────────────────
    TRAIN_FUNDING_RATE = _cx.FUTURES_FUNDING_RATE     # 0.0001 = 0.01%/8h
    # ── Таймфрейм — нужен для правильной дискретизации funding и APY ──────────────
    TRAIN_BAR         = _cx.BAR                       # 60 для 1m, 1 для 1h
    # ── Начальный капитал ─────────────────────────────────────────────────────────
    TRAIN_INITIAL_CAPITAL = float(_cx.INITIAL_CAPITAL)
    _stop_str = f"{TRAIN_STOP_PCT:.0%} ({TRAIN_STOP_MODE})" if TRAIN_STOP_PCT > 0 else "OFF"
    _apy_str  = f"APY={TRAIN_APY_RATE:.0%}" if TRAIN_APY_RATE > 0 else f"APY=0 (mode={_stop_next})"
    if not _is_worker():
        print(f"  [genetics] Параметры синхронизированы из crypto_exchange: "
              f"lev={TRAIN_LEVERAGE}x  frac={TRAIN_FRACTION:.0%}  "
              f"stop={_stop_str}  fee={TRAIN_FEE}+slip={TRAIN_SLIPPAGE}  "
              f"fr={TRAIN_FUNDING_RATE}  BAR={TRAIN_BAR}  IC={TRAIN_INITIAL_CAPITAL:,.0f}  {_apy_str}")
else:
    # ── Fallback если exchange недоступен ────────────────────────────────────────
    TRAIN_FEE          = _gscfg_f(_GS, 'train_fee',       0.0004)
    TRAIN_SLIPPAGE     = _gscfg_f(_GS, 'train_slippage',  0.0001)
    TRAIN_FUTURES_FEE  = _gscfg_f(_GS, 'train_futures_fee', 0.0002)
    TRAIN_LIQ_FEE      = _gscfg_f(_GS, 'train_liq_fee',   0.005)
    TRAIN_LEVERAGE     = _gscfg_f(_GS, 'train_leverage',  2.0)
    TRAIN_FRACTION     = _gscfg_f(_GS, 'train_fraction',  0.20)
    TRAIN_MAX_POS      = _gscfg_i(_GS, 'train_max_pos',   10)
    TRAIN_STOP_PCT     = _gscfg_f(_GS, 'train_stop_pct',  0.20)
    TRAIN_STOP_MODE    = _GS.get('train_stop_mode', 'from_peak')
    TRAIN_FUNDING_RATE = _gscfg_f(_GS, 'train_funding_rate', 0.0001)
    TRAIN_APY_RATE     = _gscfg_f(_GS, 'train_apy_rate',  0.05)
    TRAIN_BAR          = _gscfg_i(_GS, 'train_bar',       60)     # BAR при fallback
    TRAIN_INITIAL_CAPITAL = _gscfg_f(_GS, 'train_initial_capital', 10000.0)

# ── [8] BC / agent seeding ────────────────────────────────────────────────────
BC_ENABLED      = _gscfg_b(_GS, 'bc_enabled',      False)   # OFF по умолчанию
AGENT_SEED_FRAC = _gscfg_f(_GS, 'agent_seed_frac', 0.20) if BC_ENABLED else 0.0
BC_LR           = _gscfg_f(_GS, 'bc_lr',           0.003)
BC_EPOCHS       = _gscfg_i(_GS, 'bc_epochs',       50)
BC_NOISE        = _gscfg_f(_GS, 'bc_noise',        0.015)
BC_MAX_BARS     = _gscfg_i(_GS, 'bc_max_bars',     4000)
BC_REGIMES_PER_TYPE = _gscfg_i(_GS, 'bc_regimes_per_type', 2)
_asl_raw = _GS.get('agent_seed_list', '').strip()
AGENT_SEED_LIST = [x.strip() for x in _asl_raw.split(',') if x.strip()] if _asl_raw else []

# ── [9] Warm-start / resume ────────────────────────────────────────────────────
CONTINUE_TRAINING    = _gscfg_b(_GS, 'continue_training',    True)
LOAD_BEST_GENOME     = _gscfg_b(_GS, 'load_best_genome',     True)
LOAD_EXTRA_GENOMES   = _gscfg_b(_GS, 'load_extra_genomes',   True)
LOAD_REGIME_GENOMES  = _gscfg_b(_GS, 'load_regime_genomes',  True)
LOAD_ISLAND_GENOMES  = _gscfg_b(_GS, 'load_island_genomes',  True)
LOAD_EXTRA_MAX       = _gscfg_i(_GS, 'load_extra_max',       0)

# ── [10] Прочее ───────────────────────────────────────────────────────────────
ARCHIVE_SIZE   = _gscfg_i(_GS, 'archive_size', 5)
SAVE_EVERY     = _gscfg_i(_GS, 'save_every',   5)
CHART_EVERY    = _gscfg_i(_GS, 'chart_every',  1)
ZERO_THRESH    = _gscfg_f(_GS, 'zero_thresh',  0.05)

MIN_RET_THRESH         = -7.0
INACTIVITY_THRESH      = 0.010
DD_DURATION_THRESH     = 0.03
MAX_DAILY_LOSS_THRESH  = -0.05
BEAR_PENALTY_THRESH    = -5.0

# ── FIX 3+4: anti-overfitting и pos_rate bonus ────────────────────────────────
# Проблема бэктест vs обучение (FINDING 4):
#   fitness=30.25 при backtest avg_ret=+1.40% — Calmar 2–3 outlier-месяцев (76%,62%,66%)
#   перевешивал 54 убыточных периода из 96.
# Проблема FutShort (FINDING 3): GeneticsBearish делал 0 шортов — не было стимула
#   открывать короткие позиции в bearish-периоды.
#
# Решения:
#   1. NEG_STREAK_PEN    — штраф за 3+ убыточных месяца подряд (аналог max drawdown duration)
#   2. POS_RATE_BONUS    — прямой бонус за % прибыльных периодов (как pos% в бэктесте)
#   3. CALMAR_WEIGHT_ADJ — снижает вес Calmar в формуле (уменьшает outlier-dominance)
#   4. SHORT_BEARISH_BONUS — бонус за прибыль в bearish-периодах через шорты
NEG_STREAK_PEN_W    = _gscfg_f(_GS, 'neg_streak_pen_w',    0.30)
NEG_STREAK_THRESH   = _gscfg_i(_GS, 'neg_streak_thresh',   3)     # серия из N подряд убытков
POS_RATE_BONUS_W    = _gscfg_f(_GS, 'pos_rate_bonus_w',    0.80)  # бонус за pos% прибыльных месяцев
CALMAR_WEIGHT_ADJ   = _gscfg_f(_GS, 'calmar_weight_adj',   0.70)  # снижаем влияние outlier-Calmar

# ── [10b] ОДНОФАЗНЫЙ АДАПТИВНЫЙ ФИТНЕС (замена 3-phase системы) ──────────────
# Причина замены: 3-фазная система создавала катастрофический клифф при
# переходе Phase1→Phase2: фитнес Phase1 ~97 (3.5×mean_r), Phase2 ~5 (0.2×mean_r).
# Архив сохранял Phase1-записи с fitness=97, которые никогда не побить в Phase2
# → вечная стагнация → 12+ рестартов → деградация.
# Решение: единая формула + динамические режимные веса.
FITNESS_PHASE2_FRAC = _gscfg_f(_GS, 'fitness_phase2_frac', 0.60)  # legacy, не используется

_TRAIN_PHASE: int = 1   # всегда 1 в однофазном режиме; оставлен для совместимости

# ── [10c] ДИНАМИЧЕСКИЕ РЕЖИМНЫЕ ВЕСА ─────────────────────────────────────────
# Если модель хорошо торгует в режиме X → снижаем вес X (не нужно переобучаться).
# Если плохо торгует в режиме Y → повышаем вес Y (нужно больше обучения).
# Это предотвращает передержку к одному типу рынка.
DYNAMIC_REGIME_WEIGHTS_ENABLED = _gscfg_b(_GS, 'dynamic_regime_weights', True)
DYNAMIC_RW_WINDOW  = _gscfg_i(_GS, 'dynamic_rw_window',  10)    # ген. в скользящем окне
DYNAMIC_RW_BETA    = _gscfg_f(_GS, 'dynamic_rw_beta',    1.0)   # сила коррекции (было 1.5 — слишком агрессивно)
DYNAMIC_RW_MIN     = _gscfg_f(_GS, 'dynamic_rw_min',     0.30)  # мин. мультипликатор веса
DYNAMIC_RW_MAX     = _gscfg_f(_GS, 'dynamic_rw_max',     2.5)   # макс. мультипликатор веса (было 4.0 — нестабильно)
DYNAMIC_RW_TARGET  = _gscfg_f(_GS, 'dynamic_rw_target',  0.5)   # целевой avg return %
# Temp-файл для передачи динамических весов в CPU-воркеры
_DYN_RW_FILE = os.path.join(tempfile.gettempdir(), "genetics_dyn_rw.json")
_PHASE_FILE = os.path.join(tempfile.gettempdir(), "genetics_phase.txt")

# ── [10d] PARETO-RANKING ПО РЕЖИМАМ ─────────────────────────────────────────
PARETO_REGIME_RANKING   = _gscfg_b(_GS, 'pareto_regime_ranking',   True)
PARETO_BONUS_WEIGHT     = _gscfg_f(_GS, 'pareto_bonus_weight',     3.0)

# ── [10e] НАПРАВЛЕННОЕ РАЗНООБРАЗИЕ ─────────────────────────────────────────
DIRECTED_DIVERSITY_ENABLED = _gscfg_b(_GS, 'directed_diversity_enabled', True)
DIRECTED_DIVERSITY_FRAC    = _gscfg_f(_GS, 'directed_diversity_frac',    0.40)


def _set_train_phase(phase: int) -> None:
    """Записывает фазу в глобал + temp-файл для CPU-воркеров."""
    global _TRAIN_PHASE
    _TRAIN_PHASE = phase
    try:
        with open(_PHASE_FILE, 'w') as _pf:
            _pf.write(str(phase))
    except Exception:
        pass


def _get_train_phase() -> int:
    """Читает фазу из файла (для CPU spawn-воркеров) или из глобала."""
    try:
        with open(_PHASE_FILE) as _pf:
            return int(_pf.read().strip())
    except Exception:
        return _TRAIN_PHASE

# ── [11] Выбор оптимальной валюты ─────────────────────────────────────────
CURRENCY_SELECTION_ENABLED = _gscfg_b(_GS, 'currency_selection_enabled', True)
TOP_CURRENCIES_N           = _gscfg_i(_GS, 'top_currencies_n',           0)   # 0=авто
CURRENCY_MOMENTUM_W        = _gscfg_f(_GS, 'currency_momentum_weight',   0.35)
CURRENCY_VOLUME_W          = _gscfg_f(_GS, 'currency_volume_weight',     0.30)
CURRENCY_EFFICIENCY_W      = _gscfg_f(_GS, 'currency_efficiency_weight', 0.35)

# ── [12] Оптимизация обмена (Swap) ─────────────────────────────────────────
SWAP_ENABLED               = _gscfg_b(_GS, 'swap_enabled',               True)
# FIX BUG_SWAP_FEE: было 0.00 → свопы бесплатны в обучении, что нереалистично.
# Агент обучается злоупотреблять одновременными sell+buy, получая нечестное
# преимущество. Установлено 0.30 = 30% от стандартной комиссии (скидка за агрегацию).
SWAP_FEE_MULTIPLIER        = _gscfg_f(_GS, 'swap_fee_multiplier',        0.30)

# РЕЖИМНЫЕ ВЕСА ДЛЯ ВЗВЕШЕННОГО ФИТНЕСА
# 3-label режимы: bearish / neutral / bullish
# bearish = strong_crash + crash + bear  (r < -5%)
# neutral = sideways_down + sideways_up  (-5% ≤ r < +5%)
# bullish = bull + strong_bull           (r ≥ +5%)
REGIME_WEIGHTS = {
    'bearish': 2.5,   # медведи — важно, но не настолько доминировать
    'neutral': 1.5,   # боковик — умеренный приоритет
    'bullish': 1.0,   # бычий — базовый вес (было 0.6 — слишком мало для 60%+ периодов)
}

# 3 острова: каждый специализируется на своём режиме
ISLAND_REGIME_GROUPS = [
    ['bearish'],   # island 0 — bearish specialist
    ['neutral'],   # island 1 — neutral specialist
    ['bullish'],   # island 2 — bullish specialist
]

# Phase 1 веса для каждого острова (используется при phase1_enabled=on)
# При phase1_enabled=off — игнорируется
ISLAND_PHASE1_REGIME_WEIGHTS = {
    0: {'bearish': 1.0, 'neutral': 0.25, 'bullish': 0.08},  # bearish island
    1: {'bearish': 0.35, 'neutral': 1.0, 'bullish': 0.35},  # neutral island
    2: {'bearish': 0.08, 'neutral': 0.25, 'bullish': 1.0},  # bullish island
}

# Быстрые lookup-таблицы: island_id ↔ primary_regime
ISLAND_TO_REGIME: Dict[int, str] = {
    0: 'bearish',
    1: 'neutral',
    2: 'bullish',
}
REGIME_TO_ISLAND: Dict[str, int] = {
    'bearish': 0,
    'neutral': 1,
    'bullish': 2,
}
# Полный набор «профильных» режимов для каждого острова
ISLAND_REGIME_MAP: Dict[int, list] = {
    0: ['bearish'],
    1: ['neutral'],
    2: ['bullish'],
}

# Флаг: агент учится выбирать валюты через нейросеть (cross-sectional rank фичи)
# При CURRENCY_LEARNING_ENABLED = True: hard mask отключается, агент учится сам
CURRENCY_LEARNING_ENABLED = _gscfg_b(_GS, 'currency_learning_enabled', True)

# Минимальная доля каждого режима в BC периодах
BC_MIN_REGIME_FRACS = {
    'bearish': 0.30,
    'neutral': 0.25,
    'bullish': 0.25,
}

def classify_period_regime(avg_coin_return_pct: float) -> str:
    """
    Классифицирует период по среднему % возврату монет → 3-label.
    bearish  = crash+bear  (r < -5%)
    neutral  = sideways    (-5% ≤ r < +5%)
    bullish  = bull+rocket (r ≥ +5%)
    Используется для:
      1. Взвешивания периодов в функции фитнеса (медвежьи важнее)
      2. Режимно-балансированного выбора периодов для BC
      3. entry[6] → _update_regime_ret_history → _dyn_rw (критично!)
    """
    r = avg_coin_return_pct
    if   r < -5.0: return 'bearish'
    elif r <  5.0: return 'neutral'
    else:          return 'bullish'


# ══════════════════════════════════════════════════════════════════════════════
# ВЫБОР ОПТИМАЛЬНОЙ ВАЛЮТЫ — режимно-адаптивный скоринг монет
# ══════════════════════════════════════════════════════════════════════════════

def map_regime_3(r: str) -> str:
    """Конвертирует любой (7-label или 3-label) режим → 3-label."""
    if r in ('bearish', 'neutral', 'bullish'):
        return r
    if r in ('strong_crash', 'crash', 'bear'):
        return 'bearish'
    if r in ('sideways_down', 'sideways_up'):
        return 'neutral'
    if r in ('bull', 'strong_bull'):
        return 'bullish'
    return 'neutral'  # unknown → neutral


def _compute_currency_scores(feat: np.ndarray, regime: str) -> np.ndarray:
    """
    Вычисляет скоринг привлекательности каждой монеты на каждом баре.

    feat:   (T, NC, NF) — предвычисленные признаки периода
    regime: строка режима рынка

    Returns: scores (T, NC) float32
      Высокий скор = монета привлекательна для торговли в данном режиме.
      Используется для маскирования действий агента: открытие позиций
      разрешается только для топ-N монет по скору.
    """
    # Ключевые признаки из предвычисленного тензора
    mom_24h  = feat[:, :, 1].astype(np.float32)   # 24h моментум      [-1, 1]
    mom_1w   = feat[:, :, 3].astype(np.float32)   # 1-недельный mom   [-1, 1]
    rsi      = feat[:, :, 4].astype(np.float32)   # RSI центрированный[-1, 1]
    ema_fast = feat[:, :, 5].astype(np.float32)   # EMA12/48          [-1, 1]
    vol_tr   = feat[:, :, 24].astype(np.float32)  # объёмный тренд    [-1, 1]
    atr_rank = feat[:, :, 22].astype(np.float32)  # ATR rank           [-1, 1]
    mom_eff  = feat[:, :, 23].astype(np.float32)  # momentum/ATR      [-1, 1]

    mw = CURRENCY_MOMENTUM_W
    vw = CURRENCY_VOLUME_W
    ew = CURRENCY_EFFICIENCY_W

    r3 = map_regime_3(regime)
    if r3 == 'bullish':
        # Бычий рынок: лидеры роста — высокий mom + эффективность + объём
        score = (mw * (0.55 * mom_24h + 0.45 * mom_1w) +
                 ew * (0.60 * mom_eff  + 0.40 * ema_fast) +
                 vw * vol_tr)

    elif r3 == 'bearish':
        # Медвежий/обвал: шортируем волатильные падающие
        score = (mw * (-0.55 * mom_24h - 0.45 * mom_1w) +
                 ew * (0.60 * atr_rank + 0.40 * np.abs(mom_eff)) +
                 vw * vol_tr)

    else:  # neutral
        # Боковой: накопление — объём + EMA + не перекуплено
        score = (mw * (0.40 * mom_eff  + 0.60 * ema_fast) +
                 ew * (0.50 * vol_tr   - 0.50 * rsi) +
                 vw * vol_tr)

    return score.astype(np.float32)   # (T, NC)


def _get_top_n_for_regime(nc: int, regime: str, top_n_override: int = 0) -> int:
    """
    Определяет кол-во топ-монет для текущего режима.

    nc             — полный набор монет в периоде
    regime         — строка режима рынка
    top_n_override — если > 0: используем вместо авто-расчёта

    Авто-проценты:
      strong_bull: 45% (концентрация на лидерах)
      bull:        55%
      sideways*:   65% (широкий отбор для боковика)
      bear:        60%
      crash*:      50% (ограниченный выбор для шортов)
    """
    if top_n_override > 0:
        return min(top_n_override, nc)
    ratios = {
        'bearish': 0.55,
        'neutral': 0.65,
        'bullish': 0.50,
    }
    ratio = ratios.get(map_regime_3(regime), 0.60)
    return max(3, int(nc * ratio))


def _apply_currency_selection(actions: np.ndarray,
                               feat:    np.ndarray,
                               regime:  str,
                               top_n:   int = 0) -> np.ndarray:
    """
    Маскирует открытие позиций для монет вне топ-N по скорингу.

    ВАЖНО: при CURRENCY_LEARNING_ENABLED=True функция отключается — агент
    учится самостоятельно выбирать монеты через cross-sectional rank фичи
    (feat[:,26] и feat[:,27]), без жёсткой внешней маски.
    """
    # Если режим обучения выбора валют — не применяем hard mask
    if CURRENCY_LEARNING_ENABLED or not CURRENCY_SELECTION_ENABLED:
        return actions

    T, NC = feat.shape[0], feat.shape[1]
    n = _get_top_n_for_regime(NC, regime, top_n)

    if n >= NC:
        return actions  # все монеты разрешены

    scores = _compute_currency_scores(feat, regime)   # (T, NC)

    # Для каждого бара: маска разрешённых монет (топ-n по скору)
    # argpartition быстрее full sort для больших NC
    if n < NC:
        part_idx  = np.argpartition(scores, NC - n, axis=1)[:, NC - n:]  # (T, n)
        mask = np.zeros((T, NC), dtype=bool)
        np.put_along_axis(mask, part_idx, True, axis=1)
    else:
        mask = np.ones((T, NC), dtype=bool)

    # Применяем маску: только для действий открытия позиций
    result = actions.copy()
    blocked = ~mask[np.newaxis, :, :]                            # (1, T, NC) → broadcast
    # BUG-FIX #9: было (==1)|(==3)|(==4) — action=3 это sell_spot (ВЫХОД), не вход!
    # action=4 = fut_long_half (не fut_short как в комментарии).
    # Все entry-действия: 1=buy_half, 2=buy_full, 4=fl_half, 5=fl_full, 6=fs_half, 7=fs_full
    open_actions = ((result >= 1) & (result <= 2)) | ((result >= 4) & (result <= 7))
    result[blocked & open_actions] = 0                           # → hold
    return result

# ── BC (Behavioral Cloning) ───────────────────────────────────────────────────
# Все BC константы загружены из settings_genetic.txt (раздел [8])
# BC_ENABLED, AGENT_SEED_FRAC, BC_LR, BC_EPOCHS, BC_NOISE, BC_MAX_BARS,
# BC_REGIMES_PER_TYPE, AGENT_SEED_LIST — уже определены выше.
N_WORKERS = 0   # 0 = auto (GPU если доступен, иначе CPU)

# ── Геном ─────────────────────────────────────────────────────────────────────
_W1s = N_INPUT * N_HIDDEN1;    _b1s = N_HIDDEN1
_W2s = N_HIDDEN1 * N_HIDDEN2;  _b2s = N_HIDDEN2
_W3s = N_HIDDEN2 * N_HIDDEN3;  _b3s = N_HIDDEN3   # 3-й скрытый слой
_W4s = N_HIDDEN3 * N_ACTIONS;  _b4s = N_ACTIONS    # выходной слой
GENOME_SIZE = _W1s + _b1s + _W2s + _b2s + _W3s + _b3s + _W4s + _b4s

# Границы блоков для per-layer sigma и блочного кроссовера (4 блока вместо 3)
_BLOCK_BOUNDS = [
    (0,                                    _W1s + _b1s),
    (_W1s + _b1s,                          _W1s + _b1s + _W2s + _b2s),
    (_W1s + _b1s + _W2s + _b2s,           _W1s + _b1s + _W2s + _b2s + _W3s + _b3s),
    (_W1s + _b1s + _W2s + _b2s + _W3s + _b3s, GENOME_SIZE),
]

import platform

# ══════════════════════════════════════════════════════════════════════════════
# ИМПОРТ БИРЖИ
# ══════════════════════════════════════════════════════════════════════════════
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import crypto_exchange as _cx
except ImportError as e:
    print(f"[ERROR] Не найден модуль ретрорасчёта Retrodate_cryptotrade/crypto_exchange.py: {e}")
    sys.exit(1)

AGENTS_DIR = os.path.join(_here, "Agents", "genetics")
os.makedirs(AGENTS_DIR, exist_ok=True)
IS_WINDOWS = platform.system() == 'Windows'

_GENETICS_AGENT_REPORTED = False
REGIME_ORDER = ['bearish', 'neutral', 'bullish']

def island_file(island_id: int) -> str:
    """Путь к файлу лучшего генома острова. Имя = режим рынка острова."""
    regime = ISLAND_TO_REGIME.get(island_id, f"island{island_id}")
    return os.path.join(AGENTS_DIR, f"best_island_{regime}.npy")

# ══════════════════════════════════════════════════════════════════════════════
# MLP с ELU активацией (без изменений vs v4)
# ══════════════════════════════════════════════════════════════════════════════

def _elu_np(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(np.clip(x, -20, 0)) - 1.0))

def _unpack(genome: np.ndarray):
    i = 0
    W1 = genome[i:i+_W1s].reshape(N_INPUT,   N_HIDDEN1); i += _W1s; b1 = genome[i:i+_b1s]; i += _b1s
    W2 = genome[i:i+_W2s].reshape(N_HIDDEN1, N_HIDDEN2); i += _W2s; b2 = genome[i:i+_b2s]; i += _b2s
    W3 = genome[i:i+_W3s].reshape(N_HIDDEN2, N_HIDDEN3); i += _W3s; b3 = genome[i:i+_b3s]; i += _b3s
    W4 = genome[i:i+_W4s].reshape(N_HIDDEN3, N_ACTIONS); i += _W4s; b4 = genome[i:i+_b4s]
    return W1, b1, W2, b2, W3, b3, W4, b4

def _fwd_np(x, W1, b1, W2, b2, W3, b3, W4, b4):
    h1 = _elu_np(x @ W1 + b1)
    h2 = _elu_np(h1 @ W2 + b2)
    h3 = _elu_np(h2 @ W3 + b3)
    z  = h3 @ W4 + b4
    z -= z.max(axis=-1, keepdims=True)
    e  = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def _xavier_pop(rng: np.random.Generator, size: int) -> np.ndarray:
    pop = np.zeros((size, GENOME_SIZE), dtype=np.float32)
    idx = 0
    for fi, fo in [(N_INPUT, N_HIDDEN1), (N_HIDDEN1, N_HIDDEN2),
                   (N_HIDDEN2, N_HIDDEN3), (N_HIDDEN3, N_ACTIONS)]:
        nw = fi * fo
        s  = np.sqrt(2.0 / fi)
        pop[:, idx:idx+nw] = rng.normal(0, s, (size, nw)).astype(np.float32)
        idx += nw + fo   # skip biases (remain 0)
    return pop

# ══════════════════════════════════════════════════════════════════════════════
# ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def precompute_features(dfp, dfv, month: int):
    BAR = _cx.BAR
    syms = sorted(dfp.columns.tolist())
    T, NC = len(dfp), len(syms)
    feat   = np.zeros((T, NC, N_INPUT), dtype=np.float32)
    pm     = dfp[syms].values.astype(np.float64)
    vm     = dfv[syms].values.astype(np.float64)
    prices = pm.astype(np.float32)

    def _mom(lb):
        lb = min(lb, T - 1)
        out = np.zeros((T, NC), dtype=np.float32)
        if lb == 0: return out
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(pm[:-lb] > 0, (pm[lb:] - pm[:-lb]) / pm[:-lb], 0.0)
        out[lb:] = np.clip(r, -1, 1)
        return out

    feat[:,:,0] = _mom(BAR)
    feat[:,:,1] = _mom(24*BAR)
    feat[:,:,2] = _mom(72*BAR)
    feat[:,:,3] = _mom(168*BAR)

    # RSI
    period = 14
    if T > period:
        delta = np.diff(pm, axis=0)
        gain  = np.maximum(delta, 0.0)
        loss  = np.maximum(-delta, 0.0)
        ag = np.zeros((T-1, NC)); al = np.zeros((T-1, NC))
        ag[0] = gain[:period].mean(axis=0)
        al[0] = loss[:period].mean(axis=0)
        alpha_r = 1.0 / period
        for t in range(1, T-1):
            ag[t] = (1-alpha_r)*ag[t-1] + alpha_r*gain[t]
            al[t] = (1-alpha_r)*al[t-1] + alpha_r*loss[t]
        rsi_v = 100.0 - 100.0 / (1.0 + ag/(al+1e-9))
        feat[1:,:,4] = np.clip((rsi_v - 50)/50.0, -1, 1).astype(np.float32)

    # EMA ratios
    def _ema_ratio(sf, ss):
        af = 2.0/(sf+1); as_ = 2.0/(ss+1)
        ef = pm[0].copy(); es = pm[0].copy()
        ratio = np.zeros((T, NC), dtype=np.float32)
        for t in range(T):
            ef = af*pm[t] + (1-af)*ef
            es = as_*pm[t] + (1-as_)*es
            ratio[t] = np.clip((ef-es)/(np.abs(es)+1e-9), -1, 1)
        return ratio

    feat[:,:,5] = _ema_ratio(12, 48)
    feat[:,:,6] = _ema_ratio(48, 168)

    _win_v = min(24 * BAR, T)
    _cs_v  = np.vstack([np.zeros((1, NC), dtype=np.float64), np.cumsum(vm, axis=0)])
    _t_arr = np.arange(1, T + 1, dtype=np.int64)
    _lb_v  = np.minimum(_t_arr, _win_v)
    _lo_v  = (_t_arr - _lb_v).clip(0)
    _vol_avg = (_cs_v[_t_arr] - _cs_v[_lo_v]) / np.maximum(_lb_v[:, None], 1) + 1e-9
    feat[1:, :, 7] = np.clip(vm[1:] / _vol_avg[:-1] - 1, -3, 3).astype(np.float32)

    # Price in 24h range
    feat[:,:,8] = 0.5
    _win_p = min(24 * BAR, T)
    _t_arr2 = np.arange(1, T + 1, dtype=np.int64)
    _lb_p   = np.minimum(_t_arr2, _win_p)
    _lo_p   = (_t_arr2 - _lb_p).clip(0)
    # rolling min/max: для каждого t берём pm[max(0,t-lb):t+1]
    # Полностью векторизовать rolling min/max без scipy сложно для переменного окна,
    # поэтому используем stride_tricks для фиксированного окна после прогрева:
    for t in range(1, min(_win_p, T)):   # только прогрев (короткий, max 24*BAR итераций)
        win = pm[:t+1]
        mn, mx = win.min(axis=0), win.max(axis=0)
        feat[t,:,8] = np.clip((pm[t]-mn)/(mx-mn+1e-9), 0, 1).astype(np.float32)
    if T > _win_p:                        # после прогрева — только скользящее окно
        from numpy.lib.stride_tricks import sliding_window_view
        _pw = sliding_window_view(pm, (_win_p + 1, NC))[:, 0, :, :]  # (T-_win_p, _win_p+1, NC)
        _mn = _pw.min(axis=1); _mx = _pw.max(axis=1)
        feat[_win_p:, :, 8] = np.clip(
            (pm[_win_p:] - _mn) / (_mx - _mn + 1e-9), 0, 1
        ).astype(np.float32)

    feat[:,:,19] = np.float32(np.sin(2*np.pi*month/12))
    feat[:,:,20] = np.float32(np.cos(2*np.pi*month/12))

    # ──  УЛУЧШЕННОЕ ДЕТЕКТИРОВАНИЕ РЕЖИМОВ ────────────────────────────────
    # Три таймфрейма: 168б (1 нед) + 336б (2 нед) + 672б (4 нед)
    # EMA-сглаживание → нет ложных смен режима от одного шумного бара
    # Тренд-брэдт (% активов с позитивным composite) → подтверждение большинства
    # feat 21 = уверенность в режиме (расстояние до ближайшего порога)
    # ──────────────────────────────────────────────────────────────────────────
    m168 = feat[:,:,3]   # 168-барный моментум, уже вычислен выше

    # 1. Дополнительные таймфреймы: 2 недели и 4 недели
    lb_336 = min(336 * BAR, T - 1)
    lb_672 = min(672 * BAR, T - 1)
    m336_raw = np.zeros((T, NC), dtype=np.float32)
    m672_raw = np.zeros((T, NC), dtype=np.float32)
    if lb_336 > 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            r336 = np.where(pm[:-lb_336] > 0,
                            (pm[lb_336:] - pm[:-lb_336]) / pm[:-lb_336], 0.)
        m336_raw[lb_336:] = np.clip(r336, -1., 1.).astype(np.float32)
    if lb_672 > 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            r672 = np.where(pm[:-lb_672] > 0,
                            (pm[lb_672:] - pm[:-lb_672]) / pm[:-lb_672], 0.)
        m672_raw[lb_672:] = np.clip(r672, -1., 1.).astype(np.float32)

    # 2. EMA-сглаженный мульти-таймфреймовый композит (per-asset)
    #    Вес 50/30/20: краткосрочный даёт реактивность, долгосрочные убирают шум
    _reg_ema_bars = max(24 * BAR, 24)   # период сглаживания ≈ 1 торговый день
    _ra  = np.float32(2.0 / (_reg_ema_bars + 1))
    _ra1 = np.float32(1.0 - _ra)
    composite_score = np.zeros((T, NC), dtype=np.float32)
    for t in range(1, T):
        raw_t = (0.50 * m168[t] + 0.30 * m336_raw[t] + 0.20 * m672_raw[t])
        composite_score[t] = _ra * raw_t + _ra1 * composite_score[t - 1]

    # 3. EMA тренд-брэдт: % активов с положительным composite score
    #    Стабильнее шумного 24h price breadth — отражает трендовое большинство
    _ba  = np.float32(2.0 / (max(12 * BAR, 12) + 1))
    _ba1 = np.float32(1.0 - _ba)
    ema_trend_brd    = np.zeros(T, dtype=np.float32)
    ema_trend_brd[0] = np.float32(0.5)
    for t in range(1, T):
        inst             = np.float32((composite_score[t] > 0.).mean())
        ema_trend_brd[t] = _ba * inst + _ba1 * ema_trend_brd[t - 1]

    # Пороги классификации и минимальный прогрев
    # Окно истории: вычисляется в __init__ — BAR может измениться после импорта.
    # Период обучения = 1 месяц ≈ 43200 баров → режимные фичи 9-15 почти
    # никогда не получали ненулевых значений в обучении (только последние 2 дня),
    # тогда как в live они появляются после накопления 28 дней истории.
    # Фикс: warmup = 48*BAR (2 дня) — достаточно для сглаживания EMA, но
    # режимные фичи теперь активны почти весь период обучения.
    _regime_warmup  = max(48 * BAR, 48)
    _reg_thresholds = np.array([-0.025, 0.015], dtype=np.float32)  # 3-label: bearish/neutral/bullish
    last_reg  = np.zeros(3, dtype=np.float32)
    last_conf = np.float32(0.0)

    for t0 in range(0, T, max(1, BAR)):
        t1 = min(t0 + BAR, T)

        # feat 17: EMA тренд-брэдт (заменяет 24h price breadth)
        feat[t0:t1, :, 17] = ema_trend_brd[t0]

        # feat 18: глобальная волатильность
        if t0 >= 50:
            rr = np.diff(pm[max(0, t0 - 50):t0 + 1], axis=0)
            bp = pm[max(0, t0 - 50):t0]
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(bp > 0, rr / bp, 0.0)
            gv = float(np.std(r[np.isfinite(r)])) / 0.02
            feat[t0:t1, :, 18] = np.float32(min(gv, 5.0))

        # feat 9-15 (one-hot режим) и feat 21 (confidence) — только после прогрева
        if t0 >= _regime_warmup:
            cs     = float(composite_score[t0].mean())
            br     = float(ema_trend_brd[t0])
            # Брэдт-коррекция: сдвигает сигнал в сторону большинства активов
            signal = cs + (br - 0.5) * 0.06

            # 3-label one-hot: 0=bearish 1=neutral 2=bullish
            if   signal < -0.025: rid = 0   # bearish  (strong_crash+crash+bear)
            elif signal <  0.015: rid = 1   # neutral  (sideways_down+sideways_up)
            else:                 rid = 2   # bullish  (bull+strong_bull)

            # Confidence: расстояние до ближайшего порога
            _r3_thresholds = np.array([-0.025, 0.015], dtype=np.float32)
            conf = float(np.min(np.abs(signal - _r3_thresholds))) / 0.025
            conf = float(np.clip(conf, 0.0, 1.0))

            last_reg      = np.zeros(3, dtype=np.float32)
            last_reg[rid] = 1.0
            last_conf     = np.float32(conf)

        feat[t0:t1, :, 9:12] = last_reg[None, None, :]
        feat[t0:t1, :, 21]   = last_conf

    # Noise
    if T >= 100:
        rr = np.diff(pm[:min(T,1000),:min(NC,10)], axis=0)
        bp = pm[:min(T,1000)-1,:min(NC,10)]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(bp>0, rr/bp, 0.0)
        r = r[np.isfinite(r)]
        if len(r) > 10:
            s = float(np.std(r)) + 1e-9
            noise = float((np.abs(r) > 3*s).mean()) / 0.05 * 4.0 / 10.0
            feat[:,:,16] = np.float32(min(noise, 1.0))

    # ATR-rank (feat 22) и momentum efficiency (feat 23)
    atr_window = max(14*BAR, 14)
    atr_matrix = np.zeros((T, NC), dtype=np.float32)
    if T > atr_window + 1:
        from numpy.lib.stride_tricks import sliding_window_view
        _sw = sliding_window_view(pm, (atr_window + 1, NC))[:, 0, :, :]  # (T-atr_window, win+1, NC)
        _hi = _sw.max(axis=1); _lo = _sw.min(axis=1)
        _mid = (_hi + _lo) / 2.0 + 1e-9
        atr_vals = ((_hi - _lo) / _mid).astype(np.float32)  # (T-atr_window, NC)
        atr_matrix[atr_window:] = atr_vals
        # ATR rank — argsort of argsort per row
        _ranks = np.argsort(np.argsort(atr_vals, axis=1), axis=1).astype(np.float32)
        _ranks /= (NC - 1 + 1e-9)
        feat[atr_window:, :, 22] = np.clip(_ranks * 2.0 - 1.0, -1.0, 1.0)
    else:
        for t in range(atr_window, T):  # fallback для коротких периодов
            lb  = max(1, min(atr_window, t))
            win = pm[t-lb:t+1]
            hi  = win.max(axis=0); lo = win.min(axis=0); mid = win.mean(axis=0) + 1e-9
            atr_matrix[t] = (hi - lo) / mid
            ranks = np.argsort(np.argsort(atr_matrix[t])).astype(np.float32) / (NC - 1 + 1e-9)
            feat[t,:,22] = np.clip(ranks * 2.0 - 1.0, -1.0, 1.0)

    # Momentum efficiency
    if T > 24*BAR + 1:
        mom_24   = feat[24*BAR+1:, :, 1]
        atr_rows = atr_matrix[24*BAR+1:]
        atr_rel  = (atr_rows / (atr_rows.mean(axis=1, keepdims=True) + 1e-9)).clip(0.01, 10.0)
        feat[24*BAR+1:, :, 23] = np.clip(mom_24 / atr_rel, -2.0, 2.0).astype(np.float32) / 2.0

    # Feature 24: volume EMA trend
    ema12_vol = np.zeros((T, NC), dtype=np.float64)
    ema48_vol = np.zeros((T, NC), dtype=np.float64)
    a12 = 2.0 / (12*BAR + 1); a48 = 2.0 / (48*BAR + 1)
    ema12_vol[0] = vm[0]; ema48_vol[0] = vm[0]
    for t in range(1, T):
        ema12_vol[t] = a12 * vm[t] + (1-a12) * ema12_vol[t-1]
        ema48_vol[t] = a48 * vm[t] + (1-a48) * ema48_vol[t-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_trend = np.where(ema48_vol > 0, (ema12_vol / (ema48_vol + 1e-9)) - 1.0, 0.0)
    feat[:,:,24] = np.clip(vol_trend, -2.0, 2.0).astype(np.float32) / 2.0

    # Feature 25: week range position
    week_bars = 7 * 24 * BAR
    feat[:,:,25] = 0.5
    for t in range(1, min(week_bars, T)):   # только прогрев
        lb  = max(1, t)
        win = pm[:t+1]
        mn, mx = win.min(axis=0), win.max(axis=0)
        feat[t,:,25] = np.clip((pm[t]-mn)/(mx-mn+1e-9), 0, 1).astype(np.float32)
    if T > week_bars:
        from numpy.lib.stride_tricks import sliding_window_view
        _ww = sliding_window_view(pm, (week_bars + 1, NC))[:, 0, :, :]
        _mn7 = _ww.min(axis=1); _mx7 = _ww.max(axis=1)
        feat[week_bars:, :, 25] = np.clip(
            (pm[week_bars:] - _mn7) / (_mx7 - _mn7 + 1e-9), 0, 1
        ).astype(np.float32)

    # ── НОВЫЕ ФИЧИ: cross-sectional rank для обучения выбора валют ─────────────
    # Feature 26: ранг монеты по 24h-моментуму среди всех монет периода (0=хуже всех, 1=лучшая)
    # Позволяет нейросети видеть "я — топ монета по росту" vs "я — аутсайдер"
    if NC > 1:
        m24h_cs = feat[:, :, 1].copy()   # (T, NC) — 24h momentum
        # argsort of argsort = rank, затем нормируем в [0,1]
        _cs_ranks = np.argsort(np.argsort(m24h_cs, axis=1), axis=1).astype(np.float32)
        feat[:, :, 26] = _cs_ranks / (NC - 1 + 1e-9)

    # Feature 27: ранг монеты по volume EMA trend среди всех монет
    if NC > 1:
        vol_tr_cs = feat[:, :, 24].copy()  # (T, NC) — volume trend (feat 24)
        _vr_ranks = np.argsort(np.argsort(vol_tr_cs, axis=1), axis=1).astype(np.float32)
        feat[:, :, 27] = _vr_ranks / (NC - 1 + 1e-9)

    return feat, prices, syms


# ══════════════════════════════════════════════════════════════════════════════
# [ИСПРАВЛЕНО] СИМУЛЯТОР — Numba JIT (ускорение 50-100x vs Python-цикл)
# ══════════════════════════════════════════════════════════════════════════════
#
# БАГИ ИСПРАВЛЕНЫ:
#   BUG 1 (КРИТИЧЕСКИЙ): for t in range(T) в Python — 43,000 итераций × 60 периодов
#   = 2.58М Python-вызовов за поколение. На Windows Python-цикл ≈ 5-15 мин/ген.
#   С Numba @njit(parallel=True): prange по G (агентам) → 50-100x ускорение.
#   При 32 CPU-ядрах: ~30-60 секунд вместо 20+ минут.
#
# Numba компилирует функцию при ПЕРВОМ вызове (~10-15 сек), затем — JIT-скорость.

if _NUMBA_OK:
    @njit(parallel=True, cache=True, fastmath=True)
    def _sim_core(actions_arr, prices, initial_capital, snap_every,
                  fee_total, fraction, leverage, stop_pct, max_pos, n_snaps,
                  swap_fee_total, funding_rate, stop_from_peak, apy_per_step,
                  liq_fee, bar_int):
        """
        Numba JIT ядро симулятора. prange(G) → каждый агент в своём потоке.
        Внутренний цикл for t in range(T) — компилируется в нативный код.

        Идентично crypto_exchange.run() + Exchange._liq_check():
          - Ликвидация фьючерсов (margin+pnl <= 0) + штраф liq_fee
          - Funding rate каждые 8*bar_int шагов (8 часов) — FIX BUG5
          - Portfolio stop: from_peak (просадка от ATH) или from_start (от IC)
          - stop_pct — ПОЛОЖИТЕЛЬНЫЙ (0.20 = 20% просадки → стоп)
          - APY на кэш пока агент остановлен (hold_APY режим) — FIX BUG6
          - action=6 — swap_buy: пониженная комиссия swap_fee_total
          bar_int: число баров в одном часе (BAR из exchange). 60 для 1m, 1 для 1h.
        """
        G  = actions_arr.shape[0]
        T  = actions_arr.shape[1]
        NC = actions_arr.shape[2]
        # FIX BUG5: funding interval = 8 часов = 8*BAR баров (как в Exchange.step())
        funding_interval = 8 * max(bar_int, 1)

        final_rets  = np.empty(G, dtype=np.float64)
        max_dds     = np.empty(G, dtype=np.float64)
        trade_rates = np.empty(G, dtype=np.float64)

        if n_snaps > 0:
            daily_pv = np.zeros((G, n_snaps), dtype=np.float32)
        else:
            daily_pv = np.zeros((G, 1), dtype=np.float32)

        for g in prange(G):               # ← параллельный цикл по агентам
            cash     = float(initial_capital)
            s_qty    = np.zeros(NC, dtype=np.float64)
            f_qty    = np.zeros(NC, dtype=np.float64)
            f_entry  = np.zeros(NC, dtype=np.float64)
            has_spot = np.zeros(NC, dtype=numba.boolean)
            has_fut  = np.zeros(NC, dtype=numba.boolean)
            stopped  = False
            peak_pv  = float(initial_capital)
            max_dd   = 0.0
            trade_cnt = 0

            for t in range(T):            # ← нативный C-цикл (не Python!)

                # ── Ликвидация фьючерсов (идентично _liq_check биржи) ────────────
                for c in range(NC):
                    if has_fut[c]:
                        mg  = abs(f_qty[c] * f_entry[c]) / leverage
                        pnl = f_qty[c] * (prices[t, c] - f_entry[c])
                        if mg + pnl <= 0.0:
                            penalty = mg * liq_fee
                            cash -= penalty
                            f_qty[c] = 0.0; f_entry[c] = 0.0; has_fut[c] = False

                # ── Funding rate каждые 8*BAR шагов = 8 часов (FIX BUG5) ─────────
                if funding_rate > 0.0 and t > 0 and t % funding_interval == 0:
                    for c in range(NC):
                        if has_fut[c]:
                            notional    = abs(f_qty[c] * prices[t, c])
                            fund_charge = notional * funding_rate
                            if f_qty[c] > 0.0:
                                cash -= fund_charge   # лонги платят
                            else:
                                cash += fund_charge   # шорты получают

                # ── Стоимость портфеля ───────────────────────────────────────────
                pv = cash
                for c in range(NC):
                    pv += s_qty[c] * prices[t, c]
                    if has_fut[c]:
                        pv += f_qty[c] * (prices[t, c] - f_entry[c])

                if pv > peak_pv:
                    peak_pv = pv
                if peak_pv > 0.0:
                    dd = (peak_pv - pv) / peak_pv * 100.0
                    if dd > max_dd:
                        max_dd = dd

                # ── Daily snapshot ───────────────────────────────────────────────
                if n_snaps > 0 and snap_every > 0:
                    si = t // snap_every
                    if si < n_snaps and (t + 1) % snap_every == 0:
                        daily_pv[g, si] = np.float32(max(pv, 0.0))

                # ── Портфельный стоп (идентично run() в crypto_exchange) ─────────
                # stop_pct ПОЛОЖИТЕЛЬНЫЙ: 0.20 = остановить при просадке 20%
                # stop_from_peak=1: from_peak (ATH), 0: from_start (от IC)
                if not stopped:
                    if stop_from_peak:
                        triggered = peak_pv > 0.0 and pv < peak_pv * (1.0 - stop_pct)
                    else:
                        triggered = pv < initial_capital * (1.0 - stop_pct)
                    if triggered:
                        for c in range(NC):
                            if has_spot[c]:
                                cash += s_qty[c] * prices[t, c] * (1.0 - fee_total)
                                s_qty[c] = 0.0; has_spot[c] = False
                            if has_fut[c]:
                                mg   = abs(f_qty[c] * f_entry[c]) / leverage
                                pnl  = f_qty[c] * (prices[t, c] - f_entry[c])
                                cash += (mg + pnl) * (1.0 - fee_total)
                                f_qty[c] = 0.0; f_entry[c] = 0.0; has_fut[c] = False
                        stopped = True

                if stopped:
                    # APY на кэш (идентично hold_APY в run() биржи)
                    if apy_per_step > 0.0:
                        cash *= (1.0 + apy_per_step)
                    continue

                # ── Кол-во открытых позиций ──────────────────────────────────────
                # n_pos = кол-во монет с хотя бы одной позицией (спот ИЛИ фут)
                n_pos = 0
                for c in range(NC):
                    if has_spot[c] or has_fut[c]:
                        n_pos += 1

                # BUG-FIX 1+2: ts пересчитывается на каждой монете от pv.

                # ── Обработка действий (spot+fut независимы, усреднение разрешено) ─
                for c in range(NC):
                    a = actions_arr[g, t, c]
                    _ts_raw = pv * fraction if pv > 0.0 else 0.0
                    ts = _ts_raw if cash >= _ts_raw else (cash if cash > 0.0 else 0.0)

                    # ── 9-action scheme ────────────────────────────────────────
                    # 1=buy_half(50%)  2=buy_full(100%)  3=sell_spot
                    # 4=fl_half  5=fl_full  6=fs_half  7=fs_full  8=close_fut
                    # 9=swap_buy (internal swap-optimisation, trainer only)

                    if a in (1, 2) and prices[t, c] > 0.0 and ts > 0.0:
                        # buy_spot (half or full): усреднение разрешено
                        ts_a = ts * 0.5 if a == 1 else ts
                        if not has_spot[c]:
                            if n_pos >= max_pos: continue
                            n_pos += 1
                        qty = ts_a / (prices[t, c] * (1.0 + fee_total))
                        cash -= ts_a; s_qty[c] += qty; has_spot[c] = True; trade_cnt += 1

                    elif a == 3 and has_spot[c]:
                        # sell_spot: закрываем всё
                        cash += s_qty[c] * prices[t, c] * (1.0 - fee_total)
                        s_qty[c] = 0.0; has_spot[c] = False
                        if not has_fut[c]: n_pos -= 1
                        trade_cnt += 1

                    elif a in (4, 5) and prices[t, c] > 0.0 and ts > 0.0:
                        # fut_long (half/full): усреднение разрешено, конфликт с шортом запрещён
                        if f_qty[c] < 0.0: continue
                        ts_a = ts * 0.5 if a == 4 else ts
                        if not has_fut[c]:
                            if n_pos >= max_pos: continue
                            n_pos += 1
                        new_qty = ts_a * leverage / (prices[t, c] * (1.0 + fee_total))
                        old_abs = abs(f_qty[c])
                        if old_abs > 0.0:
                            f_entry[c] = (f_entry[c]*old_abs + prices[t,c]*new_qty)/(old_abs+new_qty)
                        else:
                            f_entry[c] = prices[t, c]
                        f_qty[c] += new_qty; cash -= ts_a; has_fut[c] = True; trade_cnt += 1

                    elif a in (6, 7) and prices[t, c] > 0.0 and ts > 0.0:
                        # fut_short (half/full): усреднение разрешено, конфликт с лонгом запрещён
                        if f_qty[c] > 0.0: continue
                        ts_a = ts * 0.5 if a == 6 else ts
                        if not has_fut[c]:
                            if n_pos >= max_pos: continue
                            n_pos += 1
                        new_qty = ts_a * leverage / (prices[t, c] * (1.0 + fee_total))
                        old_abs = abs(f_qty[c])
                        if old_abs > 0.0:
                            f_entry[c] = (f_entry[c]*old_abs + prices[t,c]*new_qty)/(old_abs+new_qty)
                        else:
                            f_entry[c] = prices[t, c]
                        f_qty[c] -= new_qty; cash -= ts_a; has_fut[c] = True; trade_cnt += 1

                    elif a == 8 and has_fut[c]:
                        # close_fut
                        mg  = abs(f_qty[c] * f_entry[c]) / leverage
                        pnl = f_qty[c] * (prices[t, c] - f_entry[c])
                        cash += (mg + pnl) * (1.0 - fee_total)
                        f_qty[c] = 0.0; f_entry[c] = 0.0; has_fut[c] = False
                        if not has_spot[c]: n_pos -= 1
                        trade_cnt += 1

                    elif a == 9 and prices[t, c] > 0.0 and ts > 0.0:
                        # swap_buy (internal): не усредняем — только новая позиция
                        if has_spot[c]: continue
                        if n_pos >= max_pos: continue
                        qty = ts / (prices[t, c] * (1.0 + swap_fee_total))
                        cash -= ts; s_qty[c] += qty; has_spot[c] = True
                        n_pos += 1; trade_cnt += 1

            # ── Финальная ликвидация ─────────────────────────────────────────────
            for c in range(NC):
                if has_spot[c]:
                    cash += s_qty[c] * prices[T-1, c] * (1.0 - fee_total)
                if has_fut[c]:
                    mg   = abs(f_qty[c] * f_entry[c]) / leverage
                    pnl  = f_qty[c] * (prices[T-1, c] - f_entry[c])
                    cash += (mg + pnl) * (1.0 - fee_total)

            final_pv         = max(cash, 0.0)
            final_rets[g]    = (final_pv / initial_capital - 1.0) * 100.0
            max_dds[g]       = max_dd
            trade_rates[g]   = trade_cnt / max(float(T * NC), 1.0)

        return final_rets, max_dds, trade_rates, daily_pv

    def _warm_up_numba():
        """Прогрев JIT на маленьком примере — компиляция до начала тренировки."""
        dummy_act  = np.zeros((2, 10, 3), dtype=np.int8)
        dummy_p    = np.ones((10, 3), dtype=np.float64)
        _sim_core(
            dummy_act, dummy_p,
            float(TRAIN_INITIAL_CAPITAL), 5,
            float(TRAIN_FEE + TRAIN_SLIPPAGE),
            float(TRAIN_FRACTION),
            float(TRAIN_LEVERAGE),
            float(TRAIN_STOP_PCT),
            int(TRAIN_MAX_POS), 2,
            float((TRAIN_FEE + TRAIN_SLIPPAGE) * SWAP_FEE_MULTIPLIER),
            float(TRAIN_FUNDING_RATE),
            int(1 if TRAIN_STOP_MODE == 'from_peak' else 0),
            # FIX BUG6: APY делится на BAR (баров в часе), а не только на 365*24
            # Биржа: APY_RATE / (365 * 24 * BAR). TRAIN_BAR=60 при 1m, =1 при 1h.
            float(TRAIN_APY_RATE / (365 * 24 * max(1, TRAIN_BAR))),
            float(TRAIN_LIQ_FEE),
            int(TRAIN_BAR),   # FIX BUG5: передаём BAR для funding interval = 8*BAR
        )
        print("  [Numba] JIT прогрев завершён — симулятор скомпилирован")


# ══════════════════════════════════════════════════════════════════════════════
# ОПТИМИЗАЦИЯ ОБМЕНА (SWAP) — конвертация sell+buy в одну операцию
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_swaps(actions_arr: np.ndarray) -> np.ndarray:
    """
    Детектирует одновременные продажи и покупки на одном шаге времени
    и конвертирует их в своп-операцию (одна комиссия вместо двух).

    ВАЖНО: action=6 (swap_buy) используется ТОЛЬКО внутри Numba-тренера.
    В crypto_exchange.py action=6 означает close_all (аварийное закрытие).
    GeneticsAgent.act() никогда не выставляет action=6 при инференсе.

    Логика:
      На каждом шаге t, для каждого агента g:
        Если есть sell_spot (action=2) И buy_spot (action=1) одновременно →
        buy_spot меняется на swap_buy (action=6).

      В симуляторе:
        action=2 (sell): исполняется с полной комиссией fee_total
        action=6 (swap_buy): исполняется с swap_fee = fee_total * SWAP_FEE_MULTIPLIER
        Итого: (1 + SWAP_FEE_MULTIPLIER) × fee_total  vs  2 × fee_total обычно

    Полностью векторизованная операция — без Python-циклов по G и T.
    """
    if not SWAP_ENABLED:
        return actions_arr

    # has_sell[g, t] = True если есть хоть одна продажа спота в этом (g, t)
    # FIX BUG_SWAP: sell_spot = action 3 (не 2!). action=2 = buy_full.
    has_sell = np.any(actions_arr == 3, axis=2)   # (G, T) bool  ← было ==2
    # has_buy[g, t]  = True если есть хоть одна покупка buy_half в этом (g, t)
    has_buy  = np.any(actions_arr == 1, axis=2)   # (G, T) bool
    # Своп только там где sell+buy одновременно (sell → swap-buy = одна комиссия)
    is_swap  = has_sell & has_buy                  # (G, T) bool

    if not is_swap.any():
        return actions_arr                          # быстрый выход — нет свопов

    result = actions_arr.copy()
    # Меняем action=1 (buy_half) → action=9 (swap_buy) только на своп-шагах.
    # FIX BUG_SWAP: swap_buy = action 9 (не 6!). action=6 = fut_short_half.
    # Broadcast: is_swap (G, T) → (G, T, 1) → (G, T, NC)
    is_swap_buy = (actions_arr == 1) & is_swap[:, :, np.newaxis]
    result[is_swap_buy] = 9   # ← было =6 (fut_short_half) — неверно!
    return result


def simulate_batch(actions_arr: np.ndarray, prices: np.ndarray,
                   initial_capital: float,
                   snap_every: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Обёртка: если Numba доступна — JIT, иначе fallback на Python.
    Numba: 50-100x быстрее Python-цикла.

    Параметры симуляции берутся из TRAIN_* констант, синхронизированных
    с crypto_exchange при загрузке модуля (leverage, fraction, fee, stop, funding, etc.)

    Перед симуляцией применяются:
      1. _preprocess_swaps()  — конвертирует одновременные sell+buy в своп (action=6)
      2. Передаёт swap_fee_total в ядро симулятора для снижения комиссии свопа
    """
    G, T, NC = actions_arr.shape
    n_snaps = T // snap_every if snap_every > 0 else 0

    # Применяем оптимизацию свопов: buy → swap_buy при наличии sell на том же шаге
    processed = _preprocess_swaps(actions_arr)
    swap_fee  = float((TRAIN_FEE + TRAIN_SLIPPAGE) * SWAP_FEE_MULTIPLIER)

    # FIX BUG6: APY за один бар = APY_RATE / (365 * 24 * BAR).
    # Биржа: _apy_per_bar = APY_RATE / (365 * 24 * BAR).
    # Было: / (365 * 24) — завышено в TRAIN_BAR раз при 1m-таймфрейме.
    _apy_per_step = float(TRAIN_APY_RATE / (365 * 24 * max(1, TRAIN_BAR)))
    _stop_from_peak = int(1 if TRAIN_STOP_MODE == 'from_peak' else 0)
    _bar_int = int(TRAIN_BAR)   # FIX BUG5: для funding interval = 8*BAR

    if _NUMBA_OK:
        fr, md, tr, dpv = _sim_core(
            processed.astype(np.int8),
            prices.astype(np.float64),
            float(initial_capital),
            int(snap_every),
            float(TRAIN_FEE + TRAIN_SLIPPAGE),
            float(TRAIN_FRACTION),
            float(TRAIN_LEVERAGE),
            float(TRAIN_STOP_PCT),
            int(TRAIN_MAX_POS),
            int(n_snaps),
            swap_fee,
            float(TRAIN_FUNDING_RATE),
            _stop_from_peak,
            _apy_per_step,
            float(TRAIN_LIQ_FEE),
            _bar_int,
        )
        daily_pv_out = dpv if n_snaps > 0 else None
        return fr, md, tr, daily_pv_out

    # ── Fallback: оригинальный Python-цикл (медленный) ───────────────────────
    return _simulate_batch_python(processed, prices, initial_capital, snap_every)


def _simulate_batch_python(actions_arr: np.ndarray, prices: np.ndarray,
                            initial_capital: float,
                            snap_every: int = 0
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Оригинальная Python-реализация (fallback если Numba недоступна).
    Синхронизирована с Numba-ядром и crypto_exchange: лик-штраф, funding, from_peak стоп, APY."""
    G, T, NC = actions_arr.shape
    IC  = float(initial_capital)
    FEE = TRAIN_FEE + TRAIN_SLIPPAGE
    FR  = TRAIN_FRACTION
    LEV = TRAIN_LEVERAGE
    STP = TRAIN_STOP_PCT          # положительный: 0.20 = 20%
    FR_MODE = TRAIN_STOP_MODE     # 'from_peak' | 'from_start'
    FUND = TRAIN_FUNDING_RATE
    # FIX BUG6: APY за один бар = APY / (365 * 24 * BAR), как в Exchange.run()
    APY_STEP = TRAIN_APY_RATE / (365 * 24 * max(1, TRAIN_BAR))
    LIQ = TRAIN_LIQ_FEE
    SWAP_FEE = FEE * SWAP_FEE_MULTIPLIER
    # FIX BUG5: funding каждые 8*BAR баров = 8 часов, как в Exchange.step()
    FUND_INTERVAL = 8 * max(1, TRAIN_BAR)

    cash     = np.full(G, IC, dtype=np.float64)
    s_qty    = np.zeros((G, NC), np.float64)
    f_qty    = np.zeros((G, NC), np.float64)
    f_entry  = np.zeros((G, NC), np.float64)
    has_spot = np.zeros((G, NC), bool)
    has_fut  = np.zeros((G, NC), bool)
    stopped  = np.zeros(G, bool)
    peak_pv  = np.full(G, IC, dtype=np.float64)
    max_dd   = np.zeros(G, dtype=np.float64)
    trade_count = np.zeros(G, dtype=np.int32)

    # Daily snapshot setup
    n_snaps = T // snap_every if snap_every > 0 else 0
    daily_pv = np.zeros((G, n_snaps), dtype=np.float32) if n_snaps > 0 else None

    for t in range(T):
        p  = prices[t].astype(np.float64)
        at = actions_arr[:, t, :]

        # ── Ликвидация фьючерсов (маржа + PnL ≤ 0) + штраф ─────────────────
        if has_fut.any():
            mg_arr = np.abs(f_qty * f_entry) / LEV
            pnl_arr = f_qty * (p - f_entry)
            liq_mask = has_fut & (mg_arr + pnl_arr <= 0.0)
            if liq_mask.any():
                penalty = mg_arr * LIQ * liq_mask
                cash -= penalty.sum(axis=1)
                f_qty[liq_mask]   = 0.0
                f_entry[liq_mask] = 0.0
                has_fut[liq_mask] = False

        # ── Funding rate каждые 8*BAR шагов = 8 часов (FIX BUG5) ──────────────
        if FUND > 0.0 and t > 0 and t % FUND_INTERVAL == 0 and has_fut.any():
            notional = np.abs(f_qty * p)
            fund_charge = notional * FUND
            long_mask  = has_fut & (f_qty > 0)
            short_mask = has_fut & (f_qty < 0)
            cash -= (fund_charge * long_mask).sum(axis=1)
            cash += (fund_charge * short_mask).sum(axis=1)

        # ── Portfolio value + drawdown ────────────────────────────────────────
        fut_pnl = np.where(has_fut, f_qty * (p - f_entry), 0.0)
        pv = cash + (s_qty * p).sum(axis=1) + fut_pnl.sum(axis=1)
        peak_pv = np.maximum(peak_pv, pv)
        dd = np.where(peak_pv > 0, (peak_pv - pv) / peak_pv * 100.0, 0.0)
        max_dd = np.maximum(max_dd, dd)

        # ── Daily snapshot ───────────────────────────────────────────────────
        if n_snaps > 0:
            si = t // snap_every
            if si < n_snaps and (t + 1) % snap_every == 0:
                daily_pv[:, si] = np.maximum(pv, 0.0).astype(np.float32)

        # ── Портфельный стоп (from_peak или from_start) ──────────────────────
        if FR_MODE == 'from_peak':
            newly_stop = ~stopped & (peak_pv > 0) & (pv < peak_pv * (1.0 - STP))
        else:
            newly_stop = ~stopped & (pv < IC * (1.0 - STP))
        if newly_stop.any():
            for g in np.where(newly_stop)[0]:
                cash[g] += (s_qty[g] * p * (1 - FEE)).sum()
                mg_g  = np.abs(f_qty[g] * f_entry[g]) / LEV
                pnl_g = f_qty[g] * (p - f_entry[g])
                cash[g] += ((mg_g + pnl_g) * (1 - FEE) * has_fut[g]).sum()
                s_qty[g] = 0.0; has_spot[g] = False
                f_qty[g] = 0.0; has_fut[g]  = False
            stopped[newly_stop] = True

        # ── APY на кэш пока агент остановлен ─────────────────────────────────
        if stopped.any() and APY_STEP > 0.0:
            cash[stopped] *= (1.0 + APY_STEP)

        # n_pos = кол-во монет с хотя бы одной позицией (спот ИЛИ фут)
        n_pos = (has_spot | has_fut).sum(1)
        room  = (~stopped) & (n_pos < TRAIN_MAX_POS)

        def _ts(mask):
            raw = np.maximum(pv[mask] * FR, 0.0)
            return np.minimum(raw, np.maximum(cash[mask], 0.0))

        # ── 9-action scheme ────────────────────────────────────────────────────
        # 1=buy_half(50%) 2=buy_full(100%) 3=sell_spot
        # 4=fl_half 5=fl_full 6=fs_half 7=fs_full 8=close_fut 9=swap_buy

        # actions 1+2: buy_spot (half / full, усреднение разрешено)
        for _a_buy, _frac in ((1, 0.5), (2, 1.0)):
            bm = (at == _a_buy) & ~stopped[:, None]
            if not bm.any(): continue
            for c in range(NC):
                m = bm[:, c]
                if not m.any(): continue
                m_ok = m & (has_spot[:,c] | room)
                if not m_ok.any(): continue
                ts_c = _ts(m_ok) * _frac
                valid = m_ok.copy(); valid[m_ok] = ts_c > 0.0
                if not valid.any(): continue
                ts_v = ts_c[ts_c > 0.0]
                qty = ts_v / (p[c] * (1 + FEE))
                cash[valid] -= ts_v; s_qty[valid,c] += qty
                newly_opened = valid & ~has_spot[:,c]
                has_spot[valid,c] = True
                n_pos[newly_opened] += 1
                room[newly_opened] = (~stopped[newly_opened]) & (n_pos[newly_opened] < TRAIN_MAX_POS)
                trade_count[valid] += 1

        # action 3: sell_spot
        sm = (at == 3) & has_spot & ~stopped[:, None]
        if sm.any():
            for c in range(NC):
                m = sm[:, c]
                if not m.any(): continue
                cash[m] += s_qty[m,c] * p[c] * (1-FEE)
                s_qty[m,c] = 0.0; has_spot[m,c] = False
                closed = m & ~has_fut[:,c]
                n_pos[closed] -= 1
                room[closed] = (~stopped[closed]) & (n_pos[closed] < TRAIN_MAX_POS)
                trade_count[m] += 1

        # action 9: swap_buy (internal, только новые позиции)
        sbm = (at == 9) & ~has_spot & room[:, None]
        if sbm.any():
            for c in range(NC):
                m = sbm[:, c]
                if not m.any(): continue
                ts_c = _ts(m)
                valid = m.copy(); valid[m] = ts_c > 0.0
                if not valid.any(): continue
                ts_v = ts_c[ts_c > 0.0]
                qty = ts_v / (p[c] * (1 + SWAP_FEE))
                cash[valid] -= ts_v; s_qty[valid,c] += qty
                has_spot[valid,c] = True; n_pos[valid] += 1
                room[valid] = (~stopped[valid]) & (n_pos[valid] < TRAIN_MAX_POS)
                trade_count[valid] += 1

        # actions 4+5: fut_long (half/full, усреднение разрешено, конфликт с шортом запрещён)
        for _a_fl, _frac in ((4, 0.5), (5, 1.0)):
            flm = (at == _a_fl) & (f_qty >= 0) & ~stopped[:, None]
            if not flm.any(): continue
            for c in range(NC):
                m = flm[:, c]
                if not m.any(): continue
                m_ok = m & (has_fut[:,c] | room)
                if not m_ok.any(): continue
                ts_c = _ts(m_ok) * _frac
                valid = m_ok.copy(); valid[m_ok] = ts_c > 0.0
                if not valid.any(): continue
                ts_v = ts_c[ts_c > 0.0]
                new_qty = ts_v * LEV / (p[c] * (1+FEE))
                old_abs = np.abs(f_qty[valid,c])
                avg_entry = np.where(
                    old_abs > 0,
                    (f_entry[valid,c]*old_abs + p[c]*new_qty) / (old_abs + new_qty),
                    p[c])
                f_entry[valid,c] = avg_entry
                f_qty[valid,c]  += new_qty
                cash[valid]     -= ts_v
                newly_opened = valid & ~has_fut[:,c]
                has_fut[valid,c] = True
                n_pos[newly_opened] += 1
                room[newly_opened] = (~stopped[newly_opened]) & (n_pos[newly_opened] < TRAIN_MAX_POS)
                trade_count[valid] += 1

        # actions 6+7: fut_short (half/full, усреднение, конфликт с лонгом запрещён)
        for _a_fs, _frac in ((6, 0.5), (7, 1.0)):
            fsm = (at == _a_fs) & (f_qty <= 0) & ~stopped[:, None]
            if not fsm.any(): continue
            for c in range(NC):
                m = fsm[:, c]
                if not m.any(): continue
                m_ok = m & (has_fut[:,c] | room)
                if not m_ok.any(): continue
                ts_c = _ts(m_ok) * _frac
                valid = m_ok.copy(); valid[m_ok] = ts_c > 0.0
                if not valid.any(): continue
                ts_v = ts_c[ts_c > 0.0]
                new_qty = ts_v * LEV / (p[c] * (1+FEE))
                old_abs = np.abs(f_qty[valid,c])
                avg_entry = np.where(
                    old_abs > 0,
                    (f_entry[valid,c]*old_abs + p[c]*new_qty) / (old_abs + new_qty),
                    p[c])
                f_entry[valid,c] = avg_entry
                f_qty[valid,c]  -= new_qty
                cash[valid]     -= ts_v
                newly_opened = valid & ~has_fut[:,c]
                has_fut[valid,c] = True
                n_pos[newly_opened] += 1
                room[newly_opened] = (~stopped[newly_opened]) & (n_pos[newly_opened] < TRAIN_MAX_POS)
                trade_count[valid] += 1

        # action 8: close_futures
        cfm = (at == 8) & has_fut & ~stopped[:, None]
        if cfm.any():
            for c in range(NC):
                m = cfm[:, c]
                if not m.any(): continue
                pnl = f_qty[m,c] * (p[c] - f_entry[m,c])
                mg  = np.abs(f_qty[m,c] * f_entry[m,c]) / LEV
                cash[m] += (mg + pnl) * (1 - FEE)
                f_qty[m,c] = 0.0; f_entry[m,c] = 0.0; has_fut[m,c] = False
                closed = m & ~has_spot[:,c]
                n_pos[closed] -= 1
                room[closed] = (~stopped[closed]) & (n_pos[closed] < TRAIN_MAX_POS)
                trade_count[m] += 1

    fp = prices[-1].astype(np.float64)
    cash += (s_qty * fp * (1-FEE)).sum(axis=1)
    mg_f  = np.abs(f_qty * f_entry) / LEV
    pnl_f = f_qty * (fp - f_entry)
    cash += ((mg_f + pnl_f) * (1-FEE) * has_fut).sum(axis=1)
    final_pv = np.maximum(cash, 0.0)
    trade_rate = trade_count.astype(np.float64) / max(T * NC, 1)
    return (final_pv / IC - 1.0) * 100.0, max_dd, trade_rate, daily_pv


# ══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION v8 — 3-ФАЗНЫЙ АДАПТИВНЫЙ
# ══════════════════════════════════════════════════════════════════════════════

def _compute_fitness(period_rets: np.ndarray,
                     period_dds: np.ndarray,
                     period_trade_rates: np.ndarray,
                     period_daily_pvs: Optional[List[np.ndarray]] = None,
                     period_regime_weights: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """
    3-фазный адаптивный фитнес v8.

    ФАЗА 1 (gens 1..PHASE1_GENS_FRAC = 30%) — «Найди любые деньги»
      Цель: создать хоть какой-то градиент из нулевой популяции.
      Формула: 3.5×mean_ret + 2.0×calmar + subperiod_score + win-бонусы.
      НЕТ inactivity_pen, bear_pen, dd_duration — они блокируют ранний прогресс.

    ФАЗА 2 (gens PHASE1..fitness_phase2_frac = 60%) — «Делай это стабильно»
      Добавляем risk-adj метрики + умеренные штрафы (в 3× меньше Phase 3).

    ФАЗА 3 (gens FITNESS_PHASE2..N_GENERATIONS) — полная сложность v7.
    """
    G, P = period_rets.shape
    # Однофазный фитнес: phase-переменная больше не используется

    # ──  Режимные веса ───────────────────────────────────────────────────────
    if period_regime_weights is not None and len(period_regime_weights) == P:
        w_arr = np.array(period_regime_weights, dtype=np.float64)
    else:
        w_arr = np.ones(P, dtype=np.float64)
    w_norm = w_arr / (w_arr.sum() + 1e-9)

    mean_r_weighted = (period_rets * w_norm[None, :]).sum(axis=1)
    mean_r          = period_rets.mean(axis=1)
    mean_r_tiled    = mean_r_weighted[:, None]
    deviations      = period_rets - mean_r_tiled
    weighted_var    = ((deviations ** 2) * w_norm[None, :]).sum(axis=1)
    full_std        = np.sqrt(weighted_var) + 1e-6

    losses   = np.where(period_rets < 0, period_rets, 0.0)
    down_std = np.sqrt(((losses ** 2) * w_norm[None, :]).sum(axis=1))
    mean_dd  = period_dds.mean(axis=1)
    zero_p   = (np.abs(period_rets) < ZERO_THRESH).mean(axis=1)

    worst_ret = period_rets.min(axis=1)
    worst_pen = np.where(worst_ret < MIN_RET_THRESH,
                         (MIN_RET_THRESH - worst_ret) * 0.5, 0.0)

    calmar_scaled = mean_r_weighted / (mean_dd + 5.0) * 5.0
    sharpe_scaled = mean_r_weighted / (full_std + 3.0) * 3.0

    mean_trade_rate = period_trade_rates.mean(axis=1)
    inactivity_pen  = np.where(
        mean_trade_rate < INACTIVITY_THRESH,
        (INACTIVITY_THRESH - mean_trade_rate) / INACTIVITY_THRESH * 5.0,
        0.0
    )

    # ── Daily snapshot metrics ────────────────────────────────────────────────
    daily_sharpe_scores = np.zeros(G, dtype=np.float64)
    daily_wr_bonus      = np.zeros(G, dtype=np.float64)
    consistency_bonus   = np.zeros(G, dtype=np.float64)
    dd_duration_pen     = np.zeros(G, dtype=np.float64)
    max_daily_loss_pen  = np.zeros(G, dtype=np.float64)
    subperiod_score     = np.zeros(G, dtype=np.float64)

    if period_daily_pvs is not None:
        valid_data = [(pv, w_norm[i]) for i, pv in enumerate(period_daily_pvs)
                      if pv is not None and pv.shape[1] >= 3]
        if valid_data:
            per_period_sharpe  = []
            per_period_wr      = []
            per_period_consist = []
            per_period_ddd     = []
            per_period_mdlp    = []
            per_period_sub     = []
            per_period_w       = []

            for pv, pw in valid_data:
                pv_safe = np.maximum(pv, 1.0)
                dr = pv_safe[:, 1:] / pv_safe[:, :-1] - 1.0
                dr_mean = dr.mean(axis=1)
                dr_std  = dr.std(axis=1) + 1e-6
                dsh = dr_mean / dr_std * np.sqrt(min(365, dr.shape[1]))
                per_period_sharpe.append(np.clip(dsh, -5, 10))
                dwr = (dr > 0).mean(axis=1)
                per_period_wr.append(dwr)
                start_pv = pv_safe[:, 0:1]
                above_start = (pv_safe > start_pv).mean(axis=1)
                per_period_consist.append(above_start)
                peak_pv_snap = np.maximum.accumulate(pv_safe, axis=1)
                dd_snap = (peak_pv_snap - pv_safe) / (peak_pv_snap + 1e-9)
                in_dd = (dd_snap > DD_DURATION_THRESH).mean(axis=1)
                per_period_ddd.append(in_dd)
                min_daily_ret = dr.min(axis=1)
                mdlp = np.where(min_daily_ret < MAX_DAILY_LOSS_THRESH,
                                (MAX_DAILY_LOSS_THRESH - min_daily_ret) * 15.0, 0.0)
                per_period_mdlp.append(mdlp)

                # Подпериодный score: делим на 4 квартала, бонус за каждый + квартал
                T_snap = pv_safe.shape[1]
                q_size = max(1, T_snap // 4)
                sub_quarters = []
                for qi in range(4):
                    q_start = qi * q_size
                    q_end   = min(q_start + q_size, T_snap)
                    if q_end > q_start + 1:
                        q_ret = pv_safe[:, q_end-1] / pv_safe[:, q_start] - 1.0
                        sub_quarters.append((q_ret > 0).astype(np.float64))
                if sub_quarters:
                    sub_score = np.stack(sub_quarters, axis=1).mean(axis=1)  # 0..1
                    per_period_sub.append(sub_score * pw)
                per_period_w.append(pw)

            ww = np.array(per_period_w, dtype=np.float64)
            ww /= ww.sum() + 1e-9

            def _wavg(lst):
                m = np.column_stack(lst)
                return (m * ww[None, :]).sum(axis=1)

            mean_dsh  = _wavg(per_period_sharpe)
            mean_dwr  = _wavg(per_period_wr)
            mean_con  = _wavg(per_period_consist)
            mean_ddd  = _wavg(per_period_ddd)
            mean_mdlp = _wavg(per_period_mdlp)

            daily_sharpe_scores = np.clip(mean_dsh, -5, 10) * 2.5
            daily_wr_bonus      = (mean_dwr - 0.50) * 15.0
            consistency_bonus   = (mean_con - 0.50) * 8.0
            dd_duration_pen     = mean_ddd * 8.0
            max_daily_loss_pen  = mean_mdlp

            if per_period_sub:
                # per_period_sub items уже умножены на pw (sub_score * pw),
                # просто суммируем и нормируем на суммарный вес
                _sub_w_total = float(sum(per_period_w[:len(per_period_sub)]))
                subperiod_score = sum(per_period_sub) / (_sub_w_total + 1e-9)

    # ── Единые пороги win-rate бонусов ───────────────────────────────────────
    win_1pct_rate     = (period_rets >= WIN1_PROFIT_THRESH).mean(axis=1)
    win_5pct_rate     = (period_rets >= WIN_PROFIT_THRESH).mean(axis=1)
    bonus_20pct_rate  = (period_rets >= BONUS_PROFIT_THRESH).mean(axis=1)
    win_1pct_bonus    = win_1pct_rate   * 5.0 * WIN1_FITNESS_WEIGHT
    win_5pct_bonus    = win_5pct_rate   * 5.0 * FITNESS_W_WIN_5PCT
    bonus_20pct_bonus = bonus_20pct_rate * 5.0 * FITNESS_W_BONUS_20

    # ── Штраф за слабые режимы (медведи/краши) ───────────────────────────────
    q25 = np.percentile(period_rets, 25, axis=1)
    bear_regime_pen = np.where(
        q25 < BEAR_PENALTY_THRESH,
        (BEAR_PENALTY_THRESH - q25) * 0.4,
        0.0
    )
    q75 = np.percentile(period_rets, 75, axis=1)
    regime_spread = q75 - q25
    bear_regime_pen += np.where(regime_spread > 15.0,
                                (regime_spread - 15.0) * 0.06, 0.0)
    stop_pct = (period_rets <= -9.0).mean(axis=1)
    stop_pen = np.where(stop_pct > 0.20, (stop_pct - 0.20) * 20.0, 0.0)

    # ── FIX 3: Phase1 специализация — бонус за pos% (прямая метрика бэктеста) ──
    # pos_rate_bonus напрямую вознаграждает % прибыльных периодов, устраняя
    # разрыв между fitness (dominated by outliers) и реальным win rate бэктеста.
    pos_rate      = (period_rets > 0).mean(axis=1)                # 0..1
    pos_rate_bonus = (pos_rate - 0.35) * 5.0 * POS_RATE_BONUS_W  # target 35% → 0 bonus

    # ── FIX 4: штраф за серию убыточных месяцев (anti Calmar-overfitting) ──────
    # Агент с fitness=30 но 54/96 убыточными периодами имеет длинные neg-стрики.
    # Штраф прогрессивно растёт с каждым добавочным убыточным месяцем сверх NEG_STREAK_THRESH.
    # Векторизовано: для каждого агента считаем макс серию убытков.
    neg_streak_pen = np.zeros(G, dtype=np.float64)
    for gi in range(G):
        pr   = period_rets[gi]
        cur  = 0; mx = 0
        for r in pr:
            cur = cur + 1 if r < 0 else 0
            mx  = max(mx, cur)
        excess = max(0, mx - NEG_STREAK_THRESH)
        neg_streak_pen[gi] = excess * NEG_STREAK_PEN_W

    # ══ ЕДИНАЯ СБАЛАНСИРОВАННАЯ ФОРМУЛА v2 (FIX 3+4) ═══════════════════════════
    #
    # Изменения vs v1:
    #   calmar 1.5 → 1.5*CALMAR_WEIGHT_ADJ=1.05  (снижаем outlier-dominance)
    #   + pos_rate_bonus   (прямое вознаграждение за % прибыльных периодов)
    #   - neg_streak_pen   (штраф за 3+ убытков подряд — как в реальной торговле)
    #   subperiod_score 0.8 → 1.2  (больший вес внутрипериодной консистентности)
    #
    # Цель: выровнять ранжирование с бэктестом:
    #   Было: fitness→outlier-months доминируют → avg_ret ≠ backtest
    #   Стало: fitness→balanced: avg + pos_rate + consistency - neg_streaks
    #
    # Прогрессивный бонус за доходность
    progressive_return_bonus = np.tanh(np.maximum(0.0, mean_r_weighted - 0.5) / 3.0) * 0.40

    return (2.5    * mean_r_weighted                      # сильный сигнал возврата
          + 1.5    * calmar_scaled * CALMAR_WEIGHT_ADJ    # risk-adjusted (drawdown), снижен
          + 0.5    * sharpe_scaled                        # risk-adjusted (volatility)
          + 0.35   * daily_sharpe_scores                  # внутридневная стабильность
          + 1.2    * subperiod_score                      # внутрипериодная консистентность (усилена)
          + pos_rate_bonus                                # FIX 3: бонус за % прибыльных периодов
          + win_1pct_bonus                                # ≥1% победы
          + win_5pct_bonus                                # ≥5% победы
          + bonus_20pct_bonus                             # ≥20% бонус
          + TRADE_REWARD_W * np.tanh(mean_trade_rate / (TRADE_REWARD_REF + 1e-9))
          + progressive_return_bonus                      # бонус за высокую доходность
          - FITNESS_ALPHA * 0.5  * down_std               # штраф за downside volatility
          - FITNESS_BETA  * 0.5  * mean_dd                # штраф за просадку
          - FITNESS_GAMMA * 0.4  * zero_p                 # штраф за пассивность
          - FITNESS_DELTA * 0.35 * worst_pen              # штраф за катастрофы
          - FITNESS_THETA * 0.4  * bear_regime_pen        # штраф за слабые режимы
          - FITNESS_EPSILON * 0.25 * inactivity_pen       # штраф за бездействие
          - 0.20            * stop_pen                    # штраф за частые стопы
          - neg_streak_pen)                               # FIX 4: штраф за серию убытков


# ══════════════════════════════════════════════════════════════════════════════
# PARETO-RANKING ПО РЕЖИМАМ (NSGA-II style non-dominated sorting)
# ══════════════════════════════════════════════════════════════════════════════

def _pareto_regime_bonus(period_rets: np.ndarray,
                         period_regimes: List[str],
                         bonus_weight: float = PARETO_BONUS_WEIGHT
                         ) -> np.ndarray:
    """
    Вычисляет Pareto-бонус к фитнесу на основе multi-objective ranking по режимам.

    Цели (3):  mean_return_bearish, mean_return_neutral, mean_return_bullish
    Метод:     Fast non-dominated sorting (NSGA-II) → Pareto-rank → бонус

    Pareto-ranking даёт естественное давление на ВСЕ режимы одновременно,
    без конфликта весов (в отличие от dynamic_rw, который двигает ландшафт).

    Бонус: (1.0 - rank / n_fronts) × bonus_weight
    - Pareto-фронт 0 (не доминируется): полный бонус
    - Последний фронт: бонус ≈ 0

    Args:
        period_rets:    (G, P) матрица возвратов [%] по периодам
        period_regimes: список режимов для каждого периода (len=P)
        bonus_weight:   максимальный бонус для фронта 0

    Returns:
        (G,) массив бонусов к фитнесу
    """
    G, P = period_rets.shape
    if G == 0 or P == 0:
        return np.zeros(G, dtype=np.float64)

    # ── Вычисляем 3 objective для каждого генома ─────────────────────────────
    regime_labels = ['bearish', 'neutral', 'bullish']
    regime_masks = {r: np.array([1.0 if period_regimes[i] == r else 0.0
                                  for i in range(P)], dtype=np.float64)
                    for r in regime_labels}

    objectives = np.zeros((G, len(regime_labels)), dtype=np.float64)
    for ri, r in enumerate(regime_labels):
        mask = regime_masks[r]
        count = mask.sum()
        if count > 0:
            objectives[:, ri] = (period_rets * mask[None, :]).sum(axis=1) / count
        else:
            objectives[:, ri] = 0.0

    # ── Fast non-dominated sorting ───────────────────────────────────────────
    # Упрощённая версия NSGA-II: O(G² × K) где K=3 objectives
    pareto_ranks = np.zeros(G, dtype=np.int32)
    remaining = set(range(G))
    current_rank = 0

    while remaining:
        # Находим не-доминируемые точки в remaining
        front = []
        remaining_list = list(remaining)
        for i in remaining_list:
            dominated = False
            for j in remaining_list:
                if i == j:
                    continue
                # j доминирует i если j >= i по всем и j > i хотя бы по одному
                if (np.all(objectives[j] >= objectives[i]) and
                    np.any(objectives[j] > objectives[i])):
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for idx in front:
            pareto_ranks[idx] = current_rank
            remaining.discard(idx)
        current_rank += 1

    # ── Бонус: линейно от 0 до bonus_weight ─────────────────────────────────
    n_fronts = max(1, current_rank)
    bonus = (1.0 - pareto_ranks / n_fronts) * bonus_weight
    return bonus


# ══════════════════════════════════════════════════════════════════════════════
# GPU EVALUATOR — обновлён для daily snapshots
# ══════════════════════════════════════════════════════════════════════════════

class GPUEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, population, precomp, rw_override: Optional[Dict[str, float]] = None):
        G = len(population)
        rl, dl, trl = [], [], []
        daily_pvs: List[Optional[np.ndarray]] = []
        regime_weights = []

        snap_every = _get_snap_every()

        for feat, prices, syms, month, period, *extra in precomp:
            rw     = extra[0] if extra else 1.0
            regime = extra[1] if len(extra) > 1 else 'bull'
            # Применяем динамические режимные веса если переданы
            if rw_override is not None and regime in rw_override:
                rw = rw_override[regime]
            regime_weights.append(float(rw))
            actions = self._batch_forward(population, feat)
            # Выбор оптимальных валют: маскируем открытие позиций для не-топ монет
            if CURRENCY_SELECTION_ENABLED:
                actions = _apply_currency_selection(
                    actions, feat, regime, TOP_CURRENCIES_N)
            rets, dds, trs, dpv = simulate_batch(
                actions.astype(np.int32), prices, _cx.INITIAL_CAPITAL, snap_every)
            rl.append(rets); dl.append(dds); trl.append(trs)
            daily_pvs.append(dpv)

        ret_mat = np.column_stack(rl)
        dd_mat  = np.column_stack(dl)
        tr_mat  = np.column_stack(trl)
        rw_arr  = np.array(regime_weights, dtype=np.float64)
        fits    = _compute_fitness(ret_mat, dd_mat, tr_mat, daily_pvs, rw_arr)

        # ── Pareto-ranking бонус по режимам ────────────────────────────────────
        if PARETO_REGIME_RANKING:
            _p_regimes = [map_regime_3(entry[6]) if len(entry) > 6
                          and isinstance(entry[6], str) else 'neutral'
                          for entry in precomp]
            pareto_bonus = _pareto_regime_bonus(ret_mat, _p_regimes,
                                                PARETO_BONUS_WEIGHT)
            fits = fits + pareto_bonus

        return fits, [ret_mat[g].tolist() for g in range(G)]

    def _batch_forward(self, population, feat):
        import torch
        dev = self.device
        G   = len(population)
        T, NC, NF = feat.shape

        with torch.no_grad():  # FIX 1: без autograd — меньше памяти и быстрее
            pop_t = torch.from_numpy(population.astype(np.float32)).to(dev, non_blocking=True)
            i = 0
            W1 = pop_t[:, i:i+_W1s].view(G, N_INPUT,   N_HIDDEN1); i += _W1s
            b1 = pop_t[:, i:i+_b1s];                                 i += _b1s
            W2 = pop_t[:, i:i+_W2s].view(G, N_HIDDEN1, N_HIDDEN2);  i += _W2s
            b2 = pop_t[:, i:i+_b2s];                                 i += _b2s
            W3 = pop_t[:, i:i+_W3s].view(G, N_HIDDEN2, N_HIDDEN3);  i += _W3s
            b3 = pop_t[:, i:i+_b3s];                                 i += _b3s
            W4 = pop_t[:, i:i+_W4s].view(G, N_HIDDEN3, N_ACTIONS);  i += _W4s
            b4 = pop_t[:, i:i+_b4s]

            # FIX 3: reshape feat в 2D (T*NC, NF) — один трансфер на GPU
            feat_2d = torch.from_numpy(
                feat.reshape(T * NC, NF).astype(np.float32)
            ).to(dev, non_blocking=True)

            # ── Авто-расчёт безопасного chunk_bars под текущий G ──────────────────
            # Пик VRAM при одном чанке:  h1+h2 одновременно в памяти (самый тяжёлый момент)
            #   = G * (chunk*NC) * (H1 + H2) * 4 байт
            # При G=1500, NC=20, H1=128, H2=64, chunk=1024:
            #   1500 * 20480 * 192 * 4 = 23.6 GB → превышает 24 GB RTX4090 → зависание!
            # Решение: авто-уменьшаем chunk_bars пропорционально G.
            try:
                _vram_total = torch.cuda.get_device_properties(dev).total_memory
                _vram_safe  = int(_vram_total * 0.48)   # 48% VRAM — буфер для весов и остального
                _bytes_per_bar = G * NC * (N_HIDDEN1 + N_HIDDEN2) * 4   # h1+h2 пик
                _auto_chunk = max(32, int(_vram_safe / max(1, _bytes_per_bar)))
                _chunk = min(GPU_CHUNK_BARS, _auto_chunk)
                # Печатаем только один раз за жизнь объекта чтобы не спамить (96 вызовов/ген)
                if _chunk < GPU_CHUNK_BARS and not getattr(self, '_vram_chunk_warned', False):
                    _peak_gb = G * NC * _chunk * (N_HIDDEN1 + N_HIDDEN2) * 4 / 1024**3
                    print(f"  [GPU-VRAM] авто chunk: {GPU_CHUNK_BARS}→{_chunk}  "
                          f"G={G}  пик≈{_peak_gb:.1f}GB/{_vram_total/1024**3:.0f}GB")
                    self._vram_chunk_warned = True
            except Exception:
                _chunk = GPU_CHUNK_BARS

            all_acts = []
            for t0 in range(0, T, _chunk):
                t1  = min(t0 + _chunk, T)
                tc  = t1 - t0
                x   = feat_2d[t0*NC : t1*NC]   # (tc*NC, NF) — НЕ копируется G раз

                # einsum 'nf,gfh->gnh' — x broadcast без expand()
                # Пик VRAM: G*(tc*NC)*(H1+H2)*4 — авто-chunk гарантирует безопасность
                h1  = torch.nn.functional.elu(
                        torch.einsum('nf,gfh->gnh', x, W1) + b1[:, None, :])
                h2  = torch.nn.functional.elu(
                        torch.einsum('gnh,ghk->gnk', h1, W2) + b2[:, None, :])
                del h1   # FIX: освобождаем h1 до h3 — уменьшаем пик с H1+H2+H3 до H2+H3
                h3  = torch.nn.functional.elu(
                        torch.einsum('gnk,gkh->gnh', h2, W3) + b3[:, None, :])
                del h2
                out = torch.einsum('gnh,gha->gna', h3, W4) + b4[:, None, :]
                del h3

                # out: (G, tc*NC, N_ACTIONS) → reshape (G, tc, NC)
                acts = out.argmax(dim=-1).view(G, tc, NC).to(torch.int8).cpu().numpy()
                all_acts.append(acts)
                del x, out

        torch.cuda.empty_cache()  # один раз после всех чанков
        return np.concatenate(all_acts, axis=1).astype(np.int8)


def _get_snap_every() -> int:
    """Определяем интервал снапшота исходя из BAR."""
    try:
        bar = _cx.BAR
    except Exception:
        bar = 1
    return max(1, SNAPSHOT_BARS // max(1, bar)) * bar


def _detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            dev  = torch.device('cuda')
            name = torch.cuda.get_device_name(0)
            gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {name}  ({gb:.1f} GB VRAM)  - будет использован")
            return 'torch', dev
        print("  PyTorch найден, но CUDA недоступна -> CPU режим")
    except ImportError:
        print("  PyTorch не установлен -> CPU режим")
    except Exception as e:
        print(f"  GPU недоступен ({type(e).__name__}: {e}) -> CPU режим")
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# CPU PERSISTENT POOL — обновлён для daily snapshots
# ══════════════════════════════════════════════════════════════════════════════
_W_PRECOMP = None
_W_IC      = None

def _worker_init(pkl_path: str, ic: float):
    global _W_PRECOMP, _W_IC
    # Устанавливаем флаг ПЕРВЫМ — module-level код уже выполнился при spawn,
    # но _is_worker() проверяет env в реальном времени, так что это корректно
    # для любых последующих print-вызовов внутри воркера.
    import os as _os
    _os.environ['_GENETICS_WORKER'] = '1'
    _old_o, _old_e = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        import numba  # noqa – подавляем "[OK] numba найден" в воркерах
    except ImportError:
        pass
    try:
        with open(pkl_path, 'rb') as f:
            _W_PRECOMP = pickle.load(f)
        _W_IC = ic
    except Exception:
        pass
    finally:
        sys.stdout = _old_o; sys.stderr = _old_e

def _worker_task(gbytes: bytes):
    # Синхронизируем фазу фитнеса из temp-файла (устанавливается родительским процессом)
    _set_train_phase(_get_train_phase())
    genome = np.frombuffer(gbytes, dtype=np.float32).copy()
    W1, b1, W2, b2, W3, b3, W4, b4 = _unpack(genome)
    IC = _W_IC or 100_000.0
    snap_every = _get_snap_every()
    rl, dl, trl = [], [], []
    dpv_list = []
    rw_list  = []
    for entry in _W_PRECOMP:
        feat, prices, syms, month, period = entry[:5]
        rw     = entry[5] if len(entry) > 5 else 1.0
        regime = map_regime_3(entry[6]) if len(entry) > 6 else 'neutral'
        rw_list.append(float(rw))
        T, NC, _ = feat.shape
        actions = np.zeros((T, NC), dtype=np.int8)
        for t in range(T):
            actions[t] = _fwd_np(feat[t], W1, b1, W2, b2, W3, b3, W4, b4).argmax(axis=-1).astype(np.int8)
        # Выбор оптимальных валют перед симуляцией
        if CURRENCY_SELECTION_ENABLED:
            actions_3d = _apply_currency_selection(
                actions[np.newaxis].astype(np.int32), feat, regime, TOP_CURRENCIES_N)
            actions = actions_3d[0].astype(np.int8)
        rets, dds, trs, dpv = simulate_batch(
            actions[None].astype(np.int32), prices, IC, snap_every)
        rl.append(float(rets[0])); dl.append(float(dds[0])); trl.append(float(trs[0]))
        dpv_list.append(dpv)
    if not rl:
        return -100.0, []
    r_arr  = np.array(rl,  dtype=np.float64)[None]
    d_arr  = np.array(dl,  dtype=np.float64)[None]
    tr_arr = np.array(trl, dtype=np.float64)[None]
    rw_arr = np.array(rw_list, dtype=np.float64)
    return float(_compute_fitness(r_arr, d_arr, tr_arr, dpv_list, rw_arr)[0]), rl


class CPUEvaluator:
    def __init__(self, precomp, n_workers):
        import concurrent.futures as cfu
        self.n   = n_workers
        self._ex = None
        tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        tmp.close()
        self._pkl = tmp.name
        with open(self._pkl, 'wb') as f:
            pickle.dump(precomp, f, protocol=4)
        print(f"  Данные -> tmp файл ({os.path.getsize(self._pkl)/1024**2:.1f} MB)")
        if n_workers > 1:
            # Устанавливаем флаг ДО spawn — дочерние процессы наследуют его
            # и подавляют module-level prints (settings/exchange spam).
            os.environ['_GENETICS_WORKER'] = '1'
            ctx = _mp.get_context('spawn')
            self._ex = cfu.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_worker_init,
                initargs=(self._pkl, _cx.INITIAL_CAPITAL))
            print(f"  Запуск {n_workers} постоянных воркеров...", end=' ', flush=True)
            dummy = np.zeros(GENOME_SIZE, dtype=np.float32).tobytes()
            try:
                list(self._ex.map(_worker_task, [dummy]*n_workers, timeout=180))
            except Exception:
                pass
            print("done")

    def evaluate(self, population):
        import concurrent.futures as cfu
        G = len(population)
        fits = np.full(G, -np.inf)
        rets = [[] for _ in range(G)]
        tasks = [population[i].astype(np.float32).tobytes() for i in range(G)]
        if self._ex and self.n > 1:
            futs = {self._ex.submit(_worker_task, t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in cfu.as_completed(futs):
                idx = futs[fut]; done += 1
                try:
                    fits[idx], rets[idx] = fut.result(timeout=600)
                except Exception:
                    fits[idx] = -50.0
                _pbar(done, G, fits)
        else:
            for i, task in enumerate(tasks):
                fits[i], rets[i] = _worker_task(task)
                _pbar(i+1, G, fits)
        print()
        return fits, rets

    def shutdown(self):
        if self._ex:
            self._ex.shutdown(wait=False)
        try:
            os.unlink(self._pkl)
        except Exception:
            pass


def _pbar(done, total, fits):
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    step_pct = max(1, total // 10)
    if not is_tty and done % step_pct != 0 and done != total:
        return
    W = 24; f = int(W * done / total)
    bar = '#'*f + '.'*(W-f)
    v = fits[fits > -np.inf]
    s = f"best={v.max():+.1f}%" if len(v) else "-"
    end = '\r' if is_tty else '\n'
    print(f"    [{bar}] {done:2d}/{total}  {s}", end=end, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# УЛУЧШЕННЫЙ BEHAVIORAL CLONING
# ══════════════════════════════════════════════════════════════════════════════

def _bc_worker(args):
    """
     Улучшенный BC:
    - Использует до BC_MAX_PERIODS периодов
    - Добавлен action 5 в маппинг
    - 3 варианта: clean, noisy+, noisy- → больше разнообразия в популяции
    - Расширенный warm-up (100 баров)
    - [fix] Numba-сообщения подавлены — рабочий процесс тихий
    - [fix] Повторяющиеся ошибки одного типа показываются только один раз
    """
    # ── Подавляем "[OK] numba найден" из дочерних процессов ──────────────────
    import io as _io
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    _devnull = _io.StringIO()
    sys.stdout = sys.stderr = _devnull
    try:
        import numba  # noqa – тихая инициализация
    except ImportError:
        pass
    finally:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr

    agent_name, pkl_path, seed = args
    logs = []
    _seen_err_types: set = set()   # дедупликация ошибок одного класса

    import pickle as _pk
    with open(pkl_path, 'rb') as f:
        precomp = _pk.load(f)

    try:
        # Ищем в crypto_agents → crypto_players → crypto_exchange
        agent_cls = None
        try:
            import crypto_agents as _ca
            agent_cls = getattr(_ca, agent_name, None)
        except ImportError:
            pass
        if agent_cls is None:
            try:
                import crypto_players as _cp
                agent_cls = getattr(_cp, agent_name, None)
            except ImportError:
                pass
        if agent_cls is None:
            agent_cls = getattr(_cx, agent_name, None)
        if agent_cls is None:
            return agent_name, None, [f"    [BC] {agent_name} не найден (ни в crypto_agents, ни в crypto_players, ни в crypto_exchange)"]
    except Exception as e:
        return agent_name, None, [f"    [BC] {agent_name} ошибка поиска: {e}"]

    X_all, y_all = [], []
    rng_local = np.random.default_rng(seed)

    # Режимно-балансированный выбор периодов для BC
    # БЫЛО: precomp[:BC_MAX_PERIODS] — только первые 6 периодов (2021 Jan-Jun = ТОЛЬКО бычий)
    # СТАЛО: выбираем по BC_REGIMES_PER_TYPE периодов каждого режима
    regime_buckets: Dict[str, list] = {}
    for entry in precomp:
        feat_e, prices_e, syms_e, month_e, period_e = entry[:5]
        rw = entry[5] if len(entry) > 5 else 1.0
        regime_name = map_regime_3(entry[6]) if len(entry) > 6 and isinstance(entry[6], str) else \
                      'neutral'
        if regime_name not in regime_buckets:
            regime_buckets[regime_name] = []
        regime_buckets[regime_name].append(entry)

    selected_entries = []
    for regime_name, bucket in regime_buckets.items():
        n = min(BC_REGIMES_PER_TYPE, len(bucket))
        chosen = rng_local.choice(len(bucket), n, replace=False).tolist()
        for idx in chosen:
            selected_entries.append(bucket[idx])

    # Если режимов мало — дополняем рандомными
    if len(selected_entries) < 4:
        selected_entries = precomp[:min(6, len(precomp))]

    periods_to_use = selected_entries

    for entry in periods_to_use:
        feat, prices, syms, month, period = entry[:5]
        T, NC, NF = feat.shape
        BAR_LOCAL   = _cx.BAR
        # FIX: warmup = MIN(36h, T/8) — достаточно для инициализации EMA,
        # но не съедает период целиком (старый 100*BAR=6000 > BC_MAX_BARS=4000)
        warmup_bars = min(36 * BAR_LOCAL, T // 8)
        T_after = T - warmup_bars
        if BC_MAX_BARS > 0 and T_after > BC_MAX_BARS:
            t_indices = np.sort(rng_local.choice(T_after, BC_MAX_BARS, replace=False) + warmup_bars)
        else:
            t_indices = np.arange(warmup_bars, T)

        try:
            agent_inst = agent_cls()
            # ── Адаптируем вызов act() к сигнатуре агента ─────────────────────
            import inspect as _insp
            _sig       = _insp.signature(agent_inst.act)
            _has_month = 'month' in _sig.parameters
            _has_pv    = 'portfolio_value' in _sig.parameters
            _has_bi    = 'bar_index' in _sig.parameters

            def _call_act(a_inst, p_d, v_d, month_v, bar_t):
                kwargs: dict = {}
                if _has_month: kwargs['month'] = month_v
                # portfolio_value=1.0 (не None!) — агенты с from_peak DD-стопом
                # иначе пропускают все сделки (peak=None → stop не инициализируется)
                if _has_pv:    kwargs['portfolio_value'] = 1.0
                if _has_bi:    kwargs['bar_index'] = bar_t
                return a_inst.act(p_d, v_d, **kwargs)

            for t in range(warmup_bars):
                p_d = {syms[c]: float(prices[t, c]) for c in range(NC)}
                v_d = {syms[c]: float(max(0.0, 1.0 + feat[t, c, 7])) for c in range(NC)}
                try:
                    _call_act(agent_inst, p_d, v_d, month, t)
                except Exception:
                    pass

            for t in t_indices:
                p_d = {syms[c]: float(prices[t, c]) for c in range(NC)}
                v_d = {syms[c]: float(max(0.0, 1.0 + feat[t, c, 7])) for c in range(NC)}
                try:
                    raw_acts = _call_act(agent_inst, p_d, v_d, month, t)
                except Exception:
                    raw_acts = {s: 0 for s in syms}
                for c, s in enumerate(syms):
                    action = int(raw_acts.get(s, 0))
                    # BUG-FIX #8: был remap action=6→5, мотивированный устаревшим
                    # комментарием "action=6 (close_all в бирже)".
                    # Актуально: action=6 = fut_short_half (и в бирже, и в тренере).
                    # Remap 6→5 превращал шорт в fut_long_full — BC учился НЕВЕРНОМУ
                    # направлению! Удалено.
                    if action in range(N_ACTIONS):
                        X_all.append(feat[t, c].copy())
                        y_all.append(action)
        except Exception as e:
            # Показываем каждый класс ошибки только 1 раз (не 14 строк одинаковых)
            _err_key = type(e).__name__ + str(e)[:60]
            if _err_key not in _seen_err_types:
                _seen_err_types.add(_err_key)
                logs.append(f"    [BC] ошибка {agent_name}/{period}: {e}")

    if len(X_all) < 50:
        return agent_name, None, logs + [f"    [BC] {agent_name}: мало данных ({len(X_all)})"]

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    # ══════════════════════════════════════════════════════════════════════
    # FIX: ФИЛЬТР КАЧЕСТВА BC — исключаем агентов с нулевой торговой активностью
    # ══════════════════════════════════════════════════════════════════════
    # ПРОБЛЕМА (из логов): AdaptivePortfolio → 880,000 пар, hold=880000, buy=0
    #   CrossSectionalMomentumAgent → 900,000 пар, hold=900000, buy=0
    #   acc=100%, loss=0.0 — обучение прошло "идеально", НО сеть научилась ВСЕГДА держать
    #   (action=0). Эти агенты не торгуют и засоряют популяцию "hold everything" стратегией.
    #
    # Критерий фильтрации:
    #   active_ratio = доля шагов с ненулевым действием < 3% → ПРОПУСТИТЬ
    _raw_active_ratio = float((y != 0).mean())
    if _raw_active_ratio < 0.03:
        return agent_name, None, logs + [
            f"    [BC] {agent_name}: ПРОПУЩЕН (active_ratio={_raw_active_ratio:.1%} < 3% "
            f"— агент почти не торгует, засорит сеть стратегией 'hold everything')"]

    hold_idx   = np.where(y == 0)[0]
    active_idx = np.where(y != 0)[0]
    if len(active_idx) > 0 and len(hold_idx) > 5 * len(active_idx):
        keep = rng_local.choice(hold_idx, 5 * len(active_idx), replace=False)
        idx  = np.concatenate([keep, active_idx])
        X, y = X[idx], y[idx]

    logs.append(f"    [BC] {agent_name}: {len(y):,} пар  "
                f"hold={sum(y==0)} buy={sum(y==1)} sell={sum(y==2)} "
                f"long={sum(y==3)} short={sum(y==4)} cfut={sum(y==5)}  "
                f"active={_raw_active_ratio:.1%}")

    # Adam optimizer
    def he_init(fi, fo):
        return rng_local.normal(0, np.sqrt(2.0/fi), (fi, fo)).astype(np.float32)

    W1 = he_init(N_INPUT, N_HIDDEN1); b1 = np.zeros(N_HIDDEN1, np.float32)
    W2 = he_init(N_HIDDEN1, N_HIDDEN2); b2 = np.zeros(N_HIDDEN2, np.float32)
    W3 = he_init(N_HIDDEN2, N_HIDDEN3); b3 = np.zeros(N_HIDDEN3, np.float32)
    W4 = he_init(N_HIDDEN3, N_ACTIONS); b4 = np.zeros(N_ACTIONS, np.float32)

    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]

    Y_oh = np.eye(N_ACTIONS, dtype=np.float32)[y]
    N_s  = len(X); bsz = min(512, N_s)
    best_loss = np.inf; best_g = None
    step = 0

    for epoch in range(BC_EPOCHS):
        perm = rng_local.permutation(N_s)
        ep_loss = 0.0; nb = 0
        for start in range(0, N_s, bsz):
            bx = X[perm[start:start+bsz]]; by = Y_oh[perm[start:start+bsz]]; B = len(bx)
            a1 = bx @ W1 + b1
            h1 = np.where(a1 >= 0, a1, np.exp(np.clip(a1,-20,0))-1)
            a2 = h1 @ W2 + b2
            h2 = np.where(a2 >= 0, a2, np.exp(np.clip(a2,-20,0))-1)
            a3 = h2 @ W3 + b3
            h3 = np.where(a3 >= 0, a3, np.exp(np.clip(a3,-20,0))-1)
            z  = h3 @ W4 + b4; z -= z.max(1, keepdims=True)
            ez = np.exp(z); pr = ez / ez.sum(1, keepdims=True)
            loss = -np.mean(np.sum(by * np.log(pr + 1e-9), 1))
            ep_loss += loss; nb += 1
            dz  = (pr - by) / B
            dW4 = h3.T @ dz;  db4 = dz.sum(0)
            dh3 = dz @ W4.T * np.where(a3 >= 0, 1.0, h3 + 1.0)
            dW3 = h2.T @ dh3; db3_g = dh3.sum(0)
            dh2 = dh3 @ W3.T * np.where(a2 >= 0, 1.0, h2 + 1.0)
            dW2 = h1.T @ dh2; db2_g = dh2.sum(0)
            dh1 = dh2 @ W2.T * np.where(a1 >= 0, 1.0, h1 + 1.0)
            dW1 = bx.T @ dh1; db1_g = dh1.sum(0)
            grads = [dW1, db1_g, dW2, db2_g, dW3, db3_g, dW4, db4]
            step += 1
            bc1 = 1 - adam_beta1**step; bc2 = 1 - adam_beta2**step
            for pi, (p_arr, g_arr) in enumerate(zip(params, grads)):
                m_adam[pi] = adam_beta1 * m_adam[pi] + (1-adam_beta1) * g_arr
                v_adam[pi] = adam_beta2 * v_adam[pi] + (1-adam_beta2) * g_arr**2
                m_hat = m_adam[pi] / bc1; v_hat = v_adam[pi] / bc2
                p_arr -= BC_LR * m_hat / (np.sqrt(v_hat) + adam_eps)
        avg = ep_loss / max(nb, 1)
        if avg < best_loss:
            best_loss = avg
            best_g = np.concatenate([W1.ravel(), b1, W2.ravel(), b2,
                                      W3.ravel(), b3, W4.ravel(), b4]).astype(np.float32)

    if best_g is None:
        return agent_name, None, logs + [f"    [BC] {agent_name}: обучение не дало результата"]

    # BUG FIX: используем best_g (лучший чекпоинт), а не финальные веса эпохи
    _bg = best_g
    _i = 0
    _W1 = _bg[_i:_i+_W1s].reshape(N_INPUT,   N_HIDDEN1); _i += _W1s; _bv1 = _bg[_i:_i+_b1s]; _i += _b1s
    _W2 = _bg[_i:_i+_W2s].reshape(N_HIDDEN1, N_HIDDEN2); _i += _W2s; _bv2 = _bg[_i:_i+_b2s]; _i += _b2s
    _W3 = _bg[_i:_i+_W3s].reshape(N_HIDDEN2, N_HIDDEN3); _i += _W3s; _bv3 = _bg[_i:_i+_b3s]; _i += _b3s
    _W4 = _bg[_i:_i+_W4s].reshape(N_HIDDEN3, N_ACTIONS); _bv4 = _bg[_i+_W4s:_i+_W4s+_b4s]
    a1p = np.where((r1:=X @ _W1 + _bv1) >= 0, r1, np.exp(np.clip(r1,-20,0))-1)
    a2p = np.where((r2:=a1p @ _W2 + _bv2) >= 0, r2, np.exp(np.clip(r2,-20,0))-1)
    a3p = np.where((r3:=a2p @ _W3 + _bv3) >= 0, r3, np.exp(np.clip(r3,-20,0))-1)
    acc = float(((a3p @ _W4 + _bv4).argmax(1) == y).mean()) * 100
    logs.append(f"    [BC] {agent_name}: loss={best_loss:.4f}  acc={acc:.1f}%  ГОТОВ")
    return agent_name, best_g.tobytes(), logs


def _bc_run_one(a, result_q):
    """
    Цель mp.Process: запустить _bc_worker и положить результат в очередь.
    ОБЯЗАТЕЛЬНО на уровне модуля — локальные функции не пикклятся (Windows spawn).
    """
    # Устанавливаем флаг воркера ПЕРВЫМ — до любых импортов, которые могут печатать
    import os as _os
    _os.environ['_GENETICS_WORKER'] = '1'

    # Подавляем numba-сообщения при старте нового процесса
    import io as _io, sys as _sys
    _old_o, _old_e = _sys.stdout, _sys.stderr
    _sys.stdout = _sys.stderr = _io.StringIO()
    try:
        import numba  # noqa
    except ImportError:
        pass
    finally:
        _sys.stdout, _sys.stderr = _old_o, _old_e

    try:
        res = _bc_worker(a)
    except Exception as e:
        res = (a[0], None, [f"    [BC] {a[0]} ошибка процесса: {e}"])
    try:
        result_q.put(res)
    except Exception:
        pass


def _discover_agent_classes() -> list:
    import inspect

    # Классы-обёртки и служебные, которые НЕЛЬЗЯ инстанциировать без параметров
    _SKIP_NAMES = {
        'DDStopWrapper', '_GeneticsAdapter', 'GeneticsAgent',
        '_PositionTracker', '_VirtualPV', '_SubAgentSlot',
        # ══════════════════════════════════════════════════════════════════
        # FIX: агенты-долгожители — стабильно превышают BC_AGENT_TIMEOUT.
        # Причина: внутренняя инициализация за O(T²) или сложный warm-up.
        # HourlyAgent, IntraDaySeasonalityAgent: тяжёлые часовые расчёты
        # MeanReversionBBRSI, VolCompressionAgent: rolling-window O(T²)
        # PlayerStop, PlayerSwing: сложный state machine с большой историей
        # Эти агенты редко завершаются за 120с → убиваются и только тратят время.
        # ══════════════════════════════════════════════════════════════════
        'HourlyAgent', 'IntraDaySeasonalityAgent',
        'MeanReversionBBRSI', 'VolCompressionAgent',
        'PlayerStop', 'PlayerSwing',
    }

    def _can_instantiate(cls) -> bool:
        """True если класс можно создать вызовом cls() без аргументов."""
        if cls.__name__.startswith('_'):
            return False
        if cls.__name__ in _SKIP_NAMES:
            return False
        try:
            sig = inspect.signature(cls.__init__)
            required = [
                p for name, p in sig.parameters.items()
                if name != 'self'
                and p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD)
            ]
            return len(required) == 0
        except (ValueError, TypeError):
            return False

    discovered = []
    # 1. crypto_agents.py
    try:
        import crypto_agents as _ca
        for name, obj in inspect.getmembers(_ca, inspect.isclass):
            if obj.__module__ != _ca.__name__:
                continue
            if callable(getattr(obj, 'act', None)) and _can_instantiate(obj):
                discovered.append(name)
    except Exception:
        pass

    # 2. crypto_players.py — игроки тоже полезны для BC-сеяния
    try:
        import crypto_players as _cp
        for name, obj in inspect.getmembers(_cp, inspect.isclass):
            if obj.__module__ != _cp.__name__:
                continue
            if name in discovered:
                continue
            if callable(getattr(obj, 'act', None)) and _can_instantiate(obj):
                discovered.append(name)
    except Exception:
        pass

    return discovered



def _seed_from_agents(pop: np.ndarray, precomp, rng: np.random.Generator) -> np.ndarray:
    if AGENT_SEED_FRAC <= 0:
        return pop

    # Объединяем явный список с автоматически найденными агентами
    all_agents = list(AGENT_SEED_LIST)
    for name in _discover_agent_classes():
        if name not in all_agents:
            all_agents.append(name)
    if not all_agents:
        return pop

    n_seed   = min(max(1, int(POP_SIZE * AGENT_SEED_FRAC)), POP_SIZE - ELITE_SIZE)
    n_agents = len(all_agents)
    print(f"  [BC] Клонирование: {n_agents} агентов -> {n_seed} слотов  "
          f"({BC_REGIMES_PER_TYPE} периодов/режим, все режимы)")

    import pickle as _pk, concurrent.futures as cfu
    tmp = tempfile.NamedTemporaryFile(suffix='_bc.pkl', delete=False)
    tmp.close()
    with open(tmp.name, 'wb') as f:
        # передаём ВСЕ периоды — BC сам выберет по BC_REGIMES_PER_TYPE каждого режима
        _pk.dump(precomp, f, protocol=4)

    seeds = rng.integers(0, 2**31, n_agents).tolist()
    args  = [(name, tmp.name, int(seeds[i]))
             for i, name in enumerate(all_agents)]

    genomes = {}
    # ProcessPoolExecutor.__exit__ на Windows блокируется если воркер завис.
    # Используем mp.Process напрямую — можно вызвать .terminate() по таймауту.
    # ВАЖНО: target должна быть функцией уровня модуля (не local) — иначе pickle упадёт на Windows spawn.
    IS_WIN = (os.name == 'nt')
    BC_AGENT_TIMEOUT = 180   # FIX: 120→180с: больше времени для сложных агентов
    BATCH_SIZE = min(n_agents, max(1, min(4 if IS_WIN else 6,
                                         _mp.cpu_count() - 1)))

    # Устанавливаем флаг ДО spawn — дочерние процессы наследуют его и не спамят логами
    os.environ['_GENETICS_WORKER'] = '1'
    ctx = _mp.get_context('spawn')

    done = 0
    # Обрабатываем агентов батчами BATCH_SIZE
    for batch_start in range(0, n_agents, BATCH_SIZE):
        batch = args[batch_start: batch_start + BATCH_SIZE]
        result_q = ctx.Queue()
        procs = []
        for a in batch:
            p = ctx.Process(target=_bc_run_one, args=(a, result_q), daemon=True)
            p.start()
            procs.append((p, a[0]))

        # Собираем результаты с таймаутом на каждый процесс
        deadline = time.time() + BC_AGENT_TIMEOUT
        collected = 0
        collected_names: set = set()   # FIX: дедупликация по имени агента
        while collected < len(batch) and time.time() < deadline:
            try:
                ag_name, gbytes, logs = result_q.get(timeout=1.0)
                if ag_name in collected_names:
                    continue   # FIX: уже получили от этого агента — игнорируем дубль
                collected_names.add(ag_name)
                for line in logs:
                    print(line)
                if gbytes is not None:
                    genomes[ag_name] = np.frombuffer(gbytes, dtype=np.float32).copy()
                done += 1
                print(f"  [BC] {done}/{n_agents} завершён: {ag_name}")
                collected += 1
                deadline = time.time() + BC_AGENT_TIMEOUT  # сброс на каждый успех
            except Exception:
                pass  # queue.Empty — ждём дальше

        # Принудительно убиваем всё что ещё живо
        for p, name in procs:
            if p.is_alive():
                p.terminate()
                p.join(5)
                if p.is_alive():
                    p.kill()
                if name not in collected_names:   # FIX: считаем только незавершённых
                    done += 1
                    print(f"  [BC] {done}/{n_agents} завершён (таймаут/убит): {name}")

    # Удаляем временный pkl
    try:
        os.unlink(tmp.name)
    except Exception:
        pass

    NOISE_LEVELS = [0.0, BC_NOISE * 1.5, BC_NOISE * 3, BC_NOISE * 6, BC_NOISE * 12]
    successful = [n for n in all_agents if genomes.get(n) is not None
                  and len(genomes[n]) == GENOME_SIZE]
    print(f"  [BC] Успешно обучено: {len(successful)}/{n_agents} агентов")

    if not successful:
        print(f"  [BC] Засеяно 0/{n_seed} особей  (успешно: 0/{n_agents})")
        return pop

    # ══════════════════════════════════════════════════════════════════════
    # FIX: BC агенты распределяются РАВНОМЕРНО по всем островам
    # ══════════════════════════════════════════════════════════════════════
    # ПРОБЛЕМА (из логов): BC занимал слоты 8-127 (Island 0 exclusively).
    # _seed_islands_from_regimes() затем перезаписывала слоты 0-99 (Island 0 specialist zone).
    # Выживало только 28 BC слотов (100-127) из 120! Islands 1 и 2 не получали BC вообще.
    #
    # СТАЛО: каждый остров получает по n_seed/N_ISLANDS BC-геномов,
    # начиная с середины острова (после зоны island_specialist):
    #   Island 0: слоты 100-199  (после specialist slots 0-99)
    #   Island 1: слоты 300-399  (после specialist slots 200-299)
    #   Island 2: слоты 500-599  (после specialist slots 400-499)
    # Таким образом BC не пересекается с island_specialist и выживает весь.
    _island_sz    = POP_SIZE // N_ISLANDS          # 200
    _specialist_n = _island_sz // 2               # 100 (первая половина = specialist zone)
    _bc_per_island = n_seed // N_ISLANDS           # 40 на каждый остров
    _extra_bc      = n_seed - _bc_per_island * N_ISLANDS  # остаток в island 0

    # Строим список слотов: сначала для island 0, затем 1, 2 (+ extra в island 0)
    bc_slots_by_island = []
    for _iid in range(N_ISLANDS):
        _base = _iid * _island_sz + _specialist_n    # начало BC-зоны острова
        _n    = _bc_per_island + (_extra_bc if _iid == 0 else 0)
        bc_slots_by_island.extend(range(_base, min(_base + _n, (_iid + 1) * _island_sz)))

    # Распределяем геномы: round-robin по агентам с шумом
    slots_per_agent = max(1, len(bc_slots_by_island) // len(successful))
    extra_slots     = len(bc_slots_by_island) - slots_per_agent * len(successful)
    slot_cursor = 0

    for i, name in enumerate(successful):
        if slot_cursor >= len(bc_slots_by_island):
            break
        g = genomes[name]
        agent_slots = slots_per_agent + (1 if i < extra_slots else 0)
        for vi in range(agent_slots):
            if slot_cursor >= len(bc_slots_by_island):
                break
            target_slot = bc_slots_by_island[slot_cursor]
            noise_lvl   = NOISE_LEVELS[vi % len(NOISE_LEVELS)]
            pop[target_slot] = g.copy() if noise_lvl == 0.0 else \
                                (g + rng.normal(0, noise_lvl, GENOME_SIZE)).astype(np.float32)
            slot_cursor += 1

    # Лог по островам для диагностики
    for _iid in range(N_ISLANDS):
        _base = _iid * _island_sz + _specialist_n
        _n    = _bc_per_island + (_extra_bc if _iid == 0 else 0)
        _regime = ISLAND_TO_REGIME.get(_iid, str(_iid))
        print(f"    [BC] island {_iid}[{_regime:>12s}]: "
              f"слоты {_base}-{min(_base+_n-1, (_iid+1)*_island_sz-1)}  ({_n} особей)")

    print(f"  [BC] Засеяно {slot_cursor}/{n_seed} особей  "
          f"(успешно: {len(genomes)}/{n_agents})")
    return pop


# ══════════════════════════════════════════════════════════════════════════════
# ТРЕНИРОВЩИК С ISLANDS + NES + PER-LAYER SIGMA
# ══════════════════════════════════════════════════════════════════════════════

class GeneticTrainer:

    def __init__(self, precomp):
        self.precomp    = precomp
        self.rng        = np.random.default_rng(SEED)
        # Island-specific precomp для Phase 1 (строится сразу после хранения precomp)
        self.island_precomp: Dict[int, list] = {}
        self.sigma      = MUTATION_SIGMA
        self.gen        = 0
        self.best_g     = None
        self.best_fit   = -np.inf
        self.history    = []
        self._loaded_history: list = []   # предыдущая сессия — серый слой на графике
        self.stagnation    = 0
        self.restart_count = 0
        self.exploit_mode  = True
        self.archive: List[Tuple[float, np.ndarray]] = []
        self.tournament_k  = TOURNAMENT_K
        self.cma_dir = np.zeros(GENOME_SIZE, dtype=np.float32)
        # restarts без улучшения подряд
        self.restarts_no_improve = 0
        # Счётчик ГЛУБОКОЙ стагнации: сколько раз подряд сработал force-diversity без улучшения
        # При deep_stagnation_count >= DEEP_STAGNATION_THRESH → sigma escalation + partial reset
        self.deep_stagnation_count: int = 0
        # Флаг: только что найден новый best → эксплуатация (sigma→min быстро)
        self._after_new_best: bool = False
        # NES gradient magnitude history
        self.nes_grad_mags = []
        # Счётчик подряд идущих генов с низким NES (NES-collapse detection)
        self._nes_low_streak: int = 0
        # Island diversity history
        self.island_diversity_hist = []
        # История возвратов по режимам для динамических весов (скользящее окно)
        self.regime_ret_history: Dict[str, deque] = {
            r: deque(maxlen=DYNAMIC_RW_WINDOW) for r in REGIME_WEIGHTS
        }

        # Лучший геном для каждого режима рынка: {regime_name: (best_return, genome)}
        self.best_per_regime: Dict[str, Tuple[float, np.ndarray]] = {}
        # Загруженные с диска режимные геномы для Phase 1 инициализации островов
        self._regime_genomes: Dict[str, np.ndarray] = {}

        # ── Лучшие геномы по островам (п.2) ──────────────────────────────────
        # {island_id: best_fitness}  — для сравнения при обновлении
        self._island_best_fits: Dict[int, float] = {}
        # {island_id: genome}  — лучший геном каждого острова за всё время
        self._island_best_genomes: Dict[int, np.ndarray] = {}
        # Загруженные с диска best-island геномы для засеивания
        self._loaded_island_genomes: Dict[int, np.ndarray] = {}

        self.pop = _xavier_pop(self.rng, POP_SIZE)
        self._init_population()

        # Инициализация островов: делим популяцию на N_ISLANDS равных частей
        island_size = POP_SIZE // N_ISLANDS
        self.islands = [
            list(range(i * island_size, (i+1) * island_size))
            for i in range(N_ISLANDS)
        ]
        # Остаток добавляем к последнему острову
        remainder_start = N_ISLANDS * island_size
        if remainder_start < POP_SIZE:
            self.islands[-1].extend(range(remainder_start, POP_SIZE))

        # Phase 1: засеиваем острова из режимных геномов (если загружены)
        if PHASE1_ENABLED and hasattr(self, '_regime_genomes') and self._regime_genomes:
            self._seed_islands_from_regimes()

        # Phase 1: строим island_precomp — подмножества периодов по режимам
        if PHASE1_ENABLED:
            self._build_island_precomp()

        gpu_type, gpu_dev = _detect_gpu()
        use_gpu = (N_WORKERS == 0) and (gpu_type == 'torch')
        if use_gpu:
            self.ev   = GPUEvaluator(gpu_dev)
            self.mode = 'gpu'
            print("  Режим: GPU (PyTorch)")
            # ── VRAM-диагностика при инициализации ───────────────────────────────
            # Предупреждаем если GPU_CHUNK_BARS слишком большой для данного POP_SIZE.
            # Пик VRAM за 1 чанк: G*(chunk*NC)*(H1+H2)*4 байт (h1+h2 одновременно)
            try:
                import torch as _torch_diag
                _vram_gb = _torch_diag.cuda.get_device_properties(gpu_dev).total_memory / 1024**3
                _peak_default = POP_SIZE * GPU_CHUNK_BARS * N_HIDDEN1 * 4 / 1024**3  # упрощённо h1
                _bytes_per_bar = POP_SIZE * 20 * (N_HIDDEN1 + N_HIDDEN2) * 4  # h1+h2 пик (NC=20 типично)
                _safe_chunk = max(32, int(_vram_gb * 1024**3 * 0.48 / max(1, _bytes_per_bar)))
                if _safe_chunk < GPU_CHUNK_BARS:
                    print(f"  [VRAM-WARNING] GPU_CHUNK_BARS={GPU_CHUNK_BARS} велик для G={POP_SIZE}!")
                    print(f"  [VRAM-WARNING] Пик h1+h2 при chunk={GPU_CHUNK_BARS}: "
                          f"{POP_SIZE*GPU_CHUNK_BARS*20*(N_HIDDEN1+N_HIDDEN2)*4/1024**3:.1f}GB "
                          f"vs VRAM={_vram_gb:.0f}GB")
                    print(f"  [VRAM-WARNING] Авто-chunk={_safe_chunk} будет применён во время обучения")
                else:
                    _peak_gb = POP_SIZE * GPU_CHUNK_BARS * 20 * (N_HIDDEN1 + N_HIDDEN2) * 4 / 1024**3
                    print(f"  [VRAM-OK] Пик≈{_peak_gb:.1f}GB при G={POP_SIZE}, chunk={GPU_CHUNK_BARS}, VRAM={_vram_gb:.0f}GB")
            except Exception:
                pass
        else:
            nw = max(1, min(max(1, _mp.cpu_count()-2), POP_SIZE)) if not N_WORKERS else N_WORKERS
            self.ev   = CPUEvaluator(precomp, nw)
            self.mode = 'cpu'
            print(f"  Режим: CPU ({nw} воркеров)")

        np.save(os.path.join(AGENTS_DIR, "genome_size.npy"),
                np.array([GENOME_SIZE], dtype=np.int32))

    def _init_population(self):
        import glob as _glob

        # ── Шаг 1: Загрузка/восстановление состояния ─────────────────────────
        warm_pop   = None   # загруженная популяция (если есть)
        full_state = False  # True = полное состояние с gen/sigma/best_fit

        state_path = os.path.join(AGENTS_DIR, "population_state.pkl")
        if CONTINUE_TRAINING and os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
                loaded_pop = state.get('pop')
                saved_n_islands = state.get('n_islands', N_ISLANDS)
                if (loaded_pop is not None
                        and loaded_pop.shape == (POP_SIZE, GENOME_SIZE)):
                    warm_pop   = loaded_pop.astype(np.float32)
                    full_state = True
                    self.sigma      = float(state.get('sigma',      MUTATION_SIGMA))
                    self.gen        = int(state.get('gen',           0))
                    self.best_fit   = float(state.get('best_fit',    -np.inf))
                    self.stagnation = int(state.get('stagnation',    0))
                    self.restart_count = int(state.get('restarts',   0))
                    # _loaded_history = СНИМОК прошлой сессии (только для серого фона)
                    # ОБЯЗАТЕЛЬНО list() — независимая копия, не ссылка!
                    _prev_session   = state.get('history', [])
                    self._loaded_history = list(_prev_session)   # снимок (не меняется)
                    self._session_start_idx = len(_prev_session) # граница новой сессии
                    # self.history накапливает: старые + новые gens
                    self.history = list(_prev_session)
                    arc_raw = state.get('archive', [])
                    self.archive = [(f, np.array(g, dtype=np.float32))
                                    for f, g in arc_raw
                                    if len(g) == GENOME_SIZE]
                    cma = state.get('cma_dir')
                    if cma is not None and len(cma) == GENOME_SIZE:
                        self.cma_dir = np.array(cma, dtype=np.float32)
                    # Restore per-island best fitness (for correct save thresholds)
                    saved_ibf = state.get('island_best_fits', {})
                    self._island_best_fits = {int(k): float(v)
                                              for k, v in saved_ibf.items()}
                    # Restore best_per_regime: из pkl (предпочтительно) или из .npy файлов
                    bpr_raw = state.get('best_per_regime', {})
                    if bpr_raw:
                        self.best_per_regime = {
                            r: (float(ret), np.array(g, dtype=np.float32))
                            for r, (ret, g) in bpr_raw.items()
                            if len(g) == GENOME_SIZE
                        }
                        print(f"  [warm-start] best_per_regime загружен: "
                              f"{list(self.best_per_regime.keys())}")
                    # Warn if island count changed
                    if saved_n_islands != N_ISLANDS:
                        print(f"  [warm-start] ⚠ N_ISLANDS changed: "
                              f"{saved_n_islands} → {N_ISLANDS}  (island files re-seeded)")
                    # ── ИСПРАВЛЕНИЕ: Очищаем архив от Phase1-геномов ─────────────
                    # Phase1 фитнес (3.5×mean_r) → до ~97.
                    # Единый фитнес (2.5×mean_r + risk penalties) → до ~20.
                    # Phase1-записи в архиве создавали непреодолимую планку:
                    # best_fit=97 никогда не побить в новой шкале → вечная стагнация.
                    _FIT_PHASE1_THRESHOLD = 25.0
                    _old_bf = self.best_fit
                    if self.best_fit > _FIT_PHASE1_THRESHOLD:
                        self.best_fit = -np.inf
                        print(f"  [warm-start] ⚠ best_fit={_old_bf:+.4f} из Phase1 сброшен "
                              f"(> {_FIT_PHASE1_THRESHOLD} → несовместимо с новой шкалой фитнеса)")
                    _n_arc_before = len(self.archive)
                    self.archive = [(f, g) for f, g in self.archive
                                    if f <= _FIT_PHASE1_THRESHOLD]
                    if len(self.archive) < _n_arc_before:
                        print(f"  [warm-start] ⚠ Архив очищен от Phase1-записей: "
                              f"{_n_arc_before} → {len(self.archive)} "
                              f"(удалены с fitness > {_FIT_PHASE1_THRESHOLD})")
                    print(f"  [warm-start] ПОЛНАЯ популяция загружена: "
                          f"gen={self.gen}  sigma={self.sigma:.4f}  "
                          f"best_fit={self.best_fit:+.4f}")
                elif (loaded_pop is not None
                      and loaded_pop.ndim == 2
                      and loaded_pop.shape[1] == GENOME_SIZE):
                    warm_pop = loaded_pop.astype(np.float32)
                    self.sigma = float(state.get('sigma', MUTATION_SIGMA))
                    arc_raw = state.get('archive', [])
                    self.archive = [(f, np.array(g, dtype=np.float32))
                                    for f, g in arc_raw
                                    if len(g) == GENOME_SIZE]
                    n_warm = warm_pop.shape[0]
                    print(f"  [warm-start] Частичная загрузка: {n_warm}/{POP_SIZE}"
                          + (f"  (POP_SIZE изменился: {n_warm}→{POP_SIZE})"
                             if n_warm != POP_SIZE else ""))
                else:
                    print(f"  [warm-start] несовместимый геном, игнорируем")
            except Exception as e:
                print(f"  [warm-start] ошибка чтения state: {e}")

        # Дополнительно: best_genome.npy и archive/*.npy / gen*.npy
        extra_genomes: List[np.ndarray] = []
        best_path = os.path.join(AGENTS_DIR, "best_genome.npy")
        if LOAD_BEST_GENOME and os.path.exists(best_path):
            try:
                g = np.load(best_path).astype(np.float32).ravel()
                if len(g) == GENOME_SIZE:
                    extra_genomes.append(g)
                    self.best_g = g.copy()
                    print("  [warm-start] best_genome загружен")
            except Exception:
                pass

        if LOAD_EXTRA_GENOMES:
            skip_names = {"genome_size.npy", "best_genome.npy"}
            npy_files  = sorted(_glob.glob(os.path.join(AGENTS_DIR, "*.npy")))
            if LOAD_EXTRA_MAX > 0:
                npy_files = npy_files[:LOAD_EXTRA_MAX]
            for p in npy_files:
                if os.path.basename(p) in skip_names:
                    continue
                try:
                    g = np.load(p).astype(np.float32).ravel()
                    if len(g) == GENOME_SIZE:
                        extra_genomes.append(g)
                except Exception:
                    pass
            if extra_genomes:
                print(f"  [warm-start] Доп. геномов из .npy: {len(extra_genomes)}")

        # Загружаем лучшие геномы по режимам для Phase 1 засеивания островов
        if PHASE1_ENABLED and LOAD_REGIME_GENOMES:
            loaded_regimes = []
            for regime in REGIME_ORDER:
                rpath = os.path.join(AGENTS_DIR, f"best_genome_regime_{regime}.npy")
                if os.path.exists(rpath):
                    try:
                        g = np.load(rpath).astype(np.float32).ravel()
                        if len(g) == GENOME_SIZE:
                            self._regime_genomes[regime] = g
                            loaded_regimes.append(regime)
                            # Fallback: заполняем best_per_regime из .npy если pkl не содержал
                            if regime not in self.best_per_regime:
                                self.best_per_regime[regime] = (0.0, g.copy())
                    except Exception:
                        pass
            if loaded_regimes:
                print(f"  [Phase1] Загружено режимных геномов: {len(loaded_regimes)}  "
                      f"({', '.join(loaded_regimes)})")

        # ── Загрузка лучших геномов островов ─────────────────────────────────
        if LOAD_ISLAND_GENOMES:
            loaded_islands = []
            for iid in range(N_ISLANDS):
                ipath = island_file(iid)
                if os.path.exists(ipath):
                    try:
                        g = np.load(ipath).astype(np.float32).ravel()
                        if len(g) == GENOME_SIZE:
                            self._loaded_island_genomes[iid] = g
                            loaded_islands.append(
                                f"{ISLAND_TO_REGIME.get(iid, str(iid))}"
                            )
                    except Exception:
                        pass
            if loaded_islands:
                print(f"  [Islands] Загружено геномов: {len(loaded_islands)}  "
                      f"({', '.join(loaded_islands)})")

        # ── Шаг 2: BC-сеяние от агентов/игроков ──────────────────────────────
        bc_pop = None
        if BC_ENABLED and AGENT_SEED_FRAC > 0 and self.precomp:
            # Создаём временную популяцию только для BC — не трогаем self.pop
            _tmp = _xavier_pop(self.rng, POP_SIZE)
            bc_pop = _seed_from_agents(_tmp, self.precomp, self.rng)
            # bc_pop[ELITE_SIZE : ELITE_SIZE+n_seed] = BC-геномы
            # bc_pop[ELITE_SIZE+n_seed :] = случайные xavier (нам не нужны)

        # ── Шаг 3: Сборка финальной популяции ────────────────────────────────
        #
        # Приоритет слотов:
        #   [0  ..  ELITE_SIZE-1]  — элита из warm-start (или xavier)
        #   [ELITE_SIZE .. BC_END] — BC-агенты (30% от POP_SIZE)
        #   [BC_END .. POP_SIZE-1] — остаток warm-start | extra_genomes | xavier
        #
        # Если warm-start есть — элита ВСЕГДА берётся из него.
        # BC-агенты добавляются в дополнительные слоты, не конкурируют с элитой.

        n_bc_slots = max(0, int(POP_SIZE * AGENT_SEED_FRAC)) if bc_pop is not None else 0

        # Инициализируем self.pop
        if warm_pop is not None:
            if warm_pop.shape[0] == POP_SIZE:
                self.pop = warm_pop.copy()
            else:
                n_copy = min(warm_pop.shape[0], POP_SIZE)
                self.pop[:n_copy] = warm_pop[:n_copy]
                print(f"  [warm-start] Частичная вставка {n_copy}/{POP_SIZE} → "
                      f"остаток ({POP_SIZE - n_copy}) заполнен xavier")

        # Вставляем extra_genomes (только если нет warm или частичная загрузка)
        _partial_warm = warm_pop is not None and warm_pop.shape[0] < POP_SIZE
        if (warm_pop is None or _partial_warm) and extra_genomes:
            slot = max(ELITE_SIZE, warm_pop.shape[0] if _partial_warm else ELITE_SIZE)
            for g in extra_genomes:
                if slot >= POP_SIZE:
                    break
                self.pop[slot] = g
                slot += 1
            print(f"  [warm-start] extra геномов -> особи #{max(ELITE_SIZE, warm_pop.shape[0] if _partial_warm else ELITE_SIZE)}..#{slot-1}")

        # ══════════════════════════════════════════════════════════════════
        # FIX: BC-геномы распределяются по ВСЕМ островам, не только island 0
        # ══════════════════════════════════════════════════════════════════
        # БЫЛО: bc_pop[ELITE_SIZE:ELITE_SIZE+n_bc_slots] → все слоты 8-127 в Island 0.
        #   _seed_islands_from_regimes() затем перезаписывала слоты 0-99 (specialist zone Island 0).
        #   Выживало только 28 BC-геномов из 120. Islands 1 и 2 BC не получали совсем.
        #
        # СТАЛО: BC размещается в «свободных зонах» (вторая половина каждого острова),
        #   которые не перекрываются с island_specialist:
        #     Island 0: слоты 100-199
        #     Island 1: слоты 300-399
        #     Island 2: слоты 500-599
        if bc_pop is not None and n_bc_slots > 0:
            _island_sz     = POP_SIZE // N_ISLANDS   # 200
            _specialist_n  = _island_sz // 2         # 100
            _bc_per_island = n_bc_slots // N_ISLANDS
            _extra_bc      = n_bc_slots - _bc_per_island * N_ISLANDS
            _placed = 0
            for _iid in range(N_ISLANDS):
                _base = _iid * _island_sz + _specialist_n
                _n    = _bc_per_island + (_extra_bc if _iid == 0 else 0)
                _src_start = _iid * _bc_per_island + (0 if _iid > 0 else 0)
                # Берём BC-геномы из bc_pop в зоне island 0 (они там лежат последовательно)
                # и раскладываем по свободным зонам каждого острова
                for _k in range(_n):
                    _dst = _base + _k
                    _src = ELITE_SIZE + _placed
                    if _dst < (_iid + 1) * _island_sz and _src < POP_SIZE:
                        self.pop[_dst] = bc_pop[_src]
                        _placed += 1
            print(f"  [init] BC-геномы распределены по всем островам ({_placed} особей):")
            for _iid in range(N_ISLANDS):
                _base = _iid * _island_sz + _specialist_n
                _n    = _bc_per_island + (_extra_bc if _iid == 0 else 0)
                _rname = ISLAND_TO_REGIME.get(_iid, str(_iid))
                print(f"    island {_iid}[{_rname:>12s}]: слоты {_base}-{_base+_n-1}  ({_n} особей)")

        bc_end = ELITE_SIZE  # больше не используем bc_end для маршрутизации

        if warm_pop is not None:
            print(f"  [init] Итого: {ELITE_SIZE} элита (warm) + "
                  f"{n_bc_slots} BC-агенты (по островам) + "
                  f"{POP_SIZE - ELITE_SIZE - n_bc_slots} из предыдущей популяции")

        # ── Phase 2 warm-seed: засеваем популяцию мутантами best_g ──────────
        # При warm-start с PHASE2_ENABLED: non-elite слоты заполняются micro-мутантами
        # лучшего генома (best_g). Это гарантирует, что Phase 2 начинается
        # с плотного облака вокруг best_g, а не с полностью случайной популяции.
        # Только если best_g реально загружен (warm-start), иначе бессмысленно.
        if PHASE2_ENABLED and self.best_g is not None and warm_pop is not None:
            _t0_p2seed = time.time()
            n_p2_seed = POP_SIZE - ELITE_SIZE   # слоты ELITE_SIZE:POP_SIZE
            # VECTORIZED: генерируем все мутанты сразу — ~100x быстрее Python-цикла
            # noise_scales: (N,) — индивидуальный масштаб шума для каждой особи
            _noise_scales = (self.rng.uniform(0.5, 2.5, n_p2_seed) * PHASE2_SIGMA
                             ).astype(np.float32)
            # masks: (N, G) boolean — какие гены мутируют
            _p2_masks = self.rng.random((n_p2_seed, GENOME_SIZE)) < MUTATION_RATE
            # noise_matrix: (N, G) — шум масштабирован по строкам
            _p2_noise = (self.rng.standard_normal((n_p2_seed, GENOME_SIZE)).astype(np.float32)
                         * _noise_scales[:, None])
            _p2_noise *= _p2_masks   # обнуляем незамаскированные гены
            _p2_candidates = np.clip(
                self.best_g[None, :] + _p2_noise, -5.0, 5.0
            ).astype(np.float32)
            self.pop[ELITE_SIZE:POP_SIZE] = _p2_candidates
            print(f"  [Phase2-seed] {n_p2_seed} особей засеяны как micro-мутанты best_g "
                  f"(σ≈{PHASE2_SIGMA:.3f}, {time.time()-_t0_p2seed:.2f}s — готовы к стабилизации)")

    # ──  Построение island-specific precomp для Phase 1 ─────────────────
    def _build_island_precomp(self):
        """
        Phase 1 v2: каждый остров обучается на ПОЛНОМ наборе периодов,
        но с островными весами периодов вместо жёсткого разбиения по подвыборкам.

        Преимущества перед старым подходом (обучение только на своих периодах):
          ✓ Нет катастрофического обвала фитнеса при переходе Phase1→Phase2
          ✓ Агент с самого начала видит все рыночные условия
          ✓ Специализация достигается через усиленные веса своего режима,
            а не через исключение других периодов
          ✓ Плавный переход к Phase 2 — веса периодов становятся равными (REGIME_WEIGHTS)

        Структура island_precomp[island_id]:
          Список тех же entry из self.precomp, но с модифицированным entry[5] (rw),
          отражающим островную специализацию.
        """
        self.island_precomp = {}
        for island_id in range(N_ISLANDS):
            island_weights = ISLAND_PHASE1_REGIME_WEIGHTS.get(island_id, {})
            weighted_precomp = []
            for entry in self.precomp:
                regime = map_regime_3(entry[6]) if len(entry) > 6 and isinstance(entry[6], str) else 'neutral'
                # FIX: используем ТОЛЬКО island_rw без умножения на global REGIME_WEIGHTS.
                # Было: final_rw = island_rw × global_rw (REGIME_WEIGHTS).
                # Это уничтожало специализацию — strong_crash (global=5.0) всегда доминировал.
                # Island 1 (bear): strong_crash: 0.30×5.0×19пер=28.5 vs bear: 1.0×3.0×6пер=18.0
                # → медвежий остров учился на обвалах вместо медвежьего рынка!
                final_rw = island_weights.get(regime, PHASE1_MIN_WEIGHT)

                new_entry = list(entry)
                new_entry[5] = final_rw
                weighted_precomp.append(tuple(new_entry))

            self.island_precomp[island_id] = weighted_precomp
            regime_name = ISLAND_TO_REGIME.get(island_id, str(island_id))

            # Детальная статистика: суммарный вес и кол-во периодов по режимам
            rw_by_regime:  Dict[str, float] = {}
            cnt_by_regime: Dict[str, int]   = {}
            for entry in weighted_precomp:
                r = map_regime_3(entry[6]) if len(entry) > 6 else 'neutral'
                rw_by_regime[r]  = rw_by_regime.get(r, 0.0) + float(entry[5])
                cnt_by_regime[r] = cnt_by_regime.get(r, 0)  + 1
            all_r = sorted(rw_by_regime.items(), key=lambda x: -x[1])
            top3_str = '  '.join(
                f"{r}:{w:.2f}(n={cnt_by_regime[r]})" for r, w in all_r[:3])
            print(f"  [Phase1] Island {island_id} [{regime_name:>12s}]: "
                  f"{len(weighted_precomp)} периодов  top3_weights: {top3_str}")

    @staticmethod
    def _period_regime(rw: float, entry=None) -> str:
        """Преобразует режимный вес обратно в имя режима.
        Если entry передан — использует строку режима напрямую (entry[6])."""
        if entry is not None and len(entry) > 6 and isinstance(entry[6], str):
            return map_regime_3(entry[6])
        return next((k for k, v in REGIME_WEIGHTS.items()
                     if abs(v - rw) < 0.01), 'bull')

    def _compute_dynamic_regime_weights(self) -> Optional[Dict[str, float]]:
        """
        Вычисляет динамические веса режимов на основе скользящего окна истории.

        Логика:
          - Если mean_ret по режиму > DYNAMIC_RW_TARGET → снижаем вес
            (модель хорошо торгует здесь, не нужно переобучаться)
          - Если mean_ret < DYNAMIC_RW_TARGET → повышаем вес
            (модель слабая здесь, нужно больше обучения)

        Формула мультипликатора:
          mult = clip(1.0 - β × (mean_ret − target) / ref, rw_min, rw_max)
          где ref = 5.0% (нормировка к ожидаемому диапазону ±5%)

        Применяется ТОЛЬКО в Phase 2 (не island Phase1) чтобы не ломать
        специализацию островов. В Phase1 острова используют ISLAND_PHASE1_REGIME_WEIGHTS.
        """
        if not DYNAMIC_REGIME_WEIGHTS_ENABLED:
            return None

        result: Dict[str, float] = {}
        any_history = False
        for regime, base_w in REGIME_WEIGHTS.items():
            history = self.regime_ret_history.get(map_regime_3(regime))
            if not history or len(history) < 3:
                result[regime] = base_w   # недостаточно данных — базовый вес
                continue
            any_history = True
            mean_ret = float(np.mean(history))
            # Нормируем на ±5% диапазон → multiplier ∈ [DYNAMIC_RW_MIN, DYNAMIC_RW_MAX]
            deviation = (mean_ret - DYNAMIC_RW_TARGET) / 5.0
            mult = 1.0 - DYNAMIC_RW_BETA * deviation
            mult = float(np.clip(mult, DYNAMIC_RW_MIN, DYNAMIC_RW_MAX))
            result[regime] = base_w * mult

        if not any_history:
            return None   # нет данных — не применяем

        # Нормируем чтобы сумма весов сохранялась как у базовых REGIME_WEIGHTS
        base_sum    = sum(REGIME_WEIGHTS.values())
        dynamic_sum = sum(result.values()) + 1e-9
        scale = base_sum / dynamic_sum
        result = {r: w * scale for r, w in result.items()}

        return result

    def _update_regime_ret_history(self, br: list):
        """Обновляет историю возвратов по режимам после каждого поколения.
        br — список возвратов лучшего агента для каждого периода (по self.precomp).
        Группирует по режиму и добавляет mean_ret по режиму в скользящее окно.
        """
        if not br or not DYNAMIC_REGIME_WEIGHTS_ENABLED:
            return
        regime_rets: Dict[str, list] = {}
        for pi, entry in enumerate(self.precomp):
            raw_reg = entry[6] if len(entry) > 6 and isinstance(entry[6], str) else None
            if raw_reg and pi < len(br):
                reg = map_regime_3(raw_reg)   # гарантируем 3-label
                regime_rets.setdefault(reg, []).append(float(br[pi]))
        for reg, vals in regime_rets.items():
            if reg in self.regime_ret_history and vals:
                self.regime_ret_history[reg].append(float(np.mean(vals)))

    @staticmethod
    def _batch_forward_numpy(population: np.ndarray, feat: np.ndarray) -> np.ndarray:
        """
        Векторизованный numpy forward pass (CPU Phase 1).
        Аналог GPUEvaluator._batch_forward, но без CUDA.
        Chunk-based по времени для ограничения пикового расхода памяти.
        Возвращает actions: (G, T, NC) int8.
        """
        G = len(population)
        T, NC, NF = feat.shape
        # Распаковываем все геномы сразу — быстрее одиночных _unpack
        i = 0
        W1 = population[:, i:i+_W1s].reshape(G, N_INPUT,   N_HIDDEN1); i += _W1s
        b1 = population[:, i:i+_b1s];                                    i += _b1s
        W2 = population[:, i:i+_W2s].reshape(G, N_HIDDEN1, N_HIDDEN2);  i += _W2s
        b2 = population[:, i:i+_b2s];                                    i += _b2s
        W3 = population[:, i:i+_W3s].reshape(G, N_HIDDEN2, N_HIDDEN3);  i += _W3s
        b3 = population[:, i:i+_b3s];                                    i += _b3s
        W4 = population[:, i:i+_W4s].reshape(G, N_HIDDEN3, N_ACTIONS);  i += _W4s
        b4 = population[:, i:i+_b4s]

        CHUNK = 64  # баров в чанке: (G=85, 64*NC, H1) ≈ 200 MB пик
        feat_2d = feat.reshape(T * NC, NF).astype(np.float32)
        all_acts = []

        for t0 in range(0, T, CHUNK):
            t1 = min(t0 + CHUNK, T)
            tc = t1 - t0
            x = feat_2d[t0*NC : t1*NC]          # (tc*NC, NF)

            # einsum 'nf,gfh->gnh' — x broadcast по G без expand
            h1_r = np.einsum('nf,gfh->gnh', x, W1) + b1[:, None, :]
            h1   = np.where(h1_r >= 0, h1_r,
                            np.expm1(np.clip(h1_r, -20.0, 0.0)))

            h2_r = np.einsum('gnh,ghk->gnk', h1, W2) + b2[:, None, :]
            h2   = np.where(h2_r >= 0, h2_r,
                            np.expm1(np.clip(h2_r, -20.0, 0.0)))

            h3_r = np.einsum('gnk,gkm->gnm', h2, W3) + b3[:, None, :]
            h3   = np.where(h3_r >= 0, h3_r,
                            np.expm1(np.clip(h3_r, -20.0, 0.0)))

            out  = np.einsum('gnm,gma->gna', h3, W4) + b4[:, None, :]
            acts = out.argmax(axis=-1).reshape(G, tc, NC).astype(np.int8)
            all_acts.append(acts)
            del h1_r, h1, h2_r, h2, h3_r, h3, out   # освобождаем RAM

        return np.concatenate(all_acts, axis=1)       # (G, T, NC)

    def _eval_island_batch(self, island_pop: np.ndarray, island_precomp: list,
                            rw_override: Optional[Dict[str, float]] = None):
        """
        Векторизованная оценка острова на island-specific precomp.
        GPU режим: использует GPUEvaluator.evaluate напрямую.
        CPU режим: numpy einsum forward + Numba simulate_batch.

        rw_override: если задан и мы НЕ в Phase1-island режиме, переопределяет
                     веса режимов. Для Phase1 island_precomp уже содержит
                     island-specific веса → rw_override игнорируется.

        Возвращает (fits[G], rets_list[G]).
        """
        G = len(island_pop)
        IC = _cx.INITIAL_CAPITAL
        snap_every = _get_snap_every()

        if self.mode == 'gpu':
            return self.ev.evaluate(island_pop, island_precomp, rw_override=rw_override)

        # ── CPU path: батчевый numpy forward ──────────────────────────────
        rl_all, dl_all, trl_all = [], [], []
        dpv_all: List[Optional[np.ndarray]] = []
        rw_list: List[float] = []

        for entry in island_precomp:
            feat, prices, syms, month, period = entry[:5]
            rw     = float(entry[5]) if len(entry) > 5 else 1.0
            regime = map_regime_3(entry[6]) if len(entry) > 6 else 'neutral'
            rw_list.append(rw)

            # Векторизованный forward pass для всего острова
            actions_all = self._batch_forward_numpy(island_pop, feat)  # (G,T,NC)
            # Выбор оптимальных валют
            if CURRENCY_SELECTION_ENABLED:
                actions_all = _apply_currency_selection(
                    actions_all.astype(np.int32), feat, regime, TOP_CURRENCIES_N
                ).astype(np.int8)

            rets_g, dds_g, trs_g, dpv_g = simulate_batch(
                actions_all.astype(np.int32), prices, IC, snap_every)
            rl_all.append(rets_g)
            dl_all.append(dds_g)
            trl_all.append(trs_g)
            dpv_all.append(dpv_g)

        if not rl_all:
            return np.full(G, -np.inf), [[] for _ in range(G)]

        ret_mat = np.column_stack(rl_all)   # (G, P)
        dd_mat  = np.column_stack(dl_all)
        tr_mat  = np.column_stack(trl_all)
        rw_arr  = np.array(rw_list, dtype=np.float64)
        fits    = _compute_fitness(ret_mat, dd_mat, tr_mat, dpv_all, rw_arr)
        rets_list = [ret_mat[g].tolist() for g in range(G)]
        return fits, rets_list

    # ──  Phase 1: засеивание островов режимными геномами ────────────────
    def _seed_islands_from_regimes(self):
        """
        Каждый остров соответствует ровно одному режиму рынка.
        Засеиваем первую половину острова лучшим известным геномом для этого режима:
          1. best_island_{regime}.npy — накопленный специалист (высший приоритет)
          2. best_genome_regime_{regime}.npy — режимный геном из прошлых сессий
        Вторая половина острова сохраняет текущую популяцию (разнообразие).
        """
        if not self._regime_genomes and not self._loaded_island_genomes:
            return
        seeded = 0
        for island_id, regimes in enumerate(ISLAND_REGIME_GROUPS):
            if island_id >= len(self.islands):
                break
            island_idx = self.islands[island_id]
            regime     = regimes[0]   # строго один режим на остров

            # Приоритет 1: island-specialist (накоплен этим же островом)
            best_g = self._loaded_island_genomes.get(island_id)
            source_label = f"island_specialist"

            # Приоритет 2: режимный геном (общая сессия)
            if best_g is None and regime in self._regime_genomes:
                best_g = self._regime_genomes[regime]
                source_label = f"regime_genome"

            if best_g is None:
                continue

            n_seed = max(1, len(island_idx) // 2)
            for k in range(n_seed):
                slot = island_idx[k]
                noise_scale = self.sigma * 0.3 * (k / max(n_seed - 1, 1))
                if noise_scale < 1e-6:
                    self.pop[slot] = best_g.copy()
                else:
                    self.pop[slot] = (
                        best_g + self.rng.normal(0, noise_scale, GENOME_SIZE).astype(np.float32)
                    )
                seeded += 1

            print(f"  [Phase1] Island {island_id} [{regime:>14s}] "
                  f"← {source_label}  ({n_seed} seeded)")

        if seeded:
            phase1_gens = int(N_GENERATIONS * PHASE1_GENS_FRAC)
            print(f"  [Phase1] Total seeded {seeded}/{POP_SIZE}  "
                  f"(phase1: gen 1..{phase1_gens}, no migration)")

    # ──  Сохранение лучшего генома по режимам ───────────────────────────
    def _save_best_per_regime(self, br: list, genome: Optional[np.ndarray] = None):
        """
        Обновляет best_genome_regime_{regime}.npy при улучшении среднего
        возврата по этому режиму. Одновременно обновляет best_island_{regime}.npy
        (файл острова, специализированного на этом режиме).
        """
        g = genome if genome is not None else self.best_g
        if g is None:
            return
        updated = []
        for i, entry in enumerate(self.precomp):
            if i >= len(br):
                break
            rw     = entry[5]
            regime = map_regime_3(entry[6]) if len(entry) > 6 and isinstance(entry[6], str) else None
            if regime is None:
                continue
            ret = float(br[i])
            cur = self.best_per_regime.get(regime)
            if cur is None or ret > cur[0]:
                self.best_per_regime[regime] = (ret, g.copy())
                # Сохраняем режимный геном
                regime_path = os.path.join(AGENTS_DIR, f"best_genome_regime_{regime}.npy")
                np.save(regime_path, g)
                # Синхронизируем файл острова (best_island_{regime}.npy)
                iid = REGIME_TO_ISLAND.get(regime)
                if iid is not None:
                    np.save(island_file(iid), g)
                    # Обновляем in-memory best для острова тоже
                    cur_island_fit = self._island_best_fits.get(iid, -np.inf)
                    if ret > cur_island_fit:
                        self._island_best_fits[iid] = ret
                        self._island_best_genomes[iid] = g.copy()
                updated.append(f"{regime}({ret:+.1f}%)")
        if updated:
            print(f"  [Regime] Updated: {', '.join(updated)}")

    # ──  Сохранение лучших геномов по островам ───────────────────────────
    def _save_island_bests(self, fits: np.ndarray):
        """
        После каждого поколения: для каждого острова — лучший индивид.
        Если улучшился → сохраняем в best_island_{regime}.npy.
        Также синхронизируем best_genome_regime_{regime}.npy.
        Дополнительно: best_island_{regime}_meta.json — метаданные для _RegimeGeneticsAgent.
        """
        updated = []
        for island_id, island_idx in enumerate(self.islands):
            if not island_idx:
                continue
            island_fits    = fits[island_idx]
            best_local_k   = int(np.argmax(island_fits))
            best_fit_local = float(island_fits[best_local_k])
            best_slot      = island_idx[best_local_k]
            regime         = ISLAND_TO_REGIME.get(island_id, f"island{island_id}")

            cur_best = self._island_best_fits.get(island_id, -np.inf)
            if best_fit_local > cur_best:
                self._island_best_fits[island_id]   = best_fit_local
                self._island_best_genomes[island_id] = self.pop[best_slot].copy()
                # Файл острова: best_island_{regime}.npy
                np.save(island_file(island_id), self.pop[best_slot])
                # Синхронизируем режимный геном
                regime_path = os.path.join(AGENTS_DIR, f"best_genome_regime_{regime}.npy")
                np.save(regime_path, self.pop[best_slot])
                # ── Метаданные для _RegimeGeneticsAgent (аналог best_genome_meta.json) ──
                _island_meta = {
                    'gen':         self.gen,
                    'fitness':     best_fit_local,
                    'regime':      regime,
                    'island_id':   island_id,
                    'max_pos':     int(TRAIN_MAX_POS),
                    'genome_size': int(GENOME_SIZE),
                }
                try:
                    _meta_path = os.path.join(AGENTS_DIR, f"best_island_{regime}_meta.json")
                    with open(_meta_path, 'w') as _mf:
                        json.dump(_island_meta, _mf)
                except Exception:
                    pass
                updated.append(f"{regime}({best_fit_local:+.4f})")

        if updated:
            print(f"  [Islands] Updated: {', '.join(updated)}")

    def _save_state(self):
        # Сериализуем best_per_regime: (ret, genome) → (ret, genome.tolist())
        bpr_serial = {r: (float(ret), g.tolist())
                      for r, (ret, g) in self.best_per_regime.items()}
        state = {
            'pop':             self.pop,
            'sigma':           self.sigma,
            'gen':             self.gen,
            'best_fit':        self.best_fit,
            'stagnation':      self.stagnation,
            'restarts':        self.restart_count,
            'history':         self.history,
            'archive':         [(f, g.tolist()) for f, g in self.archive],
            'cma_dir':         self.cma_dir.tolist(),
            'best_per_regime': bpr_serial,
            # Island / regime metadata
            'island_to_regime':    ISLAND_TO_REGIME,
            'island_best_fits':    dict(self._island_best_fits),
            'n_islands':           N_ISLANDS,
        }
        path = os.path.join(AGENTS_DIR, "population_state.pkl")
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=4)

    # ── Кроссовер ─────────────────────────────────────────────────────────────
    def _block_cx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        child = p1.copy()
        for start, end in _BLOCK_BOUNDS:
            if self.rng.random() < 0.5:
                child[start:end] = p2[start:end]
        return child

    # ── DE мутант ─────────────────────────────────────────────────────────────
    def _de_mutant(self, island_fits: np.ndarray, island_pop: np.ndarray,
                   best: np.ndarray) -> np.ndarray:
        """DE мутант в пределах острова."""
        elite_k  = max(3, len(island_pop) // 5)
        elite_pool = np.argsort(island_fits)[::-1][:elite_k]
        r1, r2   = self.rng.choice(elite_pool, 2, replace=False)
        f_scale  = max(0.15, 1.0 - self.stagnation / (STAGNATION_GENS * 2))
        trial    = best + DE_F * f_scale * (island_pop[r1] - island_pop[r2])
        return trial.astype(np.float32)

    # ── CMA-inspired мутант ───────────────────────────────────────────────────
    def _cma_mutant(self, base: np.ndarray) -> np.ndarray:
        mask   = self.rng.random(GENOME_SIZE) < MUTATION_RATE
        noise  = self.rng.normal(0, self.sigma, GENOME_SIZE).astype(np.float32)
        cma_norm = self.cma_dir / (np.linalg.norm(self.cma_dir) + 1e-8)
        delta  = mask * (noise + CMA_BLEND * cma_norm * self.sigma * GENOME_SIZE**0.5)
        return (base + delta).astype(np.float32)

    # ── Локальная мутация ─────────────────────────────────────────────────────
    def _local_mutant(self, best: np.ndarray) -> np.ndarray:
        sigma_local = max(self.sigma * LOCAL_SIGMA_MULT, 5e-4)
        m = self.rng.random(GENOME_SIZE) < MUTATION_RATE
        return (best + m * self.rng.normal(0, sigma_local, GENOME_SIZE)).astype(np.float32)

    # ──  Per-layer мутация ────────────────────────────────────────────────
    def _mut_layerwise(self, g: np.ndarray) -> np.ndarray:
        """
        Мутация с разными sigma для каждого слоя:
        W1: σ×1.2 — входной слой, исследуем признаки активно
        W2: σ×0.8 — средний слой, точная настройка
        W3: σ×1.5 — выходной слой, смело пробуем новые решения
        """
        result = g.copy()
        for (start, end), mult in zip(_BLOCK_BOUNDS, LAYER_SIGMA_MULT):
            size  = end - start
            m     = self.rng.random(size) < MUTATION_RATE
            noise = self.rng.normal(0, self.sigma * mult, size).astype(np.float32)
            result[start:end] += m * noise
        return result

    # ── Стандартная мутация (fallback) ────────────────────────────────────────
    def _mut(self, g: np.ndarray) -> np.ndarray:
        m = self.rng.random(GENOME_SIZE) < MUTATION_RATE
        return g + m * self.rng.normal(0, self.sigma, GENOME_SIZE).astype(np.float32)

    # ── Lévy flight мутация — тяжёлые хвосты для побега из локального минимума ──
    def _levy_mutant(self, g: np.ndarray) -> np.ndarray:
        """
        Lévy-flight мутация: редкие крупные прыжки + обычные малые шаги.
        Распределение Lévy: P(x) ~ x^{-(1+α)}, α = LEVY_ALPHA (1 < α < 2).
        Аппроксимация через Chambers-Mallows-Stuck метод (точная генерация).

        Преимущества перед Gaussian:
          - Может перепрыгнуть широкие плато фитнеса
          - Охватывает далёкие регионы пространства геномов
          - 90%+ шагов всё ещё малые — не разрушает хорошие гены
        """
        α = LEVY_ALPHA
        # Chambers-Mallows-Stuck генерация α-стабильного распределения
        # (эффективная аппроксимация через нормальные + экспоненциальные r.v.)
        U = self.rng.uniform(-np.pi/2, np.pi/2, GENOME_SIZE)
        W = self.rng.exponential(1.0, GENOME_SIZE)
        Φ = 0.0  # симметричное распределение
        # Формула Chambers et al. для symmetric α-stable:
        levy_step = (np.sin(α * (U + Φ)) / np.cos(U) ** (1.0/α) *
                     (np.cos(U - α * (U + Φ)) / W) ** ((1.0 - α) / α))
        levy_step = np.clip(levy_step, -10.0, 10.0)  # обрезаем экстремальные значения

        # Маска мутации: Lévy прыжок только для части генов
        m = self.rng.random(GENOME_SIZE) < MUTATION_RATE
        delta = m * levy_step * LEVY_SCALE * self.sigma
        return (g + delta).astype(np.float32)

    # ── Cosine annealing sigma ─────────────────────────────────────────────────
    def _cosine_sigma(self) -> float:
        """
        Cosine Annealing with Warm Restarts для sigma (аналог SGDR).
        σ(t) = σ_min + 0.5*(σ_max - σ_min) * (1 + cos(π * t_cur/T_cur))

        DEPRECATED: используется как fallback если improvement-based sigma отключён.
        """
        if not COSINE_SIGMA_ENABLED:
            return max(SIGMA_MIN, self.sigma * SIGMA_DECAY)

        sigma_max = MUTATION_SIGMA
        t         = self.gen - 1
        T0        = COSINE_T0

        cur_T = T0
        cur_t = t
        while cur_t >= cur_T:
            cur_t -= cur_T
            cur_T  = max(2, int(cur_T * COSINE_TMULT))

        cos_val = np.cos(np.pi * cur_t / max(1, cur_T))
        sigma   = SIGMA_MIN + 0.5 * (sigma_max - SIGMA_MIN) * (1.0 + cos_val)
        return float(sigma)

    def _improvement_sigma(self) -> float:
        """
        Improvement-based sigma management (замена cosine annealing).

        Логика:
        - При улучшении: sigma снижается (эксплуатируем найденный регион)
        - При стагнации: sigma ПЛАВНО растёт (с ограниченным upper bound)
        - Upper bound = min(MUTATION_SIGMA, 0.5 × текущая sigma + step)
          Это предотвращает деструктивные burst-ы (escape_sigma_mult=4.5 убивал решения)

        Формула:
          improved:    sigma *= 0.80  (но не ниже SIGMA_MIN)
          stagnation:  sigma += (MUTATION_SIGMA - sigma) × 0.08  (мягкий рост)
          hard cap:    sigma ≤ MUTATION_SIGMA × 0.60  (никогда не выше 60% начальной)

        Returns: новое значение sigma
        """
        stag = self.stagnation
        s = self.sigma

        if stag == 0:
            # Недавнее улучшение — мягкое снижение
            s = max(SIGMA_MIN, s * 0.80)
        else:
            # Стагнация — плавный рост к умеренному потолку
            # Чем дольше стагнация, тем больше шаг (но с soft cap)
            growth_rate = 0.08 * min(stag / STAGNATION_GENS, 2.0)
            ceiling = MUTATION_SIGMA * 0.60  # никогда не выше 60% начальной sigma
            s = s + (ceiling - s) * growth_rate
            s = min(s, ceiling)

        return float(max(SIGMA_MIN, s))

    def _sel(self, fits: np.ndarray, candidates: Optional[np.ndarray] = None) -> int:
        if candidates is not None and len(candidates) > 0:
            n = min(self.tournament_k, len(candidates))
            c = self.rng.choice(candidates, n, replace=False)
        else:
            c = self.rng.integers(0, POP_SIZE, self.tournament_k)
        return int(c[np.argmax(fits[c])])

    # ── Обновление архива ─────────────────────────────────────────────────────
    def _update_archive(self, fits: np.ndarray):
        order = np.argsort(fits)[::-1]
        for idx in order[:ARCHIVE_SIZE]:
            f = float(fits[idx])
            g = self.pop[idx].copy()
            self.archive.append((f, g))
            self.archive.sort(key=lambda x: -x[0])
            self.archive = self.archive[:ARCHIVE_SIZE]

    def _archive_best_fitness(self, current_fits: np.ndarray) -> float:
        """
        Возвращает максимальный fitness в архиве, скорректированный на текущий масштаб.

        ПРОБЛЕМА (gen62→63): архив хранит геном gen62 с fitness=+30.25,
        полученным при dyn_rw={bullish×0.5, bearish×0.8}. В gen63 веса меняются
        (bullish×0.4, bearish×0.9) → тот же геном стоит ~+20, НО в архиве написано +30.25.
        Каждое поколение arch_best_f > cur_best_f → архивный геном подставляется
        в nxt[0] → нет давления на эволюцию → стагнация 50+ генов.

        Решение: если текущая популяция улучшилась с момента последнего «рекорда»,
        приводим archive[0] к реалистичному fitness через decay.
        Decay rate: чем дольше геном в архиве без нового рекорда, тем сильнее сдувается.
        """
        if not self.archive:
            return -np.inf
        # Стагнация в поколениях с момента последнего NEW BEST в архиве
        stag = getattr(self, '_archive_stag', 0)
        arch_f = self.archive[0][0]
        if stag == 0:
            return arch_f
        # Decay: каждые 5 генов стагнации → -5% от текущего best для архивного геном
        # Это плавно «уступает» архивный геном потомкам без полного удаления
        cur_best = float(current_fits[np.argmax(current_fits)]) if len(current_fits) else -np.inf
        decay_factor = max(0.0, 1.0 - stag * 0.015)   # 1.5% per stagnation gen
        effective_arch_f = arch_f * decay_factor
        # Никогда не давать архивному геному преимущество больше чем ARCHIVE_ADVANTAGE_MAX
        ARCHIVE_ADVANTAGE_MAX = _gscfg_f(_GS, 'archive_advantage_max', 3.0)
        if effective_arch_f > cur_best + ARCHIVE_ADVANTAGE_MAX:
            effective_arch_f = cur_best + ARCHIVE_ADVANTAGE_MAX
        return effective_arch_f

    # ── Обновление CMA direction memory ──────────────────────────────────────
    def _update_cma(self, new_best: np.ndarray, old_best: np.ndarray):
        delta = new_best - old_best
        norm  = np.linalg.norm(delta)
        if norm > 1e-8:
            self.cma_dir = ((1 - CMA_MEMORY_LR) * self.cma_dir
                            + CMA_MEMORY_LR * delta / norm * self.sigma)

    # ──  NES Gradient Step ───────────────────────────────────────────────
    def _nes_step(self, fits: np.ndarray) -> Optional[np.ndarray]:
        """
        NES (Natural Evolution Strategy) шаг.
        Оценивает «градиент» по пространству геномов, используя текущую
        популяцию как совокупность возмущений вокруг лучшего индивида.

        Метод:
        1. Берём топ NES_FRAC особей как NES-выборку
        2. Ранжируем fitness (ранги более устойчивы чем абсолютные значения)
        3. Вычисляем fitness-weighted mean delta от лучшего
        4. Обновляем центр в направлении «градиента»

        Returns: новый центральный геном или None если недостаточно данных
        """
        if self.best_g is None:
            return None

        n_nes = max(NES_MIN_SAMPLES, int(POP_SIZE * NES_FRAC))
        top_idx = np.argsort(fits)[::-1][:n_nes]

        if len(top_idx) < NES_MIN_SAMPLES:
            return None

        # Rank-based weights (устойчивее абсолютных fitness)
        ranks = np.argsort(np.argsort(fits[top_idx])).astype(np.float64)
        # Нормализуем в [-0.5, 0.5] как в CMA-ES
        w = (ranks / (len(ranks) - 1 + 1e-8)) - 0.5
        w = w / (np.abs(w).sum() + 1e-8)

        # Отклонения от лучшего
        deltas = self.pop[top_idx].astype(np.float64) - self.best_g.astype(np.float64)

        # NES gradient (fitness-weighted perturbations)
        grad = (w[:, None] * deltas).sum(axis=0).astype(np.float32)

        # Gradient magnitude для мониторинга
        grad_mag = float(np.linalg.norm(grad))
        self.nes_grad_mags.append(grad_mag)

        # ── Clip gradient magnitude (предотвращает взрывной NES шаг) ─────────
        # При sigma=0.25 и 14345 параметрах NES gradient может достигать 3-5,
        # что даёт step = lr × grad = 0.08 × 3.0 × 14345 = огромный сдвиг.
        # Нормируем: если ||grad|| > 1.0 → делим на ||grad|| (unit grad) × 1.0.
        MAX_GRAD_MAG = 1.0
        if grad_mag > MAX_GRAD_MAG:
            grad = grad * (MAX_GRAD_MAG / (grad_mag + 1e-9))

        # ── NES авто-буст при gradient collapse ─────────────────────────────
        # В логе G005–G034: nes < 0.18 на протяжении 30 поколений → обучение стоит.
        # Если 3 подряд поколения nes_grad < 0.30: популяция сколлапсировала,
        # делtas ≈ 0 → буст lr×2.5 чтобы NES сделал реальный шаг.
        _NES_LOW_THRESH = 0.30
        _NES_BOOST_GENS = 3
        _nes_lr_eff = NES_LR
        if (len(self.nes_grad_mags) >= _NES_BOOST_GENS and
                all(m < _NES_LOW_THRESH
                    for m in self.nes_grad_mags[-_NES_BOOST_GENS:])):
            _nes_lr_eff = NES_LR * 2.5
            if not getattr(self, '_nes_boost_logged', False):
                print(f"  [NES-boost] grad collapse × {_NES_BOOST_GENS}gens "
                      f"< {_NES_LOW_THRESH} → nes_lr×2.5 "
                      f"({NES_LR:.3f}→{_nes_lr_eff:.3f})")
                self._nes_boost_logged = True
        else:
            self._nes_boost_logged = False

        # Обновляем центр
        new_center = self.best_g + _nes_lr_eff * grad
        return new_center.astype(np.float32)

    # ──  Sigma restart (improvement-based, без деструктивных burst-ов) ────
    def _check_restart(self, improved: bool):
        if improved:
            self.stagnation           = 0
            self.exploit_mode         = True
            self.tournament_k         = TOURNAMENT_K
            self.restarts_no_improve  = 0
            self.deep_stagnation_count = 0   # сброс при улучшении
            # ── Снижение sigma при новом лучшем (мягче: ×0.80 вместо ×0.65) ───
            if EXPLOIT_SIGMA_ON_BEST and self.sigma > EXPLOIT_SIGMA_ON_BEST_FLOOR:
                new_s = max(EXPLOIT_SIGMA_ON_BEST_FLOOR,
                            self.sigma * EXPLOIT_SIGMA_ON_BEST_MULT)
                print(f"  [EXPLOIT-BEST] sigma: {self.sigma:.4f} → {new_s:.4f}  "
                      f"(new best — exploiting neighbourhood)")
                self.sigma = new_s
                self._after_new_best = True
            else:
                self._after_new_best = True
            return

        self.stagnation += 1
        if self.stagnation < STAGNATION_GENS:
            return

        self.stagnation    = 0
        self.restart_count += 1
        self.restarts_no_improve += 1

        # ── Improvement-based: вместо exploit/escape чередования → плавный рост ─
        # Старый escape: sigma×4.5 → разрушал решение.
        # Новый: _improvement_sigma() уже плавно поднимает sigma при стагнации.
        # Здесь только фиксируем restart-event и чуть подталкиваем sigma вверх.
        sigma_bump = min(MUTATION_SIGMA * 0.60,
                         self.sigma * 1.5)  # мягкий подъём, не деструктивный
        old_s = self.sigma
        self.sigma = max(self.sigma, sigma_bump)
        self._after_new_best = False
        print(f"  [RESTART #{self.restart_count}] sigma: {old_s:.4f}→{self.sigma:.4f}  "
              f"rni={self.restarts_no_improve}  "
              f"(improvement-based, cap={MUTATION_SIGMA * 0.60:.3f})")

        self.tournament_k = min(TOURNAMENT_K_MAX,
                                self.tournament_k + TOURNAMENT_PRESSURE_STEP)

    def _check_restart_phase2(self, improved: bool):
        """Phase 2 (стабилизация): только micro-exploit, без escape-burst.
        Sigma в Phase2 принудительно держится низкой (PHASE2_SIGMA) при eval,
        поэтому здесь управляем только stagnation-счётчиком и tournament_k.
        """
        if improved:
            self.stagnation      = 0
            self.exploit_mode    = True
            self.tournament_k    = TOURNAMENT_K
            self.restarts_no_improve = 0
            self._after_new_best = True
            return

        self.stagnation += 1
        if self.stagnation < STAGNATION_GENS:
            return

        self.stagnation    = 0
        self.restart_count += 1
        # В Phase 2 не делаем escape — только слегка поднимаем tournament_k
        # чтобы усилить селективное давление и найти лучших соседей
        self.tournament_k = min(TOURNAMENT_K_MAX,
                                self.tournament_k + TOURNAMENT_PRESSURE_STEP)
        print(f"  [Phase2-stag #{self.restart_count}] tk→{self.tournament_k}  "
              f"(no sigma burst in stabilization phase)")

    # ── Инъекция разнообразия ─────────────────────────────────────────────────
    def _inject_diversity(self, fits: np.ndarray, force_heavy: bool = False):
        top_k   = max(5, POP_SIZE // 10)
        top_idx = np.argsort(fits)[::-1][:top_k]
        top_pop = self.pop[top_idx]
        norms   = np.linalg.norm(top_pop, axis=1, keepdims=True) + 1e-8
        normed  = top_pop / norms
        sim_matrix = normed @ normed.T
        np.fill_diagonal(sim_matrix, 0.0)
        avg_sim = sim_matrix.sum() / max(1, top_k * (top_k - 1))
        collapsed = force_heavy or (float(avg_sim) > (1.0 - DIVERSITY_THRESHOLD))

        n_replace = max(1, int(POP_SIZE * DIVERSITY_FRAC))
        worst_idx = np.argsort(fits)[:n_replace]

        if collapsed:
            # ── ТЯЖЁЛАЯ ИНЪЕКЦИЯ (полный коллапс или NES-коллапс) ──────────────
            # Увеличенная доля: DIVERSITY_FRAC * 1.5 (но не более ~55% популяции)
            extra = max(1, int(POP_SIZE * DIVERSITY_FRAC * 0.5))
            total_replace = min(int(POP_SIZE * 0.55), n_replace + extra)
            elite_slots = set(np.argsort(fits)[::-1][:ELITE_SIZE].tolist())
            all_worst   = np.argsort(fits)[:total_replace].tolist()
            safe_worst  = [i for i in all_worst if i not in elite_slots]

            if safe_worst:
                # ⓪ Направленное разнообразие: кроссовер лучших по разным режимам
                # Создаёт потомков, наследующих навыки из bearish И bullish одновременно
                n_directed = 0
                if DIRECTED_DIVERSITY_ENABLED and self.best_per_regime:
                    n_directed = max(1, int(len(safe_worst) * DIRECTED_DIVERSITY_FRAC))
                    regime_genomes = [(r, g.copy()) for r, (ret, g) in
                                      self.best_per_regime.items()]
                    if len(regime_genomes) >= 2:
                        for di in range(min(n_directed, len(safe_worst))):
                            # Берём два разных режимных генома и скрещиваем
                            r1_idx = di % len(regime_genomes)
                            r2_idx = (di + 1) % len(regime_genomes)
                            g1 = regime_genomes[r1_idx][1]
                            g2 = regime_genomes[r2_idx][1]
                            child = self._block_cx(g1, g2)
                            # Малый шум — сохраняем знания обоих режимов
                            noise_s = self.sigma * 0.8
                            child = np.clip(
                                child + self.rng.normal(0, noise_s, GENOME_SIZE),
                                -5.0, 5.0).astype(np.float32)
                            self.pop[safe_worst[di]] = child
                    else:
                        n_directed = 0  # недостаточно режимных геномов

                # ① Xavier-случайные геномы для ~60% оставшихся
                remaining_slots = safe_worst[n_directed:]
                n_xavier = max(1, int(len(remaining_slots) * 0.60))
                if n_xavier > 0 and len(remaining_slots) > 0:
                    self.pop[np.array(remaining_slots[:n_xavier])] = _xavier_pop(self.rng, n_xavier)

                # ② Кроссовер архивных геномов между собой для ~25% оставшихся
                # Даёт геномы, семантически похожие на лучших, но структурно разные
                arc_slots = remaining_slots[n_xavier: n_xavier + max(1, int(len(remaining_slots) * 0.25))]
                if self.archive and len(arc_slots) > 0:
                    arc_genomes = [g for _, g in self.archive]
                    for k, slot in enumerate(arc_slots):
                        a1 = arc_genomes[k % len(arc_genomes)]
                        a2 = arc_genomes[(k + 1) % len(arc_genomes)]
                        # Блочный кроссовер + крупный шум
                        child = self._block_cx(a1, a2)
                        noise_s = max(MUTATION_SIGMA * 1.2, self.sigma * 3.0)
                        self.pop[slot] = np.clip(
                            child + self.rng.normal(0, noise_s, GENOME_SIZE),
                            -5.0, 5.0).astype(np.float32)

                # ③ Послойная инъекция: переинициализируем один случайный слой
                #    у ~15% заменяемых (сохраняет другие слои, трясёт только один)
                if LAYER_INJECTION_ENABLED:
                    layer_slots = remaining_slots[n_xavier + len(arc_slots):]
                    if layer_slots:
                        for slot in layer_slots:
                            base = self.best_g.copy() if self.best_g is not None \
                                   else self.pop[top_idx[0]].copy()
                            layer_idx = self.rng.integers(0, len(_BLOCK_BOUNDS))
                            start, end = _BLOCK_BOUNDS[layer_idx]
                            # Переинициализируем слой Xavier + мутируем остальное малым шумом
                            fi_fo_list = [(N_INPUT, N_HIDDEN1), (N_HIDDEN1, N_HIDDEN2),
                                          (N_HIDDEN2, N_HIDDEN3), (N_HIDDEN3, N_ACTIONS)]
                            fi, fo = fi_fo_list[layer_idx]
                            limit = np.sqrt(6.0 / (fi + fo))
                            base[start:end] = self.rng.uniform(-limit, limit, end - start).astype(np.float32)
                            noise_rest = self.sigma * 1.5
                            mask = np.ones(GENOME_SIZE, dtype=bool)
                            mask[start:end] = False
                            base[mask] += self.rng.normal(0, noise_rest, mask.sum()).astype(np.float32)
                            self.pop[slot] = np.clip(base, -5.0, 5.0).astype(np.float32)

            # Sigma bump: умеренный (без деструктивного burst)
            # Старый burst ×3.0 разрушал решения. Теперь: мягкий подъём до 60% начальной.
            old_sigma   = self.sigma
            _ceiling    = MUTATION_SIGMA * 0.60  # cap = 60% начальной sigma
            bump_target = _ceiling if force_heavy else min(_ceiling, self.sigma * 1.8)
            self.sigma  = max(self.sigma, bump_target)
            self._after_new_best = False
            reason = "NES-collapse" if force_heavy else f"sim={avg_sim:.3f}"
            _n_layer = len(layer_slots) if LAYER_INJECTION_ENABLED else 0
            print(f"  [DIVERSITY HEAVY] {reason}  заменено={len(safe_worst)}  "
                  f"sigma: {old_sigma:.4f}→{self.sigma:.4f}  "
                  f"(directed={n_directed} xavier={n_xavier} "
                  f"arc_cx={len(arc_slots)} layer_inj={_n_layer})")
        else:
            # ── МЯГКАЯ ИНЪЕКЦИЯ: замещаем худших архивными геномами с шумом ──────
            # Добавляем часть Xavier-случайных для поддержания генетического разнообразия
            n_archive = max(1, int(n_replace * 0.65))
            n_random  = n_replace - n_archive
            if self.archive and len(self.archive) >= 2:
                for k, idx in enumerate(worst_idx[:n_archive]):
                    arc_idx = k % len(self.archive)
                    _, arc_g = self.archive[arc_idx]
                    noise_scale = self.sigma * 1.8
                    self.pop[idx] = np.clip(
                        arc_g + self.rng.normal(0, noise_scale, GENOME_SIZE),
                        -5.0, 5.0).astype(np.float32)
            else:
                self.pop[worst_idx[:n_archive]] = _xavier_pop(self.rng, n_archive)
            # Случайная доля для увеличения генетического разнообразия
            if n_random > 0:
                self.pop[worst_idx[n_archive:]] = _xavier_pop(self.rng, n_random)
            print(f"  [DIVERSITY] sim={avg_sim:.3f}  заменено={n_replace}  "
                  f"(arch={n_archive} rand={n_random})")

    # ──  Island Migration ─────────────────────────────────────────────────
    def _island_migrate(self, fits: np.ndarray):
        """
        Island migration v2:
        - Лучший остров (по max fitness) отправляет мигрантов на ВСЕ острова
        - Остальные острова обмениваются только с соседями
        - Extra slot: принимают топ-1 из глобального архива

        Это ускоряет распространение хороших решений и поддерживает разнообразие.
        """
        n_islands = len(self.islands)
        if n_islands < 2:
            return

        # Лучший остров по max fitness
        island_max_fits = []
        for island_idx in self.islands:
            island_fits = fits[island_idx]
            island_max_fits.append(float(fits[island_idx[int(np.argmax(island_fits))]]))
        best_island_id = int(np.argmax(island_max_fits))

        emigrants = {}
        for i, island_idx in enumerate(self.islands):
            island_fits = fits[island_idx]
            top_local   = np.argsort(island_fits)[::-1][:MIGRATION_SIZE]
            emigrants[i] = [(float(island_fits[k]),
                             self.pop[island_idx[k]].copy())
                            for k in top_local]

        migrated_pairs = []
        for i, island_idx in enumerate(self.islands):
            island_fits = fits[island_idx]
            worst_local = np.argsort(island_fits)[:MIGRATION_SIZE]

            # Лучший остров → все острова (глобальное распространение)
            if best_island_id != i and island_max_fits[best_island_id] > island_max_fits[i]:
                src = best_island_id
            else:
                # Иначе — ближайший сосед
                candidates = [j for j in range(n_islands) if j != i]
                adj = [j for j in candidates if abs(j - i) == 1]
                src = int(self.rng.choice(adj if adj else candidates))

            for k, (f_e, g_e) in enumerate(emigrants[src]):
                if k < len(worst_local):
                    slot = island_idx[worst_local[k]]
                    self.pop[slot] = g_e.copy()

            r_dst = ISLAND_TO_REGIME.get(i,   str(i))
            r_src = ISLAND_TO_REGIME.get(src, str(src))
            migrated_pairs.append(f"{r_src}→{r_dst}")

        # Архивный геном → случайный худший в худшем острове (одна инъекция)
        if self.archive:
            worst_island_id = int(np.argmin(island_max_fits))
            worst_iidx      = self.islands[worst_island_id]
            wfits           = fits[worst_iidx]
            worst_slot      = worst_iidx[int(np.argmin(wfits))]
            _, arc_g        = self.archive[0]
            noise           = self.rng.normal(0, self.sigma * 0.5, GENOME_SIZE).astype(np.float32)
            self.pop[worst_slot] = (arc_g + noise).astype(np.float32)
            migrated_pairs.append(f"archive→{ISLAND_TO_REGIME.get(worst_island_id,'?')}")

        # Измеряем разнообразие между островами
        island_bests = []
        for island_idx in self.islands:
            bi = island_idx[int(np.argmax(fits[island_idx]))]
            island_bests.append(self.pop[bi])
        if len(island_bests) >= 2:
            dists = []
            for a in range(len(island_bests)):
                for b in range(a+1, len(island_bests)):
                    d = float(np.linalg.norm(island_bests[a] - island_bests[b]))
                    dists.append(d)
            div = float(np.mean(dists))
            self.island_diversity_hist.append(div)
            fits_str = '  '.join(f"{ISLAND_TO_REGIME.get(ii,'?')}:{island_max_fits[ii]:+.2f}"
                                  for ii in range(n_islands))
            print(f"  [MIGRATION] diversity={div:.3f}  island_fits: {fits_str}  "
                  f"routes: {', '.join(migrated_pairs)}")

    # ── Режимно-балансированная инъекция (Phase 2 only) ──────────────────────
    def _regime_balance_inject(self, fits: np.ndarray):
        """
        Phase 2: каждый остров получает инъекцию лучшего генома для своих
        «слабых» режимов. Это предотвращает скатывание в локальный минимум
        одного режима (например, агент учится только на bull, игнорируя crash).

        Логика:
          1. Для каждого острова считаем avg_ret по его профильным режимам
          2. Слабейшие N слотов острова заменяем мутантами от best_per_regime
             соответствующего слабого режима
          3. Мутация небольшая (σ×0.3) — это инъекция направления, не перезапуск
        """
        if not self.best_per_regime:
            return

        N_INJECT = max(2, MIGRATION_SIZE)   # слотов на остров
        injected_info = []

        for island_id, island_idx in enumerate(self.islands):
            if not island_idx:
                continue
            island_fits = fits[np.array(island_idx)]

            # Профильные режимы острова (из ISLAND_REGIME_MAP или fallback)
            island_regime = ISLAND_TO_REGIME.get(island_id, '')
            # Все режимы, ассоциированные с этим островом (из Phase 1 конфига)
            island_regimes_set = set(ISLAND_REGIME_MAP.get(island_id, [island_regime]))

            # Режимы вне профиля острова — «чужие», агент в них обычно слабее
            off_profile = [r for r in self.best_per_regime
                           if r not in island_regimes_set]
            if not off_profile:
                off_profile = list(self.best_per_regime.keys())

            # Сортируем по возврату — слабейшие режимы первыми
            off_profile.sort(key=lambda r: self.best_per_regime[r][0])

            # Берём худшие N_INJECT слотов острова для замены
            worst_slots = [island_idx[i] for i in np.argsort(island_fits)[:N_INJECT]]

            for slot_i, (slot, regime) in enumerate(
                    zip(worst_slots, off_profile[:len(worst_slots)])):
                _, donor_g = self.best_per_regime[regime]
                noise = self.rng.normal(0, self.sigma * 0.3,
                                        GENOME_SIZE).astype(np.float32)
                self.pop[slot] = np.clip(donor_g + noise, -5.0, 5.0)

            if worst_slots and off_profile:
                injected_info.append(
                    f"island{island_id}←{'+'.join(off_profile[:N_INJECT])}")

        if injected_info:
            print(f"  [RegimeBalance] {', '.join(injected_info)}")

    # ──  Формирование нового поколения с island structure ─────────────────
    def _next(self, fits: np.ndarray) -> np.ndarray:

        order      = np.argsort(fits)[::-1]
        arch_best_g = self.archive[0][1] if self.archive else None
        # ── Используем скорректированный fitness архива (FIX 2: stale archive) ──
        # Вместо self.archive[0][0] используем _archive_best_fitness(), которая
        # снижает «ценность» архивного генома при долгой стагнации.
        arch_best_f  = self._archive_best_fitness(fits) if self.archive else -np.inf
        arch_raw_f   = self.archive[0][0] if self.archive else -np.inf
        cur_best_f  = float(fits[order[0]])
        best_g = arch_best_g if arch_best_f > cur_best_f else self.pop[order[0]]

        # Обновляем счётчик стагнации архива
        if cur_best_f >= arch_raw_f:
            self._archive_stag = 0   # популяция догнала архив
        else:
            self._archive_stag = getattr(self, '_archive_stag', 0) + 1

        nxt = np.zeros_like(self.pop)

        # ── ЭЛИТА: глобальный топ ────────────────────────────────────────────
        for i in range(ELITE_SIZE):
            nxt[i] = self.pop[order[i]].copy()
        if arch_best_f > cur_best_f:
            nxt[0] = arch_best_g.copy()

        # NES: улучшенный центр как одна из элитных особей
        nes_center = self._nes_step(fits)
        if nes_center is not None:
            nxt[ELITE_SIZE - 1] = nes_center

        # ── РЕЖИМНАЯ ЭЛИТА ────────────────────────────────────────────────────
        # Добавляем N_REGIMES слотов под лучшие геномы каждого режима + мутантов.
        # Гарантирует сохранение прогресса по слабым режимам всегда, не только в Phase2.
        phase2_regime_slots = 0
        slot = ELITE_SIZE
        if self.best_per_regime:
            _regimes_sorted = sorted(
                self.best_per_regime.items(),
                key=lambda kv: kv[1][0]   # по среднему возврату — слабые первыми
            )
            n_regime_elite = min(len(REGIME_WEIGHTS), len(_regimes_sorted))
            # Слоты: 1 точный + 2 мутанта + 2 межрежимных кроссовера = 5 на режим
            # Межрежимные кроссоверы — ключ к мульти-режимной способности:
            # bearish_best × bullish_best → потомок, наследующий навыки обоих
            _slots_per_regime = 5
            _all_regime_genomes = [(regime, rg) for regime, (ret, rg)
                                   in _regimes_sorted]
            for r_idx, (regime, (ret, rg)) in enumerate(_regimes_sorted):
                if slot + _slots_per_regime > POP_SIZE:
                    break
                # Точная копия лучшего генома этого режима
                nxt[slot] = rg.copy()
                slot += 1
                phase2_regime_slots += 1
                # Мутант 1 — малое возмущение (эксплуатация)
                noise1 = self.rng.normal(0, self.sigma * 0.5, GENOME_SIZE).astype(np.float32)
                nxt[slot] = np.clip(rg + noise1, -5.0, 5.0)
                slot += 1
                phase2_regime_slots += 1
                # Мутант 2 — кроссовер с глобальным лучшим (смешение знаний)
                nxt[slot] = self._block_cx(rg, best_g)
                slot += 1
                phase2_regime_slots += 1
                # Мутант 3+4 — кроссовер с ДРУГИМИ режимами (мульти-режимный навык)
                # Это заменяет regime-specific heads: вместо раздельных голов,
                # кроссовер лучших из разных режимов создаёт потомков с навыками обоих
                for cr_idx in range(2):
                    if slot >= POP_SIZE:
                        break
                    other_idx = (r_idx + cr_idx + 1) % len(_all_regime_genomes)
                    _, other_rg = _all_regime_genomes[other_idx]
                    child = self._block_cx(rg, other_rg)
                    # Малый шум для исследования вокруг кроссовера
                    noise_cr = self.rng.normal(0, self.sigma * 0.3, GENOME_SIZE).astype(np.float32)
                    nxt[slot] = np.clip(child + noise_cr, -5.0, 5.0)
                    slot += 1
                    phase2_regime_slots += 1

        # Каждый остров формирует своё потомство
        # BUG-FIX #2: slot НЕ сбрасывается к ELITE_SIZE.
        # Иначе режимная элита (nxt[ELITE_SIZE..slot-1]) затирается первыми
        # local_mutant-ами островного цикла. Продолжаем с текущей позиции.
        island_size = POP_SIZE // len(self.islands)
        # slot уже указывает за режимную элиту — НЕ делаем slot = ELITE_SIZE

        for island_idx in self.islands:
            if slot >= POP_SIZE:
                break

            island_fits = fits[island_idx]
            island_pop  = self.pop[np.array(island_idx)]
            island_best_local = island_idx[int(np.argmax(island_fits))]
            island_best_g     = self.pop[island_best_local]

            # Доли offspring для острова
            # FIX: используем реально оставшееся место, а не island_size - ELITE_SIZE//N
            # Иначе при большой режимной элите (до 9 слотов) первый остров занимает
            # больше места чем доступно → slot выходит за POP_SIZE раньше времени.
            remaining = POP_SIZE - slot
            if remaining <= 0:
                break
            n_island_slots = min(island_size - (ELITE_SIZE // len(self.islands)), remaining)
            n_local = max(1, int(n_island_slots * LOCAL_FRAC))
            n_de    = max(1, int(n_island_slots * DE_FRACTION))
            n_cma   = max(1, int(n_island_slots * 0.15))
            # Lévy мутанты: заменяют часть стандартных потомков
            n_levy  = max(1, int(n_island_slots * LEVY_FRAC)) if LEVY_ENABLED else 0

            # Используем глобального лучшего для local/CMA, island-лучшего для DE
            for _ in range(n_local):
                if slot >= POP_SIZE: break
                nxt[slot] = self._local_mutant(best_g)
                slot += 1

            for _ in range(n_de):
                if slot >= POP_SIZE: break
                nxt[slot] = self._de_mutant(island_fits, island_pop, island_best_g)
                slot += 1

            for _ in range(n_cma):
                if slot >= POP_SIZE: break
                nxt[slot] = self._cma_mutant(best_g)
                slot += 1

            # Lévy-flight: большие прыжки от island-лучшего
            for _ in range(n_levy):
                if slot >= POP_SIZE: break
                nxt[slot] = self._levy_mutant(island_best_g)
                slot += 1

        # Остальные: блочный кроссовер + per-layer мутация
        levy_remaining = max(0, int((POP_SIZE - slot) * LEVY_FRAC)) if LEVY_ENABLED else 0
        levy_slots = set(self.rng.choice(range(slot, POP_SIZE),
                                         min(levy_remaining, POP_SIZE - slot),
                                         replace=False).tolist())

        while slot < POP_SIZE:
            p1_idx = self._sel(fits)
            if self.rng.random() < CROSSOVER_RATE:
                p2_idx = self._sel(fits)
                child  = self._block_cx(self.pop[p1_idx], self.pop[p2_idx])
            else:
                child  = self.pop[p1_idx].copy()

            if slot in levy_slots:
                nxt[slot] = self._levy_mutant(child)
            elif self.rng.random() < 0.70:
                nxt[slot] = self._mut_layerwise(child)
            else:
                nxt[slot] = self._mut(child)
            slot += 1

        # ── Sigma update: improvement-based (привязано к реальному прогрессу) ───
        # Старый cosine annealing двигал sigma по таймеру, не учитывая улучшения.
        # Новый _improvement_sigma:
        #   - снижает при улучшении (эксплуатация)
        #   - плавно растёт при стагнации (исследование)
        #   - hard cap = 60% от начальной sigma (нет деструктивных burst-ов)
        if self._after_new_best:
            self.sigma = max(SIGMA_MIN, self.sigma * 0.80)
            if self.sigma <= SIGMA_MIN * 1.15:
                self._after_new_best = False
        else:
            self.sigma = self._improvement_sigma()
        return nxt

    # ── Главный цикл ──────────────────────────────────────────────────────────
    def run(self):
        _results_dir = os.path.join(os.path.dirname(AGENTS_DIR), "..", "results")
        os.makedirs(_results_dir, exist_ok=True)
        _run_tag  = time.strftime("%Y%m%d_%H%M%S")
        _log_path = os.path.join(_results_dir, f"genetics_run_{_run_tag}.log")
        _log_f    = open(_log_path, "w", encoding="utf-8", buffering=1)

        def _glog(msg, *, also_print=True):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            _log_f.write(line + "\n")
            if also_print:
                print(line)

        header = (f"Pop={POP_SIZE}  Gens={N_GENERATIONS}  "
                  f"Genome={GENOME_SIZE}  Periods={len(self.precomp)}  "
                  f"Islands={N_ISLANDS}  snap_every={_get_snap_every()}bars")
        print("\n" + "="*70)
        print(f"  {header}")
        print("="*70)
        _glog(f"START  {header}", also_print=False)

        snap_every = _get_snap_every()
        _glog(f"  Daily snapshot: every {snap_every} bars", also_print=True)

        # Phase 1 информация
        _phase1_gens = int(N_GENERATIONS * PHASE1_GENS_FRAC) if PHASE1_ENABLED else 0
        _phase2_gens = int(N_GENERATIONS * FITNESS_PHASE2_FRAC)   # сессионно-относительный
        # ── Phase 2 (Stabilization) / Phase 3 (Dynamic) ─────────────────────
        _phase2_end  = int(N_GENERATIONS * PHASE2_GENS_FRAC) if PHASE2_ENABLED else 0
        # Phase 3 начинается сразу после Phase 2 (или с gen 0 если Phase2 off)
        if PHASE1_ENABLED:
            print(f"  Phase 1: ген 1..{_phase1_gens} ({PHASE1_GENS_FRAC*100:.0f}%) — "
                  f"острова обучаются на ПОЛНЫХ данных с островными весами периодов")
            print(f"  Phase 1: нет обрыва фитнеса при переходе (полный датасет с gen 1)")
            print(f"  Phase 2: ген {_phase1_gens+1}..{N_GENERATIONS} — "
                  f"нормальная миграция (каждые {MIGRATION_INTERVAL} ген)")
        else:
            print(f"  Phase 1: ОТКЛЮЧЕНА — нормальная миграция с gen 1")
        if PHASE2_ENABLED:
            print(f"  Phase 2 (stabilization): ген 1..{_phase2_end} "
                  f"({PHASE2_GENS_FRAC*100:.0f}%) — micro-mutations σ={PHASE2_SIGMA:.3f}, "
                  f"dynamic_rw=OFF, diversity freeze")
            print(f"  Phase 3 (dynamic):       ген {_phase2_end+1}..{N_GENERATIONS} "
                  f"— σ restart={PHASE3_SIGMA_RESTART:.3f}, dynamic_rw=ON")
        else:
            print(f"  Phase 2 (stabilization): ОТКЛЮЧЕНА — dynamic_rw с gen 1")
        print(f"  Currency learning: {'ON (сеть учится выбирать монеты)' if CURRENCY_LEARNING_ENABLED else 'OFF (жёсткая маска)'}")

        # _gen_offset: только для нумерации генов в логах и на графике
        # ВСЕ ФАЗОВЫЕ ПРОВЕРКИ используют g (0-индекс, относительный к сессии)
        _gen_offset = len(getattr(self, '_loaded_history', []))

        try:
            for g in range(N_GENERATIONS):
                self.gen = g + 1 + _gen_offset
                t0 = time.time()
                # ── ФАЗЫ: только сессионно-относительный индекс g (0-индекс) ────
                # _gen_offset НЕ влияет на фазы — только на нумерацию генов
                _in_phase1_now = PHASE1_ENABLED and (g < _phase1_gens)
                # Phase 2 (stabilization): micro-мутации, dynamic_rw выключен
                _in_phase2_now = PHASE2_ENABLED and (not _in_phase1_now) and (g < _phase2_end)
                # Phase 3: нормальный режим после стабилизации
                _in_phase3_now = not _in_phase1_now and not _in_phase2_now

                # ── При входе в Phase 3: перезапускаем sigma умеренно ───────────
                if _in_phase3_now and g == _phase2_end and PHASE2_ENABLED:
                    _old_s = self.sigma
                    self.sigma = PHASE3_SIGMA_RESTART
                    self._after_new_best = False
                    print(f"  [Phase2 → Phase3] sigma: {_old_s:.4f} → {self.sigma:.4f}  "
                          f"(exploration restart, dynamic_rw=ON)")

                # Однофазный фитнес — нет переключения (устраняет Phase1-клифф).
                # _set_train_phase вызывается для совместимости с CPU-воркерами
                # (они используют _get_train_phase()), но значение всегда 1.
                _fitness_phase = 1
                _set_train_phase(_fitness_phase)

                # ── Динамические режимные веса (только Phase3, после стабилизации) ─
                _dyn_rw: Optional[Dict[str, float]] = None
                if _in_phase3_now:
                    _dyn_rw = self._compute_dynamic_regime_weights()

                # ── В Phase 2: применяем micro-sigma ─────────────────────────────
                if _in_phase2_now:
                    _saved_sigma = self.sigma
                    self.sigma = PHASE2_SIGMA
                else:
                    _saved_sigma = None

                print(f"\n  GEN {self.gen}/{N_GENERATIONS + _gen_offset}  "
                      f"sigma={self.sigma:.4f}  "
                      f"stag={self.stagnation}/{STAGNATION_GENS}  "
                      f"restarts={self.restart_count}  "
                      f"tk={self.tournament_k}  "
                      f"[{self.mode.upper()}]"
                      f"  [FitPhase={_fitness_phase}]"
                      f"{'  [Phase1-weighted]' if _in_phase1_now else ''}"
                      f"{'  [Phase2-stabilize]' if _in_phase2_now else ''}"
                      f"{'  [Phase3-dynamic]'   if _in_phase3_now else ''}") 

                # ── Оценка популяции ────────────────────────────────────────
                if _in_phase1_now:
                    fits = np.full(POP_SIZE, -np.inf, dtype=np.float64)
                    rets: List[list] = [[] for _ in range(POP_SIZE)]
                    _island_phase1_stats: Dict[int, dict] = {}
                    for _iid, _iidx in enumerate(self.islands):
                        _ip   = self.island_precomp.get(_iid) or self.precomp
                        _ipop = self.pop[np.array(_iidx)]
                        _ifits, _irets = self._eval_island_batch(_ipop, _ip)
                        for _local_k, _global_k in enumerate(_iidx):
                            fits[_global_k] = _ifits[_local_k]
                            rets[_global_k] = _irets[_local_k]
                        # Детальная статистика острова: avg_ret лучшего агента по режимам
                        _valid_f = _ifits[np.isfinite(_ifits)]
                        _best_k  = int(np.argmax(_ifits))
                        _best_br = _irets[_best_k] if _irets else []
                        _regime_avgs: Dict[str, list] = {}
                        for _pi, _entry in enumerate(_ip):
                            _reg = map_regime_3(_entry[6]) if len(_entry) > 6 else 'neutral'
                            if _pi < len(_best_br):
                                _regime_avgs.setdefault(_reg, []).append(_best_br[_pi])
                        _regime_summary = '  '.join(
                            f"{_r}:{np.mean(_v):+.2f}%"
                            for _r, _v in sorted(
                                _regime_avgs.items(),
                                key=lambda x: -REGIME_WEIGHTS.get(x[0], 1.0))
                            if _v
                        )
                        _island_phase1_stats[_iid] = {
                            'best':  float(_ifits[_best_k]),
                            'mean':  float(_valid_f.mean()) if len(_valid_f) else 0.0,
                            'n':     len(_iidx),
                            'regime_avgs': _regime_summary,
                        }
                    # Печатаем детали по островам в консоль и лог
                    print(f"  [Phase1 G{self.gen:03d}] детали островов:")
                    for _iid, _stat in _island_phase1_stats.items():
                        _rname = ISLAND_TO_REGIME.get(_iid, str(_iid))
                        _line  = (f"    island[{_iid}:{_rname:>12s}]  "
                                  f"best={_stat['best']:+.4f}  mean={_stat['mean']:+.4f}  "
                                  f"n={_stat['n']}  regime_rets: {_stat['regime_avgs']}")
                        print(_line)
                        _glog(_line, also_print=False)
                else:
                    # Phase 2: вся популяция на полном наборе периодов
                    # Передаём динамические веса режимов для противодействия передержке
                    if self.mode == 'gpu':
                        fits, rets = self.ev.evaluate(self.pop, self.precomp,
                                                      rw_override=_dyn_rw)
                    else:
                        fits, rets = self.ev.evaluate(self.pop)

                v   = fits[np.isfinite(fits)]
                bi  = int(np.argmax(fits))
                bf  = float(fits[bi]); br = rets[bi]
                ela = time.time() - t0

                mean_r   = float(v.mean()) if len(v) else 0.0
                worst_r  = float(v.min())  if len(v) else 0.0
                wr  = sum(1 for r in br if r >= WIN_PROFIT_THRESH)  if br else 0
                bw  = sum(1 for r in br if r >= BONUS_PROFIT_THRESH) if br else 0
                zr  = sum(1 for r in br if abs(r) < ZERO_THRESH)     if br else 0
                # ── ПОЯСНЕНИЕ win-метрик в логе ─────────────────────────────────
                # win≥N%  — кол-во периодов где return финального 60-бар снимка ≥ N%
                #           (НЕ полный месяц! snapshot = последние 60 баров)
                # pos%    — % периодов с положительным ПОЛНЫМ return (как в бэктесте)
                # Именно pos% совпадает с win_rate в backtest_results.xlsx
                pos_months = sum(1 for r in br if r > 0) if br else 0
                pos_pct    = f"{pos_months/len(br)*100:.0f}%" if br else "n/a"
                avg_ret  = f"{np.mean(br):>+6.2f}%" if br else "n/a"
                new_best_tag = " ★NEW" if bf > self.best_fit else ""
                nes_mag = f"{self.nes_grad_mags[-1]:.3f}" if self.nes_grad_mags else "n/a"

                # BUG-FIX 3: В Phase 1 лучший геном выбирается по island-weighted
                # фитнесу (один остров завышает веса "своих" режимов). avg_ret,
                # win≥5%, big_win≥20% отражают специализированного island-агента,
                # а не сбалансированный результат по всем 96 периодам.
                # Добавляем явную пометку [Phase1-island] в лог.
                phase1_warn = "  ⚠ island-weighted (Phase1)" if _in_phase1_now else ""
                phase2_warn = "  [stabilize]" if _in_phase2_now else ""

                # ── Восстанавливаем sigma после Phase2-eval ──────────────────────
                if _saved_sigma is not None:
                    self.sigma = _saved_sigma

                # Показываем топ-3 усиленных режима в логе (когда динамические веса активны)
                _dyn_rw_str = ""
                if _dyn_rw and _in_phase3_now:
                    _rw_sorted = sorted(_dyn_rw.items(),
                                        key=lambda kv: kv[1] / (REGIME_WEIGHTS.get(kv[0], 1.0) + 1e-9),
                                        reverse=True)[:3]
                    _dyn_rw_str = "  dyn↑[" + " ".join(
                        f"{r}×{w/REGIME_WEIGHTS.get(r,1.0):.1f}"
                        for r, w in _rw_sorted
                    ) + "]"

                # ── NES-коллапс детект (только Phase 3) ──────────────────────────
                if _in_phase3_now:
                    _cur_nes = self.nes_grad_mags[-1] if self.nes_grad_mags else 1.0
                    if _cur_nes < NES_COLLAPSE_THRESHOLD:
                        self._nes_low_streak += 1
                    else:
                        self._nes_low_streak = 0
                    if self._nes_low_streak >= NES_COLLAPSE_GENS:
                        print(f"  [NES-COLLAPSE] NES={_cur_nes:.3f} < {NES_COLLAPSE_THRESHOLD} "
                              f"на протяжении {self._nes_low_streak} генов → тяжёлая инъекция")
                        self._inject_diversity(fits, force_heavy=True)
                        self._nes_low_streak = 0
                else:
                    self._nes_low_streak = 0

                # Обновляем историю возвратов по режимам (для следующего поколения)
                if br and not _in_phase1_now:
                    self._update_regime_ret_history(br)

                gen_line = (
                    f"  G{self.gen:03d}  best={bf:>+8.4f}  mean={mean_r:>+8.4f}  "
                    f"worst={worst_r:>+8.4f}  avg_ret={avg_ret}  pos%={pos_pct}  "
                    f"win≥{WIN_PROFIT_THRESH:.0f}%={wr}  "
                    f"big_win≥{BONUS_PROFIT_THRESH:.0f}%={bw}  "
                    f"zero={zr}/{len(br) if br else 0}  "
                    f"nes={nes_mag}  t={ela:.1f}s{new_best_tag}{phase1_warn}{phase2_warn}{_dyn_rw_str}"
                )
                print(gen_line)
                _glog(gen_line, also_print=False)

                improved = bf > self.best_fit
                if improved:
                    old_best_g = self.best_g.copy() if self.best_g is not None else self.pop[bi].copy()
                    self.best_fit = bf
                    self.best_g   = self.pop[bi].copy()
                    # Бэкап предыдущего best_genome перед перезаписью.
                    # Гарантирует, что хорошо обученный геном не теряется при перезапуске.
                    best_path = os.path.join(AGENTS_DIR, "best_genome.npy")
                    if os.path.exists(best_path):
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        bak = os.path.join(AGENTS_DIR, f"best_genome_bak_{ts}_fit{self.best_fit:+.4f}.npy")
                        try:
                            import shutil as _sh
                            _sh.copy2(best_path, bak)
                        except Exception:
                            pass
                    np.save(best_path, self.best_g)
                    # Сохраняем метаданные: поколение и фитнес — для GeneticsAgent
                    _meta = {'gen': self.gen, 'fitness': float(bf),
                             'max_pos': int(TRAIN_MAX_POS), 'genome_size': int(GENOME_SIZE)}
                    try:
                        import json as _json
                        with open(os.path.join(AGENTS_DIR, "best_genome_meta.json"), 'w') as _mf:
                            _json.dump(_meta, _mf)
                    except Exception:
                        pass
                    _glog(f"  ★ NEW BEST  fitness={bf:+.4f}")
                    self._update_cma(self.best_g, old_best_g)
                    # В Phase 1 br содержит возвраты только по island-периодам — индексы
                    # не совпадают с self.precomp → пропускаем _save_best_per_regime.
                    # В Phase 2 маппинг корректен.
                    if br and not _in_phase1_now:
                        self._save_best_per_regime(br)

                # В Phase 2 (стабилизация): подавляем escape-burst sigma, только exploit
                if _in_phase2_now:
                    self._check_restart_phase2(improved)
                else:
                    self._check_restart(improved)
                # Архив глобальных лучших — только в Phase 2/3 (в Phase 1 фитнесы
                # island-specific и несопоставимы между островами)
                if not _in_phase1_now:
                    self._update_archive(fits)
                # Сохраняем лучших по островам каждые 3 гена (не каждый — IO-overhead)
                if self.gen % 3 == 0 or improved:
                    self._save_island_bests(fits)
                    # Обновляем режимные геномы от лучших особей каждого острова.
                    # В Phase 1: br лучшего острова совпадает с island_precomp[island_id].
                    # В Phase 2: br лучшего в острове по всему precomp — корректно.
                    if rets:
                        for island_id, island_idx in enumerate(self.islands):
                            if not island_idx:
                                continue
                            island_fits_local = fits[island_idx]
                            best_k = int(np.argmax(island_fits_local))
                            best_slot = island_idx[best_k]
                            island_br = rets[best_slot]
                            if island_br and not _in_phase1_now:
                                self._save_best_per_regime(
                                    island_br, genome=self.pop[best_slot])

                # Force diversity inject после MAX_RESTARTS_NO_IMPROVE
                # (только в Phase 3 — в Phase 2 популяция намеренно гомогенна вокруг best_g)
                if _in_phase3_now and self.restarts_no_improve >= MAX_RESTARTS_NO_IMPROVE:
                    self.deep_stagnation_count += 1
                    print(f"  [FORCE DIVERSITY] {self.restarts_no_improve} рестартов без улучшения  "
                          f"deep_stag={self.deep_stagnation_count}")
                    self._inject_diversity(fits, force_heavy=True)
                    self.restarts_no_improve = 0

                    # ── ГЛУБОКАЯ СТАГНАЦИЯ: эскейп когда diversity не помогает ──────
                    # Если deep_stagnation_count >= 4: популяция застряла в локальном минимуме.
                    # Делаем ЧАСТИЧНЫЙ СБРОС: 60% популяции → xavier + высокая sigma.
                    # Элита (ELITE_SIZE особей) сохраняется.
                    _DEEP_STAG_THRESH = _gscfg_i(_GS, 'deep_stagnation_thresh', 4)
                    if self.deep_stagnation_count >= _DEEP_STAG_THRESH:
                        _n_reset = int(POP_SIZE * 0.60)
                        _elite_idx = set(np.argsort(fits)[::-1][:ELITE_SIZE].tolist())
                        _reset_slots = [i for i in np.argsort(fits)[:_n_reset].tolist()
                                        if i not in _elite_idx]
                        # 40% новых xavier + 60% мутанты best_g с высокой sigma
                        _n_xavier = max(1, int(len(_reset_slots) * 0.40))
                        _n_mutant = len(_reset_slots) - _n_xavier
                        _xavier_new = _xavier_pop(self.rng, _n_xavier)
                        for _k, _sl in enumerate(_reset_slots[:_n_xavier]):
                            self.pop[_sl] = _xavier_new[_k]
                        if _n_mutant > 0 and self.best_g is not None:
                            _esc_sigma = min(MUTATION_SIGMA * 1.2, SIGMA_MIN * 8)
                            _esc_noise = (self.rng.standard_normal((_n_mutant, GENOME_SIZE))
                                          * _esc_sigma).astype(np.float32)
                            _esc_mask  = self.rng.random((_n_mutant, GENOME_SIZE)) < 0.35
                            _esc_noise *= _esc_mask
                            _esc_pop   = np.clip(self.best_g[None, :] + _esc_noise, -5.0, 5.0)
                            for _k, _sl in enumerate(_reset_slots[_n_xavier:]):
                                self.pop[_sl] = _esc_pop[_k].astype(np.float32)
                        # Повышаем sigma для следующих генов
                        _old_sigma = self.sigma
                        self.sigma = min(MUTATION_SIGMA * 1.5, MUTATION_SIGMA * 0.9)
                        self.sigma = max(self.sigma, SIGMA_MIN * 4)
                        self.deep_stagnation_count = 0
                        print(f"  [DEEP ESCAPE] {len(_reset_slots)} слотов сброшено "
                              f"({_n_xavier} xavier + {_n_mutant} мутанты)  "
                              f"sigma: {_old_sigma:.4f}→{self.sigma:.4f}")

                # ── Ранний детект нарастающего коллапса (не ждём DIVERSITY_INTERVAL) ─
                # При 3 последовательных new best (gen57→58→59) NES+elite выталкивает
                # всю популяцию к best_g → sim=1.000 к gen60. Проверяем diversity
                # каждый ген когда stagnation==0 (только что было улучшение).
                # Срабатывает ТОЛЬКО если sim > 0.90 (избегаем лишних вычислений).
                elif _in_phase3_now and self.stagnation == 0:
                    _top_k_early = max(5, POP_SIZE // 10)
                    _top_idx_e   = np.argsort(fits)[::-1][:_top_k_early]
                    _top_pop_e   = self.pop[_top_idx_e]
                    _norms_e     = np.linalg.norm(_top_pop_e, axis=1, keepdims=True) + 1e-8
                    _normed_e    = _top_pop_e / _norms_e
                    _sim_e       = (_normed_e @ _normed_e.T)
                    np.fill_diagonal(_sim_e, 0.0)
                    _avg_sim_e   = _sim_e.sum() / max(1, _top_k_early * (_top_k_early - 1))
                    if _avg_sim_e > 0.90:
                        print(f"  [EARLY COLLAPSE] sim={_avg_sim_e:.3f} — превентивная инъекция")
                        self._inject_diversity(fits)

                # Регулярная инъекция — только в Phase 3
                if _in_phase3_now and self.gen % DIVERSITY_INTERVAL == 0:
                    self._inject_diversity(fits)

                # Island migration: в Phase 1 подавляем (острова специализируются независимо)
                if _in_phase1_now:
                    # Последний ген Phase 1: g == _phase1_gens - 1
                    if g == _phase1_gens - 1:
                        print(f"  [Phase1 → Phase2] Конец фазы 1 — кросс-островная миграция активирована")
                        if self.best_g is not None:
                            self.archive.append((self.best_fit, self.best_g.copy()))
                            self.archive.sort(key=lambda x: -x[0])
                            self.archive = self.archive[:ARCHIVE_SIZE]
                        _old_p1_best = self.best_fit
                        self.best_fit        = -np.inf
                        self.stagnation      = 0
                        self.restarts_no_improve = 0
                        # ── КРИТИЧНО: очищаем архив от Phase1-scale записей ─────────
                        # Phase1 фитнес (3.5×mean_r) может достигать ~97+.
                        # Новая единая формула (2.5×mean_r + штрафы) даёт max ~20.
                        # Phase1-записи в архиве делают best_fit недостижимым →
                        # _inject_diversity всегда реинжектирует crash-специалиста →
                        # деградация. Очищаем ПЕРЕД записью best_fit=-inf.
                        _FIT_TRANSITION_THRESHOLD = 25.0
                        _arc_before = len(self.archive)
                        self.archive = [(f, g) for f, g in self.archive
                                        if f <= _FIT_TRANSITION_THRESHOLD]
                        # Сбрасываем историю возвратов — Phase1 данные не релевантны
                        for _r in self.regime_ret_history:
                            self.regime_ret_history[_r].clear()
                        print(f"  [Phase1 → Phase2] best_fit сброшен: "
                              f"{_old_p1_best:+.4f} → -∞  (Phase1-геном в архиве)")
                        if _arc_before != len(self.archive):
                            print(f"  [Phase1 → Phase2] Архив очищен от Phase1-записей: "
                                  f"{_arc_before} → {len(self.archive)} "
                                  f"(порог > {_FIT_TRANSITION_THRESHOLD})")
                elif self.gen % MIGRATION_INTERVAL == 0:
                    self._island_migrate(fits)
                    # Инъекция режимных геномов в слабые острова — только в Phase 3
                    if _in_phase3_now and self.best_per_regime:
                        self._regime_balance_inject(fits)

                if self.gen % SAVE_EVERY == 0:
                    np.save(os.path.join(AGENTS_DIR, f"gen{self.gen:04d}.npy"),
                            self.pop[bi])
                    self._save_state()

                # Per-island best fitness snapshot for chart
                island_bests_snap = {
                    ISLAND_TO_REGIME.get(iid, str(iid)):
                        float(self._island_best_fits.get(iid, -np.inf))
                    for iid in range(len(self.islands))
                }
                self.history.append({
                    'gen': self.gen, 'best': bf,
                    'mean': mean_r,
                    'worst': worst_r,
                    'mean_ret': float(np.mean(br)) if br else 0.0,
                    'std_ret':  float(np.std(br))  if br else 0.0,
                    'win':  wr / max(len(br), 1),
                    'zero': zr / max(len(br), 1),
                    'sigma': self.sigma,
                    'stagnation': self.stagnation,
                    'restarts': self.restart_count,
                    'tournament_k': self.tournament_k,
                    'nes_grad': self.nes_grad_mags[-1] if self.nes_grad_mags else 0.0,
                    'island_div': self.island_diversity_hist[-1] if self.island_diversity_hist else 0.0,
                    'island_bests': island_bests_snap,
                    'sec': round(ela, 1), 'rets': br
                })
                self._save_log()

                if self.gen % CHART_EVERY == 0 or self.gen == N_GENERATIONS:
                    self._chart()

                self.pop = self._next(fits)

        finally:
            if self.mode == 'cpu':
                self.ev.shutdown()
            _glog(
                f"DONE  Best={self.best_fit:+.4f}  Restarts={self.restart_count}  "
                f"лог: {_log_path}"
            )
            _log_f.close()

        print("\n" + "="*70)
        print(f"  DONE  Best={self.best_fit:+.4f}  Restarts={self.restart_count}")
        print(f"  Archive top-{ARCHIVE_SIZE}: {[f'{f:+.4f}' for f,_ in self.archive]}")
        self._save_state()
        self._save_archive()
        self._report()

    def _save_log(self):
        with open(os.path.join(AGENTS_DIR, "training_log.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

    def _save_archive(self):
        for rank, (fit, genome) in enumerate(self.archive):
            path = os.path.join(AGENTS_DIR, f"archive_rank{rank+1}.npy")
            np.save(path, genome)
            print(f"  Archive #{rank+1}: {fit:+.4f} -> {path}")

    def _report(self):
        last = self.history[-1]['rets'] if self.history else []
        rep  = {
            'genome_size': GENOME_SIZE, 'pop': POP_SIZE, 'gens': N_GENERATIONS,
            'alpha': FITNESS_ALPHA, 'beta': FITNESS_BETA,
            'gamma': FITNESS_GAMMA, 'delta': FITNESS_DELTA,
            'epsilon': FITNESS_EPSILON,
            'zeta': FITNESS_ZETA, 'eta': FITNESS_ETA,
            'best_fitness': self.best_fit,
            'mean_ret':  float(np.mean(last)) if last else None,
            'std_ret':   float(np.std(last))  if last else None,
            'periods':   len(self.precomp), 'period_rets': last,
            'sigma_restarts': self.restart_count,
            'n_islands': N_ISLANDS,
            'arch': {'in': N_INPUT, 'h1': N_HIDDEN1, 'h2': N_HIDDEN2, 'out': N_ACTIONS,
                     'activation': 'ELU', 'version': 'v7-regime_balanced'}
        }
        with open(os.path.join(AGENTS_DIR, "training_report.json"), 'w') as f:
            json.dump(rep, f, indent=2)
        self._chart()
        print(f"  Report: {AGENTS_DIR}/training_report.json")

    def _chart(self):
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gspec
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D

            # ── Палитра ──────────────────────────────────────────────────────
            D  = "#0D1117"; M  = "#161B22"; G  = "#21262D"
            CY = "#58A6FF"; GR = "#3FB950"; RE = "#F85149"
            GO = "#F0C040"; PU = "#BC8CFF"; OR = "#FF8C00"; TE = "#79C0FF"
            PI = "#FF6EB4"; LG = "#7EE787"

            # Цвета режимов рынка
            REGIME_COLORS = {
                'bearish': "#FF6B6B",
                'neutral': "#AAAAAA",
                'bullish': "#4CAF50",
            }
            REGIME_LABELS = {
                'bearish': 'Bearish',
                'neutral': 'Neutral',
                'bullish': 'Bullish',
            }
            REGIMES_ORDERED = ['bearish', 'neutral', 'bullish']

            def sty(ax, grid_alpha=0.4):
                ax.set_facecolor(M)
                ax.tick_params(colors='#888', labelsize=8)
                [sp.set_color(G) for sp in ax.spines.values()]
                ax.grid(True, color=G, lw=grid_alpha)

            def title(ax, txt, fs=9):
                ax.set_title(txt, color='white', fontsize=fs, fontweight='bold', pad=5)

            # ── История ──────────────────────────────────────────────────────
            # _loaded_history = снимок прошлой сессии (только для серого фона, не меняется)
            # self.history    = все поколения (старые + новые), накапливается в run()
            _snapshot_idx = getattr(self, '_session_start_idx', 0)
            _prev_hist = getattr(self, '_loaded_history', [])   # снимок прошлой сессии

            # Новые поколения — срез от начала текущей сессии
            _new_hist  = self.history[_snapshot_idx:]

            # Полная слитая история без дублей (для островов, режимов и т.д.)
            _all_hist  = self.history

            # Новые поколения для основных цветных линий
            gens  = [h['gen']      for h in _new_hist]
            best  = [h['best']     for h in _new_hist]
            mean_ = [h['mean']     for h in _new_hist]
            worst = [h['worst']    for h in _new_hist]
            sigs  = [h['sigma']    for h in _new_hist]
            bmr   = [h['mean_ret'] for h in _new_hist]
            bsr   = [h['std_ret']  for h in _new_hist]
            wrs   = [h['win']*100  for h in _new_hist]
            zps   = [h.get('zero',0)*100 for h in _new_hist]
            rsts  = [h.get('restarts',0) for h in _new_hist]
            tks   = [h.get('tournament_k', TOURNAMENT_K) for h in _new_hist]
            nes_g = [h.get('nes_grad', 0.0) for h in _new_hist]
            idiv  = [h.get('island_div', 0.0) for h in _new_hist]

            # Загруженные (предыдущая сессия) — для серого фона
            prev_gens  = [h['gen']   for h in _prev_hist]
            prev_best  = [h['best']  for h in _prev_hist]
            prev_mean  = [h['mean']  for h in _prev_hist]
            prev_worst = [h['worst'] for h in _prev_hist]

            # Per-island best fitness history (объединяем loaded + new)
            island_fit_hist: Dict[str, list] = {r: [] for r in REGIME_ORDER}
            for h in _all_hist:
                ibf = h.get('island_bests', {})
                for r in REGIME_ORDER:
                    v = ibf.get(r, None)
                    island_fit_hist[r].append(v if v is not None and v > -1e10 else None)

            # ── Разбивка периодов по режимам (из precomp) ────────────────────
            # BUG-FIX 4: Старый код искал режим по весу (entry[5]), что ломалось
            # в Phase 1 (island_precomp переопределяет rw) и при совпадении весов.
            # Правильный режим хранится напрямую в entry[6] — используем его.
            period_regimes = []
            for entry in self.precomp:
                reg = map_regime_3(entry[6]) if len(entry) > 6 and isinstance(entry[6], str) else 'neutral'
                period_regimes.append(reg)

            # ── Для каждого поколения — словарь {regime: [rets]} ─────────────
            def _regime_stats_for_rets(rets_list):
                """Разбивает список возвратов на словарь {режим: [возвраты]}."""
                rd: Dict[str, list] = {r: [] for r in REGIMES_ORDERED}
                for i, r in enumerate(rets_list):
                    if i < len(period_regimes):
                        rd[period_regimes[i]].append(r)
                return rd

            # Текущие статистики по режимам — из последнего доступного поколения
            _last_hist = _all_hist[-1] if _all_hist else None
            cur_rets  = _last_hist['rets'] if _last_hist else []
            cur_rd    = _regime_stats_for_rets(cur_rets)

            def _rs(vals):
                """avg, std, win≥1%, win≥5%, win≥20%, count"""
                if not vals:
                    return 0., 0., 0., 0., 0., 0
                a = np.array(vals, dtype=np.float64)
                n = len(a)
                return (float(a.mean()), float(a.std()),
                        float((a>=WIN1_PROFIT_THRESH).mean()*100),
                        float((a>=WIN_PROFIT_THRESH).mean()*100),
                        float((a>=BONUS_PROFIT_THRESH).mean()*100),
                        n)

            cur_stats = {r: _rs(cur_rd[r]) for r in REGIMES_ORDERED}

            # История среднего возврата по режимам (для трендовых графиков)
            regime_avg_hist: Dict[str, list] = {r: [] for r in REGIMES_ORDERED}
            regime_w5_hist:  Dict[str, list] = {r: [] for r in REGIMES_ORDERED}
            for h in _new_hist:   # только поколения текущей сессии (синхронно с gens)
                rd = _regime_stats_for_rets(h.get('rets', []))
                for r in REGIMES_ORDERED:
                    s = _rs(rd[r])
                    regime_avg_hist[r].append(s[0])  # mean return
                    regime_w5_hist[r].append(s[3])   # win≥5%

            # ════════════════════════════════════════════════════════════════
            # КОМПОНОВКА:  8 строк × 2 столбца
            # 0   [=======  Fitness evolution  =======]  full
            # 1   [=======  Period returns bar  ========]  full  (с раскраской режимов)
            # ─────── РАЗДЕЛ: АНАЛИЗ ПО РЕЖИМАМ ──────────────────────────────
            # 2   [  Текущий avg по режимам  ] [  Win rate 5% / 20%  ]
            # 3   [  Тепловая карта режимов  (full width)  ]
            # 4   [  Тренд avg по режимам    ] [  Тренд win≥5%       ]
            # ─────── ТЕХНИЧЕСКИЕ МЕТРИКИ ─────────────────────────────────────
            # 5   [  Avg return ± std        ] [  Win / Zero rate    ]
            # 6   [  Sigma + restarts        ] [  NES + Island div   ]
            # ════════════════════════════════════════════════════════════════
            fig = plt.figure(figsize=(22, 28), facecolor=D)
            gs = gspec.GridSpec(
                7, 2, figure=fig,
                hspace=0.52, wspace=0.30,
                left=0.06, right=0.97, top=0.97, bottom=0.03,
                height_ratios=[2.0, 1.8, 1.6, 2.0, 1.8, 1.6, 1.6]
            )

            # ── ROW 0: Fitness evolution ──────────────────────────────────────
            ax1 = fig.add_subplot(gs[0, :])
            sty(ax1)
            # Предыдущая сессия — одна опорная точка (не загромождать график)
            if prev_gens:
                ref_x = prev_gens[-1]
                ref_best = prev_best[-1]
                ref_mean = prev_mean[-1]
                ax1.scatter([ref_x], [ref_best], color='#888888', s=50,
                            marker='D', zorder=4, label='Prev Best', alpha=0.8)
                ax1.scatter([ref_x], [ref_mean], color='#666666', s=30,
                            marker='D', zorder=3, label='Prev Mean', alpha=0.6)
                if gens:
                    ax1.axvline(ref_x, color='#888888', lw=1.0,
                                ls='--', alpha=0.5, label='Session start')
            if gens:
                ax1.fill_between(gens, worst, best, color=CY, alpha=0.10)
                ax1.plot(gens, best,  color=GR, lw=2.5, label='Best', zorder=3)
                ax1.plot(gens, mean_, color=GO, lw=1.6, ls='--', label='Mean', zorder=2)
                ax1.plot(gens, worst, color=RE, lw=0.9, ls=':', label='Worst')
                ax1.axhline(0, color='white', lw=0.7, ls='--', alpha=0.4)
                for i in range(1, len(gens)):
                    if rsts[i] > rsts[i-1]:
                        ax1.axvline(gens[i], color=OR, lw=1.0, ls=':', alpha=0.8)
                all_gens_for_vlines = gens
                for dg in range(DIVERSITY_INTERVAL, max(all_gens_for_vlines)+2, DIVERSITY_INTERVAL):
                    ax1.axvline(dg, color=PU, lw=0.8, ls='--', alpha=0.4)
                for mg in range(MIGRATION_INTERVAL, max(all_gens_for_vlines)+2, MIGRATION_INTERVAL):
                    ax1.axvline(mg, color=TE, lw=0.7, ls='-.', alpha=0.35)
                custom = [
                    Line2D([0],[0],color=OR,lw=1,ls=':',  label='σ-restart'),
                    Line2D([0],[0],color=PU,lw=1,ls='--', label='diversity'),
                    Line2D([0],[0],color=TE,lw=1,ls='-.', label='migration'),
                ]
                h0, l0 = ax1.get_legend_handles_labels()
                ax1.legend(h0+custom, l0+['σ-restart','diversity','migration'],
                           fontsize=8, loc='lower right')
            _all_best = (prev_best if prev_best else []) + (best if best else [])
            title(ax1,
                  f"Genetics v7  Pop={POP_SIZE}  Islands={N_ISLANDS}  "
                  f"{N_INPUT}→{N_HIDDEN1}→{N_HIDDEN2}→{N_HIDDEN3}→{N_ACTIONS}  "
                  f"Mode={self.mode.upper()}  Best: {max(_all_best) if _all_best else 0:+.4f}  "
                  f"σ={self.sigma:.4f}  restarts={self.restart_count}",
                  fs=10)
            ax1.set_ylabel("Fitness", color='#888')
            ax1.set_xlabel("Generation", color='#888')

            # ── ROW 1: Period P/L с equity curve и аннотациями $$ ─────────────
            ax2 = fig.add_subplot(gs[1, :])
            sty(ax2)
            if cur_rets:
                xs = list(range(len(cur_rets)))
                n_p = len(cur_rets)

                # ── Вычисляем $ P/L для каждого периода ──────────────────────
                init_cap = float(getattr(_cx, 'INITIAL_CAPITAL', 10000)) if _CX_OK else 10000.0
                pnl_dollars = [init_cap * r / 100.0 for r in cur_rets]
                # Кумулятивный equity: каждый период — независимый (одинаковый стартовый капитал)
                cumulative_equity = []
                running = init_cap
                for r in cur_rets:
                    running = running * (1.0 + r / 100.0)
                    cumulative_equity.append(running)
                total_pnl = cumulative_equity[-1] - init_cap

                bar_colors = [REGIME_COLORS.get(period_regimes[i], GR)
                              if i < len(period_regimes) else GR
                              for i in xs]
                def _dim(hex_col, factor=0.5):
                    rgb = matplotlib.colors.to_rgb(hex_col)
                    return (rgb[0]*factor, rgb[1]*factor, rgb[2]*factor, 0.75)
                final_colors = [
                    matplotlib.colors.to_rgba(c) if cur_rets[i] >= 0 else _dim(c)
                    for i, c in enumerate(bar_colors)
                ]

                # Бары: P/L в долларах (не %)
                ax2.bar(xs, pnl_dollars, color=final_colors, alpha=0.9, width=0.9, zorder=2)
                ax2.axhline(0, color='white', lw=0.9, zorder=3)

                # Пороговые линии в долларах
                thresh_mini = init_cap * WIN1_PROFIT_THRESH / 100.0
                thresh_win  = init_cap * WIN_PROFIT_THRESH / 100.0
                thresh_big  = init_cap * BONUS_PROFIT_THRESH / 100.0
                ax2.axhline( thresh_mini, color=GO,  lw=0.8, ls=':',  alpha=0.6,
                             label=f'+${thresh_mini:,.0f} ({WIN1_PROFIT_THRESH:.0f}%)', zorder=3)
                ax2.axhline( thresh_win,  color=LG,  lw=1.0, ls='--', alpha=0.6,
                             label=f'+${thresh_win:,.0f} ({WIN_PROFIT_THRESH:.0f}%)', zorder=3)
                ax2.axhline( thresh_big,  color=CY,  lw=1.0, ls=':',  alpha=0.6,
                             label=f'+${thresh_big:,.0f} ({BONUS_PROFIT_THRESH:.0f}%)', zorder=3)

                # ── Аннотации $ на ключевых барах (топ-3 и bottom-3) ──────────
                sorted_idx = sorted(range(n_p), key=lambda i: pnl_dollars[i])
                annotate_idx = set(sorted_idx[:3] + sorted_idx[-3:])
                # Также аннотируем бары > thresh_big или < -thresh_win
                for i in range(n_p):
                    if abs(pnl_dollars[i]) >= thresh_win:
                        annotate_idx.add(i)
                for i in annotate_idx:
                    v = pnl_dollars[i]
                    lbl = f"${v:+,.0f}\n({cur_rets[i]:+.1f}%)"
                    yoff = 3 if v >= 0 else -3
                    ax2.annotate(lbl, xy=(i, v), xytext=(0, yoff),
                                 textcoords='offset points', ha='center',
                                 va='bottom' if v >= 0 else 'top',
                                 fontsize=5.5, color='white', fontweight='bold',
                                 alpha=0.9)

                # ── Equity curve на вторичной оси ─────────────────────────────
                ax2r = ax2.twinx()
                ax2r.plot(xs, cumulative_equity, color=GO, lw=2.2, alpha=0.85,
                          zorder=5, label='Equity')
                ax2r.axhline(init_cap, color=GO, lw=0.7, ls='--', alpha=0.4)
                # Подсветка max drawdown на equity
                eq_arr = np.array(cumulative_equity)
                eq_peak = np.maximum.accumulate(eq_arr)
                eq_dd = (eq_peak - eq_arr) / (eq_peak + 1e-9) * 100
                max_dd_idx = int(np.argmax(eq_dd))
                max_dd_val = float(eq_dd[max_dd_idx])
                if max_dd_val > 1.0:
                    ax2r.annotate(f"MaxDD: {max_dd_val:.1f}%",
                                  xy=(max_dd_idx, cumulative_equity[max_dd_idx]),
                                  fontsize=7, color=RE, fontweight='bold',
                                  xytext=(10, -15), textcoords='offset points',
                                  arrowprops=dict(arrowstyle='->', color=RE, lw=1.2))
                ax2r.set_ylabel("Equity $", color=GO, fontsize=8)
                ax2r.tick_params(colors=GO, labelsize=7)
                ax2r.spines['right'].set_color(GO)
                ax2r.legend(fontsize=7, loc='upper right', framealpha=0.3, edgecolor=G)

                labs = [entry[4] for entry in self.precomp[:len(cur_rets)]]
                step = max(1, n_p // 24)
                ax2.set_xticks(xs[::step])
                ax2.set_xticklabels(labs[::step], rotation=45, ha='right',
                                    fontsize=6.5, color='white')

                # Мини-легенда режимов поверх графика
                patches = [mpatches.Patch(color=REGIME_COLORS[r],
                                          label=REGIME_LABELS[r])
                           for r in REGIMES_ORDERED if any(pr == r for pr in period_regimes)]
                ax2.legend(handles=patches + ax2.get_legend_handles_labels()[0][-2:],
                           fontsize=7, loc='upper left', ncol=6,
                           framealpha=0.3, edgecolor=G)

                wr2  = sum(1 for r in cur_rets if r > 0)
                w1   = sum(1 for r in cur_rets if r >= WIN1_PROFIT_THRESH)
                w5   = sum(1 for r in cur_rets if r >= WIN_PROFIT_THRESH)
                w20  = sum(1 for r in cur_rets if r >= BONUS_PROFIT_THRESH)
                zr2  = sum(1 for r in cur_rets if abs(r) < ZERO_THRESH)
                avg_pnl = np.mean(pnl_dollars)
                title(ax2,
                      f"Gen {self.gen} — лучшая особь:  "
                      f"avg={np.mean(cur_rets):+.2f}% (${avg_pnl:+,.0f}/период)  "
                      f"total P/L=${total_pnl:+,.0f}  "
                      f"win≥{WIN_PROFIT_THRESH:.0f}%={w5}/{n_p}  "
                      f"big_win≥{BONUS_PROFIT_THRESH:.0f}%={w20}/{n_p}  "
                      f"zero={zr2}/{n_p}  "
                      f"MaxDD={max_dd_val:.1f}%")
                ax2.set_ylabel("P/L per period ($)", color='#888')

            # ════════════════════════════════════════════════════════════════
            # РАЗДЕЛ: АНАЛИЗ ПО РЕЖИМАМ РЫНКА
            # ════════════════════════════════════════════════════════════════
            # Метка раздела
            fig.text(0.50, 0.605, "─── REGIME ANALYSIS ───",
                     color='#58A6FF', fontsize=10, fontweight='bold',
                     ha='center', va='center', alpha=0.8)

            # ── ROW 2 Left: Средний возврат по режимам (текущий ген) ──────────
            ax3 = fig.add_subplot(gs[2, 0])
            sty(ax3)
            r_avgs  = [cur_stats[r][0] for r in REGIMES_ORDERED]
            r_cnts  = [cur_stats[r][5] for r in REGIMES_ORDERED]
            r_cols  = [REGIME_COLORS[r] for r in REGIMES_ORDERED]
            r_lbls  = [f"{REGIME_LABELS[r]}\n(n={cur_stats[r][5]})"
                       for r in REGIMES_ORDERED]
            bars3 = ax3.bar(range(len(REGIMES_ORDERED)), r_avgs,
                            color=[matplotlib.colors.to_rgba(c) if v >= 0
                                   else tuple(max(0,x*0.6) for x in matplotlib.colors.to_rgb(c))+(0.85,)
                                   for c, v in zip(r_cols, r_avgs)],
                            alpha=0.9, width=0.65, zorder=2)
            # BUG FIX: итерируем по enumerate, не bars3.patches.index (падал ValueError)
            for bi, (bar, val) in enumerate(zip(bars3, r_avgs)):
                if cur_stats[REGIMES_ORDERED[bi]][5] > 0:
                    ypos = val + (0.5 if val >= 0 else -0.5)
                    ax3.text(bar.get_x() + bar.get_width()/2, ypos,
                             f"{val:+.1f}%", ha='center', va='bottom' if val >= 0 else 'top',
                             color='white', fontsize=7.5, fontweight='bold')
            ax3.axhline(0,                   color='white', lw=0.9, zorder=3)
            ax3.axhline( WIN_PROFIT_THRESH,  color=LG, lw=0.9, ls='--', alpha=0.6)
            ax3.axhline(-WIN_PROFIT_THRESH,  color=RE, lw=0.9, ls='--', alpha=0.4)
            ax3.set_xticks(range(len(REGIMES_ORDERED)))
            ax3.set_xticklabels([REGIME_LABELS[r] for r in REGIMES_ORDERED],
                                rotation=35, ha='right', fontsize=7.5, color='white')
            title(ax3, f"Avg return by market regime  (gen {self.gen})")
            ax3.set_ylabel("Avg Return %", color='#888')

            # ── ROW 2 Right: Win rates ≥1% / ≥5% / ≥20% по режимам ────────────────
            ax4 = fig.add_subplot(gs[2, 1])
            sty(ax4)
            x4 = np.arange(len(REGIMES_ORDERED))
            w4 = 0.27
            ax4.bar(x4 - w4,  [cur_stats[r][2] for r in REGIMES_ORDERED],
                    width=w4, color=GO,  alpha=0.85, label=f'≥{WIN1_PROFIT_THRESH:.0f}% (mini)',  zorder=2)
            ax4.bar(x4,       [cur_stats[r][3] for r in REGIMES_ORDERED],
                    width=w4, color=LG,  alpha=0.9,  label=f'≥{WIN_PROFIT_THRESH:.0f}% (win)',    zorder=2)
            ax4.bar(x4 + w4,  [cur_stats[r][4] for r in REGIMES_ORDERED],
                    width=w4, color=CY,  alpha=0.9,  label=f'≥{BONUS_PROFIT_THRESH:.0f}% (big_win)', zorder=2)
            # Подписи "X%\n(Nп)" — устраняет путаницу % vs кол-во периодов
            for xi, r in enumerate(REGIMES_ORDERED):
                v1  = cur_stats[r][2]; v5  = cur_stats[r][3]; v20 = cur_stats[r][4]
                n_r = max(1, cur_stats[r][5])
                if v1  > 2: ax4.text(xi - w4, v1  + 1, f"{v1:.0f}%\n({v1/100*n_r:.0f}п)",  ha='center', fontsize=5.5, color='white')
                if v5  > 2: ax4.text(xi,      v5  + 1, f"{v5:.0f}%\n({v5/100*n_r:.0f}п)",  ha='center', fontsize=5.5, color='white')
                if v20 > 2: ax4.text(xi + w4, v20 + 1, f"{v20:.0f}%\n({v20/100*n_r:.0f}п)", ha='center', fontsize=5.5, color=CY)
            ax4.axhline(50,  color='white', lw=0.7, ls='--', alpha=0.5, label='50%')
            ax4.axhline(100, color=G, lw=0.5, alpha=0.3)
            ax4.set_ylim(0, 115)
            ax4.set_xticks(x4)
            ax4.set_xticklabels([REGIME_LABELS[r] for r in REGIMES_ORDERED],
                                rotation=35, ha='right', fontsize=7.5, color='white')
            ax4.legend(fontsize=7.5, loc='upper right')
            title(ax4, f"Win-rate by market regime  (gen {self.gen})")
            ax4.set_ylabel("Win Rate %", color='#888')

            # ── ROW 3: Тепловая карта метрик × режим (full width) ────────────
            ax_hm = fig.add_subplot(gs[3, :])
            sty(ax_hm)
            # Матрица: строки = режимы, столбцы = метрики
            metric_names = ['Avg\nReturn%', 'Std\nReturn%',
                            f'Win≥{WIN1_PROFIT_THRESH:.0f}%\n(mini)',
                            f'Win≥{WIN_PROFIT_THRESH:.0f}%\n(win)',
                            f'Win≥{BONUS_PROFIT_THRESH:.0f}%\n(big_win)',
                            'n\nperiods']
            hm_data = np.zeros((len(REGIMES_ORDERED), len(metric_names)))
            for ri, r in enumerate(REGIMES_ORDERED):
                s = cur_stats[r]
                hm_data[ri] = [s[0], s[1], s[2], s[3], s[4], s[5]]

            # Нормируем каждый столбец для цвета (независимо)
            hm_norm = np.zeros_like(hm_data)
            for ci in range(len(metric_names)):
                col = hm_data[:, ci]
                mn, mx = col.min(), col.max()
                if mx > mn:
                    hm_norm[:, ci] = (col - mn) / (mx - mn)
                else:
                    # FIX: win-rate колонки (ci=2,3,4) — если все = 0.0 → norm=0.0 (красный)
                    # Раньше было 0.5 для всех → YlGn(0.5) = medium-green при 0% win rate!
                    hm_norm[:, ci] = 0.0 if (ci in (2, 3, 4) and mx == 0.0) else 0.5

            # Для Avg Return — центрируем вокруг нуля: зелёный=плюс, красный=минус
            avg_col = hm_data[:, 0]
            abs_max = max(abs(avg_col.min()), abs(avg_col.max()), 1e-3)
            hm_norm[:, 0] = (avg_col / abs_max + 1) / 2.0

            # FIX: win-rate cmap = RdYlGn (0%=красный, 50%=жёлтый, 100%=зелёный)
            # БЫЛО YlGn: нижняя половина шкалы уже зелёная → 0% выглядело как "норма"
            cmap_list = [
                plt.cm.RdYlGn,   # avg return (центрировано по 0)
                plt.cm.RdYlGn_r, # std (ниже = лучше)
                plt.cm.RdYlGn,   # win≥1%  — 0%=красный, 100%=зелёный
                plt.cm.RdYlGn,   # win≥5%  — 0%=красный, 100%=зелёный
                plt.cm.RdYlGn,   # win≥20% — 0%=красный, 100%=зелёный
                plt.cm.Blues,    # count
            ]
            cell_w = 1.0 / len(metric_names)
            cell_h = 1.0 / len(REGIMES_ORDERED)
            ax_hm.set_xlim(0, 1); ax_hm.set_ylim(0, 1)
            ax_hm.axis('off')
            for ri, r in enumerate(REGIMES_ORDERED):
                y_pos = 1.0 - (ri + 1) * cell_h
                # Метка режима слева
                ax_hm.text(-0.01, y_pos + cell_h*0.5,
                           REGIME_LABELS[r], color=REGIME_COLORS[r],
                           fontsize=9, fontweight='bold', va='center', ha='right')
                for ci in range(len(metric_names)):
                    x_pos = ci * cell_w
                    norm_val = float(hm_norm[ri, ci])
                    raw_val  = hm_data[ri, ci]
                    bg_color = cmap_list[ci](norm_val)
                    rect = plt.Rectangle((x_pos + 0.003, y_pos + 0.015),
                                         cell_w * 0.94, cell_h * 0.85,
                                         facecolor=bg_color, edgecolor=G, lw=0.5,
                                         transform=ax_hm.transData)
                    ax_hm.add_patch(rect)
                    # Значение ячейки
                    n_periods = hm_data[ri, 5]
                    if ci == 5:
                        val_str = f"{int(raw_val)}"
                    elif ci == 0:
                        val_str = f"{raw_val:+.1f}%"
                    elif ci in (2, 3, 4):
                        n_win = raw_val / 100 * max(n_periods, 1)
                        val_str = f"{raw_val:.1f}%\n({n_win:.0f}п)"
                    else:
                        val_str = f"{raw_val:.1f}%"
                    text_col = 'black'
                    ax_hm.text(x_pos + cell_w*0.5, y_pos + cell_h*0.45,
                               val_str, ha='center', va='center',
                               fontsize=7.5, fontweight='bold', color=text_col)
            # Заголовки столбцов
            for ci, mn in enumerate(metric_names):
                ax_hm.text(ci*cell_w + cell_w*0.5, 1.02, mn,
                           ha='center', va='bottom', fontsize=8, color='#aaa')
            title(ax_hm, f"Learnability heatmap by market regime  (gen {self.gen})",
                  fs=10)

            # ── ROW 4 Left: Тренд avg return по режимам ──────────────────────
            ax5 = fig.add_subplot(gs[4, 0])
            sty(ax5)
            if gens:
                for r in REGIMES_ORDERED:
                    vals = regime_avg_hist[r]
                    if any(v != 0 for v in vals):
                        ax5.plot(gens, vals, color=REGIME_COLORS[r],
                                 lw=1.6, label=REGIME_LABELS[r], alpha=0.9)
                ax5.axhline(0,                  color='white', lw=0.7, ls='--', alpha=0.4)
                ax5.axhline(WIN_PROFIT_THRESH,  color=LG, lw=0.8, ls=':', alpha=0.5)
                ax5.axhline(-WIN_PROFIT_THRESH, color=RE, lw=0.8, ls=':', alpha=0.4)
                ax5.legend(fontsize=7, loc='upper left', ncol=2,
                           framealpha=0.3, edgecolor=G)
            title(ax5, "Trend: avg return by regime")
            ax5.set_ylabel("Avg Return %", color='#888')
            ax5.set_xlabel("Generation", color='#888')

            # ── ROW 4 Right: Тренд win≥5% по режимам ─────────────────────────
            ax6 = fig.add_subplot(gs[4, 1])
            sty(ax6)
            if gens:
                for r in REGIMES_ORDERED:
                    vals = regime_w5_hist[r]
                    if any(v != 0 for v in vals):
                        ax6.plot(gens, vals, color=REGIME_COLORS[r],
                                 lw=1.6, label=REGIME_LABELS[r], alpha=0.9)
                ax6.axhline(50, color='white', lw=0.7, ls='--', alpha=0.5, label='50%')
                ax6.set_ylim(-5, 105)
                ax6.legend(fontsize=7, loc='lower right', ncol=2,
                           framealpha=0.3, edgecolor=G)
            title(ax6, f"Trend: periods with profit ≥{WIN_PROFIT_THRESH:.0f}% (win)")
            ax6.set_ylabel(f"Win≥{WIN_PROFIT_THRESH:.0f}% %", color='#888')
            ax6.set_xlabel("Generation", color='#888')

            # ════════════════════════════════════════════════════════════════
            # ТЕХНИЧЕСКИЕ МЕТРИКИ
            # ════════════════════════════════════════════════════════════════
            fig.text(0.50, 0.268, "─── TECHNICAL METRICS ───",
                     color='#BC8CFF', fontsize=10, fontweight='bold',
                     ha='center', va='center', alpha=0.8)

            # ── ROW 5 Left: Avg return ± std ─────────────────────────────────
            ax7 = fig.add_subplot(gs[5, 0])
            sty(ax7)
            if gens:
                ax7.bar(gens, bmr, color=[GR if v >= 0 else RE for v in bmr],
                        alpha=0.8, width=0.8, zorder=2)
                ax7.errorbar(gens, bmr, yerr=bsr, fmt='none',
                             color='white', alpha=0.4, capsize=2, lw=0.7)
                ax7.axhline(0, color='white', lw=0.8)
            title(ax7, "Best: avg return ± std")
            ax7.set_ylabel("Avg %", color='#888'); ax7.set_xlabel("Generation", color='#888')

            # ── ROW 5 Right: Win / Zero rate ──────────────────────────────────
            ax8 = fig.add_subplot(gs[5, 1])
            sty(ax8)
            if gens:
                ax8.plot(gens, wrs, color=CY,  lw=2,   label=f'Win≥{WIN_PROFIT_THRESH:.0f}%')
                ax8.plot(gens, zps, color=OR,  lw=1.5, ls='--', label='Zero %')
                ax8.axhline(50, color='white', lw=0.7, ls='--', alpha=0.5)
                ax8r = ax8.twinx()
                ax8r.plot(gens, tks, color=PU, lw=1.5, ls=':', label='Tournament K')
                ax8r.tick_params(colors='#888', labelsize=8)
                ax8r.set_ylabel("Tournament K", color='#888')
                ax8r.legend(fontsize=7, loc='upper right')
                ax8.legend(fontsize=7, loc='lower right')
            title(ax8, f"Win≥{WIN_PROFIT_THRESH:.0f}% / Zero periods / Tournament K")
            ax8.set_ylabel(f"Win≥{WIN_PROFIT_THRESH:.0f}% / Zero %", color='#888')
            ax8.set_xlabel("Generation", color='#888')

            # ── ROW 6 Left: Sigma ─────────────────────────────────────────────
            ax9 = fig.add_subplot(gs[6, 0])
            sty(ax9)
            if gens:
                ax9.plot(gens, sigs, color=PU, lw=2, zorder=2)
                ax9.axhline(SIGMA_MIN, color=OR, lw=0.8, ls=':', alpha=0.6)
                ax9.fill_between(gens, SIGMA_MIN, sigs, color=PU, alpha=0.10)
                for i in range(1, len(gens)):
                    if rsts[i] > rsts[i-1]:
                        ax9.axvline(gens[i], color=OR, lw=1.2, ls=':')
                        ax9.annotate('+σ', xy=(gens[i], sigs[i]),
                                     color=OR, fontsize=7)
            title(ax9, f"Sigma  (total restarts={self.restart_count})")
            ax9.set_ylabel("σ", color='#888'); ax9.set_xlabel("Generation", color='#888')

            # ── ROW 6 Right: Per-island fitness evolution ──────────────────────
            ax10 = fig.add_subplot(gs[6, 1])
            sty(ax10)
            _all_gens_for_islands = [h['gen'] for h in _all_hist]
            if _all_gens_for_islands:
                for r in REGIME_ORDER:
                    vals = island_fit_hist[r]
                    gx = [_all_gens_for_islands[i] for i, v in enumerate(vals) if v is not None]
                    vy = [v for v in vals if v is not None]
                    if gx:
                        _prev_last_gen = prev_gens[-1] if prev_gens else -1
                        prev_gx = [g for g in gx if g <= _prev_last_gen]
                        prev_vy = vy[:len(prev_gx)]
                        new_gx  = gx[len(prev_gx):]
                        new_vy  = vy[len(prev_gx):]
                        # Предыдущая сессия — только последняя точка как маркер
                        if prev_gx:
                            ax10.scatter([prev_gx[-1]], [prev_vy[-1]],
                                         color=REGIME_COLORS.get(r, CY),
                                         s=25, alpha=0.5, marker='D', zorder=2)
                        if new_gx:
                            ax10.plot(new_gx, new_vy,
                                      color=REGIME_COLORS.get(r, CY),
                                      lw=1.8, label=r, alpha=0.9,
                                      marker='o', markersize=2)
                ax10.axhline(0, color='white', lw=0.6, ls='--', alpha=0.4)
                ax10.legend(fontsize=6.5, loc='lower right', ncol=2,
                            framealpha=0.3, edgecolor=G)
            title(ax10, f"Island best fitness by regime  ({N_ISLANDS} islands: bearish/neutral/bullish)")
            ax10.set_ylabel("Best Fitness", color='#888')
            ax10.set_xlabel("Generation", color='#888')

            # ── Inset: NES + island diversity in a shared compact space ────────
            # These go as annotations on ax9 secondary axis to save rows
            if any(v > 0 for v in nes_g):
                ax9b = ax9.twinx()
                ax9b.plot(gens, nes_g, color=TE, lw=1.2, ls='--',
                          alpha=0.6, label='NES |∇|')
                ax9b.axhline(0.30, color=OR, lw=0.8, ls=':', alpha=0.7,
                             label='collapse threshold (0.30)')
                ax9b.tick_params(colors='#888', labelsize=7)
                ax9b.set_ylabel("|∇| NES", color='#888', fontsize=7)
                ax9b.legend(fontsize=6.5, loc='upper right')

            out = os.path.join(AGENTS_DIR, "training_progress.png")
            plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=D)
            plt.close(fig)
            print(f"  Chart: {out}")
        except Exception as e:
            import traceback
            print(f"  [chart error: {e}]")
            traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# АГЕНТ ДЛЯ RETRODATE_CRYPTOTRADE/CRYPTO_EXCHANGE.PY
# ══════════════════════════════════════════════════════════════════════════════

class GeneticsAgent:
    """
    Нейросетевой агент, обученный генетическим алгоритмом.
    Используется в Retrodate_cryptotrade/crypto_exchange.py как backtest-агент.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  АРХИТЕКТУРА: 28→128→64→32→9  (ELU)  — 14,345 параметров           │
    ├──────────────────────────────────────────────────────────────────────┤
    │  ВХОДНЫЕ ФИЧИ (28)                                                   │
    │  0:    mom_1h      (1-барный моментум, нормирован [-1,1])           │
    │  1:    mom_24h                                                       │
    │  2:    mom_72h                                                       │
    │  3:    mom_168h                                                      │
    │  4:    rsi         (RSI-50)/50                                       │
    │  5:    ema_fast    (EMA12/EMA48 - 1)                                │
    │  6:    ema_slow    (EMA48/EMA168 - 1)                               │
    │  7:    vol_ratio   (объём / средний объём - 1)                      │
    │  8:    bb_score    (позиция в Bollinger Bands)                      │
    │  9-11: regime_onehot [bearish, neutral, bullish]                    │
    │  12:   spot_pnl    ← live-only (в тренере = 0)                     │
    │  13:   fut_pnl     ← live-only (в тренере = 0)                     │
    │  14:   pos_size    ← live-only (в тренере = 0)                     │
    │  15-18: технические (price noise, global vol, EMA breadth, ...)    │
    │  19:   sin_month   (сезонность)                                     │
    │  20:   cos_month                                                     │
    │  21:   regime_conf (уверенность детектора режима)                  │
    │  22:   atr_rank    (ATR rank [-1,1])                                │
    │  23:   mom_eff     (mom/ATR efficiency)                             │
    │  24:   vol_trend   (объёмный тренд)                                 │
    │  25:   (резерв)                                                     │
    │  26:   cross_mom_rank  (cross-sectional momentum rank)             │
    │  27:   cross_vol_rank  (cross-sectional volume rank)               │
    ├──────────────────────────────────────────────────────────────────────┤
    │  ДЕЙСТВИЯ (9-action scheme)                                          │
    │  0=hold  1=buy_half  2=buy_full  3=sell_spot                        │
    │  4=fl_half  5=fl_full  6=fs_half  7=fs_full  8=close_fut            │
    ├──────────────────────────────────────────────────────────────────────┤
    │  ПОЗИЦИЯ (инференс)                                                  │
    │  spot_qty/spot_entry — спот с усреднением                           │
    │  fut_qty/fut_entry   — фьючерс с усреднением                        │
    │  pos[s] — legacy: None/'spot'/'fut_l'/'fut_s'/'spot+fut'           │
    ├──────────────────────────────────────────────────────────────────────┤
    │  СИНХРОНИЗАЦИЯ                                                       │
    │  update_from_exchange(s, sq, se, fq, fe) — обновляет P&L фичи      │
    │  _GeneticsAdapter.sync_from_exchange() вызывается перед каждым BAR │
    └──────────────────────────────────────────────────────────────────────┘
    """
    MAX_POS = TRAIN_MAX_POS

    def __init__(self, genome=None):
        if genome is None:
            p = os.path.join(AGENTS_DIR, "best_genome.npy")
            genome = np.load(p).astype(np.float32) if os.path.exists(p) else \
                     np.random.default_rng(SEED).normal(0, .3, GENOME_SIZE).astype(np.float32)
        genome = np.asarray(genome, dtype=np.float32)
        if len(genome) != GENOME_SIZE:
            print(f"  [GeneticsAgent] ⚠ геном {len(genome)} параметров, ожидается {GENOME_SIZE}.")
            if len(genome) < GENOME_SIZE:
                # Определяем N_INPUT старого генома из его размера.
                # Все слои кроме W1 фиксированы: b1+W2+b2+W3+b3+W4+b4 = GENOME_SIZE - _W1s
                _fixed_tail = GENOME_SIZE - _W1s   # всё после W1 = неизменно
                _old_W1s    = len(genome) - _fixed_tail   # W1 старого генома
                _delta      = GENOME_SIZE - len(genome)   # сколько нужно добавить
                if _old_W1s > 0 and _delta > 0:
                    # КРИТИЧНО: нули вставляем ПОСЛЕ последней строки старого W1,
                    # а НЕ в конец всего генома. Так b1, W2, b2, W3, ... остаются
                    # на своих позициях и не перемешиваются с нулями.
                    #   [W1_old (4992) | zeros (384) | b1 | W2 | b2 | W3 | b3 | W4 | b4]
                    #                   ↑ вставить сюда
                    padded         = np.zeros(GENOME_SIZE, dtype=np.float32)
                    padded[:_old_W1s]         = genome[:_old_W1s]   # W1 старых фич 0..N_old-1
                    padded[_W1s:]             = genome[_old_W1s:]   # b1,W2,b2,... без смещения
                    # padded[_old_W1s:_W1s]  = 0.0  (уже нули — новые строки W1 для фич 26,27)
                    old_n = _old_W1s // N_HIDDEN1
                    print(f"  [GeneticsAgent] Геном от старой архитектуры (N_INPUT={old_n} → теперь N_INPUT={N_INPUT}).")
                    print(f"  [GeneticsAgent] Вставлено {_delta} нулей в W1 (позиции {_old_W1s}..{_W1s-1}).")
                    print(f"  [GeneticsAgent] b1, W2, W3, W4 — сохранены корректно.")
                    print(f"  [GeneticsAgent] ⚠ РЕКОМЕНДУЕТСЯ ПЕРЕОБУЧЕНИЕ для использования фич {old_n}..{N_INPUT-1}.")
                    genome = padded
                else:
                    print(f"  [GeneticsAgent] Неожиданный размер генома — random init.")
                    genome = np.random.default_rng(SEED).normal(0, .3, GENOME_SIZE).astype(np.float32)
            else:
                print(f"  [GeneticsAgent] Геном слишком большой ({len(genome)}), обрезаем до {GENOME_SIZE}.")
                genome = genome[:GENOME_SIZE]
        self.genome = genome
        self._w     = _unpack(self.genome)
        self.ph: Dict[str, deque] = {}
        self.vh: Dict[str, deque] = {}
        # Расширенное отслеживание позиций: спот и фьючерс независимы
        self.spot_qty:   Dict[str, float] = {}   # qty монеты в споте (0 = нет позиции)
        self.spot_entry: Dict[str, float] = {}   # средняя цена входа в спот
        self.fut_qty:    Dict[str, float] = {}   # qty фьючерса (>0 лонг, <0 шорт, 0 нет)
        self.fut_entry:  Dict[str, float] = {}   # средняя цена входа в фьючерс
        # Обратная совместимость: self.pos используется некоторыми агентами-обёртками
        self.pos: Dict[str, Optional[str]] = {}
        self.t = 0; self._regime = 'unknown'; self._breadth = 0.5; self._regime_conf = 0.0
        self._ema_breadth = 0.5   # EMA-сглаженный брэдт (feat 17), как в precompute_features

        # Метаданные печатаем только один раз во всём процессе (не 4x при создании игроков)
        global _GENETICS_AGENT_REPORTED
        if not _GENETICS_AGENT_REPORTED:
            try:
                import json as _json
                _meta_path = os.path.join(AGENTS_DIR, "best_genome_meta.json")
                if os.path.exists(_meta_path):
                    with open(_meta_path) as _mf:
                        _meta = _json.load(_mf)
                    _gen  = int(_meta.get('gen', 0))
                    _fit  = float(_meta.get('fitness', 0))
                    _mpos = int(_meta.get('max_pos', TRAIN_MAX_POS))
                    if _gen < 30:
                        print(f"  [GeneticsAgent] ⚠ ВНИМАНИЕ: геном обучен только {_gen} "
                              f"ген. из рекомендуемых 100+. Бэктест будет ненадёжным! "
                              f"(fitness={_fit:+.4f})")
                    else:
                        print(f"  [GeneticsAgent] геном gen={_gen}  fitness={_fit:+.4f}  "
                              f"max_pos={_mpos}")
                    if _mpos != self.MAX_POS:
                        print(f"  [GeneticsAgent] MAX_POS: {self.MAX_POS} → {_mpos} (из метаданных)")
                    _GENETICS_AGENT_REPORTED = True
            except Exception:
                pass
        # Всегда применяем MAX_POS из метаданных, даже если уже сообщали
        try:
            import json as _json
            _meta_path = os.path.join(AGENTS_DIR, "best_genome_meta.json")
            if os.path.exists(_meta_path):
                with open(_meta_path) as _mf:
                    _meta = _json.load(_mf)
                _mpos = int(_meta.get('max_pos', TRAIN_MAX_POS))
                if _mpos != self.MAX_POS:
                    self.MAX_POS = _mpos
        except Exception:
            pass

    def _upd(self, month):
        BAR = _cx.BAR
        # Мульти-таймфреймовый тренд-брэдт: % активов с позитивным composite
        brd_count = 0; brd_total = 0
        for s, h in self.ph.items():
            hl = list(h); n = len(hl)
            if n < 168 * BAR + 1:
                continue
            p_cur  = hl[-1] + 1e-9
            m168_v = float(np.clip((p_cur - hl[-168*BAR]) / (hl[-168*BAR] + 1e-9), -1, 1))
            lb336  = min(336 * BAR, n - 1)
            lb672  = min(672 * BAR, n - 1)
            m336_v = float(np.clip((p_cur - hl[-lb336]) / (hl[-lb336] + 1e-9), -1, 1)) \
                     if lb336 >= 168 * BAR else m168_v
            m672_v = float(np.clip((p_cur - hl[-lb672]) / (hl[-lb672] + 1e-9), -1, 1)) \
                     if lb672 >= 336 * BAR else m336_v
            composite = 0.50 * m168_v + 0.30 * m336_v + 0.20 * m672_v
            brd_total += 1
            if composite > 0.0:
                brd_count += 1
        self._breadth = brd_count / brd_total if brd_total else 0.5
        # FIX BUG3: в precompute_features feat[17] = EMA-сглаженный брэдт (строки 704-710).
        # Период EMA = max(12*BAR, 12) баров. _upd() вызывается раз в BAR баров.
        _ba = float(2.0 / (max(12 * BAR, 12) + 1))
        self._ema_breadth = _ba * self._breadth + (1.0 - _ba) * self._ema_breadth
        try: self._regime = _cx._detect_regime_live(self.ph, current_month=month)
        except Exception: pass
        # Обновляем confidence
        self._regime_conf = self._compute_regime_conf()

    def _compute_regime_conf(self) -> float:
        """ Confidence: насколько чётко выражен текущий режим.
        Возвращает расстояние до ближайшего порога классификации, [0,1].
        """
        BAR = _cx.BAR
        scores = []
        for s, h in self.ph.items():
            hl = list(h); n = len(hl)
            if n < 168 * BAR + 1:
                continue
            p_cur  = hl[-1] + 1e-9
            m168_v = float(np.clip((p_cur - hl[-168*BAR]) / (hl[-168*BAR] + 1e-9), -1, 1))
            lb336  = min(336 * BAR, n - 1)
            lb672  = min(672 * BAR, n - 1)
            m336_v = float(np.clip((p_cur - hl[-lb336]) / (hl[-lb336] + 1e-9), -1, 1)) \
                     if lb336 >= 168 * BAR else m168_v
            m672_v = float(np.clip((p_cur - hl[-lb672]) / (hl[-lb672] + 1e-9), -1, 1)) \
                     if lb672 >= 336 * BAR else m336_v
            scores.append(0.50 * m168_v + 0.30 * m336_v + 0.20 * m672_v)
        if not scores:
            return 0.0
        cs     = float(np.mean(scores))
        br     = self._ema_breadth   # FIX: EMA-сглаженный брэдт как в precompute_features
        signal = cs + (br - 0.5) * 0.06
        thresholds = np.array([-0.14, -0.07, -0.025, -0.008, 0.015, 0.07])
        conf   = float(np.min(np.abs(signal - thresholds))) / 0.05
        return float(np.clip(conf, 0.0, 1.0))

    def act(self, prices, volumes, month=1, portfolio_value=None):
        BAR = _cx.BAR
        for s, p in prices.items():
            if s not in self.ph:
                self.ph[s] = deque(maxlen=700*BAR)
                self.vh[s] = deque(maxlen=300*BAR)
                self.spot_qty[s]   = 0.0
                self.spot_entry[s] = 0.0
                self.fut_qty[s]    = 0.0
                self.fut_entry[s]  = 0.0
                self.pos[s] = None   # обратная совместимость
            self.ph[s].append(p); self.vh[s].append(volumes.get(s, 0))
        self.t += 1
        if self.t % BAR == 0: self._upd(month)
        if all(len(h) < 25 for h in self.ph.values()): return {s: 0 for s in prices}

        syms = sorted(prices.keys()); NC = len(syms)
        if NC == 0: return {}

        x  = np.zeros((NC, N_INPUT), dtype=np.float32)
        ms = float(np.sin(2*np.pi*month/12)); mc = float(np.cos(2*np.pi*month/12))
        rv = np.zeros(3, dtype=np.float32)
        _r3 = map_regime_3(self._regime)
        if _r3 in REGIME_ORDER: rv[REGIME_ORDER.index(_r3)] = 1.0

        all_atr = []

        for i, s in enumerate(syms):
            h = list(self.ph[s])
            if len(h) < 25: continue
            p_cur = h[-1]
            def m(lb):
                lb = min(lb, len(h)-1)
                b = h[-lb]
                return float(np.clip((p_cur-b)/(b+1e-9),-1,1)) if b>0 else 0.0

            x[i,0]=m(BAR); x[i,1]=m(24*BAR)
            x[i,2]=m(72*BAR)  if len(h)>=72*BAR+1  else 0.0
            x[i,3]=m(168*BAR) if len(h)>=168*BAR+1 else 0.0
            x[i,4]=float(np.clip((_cx.rsi(h,14)-50)/50,-1,1))
            tl=h[-min(300*BAR,len(h)):]; e12=_cx.ema(tl,12); e48=_cx.ema(tl,48)
            e168=_cx.ema(tl,168) if len(tl)>=168*5 else e48
            x[i,5]=float(np.clip((e12-e48)/(abs(e48)+1e-9),-1,1))
            x[i,6]=float(np.clip((e48-e168)/(abs(e168)+1e-9),-1,1))
            vh=list(self.vh[s])
            # FIX BUG2: окно усреднения объёма = 24*BAR баров (= 24 часа),
            # как в precompute_features (_win_v = min(24*BAR, T)).
            # Было vh[-24:] → 24 минуты при 1m-таймфрейме. Теперь vh[-24*BAR:].
            _vol_win = min(24*BAR, len(vh))
            if len(vh)>=25: x[i,7]=float(np.clip(vh[-1]/(np.mean(vh[-_vol_win:])+1e-9)-1,-3,3))
            if len(h)>=24*BAR:
                w=h[-24*BAR:]; mn,mx=min(w),max(w)
                x[i,8]=float((p_cur-mn)/(mx-mn+1e-9))
            else: x[i,8]=0.5
            x[i,9:12]=rv; x[i,17]=self._ema_breadth; x[i,19]=ms; x[i,20]=mc
            x[i,21]=self._regime_conf   # confidence вместо нуля

            # ── Позиционные фичи 12-14 (live-only, в тренере = 0) ────────────
            # feat 12: спот P&L нормированный [-1, 1]  (0 = нет позиции)
            # feat 13: фьючерс P&L нормированный [-1, 1] (0 = нет позиции)
            # feat 14: суммарный размер позиции / TRADE_FRACTION (сколько раз усреднили)
            p_cur_f = float(p_cur)
            _sq = self.spot_qty.get(s, 0.0)
            _se = self.spot_entry.get(s, 0.0)
            if _sq > 0.0 and _se > 0.0:
                x[i, 12] = float(np.clip((_sq * p_cur_f / (_sq * _se) - 1.0) * 5.0, -1, 1))
            _fq = self.fut_qty.get(s, 0.0)
            _fe = self.fut_entry.get(s, 0.0)
            if _fq != 0.0 and _fe > 0.0:
                x[i, 13] = float(np.clip(_fq * (p_cur_f - _fe) / (_fe * 0.10 + 1e-9), -1, 1))
            _pv_ref = float(portfolio_value) if portfolio_value else 1.0
            _tf = float(getattr(_cx, 'TRADE_FRACTION', 0.15))
            _pos_val = (_sq * p_cur_f + abs(_fq) * _fe / getattr(_cx, 'LEVERAGE', 2.0)) \
                       if (_sq > 0 or _fq != 0) else 0.0
            x[i, 14] = float(np.clip(_pos_val / (max(_pv_ref, 1.0) * _tf + 1e-9) / 3.0, 0, 1))

            if len(h) >= 14*BAR:
                w_atr = h[-14*BAR:]; hi_a=max(w_atr); lo_a=min(w_atr)
                mid_a=(hi_a+lo_a)/2+1e-9
                atr_val = (hi_a-lo_a)/mid_a
                all_atr.append((i, atr_val))
            else:
                all_atr.append((i, 0.0))

            if len(vh) >= 48:
                v_arr = np.array(vh[-min(500*BAR, len(vh)):], dtype=np.float64)
                # FIX BUG1 (КРИТИЧЕСКИЙ): периоды EMA должны быть 12*BAR и 48*BAR баров,
                # идентично precompute_features (строка 812):
                #   a12 = 2.0 / (12*BAR + 1)
                #   a48 = 2.0 / (48*BAR + 1)
                # Было 2.0/(12+1) и 2.0/(48+1) — при BAR=60 это в 60 раз быстрее,
                # нейросеть видела совершенно другой сигнал → случайные действия.
                a12_ = 2.0/(12*BAR+1); a48_ = 2.0/(48*BAR+1)
                e12v = v_arr[0]; e48v = v_arr[0]
                for vv in v_arr[1:]:
                    e12v = a12_*vv + (1-a12_)*e12v
                    e48v = a48_*vv + (1-a48_)*e48v
                x[i,24] = float(np.clip((e12v/(e48v+1e-9) - 1.0) / 2.0, -1, 1))

            if len(h) >= 7*24*BAR:
                w7 = h[-7*24*BAR:]
                mn7, mx7 = min(w7), max(w7)
                x[i,25] = float(np.clip((p_cur-mn7)/(mx7-mn7+1e-9), 0, 1))
            else:
                x[i,25] = 0.5

        if all_atr:
            atr_vals = np.array([v for _, v in all_atr])
            ranks    = np.argsort(np.argsort(atr_vals)).astype(np.float32) / (len(all_atr)-1+1e-9)
            atr_mean = atr_vals.mean() + 1e-9
            for k, (i, atr_v) in enumerate(all_atr):
                x[i,22] = float(np.clip(ranks[k]*2-1, -1, 1))
                mom_24 = float(x[i,1])
                atr_rel = float(np.clip(atr_v / atr_mean, 0.01, 10.0))
                x[i,23] = float(np.clip(mom_24 / atr_rel / 2.0, -1, 1))

        # ── Feature 18 (global_vol) и Feature 16 (noise) ────────────────────
        # Эти признаки вычислялись в precompute_features() при обучении,
        # но отсутствовали в GeneticsAgent.act() → сеть видела нули вместо
        # реальных значений → нарушение train/inference = нет торговли.
        #
        # Feature 18: стандартное отклонение доходностей за последние 50 баров
        #             нормировано на 2% (референс волатильность крипты), clip [0,5].
        # Feature 16: доля "аномальных" доходностей (|r| > 3σ) = рыночный шум,
        #             нормировано так же, как в precompute_features, clip [0,1].
        _live_rets = []
        for _s, _h in self.ph.items():
            _hl = list(_h)
            _n  = min(50, len(_hl))
            if _n >= 2:
                _p_arr = np.array(_hl[-_n:], dtype=np.float64)
                _r     = np.diff(_p_arr) / (_p_arr[:-1] + 1e-9)
                _live_rets.extend(_r[np.isfinite(_r)].tolist())
        if _live_rets:
            _gv = float(np.std(_live_rets)) / 0.02
            _global_vol = float(min(_gv, 5.0))
        else:
            _global_vol = 0.0
        x[:, 18] = np.float32(_global_vol)   # FIX: был всегда 0.0

        _noise_rets = []
        for _s, _h in list(self.ph.items())[:10]:   # до 10 монет, как в precompute
            _hl = list(_h)
            _n  = min(1000, len(_hl))
            if _n >= 2:
                _p_arr = np.array(_hl[-_n:], dtype=np.float64)
                _r     = np.diff(_p_arr) / (_p_arr[:-1] + 1e-9)
                _noise_rets.extend(_r[np.isfinite(_r)].tolist())
        if len(_noise_rets) > 10:
            _r_arr   = np.array(_noise_rets)
            _std_r   = float(np.std(_r_arr)) + 1e-9
            _noise_f = float((np.abs(_r_arr) > 3 * _std_r).mean()) / 0.05 * 4.0 / 10.0
            x[:, 16] = np.float32(min(_noise_f, 1.0))  # FIX: был всегда 0.0

        # ── FIX: Feature 26 (cross-sectional momentum rank) ──────────────────
        # Feature 26 и 27 добавлены в v11 для обучения нейросети выбору монет.
        # КРИТИЧНО: должны вычисляться ИДЕНТИЧНО precompute_features() —
        # иначе train/inference расхождение → сеть видит другое распределение.
        #
        # Feature 26: ранг монеты по 24h momentum среди всех NC монет [0,1]
        #   0 = монета с наихудшим моментумом, 1 = монета с наилучшим
        # Feature 27: ранг монеты по volume EMA trend среди всех NC монет [0,1]
        if NC > 1 and N_INPUT >= 27:
            m24h_live = x[:, 1].copy()   # (NC,) — уже заполнен выше
            _cs_ranks26 = np.argsort(np.argsort(m24h_live)).astype(np.float32)
            x[:, 26] = _cs_ranks26 / (NC - 1 + 1e-9)

        if NC > 1 and N_INPUT >= 28:
            vol_tr_live = x[:, 24].copy()  # (NC,) — volume EMA trend
            _cs_ranks27 = np.argsort(np.argsort(vol_tr_live)).astype(np.float32)
            x[:, 27] = _cs_ranks27 / (NC - 1 + 1e-9)

        W1,b1,W2,b2,W3,b3,W4,b4 = self._w
        raw  = _fwd_np(x, W1, b1, W2, b2, W3, b3, W4, b4).argmax(axis=1)
        acts = {s: 0 for s in prices}

        # ── Выбор оптимальных валют (live-режим) ─────────────────────────────
        # При CURRENCY_LEARNING_ENABLED=True: нейросеть сама выбирает монеты через
        # cross-sectional rank фичи 26-27. Hard mask не применяется.
        #
        # При CURRENCY_LEARNING_ENABLED=False (обратная совместимость):
        # применяем rule-based скоринг + hard mask топ-N.
        #
        # ВАЖНО: GeneticsSymbolFilter в crypto_agents.py применяет ДОПОЛНИТЕЛЬНЫЙ
        # фильтр (TREND/LIQUID теги) поверх этого в live-режиме. Это намеренно:
        # нейросеть выбирает монеты по данным, теговый фильтр — по поведению.
        # При CURRENCY_LEARNING_ENABLED=True двойного ограничения нет: нейросеть
        # уже обучена учитывать ликвидность/трендовость через признаки 22-27.
        scores = None
        if CURRENCY_SELECTION_ENABLED and not CURRENCY_LEARNING_ENABLED and len(syms) > 1:
            feat_1bar   = x[np.newaxis, :, :]                                  # (1, NC, NF)
            scores_1bar = _compute_currency_scores(feat_1bar, self._regime)     # (1, NC)
            scores      = scores_1bar[0]                                        # (NC,)
            n_top = _get_top_n_for_regime(len(syms), self._regime, TOP_CURRENCIES_N)
            if n_top < len(syms):
                top_set = set(np.argsort(scores)[-n_top:].tolist())
                for i in range(len(syms)):
                    # FIX CRITICAL: entry-действия = 1,2 (buy_half/full), 4,5 (fl_half/full),
                    # 6,7 (fs_half/full) — все открытия позиций в 9-action training scheme.
                    if i not in top_set and int(raw[i]) in (1, 2, 4, 5, 6, 7):
                        raw[i] = 0  # → hold

        # n_pos = кол-во монет с хотя бы одной позицией (спот ИЛИ фут)
        n = sum(1 for s in syms if self.spot_qty.get(s,0.0)>0 or self.fut_qty.get(s,0.0)!=0)

        # ── 9-action scheme (ИДЕНТИЧЕН _sim_core тренера) ──────────────────
        # 0=hold  1=buy_half  2=buy_full  3=sell_spot
        # 4=fl_half  5=fl_full  6=fs_half  7=fs_full  8=close_fut
        #
        # FIX CRITICAL: предыдущая версия использовала ДРУГУЮ нумерацию
        # (1=buy, 2=sell, 3=fl, 4=fs, 5=close) — полное несовпадение с тренером.
        # Нейросеть, обученная выдавать action=2 (buy_full), попадала в sell_spot!
        # Теперь act() интерпретирует raw 0-8 ТОЧНО как _sim_core.
        #
        # Маппинг на внешние action codes (для биржи):
        #   raw 1,2 → acts[s] = 1 (buy_spot)    — биржа сама выбирает размер
        #   raw 3   → acts[s] = 2 (sell_spot)
        #   raw 4,5 → acts[s] = 3 (fut_long)    — биржа сама выбирает размер
        #   raw 6,7 → acts[s] = 4 (fut_short)   — биржа сама выбирает размер
        #   raw 8   → acts[s] = 5 (close_fut)

        for i, s in enumerate(syms):
            a   = int(raw[i])
            p_s = float(prices[s])
            sq  = self.spot_qty.get(s, 0.0)
            fq  = self.fut_qty.get(s, 0.0)
            had_any = (sq > 0 or fq != 0)

            if a in (1, 2):
                # buy_spot (half/full): усреднение разрешено
                if sq <= 0 and n >= self.MAX_POS:
                    continue   # нет места для новой монеты
                acts[s] = 1
                # Отслеживаем средний entry (приблизительно — точный ts биржа знает)
                if sq > 0 and self.spot_entry.get(s, 0.0) > 0:
                    pass  # entry обновится после ответа биржи (см. update_from_exchange)
                else:
                    self.spot_entry[s] = p_s
                if not had_any:
                    n += 1

            elif a == 3 and sq > 0:
                # sell_spot
                acts[s] = 2
                self.spot_qty[s]   = 0.0
                self.spot_entry[s] = 0.0
                if fq == 0: n -= 1

            elif a in (4, 5):
                # fut_long (half/full): усреднение в лонг, конфликт с шортом запрещён
                if fq < 0:
                    continue   # уже шорт
                if fq == 0 and sq == 0 and n >= self.MAX_POS:
                    continue
                acts[s] = 3
                if fq == 0:
                    self.fut_entry[s] = p_s
                    if sq == 0: n += 1
                # entry обновится после update_from_exchange

            elif a in (6, 7):
                # fut_short (half/full): усреднение в шорт, конфликт с лонгом запрещён
                if fq > 0:
                    continue   # уже лонг
                if fq == 0 and sq == 0 and n >= self.MAX_POS:
                    continue
                acts[s] = 4
                if fq == 0:
                    self.fut_entry[s] = p_s
                    if sq == 0: n += 1

            elif a == 8 and fq != 0:
                # close_fut
                acts[s] = 5
                self.fut_qty[s]   = 0.0
                self.fut_entry[s] = 0.0
                if sq == 0: n -= 1

        # ── FIX: swap-оптимизация ОТКЛЮЧЕНА в live-режиме ────────────────────
        # action=6 в ТРЕНЕРЕ = swap_buy; в БИРЖЕ = close_all → конфликт.
        # GeneticsAgent никогда не выставляет action=6 в live.

        # ── Обратная совместимость: синхронизируем self.pos ──────────────────
        for s in syms:
            sq = self.spot_qty.get(s, 0.0)
            fq = self.fut_qty.get(s, 0.0)
            if sq > 0 and fq != 0:
                self.pos[s] = 'spot+fut'
            elif sq > 0:
                self.pos[s] = 'spot'
            elif fq > 0:
                self.pos[s] = 'fut_l'
            elif fq < 0:
                self.pos[s] = 'fut_s'
            else:
                self.pos[s] = None

        return acts

    def update_from_exchange(self, symbol: str, spot_qty: float, spot_entry: float,
                              fut_qty: float, fut_entry: float):
        """
        Синхронизирует внутреннее состояние агента с реальными данными биржи.
        Вызывается из crypto_players.py после каждого tick чтобы:
         - обновить средневзвешенный entry при усреднении
         - точно отслеживать qty для корректного расчёта P&L в фичах 12-13

        spot_qty   — реальный qty монеты на споте (0 если нет позиции)
        spot_entry — средняя цена входа в спот
        fut_qty    — qty фьючерса (>0 лонг, <0 шорт, 0 нет)
        fut_entry  — средняя цена входа в фьючерс
        """
        self.spot_qty[symbol]   = float(spot_qty)
        self.spot_entry[symbol] = float(spot_entry)
        self.fut_qty[symbol]    = float(fut_qty)
        self.fut_entry[symbol]  = float(fut_entry)
        # Синхронизируем pos для обратной совместимости
        if spot_qty > 0 and fut_qty != 0:
            self.pos[symbol] = 'spot+fut'
        elif spot_qty > 0:
            self.pos[symbol] = 'spot'
        elif fut_qty > 0:
            self.pos[symbol] = 'fut_l'
        elif fut_qty < 0:
            self.pos[symbol] = 'fut_s'
        else:
            self.pos[symbol] = None


# ══════════════════════════════════════════════════════════════════════════════
# РЕЖИМНО-СПЕЦИАЛИЗИРОВАННЫЕ АГЕНТЫ (Island Best Genomes)
# ══════════════════════════════════════════════════════════════════════════════
# Каждый из трёх агентов загружает лучший геном своего острова (bearish/neutral/
# bullish) и действует как полноценный GeneticsAgent с той же архитектурой.
#
# Ключевые отличия от GeneticsAgent (best_genome.npy):
#  1. Геном специализирован на конкретном режиме рынка — выше точность в нём.
#  2. Параллельная работа всех трёх позволяет ансамблю покрыть все режимы.
#  3. self._native_regime — профильный режим острова. Можно использовать
#     в crypto_players.py для режимно-условного взвешивания в ансамбле.
#
# Пример интеграции в ансамбле (crypto_players.py):
#   detected = exchange.get_regime()
#   weights  = {'GeneticsBearish': 0.6 if detected=='bearish' else 0.2,
#               'GeneticsNeutral': 0.6 if detected=='neutral' else 0.2,
#               'GeneticsBullish': 0.6 if detected=='bullish' else 0.2}
# ══════════════════════════════════════════════════════════════════════════════

class _RegimeGeneticsAgent(GeneticsAgent):
    """
    Базовый класс для режимно-специализированных агентов.
    Загружает геном из best_island_{regime}.npy (сохраняется тренером каждые 3 гена).
    Поведение идентично GeneticsAgent — только другой файл генома и метка режима.
    """
    _REGIME_LABEL: str = 'unknown'     # переопределяется в подклассах

    def __init__(self, genome=None):
        if genome is None:
            # Ищем геном острова по профильному режиму
            regime_file = island_file(REGIME_TO_ISLAND.get(self._REGIME_LABEL, 0))
            fallback    = os.path.join(AGENTS_DIR, "best_genome.npy")
            if os.path.exists(regime_file):
                genome = np.load(regime_file).astype(np.float32)
            elif os.path.exists(fallback):
                genome = np.load(fallback).astype(np.float32)
            else:
                genome = np.random.default_rng(SEED).normal(0, .3, GENOME_SIZE).astype(np.float32)
        super().__init__(genome=genome)
        # Профильный режим острова — используется внешними ансамблями
        self._native_regime: str = self._REGIME_LABEL
        # Загружаем и печатаем метаданные острова
        try:
            _meta_path = os.path.join(AGENTS_DIR,
                                      f"best_island_{self._REGIME_LABEL}_meta.json")
            if os.path.exists(_meta_path):
                with open(_meta_path) as _mf:
                    _meta = json.load(_mf)
                _gen  = int(_meta.get('gen', 0))
                _fit  = float(_meta.get('fitness', 0.0))
                _mpos = int(_meta.get('max_pos', TRAIN_MAX_POS))
                print(f"  [{type(self).__name__}] regime={self._REGIME_LABEL}  "
                      f"gen={_gen}  fitness={_fit:+.4f}  max_pos={_mpos}")
                if _mpos != self.MAX_POS:
                    self.MAX_POS = _mpos
        except Exception:
            pass

    @classmethod
    def genome_path(cls) -> str:
        """Путь к файлу генома этого агента на диске."""
        return island_file(REGIME_TO_ISLAND.get(cls._REGIME_LABEL, 0))

    @classmethod
    def is_available(cls) -> bool:
        """True если геном сохранён на диске (остров обучен)."""
        return os.path.exists(cls.genome_path())


class GeneticsBearishAgent(_RegimeGeneticsAgent):
    """
    Специалист медвежьего рынка (island 0).
    Загружает best_island_bearish.npy — лучший геном острова bearish.

    Обучен на периодах с avg_coin_return < -5%:
      • Умеет шортировать падающие монеты (fut_short)
      • Минимизирует просадку в crash-режиме
      • Быстро закрывает позиции при развороте

    Наиболее эффективен когда _regime ∈ ('bearish',).
    """
    _REGIME_LABEL = 'bearish'


class GeneticsNeutralAgent(_RegimeGeneticsAgent):
    """
    Специалист нейтрального/бокового рынка (island 1).
    Загружает best_island_neutral.npy — лучший геном острова neutral.

    Обучен на периодах с avg_coin_return ∈ [-5%, +5%):
      • Торгует range-bound движения (BB mean-reversion)
      • Аккумулирует позиции в боковике перед пробоем
      • Низкая активность → меньше комиссий в flatline

    Наиболее эффективен когда _regime ∈ ('neutral',).
    """
    _REGIME_LABEL = 'neutral'


class GeneticsBullishAgent(_RegimeGeneticsAgent):
    """
    Специалист бычьего рынка (island 2).
    Загружает best_island_bullish.npy — лучший геном острова bullish.

    Обучен на периодах с avg_coin_return ≥ +5%:
      • Агрессивно покупает лидеров роста (momentum following)
      • Удерживает позиции дольше (no premature sell)
      • Использует спот + фьючерс лонг для усиления

    Наиболее эффективен когда _regime ∈ ('bullish',).
    """
    _REGIME_LABEL = 'bullish'


# Удобный словарь: режим → класс агента (для динамического создания)
REGIME_AGENT_CLASSES: Dict[str, type] = {
    'bearish': GeneticsBearishAgent,
    'neutral': GeneticsNeutralAgent,
    'bullish': GeneticsBullishAgent,
}


def make_regime_agents() -> Dict[str, 'GeneticsAgent']:
    """
    Создаёт и возвращает все три режимных агента если их геномы доступны.

    Использование в crypto_agents.make_agents():
        from crypto_genetics import make_regime_agents
        regime_agents = make_regime_agents()
        for name, agent in regime_agents.items():
            raw[name] = agent

    Returns: dict {name: agent} только для доступных агентов
    """
    result: Dict[str, GeneticsAgent] = {}
    labels = {
        'GeneticsBearish': GeneticsBearishAgent,
        'GeneticsNeutral': GeneticsNeutralAgent,
        'GeneticsBullish': GeneticsBullishAgent,
    }
    for name, cls in labels.items():
        if cls.is_available():
            try:
                agent = cls()
                result[name] = agent
            except Exception as e:
                print(f"  [RegimeAgents] ⚠ {name} ошибка загрузки: {e}")
        else:
            print(f"  [RegimeAgents] {name}: геном не найден "
                  f"({os.path.basename(cls.genome_path())}) — пропуск. "
                  f"Запустите crypto_genetics.py.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_data_dir() -> str:
    data_dir = _cx.DATA_DIR
    if os.path.isabs(data_dir) and os.path.isdir(data_dir):
        return data_dir
    cx_dir  = os.path.dirname(os.path.abspath(_cx.__file__))
    resolved = os.path.join(cx_dir, data_dir)
    if os.path.isdir(resolved):
        return resolved
    if os.path.isdir(data_dir):
        return data_dir
    return resolved


def _load_precomp():
    """
    Загружает и предвычисляет признаки для всех периодов.
    ── Оптимизация загрузки v2 (п.1) ──────────────────────────────────────────
    Фаза A: каждый уникальный годовой CSV читается с диска ровно ОДИН раз
            (не 12 раз как раньше!). Экономия IO: до 10-12x на годовых данных.
    Фаза B: вычисление фич — параллельно, до 24 потоков (numpy releases GIL).
    ─────────────────────────────────────────────────────────────────────────
    """
    import re, concurrent.futures as _cfu, pandas as pd
    cfg = _cx._CFG; tf = {'1m':'1m','1h':'1h','1d':'1d'}.get(_cx.TIMEFRAME,'1m')
    def ym(s, fy, fm=1):
        mt = re.match(r'(\d{4})-(\d{1,2})', str(s))
        if mt: return int(mt.group(1)), int(mt.group(2))
        mt = re.match(r'(\d{4})', str(s))
        if mt: return int(mt.group(1)), fm
        return fy, fm
    sd = cfg.get('start_date',''); ed = cfg.get('end_date','')
    ys, ms = ym(sd, 2022, 1) if sd else (2022, 1)
    ye, me = ym(ed, 2025, 12) if ed else (2025, 12)
    print(f"  Period: {ys}-{ms:02d} -> {ye}-{me:02d}")
    data_dir = _resolve_data_dir()
    print(f"  Data dir: {data_dir}")

    # 1. Составляем список задач (year, month, fname, fpath)
    tasks = []
    for year in range(ys, ye+1):
        fname = f"crypto_{tf}_{year}_all_symbols.csv"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: no {fname}  (проверено: {fpath})"); continue
        for month in range(ms if year==ys else 1, me+1 if year==ye else 13):
            tasks.append((year, month, fname, fpath))

    print(f"  Всего периодов для загрузки: {len(tasks)}")
    # 2. Функция загрузки одного периода (запускается в потоке)
    def _load_one(task):
        year, month, fname, fpath = task
        last   = calendar.monthrange(year, month)[1]
        period = f"{year}-{month:02d}"
        try:
            # ИСПРАВЛЕНО: передаём полный абсолютный путь fpath напрямую.
            # Старый код мутировал _cx.DATA_DIR из нескольких потоков → гонка данных → зависание.
            # load_data(fpath, ...): filepath = os.path.join(DATA_DIR, fpath).
            # На Windows os.path.join(relative, C:\abs\path) → C:\abs\path (абс. путь побеждает).
            # Если os.path.join даст несуществующий путь, load_data fallback: filepath = fpath.
            dfp, dfv = _cx.load_data(fpath, f"{year}-{month:02d}-01",
                                     f"{year}-{month:02d}-{last}")
            if dfp is None or len(dfp) == 0:
                return None
            t0 = time.time()
            feat, prices, syms = precompute_features(dfp, dfv, month)
            if prices.shape[0] > 1:
                p0 = prices[0].astype(np.float64)
                p1 = prices[-1].astype(np.float64)
                valid = p0 > 0
                avg_ret_pct = float(np.mean((p1[valid] - p0[valid]) / p0[valid]) * 100.0) \
                              if valid.any() else 0.0
            else:
                avg_ret_pct = 0.0
            regime = classify_period_regime(avg_ret_pct)
            rw     = REGIME_WEIGHTS[regime]
            elapsed = time.time() - t0
            return (feat, prices, syms, month, period, rw, regime, avg_ret_pct, elapsed)
        except Exception as ex:
            return ('error', period, str(ex))

    # 3. Параллельное выполнение.
    # ── УСКОРЕНИЕ v2 (п.1): ──────────────────────────────────────────────────
    # Раньше: каждый из 96 потоков читал годовой CSV с диска (~12x повторных чтений
    #          одного файла для 12 месяцев года).
    # Теперь:
    #   Фаза A (IO-bound): читаем каждый уникальный CSV файл ОДИН РАЗ в память.
    #                       Параллельно через потоки (GIL освобождается в pandas IO).
    #   Фаза B (CPU-bound): вычисляем фичи каждого месяца из уже загруженных DataFrame.
    #                       Увеличили потоки: numpy releases GIL → хорошо масштабируется.
    # ─────────────────────────────────────────────────────────────────────────
    unique_fpaths = list(dict.fromkeys(fpath for _, _, _, fpath in tasks))
    csv_cache: dict = {}   # fpath -> (dfp_full, dfv_full) или None

    # Фаза A: параллельная загрузка уникальных CSV в память
    n_io = min(len(unique_fpaths), max(1, min(_mp.cpu_count(), 16)))
    print(f"  (фаза A: {len(unique_fpaths)} уникальных CSV, {n_io} IO-потоков | "
          f"фаза B: фичи {len(tasks)} периодов)")
    t_io_start = time.time()

    def _load_csv_raw(fpath_item):
        """Загружает полный годовой CSV в DataFrame (price + volume). GIL освобождается в pandas IO."""
        try:
            dfp, dfv = _cx.load_data(fpath_item, None, None)   # None → весь файл без фильтрации
            if dfp is None or len(dfp) == 0:
                return fpath_item, None
            return fpath_item, (dfp, dfv)
        except Exception as ex:
            return fpath_item, None

    with _cfu.ThreadPoolExecutor(max_workers=n_io) as pool:
        for fpath_item, result in pool.map(_load_csv_raw, unique_fpaths):
            csv_cache[fpath_item] = result
            print(f"\r  Загружено CSV: {len(csv_cache)}/{len(unique_fpaths)}", end='', flush=True)
    print(f"  ({time.time()-t_io_start:.1f}с)", flush=True)

    # Фаза B: параллельное вычисление фич по месяцам (CPU-bound, numpy)
    # Больше потоков — numpy/numba releases GIL → хорошо масштабируется на 32 ядрах.
    n_cpu = min(len(tasks), max(1, min(_mp.cpu_count(), 24)))
    print(f"  Фаза B: вычисление признаков {len(tasks)} периодов "
          f"({n_cpu} потоков)...", flush=True)

    def _load_one(task):
        year, month, fname, fpath = task
        last   = calendar.monthrange(year, month)[1]
        period = f"{year}-{month:02d}"
        try:
            raw = csv_cache.get(fpath)
            if raw is None:
                return ('error', period, f"CSV не загружен: {fpath}")
            dfp_full, dfv_full = raw
            # ── Фильтрация в памяти (без обращения к диску) ──────────────────
            date_start = pd.Timestamp(f"{year}-{month:02d}-01", tz='UTC')
            date_end   = pd.Timestamp(f"{year}-{month:02d}-{last}", tz='UTC') \
                         + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask = (dfp_full.index >= date_start) & (dfp_full.index <= date_end)
            dfp = dfp_full.loc[mask]
            dfv = dfv_full.loc[mask]
            if dfp is None or len(dfp) == 0:
                return None
            t0 = time.time()
            feat, prices, syms = precompute_features(dfp, dfv, month)
            if prices.shape[0] > 1:
                p0 = prices[0].astype(np.float64)
                p1 = prices[-1].astype(np.float64)
                valid = p0 > 0
                avg_ret_pct = float(np.mean((p1[valid] - p0[valid]) / p0[valid]) * 100.0) \
                              if valid.any() else 0.0
            else:
                avg_ret_pct = 0.0
            regime = classify_period_regime(avg_ret_pct)
            rw     = REGIME_WEIGHTS[regime]
            elapsed = time.time() - t0
            return (feat, prices, syms, month, period, rw, regime, avg_ret_pct, elapsed)
        except Exception as ex:
            return ('error', period, str(ex))

    results_raw = [None] * len(tasks)
    done_count  = [0]

    with _cfu.ThreadPoolExecutor(max_workers=n_cpu) as pool:
        fut_map = {pool.submit(_load_one, t): i for i, t in enumerate(tasks)}
        for fut in _cfu.as_completed(fut_map):
            idx = fut_map[fut]
            results_raw[idx] = fut.result()
            done_count[0] += 1
            print(f"\r  Признаки {done_count[0]}/{len(tasks)}...", end='', flush=True)
    print()

    # 4. Выводим результаты в порядке периодов и собираем precomp
    precomp = []
    for res in results_raw:
        if res is None:
            continue
        # FIX: res[0] == 'error' падал с ValueError если res[0] — numpy-массив feat.
        # Используем isinstance для надёжной проверки типа.
        if isinstance(res[0], str) and res[0] == 'error':
            print(f"  ERROR {res[1]}: {res[2]}")
            continue
        feat, prices, syms, month, period, rw, regime, avg_ret_pct, elapsed = res
        print(f"  Loading {period}... ok {feat.shape[0]:,}bars x {feat.shape[1]}coins  "
              f"regime={regime}({avg_ret_pct:+.1f}%)  w={rw}  {elapsed:.1f}s")
        precomp.append((feat, prices, syms, month, period, rw, regime))

    # 5. Статистика режимов
    regimes_seen = {}
    for entry in precomp:
        rw = entry[5]
        regime_name = map_regime_3(entry[6]) if len(entry) > 6 and isinstance(entry[6], str) else f'w={rw}'
        regimes_seen[regime_name] = regimes_seen.get(regime_name, 0) + 1
    print(f"\n   Распределение режимов: {regimes_seen}")
    return precomp


if __name__ == '__main__':
    _mp.freeze_support()

    print("="*70)
    print("  GENETICS AGENT TRAINER  v11 (3-islands + levy-flight + cosine-sigma + currency-learning)")
    print("="*70)
    print(f"  {platform.system()}  CPU: {_mp.cpu_count()} cores"
          f"  BAR={_cx.BAR}  Capital={_cx.INITIAL_CAPITAL:,.0f}")
    print()
    print("  CONFIG:")
    print(f"  Population:    {POP_SIZE}     Generations: {N_GENERATIONS}")
    print(f"  Genome:        {GENOME_SIZE}  float32")
    print(f"  Arch:          {N_INPUT}→{N_HIDDEN1}→{N_HIDDEN2}→{N_HIDDEN3}→{N_ACTIONS}  (ELU)")
    print(f"  Elite:         {ELITE_SIZE}     Tournament K: {TOURNAMENT_K}..{TOURNAMENT_K_MAX}")
    print(f"  Mutation:      p={MUTATION_RATE:.0%}  sigma=cosine({SIGMA_MIN}..{MUTATION_SIGMA})  T0={COSINE_T0}")
    print(f"  Lévy flight:   {'ON' if LEVY_ENABLED else 'OFF'}  frac={LEVY_FRAC:.0%}  α={LEVY_ALPHA}  scale={LEVY_SCALE}")
    print(f"  Per-layer σ:   W1×{LAYER_SIGMA_MULT[0]} / W2×{LAYER_SIGMA_MULT[1]} / W3×{LAYER_SIGMA_MULT[2]}")
    _p1g = int(N_GENERATIONS * PHASE1_GENS_FRAC) if PHASE1_ENABLED else 0
    print(f"  Islands:       {N_ISLANDS} (crash/bear/bull)  migration every {MIGRATION_INTERVAL} gens")
    for _iid, _rgs in enumerate(ISLAND_REGIME_GROUPS):
        _fname = os.path.basename(island_file(_iid))
        _rgs_str = '+'.join(_rgs)
        print(f"    island {_iid}: [{_rgs_str:>30s}]  → {_fname}")
    print(f"  Phase 1:       {'ENABLED' if PHASE1_ENABLED else 'DISABLED'}  "
          f"gens=1..{_p1g} ({PHASE1_GENS_FRAC*100:.0f}%)  [FULL data + island weights — no cliff!]")
    print(f"  Currency:      {'LEARNING (network learns coin selection via cross-sec. rank features 26-27)' if CURRENCY_LEARNING_ENABLED else 'RULE-BASED (hard mask)'}")
    print(f"  NES:           {NES_FRAC*100:.0f}% samples  lr={NES_LR}")
    print(f"  DE:            {DE_FRACTION*100:.0f}% offspring  F={DE_F}")
    print(f"  CMA memory:    lr={CMA_MEMORY_LR}  blend={CMA_BLEND}")
    print(f"  Diversity:     every {DIVERSITY_INTERVAL} gens  "
          f"force after {MAX_RESTARTS_NO_IMPROVE} restarts no improve")
    print(f"  Snapshot:      every {SNAPSHOT_BARS} bars (daily fitness)")
    print(f"  Actions:       0=hold  1=buy_half(50%)  2=buy_full(100%)  3=sell_spot")
    print(f"                 4=fl_half  5=fl_full  6=fs_half  7=fs_full  8=close_fut")
    print(f"  Pos rules:     spot+fut independent | averaging allowed | no dir-conflict")
    print(f"  Input feats:   0-25=original  26=cross_mom_rank  27=cross_vol_rank")
    print(f"  Fitness:       mean*{FITNESS_W_MEAN} + calmar*{FITNESS_W_CALMAR} "
          f"+ sharpe*{FITNESS_W_SHARPE} + daily_sharpe*{FITNESS_W_DAILY_SHARPE} "
          f"+ daily_wr*{FITNESS_W_DAILY_WINRATE} + consistency*{FITNESS_W_CONSISTENCY} "
          f"+ win≥{WIN1_PROFIT_THRESH:.0f}%*{WIN1_FITNESS_WEIGHT} "
          f"+ win≥{WIN_PROFIT_THRESH:.0f}%*{FITNESS_W_WIN_5PCT} "
          f"+ win≥{BONUS_PROFIT_THRESH:.0f}%*{FITNESS_W_BONUS_20}")
    _bc_status = f"ON  agents={'auto' if not AGENT_SEED_LIST else len(AGENT_SEED_LIST)}  epochs={BC_EPOCHS}" if BC_ENABLED else "OFF  (включить: bc_enabled=on в settings_genetic.txt)"
    print(f"  BC:            {_bc_status}")
    _ws = []
    if CONTINUE_TRAINING:    _ws.append("population")
    if LOAD_BEST_GENOME:     _ws.append("best_genome")
    if LOAD_EXTRA_GENOMES:   _ws.append("extra_npy")
    if LOAD_REGIME_GENOMES:  _ws.append("regime_genomes")
    if LOAD_ISLAND_GENOMES:  _ws.append("island_genomes")
    print(f"  Warm-start:    {', '.join(_ws) if _ws else 'DISABLED'}")
    print(f"  Archive:       top-{ARCHIVE_SIZE} genomes")
    print(f"  JIT:           {'Numba OK (50-100x speedup)' if _NUMBA_OK else 'Numba NOT installed — slow mode!'}")
    print()
    print("  Loading features...")
    precomp = _load_precomp()
    if not precomp:
        print("[ERROR] No data!"); sys.exit(1)
    total_b = sum(f.shape[0] for f, _, _, _, _, *_ in precomp)
    print(f"\n  Periods: {len(precomp)}  Total bars: {total_b:,}")
    print(f"  Simulations per generation: {POP_SIZE*len(precomp)}")
    snap = _get_snap_every()
    avg_snaps = total_b // len(precomp) // snap if len(precomp) > 0 else 0
    print(f"  Daily snapshots per period: ~{avg_snaps}")
    print()

    trainer = GeneticTrainer(precomp)

    # ── Numba JIT прогрев (компиляция simulate_batch до старта тренировки) ──
    if _NUMBA_OK:
        print("  [Numba] Компиляция JIT-симулятора (10-15 сек, только первый раз)...")
        t_jit = time.time()
        _warm_up_numba()
        print(f"  [Numba] Готово за {time.time()-t_jit:.1f}с  "
              f"(следующие запуски — мгновенно из кэша)")

    trainer.run()

    print(f"\n  Results: {AGENTS_DIR}")
    print("  Integration:")
    print("    from crypto_genetics_v5 import GeneticsAgent")
    print("    agents['Genetics'] = GeneticsAgent()")
