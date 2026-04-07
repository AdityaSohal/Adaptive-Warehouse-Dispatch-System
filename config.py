"""
config.py — Central configuration for the Adaptive Warehouse Dispatch System.
Edit values here; every other module imports from this file.
"""

# ─── Simulation ────────────────────────────────────────────────────────────────
GRID_SIZE          = 22          # cells per side  (22×22 grid)
RANDOM_SEED        = 42

# ─── Layout counts ─────────────────────────────────────────────────────────────
NUM_PICKUP_ZONES   = 6           # shelf / pickup stations
NUM_DROP_ZONES     = 5           # delivery drop points
NUM_CHARGING       = 3           # charging stations
NUM_SHELF_ROWS     = 3           # rows of decorative shelf obstacles
SHELF_ROW_LEN      = 4           # shelves per row

# ─── Fleet composition ─────────────────────────────────────────────────────────
NUM_FAST_AGENTS    = 4           # AgentRole.FAST
NUM_HEAVY_AGENTS   = 2           # AgentRole.HEAVY

# ─── Agent physics ─────────────────────────────────────────────────────────────
FAST_SPEED         = 2.0         # steps/tick (pathfinding weight)
HEAVY_SPEED        = 1.0
FAST_CAPACITY      = 15.0        # kg
HEAVY_CAPACITY     = 40.0        # kg
FAST_DRAIN         = 1.4         # battery % lost per movement step
HEAVY_DRAIN        = 0.9
CHARGE_RATE        = 9.0         # battery % gained per charging tick
BATTERY_FULL       = 100.0
BATTERY_LOW        = 28.0        # threshold → route to charger
BATTERY_CRITICAL   = 12.0        # must charge regardless
BATTERY_RESUME     = 90.0        # resume work after charging to this level

# ─── RL / Bandit ───────────────────────────────────────────────────────────────
BANDIT_EPSILON     = 0.20        # ε-greedy exploration for strategy bandit
BANDIT_UCB_C       = 1.5         # UCB1 exploration constant
QL_ALPHA           = 0.15        # Q-learning rate
QL_GAMMA           = 0.92        # discount factor
QL_EPSILON_START   = 0.50        # initial agent exploration rate
QL_EPSILON_MIN     = 0.05        # minimum exploration rate
QL_EPSILON_DECAY   = 0.9975      # per-update multiplicative decay
SPEC_THRESHOLD     = 5           # tasks in category before specialisation locks

# ─── Task / Order ──────────────────────────────────────────────────────────────
TASKS_PER_TICK_P   = 0.14        # probability of spawning a task each tick
MAX_QUEUE          = 14
INITIAL_TASKS      = 6

# Priority deadlines in seconds  {priority_value: seconds}
PRIORITY_DEADLINES = {5: 14, 4: 28, 3: 45, 2: 65, 1: 95}

# Category thresholds
HEAVY_WEIGHT_KG    = 14.0
SHORT_DIST_CELLS   = 5

# ─── Environment ───────────────────────────────────────────────────────────────
NUM_CONGESTION     = 14
DYNAMIC_CONGESTION = True        # refresh congestion zones every N ticks
CONGESTION_REFRESH = 22
DEADLOCK_WAIT      = 6           # ticks waiting → force replan
RESERVATION_HORIZON= 30          # ticks ahead in space-time table

# ─── Visualisation ─────────────────────────────────────────────────────────────
WINDOW_W           = 1380
WINDOW_H           = 800
SIDEBAR_W          = 330
TARGET_FPS         = 60
TICK_MS_DEFAULT    = 110         # ms between simulation ticks
TICK_MS_MIN        = 25
TICK_MS_MAX        = 700
AGENT_LERP         = 0.28        # pixel-position lerp speed  (0–1, higher = snappier)
PATH_DRAW_STEPS    = 14          # how many future steps to render per agent
CHART_HISTORY      = 80          # ticks of rolling-window metric chart

# ─── Colour palette (all tuples) ───────────────────────────────────────────────
COL = dict(
    bg           = (11,  13,  20),
    grid_cell    = (16,  19,  29),
    grid_line    = (24,  28,  44),
    shelf_bg     = (28,  18,   8),
    shelf_border = (72,  44,  16),
    shelf_item   = [(200, 147,  90), (180, 110,  60), (220, 170, 100), (160,  90,  40)],
    pickup_bg    = (10,  28,  48),
    pickup_bdr   = (30,  90, 160),
    drop_bg      = (10,  28,  16),
    drop_bdr     = (30, 130,  60),
    charger_idle = (12,  28,  55),
    charger_bdr  = (30,  90, 220),
    charger_occ  = (20,  55, 140),
    congestion   = (120,  20,  20, 50),
    agent_fast   = ( 3, 155, 229),
    agent_heavy  = (230,  74,  25),
    path_fast    = ( 41, 182, 246, 80),
    path_heavy   = (255, 138,  50, 80),
    panel_bg     = (14,  17,  26),
    panel_bdr    = (30,  36,  58),
    text         = (200, 206, 220),
    text_dim     = ( 90,  98, 128),
    white        = (255, 255, 255),
    green        = ( 63, 185,  80),
    red          = (248,  81,  73),
    orange       = (255, 123,  36),
    yellow       = (210, 153,  34),
    blue         = ( 88, 166, 255),
    priority     = {5:(248,81,73), 4:(255,123,36), 3:(210,153,34), 2:(88,166,255), 1:(99,110,130)},
    state        = {0:(60,65,90), 1:(29,182,246), 2:(63,185,80), 3:(255,215,50), 4:(30,100,230), 5:(230,70,60)},
)
