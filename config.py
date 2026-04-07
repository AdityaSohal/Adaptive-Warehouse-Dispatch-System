"""
config.py — Central configuration for the Adaptive Warehouse Dispatch System v3.
All tunable constants live here. Every other module imports from this file.
"""

# ─── Simulation ────────────────────────────────────────────────────────────────
GRID_SIZE          = 22
RANDOM_SEED        = 42

# ─── Layout counts ─────────────────────────────────────────────────────────────
NUM_PICKUP_ZONES   = 6
NUM_DROP_ZONES     = 5
NUM_CHARGING       = 3
NUM_SHELF_ROWS     = 3
SHELF_ROW_LEN      = 4

# ─── Fleet composition ─────────────────────────────────────────────────────────
NUM_FAST_AGENTS    = 4
NUM_HEAVY_AGENTS   = 2

# ─── Agent physics ─────────────────────────────────────────────────────────────
FAST_SPEED         = 2.0
HEAVY_SPEED        = 1.0
FAST_CAPACITY      = 15.0
HEAVY_CAPACITY     = 40.0
FAST_DRAIN         = 1.4
HEAVY_DRAIN        = 0.9
CHARGE_RATE        = 9.0
BATTERY_FULL       = 100.0
BATTERY_LOW        = 28.0
BATTERY_CRITICAL   = 12.0
BATTERY_RESUME     = 90.0
# Predictive charging: if predicted battery at task completion < this, charge first
BATTERY_PREDICTIVE = 35.0

# ─── CBS Path Planning ─────────────────────────────────────────────────────────
CBS_MAX_ITERATIONS     = 200     # max conflict-tree nodes before fallback
CBS_REPLAN_INTERVAL    = 10      # ticks between global CBS replanning passes
RESERVATION_HORIZON    = 30      # ticks ahead in space-time table
CORRIDOR_LANE_WIDTH    = 1       # cells — one-way lanes in corridors (0 = disabled)

# ─── ORCA Collision Avoidance ──────────────────────────────────────────────────
ORCA_TAU               = 3.0     # seconds ahead for velocity-obstacle computation
ORCA_NEIGHBOR_DIST     = 4.0     # cells — radius to consider other agents
ORCA_MAX_SPEED         = 2.0     # max correction speed
ORCA_TIME_HORIZON      = 5.0     # time horizon for ORCA half-planes
TTC_EMERGENCY_THRESH   = 2.0     # ticks — time-to-collision that triggers e-stop

# ─── Deadlock Detection ────────────────────────────────────────────────────────
DEADLOCK_WAIT          = 6       # ticks waiting → flag as potential deadlock
DEADLOCK_CYCLE_CHECK   = 4       # min cycle length for wait-for graph detection
PUSH_SWAP_RADIUS       = 3       # cells — search radius for push-and-swap escape zone
DEADLOCK_TOKEN_TTL     = 20      # ticks — how long a push token stays active

# ─── Task / Order ──────────────────────────────────────────────────────────────
TASKS_PER_TICK_P       = 0.14
MAX_QUEUE              = 14
INITIAL_TASKS          = 6
PRIORITY_DEADLINES     = {5: 14, 4: 28, 3: 45, 2: 65, 1: 95}
HEAVY_WEIGHT_KG        = 14.0
SHORT_DIST_CELLS       = 5

# ─── Charging Strategy ─────────────────────────────────────────────────────────
CHARGE_RESERVE_WINDOW  = 15      # ticks — reserve charger this far ahead
CHARGE_MAX_QUEUE       = 2       # max agents queued per charger
CHARGE_PRIORITY_BONUS  = 5.0    # reward bonus for completing charge before critical

# ─── Congestion Heatmap ────────────────────────────────────────────────────────
NUM_CONGESTION         = 14
DYNAMIC_CONGESTION     = True
CONGESTION_REFRESH     = 22
HEATMAP_ALPHA          = 0.05    # EMA decay for learned heatmap (lower = slower to forget)
HEATMAP_PENALTY_SCALE  = 2.0    # multiplier on heatmap cost in A* edge weights

# ─── RL / Bandit ───────────────────────────────────────────────────────────────
BANDIT_UCB_C           = 1.5
QL_ALPHA               = 0.15
QL_GAMMA               = 0.92
QL_EPSILON_START       = 0.50
QL_EPSILON_MIN         = 0.05
QL_EPSILON_DECAY       = 0.9975
SPEC_THRESHOLD         = 5

# ─── MAPPO Training ────────────────────────────────────────────────────────────
MAPPO_ENABLED          = True    # set False to skip neural training
MAPPO_HIDDEN_DIM       = 128
MAPPO_LR               = 3e-4
MAPPO_GAMMA            = 0.99
MAPPO_GAE_LAMBDA       = 0.95
MAPPO_CLIP_EPS         = 0.2
MAPPO_ENTROPY_COEF     = 0.01
MAPPO_VF_COEF          = 0.5
MAPPO_UPDATE_EPOCHS    = 4
MAPPO_BATCH_SIZE       = 64
MAPPO_ROLLOUT_LEN      = 128     # steps per rollout before an update
MAPPO_MODEL_PATH       = "models/mappo_checkpoint.pt"

# ─── Reward shaping (MAPPO) ────────────────────────────────────────────────────
REWARD_DELIVERY        =  5.0    # per successful delivery
REWARD_PRIORITY_SCALE  =  1.5    # multiplied by priority level
PENALTY_COLLISION      = -8.0
PENALTY_DEADLOCK       = -4.0
PENALTY_IDLE           = -0.05   # per tick idle with tasks pending
PENALTY_FLAT_BATTERY   = -15.0
REWARD_CHARGE_COMPLETE =  2.0
PENALTY_CONGESTION     = -0.3    # per step through a high-cost cell

# ─── Model persistence ─────────────────────────────────────────────────────────
MODEL_SAVE_INTERVAL    = 1       # save after every N episodes
MODEL_DIR              = "models"

# ─── Environment ───────────────────────────────────────────────────────────────
BANDIT_EPSILON         = 0.20    # kept for legacy; UCB1 is primary

# ─── Visualisation ─────────────────────────────────────────────────────────────
WINDOW_W           = 1380
WINDOW_H           = 800
SIDEBAR_W          = 330
TARGET_FPS         = 60
TICK_MS_DEFAULT    = 110
TICK_MS_MIN        = 25
TICK_MS_MAX        = 700
AGENT_LERP         = 0.28
PATH_DRAW_STEPS    = 14
CHART_HISTORY      = 80

# ─── Colour palette ────────────────────────────────────────────────────────────
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
    heatmap      = (200,  50,  10, 60),   # learned heatmap overlay
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
