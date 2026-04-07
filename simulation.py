"""
simulation.py  —  Live Pygame Visualization
Run: python simulation.py

Controls:
  SPACE     — pause / resume
  +/-       — speed up / slow down
  R         — reset
  Q / ESC   — quit
  Click agent — inspect in sidebar
"""

import sys
import math
import random
import pygame

from agent      import AgentRole, AgentState
from task       import Task, Priority, generate_tasks
from environment import WarehouseEnvironment
from dispatcher  import Dispatcher

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CFG = {
    'grid_size':       20,
    'num_pickup':       5,
    'num_drop':         5,
    'num_congestion':  12,
    'tasks_per_tick':   0.15,   # probability of spawning a task each tick
    'max_tasks_queue':  12,
    'tick_ms':         120,     # ms between ticks (lower = faster)
    'min_tick_ms':      30,
    'max_tick_ms':     600,
    'random_seed':      7,
    'window_w':       1280,
    'window_h':        760,
}

random.seed(CFG['random_seed'])

# ─────────────────────────────────────────────────────────────
# COLORS  (all hardcoded — physical scene, must not invert)
# ─────────────────────────────────────────────────────────────
C = {
    'bg':          (15,  17,  26),
    'grid':        (28,  33,  50),
    'grid_line':   (40,  46,  68),
    'pickup':      (29, 158, 117),
    'drop':        (200,160,  40),
    'congestion':  (180,  40,  40),
    'charger':     (80,  140, 220),
    'charger_occ': (40,   80, 160),
    'path_fast':   (100, 200, 255),
    'path_heavy':  (255, 180,  60),
    'text':        (220, 220, 230),
    'text_dim':    (120, 125, 145),
    'panel':       (22,  25,  40),
    'panel_border':(50,  55,  80),
    'white':       (255, 255, 255),
    'black':       (0,   0,   0),
    'green':       (80,  200, 100),
    'red':         (220,  60,  60),
    'orange':      (255, 150,  40),
    'yellow':      (255, 220,  50),
    'priority':    {
        Priority.LOW:      (100, 180, 100),
        Priority.NORMAL:   (100, 160, 220),
        Priority.HIGH:     (255, 200,  50),
        Priority.URGENT:   (255, 140,  30),
        Priority.CRITICAL: (220,  60,  60),
    },
    'agent_fast':  (79, 195, 247),
    'agent_heavy': (255, 138,  96),
    'state_color': {
        AgentState.IDLE:       (130, 130, 150),
        AgentState.TO_PICKUP:  ( 79, 195, 247),
        AgentState.TO_DROP:    ( 80, 200, 100),
        AgentState.TO_CHARGER: (255, 220,  50),
        AgentState.CHARGING:   ( 80, 140, 220),
        AgentState.WAITING:    (220,  60,  60),
    },
}

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
SIDEBAR_W   = 300
GRID_PANEL_W = CFG['window_w'] - SIDEBAR_W
GRID_PANEL_H = CFG['window_h']
CELL        = GRID_PANEL_W // CFG['grid_size']
GRID_OFF_X  = (GRID_PANEL_W - CFG['grid_size'] * CELL) // 2
GRID_OFF_Y  = (GRID_PANEL_H - CFG['grid_size'] * CELL) // 2


def cell_center(x, y):
    return (GRID_OFF_X + x * CELL + CELL // 2,
            GRID_OFF_Y + y * CELL + CELL // 2)


def cell_rect(x, y):
    return pygame.Rect(GRID_OFF_X + x * CELL, GRID_OFF_Y + y * CELL, CELL, CELL)


# ─────────────────────────────────────────────────────────────
# MAIN SIMULATION CLASS
# ─────────────────────────────────────────────────────────────
class WarehouseSimulation:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((CFG['window_w'], CFG['window_h']))
        pygame.display.set_caption("Adaptive Warehouse Dispatch System")
        self.clock   = pygame.time.Clock()

        self.font_lg  = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 12)
        self.font_sm  = pygame.font.SysFont("monospace", 10)
        self.font_hd  = pygame.font.SysFont("monospace", 18, bold=True)

        self._init_sim()

    def _init_sim(self):
        self.env = WarehouseEnvironment(
            grid_size=CFG['grid_size'],
            num_pickup=CFG['num_pickup'],
            num_drop=CFG['num_drop'],
            num_congestion=CFG['num_congestion'],
            dynamic_traffic=True,
        )
        self.dispatcher = Dispatcher(self.env, epsilon=0.2)

        # Seed with initial tasks
        init_tasks = generate_tasks(6, CFG['grid_size'],
                                    self.env.pickup_zones, self.env.drop_zones)
        self.dispatcher.add_tasks(init_tasks)

        self.paused      = False
        self.tick_ms     = CFG['tick_ms']
        self.selected_id = None          # selected agent id
        self.tick_rewards: list[float] = []
        self.total_delivered   = 0
        self.total_failed      = 0
        self.log: list[str]    = []
        self._elapsed          = 0

    # ──────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────
    def run(self):
        while True:
            dt = self.clock.tick(60)
            self._handle_events()

            if not self.paused:
                self._elapsed += dt
                if self._elapsed >= self.tick_ms:
                    self._elapsed = 0
                    self._do_tick()

            self._draw()
            pygame.display.flip()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.tick_ms = max(CFG['min_tick_ms'], self.tick_ms - 20)
                elif event.key == pygame.K_MINUS:
                    self.tick_ms = min(CFG['max_tick_ms'], self.tick_ms + 20)
                elif event.key == pygame.K_r:
                    self._init_sim()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)

    def _handle_click(self, pos):
        mx, my = pos
        if mx >= GRID_PANEL_W:
            return  # clicked sidebar
        for agent in self.env.agents:
            cx, cy = cell_center(agent.x, agent.y)
            if math.dist((mx, my), (cx, cy)) < CELL * 0.55:
                self.selected_id = agent.id
                return
        self.selected_id = None

    def _do_tick(self):
        # Possibly spawn new tasks
        pending = [t for t in self.dispatcher.pending_tasks
                   if t.status.name == 'PENDING']
        if len(pending) < CFG['max_tasks_queue'] and random.random() < CFG['tasks_per_tick']:
            new = generate_tasks(1, CFG['grid_size'],
                                 self.env.pickup_zones, self.env.drop_zones)
            self.dispatcher.add_tasks(new)

        stats = self.dispatcher.tick()

        # Collect rewards
        for r in stats['rewards']:
            self.tick_rewards.append(r)
            self.total_delivered += 1

        # Count failures
        self.total_failed = len([t for t in self.dispatcher.all_tasks
                                  if t.status.name == 'FAILED'])

        # Log collisions / deadlocks
        if stats['collisions'] > 0:
            self._add_log(f"[tick {self.env.tick}] collision detected")
        if stats['deadlocks'] > 0:
            self._add_log(f"[tick {self.env.tick}] deadlock resolved")

    def _add_log(self, msg: str):
        self.log.append(msg)
        if len(self.log) > 8:
            self.log.pop(0)

    # ──────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────
    def _draw(self):
        self.screen.fill(C['bg'])
        self._draw_grid()
        self._draw_zones()
        self._draw_chargers()
        self._draw_paths()
        self._draw_tasks_on_grid()
        self._draw_agents()
        self._draw_sidebar()

        if self.paused:
            self._draw_paused()

    def _draw_grid(self):
        gs = CFG['grid_size']
        for x in range(gs):
            for y in range(gs):
                r = cell_rect(x, y)
                pygame.draw.rect(self.screen, C['grid'], r)
        # Grid lines
        for x in range(gs + 1):
            px = GRID_OFF_X + x * CELL
            pygame.draw.line(self.screen, C['grid_line'],
                             (px, GRID_OFF_Y),
                             (px, GRID_OFF_Y + gs * CELL), 1)
        for y in range(gs + 1):
            py = GRID_OFF_Y + y * CELL
            pygame.draw.line(self.screen, C['grid_line'],
                             (GRID_OFF_X, py),
                             (GRID_OFF_X + gs * CELL, py), 1)

    def _draw_zones(self):
        for (x, y) in self.env.congestion_zones:
            r = cell_rect(x, y)
            s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            s.fill((*C['congestion'], 70))
            self.screen.blit(s, r)

        for (x, y) in self.env.pickup_zones:
            r = cell_rect(x, y)
            s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            s.fill((*C['pickup'], 100))
            self.screen.blit(s, r)
            lbl = self.font_sm.render("P", True, C['white'])
            self.screen.blit(lbl, (r.x + CELL//2 - 4, r.y + CELL//2 - 5))

        for (x, y) in self.env.drop_zones:
            r = cell_rect(x, y)
            s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            s.fill((*C['drop'], 100))
            self.screen.blit(s, r)
            lbl = self.font_sm.render("D", True, C['black'])
            self.screen.blit(lbl, (r.x + CELL//2 - 4, r.y + CELL//2 - 5))

    def _draw_chargers(self):
        for station in self.env.charging_stations:
            x, y = station.x, station.y
            r    = cell_rect(x, y)
            col  = C['charger_occ'] if not station.is_free else C['charger']
            pygame.draw.rect(self.screen, col, r, border_radius=3)
            lbl = self.font_sm.render("Z", True, C['white'])
            self.screen.blit(lbl, (r.x + CELL//2 - 4, r.y + CELL//2 - 5))

    def _draw_paths(self):
        for agent in self.env.agents:
            if not agent.path:
                continue
            col = C['path_fast'] if agent.role == AgentRole.FAST else C['path_heavy']
            pts = [cell_center(agent.x, agent.y)]
            for (px, py) in agent.path[:12]:
                pts.append(cell_center(px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (*col, 120), False, pts, 1)

    def _draw_tasks_on_grid(self):
        """Draw small diamonds at pickup and drops of active tasks."""
        for agent in self.env.agents:
            task = agent.current_task
            if not task:
                continue
            col = C['priority'].get(task.priority, C['white'])
            # Pickup marker
            if task.status.name in ('ASSIGNED', 'PENDING'):
                cx, cy = cell_center(*task.pickup)
                pts = [(cx, cy-6), (cx+6, cy), (cx, cy+6), (cx-6, cy)]
                pygame.draw.polygon(self.screen, col, pts)
            # Drop marker
            cx, cy = cell_center(*task.drop)
            pts = [(cx, cy-6), (cx+6, cy), (cx, cy+6), (cx-6, cy)]
            pygame.draw.polygon(self.screen, col, pts, 2)

    def _draw_agents(self):
        for agent in self.env.agents:
            cx, cy = cell_center(agent.x, agent.y)
            radius = CELL // 2 - 2

            # Body color by role
            body_col = (C['agent_fast'] if agent.role == AgentRole.FAST
                        else C['agent_heavy'])

            # Highlight selected
            if self.selected_id == agent.id:
                pygame.draw.circle(self.screen, C['white'], (cx, cy), radius + 4, 2)

            # State ring
            state_col = C['state_color'].get(agent.state, C['text_dim'])
            pygame.draw.circle(self.screen, state_col, (cx, cy), radius + 2)
            pygame.draw.circle(self.screen, body_col,  (cx, cy), radius)

            # ID label
            lbl = self.font_md.render(str(agent.id), True, C['black'])
            self.screen.blit(lbl, (cx - lbl.get_width()//2,
                                   cy - lbl.get_height()//2))

            # Battery bar above agent
            bar_w = CELL - 4
            bar_h = 4
            bx    = cx - bar_w // 2
            by    = cy - radius - 8
            pygame.draw.rect(self.screen, (60, 60, 80),
                             (bx, by, bar_w, bar_h), border_radius=2)
            fill  = int(bar_w * agent.battery / 100)
            bat_col = (C['green'] if agent.battery > 40
                       else C['orange'] if agent.battery > 20
                       else C['red'])
            pygame.draw.rect(self.screen, bat_col,
                             (bx, by, fill, bar_h), border_radius=2)

    # ──────────────────────────────────────────
    # Sidebar
    # ──────────────────────────────────────────
    def _draw_sidebar(self):
        sx = GRID_PANEL_W
        panel = pygame.Rect(sx, 0, SIDEBAR_W, CFG['window_h'])
        pygame.draw.rect(self.screen, C['panel'], panel)
        pygame.draw.line(self.screen, C['panel_border'],
                         (sx, 0), (sx, CFG['window_h']), 1)

        x, y = sx + 12, 12

        # ── Title
        t = self.font_hd.render("WAREHOUSE AI", True, C['agent_fast'])
        self.screen.blit(t, (x, y)); y += 26

        # ── Tick / speed
        info = f"Tick: {self.env.tick}   Speed: {1000//max(1,self.tick_ms)}x"
        self.screen.blit(self.font_sm.render(info, True, C['text_dim']), (x, y)); y += 18

        # ── KPIs
        y += 6
        self._sidebar_kpi(x, y, "Delivered",  str(self.total_delivered), C['green'])
        self._sidebar_kpi(x + 138, y, "Failed", str(self.total_failed), C['red'])
        y += 28
        self._sidebar_kpi(x, y, "Collisions",
                          str(self.env.collision_events), C['orange'])
        self._sidebar_kpi(x + 138, y, "Deadlocks",
                          str(self.env.deadlocks_resolved), C['yellow'])
        y += 28

        # ── Best strategy
        _, best = self.dispatcher.best_strategy()
        t = self.font_sm.render(f"Strategy: {best}", True, C['agent_heavy'])
        self.screen.blit(t, (x, y)); y += 20

        # ── Order queue
        pygame.draw.line(self.screen, C['panel_border'], (sx+8, y), (sx+SIDEBAR_W-8, y), 1)
        y += 8
        t = self.font_md.render("ORDER QUEUE", True, C['text'])
        self.screen.blit(t, (x, y)); y += 18

        pending = sorted(
            [t for t in self.dispatcher.pending_tasks if t.status.name == 'PENDING'],
            key=lambda t: -t.priority_score
        )[:8]

        for task in pending:
            col  = C['priority'].get(task.priority, C['white'])
            tr   = max(0.0, task.time_remaining)
            line = f"#{task.id} {task.priority.name[:3]} {task.category[:3]} {tr:.0f}s"
            pygame.draw.rect(self.screen, col,
                             pygame.Rect(x-2, y-1, 4, 14), border_radius=2)
            t2 = self.font_sm.render(line, True, C['text'])
            self.screen.blit(t2, (x+8, y)); y += 16
            if y > CFG['window_h'] - 240:
                break

        # ── Agent details
        pygame.draw.line(self.screen, C['panel_border'], (sx+8, y+4), (sx+SIDEBAR_W-8, y+4), 1)
        y += 14
        t = self.font_md.render("AGENTS", True, C['text'])
        self.screen.blit(t, (x, y)); y += 18

        for agent in self.env.agents:
            hl  = (agent.id == self.selected_id)
            col = C['agent_fast'] if agent.role == AgentRole.FAST else C['agent_heavy']
            if hl:
                pygame.draw.rect(self.screen, (35, 40, 65),
                                 pygame.Rect(sx+4, y-2, SIDEBAR_W-8, 48),
                                 border_radius=4)
            r_tag = "FAST " if agent.role == AgentRole.FAST else "HEAVY"
            line1 = f"[{agent.id}] {r_tag}  {agent.state.name[:8]}"
            line2 = (f"  batt={agent.battery:.0f}%  "
                     f"tasks={agent.tasks_completed}  "
                     f"spec={agent.specialization[:6]}")
            t1 = self.font_sm.render(line1, True, col)
            t2 = self.font_sm.render(line2, True, C['text_dim'])
            self.screen.blit(t1, (x, y));    y += 14
            self.screen.blit(t2, (x, y));    y += 14

            if hl and agent.current_task:
                tk = agent.current_task
                t3 = self.font_sm.render(
                    f"  Task #{tk.id} {tk.priority.name} {tk.category}",
                    True, C['priority'].get(tk.priority, C['white']))
                self.screen.blit(t3, (x, y)); y += 14
            y += 2
            if y > CFG['window_h'] - 120:
                break

        # ── Event log
        pygame.draw.line(self.screen, C['panel_border'],
                         (sx+8, y+4), (sx+SIDEBAR_W-8, y+4), 1)
        y += 14
        t = self.font_md.render("EVENT LOG", True, C['text'])
        self.screen.blit(t, (x, y)); y += 16
        for entry in self.log[-5:]:
            t2 = self.font_sm.render(entry[:34], True, C['text_dim'])
            self.screen.blit(t2, (x, y)); y += 14

        # ── Controls hint
        hint = "[SPC] pause  [+/-] speed  [R] reset  [Q] quit"
        th = self.font_sm.render(hint, True, C['text_dim'])
        self.screen.blit(th, (sx + 8, CFG['window_h'] - 18))

    def _sidebar_kpi(self, x, y, label, value, col):
        tl = self.font_sm.render(label, True, C['text_dim'])
        tv = self.font_lg.render(value, True, col)
        self.screen.blit(tl, (x, y))
        self.screen.blit(tv, (x, y + 12))

    def _draw_paused(self):
        s = pygame.Surface((GRID_PANEL_W, GRID_PANEL_H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 100))
        self.screen.blit(s, (0, 0))
        t = self.font_hd.render("PAUSED  [SPACE to resume]", True, C['white'])
        rx = GRID_PANEL_W // 2 - t.get_width() // 2
        ry = GRID_PANEL_H // 2 - t.get_height() // 2
        self.screen.blit(t, (rx, ry))


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sim = WarehouseSimulation()
    sim.run()
