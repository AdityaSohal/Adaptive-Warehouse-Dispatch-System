"""
simulation.py — Live Pygame visualiser for the Adaptive Warehouse Dispatch System.

Inspired by real AMR dashboards (Amazon Robotics, Locus Robotics, 6RS Chuck).

Controls
─────────
  SPACE     pause / resume
  +/-       speed up / slow down
  R         reset simulation
  Q / ESC   quit
  Click     select / deselect an agent
"""

from __future__ import annotations

import math
import random
import sys
import time

import pygame

from agent       import AgentRole, AgentState
from config      import (
    GRID_SIZE, RANDOM_SEED,
    INITIAL_TASKS, TASKS_PER_TICK_P, MAX_QUEUE,
    WINDOW_W, WINDOW_H, SIDEBAR_W,
    TARGET_FPS, TICK_MS_DEFAULT, TICK_MS_MIN, TICK_MS_MAX,
    AGENT_LERP, PATH_DRAW_STEPS, CHART_HISTORY,
    COL,
    NUM_FAST_AGENTS, NUM_HEAVY_AGENTS,
)
from dispatcher  import Dispatcher
from environment import WarehouseEnvironment
from strategies  import STRATEGY_NAMES
from task        import Task, Priority, generate_tasks

# ── Layout constants ────────────────────────────────────────────────────────────
GRID_PX = WINDOW_W - SIDEBAR_W          # pixel width of the grid panel
CELL    = GRID_PX // GRID_SIZE          # pixel size per cell
GRID_OX = (GRID_PX - GRID_SIZE * CELL) // 2   # left offset
GRID_OY = (WINDOW_H - GRID_SIZE * CELL) // 2  # top offset


def cell_center(x: int, y: int) -> tuple[int, int]:
    return (GRID_OX + x * CELL + CELL // 2, GRID_OY + y * CELL + CELL // 2)


def cell_rect(x: int, y: int) -> pygame.Rect:
    return pygame.Rect(GRID_OX + x * CELL, GRID_OY + y * CELL, CELL, CELL)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _col(key, alpha=None):
    c = COL[key]
    if alpha is not None and len(c) == 3:
        return (*c, alpha)
    return c


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ══════════════════════════════════════════════════════════════════════════════
class WarehouseSimulation:

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Adaptive Warehouse Dispatch  ·  AMR Simulation")
        self.clock  = pygame.time.Clock()

        # Fonts
        mono = "monospace"
        self.f10  = pygame.font.SysFont(mono, 10)
        self.f11  = pygame.font.SysFont(mono, 11)
        self.f12  = pygame.font.SysFont(mono, 12)
        self.f13  = pygame.font.SysFont(mono, 13, bold=True)
        self.f15  = pygame.font.SysFont(mono, 15, bold=True)
        self.f18  = pygame.font.SysFont(mono, 18, bold=True)
        self.f20  = pygame.font.SysFont(mono, 20, bold=True)

        # Surfaces for transparency
        self._surf_cell = pygame.Surface((CELL, CELL), pygame.SRCALPHA)

        self._init_sim()

    # ── Simulation init / reset ──────────────────────────────────────────────────
    def _init_sim(self) -> None:
        random.seed(RANDOM_SEED)
        self.env        = WarehouseEnvironment(seed=RANDOM_SEED)
        self.dispatcher = Dispatcher(self.env)

        # Seed with initial tasks
        init_tasks = generate_tasks(
            INITIAL_TASKS, GRID_SIZE,
            self.env.pickup_zones, self.env.drop_zones,
        )
        self.dispatcher.add_tasks(init_tasks)

        # Init pixel positions for agents
        for a in self.env.agents:
            cx, cy = cell_center(a.x, a.y)
            a.px, a.py = float(cx), float(cy)

        self.paused       = False
        self.tick_ms      = TICK_MS_DEFAULT
        self.selected_id: int | None = None
        self._elapsed     = 0

        # Metrics history for charts
        self.reward_hist: list[float] = []
        self.eff_hist:    list[float] = []
        self.queue_hist:  list[int]   = []

        self.event_log: list[str] = []
        self.total_delivered = 0
        self.total_failed    = 0

        # Flash overlay for deliveries
        self._flash_cells: list[tuple[tuple, float, tuple]] = []  # (cell, expire_time, col)

    # ── Main loop ────────────────────────────────────────────────────────────────
    def run(self) -> None:
        while True:
            dt = self.clock.tick(TARGET_FPS)
            self._handle_events()

            if not self.paused:
                self._elapsed += dt
                if self._elapsed >= self.tick_ms:
                    self._elapsed = 0
                    self._do_tick()

            # Lerp pixel positions every frame
            for a in self.env.agents:
                cx, cy = cell_center(a.x, a.y)
                a.lerp_pixel(cx, cy)

            self._draw()
            pygame.display.flip()

    # ── Events ───────────────────────────────────────────────────────────────────
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    self._quit()
                elif k == pygame.K_SPACE:
                    self.paused = not self.paused
                elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.tick_ms = max(TICK_MS_MIN, self.tick_ms - 20)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.tick_ms = min(TICK_MS_MAX, self.tick_ms + 20)
                elif k == pygame.K_r:
                    self._init_sim()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(event.pos)

    def _handle_click(self, pos: tuple) -> None:
        mx, my = pos
        if mx >= GRID_PX:
            return
        for agent in self.env.agents:
            cx, cy = cell_center(agent.x, agent.y)
            if math.dist((mx, my), (cx, cy)) < CELL * 0.55:
                self.selected_id = agent.id if self.selected_id != agent.id else None
                return
        self.selected_id = None

    @staticmethod
    def _quit() -> None:
        pygame.quit()
        sys.exit()

    # ── Simulation tick ──────────────────────────────────────────────────────────
    def _do_tick(self) -> None:
        # Possibly spawn new tasks
        pending = sum(1 for t in self.dispatcher.pending_tasks if t.status.name == 'PENDING')
        new_tasks = []
        if pending < MAX_QUEUE and random.random() < TASKS_PER_TICK_P:
            t = generate_tasks(1, GRID_SIZE, self.env.pickup_zones, self.env.drop_zones)[0]
            new_tasks = [t]
            self._log(f"[{self.env.tick+1}] new order #{t.id} [{t.priority.name}]")

        stats = self.dispatcher.tick(new_tasks)

        # Track deliveries
        for r in stats['rewards']:
            self.total_delivered += 1
            self._log(f"[{self.env.tick}] delivered ✓  reward={r:.1f}")
        self.total_failed = self.dispatcher.total_failed()

        # Flash delivered drop zones
        for a in self.env.agents:
            if a.state == AgentState.IDLE and not a.carrying and a.tasks_completed > 0:
                if a.location in self.env.drop_zones:
                    self._flash_cells.append((
                        a.location,
                        time.time() + 0.5,
                        COL['green'],
                    ))

        # Update charts
        self.reward_hist.append(self.dispatcher.avg_reward())
        self.eff_hist.append(self.dispatcher.efficiency() * 100)
        self.queue_hist.append(stats['pending'])
        for lst in (self.reward_hist, self.eff_hist, self.queue_hist):
            if len(lst) > CHART_HISTORY:
                lst.pop(0)

        if stats['deadlocks'] > 0:
            self._log(f"[{self.env.tick}] deadlock resolved")

    def _log(self, msg: str) -> None:
        self.event_log.append(msg)
        if len(self.event_log) > 10:
            self.event_log.pop(0)

    # ── Master draw ──────────────────────────────────────────────────────────────
    def _draw(self) -> None:
        self.screen.fill(COL['bg'])
        self._draw_grid()
        self._draw_obstacles()
        self._draw_zones()
        self._draw_chargers()
        self._draw_congestion()
        self._draw_flash()
        self._draw_paths()
        self._draw_task_markers()
        self._draw_agents()
        self._draw_sidebar()
        if self.paused:
            self._draw_paused_overlay()

    # ── Grid ─────────────────────────────────────────────────────────────────────
    def _draw_grid(self) -> None:
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                r = cell_rect(x, y)
                pygame.draw.rect(self.screen, COL['grid_cell'], r)
        for x in range(GRID_SIZE + 1):
            px = GRID_OX + x * CELL
            pygame.draw.line(self.screen, COL['grid_line'],
                             (px, GRID_OY), (px, GRID_OY + GRID_SIZE * CELL))
        for y in range(GRID_SIZE + 1):
            py = GRID_OY + y * CELL
            pygame.draw.line(self.screen, COL['grid_line'],
                             (GRID_OX, py), (GRID_OX + GRID_SIZE * CELL, py))

    # ── Shelf obstacles ──────────────────────────────────────────────────────────
    def _draw_obstacles(self) -> None:
        SHELF_COLORS = COL['shelf_item']
        for (sx, sy) in self.env.obstacles:
            r = cell_rect(sx, sy)
            pygame.draw.rect(self.screen, COL['shelf_bg'], r)
            pygame.draw.rect(self.screen, COL['shelf_border'], r, 1)

            # Draw stacked boxes on the shelf
            bw = max(4, CELL // 2 - 3)
            bh = max(3, CELL // 3 - 2)
            for row in range(2):
                for col in range(2):
                    ix = r.x + 2 + col * (bw + 2)
                    iy = r.y + 2 + row * (bh + 2)
                    if ix + bw <= r.right and iy + bh <= r.bottom:
                        color_idx = (sx * 3 + sy * 7 + row * 2 + col) % len(SHELF_COLORS)
                        pygame.draw.rect(self.screen, SHELF_COLORS[color_idx],
                                         (ix, iy, bw, bh))
                        pygame.draw.rect(self.screen, (20, 10, 0), (ix, iy, bw, bh), 1)
                        # Plank line
                        pygame.draw.line(self.screen, (20, 10, 0),
                                         (ix, iy + bh // 2), (ix + bw, iy + bh // 2))

    # ── Pickup & drop zones ──────────────────────────────────────────────────────
    def _draw_zones(self) -> None:
        for (x, y) in self.env.pickup_zones:
            r   = cell_rect(x, y)
            inv = self.env.inventory.get((x, y), 0)

            # Background
            s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            s.fill((*COL['pickup_bg'], 200))
            self.screen.blit(s, r)
            pygame.draw.rect(self.screen, COL['pickup_bdr'], r, 2)

            # Stacked box icon (shows inventory level)
            boxes = min(inv, 4)
            bw, bh = max(4, CELL // 3), max(3, CELL // 4)
            for i in range(boxes):
                row = i // 2
                col = i % 2
                ix = r.x + 3 + col * (bw + 1)
                iy = r.y + r.height - bh - 2 - row * (bh + 1)
                if ix + bw <= r.right and iy >= r.y:
                    ci = i % len(COL['shelf_item'])
                    pygame.draw.rect(self.screen, COL['shelf_item'][ci],
                                     (ix, iy, bw, bh))
                    pygame.draw.rect(self.screen, (20, 10, 0), (ix, iy, bw, bh), 1)

            # Label
            lbl = self.f10.render(f"P{inv}", True, COL['pickup_bdr'])
            self.screen.blit(lbl, (r.x + 2, r.y + 2))

        for (x, y) in self.env.drop_zones:
            r = cell_rect(x, y)
            s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            s.fill((*COL['drop_bg'], 200))
            self.screen.blit(s, r)
            pygame.draw.rect(self.screen, COL['drop_bdr'], r, 2)

            # Target crosshair
            cx, cy = r.centerx, r.centery
            sz = CELL // 3
            pygame.draw.line(self.screen, COL['drop_bdr'],
                             (cx - sz, cy), (cx + sz, cy), 1)
            pygame.draw.line(self.screen, COL['drop_bdr'],
                             (cx, cy - sz), (cx, cy + sz), 1)
            pygame.draw.circle(self.screen, COL['drop_bdr'], (cx, cy), sz - 2, 1)

            lbl = self.f10.render("D", True, COL['drop_bdr'])
            self.screen.blit(lbl, (r.x + 2, r.y + 2))

    # ── Charging stations ────────────────────────────────────────────────────────
    def _draw_chargers(self) -> None:
        t = pygame.time.get_ticks() / 1000.0
        for station in self.env.charging_stations:
            r   = cell_rect(station.x, station.y)
            occ = not station.is_free
            bg  = COL['charger_occ'] if occ else COL['charger_idle']
            bdr = COL['charger_bdr']

            pygame.draw.rect(self.screen, bg, r, border_radius=3)
            pygame.draw.rect(self.screen, bdr, r, 2, border_radius=3)

            # Lightning bolt
            cx, cy = r.centerx, r.centery
            bolt = [
                (cx + 3, r.y + 3),
                (cx - 1, cy - 1),
                (cx + 2, cy - 1),
                (cx - 3, r.bottom - 3),
                (cx + 1, cy + 2),
                (cx - 2, cy + 2),
            ]
            pulse_col = bdr
            if occ:
                alpha = int(160 + 95 * math.sin(t * 3))
                pulse_col = (min(255, bdr[0] + 60), min(255, bdr[1] + 40), min(255, bdr[2] + 20))
            pygame.draw.polygon(self.screen, pulse_col, bolt)

    # ── Congestion zones ─────────────────────────────────────────────────────────
    def _draw_congestion(self) -> None:
        s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        s.fill((180, 30, 30, 40))
        for (x, y) in self.env.congestion_zones:
            self.screen.blit(s, cell_rect(x, y))

    # ── Flash cells (delivery confirmation) ──────────────────────────────────────
    def _draw_flash(self) -> None:
        now = time.time()
        self._flash_cells = [(c, e, col) for c, e, col in self._flash_cells if e > now]
        for (cell, expire, col) in self._flash_cells:
            remaining = expire - now
            alpha = int(200 * remaining / 0.5)
            s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            s.fill((*col, alpha))
            self.screen.blit(s, cell_rect(*cell))

    # ── Agent paths ──────────────────────────────────────────────────────────────
    def _draw_paths(self) -> None:
        for agent in self.env.agents:
            if not agent.path:
                continue
            is_fast = agent.role == AgentRole.FAST
            col     = (*COL['agent_fast'], 50) if is_fast else (*COL['agent_heavy'], 50)
            pts = [(int(agent.px), int(agent.py))]
            for (px, py) in agent.path[:PATH_DRAW_STEPS]:
                pts.append(cell_center(px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, col, False, pts, 1)

    # ── Task markers on grid ─────────────────────────────────────────────────────
    def _draw_task_markers(self) -> None:
        for agent in self.env.agents:
            task = agent.current_task
            if not task:
                continue
            col = task.color

            # Pickup marker (diamond) — while agent is heading there
            if agent.state == AgentState.TO_PICKUP:
                cx, cy = cell_center(*task.pickup)
                pts = [(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)]
                pygame.draw.polygon(self.screen, col, pts)
                pygame.draw.polygon(self.screen, COL['bg'], pts, 1)

            # Drop marker (open diamond)
            cx, cy = cell_center(*task.drop)
            pts = [(cx, cy - 5), (cx + 5, cy), (cx, cy + 5), (cx - 5, cy)]
            pygame.draw.polygon(self.screen, col, pts, 2)

    # ── Agents ───────────────────────────────────────────────────────────────────
    def _draw_agents(self) -> None:
        for agent in self.env.agents:
            self._draw_agent(agent)

    def _draw_agent(self, agent: Agent) -> None:
        px, py = int(agent.px), int(agent.py)
        radius = max(8, CELL // 2 - 3)

        body_col  = COL['agent_fast'] if agent.role == AgentRole.FAST else COL['agent_heavy']
        state_col = COL['state'].get(agent.state.value, COL['text_dim'])

        # Selected highlight
        if self.selected_id == agent.id:
            pygame.draw.circle(self.screen, COL['white'], (px, py), radius + 5, 2)

        # Outer state ring
        pygame.draw.circle(self.screen, state_col, (px, py), radius + 3)

        # Body
        pygame.draw.circle(self.screen, body_col, (px, py), radius)
        pygame.draw.circle(self.screen, COL['bg'], (px, py), radius, 1)

        # ID text
        lbl = self.f12.render(str(agent.id), True, COL['bg'])
        self.screen.blit(lbl, (px - lbl.get_width() // 2, py - lbl.get_height() // 2))

        # Battery arc (thin ring around the body)
        bat_pct = agent.battery / 100.0
        bat_col = (COL['green'] if agent.battery > 40
                   else COL['orange'] if agent.battery > 20 else COL['red'])
        arc_rect = pygame.Rect(px - radius - 5, py - radius - 5, (radius + 5) * 2, (radius + 5) * 2)
        angle_start = -math.pi / 2
        angle_end   = angle_start + bat_pct * 2 * math.pi
        if bat_pct > 0.01:
            points = [
                (px + (radius + 4) * math.cos(a), py + (radius + 4) * math.sin(a))
                for a in [angle_start + i * (angle_end - angle_start) / 20 for i in range(21)]
            ]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, bat_col, False, [(int(p[0]), int(p[1])) for p in points], 3)

        # Package carried
        if agent.carrying and agent.current_task:
            col  = agent.current_task.color
            pw, ph = max(10, CELL // 2), max(7, CELL // 3)
            pkx, pky = px - pw // 2, py - radius - ph - 3
            pygame.draw.rect(self.screen, col, (pkx, pky, pw, ph), border_radius=2)
            pygame.draw.rect(self.screen, COL['bg'], (pkx, pky, pw, ph), 1, border_radius=2)
            # Strapping lines
            pygame.draw.line(self.screen, COL['bg'],
                             (px, pky), (px, pky + ph), 1)
            pygame.draw.line(self.screen, COL['bg'],
                             (pkx, pky + ph // 2), (pkx + pw, pky + ph // 2), 1)

        # Charging indicator
        if agent.state == AgentState.CHARGING:
            t = pygame.time.get_ticks() / 500.0
            alpha = int(128 + 127 * math.sin(t))
            bolt_col = (*COL['blue'], alpha)
            lbl2 = self.f11.render("⚡", True, COL['blue'])
            self.screen.blit(lbl2, (px - lbl2.get_width() // 2, py - radius - 16))

    # ══════════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════════
    def _draw_sidebar(self) -> None:
        sx = GRID_PX
        pygame.draw.rect(self.screen, COL['panel_bg'],
                         pygame.Rect(sx, 0, SIDEBAR_W, WINDOW_H))
        pygame.draw.line(self.screen, COL['panel_bdr'],
                         (sx, 0), (sx, WINDOW_H), 1)

        x = sx + 12
        y = 10

        # ── Title bar ──
        pygame.draw.rect(self.screen, (18, 22, 38),
                         pygame.Rect(sx, 0, SIDEBAR_W, 40))
        title = self.f18.render("◈ WAREHOUSE AI", True, COL['agent_fast'])
        self.screen.blit(title, (x, y))
        speed_lbl = self.f10.render(
            f"tick {self.env.tick}  ·  {1000 // max(1, self.tick_ms)}×  "
            + ("▶" if not self.paused else "⏸"),
            True, COL['text_dim'],
        )
        self.screen.blit(speed_lbl, (x, y + 22))
        y += 48

        # ── KPI strip ──
        kpis = [
            ("Delivered", str(self.total_delivered),       COL['green']),
            ("Failed",    str(self.total_failed),           COL['red']),
            ("Eff%",      f"{self.dispatcher.efficiency()*100:.0f}%", COL['blue']),
            ("Charges",   str(self.env.charges_completed),  COL['yellow']),
        ]
        kw = (SIDEBAR_W - 24) // len(kpis)
        for i, (label, val, col) in enumerate(kpis):
            kx = x + i * kw
            pygame.draw.rect(self.screen, (18, 22, 36),
                             pygame.Rect(kx, y, kw - 4, 38), border_radius=4)
            self.screen.blit(self.f10.render(label, True, COL['text_dim']), (kx + 4, y + 3))
            self.screen.blit(self.f15.render(val, True, col), (kx + 4, y + 16))
        y += 46

        # ── Mini chart: efficiency ──
        y = self._draw_mini_chart(x, y, self.eff_hist, "Efficiency %",
                                  COL['blue'], 0, 100) + 6

        # ── Mini chart: queue depth ──
        y = self._draw_mini_chart(x, y, self.queue_hist, "Queue depth",
                                  COL['orange'], 0, MAX_QUEUE) + 8

        # ── Strategy bandit panel ──
        y = self._draw_strategy_panel(x, y) + 8

        # ── Agent fleet ──
        y = self._draw_agent_fleet(x, y)

        # ── Order queue ──
        y = self._draw_order_queue(x, y)

        # ── Event log ──
        self._draw_event_log(x)

        # ── Controls hint at bottom ──
        hint = "[SPC] pause  [+/-] speed  [R] reset  [Q] quit  click=select"
        ht = self.f10.render(hint, True, COL['text_dim'])
        self.screen.blit(ht, (sx + 8, WINDOW_H - 14))

    def _divider(self, x: int, y: int) -> int:
        pygame.draw.line(self.screen, COL['panel_bdr'],
                         (x - 4, y), (x + SIDEBAR_W - 20, y))
        return y + 6

    def _section_title(self, x: int, y: int, text: str) -> int:
        lbl = self.f10.render(text.upper(), True, COL['text_dim'])
        self.screen.blit(lbl, (x, y))
        return y + 14

    def _draw_mini_chart(
        self, x: int, y: int, data: list[float], title: str,
        col: tuple, ymin: float, ymax: float,
    ) -> int:
        cw, ch = SIDEBAR_W - 24, 36
        self._section_title(x, y, title)
        y += 12
        pygame.draw.rect(self.screen, (16, 20, 32),
                         pygame.Rect(x, y, cw, ch), border_radius=3)
        if len(data) >= 2:
            step = cw / max(len(data) - 1, 1)
            pts  = []
            for i, v in enumerate(data):
                nx = int(x + i * step)
                ny = int(y + ch - (v - ymin) / max(ymax - ymin, 1) * ch)
                pts.append((nx, max(y, min(y + ch, ny))))
            pygame.draw.lines(self.screen, col, False, pts, 2)
        # Latest value
        if data:
            val_lbl = self.f10.render(f"{data[-1]:.1f}", True, col)
            self.screen.blit(val_lbl, (x + cw - val_lbl.get_width() - 2, y + 2))
        return y + ch

    def _draw_strategy_panel(self, x: int, y: int) -> int:
        y = self._section_title(x, y, "UCB1 Strategy Bandit")
        avgs   = self.dispatcher.bandit.avg_rewards()
        best_i, best_name = self.dispatcher.bandit.best_arm()
        bar_w  = SIDEBAR_W - 24
        for i, name in enumerate(STRATEGY_NAMES):
            pulls = self.dispatcher.bandit.counts[i]
            avg   = avgs[i]
            is_best = i == best_i

            col = COL['green'] if is_best else COL['text_dim']
            prefix = "▶ " if is_best else "  "
            row_lbl = self.f10.render(
                f"{prefix}{name:<12} {avg:5.2f} ({pulls})",
                True, col,
            )
            self.screen.blit(row_lbl, (x, y))
            y += 13
        return y

    def _draw_agent_fleet(self, x: int, y: int) -> int:
        y = self._divider(x, y)
        y = self._section_title(x, y, "Fleet")
        selected = None

        for agent in self.env.agents:
            is_sel = agent.id == self.selected_id
            if is_sel:
                selected = agent
                continue
            y = self._draw_agent_row(x, y, agent, compact=True)
            if y > WINDOW_H - 220:
                break

        # Selected agent expanded
        if selected:
            pygame.draw.rect(self.screen, (20, 26, 46),
                             pygame.Rect(x - 4, y - 2, SIDEBAR_W - 16, 80),
                             border_radius=4)
            y = self._draw_agent_row(x, y, selected, compact=False)

        return y + 4

    def _draw_agent_row(self, x: int, y: int, agent: Agent, compact: bool) -> int:
        col = COL['agent_fast'] if agent.role == AgentRole.FAST else COL['agent_heavy']
        role_tag = "⚡F" if agent.role == AgentRole.FAST else "🏗H"
        state_name = agent.state.name[:6]

        # Left accent bar
        pygame.draw.rect(self.screen, col,
                         pygame.Rect(x - 4, y, 3, 28 if not compact else 20), border_radius=1)

        lbl = self.f11.render(
            f"R{agent.id} {role_tag}  {state_name:<6}  {agent.battery:.0f}%",
            True, col,
        )
        self.screen.blit(lbl, (x, y))
        y += 13

        # Battery bar
        bw  = SIDEBAR_W - 28
        bat_col = (COL['green'] if agent.battery > 40
                   else COL['orange'] if agent.battery > 20 else COL['red'])
        pygame.draw.rect(self.screen, (30, 35, 55), (x, y, bw, 5), border_radius=2)
        pygame.draw.rect(self.screen, bat_col,
                         (x, y, int(bw * agent.battery / 100), 5), border_radius=2)
        y += 8

        if not compact:
            spec_lbl = self.f10.render(
                f"  tasks={agent.tasks_completed}  spec={agent.specialization}"
                f"  ε={agent.ql.epsilon*100:.0f}%  Q-upd={agent.ql.updates}",
                True, COL['text_dim'],
            )
            self.screen.blit(spec_lbl, (x, y))
            y += 13
            if agent.current_task:
                tk   = agent.current_task
                t_lbl = self.f10.render(
                    f"  task #{tk.id}  {tk.priority.name}  {tk.category}  {tk.weight:.0f}kg",
                    True, tk.color,
                )
                self.screen.blit(t_lbl, (x, y))
                y += 13

        return y + 4

    def _draw_order_queue(self, x: int, y: int) -> int:
        y = self._divider(x, y)
        if y > WINDOW_H - 160:
            return y
        y = self._section_title(x, y, "Order Queue")

        pending = sorted(
            [t for t in self.dispatcher.pending_tasks if t.status.name == 'PENDING'],
            key=lambda t: (-t.priority_score, t.deadline),
        )[:6]

        for task in pending:
            if y > WINDOW_H - 130:
                break
            col = task.color
            tr  = max(0.0, task.time_remaining)
            urgency = task.urgency_ratio

            # Urgency bar
            bw = SIDEBAR_W - 24
            pygame.draw.rect(self.screen, (24, 28, 46), (x, y, bw, 3))
            fill_col = col if urgency < 0.7 else COL['red']
            pygame.draw.rect(self.screen, fill_col,
                             (x, y, int(bw * min(urgency, 1.0)), 3))
            y += 4

            # Left priority bar
            pygame.draw.rect(self.screen, col, (x - 4, y, 3, 14), border_radius=1)

            status_ico = {"PENDING": "⏳", "ASSIGNED": "🚗", "IN_TRANSIT": "📦"}.get(
                task.status.name, "?"
            )
            line = f"#{task.id:04d} {task.priority.name[:4]:<4} {task.category:<5}  {tr:.0f}s"
            lbl  = self.f10.render(line, True, COL['text'])
            self.screen.blit(lbl, (x, y))
            y += 15

        return y

    def _draw_event_log(self, x: int) -> None:
        y = WINDOW_H - 90
        pygame.draw.line(self.screen, COL['panel_bdr'],
                         (x - 4, y - 4), (x + SIDEBAR_W - 20, y - 4))
        self._section_title(x, y, "Event Log")
        y += 13
        for entry in self.event_log[-5:]:
            lbl = self.f10.render(entry[:42], True, COL['text_dim'])
            self.screen.blit(lbl, (x, y))
            y += 12
            if y > WINDOW_H - 20:
                break

    # ── Pause overlay ────────────────────────────────────────────────────────────
    def _draw_paused_overlay(self) -> None:
        s = pygame.Surface((GRID_PX, WINDOW_H), pygame.SRCALPHA)
        s.fill((0, 0, 0, 110))
        self.screen.blit(s, (0, 0))
        msg  = self.f20.render("⏸  PAUSED  —  [SPACE] to resume", True, COL['white'])
        self.screen.blit(msg, (GRID_PX // 2 - msg.get_width() // 2, WINDOW_H // 2))


# ─── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sim = WarehouseSimulation()
    sim.run()
