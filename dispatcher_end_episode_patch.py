# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER CHANGE — only end_episode() needs updating.
# Replace the existing end_episode method in dispatcher.py with this:
# ─────────────────────────────────────────────────────────────────────────────

    def end_episode(self) -> dict:
        """
        Call when a simulation run (episode) completes.
        Updates MAPPO weights AND pushes cost overlay to agents,
        then saves all models.
        """
        self.episode += 1

        # ── OLD (v3): self.mappo.update()
        # ── NEW (v5): update weights + push overlay so A* is affected
        if self.mappo:
            self.mappo.update_and_apply(
                agents     = self.env.agents,
                grid_size  = self.env.grid_size,
                env        = self.env,          # passes congestion_cost in
            )

        summary = self.summary_dict()

        if self.episode % MODEL_SAVE_INTERVAL == 0:
            self.store.save_all(
                mappo     = self.mappo,
                qlearners = [a.ql for a in self.env.agents],
                bandit    = self.bandit,
                heatmap   = self.env.heatmap,
                episode   = self.episode,
                metrics   = summary,
            )

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# That's the only change needed in dispatcher.py.
# Everything else in dispatcher.py stays exactly as-is.
# ─────────────────────────────────────────────────────────────────────────────
