from __future__ import annotations

import unittest


class TestPanteonLegendProfile(unittest.TestCase):
    def test_profile_disables_flash_and_uses_v2_soft_selector(self) -> None:
        from panteon_legend.config import build_legend_profile

        profile = build_legend_profile()

        self.assertEqual(profile.name, "panteon_legend_v2")
        self.assertFalse(profile.flash_enabled)
        self.assertEqual(profile.trade_fraction, 0.10)
        self.assertEqual(profile.leverage, 3)
        self.assertEqual(profile.max_trades_per_period, 1000)
        self.assertEqual(profile.liquidity_min_adv, 5000)
        self.assertFalse(profile.apply_leverage_to_notional)
        self.assertEqual(profile.soft_selector_policy, "soft_regime_top1_24")
        self.assertEqual(profile.soft_selector_window_bars, 24)
        self.assertEqual(profile.soft_selector_min_closed_trades, 20)
        self.assertGreater(profile.soft_selector_min_score_to_trade, 0.0)
        self.assertFalse(profile.current_actionable_gate_enabled)
        self.assertTrue(profile.solo_current_actionable_gate_enabled)
        self.assertIn("Legend_Defensive", profile.execution_candidate_allow_labels)
        self.assertIn("Legend_RegimeSwitch", profile.execution_candidate_allow_labels)
        self.assertIn("Solo_MomentumScalper", profile.execution_candidate_allow_labels)
        self.assertNotIn("Legend_CurrentActorPool", profile.execution_candidate_allow_labels)

    def test_actor_pool_unites_current_core_and_legacy_style_specialists(self) -> None:
        from panteon_legend.config import legend_actor_labels

        labels = set(legend_actor_labels())

        for label in (
            "LiveOIBreakout",
            "LiveCrashHunter",
            "MomentumScalper",
            "VolBreakoutHunter",
            "ResearchValidatorAgent",
            "BullRotationAgent",
            "FundingArb",
            "NeutralLiquiditySweep",
            "AnchorFlowMomentum",
        ):
            self.assertIn(label, labels)

    def test_runtime_configs_replace_flash_gates_with_soft_selector(self) -> None:
        from panteon_legend.runtime import (
            legend_live_execution_config,
            legend_risk_config,
            legend_strategist_config,
        )

        risk = legend_risk_config()
        live = legend_live_execution_config()
        strategist = legend_strategist_config()

        self.assertEqual(risk.capital_fraction, 0.10)
        self.assertEqual(risk.max_leverage, 3)
        self.assertFalse(risk.apply_leverage_to_notional)
        self.assertEqual(risk.max_open_positions, 8)
        self.assertTrue(risk.floor_to_exchange_min_notional)
        self.assertEqual(live.max_new_opens_per_bar, 3)
        self.assertFalse(strategist.hard_policy_enabled)
        self.assertFalse(strategist.real_promotion_gate_enabled)
        self.assertFalse(strategist.v3_current_actionable_gate_enabled)
        self.assertTrue(strategist.v3_solo_current_actionable_gate_enabled)
        self.assertTrue(strategist.use_v3_rolling_score)
        self.assertTrue(strategist.use_v3_soft_shadow_score)
        self.assertEqual(strategist.v3_shadow_rolling_window_bars, 24)
        self.assertEqual(strategist.v3_shadow_rolling_min_closed_trades, 20)
        self.assertGreater(strategist.v3_min_score_to_trade, 0.0)
        self.assertEqual(strategist.v3_real_loss_kill_min_closed_trades, 0)
        self.assertIn("Legend_Defensive", strategist.v3_candidate_allow_labels)
        self.assertNotIn("Legend_CurrentActorPool", strategist.v3_candidate_allow_labels)

    def test_retrodate_config_is_five_year_no_flash_no_hard_policy(self) -> None:
        from panteon_legend.retrotest import build_legend_retrodate_config

        config = build_legend_retrodate_config(
            max_bars=25,
            include_optional_agents=True,
            optional_agent_labels=("GeneticsCore",),
        )

        self.assertEqual(config.years, (2022, 2023, 2024, 2025, 2026))
        self.assertTrue(config.include_optional_agents)
        self.assertEqual(config.optional_agent_labels, ("GeneticsCore",))
        self.assertFalse(config.flash_enabled)
        self.assertFalse(config.hard_policy_enabled)
        self.assertEqual(config.risk_capital_fraction, 0.10)
        self.assertEqual(config.risk_max_leverage, 3)
        self.assertFalse(config.apply_risk_leverage_to_notional)
        self.assertEqual(config.max_new_opens_per_bar, 3)
        self.assertEqual(config.risk_max_open_positions, 8)
        self.assertTrue(config.fixed_agent_players_enabled)
        self.assertGreaterEqual(config.solo_agent_candidate_limit, 12)
        self.assertTrue(config.use_v3_executable_soft_top1_score)
        self.assertFalse(config.current_actionable_candidate_layer_enabled)
        self.assertIn("Legend_Defensive", config.v3_candidate_allow_labels)
        self.assertNotIn("Legend_CurrentActorPool", config.v3_candidate_allow_labels)
        self.assertEqual(
            [item[0] for item in config.regime_switch_player_sets],
            ["Legend_RegimeSwitch"],
        )
        self.assertEqual(
            [item[0] for item in config.rotating_agent_player_sets],
            ["Legend_CurrentActorPool"],
        )
        self.assertEqual(
            [profile.label for profile in config.player_profiles],
            ["Legend_Consensus", "Legend_Bullish", "Legend_Defensive"],
        )

    def test_full_experiment_can_run_latest_genetics_build(self) -> None:
        from pathlib import Path

        from tools.run_legend_full_experiment import build_config

        config = build_config(
            "soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_dyn",
            Path("Results") / "unit",
            include_optional_agents=True,
        )

        self.assertTrue(config.include_optional_agents)
        self.assertIn("GeneticsCore", config.optional_agent_labels)
        self.assertIn("GeneticsResearch", [profile.label for profile in config.player_profiles])
        self.assertIn(
            ("Legend_GeneticsCore", ("GeneticsCore",)),
            config.fixed_agent_player_sets,
        )
        self.assertIn("GeneticsResearch", config.soft_allocator_execution_allow_labels)
        self.assertIn("Legend_GeneticsCore", config.soft_allocator_execution_allow_labels)
        self.assertTrue(config.soft_allocator_realized_gate_enabled)
        self.assertTrue(config.soft_allocator_execution_deny_actor_symbols)

    def test_full_experiment_soft_top1_24_profile_matches_offline_policy(self) -> None:
        from pathlib import Path

        from tools.run_legend_full_experiment import build_config

        config = build_config(
            "soft_regime_top1_24_only",
            Path("Results") / "unit",
            include_optional_agents=True,
        )

        policy = config.soft_allocator_execution_policy
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "soft_regime_top1_24")
        self.assertEqual(policy.top_k, 1)
        self.assertEqual(policy.rolling_window_bars, 24)
        self.assertEqual(policy.min_closed_trades, 20)
        self.assertEqual(policy.cash_reserve_weight, 0.0)
        self.assertEqual(policy.max_weight_per_leader, 1.0)
        self.assertEqual(policy.score_scope, "regime")
        self.assertTrue(config.soft_allocator_execution_enabled)
        self.assertTrue(config.soft_allocator_execution_soft_only)
        self.assertFalse(config.soft_allocator_realized_gate_enabled)
        self.assertEqual(config.soft_allocator_execution_allow_labels, ())
        self.assertIn("GeneticsResearch", [profile.label for profile in config.player_profiles])
        self.assertIn(
            ("Legend_GeneticsBullish", ("GeneticsBullish",)),
            config.fixed_agent_player_sets,
        )

    def test_full_experiment_soft_top1_24_confirmed_profile_matches_offline_policy(self) -> None:
        from pathlib import Path

        from tools.run_legend_full_experiment import build_config

        config = build_config(
            "soft_regime_top1_24_confirmed_only",
            Path("Results") / "unit",
            include_optional_agents=True,
        )

        policy = config.soft_allocator_execution_policy
        self.assertIsNotNone(policy)
        self.assertEqual(policy.name, "soft_regime_top1_24_confirmed")
        self.assertEqual(policy.top_k, 1)
        self.assertEqual(policy.rolling_window_bars, 24)
        self.assertEqual(policy.min_closed_trades, 50)
        self.assertEqual(policy.cash_reserve_weight, 0.0)
        self.assertEqual(policy.max_weight_per_leader, 1.0)
        self.assertEqual(policy.score_scope, "regime")
        self.assertTrue(config.soft_allocator_execution_enabled)
        self.assertTrue(config.soft_allocator_execution_soft_only)
        self.assertFalse(config.soft_allocator_realized_gate_enabled)
        self.assertEqual(config.soft_allocator_execution_allow_labels, ())


if __name__ == "__main__":
    unittest.main()
