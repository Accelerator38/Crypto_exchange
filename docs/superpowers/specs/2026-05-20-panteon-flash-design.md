# Panteon Flash Design

Date: 2026-05-20

## Goal

Panteon_Flash removes the global player-selection layer from the production decision path. On each bar, Panteon selects the best available actor for each cryptocurrency independently, because BTC, ETH, SOL, and other symbols can be in different actionable situations even when the bar has one global market regime.

## Core Design

The new decision kernel is `FlashAllocator`. It receives the current `MarketSnapshot`, a list of composed ensemble players, registered solo agents, current memory, quarantine state, and recent shadow/actionability state. It returns one `FlashDecision` per symbol.

Each decision is explicit:

- `symbol`: market symbol being decided.
- `selected_actor`: chosen actor label, or `NoTrade`.
- `actor_type`: `agent`, `ensemble`, or `no_trade`.
- `score`: comparable numeric score.
- `action`: resulting action for that symbol.
- `reason`: short machine-readable reason.
- `candidates`: full ranked audit rows with rejection reasons.
- `signal`: executable `Signal` when the selected actor produced an actionable symbol-level signal.

Ensemble players remain first-class statistical entities, but they are marked as ensemble actors rather than being a separate selection layer. Their shadow, memory, attribution, and visualizations stay separate from solo agents so debugging can answer both questions: "which individual agent worked?" and "which ensemble recipe worked?"

## Execution Flow

Legacy v3 path remains available for rollback and A/B comparisons. Flash path is opt-in through a pipeline flag.

When Flash mode is enabled:

1. Compose the existing profile and rotating-player candidates as before.
2. Run the existing shadow tournament so current shadow/actionability data stays available.
3. Build solo-agent candidates from `AgentRegistry`.
4. Ask `FlashAllocator` for one decision per market symbol.
5. Execute only the accepted per-symbol signals.
6. Emit structured debug payloads and status fields that preserve every per-symbol decision.

The old `Strategist.current_leader` does not select the real Flash signals. It can remain initialized for non-Flash runs and for tests that compare legacy behavior.

## Deployment Knobs

Flash can be enabled from exchange profile settings with `panteon_flash_enabled`, `v2_flash_enabled`, or `flash_enabled`. Environment fallbacks are `PANTEON_FLASH_ENABLED`, `PANTEON_FLASH`, `{EXCHANGE}_PANTEON_FLASH_ENABLED`, and `{EXCHANGE}_PANTEON_FLASH`.

Allocator scoring knobs:

- `panteon_flash_min_score_to_trade` / `v2_flash_min_score_to_trade` / `flash_min_score_to_trade`
- `panteon_flash_actionable_bonus` / `v2_flash_actionable_bonus` / `flash_actionable_bonus`
- `panteon_flash_no_data_score` / `v2_flash_no_data_score` / `flash_no_data_score`

## Scoring Policy

The first Flash implementation should be conservative and transparent:

- Reject quarantined actors before scoring.
- Reject actors that produce no non-HOLD action for the specific symbol.
- Score solo agents from existing `PerformanceMemory` per-regime metrics.
- Score ensemble actors from their own player metrics when available, otherwise from the mean of their component-agent metrics.
- Apply a small current-actionability bonus when shadow state reports the actor as actionable for this bar.
- Prefer higher score, then actionable signal, then stable deterministic label order.

This deliberately avoids a new complex model. The value of Flash is inspectability: every symbol gets a ranked candidate table that can later feed better opportunity models.

## Observability

Flash must make debugging easier than v3:

- `StepResult.causal_decision` includes `flash_enabled`, `flash_decisions`, and `flash_selected_actors_by_symbol`.
- Each row shows selected actor, actor type, action, score, reason, and top rejected candidates.
- `Signal.by_player` uses the selected actor label. For solo agents, `Signal.by_agent` is the same agent label. For ensemble actors, `Signal.by_agent` remains the main contributor.
- Dashboards keep separate agent and player/ensemble leaderboards.

## Testing

Focused tests cover the new kernel before production wiring:

- Per-symbol selection can choose different actors for BTC and ETH on the same bar.
- A quarantined actor is rejected even if it has the strongest signal.
- An ensemble actor is marked as `ensemble` and remains visible in decision audit rows.
- Flash mode in the main loop executes per-symbol decisions without using global selected leader fallback.

## Out Of Scope

Flash does not yet add a learned opportunity-quality model for shadow position replay. The allocator produces the structured data needed for that next step.
