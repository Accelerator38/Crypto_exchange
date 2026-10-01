# GA episode placement: structural stop

Pre-registered contract: `configs/genetic_primus_episode_attribution_v2.json` (committed before this diagnostic). Evidence class: revealed historical development, no promotion authority.

The archived schedule has 1,296 decision rows, 309 EMA-long eligible bars and 288 GA-long bars. An exact-exposure binary control can differ on at most `2 × (309 − 288) = 42` rows. The pre-registered minimum was 50 changed rows per seed. Thus even the most diverse admissible placement cannot qualify. The runner stopped before loading market prices or computing any PnL. No seed, control return, or bootstrap statistic was inspected or selected.

This is a non-identifiability finding for this particular matched-control design, not evidence that GA has or lacks incremental edge. The earlier episode-reversal experiment changed only 6–20 decisions per seed and must likewise be read as a weak diagnostic. GA search, prospective sealing, paper and live promotion remain closed.
