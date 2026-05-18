from pathlib import Path

from panteon_v2.analysis.allocation_diagnostics import analyze_trading_log


def test_analyze_trading_log_counts_no_trade_and_actionability(tmp_path: Path):
    log = tmp_path / "trading.log"
    log.write_text(
        "\n".join([
            (
                "2026-01-01 bar=1 regime=neutral leader=NoTrade "
                "selected_leader=NoTrade executed_leader=NoTrade "
                "raw_signals=0 signals=0 filled=0"
            ),
            (
                "2026-01-01 bar=2 regime=neutral leader=Alpha "
                "selected_leader=Alpha executed_leader=Alpha "
                "raw_signals=0 signals=0 filled=0"
            ),
            (
                "2026-01-01 bar=3 regime=neutral leader=Alpha "
                "selected_leader=Alpha executed_leader=Alpha "
                "raw_signals=2 signals=1 filled=1"
            ),
        ]),
        encoding="utf-8",
    )

    report = analyze_trading_log(log)

    assert report.bars == 3
    assert report.no_trade_bars == 1
    assert report.no_trade_share_pct == 100.0 / 3.0
    assert report.raw_zero_share_pct == 200.0 / 3.0
    assert report.filled_zero_share_pct == 200.0 / 3.0
    assert report.by_leader["Alpha"].bars == 2
    assert report.by_leader["Alpha"].raw_zero_bars == 1
    assert report.by_leader["Alpha"].filled == 1
