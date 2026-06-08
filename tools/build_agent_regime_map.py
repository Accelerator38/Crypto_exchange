"""Roadmap #4: карта «агент × режим» из ретро-лидербордов.

Строит матрицу прибыльности каждого актора по 8 режимам (BULLISH, BEARISH,
NEUTRAL, CRASH, RANGE_LOW_VOL, CHOPPY_DOWN, CHOPPY_UP, MIXED_ROTATIONAL) на
основе per_regime-разбивки из leaderboard_players.json одного или нескольких
ретро-прогонов. Выдаёт:
  • markdown-таблицу agent × regime (pnl_per_trade %, в скобках closed);
  • «лучший агент на режим» (routing-кандидаты);
  • JSON routing-map: regime -> [агенты с положительным per-trade эджем и выборкой].

Использование:
  python tools/build_agent_regime_map.py <retro_dir> [<retro_dir> ...] \
      [--min-closed 10] [--out Reports/agent_regime_map]

retro_dir — папка прогона (содержит leaderboard_players.json и/или
leaderboard_agents.json). Несколько папок суммируются (closed-взвешенно).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

REGIMES = [
    "bullish", "bearish", "neutral", "crash",
    "range_low_vol", "choppy_down", "choppy_up", "mixed_rotational",
]


def _load_pool(retro_dir: str) -> Dict[str, dict]:
    pool: Dict[str, dict] = {}
    for fn in ("leaderboard_players.json", "leaderboard_agents.json"):
        path = os.path.join(retro_dir, fn)
        if not os.path.isfile(path):
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        actors = data.get("players") or data.get("agents") or {}
        for label, payload in actors.items():
            if isinstance(payload, dict):
                pool.setdefault(str(label), payload)
    return pool


def _clean_label(label: str) -> str:
    # V_LiveMeanRev -> LiveMeanRev; agent:X -> X; ensemble:X -> X
    for pref in ("V_", "agent:", "ensemble:", "player:", "Solo_"):
        if label.startswith(pref):
            label = label[len(pref):]
    return label


def _accumulate(dirs: List[str]) -> Dict[str, Dict[str, Tuple[float, int, float]]]:
    """label -> regime -> (sum_pnl_pct, sum_closed, sum_wins) (closed-взвешенно)."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
    for d in dirs:
        pool = _load_pool(d)
        for raw_label, payload in pool.items():
            label = _clean_label(raw_label)
            per_regime = payload.get("per_regime") or {}
            for regime, m in per_regime.items():
                if not isinstance(m, dict):
                    continue
                reg = str(regime).lower()
                closed = int(m.get("closed_trades", 0) or 0)
                if closed <= 0:
                    continue
                pnl = float(m.get("pnl_pct", 0.0) or 0.0)
                win = float(m.get("win_rate", 0.0) or 0.0)
                cell = acc[label][reg]
                cell[0] += pnl
                cell[1] += closed
                cell[2] += win * closed / 100.0  # wins (approx)
    out: Dict[str, Dict[str, Tuple[float, int, float]]] = {}
    for label, regs in acc.items():
        out[label] = {r: (v[0], int(v[1]), v[2]) for r, v in regs.items()}
    return out


def build(dirs: List[str], min_closed: int) -> dict:
    acc = _accumulate(dirs)
    rows = []
    routing: Dict[str, List[dict]] = {r: [] for r in REGIMES}
    for label, regs in sorted(acc.items()):
        row = {"agent": label, "regimes": {}}
        for reg in REGIMES:
            pnl, closed, wins = regs.get(reg, (0.0, 0, 0.0))
            ppt = (pnl / closed) if closed > 0 else 0.0
            row["regimes"][reg] = {"ppt": ppt, "closed": closed, "cum_pnl": pnl}
            if closed >= min_closed and ppt > 0.0:
                routing[reg].append({"agent": label, "ppt": round(ppt, 4), "closed": closed})
        rows.append(row)
    for reg in routing:
        routing[reg].sort(key=lambda x: -x["ppt"])
    return {"rows": rows, "routing": routing, "min_closed": min_closed, "sources": dirs}


def to_markdown(report: dict) -> str:
    lines = ["# Карта агент × режим (per-trade PnL %, в скобках closed)", ""]
    short = {r: r[:6] for r in REGIMES}
    header = "| agent | " + " | ".join(short[r] for r in REGIMES) + " |"
    sep = "|---|" + "|".join("---" for _ in REGIMES) + "|"
    lines += [header, sep]
    for row in report["rows"]:
        cells = []
        for reg in REGIMES:
            c = row["regimes"][reg]
            if c["closed"] <= 0:
                cells.append("·")
            else:
                cells.append(f"{c['ppt']*100:+.2f}({c['closed']})")
        lines.append(f"| {row['agent']} | " + " | ".join(cells) + " |")
    lines += ["", "## Routing-кандидаты (положительный per-trade эдж, выборка ≥ min_closed)", ""]
    for reg in REGIMES:
        top = report["routing"][reg][:5]
        if top:
            picks = ", ".join(f"{t['agent']}({t['ppt']*100:+.2f}%/{t['closed']})" for t in top)
        else:
            picks = "—"
        lines.append(f"- **{reg}**: {picks}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build agent×regime profitability map from retro leaderboards.")
    ap.add_argument("dirs", nargs="+", help="Retro output dir(s) with leaderboard_players.json")
    ap.add_argument("--min-closed", type=int, default=10)
    ap.add_argument("--out", default="Reports/AgentRegimeMap/agent_regime_map")
    args = ap.parse_args(argv)

    # Разворачиваем glob-паттерны.
    dirs: List[str] = []
    for d in args.dirs:
        matched = glob.glob(d)
        dirs.extend(matched if matched else [d])
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("No valid retro dirs found.")
        return 2

    report = build(dirs, args.min_closed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    md = to_markdown(report)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    # Консоль может быть cp1251 — печатаем безопасно.
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))
    print(f"\nWritten: {args.out}.md / .json  (sources: {len(dirs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
