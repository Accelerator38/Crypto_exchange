"""Read-only evidence analysis; add a cash-caption clarification, never replay."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Reports/Exia/Genetic_Primus/compact_ioc_engineering_v1_20261008"
OUTPUT = EVIDENCE / "clarified_view"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("clarified evidence view already exists")
    receipt = json.loads((EVIDENCE / "receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, digest in receipt["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("pinned source mismatch")
    for name, digest in manifest["files"].items():
        if sha(EVIDENCE / name) != digest:
            raise ValueError("pinned artifact mismatch")
    results = json.loads((EVIDENCE / "synthetic_results.json").read_text(encoding="utf-8"))
    flat = results["flat_price_round_trip"]
    base = flat["equity"][0]["ts"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for name, label, style in (("NoTrade", "NoTrade", "--"),
                               ("flat_price_round_trip", "Круговая сделка: spread + fees + funding", "-")):
        points = results[name]["equity"]
        axes[0].plot([0] + [(p["ts"] - base) / 1000 for p in points],
                     [1000] + [float(p["equity_quote"]) for p in points], style, label=label, marker=".")
    axes[1].plot([(p["ts"] - base) / 1000 for p in flat["equity"]],
                 [float(p["cash_quote"]) for p in flat["equity"]], marker=".", color="tab:orange")
    axes[0].set(title="СИНТЕТИЧЕСКАЯ инженерная проверка — не результат стратегии", ylabel="NAV, USDT")
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel="Время событий от условного close, секунды", ylabel="Балансовый cash, USDT")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    OUTPUT.mkdir(exist_ok=False)
    fig.savefig(OUTPUT / "synthetic_equity.png", dpi=140)
    fig.savefig(OUTPUT / "synthetic_equity.svg")
    plt.close(fig)
    fee, funding, spread = flat["fees_quote"], flat["signed_funding_quote"], Decimal(".201") * Decimal(".2")
    report = f"""# Compact IOC: инженерный результат, 8 октября 2026

Реализован отдельный ограниченный offline execution kernel. Это не запуск
collector, не обучение и не исторический тест финансовой стратегии.

## Результаты

37 из 37 явных unittest прошли; дополнительно сохранены 8 синтетических
сценариев. Проверены post-close availability и latency, свежесть и clocks
книги, qty/price grid, minimums, защищённая цена, shared liquidity, partial IOC,
отмена остатка, expiry, cash и reduce-only reserves, signed funding,
drawdown-блок новых buys и deterministic checkpoint restore.

При неизменном mark 100 круговая сделка qty 0.201:
buy 100.1 → sell 99.9; начальный NAV 1000 USDT, итог {flat['equity_quote']}.
Spread = {spread} USDT; fees = {fee}; funding debit = {funding}.
Итог точно равен `1000 − spread − fees − funding`; позиция закрыта.
NoTrade остаётся 1000. Это ожидаемый убыток от издержек без alpha,
а не сравнение торговых моделей. Возобновлённый результат побитово по
каноническому JSON совпадает с непрерывным.

![Синтетический NAV и балансовый cash](synthetic_equity.png)

Нижний график показывает балансовый cash. Для доступного cash необходимо
дополнительно вычесть резервы открытых buy IOC; резервы уже контролируются
ядром. Подпись «Свободный cash» в первоначальной иллюстрации была неточной.
Исходная иллюстрация и evidence не переписаны; этот view исправляет только
подпись и пояснение, без перерасчёта ledger или изменения результата.

## Ограничения и следующий шаг

Старое исполнение на next-4h-open и новый post-close clock несовместимы:
старые labels и model hashes нельзя считать заново квалифицированными.
Прочитан лишь provenance одного frozen linear origin; inference и fitting
не выполнялись. Параметры latency/participation/risk пока инженерные примеры.
Верификация реальных fills, margin/liquidation и production adapter отсутствует.

Следующий финансовый шаг требует нового target-clock/label контракта и
исторического L2 с receive/exchange clocks и корректным восстановлением
книги, доказанных rules/fees и observed funding marks за весь интервал.
До этого август–сентябрь не пересчитывается; старые прибыльности и LCB
не меняются. Синтетические volumes не заменяют настоящую ликвидность.

Источники и artifacts сверены с исходным receipt: 10 source hashes и 9
artifact hashes. Новый view привязан к первоначальному manifest отдельным
manifest. Новых fits, market replays и network requests: 0. Все девять
authority/safety flags false; реальные торги и paper не включены.
"""
    with (OUTPUT / "REPORT_RU.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    body = {"schema": "engineering_clarified_view/1", "reason": "cash balance is not cash net of reservations",
            "original_manifest_sha256": sha(EVIDENCE / "artifact_hashes.json"),
            "original_receipt_sha256": sha(EVIDENCE / "receipt.json"),
            "analysis_source_sha256": sha(Path(__file__)), "no_new_event_or_market_replay": True,
            "files": {path.name: sha(path) for path in sorted(OUTPUT.iterdir())}}
    with (OUTPUT / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"sources_verified": len(receipt["source_hashes"]), "artifacts_verified": len(manifest["files"]),
                      "caption_clarified": True, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
