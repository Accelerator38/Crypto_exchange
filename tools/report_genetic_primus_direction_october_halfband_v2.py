"""Independent statistics audit and evidence-first report. No fitting or network."""
from __future__ import annotations
import run_genetic_primus_direction_october_halfband_v2 as adapter
import json
from pathlib import Path
import numpy as np

run=adapter.old
ROOT,OUT=run.ROOT,adapter.OUT

def metric(cm,name):
    cm=np.asarray(cm,float);diagonal=np.diagonal(cm,axis1=-2,axis2=-1)
    if name=="accuracy":return diagonal.sum(axis=-1)/cm.sum(axis=(-2,-1))
    den=cm.sum(axis=-1)+cm.sum(axis=-2)
    return np.divide(2*diagonal,den,out=np.zeros_like(diagonal),where=den!=0).mean(axis=-1)

def audit():
    c,reg=run.registered();results=run.read(OUT/"results.json");weights=[]
    for seed in c["bootstrap"]["seeds"]:
        with np.load(OUT/("bootstrap_bank_%s.npz"%seed),allow_pickle=False) as a:
            assert np.array_equal(a["indices"],run.reuse.causal.block_bank(13,1000,3,seed))
            weights.append(np.stack([np.bincount(row,minlength=13) for row in a["indices"]]))
    obs=[];all_draws=[[],[]];raw_checked=0
    for h in c["timeframes_minutes"]:
        rr=run.read(OUT/("results_%sm.json"%h))
        with np.load(OUT/("daily_confusion_%sm.npz"%h),allow_pickle=False) as a:daily=a["daily"]
        for row in rr["rows"]:
            target=c["evaluation_band_factors"].index(row["evaluation_band_factor"]);asset=c["symbols"].index(row["symbol"]);k=run.NAMES.index(row["method"])
            cm=daily[target,asset,k].sum(axis=0)
            assert np.array_equal(cm,row["confusion_down_neutral_up"])
            for name in ("accuracy","macro_f1"):assert abs(metric(cm,name)-row[name])<1e-15
        with np.load(OUT/("paired_draws_%sm.npz"%h),allow_pickle=False) as a:
            for bank in range(2):
                saved=a["bank%s"%bank];cache={};checked=[]
                for j,row in enumerate(rr["comparisons"]):
                    asset=c["symbols"].index(row["symbol"]);values=[]
                    for method in (row["candidate"],row["control"]):
                        key=(asset,method)
                        if key not in cache:cache[key]=np.einsum("bd,dij->bij",weights[bank],daily[1,asset,run.NAMES.index(method)])
                        values.append(metric(cache[key],row["metric"]))
                    difference=values[0]-values[1]
                    assert np.allclose(difference,saved[:,j],atol=1e-15,rtol=0);checked.append(difference)
                all_draws[bank].append(np.array(checked).T)
        obs.extend(r["difference"] for r in rr["comparisons"])
        # Direct raw-price boolean counters, not the confusion helper.
        opposite=np.zeros((12,8),np.int64);calls=np.zeros_like(opposite)
        for number in range(5):
            p=OUT/("origin_%02d"%number)/("tf_%sm.npz"%h)
            with np.load(p,allow_pickle=False) as a:
                for k,pred in enumerate(run.predictions(a,h,c)):
                    opposite[k]+=(((pred==1)&(a["return_pct"]<0))|((pred==-1)&(a["return_pct"]>0))).sum(axis=0)
                    calls[k]+=(pred!=0).sum(axis=0)
        for k,method in enumerate(run.NAMES):
            for asset,coin in enumerate(c["symbols"]):
                row=next(r for r in rr["rows"] if r["method"]==method and r["symbol"]==coin and r["evaluation_band_factor"]==0)
                assert opposite[k,asset]==row["opposite_count"] and calls[k,asset]==row["directional_calls"];raw_checked+=1
    obs=np.array(obs);criticals=[]
    for bank in range(2):
        draws=np.concatenate(all_draws[bank],axis=1);se=draws.std(axis=0,ddof=1);active=se>1e-12
        critical=max(0.,float(np.quantile(((obs[None,active]-draws[:,active])/se[None,active]).max(axis=1),.95)))
        assert np.allclose(obs-critical*se,[r["simultaneous_lcb_bank%s"%bank] for r in results["comparisons"]],atol=1e-14,rtol=0);criticals.append(critical)
    for p,sha in reg["data_pins"].items():assert run.digest(ROOT/p)==sha,p
    old_pins={}
    for directory,names in [("direction_ternary_hypotheses_v1_20261014",["delivery_manifest.json"]),("direction_vector_errors_v1_20261014",["delivery_manifest.json","report_adapter_manifest.json"])]:
        for name in names:old_pins.update(run.read(ROOT/"Reports/Exia/Genetic_Primus"/directory/name)["artifact_pins"])
    for p,sha in old_pins.items():assert run.digest(ROOT/p)==sha,p
    prior=run.read(ROOT/c["previous_registry"]);current=run.read(adapter.REGISTRY);assert current["objectives"][:-1]==prior["objectives"]
    run.write(OUT/"comparison_audit.json",dict(independent_metric_cells=1152,paired_comparisons=len(obs),raw_boolean_cells=raw_checked,
        independently_recomputed_max_t_criticals=criticals,old_artifact_pins_unchanged=len(old_pins),registered_data_pins_verified=len(reg["data_pins"]),
        previous_registry_entries_unchanged=True,estimator_fits=0,report_tool_sha256=run.digest(Path(__file__))))
    print(json.dumps(dict(metric_cells=1152,paired_comparisons=512,raw_boolean_cells=raw_checked,old_pins=len(old_pins),fits=0)))

def fmt(value,multiplier=100):return "—" if value is None else "%.2f"%(value*multiplier)

def report():
    import matplotlib.pyplot as plt
    c,reg=run.registered();result=run.read(OUT/"results.json");summary=run.read(OUT/"summary.json");verification=run.read(OUT/"comparison_audit.json")
    assert sum(run.read(OUT/("audit_%sm.json"%h))["models"] for h in c["timeframes_minutes"])==80
    rows=result["rows"];medians=[]
    for h in c["timeframes_minutes"]:
        for target in c["evaluation_band_factors"]:
            for method in run.NAMES:
                rr=[r for r in rows if r["minutes"]==h and r["method"]==method and r["evaluation_band_factor"]==target]
                values={}
                for key in ("accuracy","macro_f1","actual_neutral_fraction","directional_call_fraction","opposite_per_directional_call","opposite_raw_sign_per_directional_call","opposite_raw_sign_given_actual_move"):
                    a=[r[key] for r in rr if r[key] is not None];values[key]=None if not a else float(np.median(a))
                medians.append(dict(minutes=h,method=method,evaluation_band_factor=target,**values))
    names=run.NAMES[:4]+["random_uniform_3","majority_half","momentum_half"]
    fig,axes=plt.subplots(2,2,figsize=(16,9),layout="constrained")
    for ax,h in zip(axes.flat,c["timeframes_minutes"]):
        matrix=np.array([[next(r["accuracy"] for r in rows if r["minutes"]==h and r["method"]==method and r["symbol"]==coin and r["evaluation_band_factor"]==.5) for coin in c["symbols"]] for method in names])*100
        im=ax.imshow(matrix,vmin=20,vmax=65,cmap="viridis",aspect="auto");ax.set_xticks(range(8),[s.split('/')[0] for s in c["symbols"]]);ax.set_yticks(range(len(names)),names,fontsize=9)
        ax.set_title("%s мин — общие половинные метки, accuracy %%"%h)
        for i in range(len(names)):
            for j in range(8):ax.text(j,i,"%.1f"%matrix[i,j],ha="center",va="center",fontsize=9,color="white" if matrix[i,j]<45 else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Правильные прогнозы, %",shrink=.7);fig.savefig(OUT/"half_truth_accuracy_by_coin.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for ax,h in zip(axes.flat,c["timeframes_minutes"]):
        matrix=np.array([[next(r["accuracy"] for r in medians if r["minutes"]==h and r["method"]==method and r["evaluation_band_factor"]==factor) for method in run.NAMES[:4]] for factor in (1,.5)])*100
        im=ax.imshow(matrix,vmin=20,vmax=65,cmap="viridis",aspect="auto");ax.set_xticks(range(4),run.NAMES[:4],rotation=15);ax.set_yticks([0,1],["Исходная truth","Половинная truth"]);ax.set_title("%s мин — сравнивать внутри строки"%h)
        for i in range(2):
            for j in range(4):ax.text(j,i,"%.2f%%"%matrix[i,j],ha="center",va="center",color="white" if matrix[i,j]<45 else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Медиана accuracy по валютам, %",shrink=.7);fig.savefig(OUT/"cross_scoring.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(16,9),layout="constrained")
    for ax,h in zip(axes.flat,c["timeframes_minutes"]):
        matrix=np.array([[np.nan if (v:=next(r["opposite_raw_sign_per_directional_call"] for r in rows if r["minutes"]==h and r["method"]==method and r["symbol"]==coin and r["evaluation_band_factor"]==.5)) is None else 100*v for coin in c["symbols"]] for method in names])
        im=ax.imshow(matrix,vmin=25,vmax=65,cmap="YlOrRd",aspect="auto");ax.set_xticks(range(8),[s.split('/')[0] for s in c["symbols"]]);ax.set_yticks(range(len(names)),names,fontsize=9);ax.set_title("%s мин — противоположный знак / направленные вызовы"%h)
        for i in range(len(names)):
            for j in range(8):ax.text(j,i,"—" if np.isnan(matrix[i,j]) else "%.1f"%matrix[i,j],ha="center",va="center",color="white" if matrix[i,j]>48 else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Ошибки знака, % направленных вызовов",shrink=.7);fig.savefig(OUT/"opposite_raw_by_coin.png",dpi=150);plt.close(fig)
    blocks=[];fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for ax,h in zip(axes.flat,c["timeframes_minutes"]):
        with np.load(OUT/("daily_confusion_%sm.npz"%h),allow_pickle=False) as a:daily=a["daily"][1]
        for method in ("original_boost","half_boost","majority_half","random_uniform_3"):
            cm=daily[:,run.NAMES.index(method)].sum(axis=0);cumulative=cm.cumsum(axis=0)
            ax.plot(range(1,14),metric(cumulative,"accuracy")*100,marker=".",label=method)
            for block,start in enumerate(range(0,13,3)):
                part=cm[start:min(start+3,13)].sum(axis=0);blocks.append(dict(minutes=h,method=method,block=block,start_day_october=start+1,end_day_october=min(start+3,13),accuracy=float(metric(part,"accuracy")),n=int(part.sum())))
        ax.set_title("%s мин — накопленная accuracy, общая половинная truth"%h);ax.set_xlabel("День октября, UTC");ax.set_ylabel("Правильные прогнозы, %");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(OUT/"cumulative_accuracy_by_day.png",dpi=150);plt.close(fig)
    lines=["# Октябрь: переобучение на половинной нейтральной полосе","","Период нового исторического теста: 1–13 октября 2026 UTC. Дата расчёта: 14 октября 2026.","",
        "## Итог","","На общей половинной truth заново обученные half-модели точнее исходных моделей, применённых к этим более узким меткам. Но общий устойчивый выигрыш по всем зарегистрированным контролям и обеим метрикам не подтверждён: полный исследовательский gate **0/64**. Ни 80%, ни 90% accuracy не получены. Особенно слабое место — 4h: half-модели по медиане уступили majority_half. Это короткий новый исторический тест, не prospective и не финальная независимая валидация.","",
        "| Горизонт | Original logit на half truth, % | Half logit на той же truth, % | Original boost на half truth, % | Half boost на той же truth, % | Majority half, % | Random 3, % |","|---:|---:|---:|---:|---:|---:|---:|"]
    for h in c["timeframes_minutes"]:
        values=[next(r["accuracy"] for r in medians if r["minutes"]==h and r["method"]==m and r["evaluation_band_factor"]==.5) for m in ["original_logit","half_logit","original_boost","half_boost","majority_half","random_uniform_3"]]
        lines.append("| %s мин | %s |"%(h," | ".join(fmt(v) for v in values)))
    lines += ["","Все значения — медианы по восьми валютам. Разность медиан не является парной оценкой эффекта; все paired comparisons и их одновременные LCB сохранены в results.json.","",
        "## Данные и квалификация","","Разрешённая публичная REST-загрузка Bitget USDT-FUTURES market candles: 149760 строк 1m по ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP. Каждый актив — 18720 минут; последний open time 13.10 23:59 UTC. 40 symbol/batch parquet, один общий october_full8_1m.parquet, 216 raw ответов с SHA; дополнительно был один предварительный запрос 10 минут BTC. Пропусков, дублей и сдвигов нет. Хвосты 100 минут совпали между candles/history-candles в 40 случаях; оба endpoint одного провайдера, это не независимая проверка цен. Граница старого и нового массива: open/предыдущий close совпали для всех активов.","",
        "Описание формата и ограничений endpoint: [официальная документация Bitget](https://www.bitget.com/docs/catalog/market/market-data). Это OHLCV last-trade/market candles, не стакан, mark/funding или доказательство исполнимости сделок. Загрузка записана отдельно от research и runtime; все network_allowed/credentials/orders/paper/live/runtime/promotion флаги моделей false.","",
        "Историческая часть исходных OHLCV и всех 12 признаков воспроизвелась точно на четырёх горизонтах. Новые признаки вычисляются тем же frozen direction_multiscale_v1/features pipeline. Входные данные, code, scaler, folds, model SHA и cutoffs закреплены в registration.json и метаданных моделей.","",
        "## Постановка и причинность","","r = 100 × (close[t+h] / close[t] − 1). Down: r<−δ; neutral: |r|≤δ; up: r>δ. Исходные δ: 0,020 / 0,064 / 0,109 / 0,320%; half δ: 0,010 / 0,032 / 0,0545 / 0,160% для 1m/10m/30m/240m. Zero-band — диагностический чистый знак с отдельным exact-zero neutral. Все прогнозы делаются на каждом допустимом интервале, покрытие 100%; часть прогнозов neutral.","",
        "Pooled обучение по восьми активам; результаты отдельно по валютам. Пять origins: 1/4/7/10/13 октября; последний блок один день. Rolling 30d включает 7d calibration, фактический train примерно 23d минус purge. Calibration не использовалась для настройки. Только созревшие labels; purge 7 баров соответствующего горизонта; endpoints строго до cutoffs. На каждом origin новый невзвешенный scaler mean/std, только train; uniform веса всех трёх исходов. Общая выборка максимум 4000 временных строк для обеих полос; одинаковый seed и preprocessing.","",
        "## Архитектуры и бюджет","","Logistic: 12 признаков, 39 номинальных multiclass коэффициентов (softmax избыточность), C=0,1, max_iter=120, tol=1e−6. Boosting: 32 итерации, 3 дерева на итерацию, до 7 листьев, min leaf=128, learning rate=0,08, L2=10, early stopping отключён; до 672 leaf values плюс splits. Никаких новых признаков, MLP, GA, class balancing, recency или tuning в этом опыте.","",
        "Сохранено **80 моделей**, warning=0, prior fallback не потребовался. В v1 первого origin успешно выполнились logit и boost fits, но boost не сохранился из-за отсутствующего описательного поля maximum_leaf_values. v2 добавляет ровно это поле (оно не передаётся в estimator constructor), импортирует сохранённый logit без повторного fit и выполняет оставшиеся 79. **Полный фактический расход — 81 estimator fit**, включая один fit с ошибкой сохранения. Все старые файлы оставлены неизменными. Связанные registries v25/v26 сохраняют историю ошибки и metadata-only amendment, не скрывая её новым успешным названием.","",
        "## Ошибки направления","","Знаменатель следующей таблицы — только вызовы вверх/вниз. Raw opposite означает любой противоположный знак, даже внутри полосы. Beyond-band opposite требует противоположного движения за половинной полосой. Neutral forecasts не засчитываются как правильные направленные прогнозы; при нулевых calls доля не определена. Доли от всех интервалов и ошибки при фактически ненулевых изменениях доступны в подробностях.","",
        "| Мин | Модель | Направленные вызовы / все, % | Raw opposite / вызовы, % | Opposite за half полосой / вызовы, % |","|---:|---|---:|---:|---:|"]
    for h in c["timeframes_minutes"]:
        for method in ["original_boost","half_boost","half_logit","random_uniform_3"]:
            r=next(r for r in medians if r["minutes"]==h and r["method"]==method and r["evaluation_band_factor"]==.5)
            lines.append("| %s | %s | %s | %s | %s |"%(h,method,fmt(r["directional_call_fraction"]),fmt(r["opposite_raw_sign_per_directional_call"]),fmt(r["opposite_per_directional_call"])))
    lines += ["","## Неопределённость и аудит","","1152 полные строки метрик: 12 методов × 8 активов × 4 горизонта × 3 определения truth. Для обученных моделей сохранены marginal CI accuracy/macro-F1; log-loss и multiclass Brier также оцениваются на одинаковых labels. Основное семейство — 512 парных сравнений: half logit/boost против matched original и random_uniform_3/majority_half/momentum_half, accuracy и macro-F1, все активы/горизонты. Односторонние max-t LCB одновременно по семейству, в двух общих banks, 1000 повторов, circular calendar blocks 3 дня.","",
        "Всего 13 календарных дней, примерно четыре трёхдневных блока зависимости: интервалы и max-t здесь нестабильны, особенно на 4h (78 примеров на актив). Новый bootstrap seed не создаёт независимых данных. Результаты нельзя экстраполировать на год или трактовать как доказательство торговой прибыли. Это prequential replay: последующие origins законно используют созревшие ранние октябрьские labels, но не будущие относительно origin.","",
        "Пройдены 9 pure checks, включая future-suffix invariance и strict endpoints. Repeat-predict всех 80 артефактов точный; независимое накопление confusion совпало. Второй audit независимо проверил 1152 metric cells, 512 paired comparisons, обе bootstrap banks и max-t LCB; прямые boolean-счётчики raw opposite проверены в 384 ячейках. Проверено %s старых artifact hashes и %s input pins; registry entries сохранены."%(verification["old_artifact_pins_unchanged"],verification["registered_data_pins_verified"]),"",
        "![Общие половинные метки](half_truth_accuracy_by_coin.png)","","Рисунок 1. Сравнение обучающих полос на одной половинной truth по каждой валюте; не смешивает разные labels в одном ranking.","",
        "![Перекрёстная оценка](cross_scoring.png)","","Рисунок 2. Оценивать эффект модели следует внутри строки. Диагональное сравнение original/original против half/half меняет задачу и не доказывает улучшения.","",
        "![Противоположный знак](opposite_raw_by_coin.png)","","Рисунок 3. Доля любого противоположного знака среди направленных вызовов; показывает, дала ли узкая полоса сильный directional signal.","",
        "![Календарная устойчивость](cumulative_accuracy_by_day.png)","","Рисунок 4. Накопленная accuracy на общих half labels по календарным дням, pooled counts активов. Не кривая капитала и не независимые дневные наблюдения; короткий период ограничивает выводы.","",
        "## Решение и следующий этап","","Не объявлять половинную полосу победителем по улучшению относительно модели, обученной на другой truth. На 1m/10m/30m есть наблюдаемый эффект adaptation к новым labels, но сильный перевес над простыми контролями и устойчивое direction skill не подтверждены всем зарегистрированным gate. На 4h majority остаётся важным отрицательным контролем. Сохранить оба контракта и не расширять архитектуру/поиск по этим 13 дням.","",
        "Следующий небольшой шаг — зафиксировать shortlist из простых моделей и оба определения truth, затем накопить существенно более длинный новый участок. До нового опыта определить требуемую длину по зависимости результатов и величине эффекта; текущие 13 дней не решают эту задачу. Если нужна минутная информативность, отдельный приоритет — многодневные causal L2/trades features; OHLCV-порог сам по себе информацию не создаёт. Торговой прибыли, fees/funding, исполнения и promotion этот опыт не проверяет.","",
        "## Артефакты","","OHLCV parquet: Retrodate/genetic_primus_direction_october_20261001_20261013_v1/october_full8_1m.parquet. Raw provenance: тот же каталог/raw и data_audit.json. Model inputs: direction_october_halfband_v1_20261014/data, неизменные с начала опыта. Модели/forecasts/метрики: текущий v2 каталог; execution_accounting.json — полный fit budget; registration.json — контракт; comparison_audit.json/audit_*m.json — проверки; DETAILS_RU.md — все методы/валюты/labels; календарные блоки — block_results.json.","",
        "Навык научно-технических отчётов использован для явных источников, формул, знаменателей, audit trail и ограничений. Отчёты Markdown/JSON и PNG; DOCX/XLSX не создавались. Старый runtime, API-ключи, ордера, paper/live не затрагивались."]
    with (OUT/"REPORT_RU.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    details=["# Все метрики октябрьского half-band опыта","","Для каждой таблицы truth фиксирована; 1 — исходная полоса, 0,5 — половинная, 0 — exact sign. N — все допустимые интервалы, не только направленные вызовы. CI marginal, без family correction; corrected comparisons — в results.json.",""]
    for h in c["timeframes_minutes"]:
        for coin in c["symbols"]:
            for factor in c["evaluation_band_factors"]:
                selected=[r for r in rows if r["minutes"]==h and r["symbol"]==coin and r["evaluation_band_factor"]==factor]
                details += ["## %s — %s мин — truth ×%s"%(coin,h,factor),"","N=%s; истинные counts вниз/нейтр./вверх: %s."%(selected[0]["n"],selected[0]["actual_down_neutral_up"]),"",
                    "| Метод | Accuracy, % | Macro-F1 | MCC | Recall вниз/нейтр./вверх, % | Направленные вызовы, % всех | Raw opposite, % вызовов | Opposite за truth-полосой, % вызовов | Opposite за полосой, % всех | LogLoss | Brier |","|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
                for r in selected:
                    recall=" / ".join("—" if v is None else fmt(v) for v in r["recall_down_neutral_up"])
                    details.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"%(r["method"],fmt(r["accuracy"]),fmt(r["macro_f1"],1),fmt(r["mcc"],1),recall,fmt(r["directional_call_fraction"]),fmt(r["opposite_raw_sign_per_directional_call"]),fmt(r["opposite_per_directional_call"]),fmt(r["opposite_per_all_intervals"]),fmt(r.get("log_loss"),1),fmt(r.get("multiclass_brier"),1)))
                details += ["","| Модель | Marginal 95% CI accuracy bank0 / bank1, % | Marginal 95% CI macro-F1 bank0 / bank1 |","|---|---|---|"]
                for r in selected:
                    if "ci95_banks" in r:
                        ci=r["ci95_banks"];details.append("| %s | %s | %s |"%(r["method"]," / ".join("[%s; %s]"%(fmt(a[0]),fmt(a[1])) for a in ci["accuracy"])," / ".join("[%s; %s]"%(fmt(a[0],1),fmt(a[1],1)) for a in ci["macro_f1"])))
                details.append("")
    with (OUT/"DETAILS_RU.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(details)+"\n")
    run.write(OUT/"report_statistics.json",dict(medians=medians,report_tool_sha256=run.digest(Path(__file__)),estimator_fits=0))
    run.write(OUT/"block_results.json",dict(rows=blocks,truth_band_factor=.5,aggregation="pooled counts across eight assets"))
    print(json.dumps(dict(report=run.rel(OUT/"REPORT_RU.md"),figures=4,metric_cells=1152)))

def manifest():
    c,reg=run.registered()
    required=["REPORT_RU.md","DETAILS_RU.md","half_truth_accuracy_by_coin.png","cross_scoring.png","opposite_raw_by_coin.png","cumulative_accuracy_by_day.png"]
    assert all((OUT/name).is_file() for name in required)
    artifacts={run.rel(p):run.digest(p) for p in OUT.rglob("*") if p.is_file() and p.name!="delivery_manifest.json"}
    artifacts.update(adapter.pins());artifacts[run.rel(Path(__file__))]=run.digest(Path(__file__));artifacts[run.rel(adapter.REGISTRY)]=run.digest(adapter.REGISTRY)
    run.write(OUT/"delivery_manifest.json",dict(artifact_pins=artifacts,figures_visually_inspected=True,prospective=False,**{k:False for k in run.reuse.fixed.SAFETY}))
    print(json.dumps(dict(final_artifact_pins=len(artifacts))))

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["audit","report","manifest"]);args=parser.parse_args();globals()[args.stage]()
