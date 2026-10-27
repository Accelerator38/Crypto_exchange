"""Independent arithmetic audit, graphs and report for frozen October forecasts."""
from __future__ import annotations
import analyze_genetic_primus_direction_signal_attribution_v1 as analysis
import argparse
import json
from pathlib import Path
import numpy as np

run=analysis.run
ROOT,OUT=analysis.ROOT,analysis.OUT

def audit():
    c,reg=analysis.registered();pc=run.config();paired=conditional=intervals=0
    for h in pc["timeframes_minutes"]:
        result=run.read(OUT/("results_%sm.json"%h));f=analysis.frame(h)
        with np.load(OUT/("daily_%sm.npz"%h),allow_pickle=False) as a:daily=a["counts"]
        banks=[]
        for seed in pc["bootstrap"]["seeds"]:
            with np.load(analysis.PARENT/("bootstrap_bank_%s.npz"%seed),allow_pickle=False) as a:banks.append(a["indices"])
        for row in result["rows"]:
            t=c["truth_band_factors"].index(row["truth_band_factor"]);asset=pc["symbols"].index(row["symbol"]);k=analysis.METHODS.index(row["method"])
            d=daily[t,asset,k];total=d.sum(axis=0)
            assert total[0]==row["n"] and total[1]==row["model_correct"] and total[2]==row["control_correct"]
            assert total[3]==row["wins"] and total[4]==row["losses"] and (total[3]-total[4])==sum(row["correct_delta_down_neutral_up"])
            assert np.array_equal(total[7:],row["correct_delta_down_neutral_up"])
            for b,indices in enumerate(banks):
                sampled=d[indices];values=(sampled[:,:,1]-sampled[:,:,2]).sum(axis=1)/sampled[:,:,0].sum(axis=1)
                assert np.allclose(np.quantile(values,[.025,.975]),row["accuracy_increment_ci95_banks"][b],atol=1e-15,rtol=0);intervals+=1
            factor=row["truth_band_factor"];truth=run.reuse.tri.labels(f["returns"][:,asset],pc["neutral_bands_pct"][str(h)]*factor)
            prior=f["raw_priors"] if factor==0 else f["priors"][t]
            control=np.array([-1,0,1])[prior[:,asset].argmax(axis=-1)]
            if k<4:pred=np.array([-1,0,1])[f["prob"][k,:,asset].argmax(axis=-1)]
            elif k==4:pred=f["random"][:,asset]
            else:pred=run.reuse.tri.labels(f["past"][:,asset],pc["neutral_bands_pct"][str(h)]*factor)
            cube=np.bincount((truth+1)*9+(pred+1)*3+control+1,minlength=27).reshape(3,3,3)
            model_by_state=np.array([cube[i,i,:].sum() for i in range(3)])
            control_by_state=np.array([cube[i,:,i].sum() for i in range(3)])
            assert np.array_equal(model_by_state-control_by_state,row["correct_delta_down_neutral_up"])
            assert int(model_by_state.sum())==row["model_correct"] and int(control_by_state.sum())==row["control_correct"];paired+=1
        for row in result["conditional"]:
            t=c["truth_band_factors"].index(row["truth_band_factor"]);asset=pc["symbols"].index(row["symbol"]);k=analysis.METHODS.index(row["method"])
            truth=run.reuse.tri.labels(f["returns"][:,asset],pc["neutral_bands_pct"][str(h)]*row["truth_band_factor"]);move=truth!=0
            p=f["prob"][k,:,asset];prob=p[move,2]/(p[move,0]+p[move,2]);y=(truth[move]>0).astype(int)
            assert row["n"]==len(y) and abs(row["accuracy"]-((prob>.5)==y).mean())<1e-15
            assert abs(row["brier"]-np.dot(prob-y,prob-y)/len(y))<1e-15
            assert sum(b["n"] for b in row["bins"])==len(y)
            for s in row["confidence_strata"]:assert s["n"]==int((np.maximum(prob,1-prob)>=s["minimum_confidence"]).sum())
            conditional+=1
        for row in result["pooled"]:
            t=c["truth_band_factors"].index(row["truth_band_factor"]);k=analysis.METHODS.index(row["method"]);d=daily[t,:,k]
            assert row["n"]==int(d[:,:,0].sum())
            for asset,v in enumerate(row["leave_one_asset_out_increment"]):
                kept=np.delete(d,asset,axis=0);assert abs(v-(kept[:,:,1]-kept[:,:,2]).sum()/kept[:,:,0].sum())<1e-15
            for block,v in enumerate(row["leave_one_block_out_increment"]):
                days=[i for i in range(13) if i not in range(block*3,min(block*3+3,13))];kept=d[:,days]
                assert abs(v-(kept[:,:,1]-kept[:,:,2]).sum()/kept[:,:,0].sum())<1e-15
    for p,sha in {**reg["parent_artifact_pins"],**reg["parent_data_pins"]}.items():assert run.digest(ROOT/p)==sha,p
    previous=run.read(ROOT/c["previous_registry"]);current=run.read(analysis.REGISTRY);assert current["objectives"][:-1]==previous["objectives"]
    run.write(OUT/"audit.json",dict(paired_cells=paired,conditional_cells=conditional,marginal_intervals=intervals,
        independent_prediction_cube_attribution=True,independent_direction_brier=True,leave_one_out_checked=True,
        previous_registry_unchanged=True,parent_pins_unchanged=True,estimator_fits=0,report_tool_sha256=run.digest(Path(__file__))))
    print(json.dumps(dict(paired_cells=paired,conditional_cells=conditional,marginal_intervals=intervals,fits=0)))

def fmt(v,multiplier=100):return "—" if v is None else "%.2f"%(v*multiplier)

def report():
    import matplotlib.pyplot as plt
    c,reg=analysis.registered();pc=run.config();checked=run.read(OUT/"audit.json")
    results=[run.read(OUT/("results_%sm.json"%h)) for h in pc["timeframes_minutes"]]
    rows=sum([r["rows"] for r in results],[]);pooled=sum([r["pooled"] for r in results],[]);conditional=sum([r["conditional"] for r in results],[])
    directions=[]
    for h in pc["timeframes_minutes"]:
        for factor in c["truth_band_factors"]:
            for method in c["learned_methods"]:
                rr=[r for r in conditional if r["minutes"]==h and r["method"]==method and r["truth_band_factor"]==factor]
                medians={key:float(np.median([r[key] for r in rr])) for key in ("accuracy","prior_accuracy","brier_improvement","auc","move_brier_improvement")}
                total=sum(r["n"] for r in rr);high=[dict(confidence=level,n=sum(r["confidence_strata"][i]["n"] for r in rr),coverage=sum(r["confidence_strata"][i]["n"] for r in rr)/total) for i,level in enumerate((.8,.9))]
                directions.append(dict(minutes=h,method=method,truth_band_factor=factor,**medians,n_actual_moves_all_assets=total,high_confidence=high))
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout="constrained")
    for ax,h in zip(axes.flat,pc["timeframes_minutes"]):
        selected=[r for r in pooled if r["minutes"]==h and r["truth_band_factor"]==.5 and r["method"] in c["learned_methods"]]
        matrix=np.array([r["correct_delta_down_neutral_up"] for r in selected],float)/np.array([r["n"] for r in selected])[:,None]*100
        pos=np.zeros(4);neg=np.zeros(4)
        for j,(label,color) in enumerate(zip(["Истинное вниз","Истинное neutral","Истинное вверх"],["#ca5656","#999999","#3d9c71"])):
            values=matrix[:,j];ax.bar(range(4),values,bottom=np.where(values>=0,pos,neg),color=color,label=label)
            pos+=np.maximum(values,0);neg+=np.minimum(values,0)
        total=matrix.sum(axis=1);ax.plot(range(4),total,"ko",label="Суммарная добавка")
        ax.axhline(0,color="black",lw=.8);ax.set_xticks(range(4),[r["method"] for r in selected],rotation=15);ax.set_title("%s мин — добавка к majority_half"%h);ax.set_ylabel("Вклад в общую accuracy, п.п.")
        for i,value in enumerate(total):ax.annotate("%+.2f"%value,(i,value),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
    axes[0,0].legend(fontsize=8,loc="lower left");fig.savefig(OUT/"accuracy_attribution.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(16,8),layout="constrained")
    for ax,h in zip(axes.flat,pc["timeframes_minutes"]):
        matrix=np.array([[100*(next(r["accuracy"] for r in conditional if r["minutes"]==h and r["method"]==method and r["symbol"]==coin and r["truth_band_factor"]==0)-next(r["prior_accuracy"] for r in conditional if r["minutes"]==h and r["method"]==method and r["symbol"]==coin and r["truth_band_factor"]==0)) for coin in pc["symbols"]] for method in c["learned_methods"]])
        im=ax.imshow(matrix,vmin=-12,vmax=12,cmap="RdBu",aspect="auto");ax.set_xticks(range(8),[s.split('/')[0] for s in pc["symbols"]]);ax.set_yticks(range(4),c["learned_methods"]);ax.set_title("%s мин — условное направление: добавка к train prior"%h)
        for i in range(4):
            for j in range(8):ax.text(j,i,"%+.1f"%matrix[i,j],ha="center",va="center",color="white" if abs(matrix[i,j])>7 else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Добавка accuracy на фактически ненулевых изменениях, п.п.",shrink=.7);fig.savefig(OUT/"conditional_direction_increment.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout="constrained")
    for ax,h in zip(axes.flat,pc["timeframes_minutes"]):
        for method in c["learned_methods"]:
            rr=[r for r in conditional if r["minutes"]==h and r["method"]==method and r["truth_band_factor"]==.5];x=[];y=[]
            for b in range(10):
                n=sum(r["bins"][b]["n"] for r in rr)
                if n:
                    x.append(sum(r["bins"][b]["n"]*(r["bins"][b]["predicted_mean"] or 0) for r in rr)/n)
                    y.append(sum(r["bins"][b]["n"]*(r["bins"][b]["actual_positive_fraction"] or 0) for r in rr)/n)
            ax.plot(x,y,"o-",label=method)
        ax.plot([0,1],[0,1],"k--",lw=1,label="Идеальная калибровка");ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title("%s мин — p(up | фактический move), half truth"%h)
        ax.set_xlabel("Средняя предсказанная вероятность вверх");ax.set_ylabel("Фактическая доля вверх");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(OUT/"direction_calibration.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    for ax,h in zip(axes.flat,pc["timeframes_minutes"]):
        for method in c["learned_methods"]:
            r=next(r for r in pooled if r["minutes"]==h and r["method"]==method and r["truth_band_factor"]==.5)
            ax.plot(range(5),[100*b["accuracy_increment"] for b in r["blocks"]],"o-",label=method)
        ax.axhline(0,color="gray",lw=1);ax.set_xticks(range(5),["1–3","4–6","7–9","10–12","13"]);ax.set_title("%s мин — добавка к majority_half по блокам"%h);ax.set_xlabel("Дни октября UTC");ax.set_ylabel("Добавка accuracy, п.п.");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(OUT/"calendar_increment.png",dpi=150);plt.close(fig)
    lines=["# Почему трёхрежимная точность не стала сильным прогнозом направления","","Frozen данные: 1–13 октября 2026 UTC. Диагностика 14 октября 2026, без новых fits, сети и изменений прогнозов.","",
        "## Главный вывод","","У boosting на 1m и 10m положительная добавка общей трёхклассовой accuracy обеспечена главным образом распознаванием neutral, а не улучшением направленных классов. На 1m half_boost получает +3,30 п.п. к majority_half: +7,24 п.п. neutral и −3,94 п.п. суммарно down/up. На 10m +3,25 п.п. ≈ +4,17 neutral −0,91 down/up (компоненты округлены). Поэтому более высокая трёхрежимная accuracy не означает, что вектор направления угадывается на 80–90%.","",
        "Half_logit на 10m — единственный из двух half-вариантов с небольшим положительным медианным Brier improvement чистого направления в этом раскрытом отрезке. Условная direction accuracy 51,86% против 50,10% у train-only prior, AUC около 0,528. Это описательный кандидат, не подтверждённый новый winner: прежний одновременный gate остался 0/64; 13 календарных дней недостаточно для устойчивых выводов.","",
        "## Точная атрибуция общей accuracy","","Для всех интервалов ∆accuracy = [N(correct_model)−N(correct_majority)]/N. Разложение по истинным down/neutral/up точно суммируется в общую разницу. Эквивалентно ∆accuracy = (wins−losses)/N, где wins: model верна, majority ошиблась; losses: наоборот. Это арифметическая атрибуция, не причинная атрибуция торговой прибыли. Ни PnL, ни сделки здесь не вычислялись.","",
        "| Мин | Модель | ∆ к majority_half, п.п. | Вклад down, п.п. | Вклад neutral, п.п. | Вклад up, п.п. | Положительных блоков из 5 |","|---:|---|---:|---:|---:|---:|---:|"]
    for h in pc["timeframes_minutes"]:
        for method in ("half_logit","half_boost"):
            r=next(r for r in pooled if r["minutes"]==h and r["method"]==method and r["truth_band_factor"]==.5)
            components=[100*v/r["n"] for v in r["correct_delta_down_neutral_up"]]
            lines.append("| %s | %s | %+.2f | %+.2f | %+.2f | %+.2f | %s |"%(h,method,r["accuracy_increment"]*100,*components,r["positive_blocks"]))
    lines += ["","Здесь pooled counts восьми валют, а не медиана их accuracy. Поэтому цифры отличаются от медиан предыдущего отчёта. Календарные блоки: четыре по три дня и последний один день; они не равны по весу, а 5/5 положительных блоков сами по себе не являются новым significance gate.","",
        "## Качество именно направления","","Из frozen вероятностей вычисляется p_dir(up)=p(up)/(p(up)+p(down)). Для диагностики направление выбирается на ВСЕХ фактически движущихся интервалах, включая случаи, когда исходный argmax трёх состояний был neutral. Чистый знак: фактическое движение r≠0; полоса 0 сохраняет exact-zero отдельным классом. При p_dir=0,5 выбирается down. Это условная задача с изменённой оценкой решений, не основная 100%-coverage трёхклассовая accuracy и не отказная торговая политика.","",
        "Контроль — train-only Laplace prior направления для той же truth и тех же mature train indices. Raw sign prior заново вычислен только на сохранённых исходных train rows, без fit. Brier improvement = Brier_prior−Brier_model: положительное значение лучше. AUC измеряет ранжирование; высокая accuracy majority при дисбалансе не доказывает directional signal.","",
        "| Мин | Half-модель | Условная direction accuracy, % | Train prior, % | AUC | Brier improvement | Calls с уверенностью ≥80%* |","|---:|---|---:|---:|---:|---:|---:|"]
    for h in pc["timeframes_minutes"]:
        for method in ("half_logit","half_boost"):
            r=next(r for r in directions if r["minutes"]==h and r["method"]==method and r["truth_band_factor"]==0)
            lines.append("| %s | %s | %s | %s | %.4f | %+.6f | %s / %s |"%(h,method,fmt(r["accuracy"]),fmt(r["prior_accuracy"]),r["auc"],r["brier_improvement"],r["high_confidence"][0]["n"],r["n_actual_moves_all_assets"]))
    lines += ["","* Это confidence модели, не фактическая точность. Фиксированные уровни 80/90% используются только описательно; решения и thresholds не менялись. Высокоуверенных случаев единицы или ноль, поэтому нельзя заявить 80–90% достигнутой accuracy на содержательном покрытии. Таблица accuracy/AUC/Brier — медианы по восьми валютам; confidence counts — суммы.","",
        "## Устойчивость и ограничения","","Для каждой ячейки сохранены wins/losses, разложение down/neutral/up и marginal 95% CI добавки. Два общих calendar bootstrap banks исходного опыта, blocks 3 дня; нового hypothesis gate нет. Показаны исключение любой одной валюты и любого одного календарного блока. Эти проверки описательные и не создают новые независимые данные.","",
        "| Мин | Модель | ∆ pooled, п.п. | Range при исключении валюты, п.п. | Range при исключении блока, п.п. | 95% CI bank0, п.п. |","|---:|---|---:|---|---|---|"]
    for h in pc["timeframes_minutes"]:
        for method in ("half_logit","half_boost"):
            r=next(r for r in pooled if r["minutes"]==h and r["method"]==method and r["truth_band_factor"]==.5)
            a=r["leave_one_asset_out_increment"];b=r["leave_one_block_out_increment"];ci=r["accuracy_increment_ci95_banks"][0]
            lines.append("| %s | %s | %+.2f | [%+.2f; %+.2f] | [%+.2f; %+.2f] | [%+.2f; %+.2f] |"%(h,method,100*r["accuracy_increment"],100*min(a),100*max(a),100*min(b),100*max(b),100*ci[0],100*ci[1]))
    lines += ["","## Длительность нового теста","","Подготовлены 256 сценариев для эффекта +1/+2/+3/+5 п.п. к majority_half: 2 модели × 4 горизонта × 8 валют. Только четыре полных трёхдневных блока используются для оценки σ; последний неполный день остаётся в основном отчёте, но не в variance estimate. При предполагаемых независимых стационарных блоках B≈[(z(1−0,05/64)+z(0,8))σ/δ]², дни=3×ceil(B). Дополнительно используется верхняя 90% chi-square оценка σ. Ни независимость, ни стационарность здесь не доказаны; при четырёх блоках это только грубый sensitivity calculation.","",
        "Файл duration_scenarios.json содержит точечные и осторожные оценки. Они часто существенно превышают 90 дней, особенно на 4h. Это НЕ рекомендация ждать годы: огромные значения показывают, что малая выборка, слабый эффект и широкое семейство не позволяют обосновать точную длительность. Нельзя честно обещать, что 90 дней автоматически дадут достаточную power или 80% точности.","",
        "future_protocol_DRAFT.json сохраняет маленький набор: прежние uniform logit/boost, обе обучающие полосы, все восемь валют и четыре горизонта, тот же causal retrain/scaler/purge. Предложен инженерный минимум 90 новых календарных дней с началом не раньше 15 октября, но окончательный интервал НЕ запечатан, не запланирован и не запущен. Никакой автоматической загрузки, обучения, анализа или promotion. Перед sealed опытом нужно утвердить primary estimand, допустимое семейство сравнений, эффект и длительность; не выбирать их по новому тесту.","",
        "## План дальнейшей разработки","","1. Прекратить перебор нейтральных полос как способ получить 80–90% направленной accuracy: полоса меняет задачу, но не создаёт информацию.\n2. Разделить роли. Boosting 1m/10m можно рассматривать как detector нейтральности; half_logit 10m — как слабый directional candidate для новой проверки. Это гипотезы по раскрытым данным, не разрешение на замену моделей.\n3. До нового обучения выбрать одну конкретную ошибку: калибровка/регуляризация вероятностей направления либо новые причинные данные. Подбирать C/пороги на октябрьском тесте нельзя; если менять, нужен отдельный development контракт с ограниченным бюджетом и последующий новый тест.\n4. Для minute directional skill нужны проверки новых информационных признаков — многодневных синхронных trades/L2 с доказанной доступностью. OHLCV volume нельзя выдавать за истинный buy/sell imbalance. Сеть на этом этапе не открывалась; новый сбор потребует отдельного разрешения.\n5. Сначала исследовательское подтверждение, затем отдельный continuous-ledger финансовый replay с fees/funding/реалистичным исполнением. Даже подтверждённое direction skill не является автоматическим допуском к деньгам.","",
        "![Атрибуция accuracy](accuracy_attribution.png)","","Рисунок 1. Точная добавка общей accuracy по истинному классу; чёрная точка — сумма. Neutral вклад и directional signal показаны отдельно.","",
        "![Условное направление](conditional_direction_increment.png)","","Рисунок 2. Добавка условной точности чистого знака относительно причинного train-only direction prior. Фактически нулевые изменения исключены; neutral predictions модели не исключаются.","",
        "![Калибровка направления](direction_calibration.png)","","Рисунок 3. Fixed-bin калибровка вероятности up на фактических moves половинной truth. Sparse крайние bins могут быть шумными; их counts сохранены в JSON, thresholds не подбирались.","",
        "![Календарная добавка](calendar_increment.png)","","Рисунок 4. Разброс добавки к majority по заранее заданным блокам октября. Последний блок один день, не три.","",
        "## Проверки и артефакты","","Проверено %s парных ячеек, %s условных ячеек, %s marginal CI, независимая prediction-cube атрибуция, Brier и leave-one-out. Parent code/data/model hashes неизменны, registry v27 append-only. Новых estimator fits: 0; source forecasts и цены не менялись. Навык научно-технических отчётов применён для формул, знаменателей, доказательных графиков и явного разделения условной/полной accuracy. Документы Markdown/JSON, PNG; DOCX/XLSX не создавались."%(checked["paired_cells"],checked["conditional_cells"],checked["marginal_intervals"]),"",
        "registration.json — контракт диагностики; results_*m.json — ячейки, pooled и conditional; daily_*m.npz — дневные counts; duration_scenarios.json — сценарии длины; future_protocol_DRAFT.json — незапущенный проект проверки; DETAILS_RU.md — все модели и валюты; audit.json — независимые проверки; delivery_manifest.json — hashes. Все credentials/orders/paper/live/runtime/network/promotion разрешения false."]
    with (OUT/"REPORT_RU.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    details=["# Атрибуция по всем валютам и моделям","","∆ accuracy — против train-only majority той же evaluated truth. Числа в п.п. и процентах подписаны. Truth factor 0 — exact sign; 0,5 — половинная полоса; 1 — исходная. CI marginal, не поправлены для нового significance gate.",""]
    for h in pc["timeframes_minutes"]:
        for coin in pc["symbols"]:
            for factor in c["truth_band_factors"]:
                details += ["## %s — %s мин — truth ×%s"%(coin,h,factor),"","| Метод | N | Accuracy, % | Majority, % | ∆, п.п. | Вклады down / neutral / up, п.п. | Wins | Losses | CI bank0 / bank1, п.п. |","|---|---:|---:|---:|---:|---|---:|---:|---|"]
                for r in [r for r in rows if r["minutes"]==h and r["symbol"]==coin and r["truth_band_factor"]==factor]:
                    parts=" / ".join("%+.2f"%(100*v/r["n"]) for v in r["correct_delta_down_neutral_up"])
                    ci=" / ".join("[%+.2f; %+.2f]"%(100*a[0],100*a[1]) for a in r["accuracy_increment_ci95_banks"])
                    details.append("| %s | %s | %s | %s | %+.2f | %s | %s | %s | %s |"%(r["method"],r["n"],fmt(r["model_accuracy"]),fmt(r["control_accuracy"]),100*r["accuracy_increment"],parts,r["wins"],r["losses"],ci))
                details += ["","### Условное направление на всех фактических moves","","| Модель | N actual moves | Accuracy, % | Train prior, % | AUC | Brier | Prior Brier | Brier improvement | Move Brier improvement | N confidence ≥80% / ≥90% |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
                for r in [r for r in conditional if r["minutes"]==h and r["symbol"]==coin and r["truth_band_factor"]==factor]:
                    details.append("| %s | %s | %s | %s | %.4f | %.6f | %.6f | %+.6f | %+.6f | %s / %s |"%(r["method"],r["n"],fmt(r["accuracy"]),fmt(r["prior_accuracy"]),r["auc"],r["brier"],r["prior_brier"],r["brier_improvement"],r["move_brier_improvement"],r["confidence_strata"][0]["n"],r["confidence_strata"][1]["n"]))
                details += ["","Counts fixed probability bins, confidence coverage/accuracy и log-loss доступны в results_*m.json. Условное направление не заменяет исходный трёхклассовый forecast.",""]
    with (OUT/"DETAILS_RU.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(details)+"\n")
    run.write(OUT/"report_statistics.json",dict(conditional_medians=directions,pooled=pooled,report_tool_sha256=run.digest(Path(__file__)),estimator_fits=0))
    print(json.dumps(dict(report=run.rel(OUT/"REPORT_RU.md"),paired_cells=len(rows),conditional_cells=len(conditional),figures=4)))

def manifest():
    c,reg=analysis.registered()
    required=["REPORT_RU.md","DETAILS_RU.md","accuracy_attribution.png","conditional_direction_increment.png","direction_calibration.png","calendar_increment.png"]
    assert all((OUT/name).is_file() for name in required)
    artifacts={run.rel(p):run.digest(p) for p in OUT.rglob("*") if p.is_file() and p.name!="delivery_manifest.json"}
    artifacts.update(analysis.pins());artifacts[run.rel(Path(__file__))]=run.digest(Path(__file__));artifacts[run.rel(analysis.REGISTRY)]=run.digest(analysis.REGISTRY)
    run.write(OUT/"delivery_manifest.json",dict(artifact_pins=artifacts,figures_visually_inspected=True,future_protocol_unsealed=True,**{k:False for k in run.reuse.fixed.SAFETY}))
    print(json.dumps(dict(artifact_pins=len(artifacts),fits=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["audit","report","manifest"]);args=parser.parse_args();globals()[args.stage]()
