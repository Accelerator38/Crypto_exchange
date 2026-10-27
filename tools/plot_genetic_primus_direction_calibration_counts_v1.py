"""Display-only calibration QA: mark sparse bins. Does not change any metrics."""
from pathlib import Path
import json
import numpy as np
import analyze_genetic_primus_direction_signal_attribution_v1 as analysis

if __name__=="__main__":
    import matplotlib.pyplot as plt
    c,reg=analysis.registered();pc=analysis.run.config();out=analysis.OUT
    fig,axes=plt.subplots(2,2,figsize=(15,11),layout="constrained");colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728"];summary=[]
    for ax,h in zip(axes.flat,pc["timeframes_minutes"]):
        rows=analysis.run.read(out/("results_%sm.json"%h))["conditional"]
        for k,method in enumerate(c["learned_methods"]):
            selected=[r for r in rows if r["method"]==method and r["truth_band_factor"]==.5];points=[]
            for b in range(10):
                n=sum(r["bins"][b]["n"] for r in selected)
                if n:
                    x=sum(r["bins"][b]["n"]*(r["bins"][b]["predicted_mean"] or 0) for r in selected)/n
                    y=sum(r["bins"][b]["n"]*(r["bins"][b]["actual_positive_fraction"] or 0) for r in selected)/n
                    points.append((x,y,n));summary.append(dict(minutes=h,method=method,bin=b,n=n,p_mean=x,actual_up=y))
            dense=[p for p in points if p[2]>=30];sparse=[p for p in points if p[2]<30]
            if dense:
                ax.plot([p[0] for p in dense],[p[1] for p in dense],"o-",color=colors[k],label=method)
            if sparse:
                ax.scatter([p[0] for p in sparse],[p[1] for p in sparse],marker="x",s=35,color=colors[k])
                for x,y,n in sparse:ax.annotate("n=%s"%n,(x,y),xytext=(3,4+9*k),textcoords="offset points",fontsize=7,color=colors[k])
        ax.plot([0,1],[0,1],"k--",lw=1);ax.set_xlim(-.02,1.04);ax.set_ylim(-.03,1.10)
        ax.set_title("%s мин — условное направление, half truth"%h);ax.set_xlabel("Средняя p(up | move)");ax.set_ylabel("Фактическая доля up");ax.grid(alpha=.2)
        ax.text(.02,1.04,"× и n: менее 30 наблюдений; не соединены",fontsize=8)
    axes[0,0].legend(fontsize=8,loc="lower right")
    fig.savefig(out/"direction_calibration.png",dpi=150);plt.close(fig)
    analysis.run.write(out/"calibration_plot_counts.json",dict(pooled_bins=summary,plot_only_minimum_connected_bin_count=30,
        statistical_rows_unchanged=True,estimator_fits=0,plot_tool_sha256=analysis.run.digest(Path(__file__))))
    print(json.dumps(dict(pooled_calibration_bins=len(summary),sparse_bins_explicit=True,fits=0)))
