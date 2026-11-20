"""Jinja2 + Plotly (CDN) single-file interactive HTML dashboard.

Sections (per spec):
  a) Model arena summary table + fold-metric boxplots
  b) ROC / Precision-Recall / reliability curves (classification)
     or predicted-vs-actual + residual plots (regression)
  c) SHAP summary bar + distribution plots for both model families
  d) Collinearity heatmap + data-quality warnings
  e) Worst-prediction error table with local top-3 SHAP drivers

STRICT RULE: no MCP imports. Output is standalone except the Plotly CDN tag.
"""
from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from ..engine.config import MODEL_LIGHTGBM, MODEL_PYTORCH_MLP, MODEL_FAMILIES, TaskType

COLORS = {MODEL_LIGHTGBM: "#2563eb", MODEL_PYTORCH_MLP: "#e11d48"}
LABELS = {MODEL_LIGHTGBM: "LightGBM baseline", MODEL_PYTORCH_MLP: "PyTorch residual MLP"}
PLOT_CONFIG = {"responsive": True, "displayModeBar": False, "scrollZoom": False}

_METRIC_SHOW = {
    TaskType.BINARY_CLASSIFICATION: ("roc_auc", "brier"),
    TaskType.MULTICLASS_CLASSIFICATION: ("macro_f1", "log_loss"),
    TaskType.REGRESSION: ("rmse", "mae"),
}


def _esc(x) -> str:
    return html_mod.escape(str(x))


def _fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(v)


def _layout(title: str, height: int = 330, xaxis: str = "", yaxis: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"size": 14}},
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 60, "r": 24, "t": 46, "b": 44},
        "font": {"size": 11},
        "xaxis": {"title": xaxis},
        "yaxis": {"title": yaxis},
        "legend": {"orientation": "h", "x": 0, "y": 1.12, "font": {"size": 10}},
        "showlegend": True,
    }


def _chart(specs: dict, div_id: str, fig: go.Figure) -> str:
    specs[div_id] = json.loads(fig.to_json())
    return f'<div class="plot" id="{div_id}" data-spec="{div_id}"></div>'


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _box_figure(specs, ctx, div_id, metric):
    fig = go.Figure(layout=_layout(f"Fold-level {metric} (mean±std across folds)"))
    for family in MODEL_FAMILIES:
        vals = ctx["per_model"][family]["per_fold"].get(metric, [])
        fig.add_trace(go.Box(
            y=[float(v) for v in vals], name=LABELS[family], marker_color=COLORS[family],
            boxpoints="all", jitter=0.3, line={"color": COLORS[family]},
        ))
    higher = metric not in {"rmse", "mae", "log_loss", "brier", "ece"}
    fig.update_yaxes(autorange="reversed" if not higher else None)
    return _chart(specs, div_id, fig)


def _arena_table(ctx) -> str:
    task = ctx["task"]
    metrics = list(ctx["per_model"][MODEL_LIGHTGBM]["per_fold"].keys())
    primary = ctx["manifest"]["winning_metric"]
    rows = []
    for m in metrics:
        cells = [f"<code>{_esc(m)}{' ★' if m == primary else ''}</code>"]
        best_fam, best_val = None, None
        for family in MODEL_FAMILIES:
            mean = ctx["per_model"][family]["mean"].get(m, float("nan"))
            std = ctx["per_model"][family]["std"].get(m, float("nan"))
            oof = ctx["per_model"][family]["oof"].get(m, float("nan"))
            cells.append(f"<td>{_fmt(mean)} ± {_fmt(std)}</td><td class='mono'>{_fmt(oof)}</td>")
            if np.isfinite(oof):
                hi = m not in {"rmse", "mae", "log_loss", "brier", "accuracy_neg"}
                if m == "accuracy":
                    hi = True
                if best_val is None or (oof > best_val if hi else oof < best_val):
                    best_fam, best_val = family, oof
        cells.append(f"<td><span class='pill' style='background:{COLORS.get(best_fam, '#666')}'>{_esc(LABELS.get(best_fam, '—'))}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = ("<tr><th>Metric</th><th>LightGBM fold mean±std</th><th>LGBM OOF</th>"
            "<th>MLP fold mean±std</th><th>MLP OOF</th><th>Better</th></tr>")
    cal_rows = []
    for family in MODEL_FAMILIES:
        cal = {k: v for k, v in ctx["per_model"][family]["calibration"].items() if not isinstance(v, list)}
        cal_txt = ", ".join(f"{k}={_fmt(v)}" for k, v in cal.items())
        t = ctx["per_model"][family]["fit_seconds"]
        p = ctx["per_model"][family].get("n_parameters")
        extra = f" · fit {t:.1f}s" + (f" · {p:,} params" if p else "")
        cal_rows.append(f"<tr><td><span class='pill' style='background:{COLORS[family]}'>{_esc(LABELS[family])}</span></td>"
                        f"<td colspan='5' class='mono'>{_esc(cal_txt)}{extra}</td></tr>")
    return (f"<table class='tbl'>{head}{''.join(rows)}</table>"
            f"<table class='tbl sub'><tr><th>Calibration / cost</th><th></th></tr>{''.join(cal_rows)}</table>")


def _classification_curve_figures(specs, ctx, prefix=""):
    task: TaskType = ctx["task"]
    oof = ctx["oof_df"]
    y = oof["y_true_code"].to_numpy()
    out = []
    if task is TaskType.BINARY_CLASSIFICATION:
        fig = go.Figure(layout=_layout("ROC — out-of-fold", xaxis="False positive rate", yaxis="True positive rate"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line={"dash": "dot", "color": "#999"}))
        pr = go.Figure(layout=_layout("Precision-Recall — out-of-fold", xaxis="Recall", yaxis="Precision"))
        for family in MODEL_FAMILIES:
            p = oof[f"{family}_proba_1"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            fig.add_trace(go.Scatter(x=fpr.tolist(), y=tpr.tolist(), name=f"ROC {LABELS[family]}", line={"color": COLORS[family]}))
            prec, rec, _ = precision_recall_curve(y, p)
            pr.add_trace(go.Scatter(x=rec.tolist(), y=prec.tolist(), name=f"PR {LABELS[family]}", line={"color": COLORS[family]}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", visible=False))
        out.append(_chart(specs, prefix + "roc", fig))
        out.append(_chart(specs, prefix + "pr", pr))
    else:
        n_cls = len(ctx["classes"])
        y_bin = label_binarize(y, classes=list(range(n_cls)))
        fig = go.Figure(layout=_layout("Micro-average ROC — out-of-fold", xaxis="FPR", yaxis="TPR"))
        pr = go.Figure(layout=_layout("Macro-averaged PR (threshold sweep) — out-of-fold", xaxis="Threshold", yaxis="Macro precision"))
        for family in MODEL_FAMILIES:
            proba = oof[[f"{family}_proba_{c}" for c in range(n_cls)]].to_numpy()
            fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
            fig.add_trace(go.Scatter(x=fpr.tolist(), y=tpr.tolist(), name=f"ROC {LABELS[family]}", line={"color": COLORS[family]}))
            ths = np.unique(np.round(np.quantile(proba.ravel(), np.linspace(0.02, 0.98, 40)), 4))
            macro_p = []
            for th in ths:
                preds = proba >= th
                ps = []
                for c in range(n_cls):
                    tp = float(((preds[:, c]) & (y_bin[:, c] == 1)).sum())
                    fp = float(((preds[:, c]) & (y_bin[:, c] == 0)).sum())
                    ps.append(tp / (tp + fp) if tp + fp else 0.0)
                macro_p.append(float(np.mean(ps)))
            pr.add_trace(go.Scatter(x=ths.tolist(), y=macro_p, name=f"PR {LABELS[family]}", line={"color": COLORS[family]}))
        out.append(_chart(specs, prefix + "roc", fig))
        out.append(_chart(specs, prefix + "pr", pr))

    # reliability diagram (binary: proba of positive; multiclass: top-class confidence)
    rel = go.Figure(layout=_layout("Reliability / calibration — out-of-fold", xaxis="Predicted probability", yaxis="Observed frequency"))
    rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line={"dash": "dot", "color": "#999"}))
    for family in MODEL_FAMILIES:
        if task is TaskType.BINARY_CLASSIFICATION:
            curve = ctx["per_model"][family]["calibration"].get("reliability_curve", [])
            if curve:
                rel.add_trace(go.Scatter(
                    x=[b["mean_predicted"] for b in curve], y=[b["observed_rate"] for b in curve],
                    mode="lines+markers", name=LABELS[family], line={"color": COLORS[family]},
                    hovertemplate="pred=%{x:.3f} obs=%{y:.3f}",
                ))
        else:
            proba = oof[[f"{family}_proba_{c}" for c in range(len(ctx["classes"]))]].to_numpy()
            conf = proba.max(axis=1)
            correct = (proba.argmax(axis=1) == y).astype(float)
            edges = np.linspace(0, 1, 11)
            xs, ys = [], []
            for b in range(10):
                mask = (conf >= edges[b]) & (conf < edges[b + 1] if b < 9 else conf <= 1.0)
                if mask.sum():
                    xs.append(float(conf[mask].mean())); ys.append(float(correct[mask].mean()))
            rel.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=LABELS[family], line={"color": COLORS[family]}))
    out.append(_chart(specs, prefix + "rel", rel))
    return out


def _regression_figures(specs, ctx, prefix=""):
    oof = ctx["oof_df"]
    y = oof["y_true_code"].to_numpy()
    scatter = go.Figure(layout=_layout("Predicted vs actual — out-of-fold", xaxis="Actual", yaxis="Predicted"))
    resid = go.Figure(layout=_layout("Residual distribution — out-of-fold", xaxis="Residual (true − pred)", yaxis="Count"))
    lo, hi = float(min(y.min(), 0)), float(max(y.max(), 0))
    scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="identity", line={"dash": "dot", "color": "#999"}))
    for family in MODEL_FAMILIES:
        pred = oof[f"{family}_pred"].to_numpy()
        r = y - pred
        scatter.add_trace(go.Scatter(x=y.tolist(), y=pred.tolist(), mode="markers", name=LABELS[family],
                                    marker={"color": COLORS[family], "size": 4, "opacity": 0.55}))
        resid.add_trace(go.Histogram(x=r.tolist(), name=LABELS[family], marker_color=COLORS[family], opacity=0.7, nbinsx=40))
    return [_chart(specs, prefix + "scatter", scatter), _chart(specs, prefix + "resid", resid)]


def _shap_figures(specs, ctx, prefix=""):
    expl = ctx["expl"]
    frames = expl["shap_frames"]
    base_features = expl["base_features"]
    enc_map = expl["encoded_to_base"]

    def base_matrix(frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in frame.columns if not c.startswith("_")]
        agg: dict[str, np.ndarray] = {}
        for c in cols:
            b = enc_map.get(c, c)
            agg.setdefault(b, np.zeros(len(frame)))
            agg[b] += frame[c].to_numpy(dtype=float)
        return pd.DataFrame(agg, index=frame.index)

    top_names = [t["feature"] for t in expl["top_features"][:12]]
    out = []
    for family in MODEL_FAMILIES:
        arr = np.maximum(expl["gbdt_mean_abs_shap"] if family == MODEL_LIGHTGBM else expl["nn_mean_abs_shap"], 0)
        order = np.argsort(-arr)[:12]
        names = [base_features[i] for i in order][::-1]
        vals = [float(arr[i]) for i in order][::-1]
        fig = go.Figure(layout=_layout(f"Global mean|SHAP| — {LABELS[family]}", yaxis="feature", xaxis="mean|SHAP| (aggregated per feature)"))
        fig.add_trace(go.Bar(x=vals, y=names, orientation="h", marker_color=COLORS[family]))
        out.append(_chart(specs, f"{prefix}bar_{family}", fig))

        frame = frames[family]
        if len(frame) and len(top_names):
            bm = base_matrix(frame)
            use = [t for t in top_names if t in bm.columns][::-1]
            rng = np.random.default_rng(0)
            fig = go.Figure(layout=_layout(f"Per-sample SHAP distribution — {LABELS[family]}", yaxis="feature", xaxis="SHAP value (aggregated, signed)"))
            color_vals = np.full((len(bm), len(use)), np.nan)
            raw_df = ctx["df"]
            for j, feat in enumerate(use):
                if feat in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[feat]):
                    v = pd.to_numeric(raw_df[feat], errors="coerce").to_numpy(dtype=float)[frame["_pos"].to_numpy()]
                    lo_, hi_ = np.nanmin(v), np.nanmax(v)
                    if hi_ > lo_:
                        color_vals[:, j] = (v - lo_) / (hi_ - lo_)
            for j, feat in enumerate(use):
                x = bm[feat].to_numpy()
                yj = j + rng.uniform(-0.25, 0.25, len(x))
                fig.add_trace(go.Scatter(
                    x=x.tolist(), y=yj.tolist(), mode="markers", name=feat, showlegend=False,
                    marker={"size": 3, "color": color_vals[:, j].tolist(), "coloraxis": "coloraxis1", "opacity": 0.7},
                ))
            fig.update_layout(coloraxis1={"colorscale": [[0, "#2563eb"], [1, "#e11d48"]], "showscale": True,
                                         "colorbar": {"title": {"text": "feature value (norm)"}, "thickness": 8}},
                              yaxis={"tickmode": "array", "tickvals": list(range(len(use))), "ticktext": use})
            out.append(_chart(specs, f"{prefix}strip_{family}", fig))
    return out


def _agreement_figure(specs, ctx, prefix=""):
    expl = ctx["expl"]
    gb = expl["gbdt_gain_importance"]
    nn = expl["nn_mean_abs_shap"]
    feats = expl["base_features"]
    idx = np.argsort(-np.maximum(gb, nn))[:30]
    gsum, nsum = float(np.sum(gb)) or 1.0, float(np.sum(nn)) or 1.0
    rho = expl["attribution_agreement_spearman"]
    fig = go.Figure(layout=_layout(
        f"Attribution agreement: LightGBM gain vs MLP SHAP — Spearman ρ = {rho if rho is not None else 'n/a'}",
        height=380, xaxis="LightGBM gain share", yaxis="MLP mean|SHAP| share"))
    fig.add_trace(go.Scatter(
        x=[float(gb[i] / gsum) for i in idx], y=[float(nn[i] / nsum) for i in idx],
        mode="markers+text", text=[feats[i] for i in idx], textposition="top center",
        textfont={"size": 9}, marker={"size": 8, "color": "#16a34a"},
    ))
    return _chart(specs, prefix + "agree", fig)


def _data_quality_figures(specs, ctx, prefix=""):
    profile = ctx["profile"]
    out = []
    warn_html = "".join(f"<li>{_esc(w)}</li>" for w in profile.get("warnings", []))
    out.append(f"<div class='card'><h3>Data-quality warnings ({len(profile.get('warnings', []))})</h3><ul class='warnlist'>{warn_html or '<li>None detected.</li>'}</ul></div>")

    numeric_features = [c for c in ctx["feature_cols"] if c in profile.get("numeric_feature_columns", [])]
    clusters = profile.get("collinear_feature_clusters", [])
    heat_cols = sorted({c for cl in clusters for c in cl})[:15] or numeric_features[:15]
    if len(heat_cols) >= 2:
        corr = ctx["df"][heat_cols].corr().round(3)
        fig = go.Figure(go.Heatmap(z=corr.to_numpy().tolist(), x=heat_cols, y=heat_cols,
                                   zmin=-1, zmax=1, colorscale="RdBu_r", reversescale=True,
                                   colorbar={"thickness": 10}))
        fig.update_layout(_layout("Feature collinearity (|r|>0.90 flagged as clusters)", height=380))
        out.append(_chart(specs, prefix + "corr", fig))

    miss = [(c["name"], c["missing_pct"]) for c in profile["columns"] if c["missing_pct"] > 0]
    if miss:
        miss.sort(key=lambda x: -x[1])
        fig = go.Figure(layout=_layout("Missingness (% of rows)", height=max(240, 24 * len(miss) + 90)))
        fig.add_trace(go.Bar(x=[m[0] for m in miss], y=[m[1] for m in miss], marker_color="#f59e0b"))
        out.append(_chart(specs, prefix + "missing", fig))

    if profile.get("class_balance"):
        cb = profile["class_balance"]
        fig = go.Figure(layout=_layout("Class balance (target)", xaxis="class", yaxis="rows"))
        fig.add_trace(go.Bar(x=list(cb.keys()), y=list(cb.values()), marker_color="#16a34a"))
        out.append(_chart(specs, prefix + "balance", fig))
    elif ctx["task"] is TaskType.REGRESSION:
        fig = go.Figure(layout=_layout("Target distribution"))
        fig.add_trace(go.Histogram(x=ctx["df"][ctx["manifest"]["dataset"]["target_column"]].to_numpy().tolist(),
                                   marker_color="#16a34a", nbinsx=40))
        out.append(_chart(specs, prefix + "target", fig))
    return out


def _error_table(ctx) -> str:
    worst: dict = ctx["worst"]
    if not worst:
        return "<p>No worst-error rows found.</p>"
    oof = ctx["oof_df"]
    df = ctx["df"]
    task: TaskType = ctx["task"]
    expl = ctx["expl"]
    frames = expl["shap_frames"]
    enc_map = expl["encoded_to_base"]
    feat_show = ctx["feature_cols"][:10]

    def local_top3(family: str, pos: int) -> str:
        frame = frames[family]
        row = frame[frame["_pos"] == pos]
        if not len(row):
            return "—"
        agg: dict[str, float] = {}
        r = row.iloc[0]
        for c in frame.columns:
            if c.startswith("_"):
                continue
            b = enc_map.get(c, c)
            agg[b] = agg.get(b, 0.0) + float(r[c])
        ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        parts = []
        for f, v in ranked:
            sgn = "+" if v >= 0 else "−"
            parts.append(f"<span class='{ 'pos' if v >= 0 else 'neg'}'>{_esc(f)} {sgn}{abs(v):.3f}</span>")
        return " · ".join(parts) or "—"

    rows_html = []
    for group, positions in worst.items():
        for pos in positions:
            true = oof["y_true"].iloc[pos] if "y_true" in oof.columns else _fmt(oof["y_true_code"].iloc[pos], 3)
            preds = []
            for family in MODEL_FAMILIES:
                col = f"{family}_pred"
                p = oof[col].iloc[pos]
                preds.append(f"{_esc(LABELS[family].split()[0])}: {_fmt(p, 4)}")
            vals = []
            for f in feat_show:
                v = df[f].iloc[pos]
                vals.append(_esc("" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, (int, float, np.floating)) else v)))
            rows_html.append(
                f"<tr><td>{_esc(group)}</td><td class='mono'>{pos}</td>"
                f"<td>{'<br>'.join(preds)}</td><td class='mono'><b>{_esc(true)}</b></td>"
                f"<td>{' | '.join(vals)}</td>"
                + "".join(f"<td class='small'>{local_top3(fam, pos)}</td>" for fam in MODEL_FAMILIES)
                + "</tr>"
            )
    head = ("<tr><th>Error class</th><th>row</th><th>predicted</th><th>true</th>"
            f"<th>feature values ({_esc(', '.join(feat_show))})</th>"
            + "".join(f"<th>top-3 SHAP ({_esc(LABELS[f])})</th>" for f in MODEL_FAMILIES) + "</tr>")
    return f"<table class='tbl err'>{head}{''.join(rows_html)}</table>"


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #

_TEMPLATE = Template("""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root { --bg:#f5f6fa; --card:#ffffff; --ink:#111827; --mut:#6b7280; --accent:#2563eb; --border:#e5e7eb; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
header.topbar { background:#0b1220; color:#fff; padding:14px 22px; display:flex; justify-content:space-between; align-items:center; }
.brand { font-weight:700; letter-spacing:.3px; } .brand span { color:#93c5fd; }
.runmeta { color:#9ca3af; font-size:12px; }
main { padding:18px 22px 60px; max-width:1280px; margin:0 auto; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:16px 0; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }
.card .v { font-size:20px; font-weight:700; margin:4px 0 2px; font-variant-numeric:tabular-nums; }
.card .s { font-size:11px; color:var(--mut); }
.card .v.win { color:var(--accent); }
.warnbar { background:#fef3c7; border:1px solid #f59e0b; border-radius:10px; padding:10px 14px; margin:10px 0; font-size:13px; }
nav.tabs { display:flex; gap:6px; margin:18px 0 14px; position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:5; flex-wrap:wrap; }
.tab-btn { border:1px solid var(--border); background:var(--card); padding:8px 14px; border-radius:999px; cursor:pointer; font-size:13px; }
.tab-btn.active { background:#0b1220; color:#fff; border-color:#0b1220; }
.panel { display:none; } .panel.active { display:block; }
.section { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; margin:0 0 14px; }
.section h3 { margin:0 0 10px; font-size:14px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; } .grid2 > .section { margin:0; }
@media (max-width: 980px) { .grid2 { grid-template-columns:1fr; } }
.plot { width:100%; min-height:240px; }
table.tbl { width:100%; border-collapse:collapse; font-size:12.5px; margin-bottom:8px; }
table.tbl th { text-align:left; background:#f3f4f6; padding:7px 9px; border-bottom:2px solid var(--border); font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#374151; }
table.tbl td { padding:7px 9px; border-bottom:1px solid var(--border); vertical-align:top; }
table.tbl tr:hover td { background:#f9fafb; }
.mono { font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.small { font-size:11.5px; }
.pill { color:#fff; padding:2px 8px; border-radius:999px; font-size:11px; white-space:nowrap; }
.pos { color:#b91c1c; } .neg { color:#1d4ed8; }
.warnlist { margin:0; padding-left:18px; font-size:13px; line-height:1.55; }
ul.compact li { margin-bottom:3px; }
code { background:#f3f4f6; padding:1px 5px; border-radius:5px; font-size:12px; }
footer { color:var(--mut); font-size:12px; margin-top:24px; line-height:1.7; }
</style></head>
<body>
<header class="topbar">
  <div class="brand">Tabular<span>XAI</span>·MCP — Experiment Report</div>
  <div class="mono runmeta">run {{ run_id }} · {{ created_at }}</div>
</header>
<main>
<div class="cards">
{% for c in header_cards %}  <div class="card"><div class="k">{{ c.k }}</div><div class="v {{ c.cls }}">{{ c.v }}</div><div class="s">{{ c.s }}</div></div>
{% endfor %}</div>
{% if warning_banner %}<div class="warnbar">{{ warning_banner | safe }}</div>{% endif %}
<nav class="tabs">
{% for t in tabs %}  <button class="tab-btn{% if loop.first %} active{% endif %}" data-tab="{{ t.id }}">{{ t.label }}</button>
{% endfor %}</nav>
{% for t in tabs %}<section class="panel{% if loop.first %} active{% endif %}" id="panel-{{ t.id }}">{{ t.body | safe }}</section>
{% endfor %}
<footer>{{ footer | safe }}</footer>
</main>
<script>
const SPECS = {{ specs_json }};
const CFG = {{ config_json }};
const inited = new Set();
function initIn(panel){
  panel.querySelectorAll(".plot[data-spec]").forEach(el => {
    if (inited.has(el.id)) return;
    const s = SPECS[el.dataset.spec];
    Plotly.newPlot(el.id, s.data, s.layout, CFG);
    inited.add(el.id);
  });
}
document.querySelectorAll(".tab-btn").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "panel-" + btn.dataset.tab));
  initIn(document.getElementById("panel-" + btn.dataset.tab));
}));
initIn(document.querySelector(".panel.active"));
</script>
</body></html>""")


def _card(k, v, s="", cls="") -> dict:
    return {"k": _esc(k), "v": _esc(v), "s": _esc(s), "cls": cls}


def _section(title: str, inner: str) -> str:
    return f"<div class='section'><h3>{title}</h3>{inner}</div>"


def _grid2(*sections: str) -> str:
    return f"<div class='grid2'>{''.join(sections)}</div>"


def write_report(ctx: dict) -> Path:
    run_dir = Path(ctx["run_dir"])
    manifest = ctx["manifest"]
    task: TaskType = ctx["task"]
    specs: dict = {}

    # ---- header cards ------------------------------------------------------
    cmp_ = manifest["model_comparison"]
    winner = manifest["winner"]
    pm = manifest["winning_metric"]
    header_cards = [
        _card("Task", manifest["task_type"], f"{manifest['dataset']['n_rows']} rows × {manifest['dataset']['n_features']} features"),
        _card("Winner (" + pm + ")", _fmt(cmp_[winner]["oof"][pm]), LABELS[winner], "win"),
        _card("Runner-up", _fmt(cmp_[[f for f in MODEL_FAMILIES if f != winner][0]]["oof"][pm]), LABELS[[f for f in MODEL_FAMILIES if f != winner][0]]),
        _card("Calibration (winner)", json.dumps({k: v for k, v in list(manifest["calibration_summary"].items())[:1]}) if manifest["calibration_summary"] else "n/a", "see arena table"),
        _card("Attribution ρ", _fmt(manifest["attribution_agreement_spearman"], 3), "LightGBM gain vs MLP SHAP"),
        _card("Manifest size", f"{manifest.get('estimated_manifest_tokens', 'n/a')} tok", "validated < 2,000 tokens"),
    ]

    tabs = []

    # a) arena
    m1, m2 = _METRIC_SHOW[task]
    arena_body = _section("Model arena summary (fold mean±std + pooled out-of-fold)", _arena_table(ctx))
    arena_body += _grid2(
        _section("Fold metrics", _chart(specs, "box1", go.Figure()) if False else _box_figure(specs, ctx, "box1", m1)),
        _section("Fold metrics (2)", _box_figure(specs, ctx, "box2", m2)),
    )
    tabs.append({"id": "arena", "label": "Model Arena", "body": arena_body})

    # b) diagnostics
    if task.is_classification:
        diag_body = _grid2(*_classification_curve_figures(specs, ctx))
        diag_body += _section("", "")
    else:
        figs = _regression_figures(specs, ctx)
        diag_body = _grid2(figs[0], figs[1])
    diag_body = diag_body.replace("<div class='section'><h3></h3></div>", "")
    tabs.append({"id": "diag", "label": "Diagnostics", "body": diag_body})

    # c) explainability
    shap_body = _grid2(*_shap_figures(specs, ctx, prefix="shap_"))
    shap_body += _section("Cross-model attribution agreement", _agreement_figure(specs, ctx, prefix="ag_"))
    tabs.append({"id": "explain", "label": "Explainability", "body": shap_body})

    # d) data quality
    dq_body = "".join(
        s if s.startswith("<div class='section'") else _section("Diagnostics", s)
        for s in _data_quality_figures(specs, ctx, prefix="dq_")
    )
    dq_body = _section("Correlation & distribution context", dq_body)
    tabs.append({"id": "dq", "label": "Data Quality", "body": dq_body})

    # e) errors
    tabs.append({"id": "errors", "label": "Worst Predictions",
                 "body": _section("Worst-prediction rows (out-of-fold) with local top-3 SHAP drivers", _error_table(ctx))})

    warning_banner = None
    if manifest["dataset_warnings"]:
        warning_banner = (
            f"<b>{len(manifest['dataset_warnings'])} data-quality findings</b> — see Data Quality tab: "
            + "; ".join(w[:90] for w in manifest["dataset_warnings"][:3])
        )

    ap = manifest["artifact_paths"]
    footer = (
        "Generated by <b>tabular-xai-mcp</b> — dual-artifact pattern. Raw files: "
        f"<code>oof_predictions.parquet</code> <code>shap_values_gbdt.parquet</code> "
        f"<code>shap_values_nn.parquet</code> <code>metrics.json</code> (absolute paths in manifest.json artifact_paths).<br>"
        f"Agent artifact: <code>{_esc(ap.get('manifest_json', ''))}</code>"
    )

    html_out = _TEMPLATE.render(
        title=f"TabularXAI run {manifest['run_id']}",
        run_id=manifest["run_id"],
        created_at=manifest["created_at"],
        header_cards=header_cards,
        warning_banner=warning_banner,
        tabs=tabs,
        footer=footer,
        specs_json=json.dumps(specs),
        config_json=json.dumps(PLOT_CONFIG),
    )
    out_path = run_dir / "report.html"
    out_path.write_text(html_out, encoding="utf-8")
    return out_path
