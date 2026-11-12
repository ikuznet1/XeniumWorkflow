"""Small Dash building blocks: labels, plot-tab controls, the jobs panel rows."""

from dash import dcc, html

from . import core, jobs


# ─── UI micro-components ──────────────────────────────────────────────────────
def ctrl_label(text):
    return html.Div(text, style={
        "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px",
        "color": core.MUTED, "textTransform": "uppercase", "marginBottom": "6px",
    })


def stat_row(label, value):
    return html.Div([
        html.Span(label + ": ", style={"color": core.MUTED, "fontSize": "11px"}),
        html.Span(value,        style={"color": core.TEXT,  "fontSize": "12px", "fontWeight": "600"}),
    ], style={"marginBottom": "4px"})


def info_chip(label, value):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": core.MUTED}),
        html.Div(value, style={"fontSize": "13px", "fontWeight": "600", "color": core.TEXT}),
    ], style={
        "backgroundColor": core.DARK_BG, "border": f"1px solid {core.BORDER}",
        "borderRadius": "4px", "padding": "4px 10px",
    })


# ─── Plot tab control helpers ─────────────────────────────────────────────────
_PLOT_CTRL_INPUT = {
    "width": "60px", "backgroundColor": "#0d1117", "color": core.TEXT,
    "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
    "fontSize": "11px", "padding": "2px 4px",
}

def _plot_dim_controls(prefix: str):
    """Width × Height inputs for a plot tab."""
    return html.Div([
        html.Span("W", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-plot-w", type="number", placeholder="W",
                  debounce=True, style=_PLOT_CTRL_INPUT),
        html.Span("×", style={"color": core.MUTED, "fontSize": "10px", "margin": "0 2px"}),
        html.Span("H", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-plot-h", type="number", placeholder="H",
                  debounce=True, style=_PLOT_CTRL_INPUT),
    ], style={"display": "flex", "alignItems": "center", "gap": "4px",
              "padding": "4px 6px", "flexShrink": "0", "borderTop": f"1px solid {core.BORDER}"})

def _plot_normalize_and_dim_controls(prefix: str):
    """Normalize toggle + min-freq filter + dimensions for interaction/chord tabs."""
    return html.Div([
        dcc.Checklist(
            id=f"{prefix}-normalize",
            options=[{"label": html.Span(" O/E normalize", style={"fontSize": "11px", "color": core.TEXT}),
                      "value": "on"}],
            value=[], inputStyle={"marginRight": "4px"},
            labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer"},
            style={"display": "inline-flex"},
        ),
        html.Span("", style={"width": "10px"}),
        html.Span("Min%", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-min-freq", type="number", placeholder="0", min=0, max=100,
                  debounce=True, style=_PLOT_CTRL_INPUT),
        html.Span("", style={"width": "10px"}),
        html.Span("W", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-plot-w", type="number", placeholder="W",
                  debounce=True, style=_PLOT_CTRL_INPUT),
        html.Span("×", style={"color": core.MUTED, "fontSize": "10px", "margin": "0 2px"}),
        html.Span("H", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-plot-h", type="number", placeholder="H",
                  debounce=True, style=_PLOT_CTRL_INPUT),
    ], style={"display": "flex", "alignItems": "center", "gap": "4px",
              "padding": "4px 6px", "flexShrink": "0", "borderTop": f"1px solid {core.BORDER}"})

def _plot_range_and_dim_controls(prefix: str):
    """Color range (min/max) + dimensions for heatmap tabs."""
    return html.Div([
        html.Span("Color:", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-cbar-min", type="number", placeholder="Min",
                  debounce=True, style=_PLOT_CTRL_INPUT),
        html.Span("–", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-cbar-max", type="number", placeholder="Max",
                  debounce=True, style=_PLOT_CTRL_INPUT),
        html.Span("", style={"width": "12px"}),
        html.Span("W", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-plot-w", type="number", placeholder="W",
                  debounce=True, style=_PLOT_CTRL_INPUT),
        html.Span("×", style={"color": core.MUTED, "fontSize": "10px", "margin": "0 2px"}),
        html.Span("H", style={"color": core.MUTED, "fontSize": "10px"}),
        dcc.Input(id=f"{prefix}-plot-h", type="number", placeholder="H",
                  debounce=True, style=_PLOT_CTRL_INPUT),
    ], style={"display": "flex", "alignItems": "center", "gap": "4px",
              "padding": "4px 6px", "flexShrink": "0", "borderTop": f"1px solid {core.BORDER}"})


def method_label(m):
    return (m.replace("gene_expression_", "")
             .replace("_clusters", "")
             .replace("_", " ").title())

# ─── Jobs panel (rendering) ───────────────────────────────────────────────────
_JOBS_PANEL_STYLE = {
    "position": "fixed", "right": "12px", "bottom": "12px", "width": "340px",
    "maxHeight": "60vh", "overflowY": "auto", "zIndex": 2000,   # above dbc.Modal (1055)
    "backgroundColor": core.PANEL_BG, "border": f"1px solid {core.BORDER}", "borderRadius": "6px",
    "boxShadow": "0 4px 16px rgba(0,0,0,0.5)", "padding": "8px 10px",
    "fontSize": "11px", "color": core.TEXT,
}
_JOBS_PANEL_HIDDEN = {"display": "none"}
_JOB_OK_COLOR, _JOB_BAD_COLOR = "#3fb950", "#f85149"


def _fmt_elapsed(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _job_row(row: dict):
    """One job of the jobs panel: kind, label, elapsed time, step, progress bar
    (when a fraction is known) and Cancel."""
    state = row["state"]
    if state == "running" and row["cancellable"]:
        action = html.Button(
            "Cancel", id={"type": "job-cancel-btn", "job": row["id"]}, n_clicks=0,
            title="Stop this job. External tools are terminated; Python / R steps stop "
                  "after the current step.",
            style={"fontSize": "10px", "padding": "1px 8px", "cursor": "pointer",
                   "backgroundColor": core.CARD_BG, "color": core.TEXT,
                   "border": f"1px solid {core.BORDER}", "borderRadius": "4px"})
    elif state == "running":
        action = html.Span("writing results — cannot be cancelled now",
                           style={"color": core.MUTED, "fontStyle": "italic"})
    elif state == "cancelling":
        action = html.Span("stopping the tool…" if row["n_procs"]
                           else "cancelling after the current step…",
                           style={"color": "#d29922", "fontStyle": "italic"})
    elif state == "done":
        action = html.Span("✓ done", style={"color": _JOB_OK_COLOR})
    elif state == "cancelled":
        action = html.Span("✗ cancelled", style={"color": core.MUTED})
    else:
        action = html.Span("✗ failed", style={"color": _JOB_BAD_COLOR})

    parts = [
        html.Div([
            html.Span(row["kind"], style={
                "backgroundColor": core.CARD_BG, "border": f"1px solid {core.BORDER}",
                "borderRadius": "8px", "padding": "0 6px", "marginRight": "6px",
                "color": core.ACCENT, "fontSize": "10px", "whiteSpace": "nowrap"}),
            html.Span(row["label"], title=row["label"], style={
                "flex": "1", "minWidth": "0", "overflow": "hidden",
                "textOverflow": "ellipsis", "whiteSpace": "nowrap", "fontWeight": "600"}),
            html.Span(_fmt_elapsed(row["elapsed"]), className="job-elapsed",
                      style={"marginLeft": "8px", "color": core.MUTED,
                             "fontVariantNumeric": "tabular-nums"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(row["step"], title=row["step"], className="job-step", style={
            "color": core.MUTED, "marginTop": "2px", "overflow": "hidden",
            "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
    ]
    frac = row["fraction"]
    if frac is not None and state in jobs._JOB_ACTIVE:
        # one decimal below 10 %: a whole-slide run of many patches moves slowly
        pct = f"{100 * frac:.1f}" if frac < 0.0995 else f"{100 * frac:.0f}"
        parts.append(html.Div([
            html.Div(html.Div(className="job-progress-fill", style={
                "width": f"{100 * frac:.1f}%", "height": "100%", "backgroundColor": core.ACCENT,
                "borderRadius": "3px", "transition": "width 0.4s"}),
                className="job-progress", **{"data-fraction": f"{frac:.3f}"},
                style={"flex": "1", "height": "6px", "backgroundColor": core.BORDER,
                       "borderRadius": "3px", "overflow": "hidden"}),
            html.Span(f"{pct}%", style={"marginLeft": "6px", "color": core.MUTED, "width": "34px",
                                        "textAlign": "right",
                                        "fontVariantNumeric": "tabular-nums"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "4px"}))
    parts.append(html.Div(action, style={"marginTop": "4px", "textAlign": "right"}))
    return html.Div(parts, className="job-row", **{"data-job": row["id"], "data-state": state},
                    style={"padding": "6px 0", "borderTop": f"1px solid {core.BORDER}"})


def _jobs_panel_summary(rows: list) -> list:
    """Header line of the panel (the <summary>: a click on it folds the rows away)."""
    n_active = sum(r["state"] in jobs._JOB_ACTIVE for r in rows)
    return [
        html.Span("Jobs", style={"fontWeight": "600", "color": core.TEXT}),
        html.Span(f"{n_active} running" if n_active else "finished",
                  style={"marginLeft": "8px", "color": core.MUTED}),
    ]


def _jobs_panel_children(rows: list) -> list:
    return [_job_row(r) for r in rows]
