"""The Dash app object and its layout (sidebar, plots, modals); REPL switch."""

import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from . import compat, core, ui, dataview, morph, analysis, stats, controls, proximity, sdata_store, startup
from . import aggregates
from . import imgseg
from . import joint
from . import banksy
from . import segfree
from . import txfactor
from . import lrdecomp


with core._spage_lock:
    _spage_autoload_result = core._spage_state.get("result")
    _spage_autoload_genes  = core._spage_state.get("result_genes") or []

_base_gene_options = [{"label": g, "value": g} for g in dataview._sorted_gene_names]
_imp_gene_list = (sorted(_spage_autoload_result.columns) if _spage_autoload_result is not None
                  else sorted(_spage_autoload_genes))
if _imp_gene_list:
    _imp_opts    = [{"label": f"{g} [imp]", "value": f"{g} [imp]"} for g in _imp_gene_list]
    gene_options = _base_gene_options + _imp_opts
else:
    gene_options = _base_gene_options

sidebar = html.Div([
    # Header + collapsible tissue info (rebuilt dynamically on dataset change)
    html.Div(id="tissue-info-content"),

    html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 14px 0"}),

    # Subset indicator (hidden when no subset is active)
    html.Div(id="subset-indicator", style={"display": "none"},
             children=[
                 html.Div([
                     html.Span("⬡ SUBSET ACTIVE", style={
                         "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px",
                         "color": "#f0a500",
                     }),
                     html.Span(id="subset-count", style={
                         "fontSize": "10px", "color": core.MUTED, "marginLeft": "6px",
                     }),
                 ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),
                 html.Button("✕ Clear Subset", id="unsubset-btn", n_clicks=0, style={
                     "width": "100%", "padding": "3px 0", "marginTop": "4px",
                     "backgroundColor": core.CARD_BG, "color": "#f0a500",
                     "border": "1px solid #f0a500", "borderRadius": "4px",
                     "cursor": "pointer", "fontSize": "11px",
                 }),
                 html.Hr(style={"borderColor": core.BORDER, "margin": "10px 0 14px 0"}),
             ]),

    ui.ctrl_label("Segmentation Source"),
    dcc.Dropdown(
        id="seg-source",
        options=[
            {"label": "Xenium (original)", "value": "xenium"},
            {"label": "Baysor",            "value": "baysor", "disabled": True},
            {"label": "Proseg",            "value": "proseg", "disabled": True},
        ],
        value="xenium",
        clearable=False,
        style={"fontSize": "12px", "marginBottom": "6px"},
    ),
    html.Div(
        html.Button(
            "🗑 Delete this run",
            id="seg-delete-btn", n_clicks=0,
            style={
                "width": "100%", "padding": "4px 0",
                "backgroundColor": "transparent", "color": "#da3633",
                "border": "1px solid #da3633", "borderRadius": "4px",
                "cursor": "pointer", "fontSize": "11px",
            },
        ),
        id="seg-delete-div",
        style={"display": "none", "marginBottom": "14px"},
    ),

    html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 14px 0"}),

    # Color by
    ui.ctrl_label("Color By"),
    dcc.Dropdown(
        id="color-by",
        options=[
            {"label": "Cluster",                    "value": "cluster"},
            {"label": "Gene Expression",            "value": "gene"},
            {"label": "Cell Type (CellTypist)",     "value": "cell_type:celltypist", "disabled": True},
            {"label": "Cell Type (Seurat)",         "value": "cell_type:seurat",     "disabled": True},
            {"label": "Cell Type (RCTD)",           "value": "cell_type:rctd",       "disabled": True},
            {"label": "Transcript Counts",          "value": "transcript_counts"},
            {"label": "Cell Area",         "value": "cell_area"},
            {"label": "Nucleus Area",      "value": "nucleus_area"},
            {"label": "Spatial Niches",    "value": "niche",          "disabled": True},
            # ROIs auto-loaded from cache at startup (later changes: update_color_by_for_rois)
            *[{"label": f"ROI: {c}", "value": f"roi_{c}"}
              for c in dict.fromkeys(r["cls"] for r in core._roi_state["rois"])],
            # gene sets saved for this dataset (later changes: update_color_by_for_genesets)
            *[{"label": f"Gene set: {n}", "value": f"geneset:{n}"} for n in analysis._geneset_names()],
            *analysis._view_color_extra_options(),
        ],
        value="cluster",
        clearable=False,
        style={"fontSize": "12px", "marginBottom": "14px"},
    ),

    html.Div(id="cluster-ctrl", children=[
        ui.ctrl_label("Clustering Method"),
        dcc.Dropdown(
            id="cluster-method",
            options=startup.cluster_options,
            value=startup.cluster_methods[0] if startup.cluster_methods else None,
            clearable=False, style={"fontSize": "12px"},
        ),
    ], style={"marginBottom": "14px"}),

    html.Div(id="gene-ctrl", children=[
        ui.ctrl_label("Gene"),
        dcc.Dropdown(
            id="gene-selector",
            options=gene_options,
            value=dataview._default_gene,
            clearable=False, placeholder="Search gene…",
            style={"fontSize": "12px"},
        ),
    ], style={"marginBottom": "14px", "display": "none"}),

    html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 14px 0"}),

    # Boundary overlays
    ui.ctrl_label("Overlays"),
    html.Div([
        dcc.Checklist(
            id="boundary-toggles",
            options=[
                {"label": html.Span(" Xenium Cell Boundaries",    style={"color": "#00d4ff", "fontSize": "13px"}),
                 "value": "cell"},
                {"label": html.Span(" Xenium Nuclei Boundaries", style={"color": "#ff9f43", "fontSize": "13px"}),
                 "value": "nucleus"},
                {"label": html.Span(" Proseg Boundaries",  style={"color": "#2ed573", "fontSize": "13px"}),
                 "value": "proseg", "disabled": True},
                {"label": html.Span(" Baysor Boundaries",  style={"color": "#a29bfe", "fontSize": "13px"}),
                 "value": "baysor", "disabled": True},
                {"label": html.Span(" Image-segmentation Boundaries", style={"color": "#ff6b9d", "fontSize": "13px"}),
                 "value": "imgseg", "disabled": True},
            ],
            value=[],
            inputStyle={"marginRight": "6px"},
            labelStyle={
                "display": "flex", "alignItems": "center",
                "marginBottom": "7px", "cursor": "pointer",
            },
        ),
        html.Div(
            f"Zoom in to see boundaries (limit {core.BOUNDARY_CELL_LIMIT:,} cells in viewport).",
            style={"fontSize": "10px", "color": core.MUTED, "marginTop": "4px", "lineHeight": "1.4"},
        ),
        html.Div("Transcripts (WebGL view)",
                 style={"fontSize": "11px", "color": core.MUTED, "marginTop": "10px", "marginBottom": "4px"}),
        dcc.Dropdown(id="webgl-tx-genes", options=[], value=[], multi=True,
                     placeholder="Genes to draw as transcripts…",
                     style={"fontSize": "12px"}),
    ], style={"marginBottom": "10px"}),


    html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 14px 0"}),

    # Point style
    ui.ctrl_label("Point Size"),
    dcc.Slider(
        id="point-size", min=1, max=8, step=0.5, value=2,
        marks={1: "1", 4: "4", 8: "8"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),
    html.Div(style={"marginBottom": "12px"}),

    ui.ctrl_label("Opacity"),
    dcc.Slider(
        id="point-opacity", min=0.1, max=1.0, step=0.05, value=0.85,
        marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
        tooltip={"placement": "bottom", "always_visible": False},
    ),

    html.Div(style={"marginBottom": "12px"}),

    # Colorbar range controls (visible only for continuous color modes)
    html.Div(id="colorbar-ctrl", children=[
        ui.ctrl_label("Color Range"),
        html.Div([
            dcc.Input(id="colorbar-min", type="number", placeholder="Min",
                      debounce=True, style={
                          "width": "70px", "backgroundColor": "#0d1117", "color": core.TEXT,
                          "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                          "fontSize": "12px", "padding": "2px 6px",
                      }),
            html.Span("–", style={"color": core.MUTED, "margin": "0 6px"}),
            dcc.Input(id="colorbar-max", type="number", placeholder="Max",
                      debounce=True, style={
                          "width": "70px", "backgroundColor": "#0d1117", "color": core.TEXT,
                          "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                          "fontSize": "12px", "padding": "2px 6px",
                      }),
            html.Button("Reset", id="colorbar-reset-btn", n_clicks=0,
                        style={"marginLeft": "8px", "fontSize": "11px", "padding": "2px 8px",
                               "backgroundColor": "transparent", "color": core.MUTED,
                               "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                               "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
    ], style={"display": "none"}),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),

    # ── Morphology image overlay ────────────────────────────────────────
    ui.ctrl_label("Morphology Image"),
    html.Div(
        f"Loads tiles for viewports ≤{morph.MORPH_MAX_UM:,} µm. Zoom in to activate.",
        style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
    ),
    dcc.Checklist(
        id="morph-enable",
        options=[{"label": html.Span(" Enable Image Overlay", style={"fontSize": "13px", "color": core.TEXT}),
                  "value": "show"}],
        value=[],
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer"},
        style={"marginBottom": "8px"},
    ),
    html.Div(id="morph-controls", children=[
        # (wrapped so the WebGL glue can disable it and say why in a tooltip)
        html.Div(id="morph-zlevel-wrap", children=[
            ui.ctrl_label("Z-Level"),
            dcc.Dropdown(
                id="morph-zlevel",
                options=[{"label": f"Z-level {i}", "value": i} for i in range(4)],
                value=0, clearable=False,
                style={"fontSize": "12px", "marginBottom": "8px"},
            ),
        ]),
        ui.ctrl_label("Channels"),
        dcc.Checklist(
            id="morph-channels",
            options=[
                {"label": html.Span(f" {c['label']}",
                                    style={"color": f"rgb{c['color']}", "fontSize": "12px"}),
                 "value": c["value"]}
                for c in morph.MORPH_CHANNELS
            ],
            value=["dapi", "boundary"],
            inputStyle={"marginRight": "6px"},
            labelStyle={"display": "flex", "alignItems": "center",
                        "marginBottom": "5px", "cursor": "pointer"},
            style={"marginBottom": "8px"},
        ),
        ui.ctrl_label("Brightness"),
        dcc.Slider(
            id="morph-brightness", min=0.5, max=8, step=0.25, value=2,
            marks={0.5: "0.5×", 4: "4×", 8: "8×"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(style={"marginBottom": "10px"}),
        ui.ctrl_label("Image Opacity"),
        dcc.Slider(
            id="morph-opacity", min=0.1, max=1.0, step=0.05, value=0.85,
            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(style={"marginBottom": "8px"}),
    ], style={"display": "none"}),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),

    # ── Cell type annotation ────────────────────────────────────────────
    ui.ctrl_label("Cell Type Annotation"),
    html.Button(
        "Annotate Cells",
        id="annot-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#238636", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    html.Div(id="annot-status", style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px"}),
    dcc.Interval(id="annot-poll", interval=1000, disabled=True),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),

    # ── Resegmentation ──────────────────────────────────────────────────
    ui.ctrl_label("Resegmentation"),
    html.Button(
        "Resegment Cells",
        id="reseg-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#1f6feb", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    # Status line (shows whichever tool is running/done)
    html.Div(id="reseg-status", style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px", "marginBottom": "6px"}),
    # Hidden status divs for existing poll callbacks
    html.Div(id="baysor-status", style={"display": "none"}),
    html.Div(id="proseg-status", style={"display": "none"}),
    html.Div(id="imgseg-status", style={"display": "none"}),
    dcc.Interval(id="baysor-poll", interval=3000, disabled=True),
    dcc.Interval(id="proseg-poll", interval=3000, disabled=True),
    dcc.Interval(id="imgseg-poll", interval=3000, disabled=True),
    dcc.Store(id="iseg-run-request", data=0),
    dcc.Store(id="iseg-load-request", data=None),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),

    # ── SpaGE gene imputation ───────────────────────────────────────────
    ui.ctrl_label("SpaGE Gene Imputation"),
    html.Button(
        "Impute Genes",
        id="spage-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#6f42c1", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    html.Div(id="spage-status", style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px", "marginBottom": "6px"}),
    dcc.Interval(id="spage-poll", interval=3000, disabled=True),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),
    ui.ctrl_label("SPLIT Correction"),
    dbc.Button("✦ Correct Ambient RNA", id="split-modal-open-btn",
               style={"width":"100%","backgroundColor":"#e85d04","color":"white"}),
    html.Div(id="split-status",
             style={"fontSize":"0.75rem","marginTop":"4px","color":"#8b949e"}),
    dcc.Interval(id="split-poll", interval=3000, disabled=True),
    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),
    ui.ctrl_label("CellNEST Cell-Cell Comm."),
    html.Button(
        "Load CellNEST Results",
        id="cellnest-modal-open-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#0d9488", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "8px",
        },
    ),
    html.Div(id="cellnest-sidebar-status",
             style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px"}),
    dcc.Checklist(
        id="cellnest-overlay-enable",
        options=[{"label": html.Span(" Show edges on map", style={"fontSize": "11px", "color": core.TEXT}),
                  "value": "on"}],
        value=[],
        inputStyle={"marginRight": "4px"},
        labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer"},
        style={"marginTop": "4px"},
    ),
    dcc.Store(id="cellnest-version", data=0),
    dcc.Interval(id="cellnest-poll", interval=1000, disabled=True),
    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),
    ui.ctrl_label("Spatial Analysis"),
    html.Button("⬡ Spatial Analysis Tools", id="niche-modal-open-btn", n_clicks=0,
                style={"width": "100%", "backgroundColor": "#0d6efd", "color": "white",
                       "border": "none", "borderRadius": "4px", "padding": "6px 10px",
                       "fontSize": "0.82rem", "cursor": "pointer", "marginBottom": "4px"}),
    html.Div(id="niche-status", style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px",
                                       "marginBottom": "6px"}),
    html.Div(id="spatial-analysis-status",
             style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px",
                    "marginBottom": "6px"}),
    dcc.Interval(id="niche-poll", interval=2000, disabled=True),
    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),
    ui.ctrl_label("Active Counts"),
    dcc.RadioItems(
        id="counts-mode",
        options=[
            {"label": " Original",        "value": "original"},
            {"label": " SPLIT Corrected", "value": "corrected", "disabled": True},
        ],
        value="original",
        labelStyle={"display":"block","cursor":"pointer","marginBottom":"2px"},
        style={"fontSize":"0.82rem"},
    ),
    # with several samples: which of them carry SPLIT-corrected counts
    html.Div(id="counts-mode-note", style={"fontSize": "10px", "color": core.MUTED, "lineHeight": "1.35",
                                           "marginTop": "2px"}),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),
    ui.ctrl_label("ROI Annotations"),
    dcc.Checklist(
        id="roi-show-toggle",
        options=[{"label": " Show ROIs", "value": "show"}],
        value=["show"],
        style={"fontSize": "0.82rem", "marginBottom": "6px"},
    ),
    html.Button(
        "Manage / Draw ROIs",
        id="roi-manage-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": "#5b2d8e", "color": "#fff",
            "border": "none", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "6px",
        },
    ),
    html.Div(id="roi-sidebar-summary",
             style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px"}),

    html.Hr(style={"borderColor": core.BORDER, "margin": "14px 0"}),

    # UMAP toggle
    html.Button(
        "Show Plots",
        id="umap-toggle",
        n_clicks=1,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": core.CARD_BG, "color": core.TEXT,
            "border": f"1px solid {core.BORDER}", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
            "marginBottom": "6px",
        },
    ),
    html.Button(
        "Run UMAP on Reseg Cells",
        id="umap-reseg-btn",
        n_clicks=0,
        style={
            "width": "100%", "padding": "7px 0",
            "backgroundColor": core.CARD_BG, "color": core.TEXT,
            "border": f"1px solid {core.BORDER}", "borderRadius": "5px",
            "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
        },
    ),
    html.Div(id="umap-reseg-status",
             style={"fontSize": "11px", "color": core.MUTED, "minHeight": "14px",
                    "marginTop": "4px"}),
    dcc.Interval(id="umap-reseg-poll", interval=1500, disabled=True),
], style={
    "width": "240px", "minWidth": "240px",
    "backgroundColor": core.PANEL_BG, "borderRight": f"1px solid {core.BORDER}",
    "padding": "20px 16px", "overflowY": "auto",
    "height": "100vh", "boxSizing": "border-box",
})

app = dash.Dash(
    compat.entry_name(),
    external_stylesheets=[dbc.themes.DARKLY],
    title="Xenium Explorer",
    suppress_callback_exceptions=True,
)

# In-app Python REPL: off unless XENIUM_ENABLE_REPL is set ("1" = localhost requests only,
# "remote" = any client). It runs arbitrary code, so never enable "remote" on an open network.
_REPL_MODE         = os.environ.get("XENIUM_ENABLE_REPL", "").strip().lower()
_REPL_ENABLED      = _REPL_MODE in ("1", "true", "yes", "on", "remote")
_REPL_ALLOW_REMOTE = _REPL_MODE == "remote"


# Spatial renderer shown at startup: "plotly" (default) or "webgl" (deck.gl view, beta).
_DEFAULT_RENDERER = ("webgl" if os.environ.get("XENIUM_RENDERER", "").strip().lower() == "webgl"
                     else "plotly")

app.layout = html.Div([
    dcc.Store(id="selected-cell",    data=None),
    dcc.Store(id="spatial-relayout", data={}),
    dcc.Store(id="webgl-click", data=None),     # {index, source, x, y} of a cell clicked in the WebGL view
    dcc.Store(id="webgl-select", data=None),    # {indices, polygon} from a WebGL lasso / rectangle
    dcc.Store(id="annot-done",       data=0),
    dcc.Store(id="baysor-done",      data=0),
    dcc.Store(id="proseg-done",      data=0),
    dcc.Store(id="reseg-patches-confirmed", data=0),
    dcc.Store(id="spage-done",       data=0),
    dcc.Store(id="dataset-version",        data=0),
    dcc.Store(id="extra-datasets-version", data=0),
    dcc.Store(id="cache-clean-done", data=0),
    dcc.Interval(id="spage-repl-poll", interval=4000, n_intervals=0),
    dcc.Store(id="subset-version",   data=0),
    dcc.Store(id="sdata-done",       data=0),
    dcc.Store(id="split-done",        data=0),
    dcc.Store(id="niche-done",        data=0),
    dcc.Store(id="spatial-done", data=0),
    dcc.Interval(id="spatial-poll", interval=2000, disabled=True),
    # differential expression / gene-set scores / transcript density
    dcc.Store(id="de-done", data=0),
    dcc.Store(id="de-selection", data=0),        # bumped by every lasso / box selection
    dcc.Store(id="de-groups-version", data=0),
    dcc.Interval(id="de-poll", interval=700, disabled=True),
    dcc.Store(id="geneset-version", data=0),
    dcc.Interval(id="geneset-poll", interval=700, disabled=True),
    dcc.Store(id="density-done", data=0),
    dcc.Store(id="density-store", data={"on": False}),   # {on, gene, bin, opacity, max, version}
    dcc.Interval(id="density-poll", interval=700, disabled=True),
    # sample design / composition / Moran's I / ligand-receptor scores
    dcc.Store(id="design-version", data=0),
    dcc.Store(id="comp-version", data=0),
    dcc.Store(id="moran-done", data=0),
    dcc.Interval(id="moran-poll", interval=700, disabled=True),
    dcc.Store(id="lr-done", data=0),
    dcc.Interval(id="lr-poll", interval=700, disabled=True),
    stats._design_modal(),
    # confound controls: compartments / spillover / segmentation QC / stratified co-localisation
    dcc.Store(id="ctl-comp-done", data=0),
    dcc.Store(id="spill-done", data=0),
    dcc.Store(id="segqc-done", data=0),
    dcc.Store(id="coloc-done", data=0),
    dcc.Interval(id="ctl-poll", interval=700, disabled=True),
    dcc.Store(id="lrd-done", data=0),
    dcc.Interval(id="lrd-poll", interval=700, disabled=True),
    dcc.Store(id="agg-done", data=0),
    dcc.Store(id="clusterset-version", data=0),   # bumped when a cluster set is installed (Banksy, joint clusters)
    dcc.Store(id="bk-done", data=0),
    dcc.Interval(id="bk-poll", interval=700, disabled=True),
    dcc.Interval(id="agg-poll", interval=700, disabled=True),
    dcc.Store(id="jc-done", data=0),
    dcc.Interval(id="jc-poll", interval=700, disabled=True),
    dcc.Store(id="sfree-done", data=0),
    dcc.Interval(id="sfree-poll", interval=700, disabled=True),
    dcc.Store(id="counts-mode-store", data="original"),
    dcc.Store(id="roi-done",          data=0),
    dcc.Store(id="roi-pending",       data=None),
    dcc.Store(id="measure-store",     data={"active": False, "points": []}),
    dcc.Interval(id="subset-poll",   interval=500,  n_intervals=0),
    dcc.Interval(id="morph-hires-poll", interval=400, n_intervals=0, disabled=True),
    # Fires once after 500 ms to guarantee initial plot render in Dash 4
    dcc.Interval(id="startup-trigger", interval=500, max_intervals=1),
    # Jobs panel: floats above the page and the modals; its interval runs only while
    # a job is listed (see update_jobs_panel / wake_jobs_poll)
    dcc.Interval(id="jobs-poll", interval=1000, disabled=True),
    html.Details([
        html.Summary(id="jobs-panel-summary", title="Click to fold / unfold the job list",
                     style={"cursor": "pointer", "paddingBottom": "2px"}),
        html.Div(id="jobs-panel-rows"),
    ], id="jobs-panel", open=True, style=ui._JOBS_PANEL_HIDDEN),

    # ── Resegmentation modal ─────────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Resegment Cells"), close_button=True),
        dbc.ModalBody([
            # ── Patch Setup ──────────────────────────────────────────────────
            html.Div([
                html.Div("Step 1 — Prepare Patches", style={"fontSize": "13px", "fontWeight": "600", "color": core.TEXT, "marginBottom": "8px"}),
                html.Div(
                    "Set parameters then click Prepare Patches. The app will build a SpatialData object, "
                    "detect the tissue ROI, and partition it into patches. Review the overlays on the "
                    "spatial plot, then confirm before running segmentation.",
                    style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "10px", "lineHeight": "1.4"},
                ),
                html.Div([
                    html.Div([
                        ui.ctrl_label("QV Threshold"),
                        dcc.Slider(
                            id="sdata-qv", min=0, max=40, step=5, value=20,
                            marks={0: "0", 20: "20", 40: "40"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(style={"marginBottom": "10px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        ui.ctrl_label("Patch Width (µm)"),
                        dcc.Input(
                            id="sdata-patch-width", type="number", value=1000, min=100, max=10000,
                            debounce=True,
                            style={
                                "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                                "boxSizing": "border-box",
                            },
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        ui.ctrl_label("Min TX / Patch"),
                        dcc.Input(
                            id="sdata-patch-min-tx", type="number", value=10, min=1, max=1000,
                            debounce=True,
                            style={
                                "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                                "boxSizing": "border-box",
                            },
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "12px", "marginBottom": "6px"}),
                # Prepare button + status row
                html.Div([
                    html.Button(
                        "▶ Prepare Patches",
                        id="sdata-prepare-btn", n_clicks=0,
                        style={
                            "padding": "6px 14px", "backgroundColor": "#1f6feb", "color": "#fff",
                            "border": "none", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
                        },
                    ),
                    html.Button(
                        "↺ Re-build",
                        id="sdata-clear-cache-btn", n_clicks=0,
                        title="Clear disk cache and re-read from raw files",
                        style={
                            "padding": "6px 10px", "backgroundColor": core.CARD_BG, "color": core.MUTED,
                            "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "11px",
                        },
                    ),
                    html.Button(
                        "⏭ Skip — run on full tissue",
                        id="reseg-skip-patches-btn", n_clicks=0,
                        title="Skip patch building and run segmentation on the full tissue",
                        style={
                            "padding": "6px 10px", "backgroundColor": core.CARD_BG, "color": "#e3b341",
                            "border": f"1px solid #e3b341", "borderRadius": "4px",
                            "cursor": "pointer", "fontSize": "11px",
                        },
                    ),
                    html.Div(id="sdata-status",
                             style={"fontSize": "11px", "color": core.MUTED, "flex": "1", "minHeight": "16px", "alignSelf": "center"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "8px", "alignItems": "center"}),
                # Overlay checkboxes
                html.Div([
                    dcc.Checklist(
                        id="sdata-show-roi",
                        options=[{"label": html.Span(" Show tissue ROI", style={"fontSize": "11px", "color": "#52b788"}), "value": "yes"}],
                        value=[],
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"display": "flex", "alignItems": "center"},
                    ),
                    dcc.Checklist(
                        id="sdata-show-patches",
                        options=[{"label": html.Span(" Show patch grid", style={"fontSize": "11px", "color": "#ffa500"}), "value": "yes"}],
                        value=[],
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"display": "flex", "alignItems": "center"},
                    ),
                ], style={"display": "flex", "gap": "20px", "marginBottom": "10px"}),
                # Confirmation banner (hidden until patches are ready)
                html.Div(id="reseg-patch-confirm-div", children=[
                    html.Div(id="reseg-patch-confirm-msg",
                             style={"fontSize": "11px", "color": "#52b788", "marginBottom": "6px"}),
                    html.Button(
                        "✓ Patches look good — proceed to segmentation",
                        id="reseg-confirm-patches-btn",
                        n_clicks=0,
                        style={
                            "width": "100%", "padding": "7px 0",
                            "backgroundColor": "#2d6a4f", "color": "#fff",
                            "border": "none", "borderRadius": "5px",
                            "cursor": "pointer", "fontSize": "12px", "fontWeight": "600",
                            "marginBottom": "4px",
                        },
                    ),
                ], style={"display": "none"}),
                dcc.Interval(id="sdata-poll", interval=2000, disabled=True),
                # Hidden elements needed by existing callbacks
                html.Div(id="sdata-convert-btn", style={"display": "none"}),
                html.Div(id="sdata-roi-btn", style={"display": "none"}),
                html.Div(id="sdata-patches-btn", style={"display": "none"}),
            ], style={"backgroundColor": "#161b22", "borderRadius": "6px", "padding": "12px", "marginBottom": "14px"}),

            html.Hr(style={"borderColor": core.BORDER, "margin": "0 0 14px 0"}),

            html.Div([
                html.Div("Step 2 — Segmentation Settings", style={"fontSize": "13px", "fontWeight": "600", "color": core.TEXT, "marginBottom": "10px"}),
                html.Div(id="reseg-step2-overlay", style={"display": "block"}, children=[
                    html.Div("Confirm patches above to enable segmentation settings.",
                             style={"fontSize": "11px", "color": core.MUTED, "textAlign": "center", "padding": "8px 0"}),
                ]),
                html.Div(id="reseg-step2-content", style={"opacity": "0.4", "pointerEvents": "none"}, children=[
                    # Algorithm selector tabs
                    dcc.Tabs(id="reseg-algo-tabs", value="baysor", children=[
                        dcc.Tab(label="Baysor", value="baysor", style={"color": core.TEXT, "backgroundColor": core.CARD_BG},
                                selected_style={"color": "#fff", "backgroundColor": "#1f6feb", "fontWeight": "600"}),
                        dcc.Tab(label="Proseg", value="proseg", style={"color": core.TEXT, "backgroundColor": core.CARD_BG},
                                selected_style={"color": "#fff", "backgroundColor": "#2d6a4f", "fontWeight": "600"}),
                        dcc.Tab(label="Image (Cellpose / classical)", value="imgseg",
                                style={"color": core.TEXT, "backgroundColor": core.CARD_BG},
                                selected_style={"color": "#fff", "backgroundColor": "#9c3d6b", "fontWeight": "600"}),
                    ], style={"marginBottom": "16px"}),

                    # ── Baysor settings panel ────────────────────────────────────
            html.Div(id="modal-baysor-panel", children=[
                html.Div(
                    "Resegment cells using Baysor (must be installed). "
                    "See github.com/kharchenkolab/Baysor.",
                    style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
                ),
                ui.ctrl_label("Spatial Region (µm) — leave blank for full slide"),
                html.Div([
                    html.Div([
                        html.Div("X min", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-xmin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("X max", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-xmax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Div("Y min", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-ymin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("Y max", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="baysor-ymax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Button(
                    "Use Current Viewport", id="baysor-use-viewport", n_clicks=0,
                    style={"width": "100%", "padding": "4px 0", "backgroundColor": core.CARD_BG,
                           "color": core.MUTED, "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "cursor": "pointer", "fontSize": "11px", "marginBottom": "10px"},
                ),
                html.Div([
                    html.Div([
                        ui.ctrl_label("Cell Radius (µm, −1 = auto)"),
                        dcc.Input(
                            id="baysor-scale", type="number", value=20, min=-1, step=1,
                            style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                   "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        ui.ctrl_label("Scale Std (optional)"),
                        dcc.Input(
                            id="baysor-scale-std", type="number", value=None, min=0, step=0.05,
                            placeholder="e.g. 0.25",
                            style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                   "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px"}),
                html.Div([
                    html.Div([
                        ui.ctrl_label("Min Transcripts / Cell"),
                        dcc.Input(
                            id="baysor-min-mol", type="number", value=10, min=1, max=500, step=1,
                            style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                   "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                    html.Div([
                        ui.ctrl_label("Number of cell types"),
                        dcc.Input(
                            id="baysor-n_clusters", type="number", value=10, min=1, step=1,
                            style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                   "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                   "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px"},
                        ),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px"}),
                dcc.Checklist(
                    id="baysor-use-patches",
                    options=[{"label": html.Span(
                        " Use patch-based segmentation (from Step 1)",
                        style={"fontSize": "12px", "color": core.TEXT},
                    ), "value": "yes"}],
                    value=["yes"],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                ),
                dcc.Checklist(
                    id="baysor-use-prior",
                    options=[{"label": html.Span(
                        " Use Xenium nuclei as prior segmentation",
                        style={"fontSize": "12px", "color": core.TEXT},
                    ), "value": "yes"}],
                    value=["yes"],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                ),
                html.Div(id="baysor-prior-conf-div", children=[
                    ui.ctrl_label("Prior Confidence"),
                    dcc.Slider(
                        id="baysor-prior-conf", min=0.0, max=1.0, step=0.05, value=0.5,
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    ui.ctrl_label("Prior Type"),
                    dcc.RadioItems(
                        id="baysor-prior-type",
                        options=[
                            {"label": html.Span(" Nucleus boundaries", style={"fontSize": "12px", "color": core.TEXT}), "value": "nucleus"},
                            {"label": html.Span(" Cell boundaries",    style={"fontSize": "12px", "color": core.TEXT}), "value": "cell"},
                        ],
                        value="nucleus",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                    ),
                    html.Div(style={"marginBottom": "6px"}),
                ]),
            ]),

            # ── Proseg settings panel ────────────────────────────────────
            html.Div(id="modal-proseg-panel", children=[
                html.Div(
                    "Probabilistic cell segmentation (faster than Baysor). "
                    "Install: conda install bioconda::rust-proseg",
                    style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "8px", "lineHeight": "1.4"},
                ),
                ui.ctrl_label("Spatial Region (µm) — leave blank for full slide"),
                html.Div([
                    html.Div([
                        html.Div("X min", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-xmin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("X max", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-xmax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Div([
                    html.Div([
                        html.Div("Y min", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-ymin", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Div("Y max", style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "2px"}),
                        dcc.Input(id="proseg-ymax", type="number", placeholder="auto",
                                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                         "padding": "3px 6px", "fontSize": "11px", "boxSizing": "border-box"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "6px", "marginBottom": "6px"}),
                html.Button(
                    "Use Current Viewport", id="proseg-use-viewport", n_clicks=0,
                    style={"width": "100%", "padding": "4px 0", "backgroundColor": core.CARD_BG,
                           "color": core.MUTED, "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "cursor": "pointer", "fontSize": "11px", "marginBottom": "10px"},
                ),
                ui.ctrl_label("Prior Type"),
                dcc.RadioItems(
                    id="proseg-prior-type",
                    options=[
                        {"label": html.Span(" Nucleus boundaries (default)", style={"fontSize": "12px", "color": core.TEXT}), "value": "nucleus"},
                        {"label": html.Span(" Cell boundaries",              style={"fontSize": "12px", "color": core.TEXT}), "value": "cell"},
                    ],
                    value="nucleus",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
                    style={"marginBottom": "8px"},
                ),
                ui.ctrl_label("Initial voxel size (µm) — leave blank for default (4)"),
                dcc.Input(
                    id="proseg-voxel-size", type="number", placeholder="auto", min=0.1, max=10, step=0.1,
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ui.ctrl_label("Threads"),
                dcc.Input(
                    id="proseg-nthreads", type="number", placeholder="all", min=1, max=128, step=1,
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ui.ctrl_label("Components (--ncomponents)"),
                dcc.Input(
                    id="proseg-samples", type="number", placeholder="10", min=1, max=100, step=1,
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ui.ctrl_label("Recorded samples (--recorded-samples)"),
                dcc.Input(
                    id="proseg-recorded-samples", type="number", placeholder="100", min=10, max=1000, step=10,
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ui.ctrl_label("Sampler schedule (--schedule) — space-separated integers"),
                dcc.Input(
                    id="proseg-schedule", type="text", placeholder="150 150 300",
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ui.ctrl_label("Nuclear reassignment prob (--nuclear-reassignment-prob)"),
                dcc.Input(
                    id="proseg-nuclear-reassign-prob", type="number", placeholder="0.2",
                    min=0.0, max=1.0, step=0.05,
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
                ui.ctrl_label("Prior seg reassignment prob (--prior-seg-reassignment-prob)"),
                dcc.Input(
                    id="proseg-prior-seg-prob", type="number", placeholder="0.5",
                    min=0.0, max=1.0, step=0.05,
                    style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                           "padding": "4px 8px", "fontSize": "12px", "marginBottom": "8px",
                           "boxSizing": "border-box"},
                ),
            ], style={"display": "none"}),  # hidden by default; shown when proseg tab is active

            # ── Image-segmentation settings panel (Cellpose / classical) ─
            imgseg._iseg_panel(),
                ]),  # end reseg-step2-content
            ]),  # end Step 2 div
        ]),
        dbc.ModalFooter([
            html.Div(id="reseg-modal-run-status", style={"fontSize": "11px", "color": core.MUTED, "flex": "1"}),
            html.Button(
                "Run",
                id="reseg-modal-run-btn",
                n_clicks=0,
                disabled=True,
                style={
                    "padding": "7px 20px",
                    "backgroundColor": "#1f6feb", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="reseg-modal-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="reseg-modal", is_open=False, size="lg",
       style={"color": core.TEXT},
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
       backdrop=True),

# ── Cell Type Annotation modal ────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Cell Type Annotation"), close_button=True),
        dbc.ModalBody([
            html.Div(
                "Annotate cells using CellTypist, Seurat label transfer, or RCTD (spacexr).",
                style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "12px", "lineHeight": "1.4"},
            ),
            ui.ctrl_label("Method"),
            dcc.RadioItems(
                id="annot-source",
                options=[
                    {"label": html.Span(" CellTypist", style={"fontSize": "12px", "color": core.TEXT}),
                     "value": "celltypist"},
                    {"label": html.Span(" Seurat kNN label transfer", style={"fontSize": "12px", "color": core.TEXT}),
                     "value": "seurat"},
                    {"label": html.Span(" RCTD (spacexr)", style={"fontSize": "12px", "color": core.TEXT}),
                     "value": "rctd"},
                ],
                value="celltypist",
                inputStyle={"marginRight": "5px"},
                labelStyle={"display": "flex", "alignItems": "center", "marginBottom": "5px", "cursor": "pointer"},
                style={"marginBottom": "10px"},
            ),
            # CellTypist controls
            html.Div(id="annot-celltypist-div", children=[
                ui.ctrl_label("Model"),
                dcc.Dropdown(
                    id="annot-model",
                    options=[{"label": v, "value": k} for k, v in core.CELLTYPIST_MODELS.items()],
                    value="Healthy_Adult_Heart.pkl",
                    clearable=False,
                    style={"fontSize": "12px", "marginBottom": "10px",
                           "backgroundColor": core.CARD_BG, "color": core.TEXT},
                ),
            ]),
            # Seurat / RCTD shared RDS controls (shown for both)
            html.Div(id="annot-seurat-div", style={"display": "none"}, children=[
                ui.ctrl_label("RDS File Path"),
                dcc.Input(
                    id="annot-rds-path",
                    type="text",
                    value=core.REF_RDS_DEFAULT,
                    placeholder=core.REF_RDS_PLACEHOLDER,
                    style={
                        "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                        "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                        "boxSizing": "border-box",
                    },
                ),
                ui.ctrl_label("Cell Type Label Column"),
                dcc.Input(
                    id="annot-label-col",
                    type="text",
                    value="Names",
                    style={
                        "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                        "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "8px",
                        "boxSizing": "border-box",
                    },
                ),
            ]),
            # RCTD-specific controls (shown only for rctd)
            html.Div(id="annot-rctd-div", style={"display": "none"}, children=[
                html.Div(
                    "Requires the spacexr R package. In R: devtools::install_github('dmcable/spacexr')",
                    style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "8px",
                           "lineHeight": "1.4", "fontStyle": "italic"},
                ),
                ui.ctrl_label("RCTD Mode"),
                dcc.Dropdown(
                    id="annot-rctd-mode",
                    options=[
                        {"label": "full — weights across all types (best for single cells)", "value": "full"},
                        {"label": "doublet — primary + secondary type",                      "value": "doublet"},
                        {"label": "multi — multiple types per spot",                         "value": "multi"},
                    ],
                    value="full",
                    clearable=False,
                    style={"fontSize": "11px", "marginBottom": "8px",
                           "backgroundColor": core.CARD_BG, "color": core.TEXT},
                ),
                ui.ctrl_label("Max Cores"),
                dcc.Input(
                    id="annot-rctd-cores",
                    type="number", value=4, min=1, max=64, step=1,
                    style={
                        "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                        "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "4px",
                        "boxSizing": "border-box",
                    },
                ),
                ui.ctrl_label("Min UMI (cells below excluded)"),
                dcc.Input(
                    id="annot-rctd-umi-min",
                    type="number", value=20, min=1, max=500, step=1,
                    style={
                        "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                        "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "4px",
                        "boxSizing": "border-box",
                    },
                ),
                ui.ctrl_label("Min UMI sigma (UMI_min_sigma)"),
                dcc.Input(
                    id="annot-rctd-umi-min-sigma",
                    type="number", value=100, min=1, max=1000, step=1,
                    style={
                        "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                        "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                        "padding": "4px 8px", "fontSize": "11px", "marginBottom": "4px",
                        "boxSizing": "border-box",
                    },
                ),
            ]),
            html.Div(id="annot-label-report",
                     style={"borderTop": f"1px solid {core.BORDER}", "marginTop": "8px", "paddingTop": "4px"}),
        ]),
        dbc.ModalFooter([
            html.Div(id="annot-modal-status", style={"fontSize": "11px", "color": core.MUTED, "flex": "1"}),
            html.Button(
                "Run Annotation",
                id="annot-btn",
                n_clicks=0,
                style={
                    "padding": "7px 20px", "backgroundColor": "#238636", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="annot-modal-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="annot-modal", is_open=False, size="md",
       style={"color": core.TEXT},
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
       backdrop=True),

# ── SpaGE Gene Imputation modal ───────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("SpaGE Gene Imputation"), close_button=True),
        dbc.ModalBody([
            html.Div(
                "Impute expression of genes not in the Xenium panel using a scRNA-seq reference (Seurat RDS). "
                "Results are cached and auto-loaded on next startup.",
                style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "14px", "lineHeight": "1.5"},
            ),
            ui.ctrl_label("Reference .rds File"),
            dcc.Input(
                id="spage-rds-path",
                type="text",
                value=core.REF_RDS_DEFAULT,
                    placeholder=core.REF_RDS_PLACEHOLDER,
                debounce=True,
                style={
                    "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                    "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                    "padding": "4px 8px", "fontSize": "11px", "marginBottom": "12px",
                    "boxSizing": "border-box",
                },
            ),
            ui.ctrl_label("Principal Vectors (n_pv)"),
            dcc.Slider(
                id="spage-npv", min=10, max=100, step=5, value=50,
                marks={10: "10", 50: "50", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(style={"marginBottom": "12px"}),
            ui.ctrl_label("Genes to Impute"),
            html.Div(
                "Leave empty for the 200 most variable reference genes outside the panel. "
                "Separate gene names with commas or newlines. "
                "Imputation always uses the original counts (not SPLIT-corrected ones).",
                style={"fontSize": "10px", "color": core.MUTED, "marginBottom": "6px", "lineHeight": "1.4"},
            ),
            dcc.Textarea(
                id="spage-genes",
                value=core._impute_genes_default(),
                placeholder="e.g. NPPA, MYH7, TNNT2 (default: the genes in the file $XENIUM_IMPUTE_GENES)",
                style={
                    "width": "100%", "height": "100px", "backgroundColor": core.CARD_BG,
                    "color": core.TEXT, "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                    "padding": "6px 8px", "fontSize": "11px", "marginBottom": "4px",
                    "boxSizing": "border-box", "resize": "vertical",
                },
            ),
        ]),
        dbc.ModalFooter([
            html.Div(id="spage-modal-status",
                     style={"fontSize": "11px", "color": core.MUTED, "flex": "1"}),
            html.Button(
                "Run SpaGE",
                id="spage-run-btn",
                n_clicks=0,
                style={
                    "padding": "7px 20px",
                    "backgroundColor": "#6f42c1", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="spage-modal-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="spage-modal", is_open=False, size="lg",
       style={"color": core.TEXT},
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
       backdrop=True),

# ─── CellNEST modal ───────────────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Load CellNEST Results"), close_button=True),
    dbc.ModalBody([
        html.Div("Point to a CellNEST output folder (the root containing output/, metadata/ subdirs).",
                 style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "14px", "lineHeight": "1.5"}),
        ui.ctrl_label("CellNEST Output Folder"),
        dcc.Input(id="cellnest-folder-path", type="text", debounce=True,
                  placeholder="/path/to/cellnest_output",
                  style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                         "padding": "4px 8px", "fontSize": "11px", "marginBottom": "12px",
                         "boxSizing": "border-box"}),
        ui.ctrl_label("Dataset Name"),
        dcc.Dropdown(id="cellnest-dataset-dropdown", options=[], value=None,
                     style={"fontSize": "11px", "marginBottom": "12px"}),
        ui.ctrl_label("Strip suffix from cell names"),
        html.Div([
            dcc.Input(id="cellnest-strip-suffix", type="text", debounce=True,
                      placeholder="e.g. _4  (leave blank to use barcode as-is)",
                      style={"flex": "1", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                             "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                             "padding": "4px 8px", "fontSize": "11px",
                             "boxSizing": "border-box"}),
        ], style={"display": "flex", "marginBottom": "12px"}),
        html.Div(id="cellnest-modal-status",
                 style={"fontSize": "11px", "color": core.MUTED, "minHeight": "16px"}),
    ]),
    dbc.ModalFooter([
        html.Button("Detect Datasets", id="cellnest-detect-btn", n_clicks=0,
                    style={"padding": "6px 14px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                           "border": f"1px solid {core.BORDER}", "borderRadius": "5px",
                           "cursor": "pointer", "fontSize": "12px", "marginRight": "auto"}),
        html.Button("Load", id="cellnest-load-btn", n_clicks=0,
                    style={"padding": "7px 20px", "backgroundColor": "#0d9488", "color": "#fff",
                           "border": "none", "borderRadius": "5px",
                           "cursor": "pointer", "fontSize": "13px", "fontWeight": "600"}),
        dbc.Button("Close", id="cellnest-modal-close-btn", color="secondary", size="sm"),
    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
], id="cellnest-modal", is_open=False, size="lg",
   backdrop="static", style={"zIndex": 2000},
   content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT}),

# ─── Spatial Niches modal ─────────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("Spatial Analysis Tools"),
    dbc.ModalBody([
        html.Div("Cluster cells by neighborhood cell-type composition (BuildNicheAssay-style).",
                 style={"fontSize": "0.85rem", "color": core.MUTED, "marginBottom": "14px"}),
        dbc.Row([
            dbc.Col([
                html.Label("N Clusters", style={"fontSize": "0.82rem", "color": core.MUTED}),
                dcc.Input(id="niche-n-clusters", type="number", value=10, min=2, max=50,
                          style={"width": "100%", "backgroundColor": core.CARD_BG,
                                 "color": core.TEXT, "border": f"1px solid {core.BORDER}",
                                 "borderRadius": "4px", "padding": "4px 8px"}),
            ], width=6),
            dbc.Col([
                html.Label("N Neighbors", style={"fontSize": "0.82rem", "color": core.MUTED}),
                dcc.Input(id="niche-n-neighbors", type="number", value=20, min=5, max=200,
                          style={"width": "100%", "backgroundColor": core.CARD_BG,
                                 "color": core.TEXT, "border": f"1px solid {core.BORDER}",
                                 "borderRadius": "4px", "padding": "4px 8px"}),
            ], width=6),
        ], style={"marginBottom": "12px"}),
        html.Label("Cell Type Method", style={"fontSize": "0.82rem", "color": core.MUTED}),
        dcc.Dropdown(id="niche-label-method", options=[], value=None, clearable=True,
                     placeholder="None (spatial coordinates only)",
                     style={"fontSize": "12px", "marginBottom": "14px",
                            "backgroundColor": core.CARD_BG, "color": core.TEXT}),
        dbc.Button("Run", id="niche-run-btn", color="primary",
                   style={"width": "100%", "marginBottom": "8px"}),
        html.Div(id="niche-modal-status",
                 style={"fontSize": "0.82rem", "color": core.MUTED, "minHeight": "18px"}),
        html.Hr(style={"borderColor": core.BORDER, "margin": "16px 0"}),
        html.Div("Spatial Analysis Plots", style={"fontWeight": "600", "marginBottom": "6px", "color": core.TEXT}),
        html.Div("Generates co-occurrence heatmap, neighborhood enrichment, interaction graph, and chord diagram.",
                 style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "10px"}),
        dbc.Button("Generate Analysis Plots", id="spatial-run-btn", color="primary", size="sm",
                   style={"marginBottom": "8px"}),
        html.Div(id="spatial-modal-status", style={"fontSize": "11px", "color": core.MUTED}),
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="niche-modal-close-btn", outline=True,
                   style={"color": core.TEXT, "borderColor": core.BORDER}),
    ),
], id="niche-modal", is_open=False, size="md",
   style={"color": core.TEXT},
   content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
   backdrop=True),

# ─── ROI Save modal ───────────────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("Save ROI"),
    dbc.ModalBody([
        html.Div(id="roi-save-preview",
                 style={"fontSize": "0.85rem", "color": core.MUTED, "marginBottom": "10px"}),
        dbc.Label("ROI Name", style={"fontSize": "0.82rem"}),
        dbc.Input(id="roi-name-input", placeholder="e.g. Tumor_1",
                  style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "border": f"1px solid {core.BORDER}", "marginBottom": "8px"}),
        dbc.Label("ROI Class", style={"fontSize": "0.82rem"}),
        dbc.Input(id="roi-class-input", placeholder="e.g. region", value="region",
                  style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "border": f"1px solid {core.BORDER}"}),
        html.Div(id="roi-save-error",
                 style={"color": "#e74c3c", "fontSize": "0.82rem", "marginTop": "8px"}),
    ]),
    dbc.ModalFooter([
        dbc.Button("Save", id="roi-save-btn", color="primary", size="sm"),
        dbc.Button("Cancel", id="roi-save-cancel-btn", color="secondary", size="sm", className="ms-2"),
    ]),
], id="roi-save-modal", is_open=False,
   style={"color": core.TEXT},
   content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
   backdrop=True),

# ─── ROI Manager modal ────────────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("Manage ROIs"),
    dbc.ModalBody([
        html.Div(id="roi-list-div", style={"marginBottom": "18px"}),
        html.Hr(style={"borderColor": core.BORDER}),
        html.Div([
            html.P("ROI Set Operations", style={"fontWeight": "600", "fontSize": "0.9rem", "marginBottom": "8px"}),
            html.Div([
                dcc.Dropdown(id="roi-op-a", placeholder="ROI A",
                             style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "fontSize": "0.82rem", "flex": "1"}),
                dcc.Dropdown(id="roi-op", options=[
                    {"label": "Union", "value": "union"},
                    {"label": "Intersection", "value": "intersection"},
                    {"label": "Subtract A−B", "value": "subtract"},
                ], placeholder="Operation",
                    style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "fontSize": "0.82rem", "width": "130px"}),
                dcc.Dropdown(id="roi-op-b", placeholder="ROI B",
                             style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "fontSize": "0.82rem", "flex": "1"}),
            ], style={"display": "flex", "gap": "8px", "marginBottom": "8px", "alignItems": "center"}),
            html.Div([
                dbc.Input(id="roi-op-name", placeholder="Result name",
                          style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "border": f"1px solid {core.BORDER}", "flex": "1"}),
                dbc.Input(id="roi-op-class", placeholder="Result class", value="region",
                          style={"backgroundColor": core.CARD_BG, "color": core.TEXT, "border": f"1px solid {core.BORDER}", "flex": "1"}),
                dbc.Button("Apply", id="roi-op-btn", color="primary", size="sm"),
            ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
            html.Div(id="roi-op-error",
                     style={"color": "#e74c3c", "fontSize": "0.82rem", "marginTop": "6px"}),
        ]),
    ]),
    dbc.ModalFooter([
        dbc.Button("Close", id="roi-manage-close-btn", color="secondary", size="sm"),
    ]),
], id="roi-manage-modal", is_open=False, size="lg",
   style={"color": core.TEXT},
   content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
   backdrop=True),

# ─── SPLIT correction modal ───────────────────────────────────────────────────
dbc.Modal([
    dbc.ModalHeader("SPLIT Ambient RNA Correction"),
    dbc.ModalBody([
        html.P("Requires R packages: spacexr (dmcable/spacexr) and SPLIT (bdsc-tds/SPLIT).",
               style={"fontSize":"0.82rem","color":"#8b949e"}),
        html.P("⚠ Peak RAM usage ~21 GB for a full Xenium slide.",
               style={"fontSize":"0.82rem","color":"#e85d04"}),
        dbc.Label("Seurat Reference RDS path"),
        dbc.Input(id="split-rds-path", value=core.REF_RDS_DEFAULT, placeholder=core.REF_RDS_PLACEHOLDER,
                  type="text", style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Cell-type label column"),
        dbc.Input(id="split-label-col", value="Names",
                  type="text", style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Max R cores"),
        dbc.Input(id="split-max-cores", value=4, type="number",
                  min=1, max=32, style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Min UMI (cells below this are excluded from RCTD)"),
        dbc.Input(id="split-min-umi", value=10, type="number",
                  min=1, max=500, style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Label("Min UMI sigma (UMI_min_sigma)"),
        dbc.Input(id="split-min-umi-sigma", value=100, type="number",
                  min=1, max=1000, style={"fontSize":"0.82rem"}),
        html.Br(),
        dbc.Checklist(
            id="split-purify-singlets",
            options=[{"label": " Purify singlets (DO_purify_singlets)", "value": "yes"}],
            value=["yes"],
            style={"fontSize":"0.82rem"},
        ),
        html.P(
            "When enabled, ambient RNA is also removed from singlet cells. "
            "Disable if corrected expression looks too aggressively reduced.",
            style={"fontSize":"0.75rem","color":"#8b949e","marginTop":"2px"},
        ),
        html.Br(),
        dbc.Button("▶ Run SPLIT", id="split-run-btn", color="danger",
                   style={"width":"100%"}),
        html.Div(id="split-modal-status",
                 style={"marginTop":"8px","fontSize":"0.82rem","color":"#8b949e"}),
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="split-modal-close-btn", color="secondary", size="sm")
    ),
], id="split-modal", is_open=False, size="lg"),

# ── Load Sample modal ─────────────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Load Xenium Sample"), close_button=True),
        dbc.ModalBody([
            html.Div("Select an available sample or enter a custom path:",
                     style={"fontSize": "12px", "color": core.MUTED, "marginBottom": "10px"}),
            dcc.Dropdown(
                id="sample-picker",
                options=sdata_store._available_samples(),
                placeholder="Available samples\u2026",
                style={"marginBottom": "10px", "backgroundColor": core.CARD_BG, "color": core.TEXT},
            ),
            html.Div("Or enter a custom path:", style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "4px"}),
            dcc.Input(
                id="sample-custom-path",
                type="text",
                placeholder="/path/to/output-XETG\u2026",
                style={"width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                       "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                       "padding": "6px 10px", "fontSize": "12px", "boxSizing": "border-box"},
            ),
            html.Div(id="sample-load-status",
                     style={"fontSize": "11px", "color": core.MUTED, "marginTop": "8px", "minHeight": "16px"}),
        ]),
        dbc.ModalFooter([
            html.Button("Load (Replace)", id="sample-load-btn", n_clicks=0,
                        style={"padding": "7px 20px", "backgroundColor": "#1f6feb", "color": "#fff",
                               "border": "none", "borderRadius": "5px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600"}),
            html.Button("Add as Additional Sample", id="sample-add-btn", n_clicks=0,
                        style={"padding": "7px 20px", "backgroundColor": "#238636", "color": "#fff",
                               "border": "none", "borderRadius": "5px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600", "marginLeft": "8px"}),
            dbc.Button("Cancel", id="sample-modal-close-btn", color="secondary", size="sm"),
        ]),
    ], id="sample-modal", is_open=False, size="lg",
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT}),

# ── Delete segmentation run confirmation modal ────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Delete Segmentation Run"), close_button=True),
        dbc.ModalBody(
            html.Div(id="seg-delete-confirm-text",
                     style={"fontSize": "13px", "color": core.TEXT, "lineHeight": "1.5"}),
        ),
        dbc.ModalFooter([
            html.Button(
                "Delete", id="seg-delete-confirm-btn", n_clicks=0,
                style={
                    "padding": "7px 20px", "backgroundColor": "#da3633", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Cancel", id="seg-delete-cancel-btn", color="secondary", size="sm"),
        ]),
    ], id="seg-delete-modal", is_open=False,
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT}),

# ── Clean Cache confirmation modal ────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Clean Cache"), close_button=True),
        dbc.ModalBody([
            html.Div(id="cache-clean-confirm-text",
                     style={"fontSize": "13px", "color": core.TEXT}),
        ]),
        dbc.ModalFooter([
            html.Button("Delete", id="cache-clean-confirm-btn", n_clicks=0,
                        style={"padding": "7px 20px", "backgroundColor": "#da3633", "color": "#fff",
                               "border": "none", "borderRadius": "5px", "cursor": "pointer",
                               "fontSize": "13px", "fontWeight": "600"}),
            dbc.Button("Cancel", id="cache-clean-cancel-btn", color="secondary", size="sm"),
        ]),
    ], id="cache-clean-modal", is_open=False,
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT}),

# ── Save as SpatialData modal ──────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Save as SpatialData"), close_button=True),
        dbc.ModalBody([
            html.Div(
                "Export the current dataset (transcripts, images, shapes) plus any analysis results "
                "(cluster labels, cell-type annotations) to a SpatialData Zarr store.",
                style={"fontSize": "11px", "color": core.MUTED, "marginBottom": "14px", "lineHeight": "1.5"},
            ),
            ui.ctrl_label("Output Directory"),
            dcc.Input(
                id="save-sdata-dir",
                type="text",
                placeholder="/path/to/output/directory",
                debounce=True,
                style={
                    "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                    "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                    "padding": "5px 8px", "fontSize": "11px", "marginBottom": "10px",
                    "boxSizing": "border-box",
                },
            ),
            ui.ctrl_label("Filename (folder name for the .zarr store)"),
            dcc.Input(
                id="save-sdata-name",
                type="text",
                placeholder="my_analysis.zarr",
                debounce=True,
                style={
                    "width": "100%", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                    "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                    "padding": "5px 8px", "fontSize": "11px", "marginBottom": "10px",
                    "boxSizing": "border-box",
                },
            ),
            dcc.Checklist(
                id="save-sdata-roi-only",
                options=[{"label": "  Export only cells within ROIs (if ROIs are defined)", "value": "roi_only"}],
                value=[],
                style={"fontSize": "11px", "color": core.TEXT, "marginBottom": "8px"},
            ),
            html.Div(
                "Note: Exports the active segmentation. For Xenium, SpatialData must be built first "
                "(open 'Resegment Cells' → patches auto-build). Existing stores at the same path will be overwritten.",
                style={"fontSize": "10px", "color": core.MUTED, "lineHeight": "1.4"},
            ),
            dcc.Interval(id="save-sdata-poll", interval=1500, disabled=True),
        ]),
        dbc.ModalFooter([
            html.Div(id="save-sdata-status",
                     style={"fontSize": "11px", "color": core.MUTED, "flex": "1"}),
            html.Button(
                "Save",
                id="save-sdata-run-btn",
                n_clicks=0,
                style={
                    "padding": "7px 20px",
                    "backgroundColor": "#1f6feb", "color": "#fff",
                    "border": "none", "borderRadius": "5px",
                    "cursor": "pointer", "fontSize": "13px", "fontWeight": "600",
                },
            ),
            dbc.Button("Close", id="save-sdata-close-btn", color="secondary", size="sm"),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
    ], id="save-sdata-modal", is_open=False, size="lg",
       style={"color": core.TEXT},
       content_style={"backgroundColor": core.DARK_BG, "color": core.TEXT},
       backdrop=True),

    html.Div([
        # ── Hover-reveal sidebar ──────────────────────────────────────────────
        # Outer zone covers the trigger strip + sidebar; hover on either reveals it
        html.Div([
            html.Div(
                sidebar,
                id="sidebar-wrapper",
            ),
        ], id="sidebar-hover-zone", style={
            "position": "fixed", "top": "0", "left": "0",
            "height": "100vh", "zIndex": "100",
            # Only the narrow trigger strip — sidebar stays open via #sidebar-wrapper:hover
            "width": "8px",
        }),

        html.Div([
            # Plots row
            html.Div([
                html.Div([
                    dcc.Graph(
                        id="spatial-plot",
                        responsive=True,
                        figure={
                            "data": [],
                            "layout": {
                                "paper_bgcolor": core.PLOT_BG,
                                "plot_bgcolor": core.PLOT_BG,
                                "xaxis": {"visible": False},
                                "yaxis": {"visible": False},
                                "annotations": [{
                                    "text": "Loading…",
                                    "x": 0.5, "y": 0.5,
                                    "xref": "paper", "yref": "paper",
                                    "showarrow": False,
                                    "font": {"size": 18, "color": "#8b949e"},
                                }],
                                "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
                            },
                        },
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d"],
                            "toImageButtonOptions": {"format": "svg", "filename": "xenium_spatial"},
                        },
                        style={"height": "100%"},
                    ),
                    # WebGL (deck.gl) view: shown instead of the Plotly graph when selected
                    html.Div(id="webgl-view", style={"position": "absolute", "inset": "0",
                                                     "display": "none"}),
                    html.Div([
                        dcc.RadioItems(
                            id="renderer-mode",
                            options=[{"label": " Plotly", "value": "plotly"},
                                     {"label": " WebGL (beta)", "value": "webgl"}],
                            value=_DEFAULT_RENDERER,
                            inline=True,
                            inputStyle={"marginRight": "3px"},
                            labelStyle={"marginRight": "10px", "fontSize": "11px",
                                        "color": core.TEXT, "cursor": "pointer"},
                        ),
                        dcc.RadioItems(
                            id="webgl-tool",
                            options=[{"label": " Pan", "value": "pan"},
                                     {"label": " Lasso", "value": "lasso"},
                                     {"label": " Box", "value": "rect"}],
                            value="pan", inline=True,
                            inputStyle={"marginRight": "3px"},
                            labelStyle={"marginRight": "8px", "fontSize": "11px",
                                        "color": core.TEXT, "cursor": "pointer"},
                            style={"borderLeft": f"1px solid {core.BORDER}", "paddingLeft": "10px"},
                        ),
                        # shown in WebGL mode only (assets/xenium_viewer_dash.js)
                        html.Button("PNG", id="webgl-png-btn", n_clicks=0,
                                    title="Download the current WebGL view as a PNG at 2x resolution, "
                                          "with legend and scale bar (for vector SVG use the Plotly "
                                          "renderer's camera button)",
                                    style={"padding": "1px 7px", "fontSize": "11px",
                                           "marginRight": "8px", "backgroundColor": core.CARD_BG,
                                           "color": core.TEXT, "border": f"1px solid {core.BORDER}",
                                           "borderRadius": "4px", "cursor": "pointer"}),
                        html.Span(id="webgl-status", style={"fontSize": "11px", "color": core.MUTED}),
                    ], style={"position": "absolute", "bottom": "10px", "left": "10px",
                              "zIndex": "10", "display": "flex", "alignItems": "center",
                              "backgroundColor": "rgba(13,17,23,0.85)", "padding": "3px 8px",
                              "borderRadius": "4px"}),
                    html.Div([
                        html.Button("📏", id="measure-btn", n_clicks=0, title="Measure distance",
                                    style={"padding": "4px 7px", "fontSize": "14px",
                                           "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                           "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                           "cursor": "pointer", "lineHeight": "1"}),
                        html.Span(id="measure-dist-display",
                                  style={"fontSize": "11px", "color": core.TEXT,
                                         "backgroundColor": "rgba(13,17,23,0.85)",
                                         "padding": "3px 6px", "borderRadius": "4px",
                                         "marginLeft": "4px", "display": "none"}),
                    ], style={"position": "absolute", "bottom": "10px", "right": "10px",
                              "zIndex": "10", "display": "flex", "alignItems": "center",
                              "pointerEvents": "auto"}),
                ], id="spatial-panel", style={"flex": "2", "minWidth": "200px",
                                              "position": "relative"}),

                html.Div(id="panel-resize-handle", style={
                    "width": "6px", "cursor": "col-resize", "flexShrink": "0",
                    "alignSelf": "stretch", "backgroundColor": "transparent",
                    "borderRadius": "3px", "transition": "background-color 0.15s",
                }),

                html.Div([
                    dbc.Tabs(id="plots-tabs", active_tab="tab-umap",
                             style={"borderBottom": f"1px solid {core.BORDER}", "marginBottom": "6px"},
                             children=[
                        dbc.Tab(label="UMAP", tab_id="tab-umap",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    dcc.Graph(id="umap-plot", responsive=True,
                                              config={"displayModeBar": True,
                                                      "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                                      "toImageButtonOptions": {"format": "svg", "filename": "xenium_umap"}},
                                              style={"flex": "1", "minHeight": "0"}),
                                    html.Div(id="webgl-umap", style={"flex": "1", "minHeight": "0",
                                                                     "position": "relative",
                                                                     "display": "none"}),
                                    ui._plot_dim_controls("umap"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        dbc.Tab(label="Cell Type Abundance", tab_id="tab-abundance",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="cell-type-abundance-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_abundance"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="abundance-plot-container",
                                             style={"flex": "1", "minHeight": "240px"}),
                                    ui._plot_dim_controls("abundance"),
                                    stats._composition_section(),
                                    segfree._sfree_section(),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0",
                                          "overflowY": "auto"})]),
                        dbc.Tab(label="Co-occurrence", tab_id="tab-cooccurrence",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="cooccurrence-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_cooccurrence"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="cooccurrence-plot-container",
                                             style={"flex": "1", "minHeight": "0"}),
                                    ui._plot_range_and_dim_controls("cooccurrence"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        dbc.Tab(label="Neighborhood Enrichment", tab_id="tab-enrichment",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="enrichment-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_enrichment"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="enrichment-plot-container",
                                             style={"flex": "1", "minHeight": "0"}),
                                    ui._plot_range_and_dim_controls("enrichment"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        dbc.Tab(label="Interaction Graph", tab_id="tab-interaction",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="interaction-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_interaction"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="interaction-plot-container",
                                             style={"flex": "1", "minHeight": "0"}),
                                    ui._plot_normalize_and_dim_controls("interaction"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        dbc.Tab(label="Chord Diagram", tab_id="tab-chord",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="chord-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_chord"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="chord-plot-container",
                                             style={"flex": "1", "minHeight": "0"}),
                                    ui._plot_normalize_and_dim_controls("chord"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        dbc.Tab(label="Niche UMAP", tab_id="tab-niche-umap",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="niche-umap-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_niche_umap"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="niche-umap-plot-container",
                                             style={"flex": "1", "minHeight": "0"}),
                                    ui._plot_dim_controls("niche-umap"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        dbc.Tab(label="CellNEST CCC", tab_id="tab-cellnest",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    # Row 1: LR pair, top-n, color-by
                                    html.Div([
                                        html.Span("LR Pair:", style={"fontSize": "11px", "color": core.MUTED, "marginRight": "6px"}),
                                        dcc.Dropdown(
                                            id="cellnest-lr-pair",
                                            options=[], value=None, clearable=True,
                                            multi=True,
                                            style={"flex": "1", "fontSize": "11px"},
                                            placeholder="Select one or more LR pairs…",
                                        ),
                                        html.Span("Top N:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 6px"}),
                                        dcc.Input(id="cellnest-top-n", type="number", value=100, min=1, max=5000,
                                                  debounce=True,
                                                  style={"width": "70px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                                         "padding": "3px 6px", "fontSize": "11px"}),
                                        html.Span("Edge W:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 6px"}),
                                        dcc.Input(id="cellnest-edge-width", type="number", value=1, min=0.1, max=20,
                                                  step=0.1, debounce=True,
                                                  style={"width": "60px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                                         "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                                         "padding": "3px 6px", "fontSize": "11px"}),
                                        html.Span("Color By:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 6px"}),
                                        dcc.Dropdown(
                                            id="cellnest-color-by",
                                            options=[
                                                {"label": "LR Pair",         "value": "lr_pair"},
                                                {"label": "Ligand",          "value": "ligand"},
                                                {"label": "Receptor",        "value": "receptor"},
                                                {"label": "Attention Score", "value": "attention"},
                                                {"label": "Sender Cell Type", "value": "sender_type"},
                                            ],
                                            value="lr_pair", clearable=False,
                                            style={"width": "160px", "fontSize": "11px"},
                                        ),
                                    ], style={"display": "flex", "alignItems": "center", "padding": "6px 8px",
                                              "borderBottom": f"1px solid {core.BORDER}", "flexShrink": "0"}),
                                    # Row 2: filters
                                    html.Div([
                                        html.Span("Annotation:", style={"fontSize": "11px", "color": core.MUTED, "marginRight": "4px"}),
                                        dcc.Dropdown(id="cellnest-annot-col", options=[], value=None, clearable=True,
                                                     placeholder="(none)",
                                                     style={"width": "150px", "fontSize": "11px"}),
                                        html.Span("Ligand:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 4px 0 10px"}),
                                        dcc.Dropdown(id="cellnest-filter-ligand", options=[], value=None, clearable=True,
                                                     multi=True, placeholder="All",
                                                     style={"width": "130px", "fontSize": "11px"}),
                                        html.Span("Receptor:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 4px 0 10px"}),
                                        dcc.Dropdown(id="cellnest-filter-receptor", options=[], value=None, clearable=True,
                                                     multi=True, placeholder="All",
                                                     style={"width": "130px", "fontSize": "11px"}),
                                        html.Span("Sender:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 4px 0 10px"}),
                                        dcc.Dropdown(id="cellnest-filter-sender", options=[], value=None, clearable=True,
                                                     multi=True, placeholder="All",
                                                     style={"width": "130px", "fontSize": "11px"}),
                                        html.Span("Receiver:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 4px 0 10px"}),
                                        dcc.Dropdown(id="cellnest-filter-receiver", options=[], value=None, clearable=True,
                                                     multi=True, placeholder="All",
                                                     style={"width": "130px", "fontSize": "11px"}),
                                    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                                              "padding": "4px 8px", "borderBottom": f"1px solid {core.BORDER}",
                                              "flexShrink": "0", "gap": "2px"}),
                                    # Row 3: optional feature toggles
                                    html.Div([
                                        html.Span("Options:", style={"fontSize": "11px", "color": core.MUTED, "marginRight": "8px"}),
                                        dcc.Checklist(
                                            id="cellnest-options",
                                            options=[
                                                {"label": " Scale edge width by attention score", "value": "edge_weight"},
                                                {"label": " Highlight autocrine (self-loops)",    "value": "autocrine"},
                                                {"label": " Color background by interaction degree", "value": "node_degree"},
                                                {"label": " Gray out non-participating cells on map", "value": "gray_bg"},
                                            ],
                                            value=[],
                                            inline=True,
                                            labelStyle={"fontSize": "11px", "color": core.TEXT, "marginRight": "14px",
                                                        "cursor": "pointer"},
                                            style={"display": "inline-flex", "flexWrap": "wrap"},
                                        ),
                                    ], style={"display": "flex", "alignItems": "center", "padding": "4px 8px",
                                              "borderBottom": f"1px solid {core.BORDER}", "flexShrink": "0"}),
                                    html.Div([
                                        dcc.Graph(id="cellnest-bar-plot", responsive=True,
                                                  config={"displayModeBar": "hover", "modeBarButtons": [["toImage"]], "toImageButtonOptions": {"format": "svg", "filename": "xenium_cellnest_bar"}},
                                                  style={"flex": "1", "minWidth": "200px"}),
                                        dcc.Graph(id="cellnest-spatial-plot", responsive=True,
                                                  config={"displayModeBar": "hover", "modeBarButtons": [["toImage"]], "toImageButtonOptions": {"format": "svg", "filename": "xenium_cellnest_chord"}},
                                                  style={"flex": "2", "minWidth": "200px"}),
                                    ], id="cellnest-plots-container",
                                       style={"display": "flex", "flex": "1", "minHeight": "0", "gap": "6px"}),
                                    ui._plot_dim_controls("cellnest"),
                                    # ── NicheNet-style analysis panel ────────
                                    html.Details([
                                        html.Summary("▶ NicheNet-style Analysis",
                                                     style={"fontSize": "11px", "color": core.MUTED,
                                                            "cursor": "pointer", "padding": "4px 8px",
                                                            "borderTop": f"1px solid {core.BORDER}"}),
                                        html.Div([
                                            html.Div([
                                                html.Span("Top N ligands:", style={"fontSize": "11px", "color": core.MUTED, "marginRight": "6px"}),
                                                dcc.Input(id="cellnest-nichenest-top-n", type="number", value=20, min=5, max=100,
                                                          debounce=True,
                                                          style={"width": "60px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                                                 "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                                                 "padding": "3px 6px", "fontSize": "11px"}),
                                                html.Span("(Annotation column controls heatmap & dotplot)", style={"fontSize": "10px", "color": core.MUTED, "marginLeft": "10px"}),
                                            ], style={"display": "flex", "alignItems": "center", "padding": "4px 8px", "flexShrink": "0"}),
                                            html.Div([
                                                dcc.Graph(id="cellnest-lr-heatmap", responsive=True,
                                                          config={"displayModeBar": "hover", "modeBarButtons": [["toImage"]], "toImageButtonOptions": {"format": "svg", "filename": "xenium_cellnest_lr_heatmap"}},
                                                          style={"flex": "1", "minWidth": "200px", "height": "320px"}),
                                                dcc.Graph(id="cellnest-ligtarget-heatmap", responsive=True,
                                                          config={"displayModeBar": "hover", "modeBarButtons": [["toImage"]], "toImageButtonOptions": {"format": "svg", "filename": "xenium_cellnest_ligtarget_heatmap"}},
                                                          style={"flex": "1", "minWidth": "200px", "height": "320px"}),
                                                dcc.Graph(id="cellnest-dotplot", responsive=True,
                                                          config={"displayModeBar": "hover", "modeBarButtons": [["toImage"]], "toImageButtonOptions": {"format": "svg", "filename": "xenium_cellnest_dotplot"}},
                                                          style={"flex": "1", "minWidth": "200px", "height": "320px"}),
                                            ], style={"display": "flex", "gap": "6px", "padding": "0 8px 8px"}),
                                        ], style={"backgroundColor": core.CARD_BG}),
                                    ], style={"flexShrink": "0"}),
                                    # ── Downstream TF panel ─────────────────
                                    html.Details([
                                        html.Summary("▶ Downstream TF Analysis",
                                                     style={"fontSize": "11px", "color": core.MUTED,
                                                            "cursor": "pointer", "padding": "4px 8px",
                                                            "borderTop": f"1px solid {core.BORDER}"}),
                                        html.Div([
                                            html.Div([
                                                html.Span("Receptor:", style={"fontSize": "11px", "color": core.MUTED, "marginRight": "6px"}),
                                                dcc.Input(id="cellnest-downstream-receptor", type="text", debounce=True,
                                                          placeholder="e.g. CCR7",
                                                          style={"width": "110px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                                                 "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                                                 "padding": "3px 6px", "fontSize": "11px"}),
                                                html.Span("Hops:", style={"fontSize": "11px", "color": core.MUTED, "margin": "0 6px"}),
                                                dcc.Input(id="cellnest-downstream-hops", type="number", value=2, min=1, max=4,
                                                          style={"width": "50px", "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                                                 "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                                                 "padding": "3px 6px", "fontSize": "11px"}),
                                                html.Button("Find TFs", id="cellnest-downstream-btn", n_clicks=0,
                                                            style={"marginLeft": "8px", "padding": "4px 12px",
                                                                   "backgroundColor": "#0d9488", "color": "#fff",
                                                                   "border": "none", "borderRadius": "4px",
                                                                   "cursor": "pointer", "fontSize": "11px"}),
                                                html.Div(id="cellnest-downstream-status",
                                                         style={"fontSize": "11px", "color": core.MUTED, "marginLeft": "10px"}),
                                            ], style={"display": "flex", "alignItems": "center",
                                                      "padding": "6px 8px", "flexWrap": "wrap", "gap": "4px"}),
                                            html.Div([
                                                html.Div([
                                                    html.Span("TFs in panel:", style={"fontSize": "11px", "color": core.MUTED, "marginRight": "6px"}),
                                                    dcc.Dropdown(id="cellnest-downstream-tf-select", options=[], value=None,
                                                                 multi=True, placeholder="Select TF(s) to plot…",
                                                                 style={"flex": "1", "fontSize": "11px"}),
                                                ], style={"display": "flex", "alignItems": "center", "padding": "4px 8px"}),
                                                dcc.Graph(id="cellnest-downstream-plot", responsive=True,
                                                          config={"displayModeBar": "hover", "modeBarButtons": [["toImage"]], "toImageButtonOptions": {"format": "svg", "filename": "xenium_cellnest_downstream"}},
                                                          style={"height": "220px"}),
                                            ], id="cellnest-downstream-results",
                                               style={"display": "none"}),
                                        ], style={"backgroundColor": core.CARD_BG}),
                                    ], style={"flexShrink": "0", "overflowY": "auto", "maxHeight": "260px"}),
                                ], style={"display": "flex", "flexDirection": "column",
                                          "flex": "1", "minHeight": "0", "overflow": "hidden"})]),
                        dbc.Tab(label="ROI Area", tab_id="tab-roi-area",
                                label_style={"fontSize": "12px", "padding": "4px 12px"},
                                children=[html.Div([
                                    html.Div(dcc.Graph(id="roi-area-plot", responsive=True,
                                              config={"displayModeBar": "hover", "toImageButtonOptions": {"format": "svg", "filename": "xenium_roi_area"}},
                                              style={"width": "100%", "height": "100%"}),
                                             id="roi-area-plot-container",
                                             style={"flex": "1", "minHeight": "0"}),
                                    ui._plot_dim_controls("roi-area"),
                                ], style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"})]),
                        analysis._de_tab(stats._pb_controls(), proximity._PROX_DE_MODES,
                                         proximity._prox_controls()),
                        analysis._geneset_tab(),
                        analysis._density_tab(),
                        stats._moran_tab(),
                        stats._lr_tab(lrdecomp._lrd_controls(), lrdecomp._lrd_results()),
                        controls._controls_tab(),
                        controls._segqc_tab(),
                        controls._coloc_tab(),
                        aggregates._agg_tab(),
                        txfactor._txf_tab(),
                        banksy._bk_tab(),
                        joint._jc_tab(),
                    ]),
                ], id="umap-panel", style={"display": "none"}),
            ], id="plots-row", style={"display": "flex", "flex": "1", "gap": "10px", "minHeight": "0"}),

            html.Div(id="vertical-resize-handle", style={
                "height": "6px", "cursor": "row-resize", "flexShrink": "0",
                "backgroundColor": "transparent", "borderRadius": "3px",
                "transition": "background-color 0.15s",
            }),

            # ── Bottom info bar (collapsible) ───────────────────────────────
            html.Div([
                # Toggle arrow button
                html.Button(
                    "▼", id="info-bar-toggle", n_clicks=0,
                    title="Hide/show info bar",
                    style={
                        "position": "absolute", "top": "-14px", "left": "50%",
                        "transform": "translateX(-50%)",
                        "background": core.CARD_BG, "border": f"1px solid {core.BORDER}",
                        "borderRadius": "50%", "width": "24px", "height": "24px",
                        "cursor": "pointer", "fontSize": "10px", "color": core.MUTED,
                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                        "padding": "0", "lineHeight": "1",
                    },
                ),
                # Collapsible body
                html.Div(
                    id="info-bar-body",
                    style={"display": "flex", "gap": "10px", "height": "100%", "overflow": "hidden"},
                    children=[
                        # Left: cell info (scrollable)
                        html.Div(
                            id="cell-info-panel",
                            children=[html.Div("Click a cell to see details",
                                               style={"color": core.MUTED, "fontSize": "13px", "fontStyle": "italic"})],
                            style={"flex": "1", "minWidth": "0", "overflowY": "auto"},
                        ),
                        # Divider
                        html.Div(style={"width": "1px", "backgroundColor": core.BORDER, "flexShrink": "0"}),
                        # Right: server log + REPL input
                        html.Div([
                            html.Div("SERVER LOG", style={
                                "fontSize": "10px", "color": core.MUTED, "fontWeight": "600",
                                "letterSpacing": "1px", "marginBottom": "4px", "flexShrink": "0",
                            }),
                            html.Pre(
                                id="server-log",
                                style={
                                    "flex": "1", "margin": "0", "fontSize": "11px", "color": "#8b949e",
                                    "overflowY": "auto", "whiteSpace": "pre-wrap",
                                    "wordBreak": "break-all",
                                    "fontFamily": "'Fira Code','Consolas',monospace",
                                },
                            ),
                            # Python REPL input
                            html.Div([
                                html.Span(">>>", style={
                                    "color": core.ACCENT, "fontFamily": "'Fira Code','Consolas',monospace",
                                    "fontSize": "12px", "flexShrink": "0", "lineHeight": "28px",
                                }),
                                dcc.Input(
                                    id="repl-input",
                                    type="text",
                                    debounce=False,
                                    placeholder=("Python expression or statement (e.g. _run_seurat_annotation(...))"
                                                 if _REPL_ENABLED else
                                                 "REPL disabled (start with XENIUM_ENABLE_REPL=1 to enable)"),
                                    disabled=not _REPL_ENABLED,
                                    style={
                                        "flex": "1", "backgroundColor": "#0d1117",
                                        "color": core.TEXT, "border": "none", "outline": "none",
                                        "fontSize": "12px", "padding": "0 6px",
                                        "fontFamily": "'Fira Code','Consolas',monospace",
                                    },
                                    n_submit=0,
                                ),
                                html.Button("Run", id="repl-run", n_clicks=0, style={
                                    "padding": "2px 10px", "fontSize": "11px", "flexShrink": "0",
                                    "backgroundColor": core.CARD_BG, "color": core.TEXT,
                                    "border": f"1px solid {core.BORDER}", "borderRadius": "4px",
                                    "cursor": "pointer",
                                }),
                            ], style={
                                "display": "flex", "alignItems": "center", "gap": "6px",
                                "borderTop": f"1px solid {core.BORDER}", "paddingTop": "4px",
                                "flexShrink": "0",
                            }),
                            dcc.Interval(id="log-poll", interval=1500, n_intervals=0),
                        ], style={
                            "flex": "1", "minWidth": "0",
                            "display": "flex", "flexDirection": "column", "overflow": "hidden",
                        }),
                    ],
                ),
            ], id="info-bar", style={
                "position": "relative",
                "backgroundColor": core.CARD_BG, "border": f"1px solid {core.BORDER}",
                "borderRadius": "6px", "padding": "10px 16px",
                "height": "180px",  # fixed height — both columns scroll independently
                "display": "flex", "flexDirection": "column",
                "boxSizing": "border-box", "flexShrink": "0",
            }),
        ], style={
            "flex": "1", "display": "flex", "flexDirection": "column",
            "padding": "12px", "gap": "10px",
            "minWidth": "0", "height": "100vh", "boxSizing": "border-box",
        }),
    ], style={
        "display": "flex", "height": "100vh",
        "backgroundColor": core.DARK_BG, "color": core.TEXT,
        "fontFamily": "'Inter','Segoe UI',sans-serif",
    }),
], style={"margin": "0", "padding": "0"})
