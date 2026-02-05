import os
import time
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import requests

import dash
from dash import dcc, html, Input, Output, State, dash_table


# ============================================================
# CONFIG
# ============================================================
CSV_FILE_PATH = "history_data.csv"

POINTS_PER_PAGE = 300
REFRESH_MS = 4000  # auto-advance speed

WIN = 50  # rolling window (samples) for anomaly score + motion window

# Risk thresholds on anomaly_score (tune)
TH_MODERATE = 3.0
TH_HIGH = 4.0
TH_VERY_HIGH = 5.0

# Open-Meteo (Pune by default)
WEATHER_LAT = 18.5204
WEATHER_LON = 73.8567
WEATHER_TTL_SEC = 600


# ============================================================
# BUTTON STYLES (INLINE)
# ============================================================
BTN_STYLE = {
    "fontFamily": '"Segoe UI","Inter",Arial,sans-serif',
    "fontSize": "13px",
    "fontWeight": "600",
    "padding": "8px 12px",
    "borderRadius": "8px",
    "border": "1px solid #d0d7de",
    "background": "#fff",
    "color": "#111827",
    "cursor": "pointer",
    "boxShadow": "0 1px 1px rgba(0,0,0,0.06)",
}

BTN_RESET = {
    **BTN_STYLE,
    "background": "#dc2626",
    "border": "1px solid #dc2626",
    "color": "white",
}

BTN_PAUSE = {
    **BTN_STYLE,
    "background": "#2563eb",
    "border": "1px solid #2563eb",
    "color": "white",
}

BTN_DOWNLOAD = {
    **BTN_STYLE,
    "background": "#16a34a",
    "border": "1px solid #16a34a",
    "color": "white",
}


# ============================================================
# WEATHER (Open-Meteo) + cache
# ============================================================
_WEATHER_CACHE = {"ts": 0.0, "data": None, "err": ""}


def get_openmeteo_current(lat=WEATHER_LAT, lon=WEATHER_LON):
    now = time.time()
    if _WEATHER_CACHE["data"] is not None and (now - _WEATHER_CACHE["ts"]) < WEATHER_TTL_SEC:
        return _WEATHER_CACHE["data"], _WEATHER_CACHE["err"]

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m"],
            "timezone": "Asia/Kolkata",
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        j = r.json()

        cur = j.get("current", {}) or {}
        data = {
            "time": cur.get("time"),
            "temp_c": cur.get("temperature_2m"),
            "rh": cur.get("relative_humidity_2m"),
        }
        _WEATHER_CACHE["ts"] = now
        _WEATHER_CACHE["data"] = data
        _WEATHER_CACHE["err"] = ""
        return data, ""
    except Exception as e:
        if _WEATHER_CACHE["data"] is not None:
            return _WEATHER_CACHE["data"], f"Weather error (stale cache): {e}"
        return {"time": None, "temp_c": None, "rh": None}, f"Weather error: {e}"


def gauge_temp(temp_c):
    v = float(temp_c) if temp_c is not None else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": " °C"},
        title={"text": "Temperature"},
        gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#d62728"}},
    ))
    fig.update_layout(height=220, margin=dict(l=25, r=25, t=55, b=10))
    return fig


def gauge_rh(rh):
    v = float(rh) if rh is not None else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": " %"},
        title={"text": "Humidity"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f77b4"}},
    ))
    fig.update_layout(height=220, margin=dict(l=25, r=25, t=55, b=10))
    return fig


# ============================================================
# CSV + anomaly scoring cache
# ============================================================
_CACHE = {"mtime": None, "scored": None, "error": ""}


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S%z", errors="coerce")
    if ts.isna().any():
        ts2 = pd.to_datetime(s, format="%d-%m-%Y %H:%M", errors="coerce")
        ts = ts.fillna(ts2)
    if ts.isna().any():
        ts3 = pd.to_datetime(s, errors="coerce")
        ts = ts.fillna(ts3)
    return ts


def load_csv_all(file_path: str) -> pd.DataFrame:
    usecols = ["timestamp", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    df = pd.read_csv(file_path, usecols=usecols, low_memory=False)

    df["timestamp"] = _parse_timestamp_series(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    sensor_cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    for c in sensor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=sensor_cols)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("After parsing/cleaning, dataframe is empty.")
    return df


def _zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - mu) / sd


def risk_level(score: float) -> str:
    if score >= TH_VERY_HIGH:
        return "very high"
    if score >= TH_HIGH:
        return "high"
    if score >= TH_MODERATE:
        return "moderate"
    return "low"


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    gyro_mag = np.sqrt(df["GyroX"] ** 2 + df["GyroY"] ** 2 + df["GyroZ"] ** 2)

    accz_rm = df["AccZ"].rolling(WIN, min_periods=max(10, WIN // 2)).mean().bfill().ffill()
    gyro_rm = gyro_mag.rolling(WIN, min_periods=max(10, WIN // 2)).mean().bfill().ffill()

    score = np.abs(_zscore(accz_rm)) + np.abs(_zscore(gyro_rm))

    out = df.copy()
    out["anomaly_score"] = score.astype(float)
    out["risk_level"] = out["anomaly_score"].apply(risk_level)
    return out


def detect_motion_window(w: pd.DataFrame) -> str:
    acc_mag = np.sqrt(w["AccX"] ** 2 + w["AccY"] ** 2 + w["AccZ"] ** 2)
    gyro_mag = np.sqrt(w["GyroX"] ** 2 + w["GyroY"] ** 2 + w["GyroZ"] ** 2)

    acc_std = float(acc_mag.std())
    gyro_rms = float(np.sqrt(np.mean(gyro_mag.to_numpy() ** 2)))

    if gyro_rms < 0.02 and acc_std < 0.05:
        return "still"
    elif gyro_rms < 0.08 and acc_std < 0.15:
        return "tilt"
    return "shake"


def get_scored_cached() -> tuple[pd.DataFrame | None, str]:
    try:
        mtime = os.path.getmtime(CSV_FILE_PATH)
    except Exception as e:
        return None, f"File error: {e}"

    if _CACHE["scored"] is not None and _CACHE["mtime"] == mtime:
        return _CACHE["scored"], _CACHE["error"]

    try:
        t0 = time.time()
        df = load_csv_all(CSV_FILE_PATH)
        scored = compute_scores(df)
        _CACHE["mtime"] = mtime
        _CACHE["scored"] = scored
        _CACHE["error"] = f"Loaded+scored in {time.time() - t0:.1f}s"
        return scored, _CACHE["error"]
    except Exception as e:
        _CACHE["mtime"] = mtime
        _CACHE["scored"] = None
        _CACHE["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return None, _CACHE["error"]


# ============================================================
# WINDOW: table + chart for current timeframe (300 points)
# ============================================================
def anomaly_score_figure(window_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=window_df["timestamp"],
        y=window_df["anomaly_score"],
        mode="lines",
        name="Anomaly score",
        line=dict(width=2),
    ))

    for risk, color in [("moderate", "goldenrod"), ("high", "orangered"), ("very high", "red")]:
        pts = window_df[window_df["risk_level"] == risk]
        if not pts.empty:
            fig.add_trace(go.Scatter(
                x=pts["timestamp"],
                y=pts["anomaly_score"],
                mode="markers",
                name=risk,
                marker=dict(color=color, size=9, symbol="diamond"),
            ))

    fig.add_hline(y=TH_MODERATE, line_dash="dash", line_color="goldenrod")
    fig.add_hline(y=TH_HIGH, line_dash="dash", line_color="orangered")
    fig.add_hline(y=TH_VERY_HIGH, line_dash="dash", line_color="red")

    # lock vertical frame
    y_max = TH_VERY_HIGH + 0.5
    fig.update_yaxes(range=[0, y_max], autorange=False)

    fig.update_layout(
        uirevision=str(window_df["timestamp"].iloc[0]),
        title="Anomaly score over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        hovermode="x unified",
        height=520,
        margin=dict(l=40, r=20, t=55, b=50),
    )
    return fig


def build_table_for_window(scored_all: pd.DataFrame, window_df: pd.DataFrame, max_rows: int = 250):
    flagged = window_df[window_df["risk_level"].isin(["moderate", "high", "very high"])].copy()
    if flagged.empty:
        return []

    rows = []
    for idx in flagged.index:
        j0 = max(0, idx - WIN + 1)
        w = scored_all.iloc[j0: idx + 1]
        motion = detect_motion_window(w)

        rows.append({
            "timestamp": str(scored_all.at[idx, "timestamp"]),
            "motion_label": motion,
            "anomaly_score": round(float(scored_all.at[idx, "anomaly_score"]), 3),
            "risk_level": scored_all.at[idx, "risk_level"],
        })

    rows = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:max_rows]
    return rows


# ============================================================
# DASH APP (layout like screenshot)
# ============================================================

app = dash.Dash(__name__)
app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%config%}
    <style>
      <style>
  html, body {
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, Arial, sans-serif !important;
  }
  * {
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, Arial, sans-serif !important;
  }
  /* Plotly text often lives inside SVG */
  svg text, .gtitle, .g-ylabel, .g-xlabel, .glegend {
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, Arial, sans-serif !important;
  }
  .dash-table-container * {
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, Arial, sans-serif !important;
  }
</style>
    </style>
    
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
'''




app.title = "SecTr SHM Dashboard"

STYLE_APP = {"maxWidth": "1400px", "margin": "0 auto", "padding": "10px"}
STYLE_ROW = {"display": "flex", "gap": "14px", "alignItems": "stretch"}
STYLE_LEFT = {"flex": "0 1 76%"}
STYLE_RIGHT = {"flex": "0 1 24%", "display": "flex", "flexDirection": "column", "gap": "10px"}
STYLE_CARD = {"border": "1px solid #e6e6e6", "borderRadius": "10px", "background": "white", "padding": "10px"}
STYLE_CENTER = {"textAlign": "center"}
STYLE_BTNROW = {"display": "flex", "gap": "10px", "justifyContent": "center", "flexWrap": "wrap", "marginTop": "8px"}

app.layout = html.Div(
    [
        html.Div(
    style={
        "display": "flex",
        "alignItems": "center",
        "gap": "15px",
        "marginBottom": "20px",
        "padding": "10px 0",
    },
    children=[
        html.Img(
            src="assets/sectr.png",  # save your logo as assets/logo.png
            style={
                "height": "45px",
                "width": "auto",
                "flexShrink": "0",
            },
        ),
        html.H1(
            "SecTr A-SHM Dashboard",
            style={
                "margin": "0",
                "fontSize": "28px",
                "color": "#1f2937",
            },
        ),
    ],
),


        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="anomaly-score-graph",
                            figure=go.Figure(),
                            style={"height": "520px"},
                        ),
                        html.Div(
                            [
                                html.Button("Prev", id="btn-prev", n_clicks=0, style=BTN_STYLE),
                                html.Button("Next", id="btn-next", n_clicks=0, style=BTN_STYLE),
                                html.Button("Reset", id="btn-reset", n_clicks=0, style=BTN_RESET),
                                html.Button("Pause", id="btn-pause", n_clicks=0, style=BTN_PAUSE),
                                html.Button("⬇ Download table (CSV)", id="btn-dl-table", n_clicks=0, style=BTN_DOWNLOAD),
                                html.Button("⬇ Download chart (HTML)", id="btn-dl-chart", n_clicks=0, style=BTN_DOWNLOAD),
                            ],
                            style=STYLE_BTNROW,
                        ),
                    ],
                    style={**STYLE_CARD, **STYLE_LEFT},
                ),

                html.Div(
                    [
                        html.Div(dcc.Graph(id="temp-gauge"), style=STYLE_CARD),
                        html.Div(dcc.Graph(id="rh-gauge"), style=STYLE_CARD),
                    ],
                    style=STYLE_RIGHT,
                ),
            ],
            style=STYLE_ROW,
        ),

        html.Div(id="status-line", style={"marginTop": "6px", "fontSize": "13px", **STYLE_CENTER}),

        html.Div(
            [
                dash_table.DataTable(
                    id="anomaly-table",
                    columns=[
                        {"name": "Timestamp", "id": "timestamp"},
                        {"name": "Motion Label", "id": "motion_label"},
                        {"name": "Anomaly Score", "id": "anomaly_score"},
                        {"name": "Risk Level", "id": "risk_level"},
                    ],
                    data=[],
                    page_size=8,
                    sort_action="native",
                    style_table={"width": "100%"},
                    style_cell={"textAlign": "center", "padding": "8px", "fontFamily": "Arial", "fontSize": "13px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f5f7fb"},
                )
            ],
            style={**STYLE_CARD, "marginTop": "10px"},
        ),

        dcc.Store(id="page-store", data={"page": 0, "done": False}),
        dcc.Store(id="pause-store", data={"paused": False}),
        dcc.Interval(id="clock", interval=REFRESH_MS, n_intervals=0, disabled=False),

        dcc.Download(id="download-table"),
        dcc.Download(id="download-chart"),

        html.Pre(
            id="debug",
            style={
                "marginTop": "10px",
                "whiteSpace": "pre-wrap",
                "fontSize": "12px",
                "color": "#444",
                "display": "none",
            },
        ),
    ],
    style=STYLE_APP,
)


# ---- Main update callback (paging + pause + weather gauges) ----
@app.callback(
    [
        Output("anomaly-table", "data"),
        Output("anomaly-score-graph", "figure"),
        Output("temp-gauge", "figure"),
        Output("rh-gauge", "figure"),
        Output("status-line", "children"),
        Output("debug", "children"),
        Output("page-store", "data"),
        Output("clock", "disabled"),
        Output("pause-store", "data"),
        Output("btn-pause", "children"),
    ],
    [
        Input("clock", "n_intervals"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("btn-reset", "n_clicks"),
        Input("btn-pause", "n_clicks"),
    ],
    [State("page-store", "data"), State("pause-store", "data")],
)
def update(_n_intervals, _prev, _next, _reset, _pause_clicks, page_state, pause_state):
    weather, weather_err = get_openmeteo_current()
    temp_fig = gauge_temp(weather.get("temp_c"))
    rh_fig = gauge_rh(weather.get("rh"))

    scored_all, cache_msg = get_scored_cached()
    if scored_all is None:
        fig = go.Figure().update_layout(title="Anomaly score (no data)", height=520)
        status_line = (
            f" "
        )
        debug = f"{weather_err}\n{cache_msg}"
        return [], fig, temp_fig, rh_fig, status_line, debug, {"page": 0, "done": True}, True, {"paused": True}, "Resume"

    paused = bool((pause_state or {}).get("paused", False))

    total = len(scored_all)
    max_page = max(0, (total - 1) // POINTS_PER_PAGE)

    page = int((page_state or {}).get("page", 0))
    done = bool((page_state or {}).get("done", False))

    trig = (dash.callback_context.triggered[0]["prop_id"].split(".")[0]
            if dash.callback_context.triggered else "clock")

    if trig == "btn-pause":
        paused = not paused

    if trig == "btn-reset":
        page = 0
        done = False
    elif trig == "btn-prev":
        page = max(0, page - 1)
        done = False
    elif trig == "btn-next":
        page = min(max_page, page + 1)
        done = (page >= max_page)
    elif trig == "clock":
        if (not paused) and (not done):
            page = page + 1
            if page >= max_page:
                page = max_page
                done = True

    start = page * POINTS_PER_PAGE
    end = min(start + POINTS_PER_PAGE, total)
    window_df = scored_all.iloc[start:end].copy()

    fig = anomaly_score_figure(window_df)
    table_data = build_table_for_window(scored_all, window_df, max_rows=250)

    tmin = window_df["timestamp"].min()
    tmax = window_df["timestamp"].max()

    status_line = (
        f" "
        f" "
        f" "
        f" "
    )

    debug = f"{weather_err}\n{cache_msg}\ntrigger={trig}\npaused={paused}\ndone={done}\nflagged_in_window={len(table_data)}"
    clock_disabled = bool(paused or done)
    pause_btn_text = "Resume" if paused else "Pause"

    return (
        table_data,
        fig,
        temp_fig,
        rh_fig,
        status_line,
        debug,
        {"page": page, "done": done},
        clock_disabled,
        {"paused": paused},
        pause_btn_text,
    )


# ---- Download table (CSV) for current timeframe ----
@app.callback(
    Output("download-table", "data"),
    Input("btn-dl-table", "n_clicks"),
    State("anomaly-table", "data"),
    State("page-store", "data"),
    prevent_initial_call=True,
)
def download_table(n_clicks, table_rows, page_state):
    if not n_clicks:
        return dash.no_update
    df = pd.DataFrame(table_rows or [])
    page = int((page_state or {}).get("page", 0))
    fname = f"anomaly_table_page_{page+1}.csv"
    return dcc.send_data_frame(df.to_csv, fname, index=False)


# ---- Download chart (HTML) for current timeframe ----
@app.callback(
    Output("download-chart", "data"),
    Input("btn-dl-chart", "n_clicks"),
    State("anomaly-score-graph", "figure"),
    State("page-store", "data"),
    prevent_initial_call=True,
)
def download_chart(n_clicks, fig_dict, page_state):
    if not n_clicks:
        return dash.no_update
    fig = go.Figure(fig_dict)
    page = int((page_state or {}).get("page", 0))
    fname = f"anomaly_chart_page_{page+1}.html"
    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    return dict(content=html_str, filename=fname, type="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
