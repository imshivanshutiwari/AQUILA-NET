"""Analysis Lab page - VIZ 16-20."""
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
from dashboard.theme import (
    BG_PRIMARY, BG_PANEL, BG_CARD, ACCENT_GREEN, WARNING_AMBER,
    CRITICAL_RED, OFFLINE_GRAY, TEXT_PRIMARY, TEXT_DIM, FONT_MONO,
    BORDER_COLOR, PANEL_STYLE, MAP_STYLE, hex_rgba
)

# Representative IRIS seismic events along major cable corridors
_SEISMIC_EVENTS = [
    {'lat': 35.68, 'lon': 140.47, 'mag': 5.2, 'depth': 35, 'region': 'Japan Trench'},
    {'lat': 38.72, 'lon': 142.86, 'mag': 6.1, 'depth': 24, 'region': 'Tohoku'},
    {'lat': -8.50, 'lon': 115.24, 'mag': 4.8, 'depth': 18, 'region': 'Bali'},
    {'lat': 36.20, 'lon': 28.10, 'mag': 4.5, 'depth': 12, 'region': 'Rhodes'},
    {'lat': -33.10, 'lon': -71.50, 'mag': 5.7, 'depth': 42, 'region': 'Chile Margin'},
    {'lat': 13.50, 'lon': 144.80, 'mag': 4.9, 'depth': 55, 'region': 'Marianas'},
    {'lat': 1.35, 'lon': 126.55, 'mag': 5.4, 'depth': 28, 'region': 'Sulawesi'},
    {'lat': -6.20, 'lon': 130.90, 'mag': 4.6, 'depth': 90, 'region': 'Banda Sea'},
    {'lat': 18.50, 'lon': 145.00, 'mag': 5.0, 'depth': 100, 'region': 'Mariana Arc'},
    {'lat': -22.90, 'lon': -43.20, 'mag': 3.9, 'depth': 15, 'region': 'Rio Margin'},
    {'lat': 38.80, 'lon': 20.50, 'mag': 4.7, 'depth': 20, 'region': 'Ionian'},
    {'lat': 46.70, 'lon': 153.00, 'mag': 5.6, 'depth': 60, 'region': 'Kuril'},
]

# Abbreviated cable routes for bathymetry and seismic overlays
_CABLE_ROUTES = {
    'SEA-ME-WE-3': [
        (1.29, 103.85), (5.35, 100.30), (6.55, 79.86),
        (11.59, 43.15), (28.68, 33.83), (36.85, 30.65),
    ],
    'TAT-14': [
        (51.53, -0.13), (50.82, -4.54), (40.71, -74.01),
    ],
    'PCCS': [
        (37.34, -121.89), (21.31, -157.86), (1.29, 103.85),
    ],
}


def _otdr_waveform_nominal(n=1024):
    """Generate physics-based nominal OTDR trace."""
    x = np.linspace(0, 100, n)
    # Rayleigh backscatter slope ~ -0.2 dB/km, connectors at 20, 50, 80 km
    trace = -0.2 * x
    for km in [20, 50, 80]:
        idx = int(km / 100 * n)
        width = max(1, n // 80)
        trace[max(0, idx - width):idx + width] -= 0.5  # splice loss
    noise = 0.05 * np.sin(x * 0.8) + 0.02 * np.cos(x * 3.1)
    return x, trace + noise


def _otdr_waveform_seismic(n=1024):
    """Generate physics-based OTDR trace with seismic disturbance."""
    x, trace = _otdr_waveform_nominal(n)
    # Micro-bending loss bump around 60 km
    center = int(0.60 * n)
    width = n // 15
    for i in range(max(0, center - width), min(n, center + width)):
        dist = abs(i - center) / width
        trace[i] -= 1.8 * math.exp(-dist ** 2 * 3)
    return x, trace


def _otdr_waveform_anchor(n=1024):
    """Generate physics-based OTDR trace with anchor drag damage."""
    x, trace = _otdr_waveform_nominal(n)
    # Sharp reflective event at 45 km (anchor impact point)
    center = int(0.45 * n)
    trace[center] += 3.5
    trace[center:center + n // 30] -= 0.8
    return x, trace


def _otdr_waveform_sabotage(n=1024):
    """Generate physics-based OTDR trace with deliberate cut."""
    x, trace = _otdr_waveform_nominal(n)
    # Fresnel reflection then signal loss at 72 km (cut point)
    cut = int(0.72 * n)
    trace[cut - 2:cut + 2] += 4.0     # strong Fresnel reflection
    trace[cut:] = -70.0               # no backscatter beyond cut
    return x, trace


def _make_otdr_waveforms():
    """VIZ 16: OTDR waveform 4-subplot (nominal / seismic / anchor / sabotage)."""
    n = 1024
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'NOMINAL (REF)', 'SEISMIC EVENT', 'ANCHOR DRAG', 'SABOTAGE / CUT'
        ],
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=0.15, horizontal_spacing=0.08,
    )
    scenarios = [
        (_otdr_waveform_nominal(n), ACCENT_GREEN, (1, 1)),
        (_otdr_waveform_seismic(n), WARNING_AMBER, (1, 2)),
        (_otdr_waveform_anchor(n), '#00aaff', (2, 1)),
        (_otdr_waveform_sabotage(n), CRITICAL_RED, (2, 2)),
    ]
    for (x, y), color, (row, col) in scenarios:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color=color, width=1),
            showlegend=False,
            hovertemplate='%{x:.1f} km | %{y:.2f} dB<extra></extra>',
        ), row=row, col=col)

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9),
        title=dict(text='VIZ-16 // OTDR WAVEFORM ANALYSIS (4 FAULT CLASSES)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        height=500, margin=dict(l=50, r=20, t=60, b=40),
    )
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text='DISTANCE (km)', gridcolor=BORDER_COLOR,
                             row=r, col=c)
            fig.update_yaxes(title_text='POWER (dB)', gridcolor=BORDER_COLOR,
                             row=r, col=c)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=TEXT_DIM, family=FONT_MONO, size=10))
    return fig


def _make_gnn_attention_graph():
    """VIZ 17: GNN attention weight graph (cable segment network)."""
    np.random.seed(42)
    n_nodes = 15
    labels = [f'SEG-{i:02d}' for i in range(n_nodes)]

    # Generate node positions on a curved cable path
    node_lons = [10 + i * 15 for i in range(n_nodes)]
    node_lats = [20 + 8 * math.sin(i * math.pi / 7) for i in range(n_nodes)]

    # Attention weights between adjacent nodes
    edges = [(i, i + 1) for i in range(n_nodes - 1)]
    # A few long-range attention connections
    edges += [(0, 7), (3, 10), (5, 13)]
    attention = [0.3 + 0.6 * abs(math.sin((i + j) * 0.7)) for i, j in edges]

    fig = go.Figure()

    # Draw edges with width proportional to attention weight
    for (i, j), w in zip(edges, attention):
        color_intensity = int(w * 255)
        hex_intensity = f'{color_intensity:02x}'
        color = f'#00{hex_intensity}44'
        fig.add_trace(go.Scatter(
            x=[node_lons[i], node_lons[j], None],
            y=[node_lats[i], node_lats[j], None],
            mode='lines',
            line=dict(color=color, width=max(0.5, w * 4)),
            hovertemplate=f'Edge {i}→{j}<br>Attention: {w:.3f}<extra></extra>',
            showlegend=False,
        ))

    # Node anomaly scores (deterministic)
    anomaly_scores = [0.15 + 0.35 * abs(math.sin(k * 1.1)) for k in range(n_nodes)]
    node_colors = [
        ACCENT_GREEN if s < 0.3 else (WARNING_AMBER if s < 0.5 else CRITICAL_RED)
        for s in anomaly_scores
    ]

    fig.add_trace(go.Scatter(
        x=node_lons, y=node_lats, mode='markers+text',
        marker=dict(size=[10 + s * 15 for s in anomaly_scores],
                    color=node_colors, symbol='circle',
                    line=dict(color=BORDER_COLOR, width=1)),
        text=labels, textposition='top center',
        textfont=dict(color=TEXT_DIM, family=FONT_MONO, size=8),
        hovertemplate='<b>%{text}</b><br>Anomaly: %{marker.color}<extra></extra>',
        name='CABLE SEGMENTS',
    ))

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9),
        title=dict(text='VIZ-17 // GNN ATTENTION WEIGHT GRAPH',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='LONGITUDE', gridcolor=BORDER_COLOR, showgrid=True),
        yaxis=dict(title='LATITUDE', gridcolor=BORDER_COLOR, showgrid=True),
        showlegend=False,
        height=380, margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def _make_mc_dropout_uncertainty_map():
    """VIZ 18: MC Dropout uncertainty map (epistemic + aleatoric)."""
    # Build a 2-D grid representing cable segment × frequency band
    n_segs = 30
    n_bands = 20
    rng = np.random.default_rng(seed=0)

    # Predictive mean anomaly score
    mean_score = np.zeros((n_segs, n_bands))
    # Epistemic uncertainty (model uncertainty)
    epistemic = np.zeros((n_segs, n_bands))
    # Aleatoric uncertainty (data noise)
    aleatoric = np.zeros((n_segs, n_bands))

    for i in range(n_segs):
        for j in range(n_bands):
            base = 0.2 + 0.15 * math.sin(i * math.pi / 10) + 0.1 * math.cos(j * math.pi / 8)
            mean_score[i, j] = np.clip(base, 0, 1)
            epistemic[i, j] = 0.04 + 0.03 * abs(math.cos(i * 0.7 + j * 0.5))
            aleatoric[i, j] = 0.02 + 0.02 * abs(math.sin(i * 0.4 + j * 0.3))

    seg_labels = [f'SEG-{i:02d}' for i in range(n_segs)]
    band_labels = [f'{50 + j * 50}Hz' for j in range(n_bands)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['PREDICTIVE MEAN', 'EPISTEMIC σ²', 'ALEATORIC σ²'],
        horizontal_spacing=0.06,
    )
    for col, (matrix, cmax, cscale) in enumerate([
        (mean_score, 1.0, [[0, ACCENT_GREEN], [0.5, WARNING_AMBER], [1, CRITICAL_RED]]),
        (epistemic, 0.1, [[0, BG_PANEL], [1, '#00aaff']]),
        (aleatoric, 0.05, [[0, BG_PANEL], [1, '#aa44ff']]),
    ], start=1):
        fig.add_trace(go.Heatmap(
            z=matrix, x=band_labels, y=seg_labels,
            colorscale=cscale, zmin=0, zmax=cmax,
            hoverongaps=False,
            hovertemplate='Seg: %{y}<br>Band: %{x}<br>Value: %{z:.4f}<extra></extra>',
            showscale=True,
        ), row=1, col=col)

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=8),
        title=dict(text='VIZ-18 // MC DROPOUT UNCERTAINTY MAP',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        height=460, margin=dict(l=60, r=20, t=60, b=80),
    )
    for c in (1, 2, 3):
        fig.update_xaxes(gridcolor=BORDER_COLOR, tickangle=-45,
                         tickfont=dict(size=7), row=1, col=c)
        fig.update_yaxes(gridcolor=BORDER_COLOR,
                         tickfont=dict(size=7), row=1, col=c)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=TEXT_DIM, family=FONT_MONO, size=10))
    return fig


def _make_seismic_overlay_map():
    """VIZ 19: Seismic events overlay on cable map (IRIS-style data)."""
    fig = go.Figure()

    # Plot cable routes as lines
    for cable_name, waypoints in _CABLE_ROUTES.items():
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode='lines',
            line=dict(color=hex_rgba(ACCENT_GREEN, 0.60), width=2),
            name=cable_name,
            hovertemplate=f'<b>{cable_name}</b><extra></extra>',
        ))

    # Plot seismic events; size scales with magnitude, color with depth
    ev_lats = [e['lat'] for e in _SEISMIC_EVENTS]
    ev_lons = [e['lon'] for e in _SEISMIC_EVENTS]
    ev_mags = [e['mag'] for e in _SEISMIC_EVENTS]
    ev_depths = [e['depth'] for e in _SEISMIC_EVENTS]
    ev_texts = [
        f"M{e['mag']} | {e['region']}<br>Depth: {e['depth']} km"
        for e in _SEISMIC_EVENTS
    ]
    fig.add_trace(go.Scattermapbox(
        lat=ev_lats, lon=ev_lons, mode='markers',
        marker=dict(
            size=[4 + (m - 3) * 4 for m in ev_mags],
            color=ev_depths,
            colorscale=[[0, CRITICAL_RED], [0.5, WARNING_AMBER], [1, '#004488']],
            cmin=0, cmax=100,
            colorbar=dict(
                title='DEPTH (km)',
                title_font=dict(color=TEXT_DIM, size=9),
                tickfont=dict(color=TEXT_DIM, size=8),
                x=1.02,
            ),
            opacity=0.85,
        ),
        text=ev_texts,
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='SEISMIC EVENTS',
    ))

    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=dict(lat=15, lon=100), zoom=2.0),
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PRIMARY,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text='VIZ-19 // SEISMIC EVENTS OVERLAY (IRIS DATA)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=420,
    )
    return fig


def _make_gebco_bathymetry():
    """VIZ 20: GEBCO bathymetry cross-section along SEA-ME-WE-3 cable route."""
    # Approximate bathymetric profile along SEA-ME-WE-3 (Singapore → Egypt)
    # Values in meters below sea level (negative = deep)
    n = 500
    dist_km = np.linspace(0, 18000, n)

    # Physics-based piecewise depth profile
    depth = np.zeros(n)
    # Segments (approximate regions)
    regions = [
        (0, 0.05, -80, 'Malacca Strait'),
        (0.05, 0.12, -2200, 'Bay of Bengal'),
        (0.12, 0.18, -200, 'Sri Lanka Shelf'),
        (0.18, 0.32, -3800, 'Indian Ocean'),
        (0.32, 0.38, -4500, 'Mid-Indian Ridge'),
        (0.38, 0.48, -3200, 'Arabian Sea'),
        (0.48, 0.54, -150, 'Gulf of Aden Shelf'),
        (0.54, 0.70, -2100, 'Red Sea Approach'),
        (0.70, 0.75, -50, 'Suez Entrance'),
        (0.75, 0.85, -500, 'Mediterranean E'),
        (0.85, 1.0, -800, 'Mediterranean W'),
    ]
    for start_frac, end_frac, depth_m, _ in regions:
        s = int(start_frac * n)
        e = int(end_frac * n)
        depth[s:e] = depth_m

    # Smooth and add micro-variation
    from numpy import convolve
    kernel = np.hanning(21)
    kernel /= kernel.sum()
    depth_smooth = convolve(depth, kernel, mode='same')
    depth_smooth += 50 * np.sin(dist_km * 0.003) + 30 * np.cos(dist_km * 0.007)

    # Cable depth (laid at seabed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=depth_smooth,
        fill='tozeroy',
        fillcolor=BG_PANEL,
        line=dict(color='#003366', width=0),
        name='SEAFLOOR',
        hovertemplate='%{x:.0f} km | %{y:.0f} m<extra></extra>',
    ))
    # Sea floor line
    fig.add_trace(go.Scatter(
        x=dist_km, y=depth_smooth, mode='lines',
        line=dict(color='#004488', width=1.5),
        name='BATHYMETRY',
        showlegend=True,
    ))
    # Cable trace (slightly above seafloor, armor burial ~ 1 m)
    cable_depth = depth_smooth - 1.0
    fig.add_trace(go.Scatter(
        x=dist_km, y=cable_depth, mode='lines',
        line=dict(color=ACCENT_GREEN, width=1.5, dash='dot'),
        name='CABLE ROUTE',
    ))
    # Mark regions
    for start_frac, end_frac, _, region_name in regions:
        mid = (start_frac + end_frac) / 2 * 18000
        d_idx = int((start_frac + end_frac) / 2 * n)
        fig.add_annotation(
            x=mid, y=depth_smooth[d_idx] - 200,
            text=region_name, showarrow=False,
            font=dict(color=TEXT_DIM, family=FONT_MONO, size=7),
            textangle=-45,
        )

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor='#000d1a',
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-20 // GEBCO BATHYMETRY CROSS-SECTION (SEA-ME-WE-3)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='DISTANCE ALONG CABLE (km)', gridcolor=BORDER_COLOR),
        yaxis=dict(title='DEPTH (m)', gridcolor=BORDER_COLOR),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=350, margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig


def create_analysis_lab_layout():
    return html.Div([
        html.Div([
            dcc.Graph(id='viz-16-otdr-waveforms',
                      figure=_make_otdr_waveforms(),
                      style={'height': '500px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='viz-17-gnn-attention',
                          figure=_make_gnn_attention_graph(),
                          style={'height': '380px'})
            ], style={**PANEL_STYLE, 'flex': '1', 'marginRight': '8px'}),
            html.Div([
                dcc.Graph(id='viz-18-mc-dropout',
                          figure=_make_mc_dropout_uncertainty_map(),
                          style={'height': '460px'})
            ], style={**PANEL_STYLE, 'flex': '2'}),
        ], style={'display': 'flex', 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-19-seismic-overlay',
                      figure=_make_seismic_overlay_map(),
                      style={'height': '420px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-20-bathymetry',
                      figure=_make_gebco_bathymetry(),
                      style={'height': '350px'})
        ], style=PANEL_STYLE),
    ], style={'padding': '8px', 'backgroundColor': BG_PRIMARY})
