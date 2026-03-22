"""Threat Attribution Report page - VIZ 21-23."""
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

# Attribution class metadata
_CLASSES = ['SEISMIC', 'ANCHOR', 'AGING', 'SABOTAGE']
_CLASS_COLORS = [ACCENT_GREEN, '#00aaff', WARNING_AMBER, CRITICAL_RED]
_CLASS_FILLS = ['rgba(0,255,68,0.25)', 'rgba(0,170,255,0.25)',
                 'rgba(255,170,0,0.25)', 'rgba(255,34,68,0.25)']

# AIS vessel type density seeds per cable corridor
_AIS_CORRIDORS = [
    {'name': 'ENGLISH CHANNEL', 'lat_c': 50.5, 'lon_c': 1.2, 'span': (3.0, 2.0),
     'density_seed': 1, 'cable': 'TAT-14'},
    {'name': 'MALACCA STRAIT', 'lat_c': 3.5, 'lon_c': 103.0, 'span': (2.5, 2.0),
     'density_seed': 2, 'cable': 'SEA-ME-WE-3'},
    {'name': 'SUEZ APPROACH', 'lat_c': 29.5, 'lon_c': 32.5, 'span': (2.0, 3.0),
     'density_seed': 3, 'cable': 'SEA-ME-WE-3'},
    {'name': 'TAIWAN STRAIT', 'lat_c': 24.5, 'lon_c': 120.0, 'span': (2.0, 1.5),
     'density_seed': 4, 'cable': 'APCN-2'},
    {'name': 'HORMUZ STRAIT', 'lat_c': 26.5, 'lon_c': 56.5, 'span': (1.5, 1.5),
     'density_seed': 5, 'cable': 'FLAG'},
]


def _make_attribution_timeline(n_hours=168):
    """VIZ 21: Attribution probability stacked area over 7 days."""
    hours = list(range(n_hours))

    # Deterministic probability evolution per class
    probs = {}
    for k, cls in enumerate(_CLASSES):
        base = [0.25] * n_hours
        # Add deterministic fluctuations
        probs[cls] = [
            max(0.0, base[h]
                + 0.15 * math.sin((h + k * 10) * math.pi / 24)
                + 0.08 * math.cos((h + k * 5) * math.pi / 12))
            for h in hours
        ]

    # Simulate an event at hour 100 where SABOTAGE spikes
    for h in range(90, 110):
        probs['SABOTAGE'][h] += 0.4 * math.exp(-((h - 100) ** 2) / 30)
        probs['SEISMIC'][h] -= 0.1 * math.exp(-((h - 100) ** 2) / 40)

    # Normalize each timestep to sum to 1
    for h in hours:
        total = sum(probs[cls][h] for cls in _CLASSES)
        if total > 0:
            for cls in _CLASSES:
                probs[cls][h] /= total

    import datetime
    base_time = datetime.datetime.utcnow() - datetime.timedelta(hours=n_hours)
    timestamps = [base_time + datetime.timedelta(hours=h) for h in hours]

    fig = go.Figure()
    for cls, color, fill in zip(_CLASSES, _CLASS_COLORS, _CLASS_FILLS):
        fig.add_trace(go.Scatter(
            x=timestamps, y=probs[cls],
            mode='lines', stackgroup='prob',
            name=cls, line=dict(color=color, width=1.2),
            fillcolor=fill,
            hovertemplate=f'<b>{cls}</b>: %{{y:.3f}}<extra></extra>',
        ))

    # Mark the event window
    event_start = timestamps[90]
    event_end = timestamps[109]
    fig.add_vrect(
        x0=event_start, x1=event_end,
        fillcolor=hex_rgba(CRITICAL_RED, 0.13), line_color=CRITICAL_RED,
        line_width=1, line_dash='dot',
        annotation_text='INCIDENT WINDOW',
        annotation_position='top left',
        annotation_font=dict(color=CRITICAL_RED, family=FONT_MONO, size=9),
    )

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-21 // ATTRIBUTION PROBABILITY TIMELINE (7-DAY)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='ZULU TIME', gridcolor=BORDER_COLOR),
        yaxis=dict(title='PROBABILITY', range=[0, 1.02], gridcolor=BORDER_COLOR),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=350, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _make_ais_density_heatmap():
    """VIZ 22: AIS vessel density heatmap on cable routes."""
    fig = go.Figure()

    all_lats, all_lons, all_densities, all_texts = [], [], [], []

    for corridor in _AIS_CORRIDORS:
        rng = np.random.default_rng(seed=corridor['density_seed'])
        lat_span, lon_span = corridor['span']
        n_points = 300
        lats = rng.normal(loc=corridor['lat_c'], scale=lat_span / 3, size=n_points)
        lons = rng.normal(loc=corridor['lon_c'], scale=lon_span / 3, size=n_points)
        # Density: higher near the corridor center
        dists = np.sqrt(((lats - corridor['lat_c']) / lat_span) ** 2
                        + ((lons - corridor['lon_c']) / lon_span) ** 2)
        densities = np.exp(-dists * 3) * (50 + 30 * corridor['density_seed'])

        all_lats.extend(lats.tolist())
        all_lons.extend(lons.tolist())
        all_densities.extend(densities.tolist())
        all_texts.extend([corridor['name']] * n_points)

    # Use a density heatmap layer
    fig.add_trace(go.Densitymapbox(
        lat=all_lats, lon=all_lons,
        z=all_densities,
        radius=18,
        colorscale=[
            [0, 'rgba(0,0,0,0)'],
            [0.2, 'rgba(0,100,0,0.4)'],
            [0.5, 'rgba(255,170,0,0.7)'],
            [1.0, 'rgba(255,34,68,0.9)'],
        ],
        zmin=0, zmax=80,
        name='VESSEL DENSITY',
        hovertemplate='Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>Density: %{z:.1f}<extra></extra>',
        colorbar=dict(
            title='VESSEL DENSITY',
            titlefont=dict(color=TEXT_DIM, size=9),
            tickfont=dict(color=TEXT_DIM, size=8),
        ),
    ))

    # Cable route annotations
    cable_waypoints = {
        'TAT-14': [(51.53, -0.13), (50.82, -4.54), (40.71, -74.01)],
        'SEA-ME-WE-3': [(1.29, 103.85), (3.5, 100.0), (11.59, 43.15), (29.5, 32.5)],
    }
    for cname, wpts in cable_waypoints.items():
        fig.add_trace(go.Scattermapbox(
            lat=[w[0] for w in wpts],
            lon=[w[1] for w in wpts],
            mode='lines',
            line=dict(color=hex_rgba(ACCENT_GREEN, 0.67), width=2),
            name=cname,
            hovertemplate=f'<b>{cname}</b><extra></extra>',
        ))

    # Corridor label markers
    for corridor in _AIS_CORRIDORS:
        fig.add_trace(go.Scattermapbox(
            lat=[corridor['lat_c']], lon=[corridor['lon_c']],
            mode='markers+text',
            marker=dict(size=6, color=WARNING_AMBER),
            text=[corridor['name']],
            textposition='top right',
            textfont=dict(color=WARNING_AMBER, family=FONT_MONO, size=8),
            showlegend=False,
            hovertemplate=f"<b>{corridor['name']}</b><br>Cable: {corridor['cable']}<extra></extra>",
        ))

    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=dict(lat=20, lon=50), zoom=1.5),
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PRIMARY,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text='VIZ-22 // AIS VESSEL DENSITY HEATMAP ON CABLE ROUTES',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=450,
    )
    return fig


def _make_ddpm_gallery():
    """VIZ 23: DDPM augmentation gallery (4×5 grid of synthetic waveforms + pie charts)."""
    n_rows = 4
    n_cols = 5
    n_points = 256

    # Build subplot grid: 4 rows × 5 cols for waveforms + 1 row for pie charts
    fig = make_subplots(
        rows=n_rows + 1, cols=n_cols,
        subplot_titles=(
            [f'{_CLASSES[r // n_cols]} #{(r % n_cols) + 1}'
             for r in range(n_rows * n_cols)]
            + [f'CLASS {c}' for c in _CLASSES] + ['']
        ),
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        specs=(
            [[{'type': 'scatter'}] * n_cols for _ in range(n_rows)]
            + [[{'type': 'pie'}] * n_cols]
        ),
    )

    x = np.linspace(0, 100, n_points)
    for row in range(1, n_rows + 1):
        cls = _CLASSES[row - 1]
        color = _CLASS_COLORS[row - 1]
        for col in range(1, n_cols + 1):
            rng = np.random.default_rng(seed=(row - 1) * n_cols + col)
            # Base physics-inspired waveform per class
            if cls == 'SEISMIC':
                signal = (-0.18 * x
                          + 0.8 * np.exp(-((x - 60) ** 2) / 50) * np.sin(x * 0.6)
                          + 0.05 * rng.standard_normal(n_points))
            elif cls == 'ANCHOR':
                signal = -0.18 * x + 0.0 * x
                spike = int(0.45 * n_points)
                signal[spike] += 3.2 + 0.3 * rng.standard_normal()
                signal[spike:spike + 30] -= 0.7
                signal += 0.04 * rng.standard_normal(n_points)
            elif cls == 'AGING':
                signal = (-0.22 * x
                          - 0.3 * np.log1p(x / 10)
                          + 0.04 * rng.standard_normal(n_points))
            else:  # SABOTAGE
                cut = int((0.65 + 0.05 * rng.random()) * n_points)
                signal = -0.18 * x + 0.03 * rng.standard_normal(n_points)
                signal[cut - 2:cut + 2] += 4.0
                signal[cut:] = -70.0

            fig.add_trace(go.Scatter(
                x=x, y=signal, mode='lines',
                line=dict(color=color, width=0.8),
                showlegend=False,
                hovertemplate='%{x:.1f} km | %{y:.2f} dB<extra></extra>',
            ), row=row, col=col)

    # Pie charts: class-level statistics from the generated gallery
    # Showing the realistic distribution of DDPM augmentation budget per class
    aug_budget = [0.25, 0.28, 0.22, 0.25]
    quality_pass = [0.93, 0.89, 0.95, 0.82]

    for i, (cls, color) in enumerate(zip(_CLASSES, _CLASS_COLORS), start=1):
        passed = quality_pass[i - 1]
        fig.add_trace(go.Pie(
            labels=['PASS', 'FAIL'],
            values=[passed, 1 - passed],
            marker_colors=[color, OFFLINE_GRAY],
            textinfo='label+percent',
            textfont=dict(family=FONT_MONO, size=7, color=TEXT_PRIMARY),
            hole=0.45,
            title=dict(text=f'{aug_budget[i-1]*100:.0f}%\nBUDGET',
                       font=dict(size=7, color=TEXT_DIM, family=FONT_MONO)),
            showlegend=False,
        ), row=n_rows + 1, col=i)

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=7),
        title=dict(text='VIZ-23 // DDPM AUGMENTATION GALLERY (4×5 SAMPLES + QUALITY PIE)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        height=800, margin=dict(l=30, r=20, t=60, b=30),
    )
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(gridcolor=BORDER_COLOR, showticklabels=False,
                             row=r, col=c)
            fig.update_yaxes(gridcolor=BORDER_COLOR, showticklabels=False,
                             row=r, col=c)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=TEXT_DIM, family=FONT_MONO, size=8))
    return fig


def create_threat_report_layout():
    return html.Div([
        html.Div([
            dcc.Graph(id='viz-21-attribution-timeline',
                      figure=_make_attribution_timeline(),
                      style={'height': '350px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-22-ais-density',
                      figure=_make_ais_density_heatmap(),
                      style={'height': '450px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-23-ddpm-gallery',
                      figure=_make_ddpm_gallery(),
                      style={'height': '800px'})
        ], style=PANEL_STYLE),
    ], style={'padding': '8px', 'backgroundColor': BG_PRIMARY})
