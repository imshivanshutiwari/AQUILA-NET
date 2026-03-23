"""Federated Learning Status page - VIZ 11-15."""
import datetime
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

# Representative FL client nodes co-located at major cable landing points
_FL_CLIENT_NODES = [
    {'id': 'NODE-US-W', 'name': 'San Jose, CA', 'lat': 37.34, 'lon': -121.89,
     'region': 'USPACOM', 'rounds': 42, 'status': 'ACTIVE'},
    {'id': 'NODE-UK', 'name': 'Bude, UK', 'lat': 50.82, 'lon': -4.54,
     'region': 'EUCOM', 'rounds': 41, 'status': 'ACTIVE'},
    {'id': 'NODE-SG', 'name': 'Singapore', 'lat': 1.29, 'lon': 103.85,
     'region': 'INDOPACOM', 'rounds': 39, 'status': 'ACTIVE'},
    {'id': 'NODE-JP', 'name': 'Chikura, JP', 'lat': 34.93, 'lon': 139.97,
     'region': 'INDOPACOM', 'rounds': 40, 'status': 'ACTIVE'},
    {'id': 'NODE-AU', 'name': 'Perth, AU', 'lat': -31.95, 'lon': 115.86,
     'region': 'INDOPACOM', 'rounds': 38, 'status': 'ACTIVE'},
    {'id': 'NODE-FR', 'name': 'Marseille, FR', 'lat': 43.30, 'lon': 5.37,
     'region': 'EUCOM', 'rounds': 37, 'status': 'ACTIVE'},
    {'id': 'NODE-BR', 'name': 'Fortaleza, BR', 'lat': -3.72, 'lon': -38.54,
     'region': 'SOUTHCOM', 'rounds': 35, 'status': 'ACTIVE'},
    {'id': 'NODE-ZA', 'name': 'Cape Town, ZA', 'lat': -33.92, 'lon': 18.42,
     'region': 'AFRICOM', 'rounds': 33, 'status': 'DEGRADED'},
    {'id': 'NODE-IN', 'name': 'Mumbai, IN', 'lat': 19.08, 'lon': 72.88,
     'region': 'CENTCOM', 'rounds': 36, 'status': 'ACTIVE'},
    {'id': 'NODE-EG', 'name': 'Alexandria, EG', 'lat': 31.20, 'lon': 29.92,
     'region': 'CENTCOM', 'rounds': 30, 'status': 'DEGRADED'},
    {'id': 'NODE-HK', 'name': 'Hong Kong', 'lat': 22.32, 'lon': 114.17,
     'region': 'INDOPACOM', 'rounds': 38, 'status': 'ACTIVE'},
    {'id': 'NODE-US-E', 'name': 'New York, NY', 'lat': 40.71, 'lon': -74.01,
     'region': 'NORTHCOM', 'rounds': 42, 'status': 'ACTIVE'},
]

_REGION_COLORS = {
    'USPACOM': ACCENT_GREEN,
    'EUCOM': '#00aaff',
    'INDOPACOM': WARNING_AMBER,
    'SOUTHCOM': '#aa44ff',
    'AFRICOM': CRITICAL_RED,
    'CENTCOM': '#ff8800',
    'NORTHCOM': '#00ffcc',
}


def _make_fl_client_map(client_nodes=None):
    """VIZ 11: FL client node map on world map."""
    nodes = client_nodes or _FL_CLIENT_NODES

    region_groups = {}
    for node in nodes:
        r = node.get('region', 'UNKNOWN')
        region_groups.setdefault(r, []).append(node)

    fig = go.Figure()
    for region, members in region_groups.items():
        color = _REGION_COLORS.get(region, OFFLINE_GRAY)
        active = [n for n in members if n.get('status') == 'ACTIVE']
        degraded = [n for n in members if n.get('status') != 'ACTIVE']

        for group, marker_sym, label_suffix in [
            (active, 'circle', ''), (degraded, 'circle-x', ' [DEGRADED]')
        ]:
            if not group:
                continue
            lats = [n['lat'] for n in group]
            lons = [n['lon'] for n in group]
            texts = [
                f"{n['name']}<br>Region: {n.get('region','?')}<br>"
                f"Rounds: {n.get('rounds',0)}<br>Status: {n.get('status','?')}"
                for n in group
            ]
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode='markers',
                marker=dict(size=10, color=color, symbol=marker_sym),
                text=texts,
                hovertemplate='%{text}<extra></extra>',
                name=f'{region}{label_suffix}',
            ))

    # Draw lines from each client to a central aggregation point (lat=0, lon=0)
    for node in nodes:
        col = _REGION_COLORS.get(node.get('region', ''), OFFLINE_GRAY)
        line_color = hex_rgba(col, 0.50 if node.get('status') == 'ACTIVE' else 0.19)
        fig.add_trace(go.Scattermapbox(
            lat=[node['lat'], 0.0], lon=[node['lon'], 0.0],
            mode='lines',
            line=dict(color=line_color, width=0.8),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Central aggregation server marker
    fig.add_trace(go.Scattermapbox(
        lat=[0.0], lon=[0.0], mode='markers+text',
        marker=dict(size=14, color=ACCENT_GREEN, symbol='star'),
        text=['AGG SERVER'], textposition='top right',
        textfont=dict(color=ACCENT_GREEN, family=FONT_MONO, size=9),
        name='AGG SERVER',
        hovertemplate='AGGREGATION SERVER<extra></extra>',
    ))

    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=dict(lat=20, lon=10), zoom=1.2),
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PRIMARY,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text='VIZ-11 // FL CLIENT NODE MAP // GLOBAL FEDERATION',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=420,
    )
    return fig


def _make_privacy_budget_gauge(epsilon_used=0.42, epsilon_max=1.0, delta=1e-5):
    """VIZ 12: Privacy budget circular gauge (ε-δ DP)."""
    pct = epsilon_used / epsilon_max
    # Segment colors: green → amber → red
    if pct < 0.5:
        bar_color = ACCENT_GREEN
    elif pct < 0.8:
        bar_color = WARNING_AMBER
    else:
        bar_color = CRITICAL_RED

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode='gauge+number+delta',
        value=epsilon_used,
        delta={'reference': 0.0, 'increasing': {'color': CRITICAL_RED},
               'valueformat': '.3f', 'suffix': ' ε'},
        number={'valueformat': '.3f', 'suffix': ' ε',
                'font': {'color': bar_color, 'family': FONT_MONO, 'size': 28}},
        gauge={
            'axis': {
                'range': [0, epsilon_max],
                'tickwidth': 1,
                'tickcolor': TEXT_DIM,
                'tickfont': {'color': TEXT_DIM, 'family': FONT_MONO, 'size': 9},
                'nticks': 6,
            },
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': BG_PANEL,
            'borderwidth': 1,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, epsilon_max * 0.5], 'color': 'rgba(0,255,68,0.13)'},
                {'range': [epsilon_max * 0.5, epsilon_max * 0.8], 'color': 'rgba(255,170,0,0.13)'},
                {'range': [epsilon_max * 0.8, epsilon_max], 'color': 'rgba(255,34,68,0.13)'},
            ],
            'threshold': {
                'line': {'color': CRITICAL_RED, 'width': 2},
                'thickness': 0.75,
                'value': epsilon_max * 0.9,
            },
        },
        title={'text': f'ε CONSUMED<br><span style="font-size:10px">δ = {delta:.0e}'
                       f' | {pct * 100:.1f}% BUDGET USED</span>',
               'font': {'color': ACCENT_GREEN, 'family': FONT_MONO, 'size': 13}},
        domain={'x': [0.1, 0.9], 'y': [0.15, 0.95]},
    ))
    # Active clients annotation
    n_active = sum(1 for n in _FL_CLIENT_NODES if n['status'] == 'ACTIVE')
    n_total = len(_FL_CLIENT_NODES)
    fig.add_annotation(
        text=f'ACTIVE CLIENTS: {n_active}/{n_total}',
        x=0.5, y=0.05, xref='paper', yref='paper',
        showarrow=False,
        font=dict(color=TEXT_DIM, family=FONT_MONO, size=9),
    )
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-12 // PRIVACY BUDGET (ε-δ DP)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        height=300, margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _make_fl_convergence(n_rounds=42):
    """VIZ 13: FL convergence curves (global model vs client models)."""
    rounds = list(range(1, n_rounds + 1))
    global_loss = [2.2 * np.exp(-r / 15) + 0.08 for r in rounds]
    global_auroc = [0.50 + 0.44 * (1 - np.exp(-r / 12)) for r in rounds]

    # Per-region divergence from global model (illustrative)
    regions = list(_REGION_COLORS.keys())[:5]
    region_losses = {}
    for k, reg in enumerate(regions):
        offset = 0.05 * (k - 2)
        region_losses[reg] = [
            2.2 * np.exp(-r / (15 + k * 2)) + 0.08 + offset * np.exp(-r / 10)
            for r in rounds
        ]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.08,
                         subplot_titles=['GLOBAL LOSS', 'GLOBAL AUROC'])

    for reg, losses in region_losses.items():
        color = hex_rgba(_REGION_COLORS.get(reg, OFFLINE_GRAY), 0.50)
        fig.add_trace(go.Scatter(x=rounds, y=losses, mode='lines',
                                  name=reg, line=dict(color=color, width=0.8),
                                  showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=rounds, y=global_loss, mode='lines',
                              name='GLOBAL LOSS', line=dict(color=ACCENT_GREEN, width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=rounds, y=global_auroc, mode='lines',
                              name='GLOBAL AUROC', line=dict(color=WARNING_AMBER, width=2)),
                  row=2, col=1)
    fig.add_hline(y=0.95, line_dash='dash', line_color=CRITICAL_RED,
                  annotation_text='TARGET', row=2, col=1)

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-13 // FL CONVERGENCE CURVES',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=380, margin=dict(l=50, r=20, t=40, b=40),
    )
    fig.update_xaxes(gridcolor=BORDER_COLOR, title_text='ROUND', row=2, col=1)
    fig.update_yaxes(gridcolor=BORDER_COLOR, title_text='LOSS', row=1, col=1)
    fig.update_yaxes(gridcolor=BORDER_COLOR, title_text='AUROC', range=[0, 1.05],
                     row=2, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=TEXT_DIM, family=FONT_MONO, size=10))
    return fig


def _make_client_participation_timeline(n_rounds=42):
    """VIZ 14: Client participation Gantt-style timeline."""
    nodes = _FL_CLIENT_NODES
    fig = go.Figure()

    bar_height = 0.6
    y_positions = {n['id']: i for i, n in enumerate(nodes)}

    for node in nodes:
        nid = node['id']
        rounds_done = node.get('rounds', 0)
        color = ACCENT_GREEN if node['status'] == 'ACTIVE' else WARNING_AMBER
        y = y_positions[nid]

        # Each round is 1 unit wide; gaps represent missed rounds
        for r in range(1, n_rounds + 1):
            # Simulate occasional dropouts deterministically
            participated = r <= rounds_done and not (r % 13 == 0 and node['status'] == 'DEGRADED')
            fill = color if participated else BORDER_COLOR
            fig.add_trace(go.Bar(
                x=[1], y=[bar_height],
                base=[r - 1],
                orientation='h',
                marker_color=fill,
                marker_line_width=0,
                yaxis=f'y',
                offsetgroup=str(y),
                showlegend=False,
                hovertemplate=(
                    f"<b>{node['name']}</b><br>Round {r}<br>"
                    f"{'✓ PARTICIPATED' if participated else '✗ ABSENT'}<extra></extra>"
                ),
            ))

    # Legend traces
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=ACCENT_GREEN, name='PARTICIPATED'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=WARNING_AMBER, name='DEGRADED'))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=BORDER_COLOR, name='ABSENT'))

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9),
        title=dict(text='VIZ-14 // CLIENT PARTICIPATION TIMELINE',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='ROUND', gridcolor=BORDER_COLOR, range=[0, n_rounds]),
        yaxis=dict(
            tickvals=list(range(len(nodes))),
            ticktext=[f"{n['id']}" for n in nodes],
            gridcolor=BORDER_COLOR,
        ),
        barmode='overlay',
        bargap=0.0,
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=400, margin=dict(l=100, r=20, t=40, b=50),
    )
    return fig


def _make_gradient_norm_violin():
    """VIZ 15: Gradient norm distribution violin plot per client."""
    nodes = _FL_CLIENT_NODES
    fig = go.Figure()

    for k, node in enumerate(nodes):
        mean_norm = 0.05 + 0.03 * math.sin(k * 1.3)
        std_norm = 0.01 + 0.005 * abs(math.cos(k * 0.9))
        # Generate 200 samples from a log-normal distribution
        rng = np.random.default_rng(seed=k + 42)
        samples = rng.lognormal(mean=math.log(max(mean_norm, 1e-6)),
                                sigma=std_norm * 5, size=200)
        samples = np.clip(samples, 0, 0.5)

        color = ACCENT_GREEN if node['status'] == 'ACTIVE' else WARNING_AMBER
        fig.add_trace(go.Violin(
            y=samples, x=[node['id']] * len(samples),
            name=node['id'],
            box_visible=True, meanline_visible=True,
            line_color=color,
            fillcolor=hex_rgba(color, 0.20),
            opacity=0.85,
        ))

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9),
        title=dict(text='VIZ-15 // GRADIENT NORM DISTRIBUTION (PER CLIENT)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='CLIENT NODE', gridcolor=BORDER_COLOR,
                   tickangle=-35, tickfont=dict(size=8)),
        yaxis=dict(title='||∇L||', gridcolor=BORDER_COLOR),
        violinmode='overlay',
        showlegend=False,
        height=350, margin=dict(l=60, r=20, t=40, b=80),
    )
    return fig


def create_federated_status_layout(client_nodes=None, epsilon_used=0.42, n_rounds=42):
    return html.Div([
        html.Div([
            dcc.Graph(id='viz-11-fl-client-map',
                      figure=_make_fl_client_map(client_nodes),
                      style={'height': '420px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='viz-12-privacy-budget',
                          figure=_make_privacy_budget_gauge(epsilon_used),
                          style={'height': '300px'})
            ], style={**PANEL_STYLE, 'flex': '1', 'marginRight': '8px'}),
            html.Div([
                dcc.Graph(id='viz-13-fl-convergence',
                          figure=_make_fl_convergence(n_rounds),
                          style={'height': '380px'})
            ], style={**PANEL_STYLE, 'flex': '2'}),
        ], style={'display': 'flex', 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-14-client-participation',
                      figure=_make_client_participation_timeline(n_rounds),
                      style={'height': '400px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-15-gradient-norms',
                      figure=_make_gradient_norm_violin(),
                      style={'height': '350px'})
        ], style=PANEL_STYLE),
    ], style={'padding': '8px', 'backgroundColor': BG_PRIMARY})
