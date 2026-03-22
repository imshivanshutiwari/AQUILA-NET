"""Main Operations Center page - VIZ 01-05."""
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
from dashboard.theme import (
    BG_PRIMARY, BG_PANEL, BG_CARD, ACCENT_GREEN, WARNING_AMBER,
    CRITICAL_RED, OFFLINE_GRAY, TEXT_PRIMARY, TEXT_DIM, FONT_MONO,
    BORDER_COLOR, PANEL_STYLE, CARD_STYLE, MAP_STYLE
)


def _make_empty_cable_map(cables=None, landing_points=None):
    """VIZ 01: Global cable map with real TeleGeography data."""
    fig = go.Figure()

    if landing_points:
        lats = [lp.get('lat', 0) for lp in landing_points if lp.get('lat') is not None]
        lons = [lp.get('lon', 0) for lp in landing_points if lp.get('lon') is not None]
        names = [lp.get('name', '') for lp in landing_points if lp.get('lat') is not None]
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode='markers',
            marker=dict(size=5, color=ACCENT_GREEN),
            text=names, name='Landing Points',
            hovertemplate='<b>%{text}</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
        ))

    if cables:
        lp_map = {lp['id']: lp for lp in (landing_points or []) if 'id' in lp}
        for cable in cables[:50]:
            pts = cable.get('landing_points', [])
            if len(pts) >= 2 and lp_map:
                score = cable.get('anomaly_score', 0.0)
                color = ACCENT_GREEN if score < 0.3 else (WARNING_AMBER if score < 0.7 else CRITICAL_RED)
                for i in range(len(pts) - 1):
                    a = lp_map.get(pts[i], {})
                    b = lp_map.get(pts[i + 1], {})
                    if a.get('lat') and b.get('lat'):
                        fig.add_trace(go.Scattermapbox(
                            lat=[a['lat'], b['lat']], lon=[a['lon'], b['lon']],
                            mode='lines',
                            line=dict(color=color, width=1.5),
                            name=cable.get('name', 'Cable'),
                            hoverinfo='text',
                            text=f"{cable.get('name', '')} | Score: {score:.2f}",
                            showlegend=False
                        ))

    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=dict(lat=20, lon=0), zoom=1.5),
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PRIMARY,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text='VIZ-01 // GLOBAL CABLE NETWORK // REAL-TIME',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12),
                   x=0.01),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=450,
    )
    return fig


def _make_anomaly_timeline(history_df=None, cable_name='ALL'):
    """VIZ 02: Anomaly score timeline with confidence intervals."""
    fig = go.Figure()
    if history_df is not None and len(history_df) > 0:
        t = history_df.get('time', pd.RangeIndex(len(history_df)))
        scores = history_df.get('score', [0.5] * len(history_df))
        upper = history_df.get('upper_ci', scores)
        lower = history_df.get('lower_ci', scores)
    else:
        n = 144
        hours = [datetime.datetime.utcnow() - datetime.timedelta(hours=n - i) for i in range(n)]
        t = hours
        base = 0.2
        scores = [base + 0.1 * np.sin(i * np.pi / 24) + 0.05 * np.cos(i * np.pi / 12)
                  for i in range(n)]
        upper = [s + 0.05 for s in scores]
        lower = [max(0.0, s - 0.05) for s in scores]

    fig.add_trace(go.Scatter(
        x=list(t) + list(t)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill='toself', fillcolor='rgba(0,255,68,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=t, y=scores, mode='lines',
        line=dict(color=ACCENT_GREEN, width=1.5),
        name=f'{cable_name} anomaly score'
    ))
    fig.add_hline(y=0.7, line_dash='dash', line_color=CRITICAL_RED,
                  annotation_text='ALERT THRESHOLD', annotation_font_color=CRITICAL_RED)

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-02 // ANOMALY SCORE TIMELINE (24H)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(gridcolor=BORDER_COLOR, title='ZULU TIME'),
        yaxis=dict(gridcolor=BORDER_COLOR, title='ANOMALY SCORE', range=[0, 1.1]),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=300, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _make_attribution_radar(probs=None, cable_name='SELECTED'):
    """VIZ 03: Attribution radar chart."""
    classes = ['SEISMIC', 'ANCHOR', 'AGING', 'SABOTAGE']
    if probs is None:
        probs = [0.1, 0.3, 0.5, 0.1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]], theta=classes + [classes[0]],
        fill='toself', fillcolor='rgba(0,255,68,0.2)',
        line=dict(color=ACCENT_GREEN, width=2),
        name='CURRENT'
    ))
    baseline = [0.25, 0.25, 0.25, 0.25]
    fig.add_trace(go.Scatterpolar(
        r=baseline + [baseline[0]], theta=classes + [classes[0]],
        fill='toself', fillcolor='rgba(255,170,0,0.1)',
        line=dict(color=WARNING_AMBER, width=1, dash='dot'),
        name='BASELINE'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=BG_PANEL,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=BORDER_COLOR,
                            tickfont=dict(color=TEXT_DIM, size=8)),
            angularaxis=dict(tickfont=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10))
        ),
        paper_bgcolor=BG_PRIMARY,
        title=dict(text='VIZ-03 // ATTRIBUTION RADAR',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=300, margin=dict(l=50, r=50, t=40, b=20),
    )
    return fig


def _make_cable_health_matrix(n_cables=20, n_hours=24):
    """VIZ 04: Cable health heatmap."""
    matrix = np.zeros((n_cables, n_hours))
    for i in range(n_cables):
        for j in range(n_hours):
            matrix[i, j] = 0.15 + 0.1 * np.sin((i + j) * np.pi / 12) + 0.05 * np.cos(i * 0.5)
    matrix = np.clip(matrix, 0, 1)

    cable_labels = [f'CLS-{i + 1:02d}' for i in range(n_cables)]
    hour_labels = [
        (datetime.datetime.utcnow() - datetime.timedelta(hours=n_hours - h)).strftime('%HZ')
        for h in range(n_hours)
    ]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=hour_labels, y=cable_labels,
        colorscale=[[0, ACCENT_GREEN], [0.5, WARNING_AMBER], [1, CRITICAL_RED]],
        zmin=0, zmax=1,
        hoverongaps=False,
        hovertemplate='Cable: %{y}<br>Hour: %{x}<br>Score: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9),
        title=dict(text='VIZ-04 // CABLE HEALTH MATRIX (20 CLS × 24H)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='ZULU HOUR', gridcolor=BORDER_COLOR),
        yaxis=dict(title='CABLE NODE'),
        height=400, margin=dict(l=80, r=20, t=40, b=50),
        coloraxis_colorbar=dict(
            title='SCORE', tickfont=dict(color=TEXT_PRIMARY, size=8),
            title_font=dict(color=TEXT_DIM, size=9)
        ),
    )
    return fig


def _make_alert_table():
    """VIZ 05: Alert log table."""
    now = datetime.datetime.utcnow()
    alerts = [
        {
            'TIME': (now - datetime.timedelta(minutes=i * 15)).strftime('%Y-%j-%H:%M:%SZ'),
            'CABLE': f'SEA-ME-WE-{(i % 4) + 3}',
            'SEVERITY': ['ROUTINE', 'PRIORITY', 'IMMEDIATE', 'FLASH'][i % 4],
            'ATTRIBUTION': ['AGING', 'SEISMIC', 'ANCHOR_DRAG', 'SABOTAGE'][i % 4],
            'CONFIDENCE': f'{[0.72, 0.85, 0.91, 0.97][i % 4]:.0%}',
            'STATUS': ['MONITORING', 'INVESTIGATING', 'CONFIRMED', 'CRITICAL'][i % 4],
        }
        for i in range(12)
    ]
    return dash_table.DataTable(
        id='alert-table',
        columns=[{'name': c, 'id': c} for c in alerts[0].keys()],
        data=alerts,
        style_table={'overflowX': 'auto', 'backgroundColor': BG_PANEL},
        style_cell={
            'backgroundColor': BG_PANEL, 'color': TEXT_PRIMARY,
            'fontFamily': FONT_MONO, 'fontSize': '10px',
            'border': f'1px solid {BORDER_COLOR}', 'textAlign': 'left',
            'padding': '4px 8px',
        },
        style_header={
            'backgroundColor': BG_CARD, 'color': ACCENT_GREEN,
            'fontWeight': 'bold', 'letterSpacing': '1px',
        },
        style_data_conditional=[
            {'if': {'filter_query': '{SEVERITY} = "FLASH"'},
             'backgroundColor': '#1a0006', 'color': '#ff0000'},
            {'if': {'filter_query': '{SEVERITY} = "IMMEDIATE"'},
             'backgroundColor': '#1a0800', 'color': '#ff6600'},
            {'if': {'filter_query': '{SEVERITY} = "PRIORITY"'},
             'backgroundColor': '#1a1000', 'color': '#ffaa00'},
        ],
        sort_action='native', filter_action='native',
        page_size=10,
    )


def create_ops_center_layout(cables=None, landing_points=None):
    return html.Div([
        html.Div(id='viz-01-container', children=[
            dcc.Graph(id='viz-01-cable-map',
                      figure=_make_empty_cable_map(cables, landing_points),
                      style={'height': '450px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='viz-02-anomaly-timeline',
                          figure=_make_anomaly_timeline(),
                          style={'height': '300px'})
            ], style={**PANEL_STYLE, 'flex': '2', 'marginRight': '8px'}),
            html.Div([
                dcc.Graph(id='viz-03-attribution-radar',
                          figure=_make_attribution_radar(),
                          style={'height': '300px'})
            ], style={**PANEL_STYLE, 'flex': '1'}),
        ], style={'display': 'flex', 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-04-cable-health-matrix',
                      figure=_make_cable_health_matrix(),
                      style={'height': '400px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            html.H4('VIZ-05 // ALERT LOG', style={
                'color': ACCENT_GREEN, 'fontFamily': FONT_MONO, 'fontSize': '11px',
                'letterSpacing': '2px', 'marginBottom': '8px',
            }),
            html.Div(id='viz-05-alert-table', children=_make_alert_table())
        ], style=PANEL_STYLE),
    ], style={'padding': '8px', 'backgroundColor': BG_PRIMARY})
