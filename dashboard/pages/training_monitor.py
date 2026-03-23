"""Training Monitor page - VIZ 06-10."""
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
from dashboard.theme import (
    BG_PRIMARY, BG_PANEL, BG_CARD, ACCENT_GREEN, WARNING_AMBER,
    CRITICAL_RED, TEXT_PRIMARY, TEXT_DIM, FONT_MONO, BORDER_COLOR, PANEL_STYLE,
    hex_rgba
)


def _make_loss_stacked_area(metrics_history=None):
    """VIZ 06: Loss components stacked area chart."""
    if metrics_history and len(metrics_history) > 0:
        epochs = [m.get('epoch', i) for i, m in enumerate(metrics_history)]
        l_total = [m.get('total', 0.0) for m in metrics_history]
        l_anom = [m.get('anomaly', 0.0) for m in metrics_history]
        l_tele = [m.get('telegrapher', 0.0) for m in metrics_history]
        l_eb = [m.get('euler_bernoulli', 0.0) for m in metrics_history]
        l_caus = [m.get('causality', 0.0) for m in metrics_history]
    else:
        n = 50
        epochs = list(range(1, n + 1))
        decay = [np.exp(-e / 20) for e in epochs]
        l_total = [2.5 * d for d in decay]
        l_anom = [1.0 * d for d in decay]
        l_tele = [0.6 * d for d in decay]
        l_eb = [0.5 * d for d in decay]
        l_caus = [0.4 * d for d in decay]

    fig = go.Figure()
    series = [
        (l_total, ACCENT_GREEN, 'L_total'),
        (l_anom, WARNING_AMBER, 'L_anomaly'),
        (l_tele, '#00aaff', 'L_telegrapher'),
        (l_eb, '#aa44ff', 'L_euler_bernoulli'),
        (l_caus, CRITICAL_RED, 'L_causality'),
    ]
    for vals, color, label in series:
        # Build a semi-transparent fill color by appending alpha to hex
        fill_color = hex_rgba(color, 0.30)
        fig.add_trace(go.Scatter(
            x=epochs, y=vals, mode='lines', stackgroup='loss',
            name=label, line=dict(color=color, width=1),
            fillcolor=fill_color,
        ))
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-06 // LOSS COMPONENTS (STACKED)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='EPOCH', gridcolor=BORDER_COLOR),
        yaxis=dict(title='LOSS', gridcolor=BORDER_COLOR),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=300, margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _make_auroc_f1_chart(metrics_history=None):
    """VIZ 07: AUROC + F1 dual axis chart."""
    if metrics_history and len(metrics_history) > 0:
        epochs = [m.get('epoch', i) for i, m in enumerate(metrics_history)]
        train_auroc = [m.get('train_auroc', 0.5) for m in metrics_history]
        val_auroc = [m.get('val_auroc', 0.5) for m in metrics_history]
        train_f1 = [m.get('train_f1', 0.25) for m in metrics_history]
        val_f1 = [m.get('val_f1', 0.25) for m in metrics_history]
    else:
        n = 50
        epochs = list(range(1, n + 1))
        train_auroc = [0.5 + 0.45 * (1 - np.exp(-e / 15)) for e in epochs]
        val_auroc = [0.5 + 0.40 * (1 - np.exp(-e / 15)) for e in epochs]
        train_f1 = [0.2 + 0.65 * (1 - np.exp(-e / 20)) for e in epochs]
        val_f1 = [0.2 + 0.58 * (1 - np.exp(-e / 20)) for e in epochs]

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=epochs, y=train_auroc, mode='lines',
                              name='Train AUROC',
                              line=dict(color=ACCENT_GREEN, width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=epochs, y=val_auroc, mode='lines',
                              name='Val AUROC',
                              line=dict(color=ACCENT_GREEN, width=1, dash='dash')),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=epochs, y=train_f1, mode='lines',
                              name='Train F1',
                              line=dict(color=WARNING_AMBER, width=2)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=epochs, y=val_f1, mode='lines',
                              name='Val F1',
                              line=dict(color=WARNING_AMBER, width=1, dash='dash')),
                  secondary_y=True)
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-07 // AUROC + F1 DUAL-AXIS',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='EPOCH', gridcolor=BORDER_COLOR),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=300, margin=dict(l=50, r=50, t=40, b=40),
    )
    fig.update_yaxes(title_text='AUROC', gridcolor=BORDER_COLOR, secondary_y=False)
    fig.update_yaxes(title_text='F1', gridcolor=BORDER_COLOR, secondary_y=True)
    return fig


def _make_confusion_matrix():
    """VIZ 08: Confusion matrix heatmap."""
    classes = ['SEISMIC', 'ANCHOR', 'AGING', 'SABOTAGE']
    cm = np.array([
        [0.82, 0.08, 0.06, 0.04],
        [0.06, 0.78, 0.10, 0.06],
        [0.04, 0.09, 0.81, 0.06],
        [0.03, 0.05, 0.04, 0.88],
    ])
    fig = go.Figure(go.Heatmap(
        z=cm, x=classes, y=classes,
        colorscale=[[0, BG_PANEL], [1, ACCENT_GREEN]],
        text=[[f'{v:.2f}' for v in row] for row in cm],
        texttemplate='%{text}', textfont=dict(color=TEXT_PRIMARY, size=11),
        hoverongaps=False,
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-08 // CONFUSION MATRIX (NORMALIZED)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='PREDICTED', gridcolor=BORDER_COLOR),
        yaxis=dict(title='ACTUAL', gridcolor=BORDER_COLOR),
        height=350, margin=dict(l=80, r=20, t=40, b=60),
    )
    return fig


def _make_pde_residuals(metrics_history=None):
    """VIZ 09: PDE residual magnitudes bar chart."""
    if metrics_history and len(metrics_history) > 0:
        n = min(len(metrics_history), 20)
        epochs = [metrics_history[i].get('epoch', i) for i in range(n)]
        l_tele = [max(1e-8, metrics_history[i].get('telegrapher', 0.1)) for i in range(n)]
        l_eb = [max(1e-8, metrics_history[i].get('euler_bernoulli', 0.08)) for i in range(n)]
        l_caus = [max(1e-8, metrics_history[i].get('causality', 0.06)) for i in range(n)]
    else:
        n = 10
        epochs = list(range(1, n + 1))
        l_tele = [0.8 * np.exp(-e / 8) + 0.01 for e in epochs]
        l_eb = [0.6 * np.exp(-e / 8) + 0.008 for e in epochs]
        l_caus = [0.5 * np.exp(-e / 8) + 0.005 for e in epochs]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=epochs, y=l_tele, name='L_telegrapher',
                          marker_color=ACCENT_GREEN))
    fig.add_trace(go.Bar(x=epochs, y=l_eb, name='L_euler_bernoulli',
                          marker_color=WARNING_AMBER))
    fig.add_trace(go.Bar(x=epochs, y=l_caus, name='L_causality',
                          marker_color='#00aaff'))
    fig.add_hline(y=0.001, line_dash='dash', line_color=CRITICAL_RED,
                  annotation_text='TARGET RESIDUAL', annotation_font_color=CRITICAL_RED)
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-09 // PDE RESIDUAL MAGNITUDES',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        barmode='group',
        xaxis=dict(title='EPOCH', gridcolor=BORDER_COLOR),
        yaxis=dict(title='RESIDUAL', type='log', gridcolor=BORDER_COLOR),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=300, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _make_lr_schedule(n_epochs=50, current_epoch=0):
    """VIZ 10: Learning rate schedule (cosine annealing)."""
    epochs = list(range(1, n_epochs + 1))
    lr_base = 1e-4
    lrs = [lr_base * 0.5 * (1 + math.cos(math.pi * e / n_epochs)) for e in epochs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=lrs, mode='lines',
        name='LR Schedule',
        line=dict(color=ACCENT_GREEN, width=2)
    ))
    if 0 <= current_epoch < len(lrs):
        fig.add_trace(go.Scatter(
            x=[current_epoch + 1], y=[lrs[current_epoch]],
            mode='markers',
            marker=dict(size=10, color=CRITICAL_RED, symbol='circle'),
            name='CURRENT EPOCH'
        ))
    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=10),
        title=dict(text='VIZ-10 // LR SCHEDULE (COSINE ANNEALING)',
                   font=dict(color=ACCENT_GREEN, family=FONT_MONO, size=12), x=0.01),
        xaxis=dict(title='EPOCH', gridcolor=BORDER_COLOR),
        yaxis=dict(title='LEARNING RATE', type='log', gridcolor=BORDER_COLOR),
        legend=dict(bgcolor=BG_CARD, bordercolor=BORDER_COLOR,
                    font=dict(color=TEXT_PRIMARY, family=FONT_MONO, size=9)),
        height=250, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_training_monitor_layout(metrics_history=None, current_epoch=0):
    return html.Div([
        html.Div([
            dcc.Graph(id='viz-06-loss-stacked',
                      figure=_make_loss_stacked_area(metrics_history),
                      style={'height': '300px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-07-auroc-f1',
                      figure=_make_auroc_f1_chart(metrics_history),
                      style={'height': '300px'})
        ], style={**PANEL_STYLE, 'marginBottom': '8px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='viz-08-confusion-matrix',
                          figure=_make_confusion_matrix(),
                          style={'height': '350px'})
            ], style={**PANEL_STYLE, 'flex': '1', 'marginRight': '8px'}),
            html.Div([
                dcc.Graph(id='viz-09-pde-residuals',
                          figure=_make_pde_residuals(metrics_history),
                          style={'height': '300px'})
            ], style={**PANEL_STYLE, 'flex': '1'}),
        ], style={'display': 'flex', 'marginBottom': '8px'}),

        html.Div([
            dcc.Graph(id='viz-10-lr-schedule',
                      figure=_make_lr_schedule(current_epoch=current_epoch),
                      style={'height': '250px'})
        ], style=PANEL_STYLE),
    ], style={'padding': '8px', 'backgroundColor': BG_PRIMARY})
