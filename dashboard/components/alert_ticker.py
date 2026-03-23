"""Scrolling alert ticker component."""
from dash import html
from dashboard.theme import BG_PRIMARY, ACCENT_GREEN, TEXT_DIM, FONT_MONO, CRITICAL_RED, WARNING_AMBER


def create_alert_ticker():
    return html.Div([
        html.Span('◀◀ ', style={'color': ACCENT_GREEN}),
        html.Span(id='alert-ticker-content',
                  children='SYSTEM INITIALIZED // MONITORING ACTIVE // ALL CABLES NOMINAL',
                  style={'color': ACCENT_GREEN, 'fontFamily': FONT_MONO, 'fontSize': '11px',
                         'letterSpacing': '1px', 'animation': 'ticker 30s linear infinite'}),
    ], style={
        'backgroundColor': '#040d05',
        'borderTop': '1px solid #1a3a1e',
        'borderBottom': '1px solid #1a3a1e',
        'padding': '4px 16px',
        'overflow': 'hidden',
        'whiteSpace': 'nowrap',
    })
