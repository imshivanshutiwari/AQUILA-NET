"""AQUILA-NET ops header component."""
from dash import html, dcc
from dashboard.theme import (BG_PRIMARY, ACCENT_GREEN, TEXT_PRIMARY, TEXT_DIM,
                               FONT_MONO, BORDER_COLOR, BG_PANEL, CRITICAL_RED)


def create_header():
    return html.Div([
        html.Div([
            html.Div([
                html.Span('▲ ', style={'color': ACCENT_GREEN, 'fontSize': '20px'}),
                html.Span('AQUILA-NET', style={
                    'color': ACCENT_GREEN, 'fontSize': '22px',
                    'fontWeight': 'bold', 'letterSpacing': '4px'
                }),
                html.Span(' [CLASSIFIED - DEMO]', style={
                    'color': CRITICAL_RED, 'fontSize': '10px',
                    'marginLeft': '8px', 'letterSpacing': '2px'
                }),
                html.Br(),
                html.Span('UNDERSEA CABLE MONITORING SYSTEM v1.0', style={
                    'color': TEXT_DIM, 'fontSize': '10px', 'letterSpacing': '2px'
                }),
            ], style={'flex': '1', 'fontFamily': FONT_MONO}),

            html.Div([
                html.Span('STATUS: ', style={'color': TEXT_DIM, 'fontSize': '11px'}),
                html.Span('● OPERATIONAL', style={'color': ACCENT_GREEN, 'fontSize': '11px'}),
                html.Span(' | ', style={'color': TEXT_DIM}),
                html.Span('CABLES MONITORED: ', style={'color': TEXT_DIM, 'fontSize': '11px'}),
                html.Span(id='header-cable-count', children='--',
                          style={'color': ACCENT_GREEN, 'fontSize': '11px'}),
                html.Span(' | ', style={'color': TEXT_DIM}),
                html.Span('ACTIVE ALERTS: ', style={'color': TEXT_DIM, 'fontSize': '11px'}),
                html.Span(id='header-alert-count', children='0',
                          style={'color': CRITICAL_RED, 'fontSize': '11px'}),
            ], style={'flex': '1', 'textAlign': 'center', 'fontFamily': FONT_MONO}),

            html.Div([
                html.Span('🕐 ZULU TIME: ', style={'color': TEXT_DIM, 'fontSize': '11px'}),
                html.Span(id='header-zulu-time', children='----',
                          style={'color': ACCENT_GREEN, 'fontSize': '12px', 'letterSpacing': '1px'}),
                html.Br(),
                html.Span('● LIVE', style={
                    'color': ACCENT_GREEN, 'fontSize': '11px',
                    'animation': 'blink 1s step-start infinite'
                }),
            ], style={'flex': '1', 'textAlign': 'right', 'fontFamily': FONT_MONO}),
        ], style={
            'display': 'flex', 'alignItems': 'center',
            'backgroundColor': BG_PRIMARY,
            'borderBottom': f'2px solid {ACCENT_GREEN}',
            'padding': '8px 16px',
        }),
    ])
