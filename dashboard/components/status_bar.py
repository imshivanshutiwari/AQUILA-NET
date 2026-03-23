"""Bottom status bar component."""
from dash import html
from dashboard.theme import BG_PRIMARY, ACCENT_GREEN, TEXT_DIM, FONT_MONO, BORDER_COLOR, WARNING_AMBER


def create_status_bar():
    items = [
        ('SYSTEM NOMINAL', ACCENT_GREEN),
        ('DB: CONNECTED', ACCENT_GREEN),
        ('FL SERVER: --/100', ACCENT_GREEN),
        ('ε BUDGET: 0.00/1.00', ACCENT_GREEN),
        ('MODELS: LOADED', ACCENT_GREEN),
        ('DATA: LIVE', ACCENT_GREEN),
    ]
    return html.Div([
        html.Div(id='status-bar-content', children=[
            html.Span(f'[{label}] ', id=f'status-{i}',
                      style={'color': color, 'fontFamily': FONT_MONO,
                             'fontSize': '10px', 'marginRight': '12px',
                             'letterSpacing': '1px'})
            for i, (label, color) in enumerate(items)
        ], style={'display': 'inline-flex', 'flexWrap': 'wrap', 'gap': '4px'}),
    ], style={
        'backgroundColor': BG_PRIMARY,
        'borderTop': f'1px solid {BORDER_COLOR}',
        'padding': '4px 16px',
        'position': 'fixed', 'bottom': 0, 'left': 0, 'right': 0, 'zIndex': 1000,
    })
