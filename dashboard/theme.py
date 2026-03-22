"""Military dark theme constants for AQUILA-NET dashboard."""

BG_PRIMARY = '#030a04'
BG_PANEL = '#061008'
BG_CARD = '#0a1a0c'
BORDER_COLOR = '#1a3a1e'
ACCENT_GREEN = '#00ff44'
ACCENT_DIM = '#006622'
WARNING_AMBER = '#ffaa00'
CRITICAL_RED = '#ff2244'
OFFLINE_GRAY = '#445544'
TEXT_PRIMARY = '#c8ffc8'
TEXT_DIM = '#448844'
FONT_MONO = 'JetBrains Mono, Courier New, monospace'

SEVERITY_COLORS = {
    'ROUTINE': '#448844',
    'PRIORITY': '#ffaa00',
    'IMMEDIATE': '#ff6600',
    'FLASH': '#ff0000',
}

PANEL_STYLE = {
    'background': BG_PANEL,
    'border': f'1px solid {BORDER_COLOR}',
    'borderLeft': f'3px solid {ACCENT_GREEN}',
    'padding': '12px',
    'fontFamily': FONT_MONO,
}

CARD_STYLE = {
    'backgroundColor': BG_CARD,
    'border': f'1px solid {BORDER_COLOR}',
    'borderRadius': '2px',
    'padding': '10px',
    'marginBottom': '8px',
    'fontFamily': FONT_MONO,
}

MAP_STYLE = 'carto-darkmatter'

def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color string to an rgba() CSS string."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': BG_PRIMARY,
        'plot_bgcolor': BG_PANEL,
        'font': {'color': TEXT_PRIMARY, 'family': FONT_MONO},
        'colorway': [ACCENT_GREEN, WARNING_AMBER, CRITICAL_RED, '#00aaff', '#aa00ff'],
        'xaxis': {'gridcolor': BORDER_COLOR, 'linecolor': BORDER_COLOR},
        'yaxis': {'gridcolor': BORDER_COLOR, 'linecolor': BORDER_COLOR},
        'legend': {'bgcolor': BG_CARD, 'bordercolor': BORDER_COLOR},
    }
}
