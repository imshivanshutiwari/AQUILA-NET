"""Severity badge component."""
from dash import html
from dashboard.theme import SEVERITY_COLORS, FONT_MONO


def create_severity_badge(severity: str):
    color = SEVERITY_COLORS.get(severity, '#ffffff')
    return html.Span(severity, style={
        'backgroundColor': color, 'color': '#000000',
        'padding': '2px 6px', 'borderRadius': '2px',
        'fontSize': '9px', 'fontWeight': 'bold',
        'fontFamily': FONT_MONO, 'letterSpacing': '1px',
    })


def get_severity_for_score(score: float) -> str:
    if score < 0.3:
        return 'ROUTINE'
    elif score < 0.5:
        return 'PRIORITY'
    elif score < 0.7:
        return 'IMMEDIATE'
    return 'FLASH'
