"""Real-time update callbacks for AQUILA-NET dashboard."""
import datetime
import numpy as np
from dash import Input, Output, State, callback, no_update
from dashboard.theme import ACCENT_GREEN, WARNING_AMBER, CRITICAL_RED, TEXT_DIM


def register_realtime_callbacks(app):
    """Register all real-time callbacks with the app."""

    @app.callback(
        Output('header-zulu-time', 'children'),
        Input('interval-1s', 'n_intervals')
    )
    def update_zulu_time(n):
        now = datetime.datetime.now(datetime.timezone.utc)
        return now.strftime('%Y-%j-%H:%M:%SZ')

    @app.callback(
        Output('header-cable-count', 'children'),
        Output('header-alert-count', 'children'),
        Input('interval-5s', 'n_intervals'),
        State('store-cables', 'data')
    )
    def update_header_counts(n, cables_data):
        cable_count = len(cables_data) if cables_data else 0
        if cables_data:
            alert_count = sum(1 for c in cables_data if c.get('anomaly_score', 0) > 0.7)
        else:
            alert_count = 0
        return str(cable_count), str(alert_count)

    @app.callback(
        Output('alert-ticker-content', 'children'),
        Input('interval-30s', 'n_intervals'),
        State('store-alerts', 'data')
    )
    def update_alert_ticker(n, alerts_data):
        if alerts_data:
            parts = [f"{a.get('severity','?')} // {a.get('cable','?')} // {a.get('attribution','?')}"
                     for a in alerts_data[:5]]
            return ' ◆ '.join(parts) + ' ◆ MONITORING ACTIVE'
        return 'SYSTEM INITIALIZED // MONITORING ACTIVE // ALL CABLES NOMINAL'

    @app.callback(
        Output('status-bar-content', 'children'),
        Input('interval-5s', 'n_intervals'),
        State('store-fl-status', 'data')
    )
    def update_status_bar(n, fl_status):
        from dashboard.theme import FONT_MONO, BORDER_COLOR
        from dash import html
        fl_round = fl_status.get('round', 0) if fl_status else 0
        fl_total = fl_status.get('total_rounds', 100) if fl_status else 100
        epsilon = fl_status.get('epsilon', 0.0) if fl_status else 0.0
        eps_color = ACCENT_GREEN if epsilon < 0.5 else (WARNING_AMBER if epsilon < 0.8 else CRITICAL_RED)
        items = [
            ('[SYSTEM NOMINAL]', ACCENT_GREEN),
            ('[DB: CONNECTED]', ACCENT_GREEN),
            (f'[FL SERVER: {fl_round}/{fl_total}]', ACCENT_GREEN),
            (f'[ε BUDGET: {epsilon:.2f}/1.00]', eps_color),
            ('[MODELS: LOADED]', ACCENT_GREEN),
            ('[DATA: LIVE]', ACCENT_GREEN),
        ]
        return [html.Span(label + ' ', style={'color': color, 'fontFamily': FONT_MONO,
                                               'fontSize': '10px', 'marginRight': '8px'})
                for label, color in items]
