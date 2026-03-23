"""AQUILA-NET Main Dash Application."""
import os
import threading
from dash import Dash, html, dcc, page_registry
import dash_bootstrap_components as dbc

from dashboard.theme import (
    BG_PRIMARY, ACCENT_GREEN, TEXT_PRIMARY, FONT_MONO, BORDER_COLOR
)
from dashboard.components.header import create_header
from dashboard.components.status_bar import create_status_bar
from dashboard.components.alert_ticker import create_alert_ticker
from dashboard.pages.ops_center import create_ops_center_layout
from dashboard.pages.training_monitor import create_training_monitor_layout
from dashboard.pages.federated_status import create_federated_status_layout
from dashboard.pages.analysis_lab import create_analysis_lab_layout
from dashboard.pages.threat_report import create_threat_report_layout
from dashboard.callbacks.realtime_callbacks import register_realtime_callbacks
from dashboard.callbacks.analysis_callbacks import register_analysis_callbacks


_GLOBAL_APP = None  # Module-level app instance for testing


def create_app(config=None, training_mode=False, phase='all'):
    """Create and configure the AQUILA-NET Dash application."""
    global _GLOBAL_APP

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title='AQUILA-NET // UNDERSEA CABLE MONITOR',
        assets_folder=os.path.join(os.path.dirname(__file__), '..', 'assets'),
    )

    # Fetch real cable data for initial render
    cables = []
    landing_points = []
    try:
        from data.telegeography_fetcher import TeleGeographyFetcher
        fetcher = TeleGeographyFetcher()
        cables = fetcher.fetch_cables()
        landing_points = fetcher.fetch_landing_points()
    except Exception:
        cables = []
        landing_points = []

    # Build layout
    app.layout = html.Div(
        id='main-container',
        children=[
            # Intervals
            dcc.Interval(id='interval-1s', interval=1000, n_intervals=0),
            dcc.Interval(id='interval-5s', interval=5000, n_intervals=0),
            dcc.Interval(id='interval-30s', interval=30000, n_intervals=0),

            # Stores
            dcc.Store(id='store-cables', data=cables[:100] if cables else []),
            dcc.Store(id='store-alerts', data=[]),
            dcc.Store(id='store-fl-status', data={'round': 0, 'total_rounds': 100, 'epsilon': 0.0}),
            dcc.Store(id='store-anomaly-history', data={}),
            dcc.Store(id='store-training-metrics', data=[]),

            # Header
            create_header(),

            # Alert ticker
            create_alert_ticker(),

            # Navigation tabs
            dcc.Tabs(
                id='main-tabs',
                value='ops-center',
                children=[
                    dcc.Tab(label='⬛ OPS CENTER', value='ops-center',
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label='⬛ TRAINING', value='training',
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label='⬛ FEDERATED', value='federated',
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label='⬛ ANALYSIS LAB', value='analysis',
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label='⬛ THREAT REPORT', value='threat',
                            style=_tab_style(), selected_style=_tab_selected_style()),
                ],
                style={
                    'backgroundColor': BG_PRIMARY,
                    'borderBottom': f'1px solid {BORDER_COLOR}',
                    'fontFamily': FONT_MONO,
                },
                colors={'border': BORDER_COLOR, 'primary': ACCENT_GREEN, 'background': BG_PRIMARY},
            ),

            # Page content
            html.Div(id='page-content', style={'minHeight': 'calc(100vh - 200px)'}),

            # Status bar
            create_status_bar(),

            # Bottom padding for status bar
            html.Div(style={'height': '35px'}),
        ],
        style={'backgroundColor': BG_PRIMARY, 'minHeight': '100vh', 'fontFamily': FONT_MONO},
    )

    # Register tab routing callback
    from dash import Input, Output

    @app.callback(
        Output('page-content', 'children'),
        Input('main-tabs', 'value'),
    )
    def render_page(tab):
        if tab == 'ops-center':
            return create_ops_center_layout(cables=cables[:50], landing_points=landing_points)
        elif tab == 'training':
            return create_training_monitor_layout()
        elif tab == 'federated':
            return create_federated_status_layout()
        elif tab == 'analysis':
            return create_analysis_lab_layout()
        elif tab == 'threat':
            return create_threat_report_layout()
        return html.Div('PAGE NOT FOUND', style={'color': ACCENT_GREEN, 'fontFamily': FONT_MONO})

    # Register callbacks
    register_realtime_callbacks(app)
    register_analysis_callbacks(app)

    _GLOBAL_APP = app
    return app


def _tab_style():
    return {
        'backgroundColor': BG_PRIMARY,
        'color': '#448844',
        'border': f'1px solid {BORDER_COLOR}',
        'fontFamily': FONT_MONO,
        'fontSize': '10px',
        'letterSpacing': '1px',
        'padding': '6px 12px',
    }


def _tab_selected_style():
    return {
        'backgroundColor': '#0a1a0c',
        'color': ACCENT_GREEN,
        'border': f'1px solid {ACCENT_GREEN}',
        'borderBottom': 'none',
        'fontFamily': FONT_MONO,
        'fontSize': '10px',
        'letterSpacing': '1px',
        'padding': '6px 12px',
    }


class AQUILADashboard:
    """Wrapper class for the AQUILA-NET dashboard application."""

    def __init__(self, config=None):
        self.config = config or {}
        self.app = create_app(config=config)

    def run(self, host='0.0.0.0', port=8050, debug=False):
        self.app.run(host=host, port=port, debug=debug)

    @property
    def server(self):
        return self.app.server

    def get_layout(self):
        return self.app.layout


if __name__ == '__main__':
    import webbrowser
    import threading
    import time

    app = create_app()

    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:8050')

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()
    app.run(debug=False, host='0.0.0.0', port=8050)
