"""On-demand analysis callbacks for AQUILA-NET dashboard."""
import numpy as np
from dash import Input, Output, State, callback, no_update
from dashboard.pages.ops_center import (
    _make_anomaly_timeline, _make_attribution_radar, _make_cable_health_matrix
)


def register_analysis_callbacks(app):
    """Register analysis callbacks."""

    @app.callback(
        Output('viz-02-anomaly-timeline', 'figure'),
        Output('viz-03-attribution-radar', 'figure'),
        Input('viz-04-cable-health-matrix', 'clickData'),
        State('store-anomaly-history', 'data'),
        prevent_initial_call=True
    )
    def update_timeline_on_click(click_data, history_data):
        if click_data is None:
            return no_update, no_update
        point = click_data.get('points', [{}])[0]
        cable_name = point.get('y', 'UNKNOWN')
        fig_timeline = _make_anomaly_timeline(cable_name=cable_name)
        fig_radar = _make_attribution_radar(cable_name=cable_name)
        return fig_timeline, fig_radar
