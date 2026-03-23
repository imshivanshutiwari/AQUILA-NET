"""Tests for the Dash dashboard - 2 tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dash_app_creates_layout():
    from dashboard.app import AQUILADashboard
    dashboard = AQUILADashboard()
    assert dashboard.server is not None
    assert dashboard.app.layout is not None


def test_page_layouts_contain_viz_ids():
    from dashboard.pages.ops_center import create_ops_center_layout
    from dashboard.pages.training_monitor import create_training_monitor_layout
    ops = str(create_ops_center_layout())
    train = str(create_training_monitor_layout())
    combined = ops + train
    page1_ids = ["viz-01-cable-map", "viz-02-anomaly-timeline", "viz-03-attribution-radar",
                 "viz-04-cable-health-matrix", "viz-05-alert-table"]
    page2_ids = ["viz-06-loss-stacked", "viz-07-auroc-f1", "viz-08-confusion-matrix",
                 "viz-09-pde-residuals", "viz-10-lr-schedule"]
    found1 = sum(1 for v in page1_ids if v in combined)
    found2 = sum(1 for v in page2_ids if v in combined)
    assert found1 >= 4, f"Ops-center needs ≥4 IDs, found {found1}"
    assert found2 >= 4, f"Training-monitor needs ≥4 IDs, found {found2}"
