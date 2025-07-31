import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
from pathlib import Path
import sys

# Add project root to system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Dash App Setup
# ─────────────────────────────────────────────────────────────────────

def generate_dashboard(test_forecast: dict, config):
    logger.info("Launching dashboard...")

    for model_name, df in test_forecast.items():
        print(f"🔍 Checking {model_name}")
        print(f"🧾 Columns: {df.columns.tolist()}")
        print(f"🧪 Sample row: {df.iloc[0].to_dict()}")
        assert "risk_label" in df.columns, f"❌ 'risk_label' missing for model: {model_name}"

    # Use only the probabilistic framework
    model_name = "probabilistic_framework"
    if model_name not in test_forecast:
        raise ValueError("Missing probabilistic_framework results in test_forecast")

    df = test_forecast[model_name].copy()

    # Convert lab Enterococci to binary SAFE/EXCEED label
    df["true_label"] = np.where(df["Enterococci"] >= 280, "EXCEED", "SAFE")

    # ── Evaluate using risk_label ───────────────────────────────
    evaluator = Evaluator(config)
    classification_metrics = evaluator.evaluate_probabilistic_risk_labels(
        y_true=df["Enterococci"],
        y_pred=df["risk_label"],
        site_names=df["SITE_NAME"]
    )

    # ── Dash Layout ─────────────────────────────────────────────
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Probabilistic Forecast Dashboard"

    app.layout = html.Div([
        html.H1("Recreational Risk Prediction — Dashboard", style={'textAlign': 'center'}),

        html.Div([
            html.Label("Select Site:"),
            dcc.Dropdown(
                id='site-dropdown',
                options=[{'label': s, 'value': s} for s in sorted(df['SITE_NAME'].unique())],
                value=sorted(df['SITE_NAME'].unique())[0],
                style={'width': '60%'}
            )
        ], style={'padding': '10px 40px'}),

        dcc.Tabs(id='tabs', value='tab-metrics', children=[
            dcc.Tab(label='Performance Table', value='tab-metrics'),
            dcc.Tab(label='Quantile Forecast Plot', value='tab-quantile'),
            dcc.Tab(label='Model Comparison Plot', value='tab-comparison')
        ]),

        html.Div(id='tab-content')
    ])

    # ── Classification Metrics Table ───────────────────────────
    def performance_table():
        perf = classification_metrics.copy()
        return dash_table.DataTable(
            id='perf-metrics',
            columns=[{"name": c, "id": c} for c in perf.columns],
            data=perf.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '6px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
        )

    # ── Quantile Forecast Plot ────────────────────────────────
    def quantile_forecast_plot(site):
        site_df = df[df['SITE_NAME'] == site].sort_values("DateTime")

        fig = go.Figure()
        quantile_cols = [c for c in site_df.columns if c.startswith("q_")]

        for q in quantile_cols:
            fig.add_trace(go.Scatter(
                x=site_df['DateTime'],
                y=site_df[q],
                name=q,
                mode='lines',
                line=dict(dash='dot')
            ))

        fig.add_trace(go.Scatter(
            x=site_df['DateTime'],
            y=site_df['Enterococci'],
            name='Lab Measured',
            mode='lines+markers',
            line=dict(color='black')
        ))

        fig.update_layout(
            title=f"Quantile Forecast for {site}",
            xaxis_title="Date",
            yaxis_title="Enterococci (MPN/100 mL)",
            template="plotly_white"
        )
        return fig

    # ── Risk Classification Plot ──────────────────────────────
    def model_comparison_plot(site):
        site_df = df[df['SITE_NAME'] == site].sort_values("DateTime")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=site_df['DateTime'],
            y=site_df['Enterococci'],
            name='Measured Enterococci',
            mode='lines+markers',
            line=dict(color='black')
        ))
        fig.add_trace(go.Scatter(
            x=site_df['DateTime'],
            y=[280]*len(site_df),
            name='280 Threshold',
            line=dict(color='red', dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=site_df['DateTime'],
            y=[140]*len(site_df),
            name='140 Threshold',
            line=dict(color='orange', dash='dot')
        ))

        fig.update_layout(
            title=f"Risk Classification vs Measured Enterococci — {site}",
            xaxis_title="Date",
            yaxis_title="MPN/100 mL",
            template="plotly_white"
        )
        return fig

    # ── Tab Routing ────────────────────────────────────────────
    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'value'),
        Input('site-dropdown', 'value')
    )
    def render_tab(tab, site):
        if tab == 'tab-metrics':
            return performance_table()
        elif tab == 'tab-quantile':
            return dcc.Graph(figure=quantile_forecast_plot(site))
        elif tab == 'tab-comparison':
            return dcc.Graph(figure=model_comparison_plot(site))
        else:
            return html.Div("Invalid tab selection")

    return app
