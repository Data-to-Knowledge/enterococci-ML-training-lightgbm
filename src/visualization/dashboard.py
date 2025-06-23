# dashboard.py

from pathlib import Path
import sys
from functools import reduce
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

# Path setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Thresholds
SAFE_THRESHOLD = 140
EXCEEDANCE_THRESHOLD = 280
CLASS_LABELS = ["SAFE", "PRECAUTIONARY", "EXCEEDANCE"]


def classify_enterococci(value):
    if value < SAFE_THRESHOLD:
        return "SAFE"
    elif SAFE_THRESHOLD <= value <= EXCEEDANCE_THRESHOLD:
        return "PRECAUTIONARY"
    else:
        return "EXCEEDANCE"


def generate_dashboard(forecast):
    logger.info("Generating 3-class dashboard...")

    first_df = next(iter(forecast.values()))
    common_cols = [col for col in first_df.columns if col != "predictions" and not col.startswith("q_")]

    dfs = []
    for model, df in forecast.items():
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={"predictions": model})
        quantile_cols = [col for col in df_copy.columns if col.startswith("q_")]
        dfs.append(df_copy[common_cols + quantile_cols + [model]])

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_cols, how='inner'), dfs)
    data = merged_df.copy()

    model_names = list(forecast.keys())
    performance_tables = {model: create_performance_table(data, model) for model in model_names}

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div([
        html.H1("Three-Class Forecast Dashboard", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Site"),
                dcc.Dropdown(
                    id='site-dropdown',
                    options=[{'label': s, 'value': s} for s in data['SITE_NAME'].unique()],
                    value=data['SITE_NAME'].unique()[0]
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Plot Type"),
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': 'Model Comparison', 'value': 'model_comparison'},
                        {'label': 'Quantile Forecast', 'value': 'quantile_forecast'}
                    ],
                    value='model_comparison'
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Model"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': m, 'value': m} for m in model_names],
                    value=model_names[0]
                )
            ], style={'width': '30%', 'display': 'inline-block'})
        ]),
        dcc.Graph(id='forecast-graph'),
        html.H2("Performance Metrics", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='performance-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[{
                'if': {'row_index': 0},
                'backgroundColor': 'rgb(240, 240, 240)', 'fontWeight': 'bold'
            }]
        )
    ])

    @app.callback(
        Output('forecast-graph', 'figure'),
        [Input('site-dropdown', 'value'),
         Input('plot-type-dropdown', 'value'),
         Input('model-dropdown', 'value')]
    )
    def update_graph(selected_site, selected_plot_type, selected_model):
        site_data = data[data["SITE_NAME"] == selected_site].sort_values("DateTime")
        x_vals = list(range(len(site_data)))
        datetimes = site_data["DateTime"].dt.strftime('%Y-%m-%d %H:%M')

        fig = go.Figure()

        if selected_plot_type == "model_comparison":
            # Add actual
            if "Enterococci" in site_data.columns:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=site_data["Enterococci"],
                    mode='lines', name="Actual",
                    line=dict(color='red', width=3),
                    hovertext=datetimes
                ))

            for i, model in enumerate(model_names):
                if model in site_data.columns:
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=site_data[model],
                        mode='lines', name=model,
                        line=dict(width=2),
                        hovertext=datetimes
                    ))

        elif selected_plot_type == "quantile_forecast":
            lower_col = next((c for c in site_data.columns if c.startswith("q_0.1")), None)
            upper_col = next((c for c in site_data.columns if c.startswith("q_0.9")), None)
            fig.add_trace(go.Scatter(
                x=x_vals, y=site_data[selected_model],
                mode='lines', name=f"{selected_model} prediction",
                line=dict(color='blue', width=3)
            ))
            if lower_col and upper_col:
                fig.add_trace(go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=site_data[upper_col].tolist() + site_data[lower_col][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name="90% Interval",
                    showlegend=True
                ))

        fig.add_trace(go.Scatter(
            x=x_vals, y=[SAFE_THRESHOLD]*len(x_vals), mode='lines', name="SAFE threshold",
            line=dict(dash='dot', color='green')
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=[EXCEEDANCE_THRESHOLD]*len(x_vals), mode='lines', name="EXCEEDANCE threshold",
            line=dict(dash='dot', color='red')
        ))

        fig.update_layout(
            title=f"{selected_plot_type.replace('_', ' ').title()} - {selected_site}",
            xaxis_title="Time",
            yaxis_title="Enterococci (MPN/100mL)",
            hovermode='x unified',
            template='plotly_white'
        )
        return fig

    @app.callback(
        Output('performance-table', 'data'),
        Output('performance-table', 'columns'),
        Input('model-dropdown', 'value')
    )
    def update_performance_table(selected_model):
        df = performance_tables[selected_model]
        return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

    return app


def create_performance_table(df, model_col):
    df = df.copy()
    df["True_Class"] = df["Enterococci"].apply(classify_enterococci)
    df["Pred_Class"] = df[model_col].apply(classify_enterococci)

    overall = classification_report(df["True_Class"], df["Pred_Class"], labels=CLASS_LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(df["True_Class"], df["Pred_Class"], labels=CLASS_LABELS)

    perf_data = []
    overall_row = {
        "SITE_NAME": "OVERALL",
        "Accuracy": round(overall["accuracy"], 3),
        "F1_SAFE": round(overall["SAFE"]["f1-score"], 3),
        "F1_PRECAUTIONARY": round(overall["PRECAUTIONARY"]["f1-score"], 3),
        "F1_EXCEEDANCE": round(overall["EXCEEDANCE"]["f1-score"], 3),
        "Confusion_Matrix": str(matrix.tolist())
    }
    perf_data.append(overall_row)

    for site in df["SITE_NAME"].unique():
        site_df = df[df["SITE_NAME"] == site]
        site_report = classification_report(site_df["True_Class"], site_df["Pred_Class"], labels=CLASS_LABELS, output_dict=True, zero_division=0)
        site_matrix = confusion_matrix(site_df["True_Class"], site_df["Pred_Class"], labels=CLASS_LABELS)

        row = {
            "SITE_NAME": site,
            "Accuracy": round(site_report["accuracy"], 3),
            "F1_SAFE": round(site_report["SAFE"]["f1-score"], 3),
            "F1_PRECAUTIONARY": round(site_report["PRECAUTIONARY"]["f1-score"], 3),
            "F1_EXCEEDANCE": round(site_report["EXCEEDANCE"]["f1-score"], 3),
            "Confusion_Matrix": str(site_matrix.tolist())
        }
        perf_data.append(row)

    return pd.DataFrame(perf_data)
