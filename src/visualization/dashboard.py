from pathlib import Path
import sys
from functools import reduce
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from sklearn.metrics import confusion_matrix

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

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

def create_performance_table(df, model_col):
    df = df.copy()
    df["True_Class"] = df["Enterococci"].apply(classify_enterococci)
    df["Pred_Class"] = df[model_col].apply(classify_enterococci)

    def calculate_metrics_per_class(y_true, y_pred, target_class):
        labels = CLASS_LABELS
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        idx = labels.index(target_class)

        TP = matrix[idx, idx]
        FP = matrix[:, idx].sum() - TP
        FN = matrix[idx, :].sum() - TP
        TN = matrix.sum() - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return TP, FP, FN, TN, round(f1, 3)

    perf_rows = []
    for site in ["OVERALL"] + sorted(df["SITE_NAME"].unique()):
        subset = df if site == "OVERALL" else df[df["SITE_NAME"] == site]
        row = {"SITE_NAME": site}
        for cls in CLASS_LABELS:
            TP, FP, FN, TN, f1 = calculate_metrics_per_class(subset["True_Class"], subset["Pred_Class"], cls)
            row[f"{cls}_TP"] = TP
            row[f"{cls}_FP"] = FP
            row[f"{cls}_FN"] = FN
            row[f"{cls}_TN"] = TN
            row[f"{cls}_F1"] = f1
        perf_rows.append(row)

    return pd.DataFrame(perf_rows)

def create_stacked_bar_data(df, model_col):
    df = df.copy()
    df["True_Class"] = df["Enterococci"].apply(classify_enterococci)
    df["Pred_Class"] = df[model_col].apply(classify_enterococci)
    return df.groupby(["True_Class", "Pred_Class"]).size().reset_index(name="count")


def flatten_confusion_and_metrics(perf_df):
    records = []
    for _, row in perf_df.iterrows():
        site = row['SITE_NAME']
        for cls in CLASS_LABELS:
            record = {
                "SITE_NAME": site,
                "CLASS": cls,
                "TP": row.get(f"{cls}_TP", 0),
                "FP": row.get(f"{cls}_FP", 0),
                "FN": row.get(f"{cls}_FN", 0),
                "TN": row.get(f"{cls}_TN", 0),
                "F1": row.get(f"{cls}_F1", 0)
            }
            tp, fp, fn = record["TP"], record["FP"], record["FN"]
            record["Precision"] = tp / (tp + fp) if (tp + fp) else 0
            record["Recall"] = tp / (tp + fn) if (tp + fn) else 0
            record["Accuracy"] = (tp + record["TN"]) / (tp + fp + fn + record["TN"]) if (tp + fp + fn + record["TN"]) else 0
            records.append(record)
    return pd.DataFrame(records)


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
    performance_tables = {}

    for model in model_names:
        perf_df = create_performance_table(data, model)
        performance_tables[model] = perf_df

    for model in model_names:
        perf_df = create_performance_table(data, model)
        performance_tables[model] = perf_df

        # 📤 Export CSV per model
        flat_df = flatten_confusion_and_metrics(perf_df)
        out_path = Path("outputs") / f"{model}_threeclass_confusion_metrics.csv"
        out_path.parent.mkdir(exist_ok=True)
        flat_df.to_csv(out_path, index=False)
        logger.info(f"Saved confusion metrics CSV to {out_path}")


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
        html.Br(),
        html.H2("Performance Table", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='performance-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            page_size=30
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
            if "Enterococci" in site_data.columns:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=site_data["Enterococci"],
                    mode='lines', name="Actual",
                    line=dict(color='red', width=3),
                    hovertext=datetimes
                ))

            for model in model_names:
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
        [Output('performance-table', 'data'),
         Output('performance-table', 'columns')],
        [Input('model-dropdown', 'value')]
    )
    def update_performance_table(model):
        df = performance_tables[model]
        columns = [{"name": i, "id": i} for i in df.columns]
        return df.to_dict('records'), columns

    return app
