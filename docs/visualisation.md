# Visualisation

This document covers the three visualisation tools in the repo: static plots, interactive Plotly plots, and the Dash dashboard.

## Overview

| Tool | Library | Output | Location |
|------|---------|--------|----------|
| Static plots | Matplotlib | PNG images | `results/graphs_plots/` and `results/visualizations/` |
| Interactive plots | Plotly | HTML files | `results/interactive/` |
| Dashboard | Dash (built on Flask + Plotly) | Live web app | `http://localhost:8514` |

Visualisations are generated automatically during the `predict` phase of the pipeline.

## Static Plots (Matplotlib)

**Location:** `src/visualization/forecast_plots.py`

These produce traditional PNG charts. The main plots are:

### Time-Series by Site

For each beach, a line chart showing:
- **Actual Enterococci values** (from lab results)
- **Predicted values** from the model
- **Threshold lines** at 280 (exceedance, red) and 140 (precautionary, orange)
- Y-axis on a log scale (because Enterococci values span several orders of magnitude)

### Prediction Scatter Plot

Predicted vs actual values on a log-log scale. Points close to the diagonal line indicate accurate predictions. Points above the line mean the model over-predicted; points below mean it under-predicted.

### Error Distribution

Histograms showing the distribution of prediction errors:
- Absolute error histogram
- Percentage error histogram

These help identify whether errors are symmetric or skewed.

## Interactive Plots (Plotly)

**Location:** `src/visualization/interactive_plots.py`

These generate standalone HTML files that can be opened in any browser. They offer hover tooltips, zooming, and panning.

### Features

- **Site dropdown** — switch between beaches using a dropdown menu
- **Model traces** — each model framework gets its own coloured line
- **Ground truth** — actual Enterococci values shown as a red line
- **Threshold overlays** — red dotted line at 280 (exceedance), orange dotted at 140 (precautionary)
- **Two-panel layout** — if training predictions are available, the top panel shows training data and the bottom panel shows test data
- **Y-axis cap** — capped at 1,000 MPN/100mL for better readability (extreme outliers would otherwise compress the rest of the plot)

### Colour Scheme

Each model framework has a consistent colour across all visualisations:

| Model | Colour |
|-------|--------|
| Probabilistic Framework | Green (`#5ba966`) |
| LightGBM Benchmark | Pink/Purple (`#c75a93`) |
| Matrix Decomposition | Purple (`#8176cc`) |
| Ground Truth (actual values) | Red |

### How to Use

Open the HTML files in `results/interactive/` in your browser. Use the dropdown at the top to select a site. Hover over data points to see exact values and dates.

## Dash Dashboard

**Location:** `src/visualization/dashboard.py`

The Dash dashboard is a live web application that launches at the end of the predict phase. It provides an interactive exploration interface with filters and dynamic metrics.

### Launching the Dashboard

The dashboard starts automatically when you run:

```bash
python src/pipeline/main_pipeline.py --mode predict
# or
python src/pipeline/main_pipeline.py --mode all
```

It runs on `http://localhost:8514` by default. Open this URL in your browser.

### Controls

The dashboard provides several interactive controls:

| Control | What it does |
|---------|-------------|
| **Site dropdown** | Filter to a specific beach |
| **Plot type** | Switch between "Model Comparison" (all models on one chart) and "Quantile Forecast" (uncertainty bands for the probabilistic model) |
| **Model selector** | Choose which model's quantiles to display (for the quantile forecast view) |

### Model Comparison View

Shows all model predictions overlaid on the same chart with the ground truth. This makes it easy to compare how different frameworks perform at each site.

### Quantile Forecast View

Shows the probabilistic framework's output with:
- **Point prediction** as a solid line
- **Shaded uncertainty bands** between different quantile pairs (e.g. q_0.20 to q_0.975)
- Wider bands indicate more uncertainty in the prediction

### Performance Table

Below the chart, a table displays per-site classification metrics:
- Accuracy
- Sensitivity (recall for exceedances)
- Specificity (recall for safe conditions)
- TP, FP, TN, FN counts
- Log-weighted MAPE (overall, exceedance, safe)

This table updates when you change the site selection.

## Generating Visualisations Without the Full Pipeline

If you want to regenerate visualisations without retraining models, use:

```bash
python src/pipeline/main_pipeline.py --mode predict
```

This loads the saved models from `models/`, generates predictions, and creates all visualisations.

## Adding New Visualisations

If you need to add a new plot type:

1. Add a new function in the appropriate file (`forecast_plots.py` for static, `interactive_plots.py` for Plotly, `dashboard.py` for Dash)
2. Call it from the predict section of `main_pipeline.py`
3. Follow the existing colour scheme for consistency
