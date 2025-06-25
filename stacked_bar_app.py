import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix



# --- Classification function ---
def classify(value):
    if value <= 140:
        return "SAFE"
    elif 141 <= value <= 280:
        return "CAUTION"
    else:
        return "EXCEED"

# --- TP, FP, TN, FN calculator ---
def compute_confusion_components(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics = {cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in classes}

    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        metrics[cls] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
    
    return metrics

# --- Load data ---
predictions_df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions_filtered_test_only.csv")  # Update path if needed
predictions_df["True_Class"] = predictions_df["Enterococci"].apply(classify)
predictions_df["Predicted_Class"] = predictions_df["probabilistic_framework_predictions"].apply(classify)

# --- Constants ---
classes = ["SAFE", "CAUTION", "EXCEED"]
color_map = {"SAFE": "green", "CAUTION": "orange", "EXCEED": "red"}

# --- Overall Metrics ---
y_true = predictions_df["True_Class"]
y_pred = predictions_df["Predicted_Class"]

overall_metrics = compute_confusion_components(y_true, y_pred, classes)
overall_accuracy = accuracy_score(y_true, y_pred)
overall_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
overall_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
overall_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

# --- Overall Stacked Bar ---
grouped = predictions_df.groupby(["True_Class", "Predicted_Class"]).size().reset_index(name="count")
totals = grouped.groupby("True_Class")["count"].transform("sum")
grouped["percentage"] = grouped["count"] / totals * 100

fig1 = px.bar(
    grouped,
    x="True_Class",
    y="count",
    color="Predicted_Class",
    text=grouped["percentage"].apply(lambda x: f"{x:.1f}%"),
    color_discrete_map=color_map,
    category_orders={"True_Class": classes, "Predicted_Class": list(reversed(classes))},
    hover_data=["count", "percentage"]
)

fig1.update_layout(
    barmode="stack",
    title=f"Overall Prediction Breakdown — Acc: {overall_accuracy:.2%}, Prec: {overall_precision:.2%}, Rec: {overall_recall:.2%}, F1: {overall_f1:.2%}",
    xaxis_title="True Class (Lab Result)",
    yaxis_title="Sample Count",
    yaxis=dict(range=[0, 300]),
    legend_title="Predicted Class"
)
fig1.update_traces(textposition="inside")

# --- Streamlit Layout ---
st.title("Three-Class Prediction Performance Dashboard")
st.subheader("All Sites — Model Performance")
st.plotly_chart(fig1, use_container_width=True)

# --- Site-level breakdown ---
st.subheader("Per-Site Performance")
selected_site = st.selectbox("Select site", sorted(predictions_df["SITE_NAME"].unique()))
site_df = predictions_df[predictions_df["SITE_NAME"] == selected_site]

site_y_true = site_df["True_Class"]
site_y_pred = site_df["Predicted_Class"]

site_metrics = compute_confusion_components(site_y_true, site_y_pred, classes)
site_accuracy = accuracy_score(site_y_true, site_y_pred)
site_precision = precision_score(site_y_true, site_y_pred, average="weighted", zero_division=0)
site_recall = recall_score(site_y_true, site_y_pred, average="weighted", zero_division=0)
site_f1 = f1_score(site_y_true, site_y_pred, average="weighted", zero_division=0)

# --- Bar plot for selected site ---
site_grouped = site_df.groupby(["True_Class", "Predicted_Class"]).size().reset_index(name="count")
site_totals = site_grouped.groupby("True_Class")["count"].transform("sum")
site_grouped["percentage"] = site_grouped["count"] / site_totals * 100

fig2 = px.bar(
    site_grouped,
    x="True_Class",
    y="count",
    color="Predicted_Class",
    text=site_grouped["percentage"].apply(lambda x: f"{x:.1f}%"),
    color_discrete_map=color_map,
    category_orders={"True_Class": classes, "Predicted_Class": list(reversed(classes))},
    hover_data=["count", "percentage"]
)

fig2.update_layout(
    barmode="stack",
    title=f"{selected_site} — Acc: {site_accuracy:.2%}, Prec: {site_precision:.2%}, Rec: {site_recall:.2%}, F1: {site_f1:.2%}",
    xaxis_title="True Class (Lab Result)",
    yaxis_title="Sample Count",
    yaxis=dict(range=[0, 300]),
    legend_title="Predicted Class"
)
fig2.update_traces(textposition="inside")

st.plotly_chart(fig2, use_container_width=True)

# --- Optional Display of TP/FP/TN/FN ---
st.markdown("### Site Confusion Components (TP / FP / TN / FN)")
st.json(site_metrics)



#### WORKING STREMLIT CODE

# import pandas as pd
# import plotly.express as px
# import streamlit as st
# from sklearn.metrics import classification_report

# # Load predictions
# predictions_df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions_filtered_test_only.csv")  # Update path if needed

# # Classification function
# def classify(value):
#     if value <= 140:
#         return "SAFE"
#     elif 141 <= value <= 280:
#         return "CAUTION"
#     else:
#         return "EXCEED"

# # Add predicted and true classes
# predictions_df["True_Class"] = predictions_df["Enterococci"].apply(classify)
# predictions_df["Predicted_Class"] = predictions_df["probabilistic_framework_predictions"].apply(classify)

# # --- Overall metrics ---
# overall_report = classification_report(
#     predictions_df["True_Class"],
#     predictions_df["Predicted_Class"],
#     output_dict=True
# )

# overall_accuracy = overall_report["accuracy"]
# overall_precision = overall_report["weighted avg"]["precision"]
# overall_recall = overall_report["weighted avg"]["recall"]
# overall_f1 = overall_report["weighted avg"]["f1-score"]

# # --- Overall grouped data for stacked bar ---
# grouped = predictions_df.groupby(["True_Class", "Predicted_Class"]).size().reset_index(name="count")
# totals = grouped.groupby("True_Class")["count"].transform("sum")
# grouped["percentage"] = grouped["count"] / totals * 100

# fig1 = px.bar(
#     grouped,
#     x="True_Class",
#     y="count",
#     color="Predicted_Class",
#     text=grouped["percentage"].apply(lambda x: f"{x:.1f}%"),
#     color_discrete_map={"SAFE": "green", "CAUTION": "orange", "EXCEED": "red"},
#     category_orders={"True_Class": ["SAFE", "CAUTION", "EXCEED"], "Predicted_Class": ["EXCEED", "CAUTION", "SAFE"]},
#     hover_data=["count", "percentage"]
# )

# fig1.update_layout(
#     barmode="stack",
#     title=f"Overall Prediction Breakdown — Accuracy: {overall_accuracy:.2%}, Precision: {overall_precision:.2%}, Recall: {overall_recall:.2%}, F1: {overall_f1:.2%}",
#     xaxis_title="True Class (Lab Result)",
#     yaxis_title="Sample Count",
#     yaxis=dict(range=[0, 300]),
#     legend_title="Predicted Class"
# )
# fig1.update_traces(textposition="inside")

# # --- Streamlit App ---
# st.title("Enterococci Prediction Breakdown")

# # Overall chart
# st.subheader("All Sites — Model Performance")
# st.plotly_chart(fig1, use_container_width=True)

# # --- Per-site dropdown and chart ---
# site_options = predictions_df["SITE_NAME"].unique()
# selected_site = st.selectbox("Select a site to view detailed prediction accuracy:", site_options)

# site_df = predictions_df[predictions_df["SITE_NAME"] == selected_site]

# site_report = classification_report(
#     site_df["True_Class"],
#     site_df["Predicted_Class"],
#     output_dict=True
# )

# site_accuracy = site_report["accuracy"]
# site_precision = site_report["weighted avg"]["precision"]
# site_recall = site_report["weighted avg"]["recall"]
# site_f1 = site_report["weighted avg"]["f1-score"]

# site_grouped = site_df.groupby(["True_Class", "Predicted_Class"]).size().reset_index(name="count")
# site_totals = site_grouped.groupby("True_Class")["count"].transform("sum")
# site_grouped["percentage"] = site_grouped["count"] / site_totals * 100

# fig2 = px.bar(
#     site_grouped,
#     x="True_Class",
#     y="count",
#     color="Predicted_Class",
#     text=site_grouped["percentage"].apply(lambda x: f"{x:.1f}%"),
#     color_discrete_map={"SAFE": "green", "CAUTION": "orange", "EXCEED": "red"},
#     category_orders={"True_Class": ["SAFE", "CAUTION", "EXCEED"], "Predicted_Class": ["EXCEED", "CAUTION", "SAFE"]},
#     hover_data=["count", "percentage"]
# )

# fig2.update_layout(
#     barmode="stack",
#     title=f"{selected_site} — Accuracy: {site_accuracy:.2%}, Precision: {site_precision:.2%}, Recall: {site_recall:.2%}, F1: {site_f1:.2%}",
#     xaxis_title="True Class (Lab Result)",
#     yaxis_title="Sample Count",
#     yaxis=dict(range=[0, 300]),
#     legend_title="Predicted Class"
# )
# fig2.update_traces(textposition="inside")

# # Display second chart
# st.subheader(f"{selected_site} — Model Performance")
# st.plotly_chart(fig2, use_container_width=True)