import pandas as pd
import altair as alt
from sklearn.metrics import classification_report

# Load predictions CSV
df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions.csv")

# Categorise into three classes
def classify(val):
    if val < 140:
        return "SAFE"
    elif 141 <= val <= 280:
        return "CAUTION"
    else:
        return "EXCEED"

# df["True_Class"] = df["Enterococci"].apply(classify)
# df["Predicted_Class"] = df["probabilistic_framework_predictions"].apply(classify)

# # Aggregate counts
# grouped = (
#     df.groupby(["True_Class", "Predicted_Class"])
#     .size()
#     .reset_index(name="count")
# )

# # Sort categories for visual order
# class_order = ["SAFE", "CAUTION", "EXCEED"]
# grouped["True_Class"] = pd.Categorical(grouped["True_Class"], categories=class_order, ordered=True)
# grouped["Predicted_Class"] = pd.Categorical(grouped["Predicted_Class"], categories=class_order, ordered=True)

# # Cap values above 500 for better chart appearance
# grouped["count_capped"] = grouped["count"].clip(upper=500)

# # Plot with clipped counts and controlled scale
# chart = alt.Chart(grouped).mark_bar(size=40).encode(
#     x=alt.X("True_Class:N", title="Actual Class (Lab Result)"),
#     y=alt.Y("count_capped:Q", title="Sample Count (Capped)", scale=alt.Scale(domain=[0, 500])),
#     color=alt.Color("Predicted_Class:N", scale=alt.Scale(domain=class_order, range=["green", "orange", "red"])),
#     tooltip=["True_Class", "Predicted_Class", "count"],
#     order=alt.Order("Predicted_Class", sort="ascending")
# ).properties(
#     title="Stacked Bar Chart (Capped at 500): Predictions by Actual Lab Class",
#     width=500,
#     height=400
# )

# # Save chart
# chart.save(r"C:\atman_python\Dont-Swim-in-Data\results\graphs_plots\stacked_bar_predictions.png")


# --- Apply Class Labels ---
df["True_Class"] = df["Enterococci"].apply(classify)
df["Predicted_Class"] = df["probabilistic_framework_predictions"].apply(classify)

# --- Container for site-level metrics ---
site_metrics = []

# --- Get all sites (or use one if not available) ---
site_col = "SITE_NAME" if "SITE_NAME" in df.columns else None
site_list = df[site_col].unique() if site_col else ["OVERALL"]

# --- Loop over sites or single overall case ---
for site in site_list:
    subset = df if site == "OVERALL" else df[df[site_col] == site]
    report = classification_report(
        subset["True_Class"],
        subset["Predicted_Class"],
        output_dict=True,
        zero_division=0
    )
    
    for cls in ["SAFE", "CAUTION", "EXCEED"]:
        if cls in report:
            metrics = {
                "SITE": site,
                "CLASS": cls,
                "Precision": round(report[cls]["precision"], 3),
                "Recall": round(report[cls]["recall"], 3),
                "F1-Score": round(report[cls]["f1-score"], 3),
                "Support": int(report[cls]["support"])
            }
            site_metrics.append(metrics)

# --- Convert to DataFrame and Save ---
metrics_df = pd.DataFrame(site_metrics)
metrics_df.to_csv(r"C:\atman_python\Dont-Swim-in-Data\results\graphs_plots\classification_metrics_per_site.csv", index=False)
print("Saved site-level classification metrics to classification_metrics_per_site.csv")