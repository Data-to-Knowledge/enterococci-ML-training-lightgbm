## GROUPED BAR GRAPH (ONE SITE CLASS)


import pandas as pd
import altair as alt

# Read data and filter out "OVERALL"
df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\outputs\probabilistic_framework_threeclass_confusion_metrics.csv")
df = df[df['SITE_NAME'] != 'OVERALL']

metrics = ['TP', 'FP', 'TN', 'FN']
class_order = ["SAFE", "PRECAUTIONARY", "EXCEEDANCE"]

# Rename metrics for clear legend entries
metric_labels = {
    'TP': 'TP (correct)',
    'FP': 'FP (false alarm)',
    'TN': 'TN (correct reject)',
    'FN': 'FN (missed)'
}

def bar_chart_for_class(selected_class):
    data = df[df['CLASS'] == selected_class]
    data_long = data.melt(
        id_vars=['SITE_NAME', 'CLASS'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )
    data_long['Metric'] = data_long['Metric'].map(metric_labels)

    return alt.Chart(data_long).mark_bar().encode(
        x=alt.X('SITE_NAME:N', title='Site', sort=alt.EncodingSortField(field="SITE_NAME")),
        y=alt.Y('Value:Q', title='Count'),
        color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10'), title="Metric"),
        tooltip=['SITE_NAME', 'Metric', 'Value']
    ).properties(
        width=600,
        height=350,
        title=f"{selected_class}: Confusion Matrix Counts by Site"
    ).configure_axisX(
        labelAngle=-45
    )

# Explanatory markdown you can include in your report or dashboard:
legend_explanation = """
**Legend: Confusion Matrix Metrics**

- **TP (correct):** Correctly predicted this class  
- **FP (false alarm):** Predicted this class, but it was actually another  
- **TN (correct reject):** Correctly predicted another class  
- **FN (missed):** Missed this class when it should have predicted it  
"""

print(legend_explanation)

# Generate and save a chart per class
for cls in class_order:
    chart = bar_chart_for_class(cls)
    chart.save(fr"C:\atman_python\Dont-Swim-in-Data\outputs\charts_graphs\{cls}_confusion_grouped_bar.svg")
    chart.save(fr"C:\atman_python\Dont-Swim-in-Data\outputs\charts_graphs\{cls}_confusion_grouped_bar.png", scale_factor=3)

print("Charts saved for each class with clear legend labels (excluding OVERALL).")





## HEATMAP

# import pandas as pd
# import altair as alt

# df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\outputs\probabilistic_framework_threeclass_confusion_metrics.csv")
# metrics = ['TP', 'FP', 'TN', 'FN']
# class_order = ["SAFE", "PRECAUTIONARY", "EXCEEDANCE"]

# for metric in metrics:
#     heatmap = alt.Chart(df).mark_rect().encode(
#         x=alt.X('SITE_NAME:N', title='Site', sort=alt.EncodingSortField(field="SITE_NAME")),
#         y=alt.Y('CLASS:N', title='Class', sort=class_order),
#         color=alt.Color(f'{metric}:Q', title=metric, scale=alt.Scale(scheme='viridis')),
#         tooltip=['SITE_NAME', 'CLASS', f'{metric}']
#     ).properties(
#         width=600,
#         height=200,
#         title=f"{metric} Heatmap by Site and Class"
#     )

#     heatmap.save(fr"C:\atman_python\Dont-Swim-in-Data\outputs\charts_graphs\{metric}_heatmap.svg")
#     heatmap.save(fr"C:\atman_python\Dont-Swim-in-Data\outputs\charts_graphs\{metric}_heatmap.png", scale_factor=3)

# print("Heatmaps saved for TP, FP, TN, FN.")

## GROUPED BAR CHARTS

# import pandas as pd
# import altair as alt
# from altair_saver import save
# alt.renderers.set_embed_options(actions=False)  # disables Altair's interactive menu in output


# # Read the data
# df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\outputs\probabilistic_framework_threeclass_confusion_metrics.csv")

# # Melt TP, FP, TN, FN to long-form
# metrics = ['TP', 'FP', 'TN', 'FN']
# df_long = df.melt(
#     id_vars=['SITE_NAME', 'CLASS'],
#     value_vars=metrics,
#     var_name='Metric',
#     value_name='Value'
# )

# # Order class
# class_order = ["SAFE", "PRECAUTIONARY", "EXCEEDANCE"]
# df_long['CLASS'] = pd.Categorical(df_long['CLASS'], categories=class_order, ordered=True)

# # Create chart: Grouped bar, columns for class, color for metric
# chart = alt.Chart(df_long).mark_bar().encode(
#     x=alt.X('SITE_NAME:N', title='Site', sort=alt.EncodingSortField(field="SITE_NAME")),
#     y=alt.Y('Value:Q', title='Count'),
#     color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10'), title="Metric"),
#     column=alt.Column('CLASS:N', title='Class', sort=class_order),
#     tooltip=['SITE_NAME', 'CLASS', 'Metric', 'Value']
# ).properties(
#     width=90,
#     height=300,
#     title="TP, FP, TN, FN by Site and Class"
# ).configure_axisX(
#     labelAngle=-45
# )

# # ----------- Saving as Image (SVG or PNG) ----------- #
# # 1. Install the Altair export requirements (run once):
# #    pip install altair vega_datasets
# #    pip install -U kaleido
# #    pip install altair_saver



# # Save as SVG (best for publications)
# chart.save(r"C:\atman_python\Dont-Swim-in-Data\outputs\charts_graphs\site_class_confusion_grouped_bar.svg")

# # Save as PNG (high-res, for slides; change scale for even higher res)
# chart.save(r"C:\atman_python\Dont-Swim-in-Data\outputs\charts_graphs\site_class_confusion_grouped_bar.png", scale_factor=3)  # scale=3 for higher DPI

# print("Chart saved as SVG and PNG in the current directory.")
