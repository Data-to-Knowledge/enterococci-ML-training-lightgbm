import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="centered", page_title="Enterococci Model: Confusion Metrics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\outputs\probabilistic_framework_threeclass_confusion_metrics.csv")
    return df[df['SITE_NAME'] != 'OVERALL']

df = load_data()

metrics = ['TP', 'FP', 'TN', 'FN']
class_order = ["SAFE", "PRECAUTIONARY", "EXCEEDANCE"]

# Color-blind-friendly palette (Okabe-Ito)
color_palette = ['#56B4E9', '#E69F00', '#009E73', '#F0E442']

metric_labels = {
    'TP': 'TP (correct): Correctly predicted this class',
    'FP': 'FP (false alarm): Predicted this class, but it was actually another',
    'TN': 'TN (correct reject): Correctly predicted another class',
    'FN': 'FN (missed): Missed this class when it should have predicted it'
}
metric_legend_names = {
    'TP': 'TP (correct)',
    'FP': 'FP (false alarm)',
    'TN': 'TN (correct reject)',
    'FN': 'FN (missed)'
}

def legend_block():
    st.markdown("""
    **Legend: Confusion Matrix Metrics**

    - <span style='color:#56B4E9'><b>TP (correct):</b></span> Correctly predicted this class  
    - <span style='color:#E69F00'><b>FP (false alarm):</b></span> Predicted this class, but it was actually another  
    - <span style='color:#009E73'><b>TN (correct reject):</b></span> Correctly predicted another class  
    - <span style='color:#F0E442'><b>FN (missed):</b></span> Missed this class when it should have predicted it  
    """, unsafe_allow_html=True)

def bar_chart_for_class(selected_class):
    data = df[df['CLASS'] == selected_class].copy()
    data_long = data.melt(
        id_vars=['SITE_NAME', 'CLASS'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )
    data_long['Metric'] = data_long['Metric'].map(metric_legend_names)
    return alt.Chart(data_long).mark_bar(size=25).encode(
        x=alt.X('SITE_NAME:N', title='Site', sort=alt.EncodingSortField(field="SITE_NAME")),
        y=alt.Y('Value:Q', title='Count'),
        color=alt.Color('Metric:N', title="Metric",
                        scale=alt.Scale(domain=list(metric_legend_names.values()), range=color_palette)),
        tooltip=['SITE_NAME', 'Metric', 'Value']
    ).properties(
        width=620,
        height=300,
        title=f"{selected_class}: Confusion Metrics by Site"
    ).configure_axisX(
        labelAngle=-40,
        labelFontSize=11
    ).configure_axisY(
        labelFontSize=12
    ).configure_legend(
        titleFontSize=13,
        labelFontSize=12
    )

def bar_chart_for_site_metric(selected_site, selected_metric):
    # Get readable name for legend/axis
    metric_disp = metric_legend_names[selected_metric]
    # Filter data for site and metric
    data = df[df['SITE_NAME'] == selected_site][['CLASS', selected_metric]]
    return alt.Chart(data).mark_bar(size=50).encode(
        x=alt.X('CLASS:N', title='Class', sort=class_order),
        y=alt.Y(f'{selected_metric}:Q', title='Count'),
        color=alt.value(color_palette[metrics.index(selected_metric)]),
        tooltip=['CLASS', f'{selected_metric}']
    ).properties(
        width=240,
        height=340,
        title=f"{selected_site} — {metric_disp}"
    ).configure_axisX(
        labelAngle=-10,
        labelFontSize=12
    ).configure_axisY(
        labelFontSize=12
    )

st.title("Enterococci Prediction Model: Confusion Metrics (Per Class and Site)")
legend_block()

st.markdown("### Confusion Metrics by Class (All Sites)")
for cls in class_order:
    st.altair_chart(bar_chart_for_class(cls), use_container_width=True)

st.markdown("---")

st.markdown("### Explore a Single Site and Metric")
site_list = sorted(df['SITE_NAME'].unique())
metric_list = metrics
selected_site = st.selectbox("Select a site:", site_list)
selected_metric = st.selectbox("Select a metric:", [f"{m} — {metric_labels[m]}" for m in metric_list])
selected_metric_code = selected_metric.split(" ")[0]  # Get 'TP', 'FP', etc.

st.altair_chart(bar_chart_for_site_metric(selected_site, selected_metric_code), use_container_width=False)

st.caption("All charts are colour-blind friendly. Hover over any bar to see exact values.")

