import streamlit as st
import pandas as pd
import plotly.express as px
import sweetviz as sv
from ydata_profiling import ProfileReport
import tempfile
import streamlit.components.v1 as components
import os

def run_eda(df):
    st.header(":bar_chart: Data Overview")

    eda_choice = st.radio("Choose EDA Mode:", ["Built-in Fast EDA", "Sweetviz Report", "YData Profiling Report"])

    if eda_choice == "Built-in Fast EDA":
        st.subheader(":open_file_folder: Quick Data Preview")
        st.dataframe(df.head())

        st.subheader(":straight_ruler: Summary Statistics")
        st.dataframe(df.describe())

        st.subheader(":warning: Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0])

        st.subheader(":date: Time Series Plot")
        fig = px.line(df, x='ds', y='y', title="Time Series")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_choice == "Sweetviz Report":
        st.subheader(":art: Sweetviz Auto EDA Report")
        if st.button("🔍 Generate Sweetviz Report"):
            with st.spinner("Generating Sweetviz Report..."):
                report = sv.analyze(df)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                    report.show_html(tmp_file.name, open_browser=False)
                    html_content = open(tmp_file.name, 'r').read()
                    components.html(html_content, height=1000, scrolling=True)

    elif eda_choice == "YData Profiling Report":
        st.subheader(":brain: YData Profiling Auto Report")
        if st.button("🔍 Generate YData Profiling Report"):
            with st.spinner("Generating YData Profiling Report..."):
                profile = ProfileReport(df, title="YData Profiling Report", minimal=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                    profile.to_file(tmp_file.name)
                    html_content = open(tmp_file.name, 'r').read()
                    components.html(html_content, height=1000, scrolling=True)


