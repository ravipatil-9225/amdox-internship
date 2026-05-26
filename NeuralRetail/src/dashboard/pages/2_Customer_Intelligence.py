import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from export_utils import export_to_excel, export_to_pdf

st.set_page_config(page_title="Customer Intelligence | NeuralRetail", layout="wide")

# Premium CSS
st.markdown("""
<style>
    :root { --amdox-primary: #E84E1B; --amdox-secondary: #F7941D; --amdox-accent: #FBBA13; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; background: -webkit-linear-gradient(45deg, #E84E1B, #FBBA13); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { background: linear-gradient(90deg, #E84E1B, #F7941D); color: white; border: none; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(232,78,27,0.4); }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #FBBA13; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("authentication_status"):
    st.error("Please log in from the main page to access this dashboard.")
    st.stop()

API_URL = os.environ.get("NEURALRETAIL_API_URL", "https://neuralretail-api-python.onrender.com/api/v1")

@st.cache_data(ttl=900)
def get_auth_token():
    try:
        response = requests.post(f"{API_URL}/login/access-token", data={"username": "admin", "password": "admin"})
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
    return None

token = get_auth_token()

st.title("Customer Intelligence Hub")
st.markdown("Churn prediction with **SHAP explainability** and **K-Means RFM segmentation**.")
st.markdown("---")

tab1, tab2 = st.tabs(["Churn Prediction", "Customer Segmentation"])

# ── Tab 1: Churn Prediction ──────────────────────────
with tab1:
    st.subheader("Individual Churn Risk Assessment")
    
    col_input, col_spacer = st.columns([1, 2])
    with col_input:
        customer_id = st.text_input("Customer ID", "CUST-0001")
        run_churn = st.button("Predict Churn Risk", type="primary", key="churn_btn")

    if run_churn:
        if not token:
            st.error("Authentication failed.")
        else:
            with st.spinner("Running XGBoost + SHAP analysis..."):
                headers = {"Authorization": f"Bearer {token}"}
                try:
                    resp = requests.post(f"{API_URL}/predict/churn", json={"customer_id": customer_id}, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Risk KPIs
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Customer", data['customer_id'])
                        with col2:
                            prob = data['churn_probability']
                            st.metric("Churn Probability", f"{prob*100:.1f}%", delta=f"{'HIGH' if prob >= 0.7 else 'MEDIUM' if prob >= 0.4 else 'LOW'}")
                        with col3:
                            risk_colors = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}
                            st.metric("Risk Segment", data['risk_segment'])

                        st.markdown("---")

                        # SHAP Waterfall Chart
                        st.subheader("SHAP Feature Attribution")
                        st.markdown("_How each feature pushes the churn prediction higher or lower._")
                        
                        shap_vals = data['shap_values']
                        features = list(shap_vals.keys())
                        values = list(shap_vals.values())

                        colors = ['#E84E1B' if v > 0 else '#2ecc71' for v in values]

                        fig = go.Figure(go.Bar(
                            x=values,
                            y=features,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{v:+.4f}" for v in values],
                            textposition='outside',
                            textfont=dict(color='white', size=14)
                        ))
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(30,30,30,0.8)',
                            title=dict(text="SHAP Waterfall - Churn Drivers", font=dict(size=18, color='#FBBA13')),
                            xaxis=dict(title="SHAP Value (Impact on Churn)", gridcolor='#333', zeroline=True, zerolinecolor='#555'),
                            yaxis=dict(gridcolor='#333'),
                            height=350,
                            margin=dict(l=20, r=80, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Raw API Response"):
                            st.json(data)

                        # Export
                        st.markdown("---")
                        churn_df = pd.DataFrame([data])
                        exp1, exp2 = st.columns(2)
                        with exp1:
                            st.download_button("Download Excel", export_to_excel({"Churn Analysis": churn_df}), f"churn_{customer_id}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        with exp2:
                            st.download_button("Download PDF", export_to_pdf(f"Churn Report - {customer_id}", [{'heading': 'Churn Prediction', 'content': f"Customer: {data['customer_id']}\nChurn Probability: {data['churn_probability']}\nRisk: {data['risk_segment']}", 'table': churn_df}]), f"churn_{customer_id}.pdf", "application/pdf")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ── Tab 2: Customer Segmentation ──────────────────────
with tab2:
    st.subheader("K-Means RFM Customer Segmentation")
    
    if st.button("Run Segmentation", type="primary", key="seg_btn"):
        if not token:
            st.error("Authentication failed.")
        else:
            with st.spinner("Running K-Means clustering on customer base..."):
                headers = {"Authorization": f"Bearer {token}"}
                try:
                    resp = requests.post(f"{API_URL}/segment/score", json={}, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Summary KPIs
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Customers", f"{data['total_customers']:,}")
                        with col2:
                            st.metric("Segments Found", data['num_segments'])

                        st.markdown("---")

                        # Segment Distribution Donut
                        summary = data['segment_summary']
                        seg_names = list(summary.keys())
                        seg_counts = [summary[s]['count'] for s in seg_names]

                        amdox_palette = ['#E84E1B', '#F7941D', '#FBBA13', '#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#1abc9c']

                        fig_donut = go.Figure(data=[go.Pie(
                            labels=seg_names,
                            values=seg_counts,
                            hole=0.55,
                            marker=dict(colors=amdox_palette[:len(seg_names)]),
                            textinfo='label+percent',
                            textfont=dict(size=13, color='white')
                        )])
                        fig_donut.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            title=dict(text="Customer Segment Distribution", font=dict(size=18, color='#FBBA13')),
                            height=400,
                            margin=dict(l=20, r=20, t=60, b=20),
                            legend=dict(bgcolor='rgba(30,30,30,0.5)', font=dict(color='white'))
                        )
                        st.plotly_chart(fig_donut, use_container_width=True)

                        # Radar Chart for Segment Profiles
                        st.subheader("Segment RFM Profiles")
                        categories = ['Avg Recency', 'Avg Frequency', 'Avg Monetary']
                        
                        fig_radar = go.Figure()
                        for i, seg in enumerate(seg_names):
                            vals = [summary[seg]['avg_recency'], summary[seg]['avg_frequency'], summary[seg]['avg_monetary']]
                            # Normalize to 0-100 for radar
                            max_vals = [max(summary[s][k] for s in seg_names) for k in ['avg_recency', 'avg_frequency', 'avg_monetary']]
                            norm_vals = [v / max(m, 1) * 100 for v, m in zip(vals, max_vals)]
                            norm_vals.append(norm_vals[0])  # close the polygon

                            fig_radar.add_trace(go.Scatterpolar(
                                r=norm_vals,
                                theta=categories + [categories[0]],
                                name=seg,
                                line=dict(color=amdox_palette[i % len(amdox_palette)], width=2),
                                fill='toself',
                                fillcolor=f"rgba({int(amdox_palette[i % len(amdox_palette)][1:3], 16)},{int(amdox_palette[i % len(amdox_palette)][3:5], 16)},{int(amdox_palette[i % len(amdox_palette)][5:7], 16)},0.1)"
                            ))

                        fig_radar.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            polar=dict(bgcolor='rgba(30,30,30,0.8)', radialaxis=dict(visible=True, gridcolor='#444')),
                            title=dict(text="Segment Comparison Radar", font=dict(size=18, color='#FBBA13')),
                            height=450,
                            margin=dict(l=40, r=40, t=60, b=40),
                            legend=dict(bgcolor='rgba(30,30,30,0.5)', font=dict(color='white'))
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                        # Segment Summary Table
                        st.subheader("Segment Detail Table")
                        summary_df = pd.DataFrame(summary).T
                        summary_df.index.name = "Segment"
                        summary_df.columns = ['Customer Count', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Total Revenue ($)']
                        st.dataframe(summary_df.style.format({
                            'Customer Count': '{:.0f}',
                            'Avg Recency (days)': '{:.1f}',
                            'Avg Frequency': '{:.1f}',
                            'Avg Monetary ($)': '${:,.2f}',
                            'Total Revenue ($)': '${:,.2f}'
                        }), use_container_width=True)

                        with st.expander("Raw API Response"):
                            st.json(data)

                        # Export
                        st.markdown("---")
                        exp1, exp2 = st.columns(2)
                        with exp1:
                            st.download_button("Download Excel", export_to_excel({"Segments": summary_df}), "segmentation_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="seg_excel")
                        with exp2:
                            st.download_button("Download PDF", export_to_pdf("Customer Segmentation Report", [{'heading': 'RFM Segments', 'content': f"Total Customers: {data['total_customers']}\nSegments: {data['num_segments']}", 'table': summary_df.reset_index()}]), "segmentation_report.pdf", "application/pdf", key="seg_pdf")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
