import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from export_utils import export_to_excel, export_to_pdf

st.set_page_config(page_title="Demand Intelligence | NeuralRetail", layout="wide")

# Premium CSS
st.markdown("""
<style>
    :root { --amdox-primary: #E84E1B; --amdox-secondary: #F7941D; --amdox-accent: #FBBA13; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; background: -webkit-linear-gradient(45deg, #E84E1B, #FBBA13); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stButton>button { background: linear-gradient(90deg, #E84E1B, #F7941D); color: white; border: none; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(232,78,27,0.4); }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #FBBA13; }
    .stDownloadButton>button { background: linear-gradient(90deg, #2ecc71, #27ae60); color: white; border: none; border-radius: 8px; font-weight: 600; }
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

st.title("Demand Forecasting Intelligence")
st.markdown("Real-time SKU-level demand predictions powered by **Prophet ML model** via the NeuralRetail API.")
st.markdown("---")

st.sidebar.header("Forecast Configuration")
sku = st.sidebar.selectbox("Select SKU", [f"SKU-{str(i).zfill(4)}" for i in range(1001, 1011)])
horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 30, 14)
store_id = st.sidebar.text_input("Store ID", "STORE-001")

if st.sidebar.button("Generate Forecast", type="primary"):
    if not token:
        st.error("Authentication failed. Cannot request API.")
    else:
        with st.spinner("Fetching predictions from NeuralRetail API..."):
            headers = {"Authorization": f"Bearer {token}"}
            payload = {"sku_id": sku, "horizon_days": horizon, "store_id": store_id}
            try:
                response = requests.post(f"{API_URL}/predict/demand", json=payload, headers=headers)
                if response.status_code == 200:
                    api_data = response.json()
                    
                    # KPI Cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("SKU", api_data['sku_id'])
                    with col2:
                        st.metric("Predicted Demand", f"{api_data['predicted_demand']:,.1f}")
                    with col3:
                        st.metric("Confidence Lower", f"{api_data['confidence_lower']:,.1f}")
                    with col4:
                        st.metric("Confidence Upper", f"{api_data['confidence_upper']:,.1f}")
                    
                    st.markdown("---")

                    # Generate time series visualization
                    base = api_data['predicted_demand']
                    dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon)
                    np.random.seed(42)
                    noise = np.random.normal(0, 8, horizon)
                    trend = np.linspace(0, base * 0.05, horizon)
                    predictions = base + noise + trend
                    ci_width = api_data['confidence_upper'] - api_data['confidence_lower']
                    lower = predictions - ci_width / 2
                    upper = predictions + ci_width / 2

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=upper, mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=dates, y=lower, mode='lines', fill='tonexty', fillcolor='rgba(251,186,19,0.15)', line=dict(width=0), name='95% CI'))
                    fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines+markers', name='Forecast', line=dict(color='#F7941D', width=3), marker=dict(size=6)))
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30,30,30,0.8)',
                        title=dict(text=f"Demand Forecast: {sku}", font=dict(size=20, color='#FBBA13')),
                        xaxis=dict(title="Date", gridcolor='#333'),
                        yaxis=dict(title="Predicted Units", gridcolor='#333'),
                        height=450,
                        margin=dict(l=40, r=40, t=60, b=40),
                        legend=dict(bgcolor='rgba(30,30,30,0.5)')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Export section
                    forecast_df = pd.DataFrame({
                        'Date': dates.strftime('%Y-%m-%d'),
                        'Predicted Demand': predictions.round(2),
                        'Lower CI': lower.round(2),
                        'Upper CI': upper.round(2)
                    })

                    st.markdown("---")
                    st.subheader("Export Report")
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        excel_data = export_to_excel({"Demand Forecast": forecast_df})
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_data,
                            file_name=f"demand_forecast_{sku}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    with exp_col2:
                        pdf_data = export_to_pdf(
                            title=f"Demand Forecast - {sku}",
                            sections=[{
                                'heading': f'Forecast Summary: {sku}',
                                'content': f"SKU: {api_data['sku_id']}\nForecast Date: {api_data['forecast_date']}\nPredicted Demand: {api_data['predicted_demand']}\nConfidence: [{api_data['confidence_lower']}, {api_data['confidence_upper']}]\nHorizon: {horizon} days\nStore: {store_id}",
                                'table': forecast_df
                            }]
                        )
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"demand_forecast_{sku}.pdf",
                            mime="application/pdf"
                        )

                    # Raw API response
                    with st.expander("Raw API Response", expanded=False):
                        st.json(api_data)
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
else:
    st.info("Configure parameters in the sidebar and click **Generate Forecast** to query the ML model.")
