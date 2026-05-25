import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from export_utils import export_to_excel, export_to_pdf

st.set_page_config(page_title="Inventory Health | NeuralRetail", layout="wide")

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

API_URL = os.environ.get("NEURALRETAIL_API_URL", "http://127.0.0.1:8000/api/v1")

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

st.title("Inventory Optimization & Health")
st.markdown("EOQ-based reorder recommendations, ABC classification, safety stock, and dead-stock detection.")
st.markdown("---")

st.sidebar.header("Inventory Analysis")
sku_list = [f"SKU-{str(i).zfill(4)}" for i in range(1001, 1021)]
selected_skus = st.sidebar.multiselect("Select SKUs to Analyze", sku_list, default=sku_list[:5])
store_id = st.sidebar.text_input("Store ID", "STORE-001")

if st.sidebar.button("Analyze Inventory", type="primary"):
    if not token:
        st.error("Authentication failed.")
    else:
        results = []
        progress = st.progress(0)
        headers = {"Authorization": f"Bearer {token}"}

        for i, sku in enumerate(selected_skus):
            try:
                resp = requests.post(f"{API_URL}/inventory/reorder", json={"sku_id": sku, "store_id": store_id}, headers=headers)
                if resp.status_code == 200:
                    results.append(resp.json())
            except:
                pass
            progress.progress((i + 1) / len(selected_skus))

        progress.empty()

        if results:
            df = pd.DataFrame(results)

            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("SKUs Analyzed", len(df))
            with col2:
                high_risk = len(df[df['stockout_risk'] > 0.5])
                st.metric("High Stockout Risk", high_risk, delta=f"{'ALERT' if high_risk > 0 else 'OK'}")
            with col3:
                dead = len(df[df['dead_stock'] == True])
                st.metric("Dead Stock Items", dead)
            with col4:
                avg_dos = df['days_of_supply'].mean()
                st.metric("Avg Days of Supply", f"{avg_dos:.1f}")

            st.markdown("---")

            # Stockout Risk Gauge
            st.subheader("Stockout Risk by SKU")
            fig_risk = go.Figure()
            colors = ['#2ecc71' if r < 0.3 else '#F7941D' if r < 0.6 else '#E84E1B' for r in df['stockout_risk']]
            fig_risk.add_trace(go.Bar(
                x=df['sku_id'],
                y=df['stockout_risk'],
                marker_color=colors,
                text=[f"{r:.0%}" for r in df['stockout_risk']],
                textposition='outside',
                textfont=dict(color='white', size=12)
            ))
            fig_risk.add_hline(y=0.5, line_dash="dash", line_color="#E84E1B", annotation_text="High Risk Threshold")
            fig_risk.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,0.8)',
                title=dict(text="Stockout Risk Assessment", font=dict(size=18, color='#FBBA13')),
                xaxis=dict(title="SKU", gridcolor='#333'),
                yaxis=dict(title="Risk Score", gridcolor='#333', range=[0, 1.1]),
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Current Stock vs Safety Stock vs Reorder
            st.subheader("Stock Levels vs Safety Stock")
            fig_stock = go.Figure()
            fig_stock.add_trace(go.Bar(name='Current Stock', x=df['sku_id'], y=df['current_stock'], marker_color='#3498db'))
            fig_stock.add_trace(go.Bar(name='Safety Stock', x=df['sku_id'], y=df['safety_stock'], marker_color='#E84E1B'))
            fig_stock.add_trace(go.Bar(name='Reorder Qty (EOQ)', x=df['sku_id'], y=df['recommended_reorder_qty'], marker_color='#FBBA13'))
            fig_stock.update_layout(
                barmode='group',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,0.8)',
                title=dict(text="Inventory Position Overview", font=dict(size=18, color='#FBBA13')),
                xaxis=dict(gridcolor='#333'),
                yaxis=dict(title="Units", gridcolor='#333'),
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(bgcolor='rgba(30,30,30,0.5)', font=dict(color='white'))
            )
            st.plotly_chart(fig_stock, use_container_width=True)

            # ABC Classification Matrix
            st.subheader("ABC Classification")
            col_a, col_b, col_c = st.columns(3)
            for col, cls, color in [(col_a, 'A', '#2ecc71'), (col_b, 'B', '#F7941D'), (col_c, 'C', '#E84E1B')]:
                with col:
                    count = len(df[df['abc_class'] == cls])
                    skus = ', '.join(df[df['abc_class'] == cls]['sku_id'].tolist()) if count > 0 else 'None'
                    st.markdown(f"""
                    <div style="background: rgba(30,30,30,0.8); border-left: 4px solid {color}; padding: 16px; border-radius: 8px;">
                        <h3 style="margin:0; -webkit-text-fill-color: {color};">Class {cls}</h3>
                        <p style="font-size: 2rem; color: {color}; margin: 8px 0;">{count} SKUs</p>
                        <p style="color: #aaa; font-size: 0.85rem;">{skus}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Full detail table
            st.markdown("---")
            st.subheader("Detailed Inventory Report")
            display_df = df[['sku_id', 'current_stock', 'safety_stock', 'recommended_reorder_qty', 'stockout_risk', 'days_of_supply', 'abc_class', 'dead_stock']].copy()
            display_df.columns = ['SKU', 'Current Stock', 'Safety Stock', 'EOQ Reorder', 'Stockout Risk', 'Days of Supply', 'ABC Class', 'Dead Stock']
            st.dataframe(display_df.style.format({
                'Stockout Risk': '{:.1%}',
                'Days of Supply': '{:.1f}'
            }), use_container_width=True)

            # Export buttons
            st.markdown("---")
            st.subheader("Export Report")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                excel_data = export_to_excel({"Inventory Report": display_df})
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name="inventory_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with exp_col2:
                pdf_data = export_to_pdf(
                    title="Inventory Optimization Report",
                    sections=[{
                        'heading': 'Inventory Health Summary',
                        'content': f"SKUs Analyzed: {len(df)}\nHigh Stockout Risk: {len(df[df['stockout_risk'] > 0.5])}\nDead Stock Items: {len(df[df['dead_stock'] == True])}\nAvg Days of Supply: {df['days_of_supply'].mean():.1f}",
                        'table': display_df
                    }]
                )
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name="inventory_report.pdf",
                    mime="application/pdf"
                )

            with st.expander("Raw API Responses"):
                st.json(results)
        else:
            st.warning("No results returned from API.")
else:
    st.info("Select SKUs in the sidebar and click **Analyze Inventory** to run the EOQ optimization.")
