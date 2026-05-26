import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from export_utils import export_to_excel, export_to_pdf

st.set_page_config(page_title="Revenue Analytics | NeuralRetail", layout="wide")

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

st.title("Revenue & Price Intelligence")
st.markdown("Causal price elasticity analysis via **DoWhy + EconML**, what-if revenue simulation, and cross-price dynamics.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Price Elasticity", "Revenue Simulator", "Cross-Price Analysis"])

# ── Tab 1: Price Elasticity Analysis ────────────────────
with tab1:
    st.subheader("Causal Price Elasticity by Category")
    st.markdown("_Using **LinearDML** (causal) and **NonParamDML** (non-linear) from DoWhy + EconML._")

    if st.button("Compute Elasticity", type="primary", key="elasticity_btn"):
        if not token:
            st.error("Authentication failed.")
        else:
            with st.spinner("Running causal inference models (DoWhy + EconML)..."):
                headers = {"Authorization": f"Bearer {token}"}
                try:
                    resp = requests.post(f"{API_URL}/revenue/elasticity", json={}, headers=headers, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()

                        if not data:
                            st.warning("No elasticity data returned.")
                        else:
                            # KPI Cards
                            avg_elasticity = np.mean([d['elasticity_coefficient'] for d in data])
                            elastic_count = sum(1 for d in data if abs(d['nonlinear_elasticity']) > 1)
                            avg_r2 = np.mean([d['r_squared'] for d in data])

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Categories Analyzed", len(data))
                            with col2:
                                st.metric("Avg Elasticity", f"{avg_elasticity:.3f}")
                            with col3:
                                st.metric("Elastic Categories", f"{elastic_count}/{len(data)}")
                            with col4:
                                st.metric("Avg R²", f"{avg_r2:.3f}")

                            st.markdown("---")

                            # Elasticity Comparison Chart (Linear vs Non-Linear)
                            categories = [d['category'] for d in data]
                            linear_vals = [d['elasticity_coefficient'] for d in data]
                            nonlinear_vals = [d['nonlinear_elasticity'] for d in data]

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                name='LinearDML (Causal)',
                                x=categories, y=linear_vals,
                                marker_color='#3498db',
                                text=[f"{v:.3f}" for v in linear_vals],
                                textposition='outside',
                                textfont=dict(color='white', size=11)
                            ))
                            fig.add_trace(go.Bar(
                                name='NonParamDML (Non-Linear)',
                                x=categories, y=nonlinear_vals,
                                marker_color='#F7941D',
                                text=[f"{v:.3f}" for v in nonlinear_vals],
                                textposition='outside',
                                textfont=dict(color='white', size=11)
                            ))
                            fig.add_hline(y=-1.0, line_dash="dash", line_color="#E84E1B",
                                          annotation_text="Elastic Threshold (-1.0)")
                            fig.update_layout(
                                barmode='group',
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(30,30,30,0.8)',
                                title=dict(text="Price Elasticity: Linear vs Non-Linear Causal Models",
                                           font=dict(size=18, color='#FBBA13')),
                                xaxis=dict(title="Product Category", gridcolor='#333'),
                                yaxis=dict(title="Elasticity Coefficient", gridcolor='#333'),
                                height=450,
                                margin=dict(l=40, r=40, t=60, b=40),
                                legend=dict(bgcolor='rgba(30,30,30,0.5)', font=dict(color='white'))
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Interpretation Cards
                            st.subheader("Category Insights")
                            cols = st.columns(min(len(data), 3))
                            for i, d in enumerate(data):
                                with cols[i % 3]:
                                    color = '#2ecc71' if abs(d['nonlinear_elasticity']) <= 1 else '#E84E1B'
                                    icon = "📈" if abs(d['nonlinear_elasticity']) > 1 else "📊"
                                    st.markdown(f"""
                                    <div style="background: rgba(30,30,30,0.8); border-left: 4px solid {color}; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                                        <h4 style="-webkit-text-fill-color: {color}; margin:0;">{icon} {d['category']}</h4>
                                        <p style="color: #ccc; margin: 8px 0 4px 0;">Elasticity: <strong style="color: {color};">{d['nonlinear_elasticity']:.4f}</strong></p>
                                        <p style="color: #aaa; font-size: 0.85rem; margin: 0;">{d['interpretation']}</p>
                                        <p style="color: #888; font-size: 0.8rem; margin: 4px 0 0 0;">Avg Price: ${d['avg_price']:.2f} | Avg Demand: {d['avg_demand']:.0f} units</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Detail Table
                            st.markdown("---")
                            detail_df = pd.DataFrame(data)
                            detail_df.columns = ['Category', 'Linear Elasticity', 'Non-Linear Elasticity',
                                                  'R²', 'Interpretation', 'Avg Price ($)', 'Avg Demand']
                            st.dataframe(detail_df.style.format({
                                'Linear Elasticity': '{:.4f}',
                                'Non-Linear Elasticity': '{:.4f}',
                                'R²': '{:.4f}',
                                'Avg Price ($)': '${:.2f}',
                                'Avg Demand': '{:.0f}'
                            }), use_container_width=True)

                            # Export
                            st.markdown("---")
                            exp1, exp2 = st.columns(2)
                            with exp1:
                                st.download_button("Download Excel", export_to_excel({"Elasticity": detail_df}),
                                                   "elasticity_report.xlsx",
                                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                   key="elast_xl")
                            with exp2:
                                st.download_button("Download PDF", export_to_pdf(
                                    "Price Elasticity Report",
                                    [{'heading': 'Elasticity Summary',
                                      'content': f"Categories: {len(data)}\nAvg Elasticity: {avg_elasticity:.4f}\nElastic: {elastic_count}",
                                      'table': detail_df}]
                                ), "elasticity_report.pdf", "application/pdf", key="elast_pdf")

                            with st.expander("Raw API Response"):
                                st.json(data)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
    else:
        st.info("Click **Compute Elasticity** to run the DoWhy causal inference pipeline across all product categories.")


# ── Tab 2: What-If Revenue Simulator ────────────────────
with tab2:
    st.subheader("What-If Revenue Simulator")
    st.markdown("_Adjust price and promotion to project demand & revenue impact using the causal elasticity model._")

    col_config, col_spacer = st.columns([1, 1])
    with col_config:
        sim_sku = st.selectbox("Select SKU", [f"SKU-{str(i).zfill(4)}" for i in range(1001, 1021)], key="sim_sku")
        price_change = st.slider("Price Change (%)", -50, 50, 0, step=1, key="sim_price")
        promo_flag = st.checkbox("Apply Promotion (25% lift)", key="sim_promo")
        run_sim = st.button("Run Simulation", type="primary", key="sim_btn")

    if run_sim:
        if not token:
            st.error("Authentication failed.")
        else:
            with st.spinner("Running revenue simulation..."):
                headers = {"Authorization": f"Bearer {token}"}
                try:
                    resp = requests.post(f"{API_URL}/revenue/simulate", json={
                        "sku_id": sim_sku,
                        "price_change_pct": price_change,
                        "promotion_flag": promo_flag
                    }, headers=headers, timeout=30)
                    if resp.status_code == 200:
                        sim = resp.json()

                        # KPI Comparison
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            color = "#2ecc71" if sim['revenue_change_pct'] > 0 else "#E84E1B"
                            st.metric("Revenue Impact",
                                      f"${sim['projected_revenue']:,.0f}",
                                      f"{sim['revenue_change_pct']:+.1f}%")
                        with col2:
                            st.metric("Demand Impact",
                                      f"{sim['projected_demand']:,.0f} units",
                                      f"{sim['demand_change_pct']:+.1f}%")
                        with col3:
                            st.metric("Price Change",
                                      f"${sim['new_price']:.2f}",
                                      f"from ${sim['current_price']:.2f}")

                        # Waterfall Chart: Current → Price Effect → Promo Effect → Projected
                        st.markdown("---")
                        st.subheader("Revenue Waterfall")

                        price_effect = sim['projected_revenue'] / (1 + sim['promotion_lift'] / 100) - sim['current_revenue']
                        promo_effect = sim['projected_revenue'] - sim['current_revenue'] - price_effect

                        fig_waterfall = go.Figure(go.Waterfall(
                            name="Revenue",
                            orientation="v",
                            x=["Current Revenue", "Price Effect", "Promotion Lift", "Projected Revenue"],
                            y=[sim['current_revenue'], price_effect, promo_effect, 0],
                            measure=["absolute", "relative", "relative", "total"],
                            textposition="outside",
                            text=[f"${sim['current_revenue']:,.0f}", f"${price_effect:+,.0f}",
                                  f"${promo_effect:+,.0f}", f"${sim['projected_revenue']:,.0f}"],
                            textfont=dict(color='white', size=12),
                            increasing=dict(marker_color='#2ecc71'),
                            decreasing=dict(marker_color='#E84E1B'),
                            totals=dict(marker_color='#FBBA13'),
                            connector=dict(line=dict(color='#555', width=1)),
                        ))
                        fig_waterfall.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(30,30,30,0.8)',
                            title=dict(text=f"Revenue Impact Analysis — {sim_sku}",
                                       font=dict(size=18, color='#FBBA13')),
                            yaxis=dict(title="Revenue ($)", gridcolor='#333'),
                            xaxis=dict(gridcolor='#333'),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig_waterfall, use_container_width=True)

                        # Sensitivity Curve: sweep price changes from -30% to +30%
                        st.subheader("Price Sensitivity Curve")
                        sensitivity_data = []
                        for pct in range(-30, 31, 5):
                            try:
                                r = requests.post(f"{API_URL}/revenue/simulate", json={
                                    "sku_id": sim_sku, "price_change_pct": pct, "promotion_flag": False
                                }, headers=headers, timeout=10)
                                if r.status_code == 200:
                                    s = r.json()
                                    sensitivity_data.append({
                                        "Price Change (%)": pct,
                                        "Revenue ($)": s['projected_revenue'],
                                        "Demand (units)": s['projected_demand']
                                    })
                            except Exception:
                                pass

                        if sensitivity_data:
                            sens_df = pd.DataFrame(sensitivity_data)
                            fig_sens = go.Figure()
                            fig_sens.add_trace(go.Scatter(
                                x=sens_df["Price Change (%)"], y=sens_df["Revenue ($)"],
                                mode='lines+markers', name='Revenue',
                                line=dict(color='#FBBA13', width=3),
                                marker=dict(size=8)
                            ))
                            fig_sens.add_trace(go.Scatter(
                                x=sens_df["Price Change (%)"], y=sens_df["Demand (units)"],
                                mode='lines+markers', name='Demand',
                                line=dict(color='#3498db', width=3, dash='dot'),
                                marker=dict(size=8),
                                yaxis='y2'
                            ))
                            fig_sens.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(30,30,30,0.8)',
                                title=dict(text="Price Sensitivity Analysis",
                                           font=dict(size=18, color='#FBBA13')),
                                xaxis=dict(title="Price Change (%)", gridcolor='#333', zeroline=True, zerolinecolor='#555'),
                                yaxis=dict(title="Revenue ($)", gridcolor='#333', side='left'),
                                yaxis2=dict(title="Demand (units)", overlaying='y', side='right', gridcolor='#333'),
                                height=400,
                                margin=dict(l=40, r=60, t=60, b=40),
                                legend=dict(bgcolor='rgba(30,30,30,0.5)', font=dict(color='white'))
                            )
                            st.plotly_chart(fig_sens, use_container_width=True)

                        with st.expander("Raw API Response"):
                            st.json(sim)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
    else:
        st.info("Configure SKU, price change, and promotion settings, then click **Run Simulation**.")


# ── Tab 3: Cross-Price Elasticity ───────────────────────
with tab3:
    st.subheader("Cross-Price Elasticity Matrix")
    st.markdown("_How does one category's price change affect another category's demand? Computed via **Double ML**._")

    # Load categories from products data
    try:
        products = pd.read_parquet("data/bronze/products.parquet")
        categories = sorted(products['category'].unique().tolist())
    except Exception:
        categories = ["Electronics", "Groceries", "Fashion", "Home"]

    col_focal, col_comp = st.columns(2)
    with col_focal:
        focal = st.selectbox("Focal Category", categories, key="cross_focal")
    with col_comp:
        competitor_options = [c for c in categories if c != focal]
        competitor = st.selectbox("Competitor Category", competitor_options, key="cross_comp")

    if st.button("Compute Cross-Elasticity", type="primary", key="cross_btn"):
        if not token:
            st.error("Authentication failed.")
        else:
            with st.spinner("Running Double ML cross-price analysis..."):
                headers = {"Authorization": f"Bearer {token}"}
                try:
                    resp = requests.post(f"{API_URL}/revenue/cross-elasticity", json={
                        "focal_category": focal,
                        "competitor_category": competitor
                    }, headers=headers, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()

                        ce = data['cross_elasticity']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cross-Elasticity", f"{ce:.4f}")
                        with col2:
                            relationship = "Substitute" if ce > 0 else "Complement" if ce < 0 else "Independent"
                            st.metric("Relationship", relationship)
                        with col3:
                            st.metric("Direction", data['interpretation'][:30] + "...")

                        # Visual interpretation
                        st.markdown("---")
                        color = '#2ecc71' if ce > 0 else '#E84E1B' if ce < 0 else '#888'
                        icon = "🔄" if ce > 0 else "🤝" if ce < 0 else "➖"
                        st.markdown(f"""
                        <div style="background: rgba(30,30,30,0.8); border: 2px solid {color}; padding: 24px; border-radius: 12px; text-align: center;">
                            <h2 style="-webkit-text-fill-color: {color}; margin: 0;">{icon} {data['interpretation']}</h2>
                            <p style="color: #aaa; margin-top: 12px; font-size: 1.1rem;">
                                When <strong>{competitor}</strong> prices increase by 1%,
                                <strong>{focal}</strong> demand changes by <strong style="color: {color};">{ce:+.4f}%</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander("Raw API Response"):
                            st.json(data)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    # Build full heatmap across all category pairs
    st.markdown("---")
    st.subheader("Full Cross-Elasticity Heatmap")
    if st.button("Generate Full Heatmap (may take ~30s)", key="heatmap_btn"):
        if not token:
            st.error("Authentication failed.")
        else:
            with st.spinner("Computing cross-elasticity for all category pairs..."):
                headers = {"Authorization": f"Bearer {token}"}
                matrix = {}
                for f_cat in categories:
                    matrix[f_cat] = {}
                    for c_cat in categories:
                        if f_cat == c_cat:
                            matrix[f_cat][c_cat] = 0.0
                        else:
                            try:
                                r = requests.post(f"{API_URL}/revenue/cross-elasticity", json={
                                    "focal_category": f_cat,
                                    "competitor_category": c_cat
                                }, headers=headers, timeout=15)
                                if r.status_code == 200:
                                    matrix[f_cat][c_cat] = r.json()['cross_elasticity']
                                else:
                                    matrix[f_cat][c_cat] = 0.0
                            except Exception:
                                matrix[f_cat][c_cat] = 0.0

                heatmap_df = pd.DataFrame(matrix)
                fig_hm = px.imshow(
                    heatmap_df.values,
                    x=heatmap_df.columns.tolist(),
                    y=heatmap_df.index.tolist(),
                    color_continuous_scale='RdBu_r',
                    text_auto='.3f',
                    labels=dict(x="Competitor Category", y="Focal Category", color="Cross-Elasticity"),
                )
                fig_hm.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title=dict(text="Cross-Price Elasticity Heatmap",
                               font=dict(size=18, color='#FBBA13')),
                    height=500,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                st.plotly_chart(fig_hm, use_container_width=True)

                st.markdown("_**Blue**: Complement (↑ competitor price → ↓ focal demand) | **Red**: Substitute (↑ competitor price → ↑ focal demand)_")
