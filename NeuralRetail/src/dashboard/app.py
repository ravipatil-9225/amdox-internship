import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

st.set_page_config(
    page_title="NeuralRetail | Amdox",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Amdox Premium Branding CSS
st.markdown("""
<style>
    /* Main Theme */
    :root {
        --amdox-primary: #E84E1B;
        --amdox-secondary: #F7941D;
        --amdox-accent: #FBBA13;
        --bg-dark: #121212;
        --card-dark: #1E1E1E;
    }
    
    .stApp {
        background-color: var(--bg-dark);
        color: #ffffff;
    }
    
    /* Header Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: -webkit-linear-gradient(45deg, var(--amdox-primary), var(--amdox-accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--card-dark);
        border-right: 1px solid #333;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--amdox-primary) 0%, var(--amdox-secondary) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(232, 78, 27, 0.4);
    }
    
    /* Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--amdox-accent);
    }
    
    /* Custom divider */
    hr {
        border-color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login(location='main', fields={'Form name': 'NeuralRetail Login'})

if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
    
    st.title("⚡ NeuralRetail - Executive Hub")
    st.markdown("### AI-Powered Sales Intelligence & Predictive Analytics Platform")
    st.markdown("---")

    st.markdown("""
    Welcome to **NeuralRetail**, an enterprise-grade AI platform developed by **Amdox Technologies**. 

    👈 **Use the sidebar to navigate to:**
    1. **Demand Intelligence:** AI-driven SKU forecasting.
    2. **Customer Intelligence:** Churn prediction and RFM segmentation.
    3. **Inventory Health:** EOQ-based reorder recommendations.
    4. **MLOps Monitor:** Pipeline health and MLflow metrics.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy Target", "≥ 90%", "+5% vs Baseline")
    with col2:
        st.metric("Processing Throughput", "15M+ txns", "Real-time")
    with col3:
        st.metric("System Status", "Healthy", "All APIs Active")

elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')
