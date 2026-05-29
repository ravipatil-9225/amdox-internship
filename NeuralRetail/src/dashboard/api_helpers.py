"""
NeuralRetail Dashboard -- API Helper Utilities
================================================
Shared helper functions for Streamlit dashboard pages to handle:
  - Render free-tier cold starts (30-60s spin-up)
  - Retry with exponential backoff
  - Auth token caching
"""
import streamlit as st
import requests
import time
import os

API_URL = os.environ.get("NEURALRETAIL_API_URL", "https://neuralretail-api-python.onrender.com/api/v1")

# Timeout for API requests (Render free tier can take up to 60s to wake up)
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


def api_request_with_retry(method, url, retries=MAX_RETRIES, **kwargs):
    """
    Make an HTTP request with retry logic and exponential backoff.
    Handles Render cold-start timeouts gracefully.
    """
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            if method == "POST":
                resp = requests.post(url, **kwargs)
            else:
                resp = requests.get(url, **kwargs)

            # If server returned a response (even an error), return it
            return resp

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < retries:
                wait = BACKOFF_BASE ** attempt
                st.toast(f"⏳ API server waking up... retry {attempt}/{retries} in {wait}s", icon="🔄")
                time.sleep(wait)
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < retries:
                wait = BACKOFF_BASE ** attempt
                st.toast(f"⏳ Request timed out... retry {attempt}/{retries} in {wait}s", icon="⏱️")
                time.sleep(wait)
        except Exception as e:
            last_error = e
            break

    # All retries failed
    st.error(f"❌ Could not connect to API after {retries} attempts: {last_error}")
    return None


@st.cache_data(ttl=900)
def get_auth_token():
    """Get JWT auth token with retry logic for Render cold starts."""
    try:
        resp = api_request_with_retry(
            "POST",
            f"{API_URL}/login/access-token",
            data={"username": "admin", "password": "admin"},
        )
        if resp and resp.status_code == 200:
            return resp.json().get("access_token")
        elif resp:
            st.error(f"Auth failed: {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
    return None


def get_auth_headers(token):
    """Return the Authorization header dict."""
    return {"Authorization": f"Bearer {token}"} if token else {}
