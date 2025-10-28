import streamlit as st
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Derivio")

# --- ENUMS ---
class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black-Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'

# --- CACHE HELPERS ---
@st.cache_data
def get_historical_data(ticker):
    try:
        data = Ticker.get_historical_data(ticker)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_data
def get_current_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching current price for {ticker}: {str(e)}")
        return None

# --- STYLING ---
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden !important;}
[data-testid="stToolbarActionButton"],
[data-testid="stToolbar"] svg {display: none !important;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="font-family:Arial; font-size:2.3rem; font-weight:700;">
Derivio: Learn & Price Financial Options
</div>
<div style="font-size:1.2rem; color:#555; margin-bottom:20px;">
Understand options pricing through interactive lessons and real data.
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
mode = st.sidebar.selectbox("Choose Mode", ["Learn", "Calculate"])
pricing_method = st.sidebar.radio('Select Model', options=[m.value for m in OPTION_PRICING_MODEL])

# --- EDUCATIONAL CONTENT ---
def explain_black_scholes():
    st.subheader("📘 Understanding the Black-Scholes Model")
    st.markdown("""
    The **Black-Scholes Model** estimates the fair price of European call and put options.  
    It assumes no dividends, constant volatility, and a lognormal distribution of prices.

    **Formula (Call Option):**
    \n
    \\( C = S_0 N(d_1) - Ke^{-rt} N(d_2) \\)
    
    where:
    - \(S_0\): Current stock price  
    - \(K\): Strike price  
    - \(t\): Time to maturity  
    - \(r\): Risk-free rate  
    - \(σ\): Volatility  
    - \(N(x)\): Cumulative normal distribution  

    **Use Case:**  
    Traders use it to value options under stable conditions and estimate theoretical prices.

    **Interactive Insight:**  
    Try adjusting volatility and time-to-maturity below to see how price sensitivity changes.
    """)

def explain_monte_carlo():
    st.subheader("🎲 Monte Carlo Simulation")
    st.markdown("""
    The **Monte Carlo method** uses random sampling to simulate possible future stock prices.

    **Concept:**
    - Generate thousands of potential stock price paths.
    - Average the discounted payoffs to estimate the option price.

    **Formula (Simplified):**
    \n
    \\( C = e^{-rt} \\frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0) \\)
    
    **Use Case:**  
    Monte Carlo is great for complex or path-dependent options where analytical solutions are hard.
    """)

def explain_binomial():
    st.subheader("🌳 Binomial Tree Model")
    st.markdown("""
    The **Binomial Model** simulates how a stock price can move up or down each period until expiration.

    **Concept:**
    - Build a tree of possible future prices.
    - Work backwards to find today’s fair option price.

    **Formula:**
    \\( C = e^{-rt} [pC_u + (1-p)C_d] \\)
    
    where:
    - \(p = \\frac{e^{rt} - d}{u - d}\)
    - \(u, d\): up/down factors

    **Use Case:**  
    Excellent for American options, allowing early exercise before maturity.
    """)

# --- LEARN MODE ---
if mode == "Learn":
    st.info("📖 Select a pricing model from the sidebar to begin learning.")
    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        explain_black_scholes()
    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        explain_monte_carlo()
    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        explain_binomial()

    st.markdown("---")
    st.markdown("### 🧠 Try It Yourself")
    st.write("Switch to **Calculate Mode** in the sidebar to apply what you learned to real stock data.")

# --- CALCULATE MODE ---
elif mode == "Calculate":
    st.subheader(f'💻 Pricing Method: {pricing_method}')
    # (Here, reuse your existing parameter input & calculation code exactly as you have it)
    st.caption("Use this calculator to apply theoretical models to live market data.")
