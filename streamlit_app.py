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

# --- STYLING & HEADER (kept as you had) ---
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden !important;}
[data-testid="stToolbarActionButton"],
[data-testid="stToolbar"] svg {display: none !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:Arial; font-size:1.5rem; font-weight:700; margin-top:0; margin-bottom:0;">
Derivio
</div>
<div style="font-family:Arial; font-size:2.7rem; font-weight:700; line-height:1.2; margin-top:0; margin-bottom:10px;">
Learn & Price Financial Options
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR STATE ---
if "mode" not in st.session_state:
    st.session_state["mode"] = "Learn"

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Learn", "Calculate"],
    index=["Learn", "Calculate"].index(st.session_state["mode"])
)
pricing_method = st.sidebar.radio(
    'Select Model',
    options=[m.value for m in OPTION_PRICING_MODEL]
)

# --- EDUCATIONAL CONTENT FUNCTIONS (kept) ---
def explain_black_scholes():
    st.subheader("Understanding the Black-Scholes Model")
    st.markdown(r"""
    The Black-Scholes Model (BSM) is one of the most important concepts in modern finance. It provides a closed-form mathematical formula to estimate the fair value of European call and put options...
    """)
    st.markdown("### Formula (Call Option)")
    st.latex(r"C = S_0 N(d_1) - K e^{-rt} N(d_2)")
    st.latex(r"d_1 = \frac{\ln(S_0 / K) + (r + \frac{\sigma^2}{2})t}{\sigma \sqrt{t}}, \quad d_2 = d_1 - \sigma \sqrt{t}")
    # (rest omitted for brevity; keep your original content)

def explain_monte_carlo():
    st.subheader("Monte Carlo Simulation")
    st.markdown("...")  # keep your content

def explain_binomial():
    st.subheader("Binomial Tree Model")
    st.markdown("...")  # keep your content

# --- Calculator functions (prevent NameError) ---
def calculate_black_scholes():
    st.subheader("Black-Scholes Model")

    ticker = st.text_input('Ticker Symbol', 'AAPL', key="bs_ticker").upper()
    current_price = get_current_price(ticker)

    if current_price:
        st.write(f"Current Price of {ticker}: ${current_price:.2f}")
        default_strike = round(current_price, 2)
        min_strike = max(0.01, round(current_price * 0.5, 2))
        max_strike = round(current_price * 2, 2)
    else:
        default_strike, min_strike, max_strike = 100.0, 0.01, 200.0

    strike_price = st.number_input('Strike Price', min_value=min_strike, max_value=max_strike, value=default_strike, step=0.01, key="bs_strike")
    risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5, key="bs_rfr")
    sigma = st.slider('Volatility (%)', 0, 100, 20, key="bs_sigma")
    exercise_date = st.date_input('Exercise Date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365), key="bs_exdate")

    if st.button('Calculate Option Price', key="bs_calc"):
        try:
            data = get_historical_data(ticker)
            if data is not None and not data.empty:
                spot_price = Ticker.get_last_price(data, 'Close')
                days_to_maturity = max(1, (exercise_date - datetime.now().date()).days)
                BSM = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate / 100, sigma / 100)
                call_price = BSM.calculate_option_price('Call Option')
                put_price = BSM.calculate_option_price('Put Option')

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Call Option Price", f"${call_price:.2f}")
                with col2:
                    st.metric("Put Option Price", f"${put_price:.2f}")

                st.markdown('<hr/>', unsafe_allow_html=True)
                st.write("Data tail:")
                st.write(data.tail())
            else:
                st.error("Unable to fetch historical data for calculations.")
        except Exception as e:
            st.error(f"Error during calculation: {e}")

def calculate_monte_carlo():
    st.subheader("Monte Carlo Simulation")

    ticker = st.text_input('Ticker Symbol', 'AAPL', key="mc_ticker").upper()
    current_price = get_current_price(ticker)

    if current_price:
        st.write(f"Current Price of {ticker}: ${current_price:.2f}")
        default_strike = round(current_price, 2)
        min_strike = max(0.01, round(current_price * 0.5, 2))
        max_strike = round(current_price * 2, 2)
    else:
        default_strike, min_strike, max_strike = 100.0, 0.01, 200.0

    strike_price = st.number_input('Strike Price', min_value=min_strike, max_value=max_strike, value=default_strike, step=0.01, key="mc_strike")
    risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5, key="mc_rfr")
    sigma = st.slider('Volatility (%)', 0, 100, 20, key="mc_sigma")
    exercise_date = st.date_input('Exercise Date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365), key="mc_exdate")
    n_sims = st.slider('Number of Simulations', 100, 100000, 10000, key="mc_n_sims")

    if st.button('Calculate Option Price', key="mc_calc"):
        try:
            data = get_historical_data(ticker)
            if data is not None and not data.empty:
                spot_price = Ticker.get_last_price(data, 'Close')
                days_to_maturity = max(1, (exercise_date - datetime.now().date()).days)
                MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate / 100, sigma / 100, n_sims)
                MC.simulate_prices()
                call_price = MC.calculate_option_price('Call Option')
                put_price = MC.calculate_option_price('Put Option')

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Call Option Price", f"${call_price:.2f}")
                with col2:
                    st.metric("Put Option Price", f"${put_price:.2f}")

                st.markdown('<hr/>', unsafe_allow_html=True)
                st.write("Data tail:")
                st.write(data.tail())
            else:
                st.error("Unable to fetch historical data for calculations.")
        except Exception as e:
            st.error(f"Error during calculation: {e}")

def calculate_binomial():
    st.subheader("Binomial Tree Model")

    ticker = st.text_input('Ticker Symbol', 'AAPL', key="bt_ticker").upper()
    current_price = get_current_price(ticker)

    if current_price:
        st.write(f"Current Price of {ticker}: ${current_price:.2f}")
        default_strike = round(current_price, 2)
        min_strike = max(0.01, round(current_price * 0.5, 2))
        max_strike = round(current_price * 2, 2)
    else:
        default_strike, min_strike, max_strike = 100.0, 0.01, 200.0

    strike_price = st.number_input('Strike Price', min_value=min_strike, max_value=max_strike, value=default_strike, step=0.01, key="bt_strike")
    risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5, key="bt_rfr")
    sigma = st.slider('Volatility (%)', 0, 100, 20, key="bt_sigma")
    exercise_date = st.date_input('Exercise Date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365), key="bt_exdate")
    n_steps = st.slider('Number of Time Steps', 5000, 100000, 15000, key="bt_n_steps")

    if st.button('Calculate Option Price', key="bt_calc"):
        try:
            data = get_historical_data(ticker)
            if data is not None and not data.empty:
                spot_price = Ticker.get_last_price(data, 'Close')
                days_to_maturity = max(1, (exercise_date - datetime.now().date()).days)
                BT = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate / 100, sigma / 100, n_steps)
                call_price = BT.calculate_option_price('Call Option')
                put_price = BT.calculate_option_price('Put Option')

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Call Option Price", f"${call_price:.2f}")
                with col2:
                    st.metric("Put Option Price", f"${put_price:.2f}")

                st.markdown('<hr/>', unsafe_allow_html=True)
                st.write("Data tail:")
                st.write(data.tail())
            else:
                st.error("Unable to fetch historical data for calculations.")
        except Exception as e:
            st.error(f"Error during calculation: {e}")

# --- LEARN MODE UI ---
if mode == "Learn":
    st.markdown("""
    <div style="
        background-color: #E4F2FD;
        padding: 14px 18px;
        border-radius: 8px;
        color: #59A9F1;
        font-weight: 500;
        font-size: 16px;
        line-height: 1.3;
        margin-top: 20px;
        margin-bottom: 5px;">
        Select a pricing model from the sidebar to begin learning.
    </div>
    """, unsafe_allow_html=True)

    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        explain_black_scholes()
    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        explain_monte_carlo()
    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        explain_binomial()

    st.markdown("---")
    st.markdown("### Try It Yourself")
    st.write("Switch to Calculate Mode in the sidebar to apply these models to real market data.")

    # Button that safely switches to Calculate mode
    if st.button("Go to Calculator Mode", type="primary", key="to_calc"):
        st.session_state["mode"] = "Calculate"
        st.rerun()

# --- CALCULATE MODE UI ---
elif mode == "Calculate":
    st.markdown("## Option Calculator")
    # Back button
    if st.button("⬅ Back to Learn Mode", key="to_learn"):
        st.session_state["mode"] = "Learn"
        st.rerun()

    # Dispatch to the right calculator
    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        calculate_black_scholes()
    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        calculate_monte_carlo()
    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        calculate_binomial()

    st.markdown("---")
    st.write("Tip: change the model in the sidebar to switch between pricing methods.")
