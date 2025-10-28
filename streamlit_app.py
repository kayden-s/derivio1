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
<div style="font-family:Arial; font-size:1.5rem; font-weight:700; margin-bottom:0;">
Derivio
</div>
<div style="font-family:Arial; font-size:2.7rem; font-weight:700; line-height:1.2; margin-bottom:25px;">
Learn & Price Financial Options
</div>
<div style="font-size:1.2rem; color:#555; margin-bottom:25px;">
An interactive platform to understand and calculate option prices using real market data.
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
    It assumes constant volatility, no dividends, and lognormally distributed prices.

    **Formula (Call Option):**
    \\[
    C = S_0 N(d_1) - Ke^{-rt} N(d_2)
    \\]
    
    where:
    - \(S_0\): Current stock price  
    - \(K\): Strike price  
    - \(t\): Time to maturity  
    - \(r\): Risk-free rate  
    - \(σ\): Volatility  
    - \(N(x)\): Cumulative normal distribution  

    **Use Case:**  
    Used for European options and theoretical valuation under stable market assumptions.
    """)

def explain_monte_carlo():
    st.subheader("🎲 Monte Carlo Simulation")
    st.markdown("""
    The **Monte Carlo method** uses random sampling to estimate option prices.  
    It’s ideal for complex derivatives where closed-form formulas don’t exist.

    **Concept:**
    - Generate thousands of potential future stock price paths.  
    - Average the discounted payoffs to estimate the fair value.

    **Formula (Simplified):**
    \\[
    C = e^{-rt} \\frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)
    \\]

    **Use Case:**  
    Ideal for exotic or path-dependent options like Asian or barrier options.
    """)

def explain_binomial():
    st.subheader("🌳 Binomial Tree Model")
    st.markdown("""
    The **Binomial Model** models stock prices as moving up or down each time step until expiration.  
    It’s especially useful for **American options** (which can be exercised early).

    **Formula:**
    \\[
    C = e^{-rt} [pC_u + (1-p)C_d]
    \\]

    where:
    - \(p = \\frac{e^{rt} - d}{u - d}\)
    - \(u, d\): Up/down factors per step

    **Use Case:**  
    Allows flexibility for varying assumptions and early exercise.
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
    st.write("Switch to **Calculate Mode** in the sidebar to apply these models to real market data.")

# --- CALCULATE MODE ---
elif mode == "Calculate":
    st.subheader(f'💻 Pricing Method: {pricing_method}')
    st.caption("Use this calculator to apply theoretical models to live market data.")
    
    # =====================
    # BLACK-SCHOLES MODEL
    # =====================
    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        # Make text uppercase
        st.markdown(
            """
            <style>
            [data-testid="stTextInput"] input {
                text-transform: uppercase;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Parameters for Black-Scholes model
        ticker = st.text_input('Ticker Symbol', 'AAPL')
        ticker = ticker.upper()
        st.caption("Enter stock symbol (e.g., AAPL for Apple).")
    
        # Fetch current price
        current_price = get_current_price(ticker)
        
        if current_price is not None:
            st.write(f"Current Price of {ticker}: ${current_price:.2f}")
            
            # Set default and min/max values based on current price
            default_strike = round(current_price, 2)
            min_strike = max(0.1, round(current_price * 0.5, 2))
            max_strike = round(current_price * 2, 2)
            
            strike_price = st.number_input('Strike Price', 
                                           min_value=min_strike, 
                                           max_value=max_strike, 
                                           value=default_strike, 
                                           step=0.01)
            st.caption(f"Price to exercise the option. Range: \${min_strike:.2f} to \${max_strike:.2f}.")
        else:
            strike_price = st.number_input('Strike price', min_value=0.01, value=100.0, step=0.01)
            st.caption("Price to exercise the option. Enter a valid ticker to see a suggested range.")
    
        risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5)
        st.caption("Annual return of a risk-free asset.")
    
        sigma = st.slider('Volatility (%)', 0, 100, 20)
        st.caption("Expected stock price fluctuation.")
    
        exercise_date = st.date_input('Exercise Date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        st.caption("Date when the option can be exercised.")
    
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        if st.button(f'Calculate Option Price'):
            try:
                with st.spinner('Fetching data...'):
                    data = get_historical_data(ticker)
    
                if data is not None and not data.empty:
                    spot_price = Ticker.get_last_price(data, 'Close')
                    risk_free_rate = risk_free_rate / 100
                    sigma = sigma / 100
                    days_to_maturity = (exercise_date - datetime.now().date()).days
                
                    BSM = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)
                    call_option_price = BSM.calculate_option_price('Call Option')
                    put_option_price = BSM.calculate_option_price('Put Option')
                
                    st.markdown(
                        """
                        <style>
                        [data-testid="stMetricValue"],
                        [data-testid="stMetricValue"] * {
                            font-size: 2.5rem !important;
                            font-weight: 600 !important;
                            line-height: 1 !important;
                        }
                    
                        [data-testid="stMetricLabel"],
                        [data-testid="stMetricLabel"] * {
                            font-size: 1.1rem !important;
                            font-weight: 500 !important;
                        }
                    
                        [data-testid="stMetricLabel"] {
                            margin-bottom: 0.5rem !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    col1, col2, spacer = st.columns([1, 1, 2])
                    with col1:
                        st.metric("Call Option Price", f"${call_option_price:.2f}")
                    with col2:
                        st.metric("Put Option Price", f"${put_option_price:.2f}")
                    
                    st.markdown(
                        '<hr style="margin-top:15px; margin-bottom:20px; border:0; border-top:1px solid #ccc;">',
                        unsafe_allow_html=True
                    )
                
                    st.write("Data Fetched Successfully")
                    st.write(data.tail())
                    
                else:
                    st.error("Unable to proceed with calculations due to data fetching error.")
            except Exception as e:
                st.error(f"Error during calculation: {str(e)}")

    
    # =====================
    # MONTE CARLO MODEL
    # =====================
    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        # Make text uppercase
        st.markdown(
            """
            <style>
            [data-testid="stTextInput"] input {
                text-transform: uppercase;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Parameters for Monte Carlo simulation
        ticker = st.text_input('Ticker Symbol', 'AAPL')
        ticker = ticker.upper()
        st.caption("Enter stock symbol (e.g., AAPL for Apple).")
    
        # Fetch current price
        current_price = get_current_price(ticker)
        
        if current_price is not None:
            st.write(f"Current Price of {ticker}: ${current_price:.2f}")
            
            # Set default and min/max values based on current price
            default_strike = round(current_price, 2)
            min_strike = max(0.1, round(current_price * 0.5, 2))
            max_strike = round(current_price * 2, 2)
            
            strike_price = st.number_input('Strike Price', 
                                           min_value=min_strike, 
                                           max_value=max_strike, 
                                           value=default_strike, 
                                           step=0.01)
            st.caption(f"Price to exercise the option. Range: \${min_strike:.2f} to ${max_strike:.2f}.")
        else:
            strike_price = st.number_input('Strike price', min_value=0.01, value=100.0, step=0.01)
            st.caption("Price to exercise the option. Enter a valid ticker to see a suggested range.")
    
        risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5)
        st.caption("Annual return of a risk-free asset.")
    
        sigma = st.slider('Volatility (%)', 0, 100, 20)
        st.caption("Expected stock price fluctuation.")
    
        exercise_date = st.date_input('Exercise Date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        st.caption("Date when the option can be exercised.")
    
        number_of_simulations = st.slider('Number of Simulations', 100, 100000, 10000)
        st.caption("Number of price paths to simulate.")
    
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        if st.button(f'Calculate Option Price'):
            try:
                with st.spinner('Fetching data...'):
                    data = get_historical_data(ticker)
                
                if data is not None and not data.empty:
                    spot_price = Ticker.get_last_price(data, 'Close')
                    risk_free_rate = risk_free_rate / 100
                    sigma = sigma / 100
                    days_to_maturity = (exercise_date - datetime.now().date()).days
                
                    MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
                    MC.simulate_prices()
                
                    call_option_price = MC.calculate_option_price('Call Option')
                    put_option_price = MC.calculate_option_price('Put Option')
                
                    st.markdown(
                        """
                        <style>
                        [data-testid="stMetricValue"],
                        [data-testid="stMetricValue"] * {
                            font-size: 2.5rem !important;
                            font-weight: 600 !important;
                            line-height: 1 !important;
                        }
                    
                        [data-testid="stMetricLabel"],
                        [data-testid="stMetricLabel"] * {
                            font-size: 1.1rem !important;
                            font-weight: 500 !important;
                        }
                    
                        [data-testid="stMetricLabel"] {
                            margin-bottom: 0.5rem !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    col1, col2, spacer = st.columns([1, 1, 2])
                    with col1:
                        st.metric("Call Option Price", f"${call_option_price:.2f}")
                    with col2:
                        st.metric("Put Option Price", f"${put_option_price:.2f}")
                    
                    st.markdown(
                        '<hr style="margin-top:15px; margin-bottom:20px; border:0; border-top:1px solid #ccc;">',
                        unsafe_allow_html=True
                    )
                
                    st.write("Data Fetched Successfully")
                    st.write(data.tail())
                
                else:
                    st.error("Unable to proceed with calculations due to data fetching error.")
            except Exception as e:
                st.error(f"Error during calculation: {str(e)}")
    
    # =====================
    # BINOMIAL TREE MODEL
    # =====================
    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        # Make text uppercase
        st.markdown(
            """
            <style>
            [data-testid="stTextInput"] input {
                text-transform: uppercase;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Parameters for Binomial-Tree model
        ticker = st.text_input('Ticker Symbol', 'AAPL')
        ticker = ticker.upper()
        st.caption("Enter stock symbol (e.g., AAPL for Apple).")
    
        # Fetch current price
        current_price = get_current_price(ticker)
        
        if current_price is not None:
            st.write(f"Current Price of {ticker}: ${current_price:.2f}")
            
            # Set default and min/max values based on current price
            default_strike = round(current_price, 2)
            min_strike = max(0.1, round(current_price * 0.5, 2))
            max_strike = round(current_price * 2, 2)
            
            strike_price = st.number_input('Strike Price', 
                                           min_value=min_strike, 
                                           max_value=max_strike, 
                                           value=default_strike, 
                                           step=0.01)
            st.caption(f"Price to exercise the option. Range: \${min_strike:.2f} to \${max_strike:.2f}.")
        else:
            strike_price = st.number_input('Strike Price', min_value=0.01, value=100.0, step=0.01)
            st.caption("Price to exercise the option. Enter a valid ticker to see a suggested range.")
    
        risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5)
        st.caption("Annual return of a risk-free asset.")
    
        sigma = st.slider('Volatility (%)', 0, 100, 20)
        st.caption("Expected stock price fluctuation.")
    
        exercise_date = st.date_input('Exercise Date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        st.caption("Date when the option can be exercised.")
    
        number_of_time_steps = st.slider('Number of Time Steps', 5000, 100000, 15000)
        st.caption("Number of periods in the binomial tree.")
    
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        if st.button(f'Calculate Option Price'):
            try:
                with st.spinner('Fetching data...'):
                    data = get_historical_data(ticker)
                
                if data is not None and not data.empty:
                    spot_price = Ticker.get_last_price(data, 'Close')
                    risk_free_rate = risk_free_rate / 100
                    sigma = sigma / 100
                    days_to_maturity = (exercise_date - datetime.now().date()).days
                
                    BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps)
                    call_option_price = BOPM.calculate_option_price('Call Option')
                    put_option_price = BOPM.calculate_option_price('Put Option')
    
                    st.markdown(
                        """
                        <style>
                        [data-testid="stMetricValue"],
                        [data-testid="stMetricValue"] * {
                            font-size: 2.5rem !important;
                            font-weight: 600 !important;
                            line-height: 1 !important;
                        }
                    
                        [data-testid="stMetricLabel"],
                        [data-testid="stMetricLabel"] * {
                            font-size: 1.1rem !important;
                            font-weight: 500 !important;
                        }
                    
                        [data-testid="stMetricLabel"] {
                            margin-bottom: 0.5rem !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    col1, col2, spacer = st.columns([1, 1, 2])
                    with col1:
                        st.metric("Call Option Price", f"${call_option_price:.2f}")
                    with col2:
                        st.metric("Put Option Price", f"${put_option_price:.2f}")
                    
                    st.markdown(
                        '<hr style="margin-top:15px; margin-bottom:20px; border:0; border-top:1px solid #ccc;">',
                        unsafe_allow_html=True
                    )
    
                    st.write("Data Fetched Successfully")
                    st.write(data.tail())
                    
                else:
                    st.error("Unable to proceed with calculations due to data fetching error.")
            except Exception as e:
                st.error(f"Error during calculation: {str(e)}")
