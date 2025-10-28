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
<div style="font-family:Arial; font-size:1.5rem; font-weight:700; margin-top:0; margin-bottom:0;">
Derivio
</div>
<div style="font-family:Arial; font-size:2.7rem; font-weight:700; line-height:1.2; margin-top:0; margin-bottom:10px;">
Learn & Price Financial Options
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
if "mode" not in st.session_state:
    st.session_state["mode"] = "Learn"

mode = st.sidebar.selectbox("Choose Mode", ["Learn", "Calculate"], index=["Learn", "Calculate"].index(st.session_state["mode"]))
pricing_method = st.sidebar.radio('Select Model', options=[m.value for m in OPTION_PRICING_MODEL])

# --- EDUCATIONAL CONTENT ---
def explain_black_scholes():
    st.subheader("Understanding the Black-Scholes Model")
    st.markdown(r"""
    The Black-Scholes Model (BSM) is one of the most important concepts in modern finance. It provides a closed-form mathematical formula to estimate the fair value of European call and put options, which give the holder the right (but not the obligation) to buy or sell an asset at a predetermined price.

    ---
    ### Core Idea
    The model assumes that:
    - The stock price follows a lognormal distribution (prices cannot go below zero)
    - Volatility and the risk-free rate remain constant over time
    - The market is frictionless, meaning there are no transaction costs or taxes
    - The option can only be exercised at expiration (European-style)
    """)

    st.markdown("---")
    st.markdown("### Formula (Call Option)")
    st.latex(r"C = S_0 N(d_1) - K e^{-rt} N(d_2)")
    st.latex(r"d_1 = \frac{\ln(S_0 / K) + (r + \frac{\sigma^2}{2})t}{\sigma \sqrt{t}}, \quad d_2 = d_1 - \sigma \sqrt{t}")

    st.markdown(r"""
    **Definitions:**
    - **$S_0$**: Current stock price  
    - **$K$**: Strike price  
    - **$t$**: Time to maturity (in years)  
    - **$r$**: Annual risk-free interest rate  
    - **$\sigma$**: Annualized volatility of returns  
    - **$N(x)$**: Cumulative distribution function of a standard normal variable  

    ---
    ### Intuition
    - The model calculates what an option should cost based on how risky the stock is and how much time remains until expiration.  
    - If volatility or time increase, the option becomes more valuable.  
    - If interest rates rise, call options increase slightly in value, while puts decrease.

    ---
    ### Real-World Application
    - Commonly used by traders and analysts to compare market prices to theoretical values  
    - Helps detect overvalued or undervalued options  
    - Useful for risk management and derivatives trading in equity, forex, and commodities markets  

    ---
    **Best For:**  
    European-style options and markets with relatively stable volatility and no early exercise.
    """)


def explain_monte_carlo():
    st.subheader("Monte Carlo Simulation")
    st.markdown(r"""
    The Monte Carlo Simulation is a flexible, computational approach used to estimate option prices through random sampling. Instead of relying on a single formula, it simulates thousands of potential future paths of the stock price to calculate an average expected payoff.

    ---
    ### Core Idea
    - Simulate how the stock price could evolve over time using randomized returns that follow a statistical distribution (often normal)  
    - For each simulated path, calculate the payoff of the option at expiration  
    - Average all payoffs and discount them back to the present using the risk-free rate
    """)

    st.markdown("---")
    st.markdown("### Simplified Formula")
    st.latex(r"C = e^{-rt} \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)")

    st.markdown(r"""
    **Definitions:**
    - **$N$**: Number of simulations  
    - **$S_T^{(i)}$**: Simulated price at expiration for the i-th path  
    - **$r$**: Annual risk-free rate  
    - **$t$**: Time to maturity (in years)  
    - **$K$**: Strike price  

    ---
    ### Intuition
    - Think of it as running a “what-if” experiment thousands of times.  
    - Each simulation represents one possible market future.  
    - Averaging them smooths out randomness to reveal a probabilistic fair price.

    ---
    ### Advantages
    - Works with any payoff structure, including exotic or path-dependent options such as Asian or barrier options  
    - Can incorporate changing volatility, interest rates, or dividends  
    - Scales easily with computational power  

    ---
    ### Limitations
    - Computationally intensive, especially with many simulations  
    - Accuracy depends on the number of simulations, with more runs improving precision  

    ---
    **Best For:**  
    Complex derivatives, exotic options, or when analytical models like Black-Scholes cannot be applied.
    """)


def explain_binomial():
    st.subheader("Binomial Tree Model")
    st.markdown(r"""
    The Binomial Tree Model provides a step-by-step, discrete-time framework for valuing options. It models the stock price as moving up (u) or down (d) during each small time interval until expiration, forming a tree of possible outcomes.

    ---
    ### Core Idea
    - Each time step represents a possible price change  
    - At every node, the stock either increases by u or decreases by d  
    - The option value is calculated backward from expiration to the present using the risk-neutral probability
    """)

    st.markdown("---")
    st.markdown("### Key Formulas")
    st.latex(r"p = \frac{e^{rt} - d}{u - d}")
    st.latex(r"C = e^{-rt} [pC_u + (1 - p)C_d]")

    st.markdown(r"""
    **Definitions:**
    - **$C_u$**: Option value after an upward move  
    - **$C_d$**: Option value after a downward move  
    - **$u$**: Upward price multiplier  
    - **$d$**: Downward price multiplier  
    - **$r$**: Annual risk-free rate  
    - **$t$**: Time step length  

    ---
    ### Intuition
    - The model works like a decision tree.  
    - It simulates all possible paths the stock might take, computes the option value at the end of each branch, and works backward using probabilities to find today’s fair price.

    ---
    ### Advantages
    - Can model American options that allow early exercise, unlike Black-Scholes  
    - Intuitive and adaptable to dividends, changing volatility, or discrete events  
    - Accuracy increases with the number of time steps  

    ---
    ### Limitations
    - Slower than closed-form models for very large trees  
    - Requires many steps to approximate continuous processes accurately  

    ---
    **Best For:**  
    American-style options or any scenario where early exercise and flexibility are important.
    """)

# --- LEARN MODE ---
if mode == "Learn":
    st.markdown(
        """
        <div style="
            background-color: #E4F2FD;
            padding: 14px 18px;
            border-radius: 8px;
            color: #59A9F1;
            font-weight: 500;
            font-size: 16px;
            line-height: 1.3;
            margin-top: 20px;
            margin-bottom: 5px;
            ">
            Select a pricing model from the sidebar to begin learning.
        </div>
        """,
        unsafe_allow_html=True
    )

    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        explain_black_scholes()
    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        explain_monte_carlo()
    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        explain_binomial()

    st.markdown("---")
    st.markdown("### Try It Yourself")
    st.write("Switch to Calculate Mode in the sidebar to apply these models to real market data.")

    # --- Go to Calculator Button ---
    if st.button("Go to Calculator Mode", type="primary"):
        st.session_state["mode"] = "Calculate"
        st.experimental_rerun()

# --- CALCULATE MODE ---
elif mode == "Calculate":
    st.subheader(f'Pricing Method: {pricing_method}')
    
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
