import streamlit as st
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker
import matplotlib.pyplot as plt
import json

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black-Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'

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

# Main title
st.markdown(
    """
    <div style="font-family: 'Arial', sans-serif; font-size: 1.5rem; font-weight: 700; margin-bottom: 0px;">
        Derivio
    </div>
    <div style="font-family: 'Arial', sans-serif; font-size: 2.7rem; font-weight: 700; line-height: 1.2; margin-bottom: 25px;">
        Option Pricing Calculator
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar model selector
pricing_method = st.sidebar.radio('Pricing method', options=[model.value for model in OPTION_PRICING_MODEL])
st.subheader(f'Pricing method: {pricing_method}')

# --- Shared styling for uppercase ticker input ---
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

def render_common_inputs(model_name):
    """Render common UI inputs across pricing models."""
    ticker = st.text_input('Ticker Symbol', 'AAPL')
    ticker = ticker.upper()
    st.caption("Enter stock symbol (e.g., AAPL for Apple Inc.).")

    current_price = get_current_price(ticker)
    if current_price is not None:
        st.write(f"**Current Price:** ${current_price:.2f}")

        default_strike = round(current_price, 2)
        min_strike = max(0.1, round(current_price * 0.5, 2))
        max_strike = round(current_price * 2, 2)

        strike_price = st.number_input(
            'Strike Price',
            min_value=min_strike,
            max_value=max_strike,
            value=default_strike,
            step=0.01
        )
        st.caption(f"Price to exercise the option. *(Range: ${min_strike:.2f}–${max_strike:.2f})*")
    else:
        strike_price = st.number_input('Strike Price', min_value=0.01, value=100.0, step=0.01)
        st.caption("Price to exercise the option. Enter a valid ticker to view a suggested range.")

    risk_free_rate = st.slider('Risk-Free Rate (%)', 0, 100, 5)
    st.caption("Annual return of a zero-risk investment, typically based on government bonds. *(0% = no return, 100% = double your money risk-free — unrealistic)*")

    sigma = st.slider('Volatility (σ) (%)', 0, 100, 20)
    st.caption("Expected variability of the stock’s price. *(0% = no volatility, 100% = extremely volatile)*")

    exercise_date = st.date_input(
        'Exercise Date',
        min_value=datetime.today() + timedelta(days=1),
        value=datetime.today() + timedelta(days=365)
    )
    st.caption("Date when the option can be exercised.")

    return ticker, strike_price, risk_free_rate, sigma, exercise_date, current_price


# --- Black-Scholes Model ---
if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    ticker, strike_price, risk_free_rate, sigma, exercise_date, current_price = render_common_inputs("Black-Scholes")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    if st.button(f'Calculate Option Price for {ticker}'):
        try:
            with st.spinner('Fetching data...'):
                data = get_historical_data(ticker)

            if data is not None and not data.empty:
                spot_price = Ticker.get_last_price(data, 'Close')
                risk_free_rate /= 100
                sigma /= 100
                days_to_maturity = (exercise_date - datetime.now().date()).days

                BSM = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)
                call_price = BSM.calculate_option_price('Call Option')
                put_price = BSM.calculate_option_price('Put Option')

                col1, col2, _ = st.columns([1, 1, 2])
                with col1:
                    st.metric("Call Option Price", f"${call_price:.2f}")
                with col2:
                    st.metric("Put Option Price", f"${put_price:.2f}")

                st.markdown('<hr>', unsafe_allow_html=True)
                st.write("Recent Data:")
                st.write(data.tail())

                fig = Ticker.plot_data(data, ticker, 'Close')
                st.pyplot(fig)
            else:
                st.error("Unable to fetch data for calculations.")
        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")


# --- Monte Carlo Simulation ---
elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    ticker, strike_price, risk_free_rate, sigma, exercise_date, current_price = render_common_inputs("Monte Carlo")

    number_of_simulations = st.slider('Number of Simulations', 100, 100000, 10000)
    st.caption("Number of simulated price paths. More simulations improve accuracy but take longer.")

    num_of_movements = st.slider('Paths Displayed on Graph', 0, int(number_of_simulations / 10), 100)
    st.caption("Number of simulated price paths to display in the visualization.")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    if st.button(f'Calculate Option Price for {ticker}'):
        try:
            with st.spinner('Fetching data...'):
                data = get_historical_data(ticker)
            
            if data is not None and not data.empty:
                spot_price = Ticker.get_last_price(data, 'Close')
                risk_free_rate /= 100
                sigma /= 100
                days_to_maturity = (exercise_date - datetime.now().date()).days

                MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
                MC.simulate_prices()

                call_price = MC.calculate_option_price('Call Option')
                put_price = MC.calculate_option_price('Put Option')

                col1, col2, _ = st.columns([1, 1, 2])
                with col1:
                    st.metric("Call Option Price", f"${call_price:.2f}")
                with col2:
                    st.metric("Put Option Price", f"${put_price:.2f}")

                st.markdown('<hr>', unsafe_allow_html=True)
                st.write("Recent Data:")
                st.write(data.tail())

                st.pyplot(Ticker.plot_data(data, ticker, 'Close'))
                st.pyplot(MC.plot_simulation_results(num_of_movements))
            else:
                st.error("Unable to fetch data for calculations.")
        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")


# --- Binomial Model ---
elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    ticker, strike_price, risk_free_rate, sigma, exercise_date, current_price = render_common_inputs("Binomial")

    number_of_steps = st.slider('Number of Time Steps', 5000, 100000, 15000)
    st.caption("Number of periods in the binomial tree. More steps improve accuracy but take longer.")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    if st.button(f'Calculate Option Price for {ticker}'):
        try:
            with st.spinner('Fetching data...'):
                data = get_historical_data(ticker)

            if data is not None and not data.empty:
                spot_price = Ticker.get_last_price(data, 'Close')
                risk_free_rate /= 100
                sigma /= 100
                days_to_maturity = (exercise_date - datetime.now().date()).days

                BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_steps)
                call_price = BOPM.calculate_option_price('Call Option')
                put_price = BOPM.calculate_option_price('Put Option')

                col1, col2, _ = st.columns([1, 1, 2])
                with col1:
                    st.metric("Call Option Price", f"${call_price:.2f}")
                with col2:
                    st.metric("Put Option Price", f"${put_price:.2f}")

                st.markdown('<hr>', unsafe_allow_html=True)
                st.write("Recent Data:")
                st.write(data.tail())

                st.pyplot(Ticker.plot_data(data, ticker, 'Close'))
            else:
                st.error("Unable to fetch data for calculations.")
        except Exception as e:
            st.error(f"Error during calculation: {str(e)}")
