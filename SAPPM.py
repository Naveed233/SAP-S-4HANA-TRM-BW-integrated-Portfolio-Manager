import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as si
import cvxpy as cp
import matplotlib.pyplot as plt
import datetime
import requests

# =============================================================================
# 1. ASSET PRICING MODELS & RISK METRICS
# =============================================================================

# ---------- A. BONDS ----------
class Bond:
    def __init__(self, face_value, coupon_rate, maturity_date, frequency=1):
        """
        face_value: Bond's face value (e.g., 1000)
        coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
        maturity_date: Maturity date as string 'YYYY-MM-DD'
        frequency: Number of coupon payments per year (1 for annual, 2 for semi-annual, etc.)
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_date = pd.to_datetime(maturity_date)
        self.frequency = frequency

    def cash_flows(self, issue_date=None):
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        dates = []
        current_date = self.maturity_date
        while current_date > issue_date:
            dates.append(current_date)
            months = int(12 / self.frequency)
            current_date -= pd.DateOffset(months=months)
        dates = sorted(dates)
        coupon_payment = self.face_value * self.coupon_rate / self.frequency
        payments = [coupon_payment] * len(dates)
        if payments:
            payments[-1] += self.face_value
        return dates, payments

    def price(self, yield_rate, issue_date=None):
        dates, payments = self.cash_flows(issue_date)
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        price = 0.0
        for date, payment in zip(dates, payments):
            t = (date - issue_date).days / 365.0
            price += payment / (1 + yield_rate) ** t
        return price

    def duration(self, yield_rate, issue_date=None):
        dates, payments = self.cash_flows(issue_date)
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        bond_price = self.price(yield_rate, issue_date)
        duration = 0.0
        for date, payment in zip(dates, payments):
            t = (date - issue_date).days / 365.0
            duration += t * (payment / (1 + yield_rate) ** t)
        return duration / bond_price

    def convexity(self, yield_rate, issue_date=None):
        dates, payments = self.cash_flows(issue_date)
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        bond_price = self.price(yield_rate, issue_date)
        convexity = 0.0
        for date, payment in zip(dates, payments):
            t = (date - issue_date).days / 365.0
            convexity += t * (t + 1) * (payment / (1 + yield_rate) ** (t + 2))
        return convexity / bond_price

# ---------- B. DERIVATIVES ----------
# 1. Options (European & American)
class Option:
    def __init__(self, option_type, S, K, T, r, sigma, dividend_yield=0):
        """
        option_type: 'call' or 'put'
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        dividend_yield: Annual dividend yield (if any)
        """
        self.option_type = option_type.lower()
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dividend_yield = dividend_yield

    def black_scholes_price(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.dividend_yield + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == 'call':
            price = (self.S * np.exp(-self.dividend_yield * self.T) * si.norm.cdf(d1, 0, 1) -
                     self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0, 1))
        else:
            price = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0, 1) -
                     self.S * np.exp(-self.dividend_yield * self.T) * si.norm.cdf(-d1, 0, 1))
        return price

    def binomial_tree_price(self, steps=100):
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.r - self.dividend_yield) * dt) - d) / (u - d)
        asset_prices = np.array([self.S * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)])
        if self.option_type == 'call':
            option_values = np.maximum(asset_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - asset_prices, 0)
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                asset_price = self.S * (u ** (j - i)) * (d ** i)
                if self.option_type == 'call':
                    option_values[i] = max(option_values[i], asset_price - self.K)
                else:
                    option_values[i] = max(option_values[i], self.K - asset_price)
        return option_values[0]

    def greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r - self.dividend_yield + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = si.norm.cdf(d1) if self.option_type == 'call' else -si.norm.cdf(-d1)
        return {"delta": delta}

# 2. Futures
class Future:
    def __init__(self, underlying_price, cost_of_carry, T):
        self.underlying_price = underlying_price
        self.cost_of_carry = cost_of_carry
        self.T = T

    def price(self):
        return self.underlying_price * np.exp(self.cost_of_carry * self.T)

# 3. Interest Rate Swaps
class InterestRateSwap:
    def __init__(self, notional, fixed_rate, floating_rates, payment_dates, discount_factors):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rates = np.array(floating_rates)
        self.payment_dates = payment_dates
        self.discount_factors = np.array(discount_factors)

    def price(self, issue_date=None):
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        fixed_leg = sum(self.fixed_rate * self.notional * df for df in self.discount_factors)
        floating_leg = sum(fr * self.notional * df for fr, df in zip(self.floating_rates, self.discount_factors))
        return floating_leg - fixed_leg

# =============================================================================
# 2. PORTFOLIO OPTIMIZATION & RISK ANALYSIS FUNCTIONS
# =============================================================================

def black_litterman(prior_returns, covariance, P, Q, tau=0.05, omega=None):
    prior_returns = np.array(prior_returns)
    covariance = np.array(covariance)
    P = np.array(P)
    Q = np.array(Q)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * covariance) @ P.T))
    inv_term = np.linalg.inv(np.linalg.inv(tau * covariance) + P.T @ np.linalg.inv(omega) @ P)
    adjusted_returns = inv_term @ (np.linalg.inv(tau * covariance) @ prior_returns + P.T @ np.linalg.inv(omega) @ Q)
    return adjusted_returns

class PortfolioOptimizer:
    def __init__(self, assets, expected_returns, covariance):
        self.assets = assets
        self.expected_returns = np.array(expected_returns)
        self.covariance = np.array(covariance)

    def optimize(self, target_return=None):
        n = len(self.assets)
        w = cp.Variable(n)
        port_return = self.expected_returns @ w
        port_variance = cp.quad_form(w, self.covariance)
        objective = cp.Minimize(port_variance)
        constraints = [cp.sum(w) == 1, w >= 0]
        if target_return is not None:
            constraints.append(port_return >= target_return)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value, port_return.value, port_variance.value

def monte_carlo_simulation_portfolio(weights, daily_returns, daily_vol, num_simulations=1000, num_periods=252):
    weights = np.array(weights)
    portfolio_paths = []
    for i in range(num_simulations):
        asset_paths = np.random.normal(loc=daily_returns, scale=daily_vol, size=(num_periods, len(weights)))
        port_daily_returns = asset_paths @ weights
        port_value = np.cumprod(1 + port_daily_returns)
        portfolio_paths.append(port_value)
    return np.array(portfolio_paths)

# =============================================================================
# 3. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_bond_price_vs_yield(bond, yield_min, yield_max, num_points=100):
    yields = np.linspace(yield_min, yield_max, num_points)
    prices = [bond.price(y) for y in yields]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(yields, prices, label='Bond Price', color='blue')
    ax.set_xlabel("Yield")
    ax.set_ylabel("Bond Price")
    ax.set_title("Bond Price vs. Yield")
    ax.legend()
    ax.grid(True)
    return fig

def plot_bond_duration_convexity_vs_yield(bond, yield_min, yield_max, num_points=100):
    yields = np.linspace(yield_min, yield_max, num_points)
    durations = [bond.duration(y) for y in yields]
    convexities = [bond.convexity(y) for y in yields]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(yields, durations, label='Duration', color='blue')
    ax.plot(yields, convexities, label='Convexity', color='red')
    ax.set_xlabel("Yield")
    ax.set_ylabel("Metric Value")
    ax.set_title("Bond Duration & Convexity vs. Yield")
    ax.legend()
    ax.grid(True)
    return fig

def plot_option_price_vs_underlying(option, underlying_range):
    prices = []
    for S in underlying_range:
        option.S = S  # update underlying price
        prices.append(option.black_scholes_price())
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(underlying_range, prices, label=f"{option.option_type.capitalize()} Option Price", color='green')
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Option Price")
    ax.set_title("Option Price vs. Underlying Asset Price")
    ax.legend()
    ax.grid(True)
    return fig

def generate_random_portfolios(expected_returns, covariance, num_portfolios=5000):
    n = len(expected_returns)
    results = np.zeros((num_portfolios, 3))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        port_return = np.dot(weights, expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        sharpe_ratio = port_return / port_volatility if port_volatility != 0 else 0
        results[i, 0] = port_volatility
        results[i, 1] = port_return
        results[i, 2] = sharpe_ratio
        weights_record.append(weights)
    return results, weights_record

def plot_efficient_frontier(expected_returns, covariance, optimal_weight=None, optimal_return=None, optimal_volatility=None, num_portfolios=5000):
    results, _ = generate_random_portfolios(expected_returns, covariance, num_portfolios)
    volatilities = results[:, 0]
    returns = results[:, 1]
    sharpe_ratios = results[:, 2]
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.5)
    ax.set_xlabel("Volatility (Std. Deviation)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    fig.colorbar(sc, label="Sharpe Ratio")
    if optimal_weight is not None:
        ax.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label='Optimal Portfolio')
        ax.legend()
    ax.grid(True)
    return fig

def plot_asset_allocation(assets, weights):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(weights, labels=assets, autopct='%1.1f%%', startangle=140)
    ax.set_title("Portfolio Asset Allocation")
    return fig

def plot_portfolio_simulations(portfolio_paths, num_paths_to_plot=50):
    fig, ax = plt.subplots(figsize=(10,6))
    for path in portfolio_paths[:num_paths_to_plot]:
        ax.plot(path, lw=0.7, alpha=0.6)
    ax.set_title("Monte Carlo Simulation of Portfolio Value Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True)
    return fig

def plot_final_value_distribution(portfolio_paths):
    final_values = portfolio_paths[:, -1]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(final_values, bins=50, edgecolor='k', alpha=0.7)
    ax.set_title("Distribution of Final Portfolio Values")
    ax.set_xlabel("Final Portfolio Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    return fig

# =============================================================================
# 4. STREAMLIT APP
# =============================================================================

st.set_page_config(page_title="Portfolio Optimization & Asset Analysis", layout="wide")
st.title("Portfolio Optimization & Asset Analysis App")

# Sidebar navigation
page = st.sidebar.radio("Navigation", 
                          ["Bond Analysis", "Option Analysis", "Portfolio Optimization"])

# -------------------------
# PAGE 1: Bond Analysis
# -------------------------
if page == "Bond Analysis":
    st.header("Bond Analysis")
    st.markdown("Adjust the bond parameters and yield range below to see how price, duration, and convexity change.")

    # Bond parameters input
    face_value = st.sidebar.number_input("Face Value", value=1000)
    coupon_rate = st.sidebar.slider("Coupon Rate", min_value=0.0, max_value=0.15, value=0.05, step=0.005)
    maturity_date = st.sidebar.text_input("Maturity Date (YYYY-MM-DD)", value="2030-12-31")
    frequency = st.sidebar.selectbox("Coupon Frequency (per year)", options=[1, 2, 4], index=1)

    # Yield range
    yield_min = st.sidebar.slider("Minimum Yield", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
    yield_max = st.sidebar.slider("Maximum Yield", min_value=0.05, max_value=0.2, value=0.1, step=0.005)

    # Create bond instance
    bond = Bond(face_value, coupon_rate, maturity_date, frequency)
    # Show sample bond metrics for a given yield (e.g., yield = 4%)
    sample_yield = 0.04
    st.write(f"At a yield of {sample_yield*100:.1f}%, the bond price is **{bond.price(sample_yield):.2f}**, "
             f"duration is **{bond.duration(sample_yield):.2f}**, and convexity is **{bond.convexity(sample_yield):.2f}**.")

    # Plot bond price vs yield
    fig1 = plot_bond_price_vs_yield(bond, yield_min, yield_max)
    st.pyplot(fig1)

    # Plot bond duration & convexity vs yield
    fig2 = plot_bond_duration_convexity_vs_yield(bond, yield_min, yield_max)
    st.pyplot(fig2)

# -------------------------
# PAGE 2: Option Analysis
# -------------------------
elif page == "Option Analysis":
    st.header("Option Analysis")
    st.markdown("Adjust the option parameters and see the price sensitivity to the underlying asset price.")

    # Option parameters input
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    S = st.sidebar.number_input("Underlying Price (S)", value=100)
    K = st.sidebar.number_input("Strike Price (K)", value=95)
    T = st.sidebar.number_input("Time to Expiration (T in years)", value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-free Rate (r)", value=0.03, step=0.005)
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
    dividend_yield = st.sidebar.number_input("Dividend Yield", value=0.01, step=0.005)

    option = Option(option_type, S, K, T, r, sigma, dividend_yield)
    euro_price = option.black_scholes_price()
    am_price = option.binomial_tree_price(steps=100)
    st.write(f"European {option_type.capitalize()} Price (Black-Scholes): **{euro_price:.2f}**")
    st.write(f"American {option_type.capitalize()} Price (Binomial Tree): **{am_price:.2f}**")
    st.write("Option Greeks:", option.greeks())

    # Underlying price range for plotting
    underlying_range = np.linspace(S * 0.8, S * 1.2, 100)
    fig3 = plot_option_price_vs_underlying(option, underlying_range)
    st.pyplot(fig3)

# -------------------------
# PAGE 3: Portfolio Optimization
# -------------------------
elif page == "Portfolio Optimization":
    st.header("Portfolio Optimization & Risk Analysis")
    st.markdown("This section demonstrates mean-variance optimization, efficient frontier, asset allocation, and Monte Carlo simulation for risk analysis.")

    # For demonstration, we use four asset classes.
    assets = ['Stock', 'Crypto', 'Bond', 'Derivative']
    # Input expected returns (annualized)
    st.sidebar.markdown("### Expected Returns (annualized)")
    er_stock = st.sidebar.number_input("Stock", value=0.08, step=0.01)
    er_crypto = st.sidebar.number_input("Crypto", value=0.12, step=0.01)
    er_bond = st.sidebar.number_input("Bond", value=0.05, step=0.01)
    er_deriv = st.sidebar.number_input("Derivative", value=0.10, step=0.01)
    expected_returns = [er_stock, er_crypto, er_bond, er_deriv]

    # For simplicity, use a dummy covariance matrix (annualized)
    covariance = [[0.1,   0.02,  0.01, 0.03],
                  [0.02,  0.2,   0.015,0.025],
                  [0.01,  0.015, 0.05, 0.02],
                  [0.03,  0.025, 0.02, 0.15]]

    # Optional Black-Litterman view on bonds
    st.sidebar.markdown("### Black-Litterman View")
    view_bond = st.sidebar.checkbox("Apply view on Bond returns", value=False)
    if view_bond:
        P = [[0, 0, 1, 0]]
        Q = [st.sidebar.number_input("Bond view (expected return)", value=0.055, step=0.005)]
        adjusted_returns = black_litterman(expected_returns, covariance, P, Q, tau=0.05)
    else:
        adjusted_returns = expected_returns

    st.write("Adjusted Expected Returns:", np.round(adjusted_returns, 4))

    # Optimize portfolio with target return (e.g., 7% annualized)
    target_return = st.sidebar.number_input("Target Return (annualized)", value=0.07, step=0.005)
    optimizer = PortfolioOptimizer(assets, adjusted_returns, covariance)
    weights, port_return, port_variance = optimizer.optimize(target_return=target_return)
    port_volatility = np.sqrt(port_variance)
    st.write("Optimized Portfolio Weights:", np.round(weights, 4))
    st.write(f"Portfolio Expected Return: **{port_return:.2f}**")
    st.write(f"Portfolio Volatility: **{port_volatility:.2f}**")

    # Plot efficient frontier with the optimal portfolio marked
    fig4 = plot_efficient_frontier(adjusted_returns, covariance, optimal_weight=weights,
                                   optimal_return=port_return, optimal_volatility=port_volatility)
    st.pyplot(fig4)

    # Plot asset allocation pie chart
    fig5 = plot_asset_allocation(assets, weights)
    st.pyplot(fig5)

    # Monte Carlo Simulation of the portfolio
    st.markdown("### Monte Carlo Simulation")
    daily_expected_returns = np.array(expected_returns) / 252
    daily_vol = st.sidebar.number_input("Assumed Daily Volatility", value=0.01, step=0.001)
    portfolio_paths = monte_carlo_simulation_portfolio(weights, daily_expected_returns, daily_vol,
                                                       num_simulations=500, num_periods=252)
    fig6 = plot_portfolio_simulations(portfolio_paths, num_paths_to_plot=30)
    st.pyplot(fig6)
    fig7 = plot_final_value_distribution(portfolio_paths)
    st.pyplot(fig7)
