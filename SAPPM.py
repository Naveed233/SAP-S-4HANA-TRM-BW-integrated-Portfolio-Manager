import numpy as np
import pandas as pd
import scipy.stats as si
import cvxpy as cp
import matplotlib.pyplot as plt
import datetime
import requests

# =============================================================================
# 1. ASSET PRICING MODELS
# =============================================================================

# -------------------------
# A. BONDS: Pricing, Duration, Convexity
# -------------------------
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
        """
        Returns the scheduled payment dates and the corresponding cash flows.
        """
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        dates = []
        current_date = self.maturity_date
        # Generate coupon dates by subtracting coupon intervals until reaching issue_date
        while current_date > issue_date:
            dates.append(current_date)
            months = int(12 / self.frequency)
            current_date -= pd.DateOffset(months=months)
        dates = sorted(dates)
        # Each payment is coupon_rate * face_value / frequency
        coupon_payment = self.face_value * self.coupon_rate / self.frequency
        payments = [coupon_payment] * len(dates)
        # At maturity, add the face value to the last coupon payment
        if payments:
            payments[-1] += self.face_value
        return dates, payments

    def price(self, yield_rate, issue_date=None):
        """
        Price the bond using the present value of its cash flows.
        yield_rate: Annual yield (decimal)
        """
        dates, payments = self.cash_flows(issue_date)
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        price = 0.0
        for date, payment in zip(dates, payments):
            t = (date - issue_date).days / 365.0
            price += payment / (1 + yield_rate) ** t
        return price

    def duration(self, yield_rate, issue_date=None):
        """
        Calculate the Macaulay duration of the bond.
        """
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
        """
        Calculate the convexity of the bond.
        """
        dates, payments = self.cash_flows(issue_date)
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        bond_price = self.price(yield_rate, issue_date)
        convexity = 0.0
        for date, payment in zip(dates, payments):
            t = (date - issue_date).days / 365.0
            convexity += t * (t + 1) * (payment / (1 + yield_rate) ** (t + 2))
        return convexity / bond_price

# -------------------------
# B. DERIVATIVES
# -------------------------
# 1. Options (European and American)

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
        """
        Price a European option using the Black-Scholes formula.
        """
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
        """
        Price an American option using a binomial tree (can also be used for European options).
        """
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        # Risk-neutral probability
        p = (np.exp((self.r - self.dividend_yield) * dt) - d) / (u - d)
        # Initialize asset prices at maturity
        asset_prices = np.array([self.S * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)])
        # Option values at maturity
        if self.option_type == 'call':
            option_values = np.maximum(asset_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - asset_prices, 0)
        # Step back through the tree
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                # For American options, check early exercise:
                asset_price = self.S * (u ** (j - i)) * (d ** i)
                if self.option_type == 'call':
                    option_values[i] = max(option_values[i], asset_price - self.K)
                else:
                    option_values[i] = max(option_values[i], self.K - asset_price)
        return option_values[0]

    def greeks(self):
        """
        Compute and return key option Greeks for European options.
        (This is a simplified version computing only Delta.)
        """
        d1 = (np.log(self.S / self.K) + (self.r - self.dividend_yield + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = si.norm.cdf(d1) if self.option_type == 'call' else -si.norm.cdf(-d1)
        return {"delta": delta}

# 2. Futures
class Future:
    def __init__(self, underlying_price, cost_of_carry, T):
        """
        underlying_price: Current price of the underlying asset.
        cost_of_carry: Cost of carry (r - dividend yield for equities).
        T: Time to expiration in years.
        """
        self.underlying_price = underlying_price
        self.cost_of_carry = cost_of_carry
        self.T = T

    def price(self):
        """
        Price the future using the cost-of-carry model.
        F = S * exp(cost_of_carry * T)
        """
        return self.underlying_price * np.exp(self.cost_of_carry * self.T)

# 3. Interest Rate Swaps
class InterestRateSwap:
    def __init__(self, notional, fixed_rate, floating_rates, payment_dates, discount_factors):
        """
        notional: Swap notional amount.
        fixed_rate: Fixed rate of the swap.
        floating_rates: List/array of expected floating rates for each payment period.
        payment_dates: List of payment dates (datetime objects).
        discount_factors: List/array of discount factors corresponding to each payment date.
        """
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rates = np.array(floating_rates)
        self.payment_dates = payment_dates
        self.discount_factors = np.array(discount_factors)

    def price(self, issue_date=None):
        """
        Price the interest rate swap as the difference between the floating and fixed legs.
        (Simplified DCF approach.)
        """
        if issue_date is None:
            issue_date = pd.Timestamp.today()
        fixed_leg = sum(self.fixed_rate * self.notional * df for df in self.discount_factors)
        floating_leg = sum(fr * self.notional * df for fr, df in zip(self.floating_rates, self.discount_factors))
        return floating_leg - fixed_leg

# =============================================================================
# 2. PORTFOLIO OPTIMIZATION & RISK ANALYSIS
# =============================================================================

def black_litterman(prior_returns, covariance, P, Q, tau=0.05, omega=None):
    """
    Adjust expected returns using a Black-Litterman approach.
    
    prior_returns: Equilibrium market returns vector.
    covariance: Covariance matrix of asset returns.
    P: Pick matrix (views) of shape (n_views, n_assets).
    Q: Vector of views.
    tau: Scalar that scales the covariance matrix.
    omega: Uncertainty (diagonal) matrix of the views. If None, computed as diag(P * tau * covariance * P.T).
    
    Returns:
        Adjusted expected returns vector.
    """
    prior_returns = np.array(prior_returns)
    covariance = np.array(covariance)
    P = np.array(P)
    Q = np.array(Q)
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * covariance) @ P.T))
    inv_term = np.linalg.inv(np.linalg.inv(tau * covariance) + P.T @ np.linalg.inv(omega) @ P)
    adjusted_returns = inv_term @ (np.linalg.inv(tau * covariance) @ prior_returns + P.T @ np.linalg.inv(omega) @ Q)
    return adjusted_returns

# -------------------------
# Portfolio Optimizer using Markowitz Mean-Variance Framework
# -------------------------
class PortfolioOptimizer:
    def __init__(self, assets, expected_returns, covariance):
        """
        assets: List of asset names (e.g., ['Stock', 'Crypto', 'Bond', 'Derivative'])
        expected_returns: Expected annualized return vector.
        covariance: Covariance matrix (annualized) of asset returns.
        """
        self.assets = assets
        self.expected_returns = np.array(expected_returns)
        self.covariance = np.array(covariance)

    def optimize(self, target_return=None):
        """
        Solve the mean-variance optimization problem.
        Minimize portfolio variance subject to:
            - Full investment (weights sum to 1)
            - No short positions (weights >= 0)
            - Optional target return constraint.
        Returns: Optimal weights, portfolio expected return, and variance.
        """
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

# -------------------------
# Monte Carlo Simulation for Portfolio Paths
# -------------------------
def monte_carlo_simulation_portfolio(weights, daily_returns, daily_vol, num_simulations=1000, num_periods=252):
    """
    Simulate portfolio value paths.
    
    weights: Portfolio weights vector.
    daily_returns: Numpy array of expected daily returns for each asset.
    daily_vol: Assumed constant daily volatility for assets (or vector for each asset).
    num_simulations: Number of simulation runs.
    num_periods: Number of trading days.
    
    Returns:
        A simulation matrix of shape (num_simulations, num_periods) representing portfolio value paths.
    """
    weights = np.array(weights)
    n_assets = len(weights)
    portfolio_paths = []
    for i in range(num_simulations):
        # Simulate daily returns for each asset over num_periods:
        asset_paths = np.random.normal(loc=daily_returns, scale=daily_vol, size=(num_periods, n_assets))
        # Compute daily portfolio return:
        port_daily_returns = asset_paths @ weights
        # Calculate cumulative portfolio value (starting from 1)
        port_value = np.cumprod(1 + port_daily_returns)
        portfolio_paths.append(port_value)
    return np.array(portfolio_paths)

# =============================================================================
# 3. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_bond_price_vs_yield(bond, yield_min, yield_max, num_points=100):
    """
    Plot bond price as a function of yield.
    """
    yields = np.linspace(yield_min, yield_max, num_points)
    prices = [bond.price(y) for y in yields]
    plt.figure(figsize=(10,6))
    plt.plot(yields, prices, label='Bond Price', color='blue')
    plt.xlabel("Yield")
    plt.ylabel("Bond Price")
    plt.title("Bond Price vs. Yield")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bond_duration_convexity_vs_yield(bond, yield_min, yield_max, num_points=100):
    """
    Plot bond duration and convexity as a function of yield.
    """
    yields = np.linspace(yield_min, yield_max, num_points)
    durations = [bond.duration(y) for y in yields]
    convexities = [bond.convexity(y) for y in yields]
    plt.figure(figsize=(10,6))
    plt.plot(yields, durations, label='Duration', color='blue')
    plt.plot(yields, convexities, label='Convexity', color='red')
    plt.xlabel("Yield")
    plt.ylabel("Metric Value")
    plt.title("Bond Duration & Convexity vs. Yield")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_option_price_vs_underlying(option, underlying_range):
    """
    Plot option price (European, Black-Scholes) as a function of the underlying asset price.
    """
    prices = []
    for S in underlying_range:
        option.S = S  # update underlying price
        prices.append(option.black_scholes_price())
    plt.figure(figsize=(10,6))
    plt.plot(underlying_range, prices, label=f"{option.option_type.capitalize()} Option Price", color='green')
    plt.xlabel("Underlying Price")
    plt.ylabel("Option Price")
    plt.title("Option Price vs. Underlying Asset Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_random_portfolios(expected_returns, covariance, num_portfolios=5000):
    """
    Generate random portfolios to help plot an efficient frontier.
    Returns:
        results: Array with columns [volatility, return, sharpe_ratio]
        weights_record: List of portfolio weights.
    """
    n = len(expected_returns)
    results = np.zeros((num_portfolios, 3))  # vol, return, sharpe ratio
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
    """
    Plot the efficient frontier using randomly generated portfolios.
    """
    results, _ = generate_random_portfolios(expected_returns, covariance, num_portfolios)
    volatilities = results[:, 0]
    returns = results[:, 1]
    sharpe_ratios = results[:, 2]
    
    plt.figure(figsize=(10,6))
    sc = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.5)
    plt.xlabel("Volatility (Std. Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.colorbar(sc, label="Sharpe Ratio")
    if optimal_weight is not None:
        plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label='Optimal Portfolio')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_asset_allocation(assets, weights):
    """
    Plot a pie chart of asset allocation.
    """
    plt.figure(figsize=(8,8))
    plt.pie(weights, labels=assets, autopct='%1.1f%%', startangle=140)
    plt.title("Portfolio Asset Allocation")
    plt.show()

def plot_portfolio_simulations(portfolio_paths, num_paths_to_plot=50):
    """
    Plot several Monte Carlo simulated portfolio paths.
    """
    plt.figure(figsize=(10,6))
    for path in portfolio_paths[:num_paths_to_plot]:
        plt.plot(path, lw=0.7, alpha=0.6)
    plt.title("Monte Carlo Simulation of Portfolio Value Over Time")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

def plot_final_value_distribution(portfolio_paths):
    """
    Plot the distribution of final portfolio values from the Monte Carlo simulation.
    """
    final_values = portfolio_paths[:, -1]
    plt.figure(figsize=(10,6))
    plt.hist(final_values, bins=50, edgecolor='k', alpha=0.7)
    plt.title("Distribution of Final Portfolio Values")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# =============================================================================
# 4. EXAMPLE USAGE (Replace dummy data/API calls with live data as needed)
# =============================================================================
if __name__ == "__main__":
    # ---- A. Bond Example & Visualization ----
    bond = Bond(face_value=1000, coupon_rate=0.05, maturity_date="2030-12-31", frequency=2)
    bond_yield = 0.04  # Example yield of 4%
    bond_price = bond.price(bond_yield)
    bond_duration = bond.duration(bond_yield)
    bond_convexity = bond.convexity(bond_yield)
    print("Bond Price: {:.2f}, Duration: {:.2f}, Convexity: {:.2f}".format(bond_price, bond_duration, bond_convexity))
    
    # Visualize bond metrics vs. yield
    plot_bond_price_vs_yield(bond, 0.01, 0.1)
    plot_bond_duration_convexity_vs_yield(bond, 0.01, 0.1)

    # ---- B. Option Example & Visualization ----
    option = Option(option_type='call', S=100, K=95, T=1, r=0.03, sigma=0.2, dividend_yield=0.01)
    euro_call_price = option.black_scholes_price()
    american_call_price = option.binomial_tree_price(steps=100)
    greeks = option.greeks()
    print("European Call Price (Black-Scholes): {:.2f}".format(euro_call_price))
    print("American Call Price (Binomial Tree): {:.2f}".format(american_call_price))
    print("Option Greeks:", greeks)
    
    # Visualize option price sensitivity vs. underlying price
    underlying_range = np.linspace(80, 120, 100)
    plot_option_price_vs_underlying(option, underlying_range)

    # ---- C. Future Example ----
    future = Future(underlying_price=100, cost_of_carry=0.03, T=1)
    future_price = future.price()
    print("Future Price: {:.2f}".format(future_price))

    # ---- D. Swap Example ----
    # Dummy payment dates and discount factors for a three-period swap.
    payment_dates = [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"), pd.Timestamp("2026-12-31")]
    discount_factors = [0.98, 0.95, 0.92]
    swap = InterestRateSwap(notional=1000000, fixed_rate=0.04,
                              floating_rates=[0.035, 0.04, 0.045],
                              payment_dates=payment_dates,
                              discount_factors=discount_factors)
    swap_value = swap.price()
    print("Swap Value: {:.2f}".format(swap_value))

    # ---- E. Portfolio Optimization Example ----
    # Assume a portfolio with 4 asset classes: Stock, Crypto, Bond, Derivative (e.g., option exposure)
    assets = ['Stock', 'Crypto', 'Bond', 'Derivative']
    # Annualized expected returns for the assets (from CAPM, historical data, or Black-Litterman views)
    expected_returns = [0.08, 0.12, 0.05, 0.10]
    # Dummy covariance matrix (annualized)
    covariance = [[0.1,   0.02,  0.01, 0.03],
                  [0.02,  0.2,   0.015,0.025],
                  [0.01,  0.015, 0.05, 0.02],
                  [0.03,  0.025, 0.02, 0.15]]
    
    # Optionally adjust expected returns using Black-Litterman views:
    P = [[0, 0, 1, 0]]  # A view on bonds
    Q = [0.055]
    adjusted_returns = black_litterman(expected_returns, covariance, P, Q, tau=0.05)
    print("Adjusted Expected Returns (Black-Litterman):", adjusted_returns)

    optimizer = PortfolioOptimizer(assets, adjusted_returns, covariance)
    # For example, target a portfolio return of 7% per annum.
    weights, port_return, port_variance = optimizer.optimize(target_return=0.07)
    port_volatility = np.sqrt(port_variance)
    print("Optimized Portfolio Weights:", weights)
    print("Portfolio Expected Return: {:.2f}".format(port_return))
    print("Portfolio Variance: {:.4f}".format(port_variance))
    print("Portfolio Volatility: {:.2f}".format(port_volatility))
    
    # Visualize efficient frontier (marking the optimized portfolio)
    plot_efficient_frontier(adjusted_returns, covariance,
                            optimal_weight=weights,
                            optimal_return=port_return,
                            optimal_volatility=port_volatility,
                            num_portfolios=5000)
    
    # Visualize asset allocation
    plot_asset_allocation(assets, weights)
    
    # ---- F. Monte Carlo Simulation for Risk Analysis ----
    # Convert annual expected returns to daily returns (assume 252 trading days)
    daily_expected_returns = np.array(expected_returns) / 252
    daily_vol = 0.01  # Assume 1% daily volatility for simplicity
    portfolio_paths = monte_carlo_simulation_portfolio(weights, daily_expected_returns, daily_vol,
                                                       num_simulations=1000, num_periods=252)
    
    # Plot several Monte Carlo simulated portfolio paths and the distribution of final values
    plot_portfolio_simulations(portfolio_paths, num_paths_to_plot=50)
    plot_final_value_distribution(portfolio_paths)
