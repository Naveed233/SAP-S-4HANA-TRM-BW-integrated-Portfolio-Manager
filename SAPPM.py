# @title Without SAP, Running Version (Current)
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
from datetime import datetime
import seaborn as sns
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# BLACK-LITTERMAN FUNCTION
# ---------------------------
def black_litterman(prior_returns, covariance, P, Q, tau=0.05, omega=None):
    """
    Adjust expected returns using the Black-Litterman approach.
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

# ---------------------------
# Portfolio Optimizer Class
# ---------------------------
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02, fixed_expected_returns=None, asset_classes=None):
        """
        Initialize with:
         - tickers: list of ticker symbols (strings)
         - start_date, end_date: date strings (YYYY-MM-DD)
         - risk_free_rate: risk-free rate (decimal)
         - fixed_expected_returns: optional list of fixed expected returns per ticker (annualized)
         - asset_classes: list of asset class labels for each ticker (e.g., "Stock", "Crypto", "Bond", "Derivative")
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.fixed_expected_returns = fixed_expected_returns  # if provided, override computed mean returns
        self.asset_classes = asset_classes
        self.returns = None

    def fetch_data(self):
        """
        Fetch historical price data and compute daily returns.
        """
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
        logger.info(f"Data columns: {data.columns}")
        st.write("Fetched Data Preview:", data.head())

        # Handle both MultiIndex and single-level columns
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                data = data.xs('Adj Close', axis=1, level=0)
            elif 'Close' in data.columns.levels[0]:
                data = data.xs('Close', axis=1, level=0)
            else:
                st.error("Neither 'Adj Close' nor 'Close' columns are available in multi-index.")
                raise ValueError("Neither 'Adj Close' nor 'Close' columns are available in multi-index.")
        else:
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                st.error("Neither 'Adj Close' nor 'Close' columns are available.")
                raise ValueError("Neither 'Adj Close' nor 'Close' columns are available.")

        data.dropna(axis=1, how='all', inplace=True)
        if data.empty:
            logger.error("No data fetched after dropping missing tickers.")
            raise ValueError("No data fetched. Please check the tickers and date range.")
        if isinstance(data, pd.DataFrame):
            self.tickers = list(data.columns)
        else:
            self.tickers = [data.name]
            data = pd.DataFrame(data)
        self.returns = data.pct_change().dropna()
        logger.info(f"Fetched returns for {len(self.tickers)} tickers.")
        return self.tickers

    def portfolio_stats(self, weights):
        """
        Calculate portfolio annualized return, volatility, and Sharpe ratio.
        If fixed_expected_returns are provided, use them instead of computed historical means.
        """
        weights = np.array(weights)
        if len(weights) != len(self.tickers):
            raise ValueError("Weights array length does not match the number of tickers.")
        weights = weights / np.sum(weights)
        if self.fixed_expected_returns is not None:
            portfolio_return = np.dot(weights, self.fixed_expected_returns) * 252
        else:
            portfolio_return = np.dot(weights, self.returns.mean()) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def value_at_risk(self, weights, confidence_level=0.95):
        portfolio_returns = self.returns.dot(weights)
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return var

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    def maximum_drawdown(self, weights):
        portfolio_returns = self.returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        return max_drawdown

    def herfindahl_hirschman_index(self, weights):
        return np.sum(weights ** 2)

    def sharpe_ratio_objective(self, weights):
        _, _, sharpe = self.portfolio_stats(weights)
        return -sharpe  # Negative for maximization via minimization

    def optimize_sharpe_ratio(self):
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        result = minimize(
            self.sharpe_ratio_objective, initial_weights,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        if result.success:
            logger.info("Optimized portfolio for Sharpe Ratio successfully.")
            return result.x
        else:
            logger.warning(f"Optimization failed: {result.message}")
            return initial_weights

    def min_volatility(self, target_return, max_weight=0.3):
        num_assets = len(self.tickers)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets
        result = minimize(
            lambda weights: self.portfolio_stats(weights)[1],
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            logger.info("Optimized portfolio for minimum volatility successfully.")
            return result.x
        else:
            logger.warning(f"Portfolio optimization failed: {result.message}")
            return np.ones(num_assets) / num_assets

    def prepare_data_for_lstm(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.returns.values)
        X, y = [], []
        look_back = 60
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i])
            y.append(scaled_data[i])
        split = int(len(X) * 0.8)
        X_train, y_train = np.array(X[:split]), np.array(y[:split])
        X_test, y_test = np.array(X[split:]), np.array(y[split:])
        if not len(X_train) or not len(y_train):
            raise ValueError("Not enough data to create training samples. Please adjust the date range or add more data.")
        return X_train, y_train, X_test, y_test, scaler

    def train_lstm_model(self, X_train, y_train, epochs=10, batch_size=32):
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dense(units=X_train.shape[2]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    def predict_future_returns(self, model, scaler, steps=30):
        if len(self.returns) < 60:
            raise ValueError("Not enough data to make predictions. Ensure there are at least 60 days of returns data.")
        last_data = self.returns[-60:].values
        scaled_last_data = scaler.transform(last_data)
        X_test = np.array([scaled_last_data])
        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)
        future_returns = predicted[0][:steps] if len(predicted[0]) >= steps else predicted[0]
        return future_returns

    def evaluate_model(self, model, scaler, X_test, y_test):
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_inverse = scaler.inverse_transform(y_test)
        mae = mean_absolute_error(y_test_inverse, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions))
        r2 = r2_score(y_test_inverse, predictions)
        return mae, rmse, r2

    def compute_efficient_frontier(self, num_portfolios=10000):
        results = np.zeros((4, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            weights_record.append(weights)
            portfolio_return, portfolio_volatility, sharpe = self.portfolio_stats(weights)
            var = self.value_at_risk(weights, confidence_level=0.95)
            cvar = self.conditional_value_at_risk(weights, confidence_level=0.95)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = sharpe
            results[3, i] = self.herfindahl_hirschman_index(weights)
        return results, weights_record

# ---------------------------
# Translations Dictionary
# ---------------------------
languages = {
    'English': 'en',
    '日本語': 'ja'
}

translations = {
    'en': {
        "title": "Portfolio Optimization with Advanced Features",
        "user_inputs": "🔧 User Inputs",
        "select_universe": "Select an Asset Universe:",
        "custom_tickers": "Enter stock tickers separated by commas (e.g., AAPL, MSFT, TSLA):",
        "add_portfolio": "Add to My Portfolio",
        "my_portfolio": "📁 My Portfolio",
        "no_assets": "No assets added yet.",
        "optimization_parameters": "📅 Optimization Parameters",
        "start_date": "Start Date",
        "end_date": "End Date",
        "risk_free_rate": "Enter the risk-free rate (in %):",
        "investment_strategy": "Choose your Investment Strategy:",
        "strategy_risk_free": "Risk Averse Investment",
        "strategy_profit": "Profit-focused Investment",
        "target_return": "Select a specific target return (in %)",
        "use_fixed_returns": "Use Fixed Expected Returns",
        "expected_stock": "Expected Return for Stocks",
        "expected_crypto": "Expected Return for Cryptocurrencies",
        "expected_bond": "Expected Return for Bonds",
        "expected_deriv": "Expected Return for Derivatives",
        "train_lstm": "Train LSTM Model for Future Returns Prediction",
        "more_info_lstm": "ℹ️ More Information on LSTM",
        "optimize_portfolio": "Optimize Portfolio",
        "optimize_sharpe": "Optimize for Highest Sharpe Ratio",
        "compare_portfolios": "Compare Sharpe vs Base",
        "portfolio_analysis": "🔍 Portfolio Analysis & Optimization Results",
        "success_lstm": "🤖 LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset to your portfolio before training the LSTM model.",
        "error_no_assets_opt": "Please add at least one asset to your portfolio before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "🔑 Optimal Portfolio Allocation (Target Return: {target}%)",
        "performance_metrics": "📊 Portfolio Performance Metrics",
        "visual_analysis": "📊 Visual Analysis",
        "portfolio_composition": "Portfolio Composition",
        "portfolio_metrics": "Portfolio Performance Metrics",
        "correlation_heatmap": "Asset Correlation Heatmap",
        "var": "Value at Risk (VaR)",
        "cvar": "Conditional Value at Risk (CVaR)",
        "max_drawdown": "Maximum Drawdown",
        "hhi": "Herfindahl-Hirschman Index (HHI)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
        "beta": "Beta",
        "alpha": "Alpha",
        "explanation_var": "**Value at Risk (VaR):** Estimates the maximum potential loss of a portfolio over a specified time frame at a given confidence level.",
        "explanation_cvar": "**Conditional Value at Risk (CVaR):** Measures the expected loss exceeding the VaR, providing insights into tail risk.",
        "explanation_max_drawdown": "**Maximum Drawdown:** Measures the largest peak-to-trough decline in the portfolio value, indicating the worst-case scenario.",
        "explanation_hhi": "**Herfindahl-Hirschman Index (HHI):** A diversification metric that measures the concentration of investments in a portfolio.",
        "explanation_sharpe_ratio": "**Sharpe Ratio:** Measures risk-adjusted returns, indicating how much excess return you receive for the extra volatility endured.",
        "explanation_sortino_ratio": "**Sortino Ratio:** Similar to the Sharpe Ratio but only considers downside volatility.",
        "explanation_calmar_ratio": "**Calmar Ratio:** Compares the portfolio's annualized return to its maximum drawdown.",
        "explanation_beta": "**Beta:** Measures the portfolio's volatility relative to a benchmark index.",
        "explanation_alpha": "**Alpha:** Represents the portfolio's excess return relative to the expected return based on its beta.",
        "explanation_lstm": "**LSTM Model Explanation:**\nLSTM is a type of recurrent neural network effective for predicting sequences such as stock returns. Its predictions are based on historical data and should be combined with other analyses.",
        "feedback_sharpe_good": "Great! A Sharpe Ratio above 1 indicates good risk-adjusted returns.",
        "feedback_sharpe_average": "Average. A Sharpe Ratio between 0.5 and 1 suggests acceptable returns for the risk taken.",
        "feedback_sharpe_poor": "Poor. A Sharpe Ratio below 0.5 indicates insufficient returns for the level of risk.",
        "success_optimize": "Portfolio optimization completed successfully!",
        "explanation_sharpe_button": "**Optimize for Highest Sharpe Ratio:**\nOptimizing for the highest Sharpe Ratio aims to achieve the best return for the risk taken.",
        "recommendation": "Based on the above metrics, the **{better_portfolio}** portfolio is recommended for better **{better_metric}**."
    },
    'ja': {
        "title": "高度な機能を備えたポートフォリオ最適化アプリ",
        "user_inputs": "🔧 ユーザー入力",
        "select_universe": "資産ユニバースを選択してください：",
        "custom_tickers": "株式ティッカーをカンマで区切って入力してください（例：AAPL, MSFT, TSLA）：",
        "add_portfolio": "マイポートフォリオに追加",
        "my_portfolio": "📁 マイポートフォリオ",
        "no_assets": "まだ資産が追加されていません。",
        "optimization_parameters": "📅 最適化パラメータ",
        "start_date": "開始日",
        "end_date": "終了日",
        "risk_free_rate": "無リスク金利を入力してください（%）：",
        "investment_strategy": "投資戦略を選択してください：",
        "strategy_risk_free": "リスク回避型投資",
        "strategy_profit": "利益重視型投資",
        "target_return": "目標リターンを選択してください（%）",
        "use_fixed_returns": "固定期待リターンを使用",
        "expected_stock": "株式の期待リターン",
        "expected_crypto": "暗号資産の期待リターン",
        "expected_bond": "債券の期待リターン",
        "expected_deriv": "デリバティブの期待リターン",
        "train_lstm": "将来リターン予測のためLSTMモデルを訓練",
        "more_info_lstm": "ℹ️ LSTMの詳細情報",
        "optimize_portfolio": "ポートフォリオを最適化",
        "optimize_sharpe": "シャープレシオ最大化のため最適化",
        "compare_portfolios": "シャープレシオベースと比較",
        "portfolio_analysis": "🔍 ポートフォリオ分析と最適化結果",
        "success_lstm": "🤖 LSTMモデルが正常に訓練されました！",
        "error_no_assets_lstm": "LSTMモデルを訓練する前に、少なくとも1つの資産をポートフォリオに追加してください。",
        "error_no_assets_opt": "最適化する前に、少なくとも1つの資産をポートフォリオに追加してください。",
        "error_date": "開始日は終了日より前でなければなりません。",
        "allocation_title": "🔑 最適なポートフォリオ配分（目標リターン：{target}%）",
        "performance_metrics": "📊 ポートフォリオのパフォーマンス指標",
        "visual_analysis": "📊 視覚的分析",
        "portfolio_composition": "ポートフォリオ構成",
        "portfolio_metrics": "ポートフォリオのパフォーマンス指標",
        "correlation_heatmap": "資産相関ヒートマップ",
        "var": "リスク価値 (VaR)",
        "cvar": "条件付きリスク価値 (CVaR)",
        "max_drawdown": "最大ドローダウン",
        "hhi": "ハーフィンダール・ハーシュマン指数 (HHI)",
        "sharpe_ratio": "シャープレシオ",
        "sortino_ratio": "ソルティーノレシオ",
        "calmar_ratio": "カルマーレシオ",
        "beta": "ベータ",
        "alpha": "アルファ",
        "explanation_var": "**リスク価値 (VaR):** 指定された信頼水準で、特定期間内の最大損失を推定します。",
        "explanation_cvar": "**条件付きリスク価値 (CVaR):** VaRを超える損失の期待値を測定し、テールリスクに関する洞察を提供します。",
        "explanation_max_drawdown": "**最大ドローダウン:** ポートフォリオ価値のピークからの最大下落率を示します。",
        "explanation_hhi": "**ハーフィンダール・ハーシュマン指数 (HHI):** ポートフォリオ内の投資集中度を測定します。",
        "explanation_sharpe_ratio": "**シャープレシオ:** リスク調整後リターンを測定し、余分なボラティリティに対してどれだけの超過リターンが得られているかを示します。",
        "explanation_sortino_ratio": "**ソルティーノレシオ:** 下方リスクのみを考慮したリスク調整後リターンを評価します。",
        "explanation_calmar_ratio": "**カルマーレシオ:** 年率リターンを最大ドローダウンで割った値を示します。",
        "explanation_beta": "**ベータ:** ポートフォリオのボラティリティがベンチマークに比べどの程度かを示します。",
        "explanation_alpha": "**アルファ:** ベータに基づく期待リターンとの差分（超過リターン）を示します。",
        "explanation_lstm": "**LSTMモデルの説明：**\nLSTMは時系列予測に優れるニューラルネットワークです。過去のデータパターンに基づいて将来リターンを予測しますが、保証されたものではありません。",
        "feedback_sharpe_good": "素晴らしいです！シャープレシオが1以上なら、リスクに対して十分なリターンを示しています。",
        "feedback_sharpe_average": "平均的です。シャープレシオが0.5〜1なら、リスクに対するリターンは許容範囲です。",
        "feedback_sharpe_poor": "低いです。シャープレシオが0.5未満なら、リスクに見合ったリターンが得られていません。",
        "success_optimize": "ポートフォリオ最適化が正常に完了しました！",
        "explanation_sharpe_button": "**シャープレシオ最大化のため最適化：**\n最適化により、リスクに対して最も優れたリターンを提供するポートフォリオを構築します。",
        "recommendation": "上記指標に基づき、**{better_portfolio}**ポートフォリオはより良い**{better_metric}**を提供すると推奨されます。"
    }
}

# ---------------------------
# Helper Functions
# ---------------------------
def extract_ticker(asset_string):
    return asset_string.split(' - ')[0].strip() if ' - ' in asset_string else asset_string.strip()

def get_translated_text(lang, key):
    return translations.get(lang, translations['en']).get(key, key)

def analyze_var(var):
    if var < -0.05:
        return "High Risk: Significant potential loss."
    elif -0.05 <= var < -0.02:
        return "Moderate Risk: Moderate potential loss."
    else:
        return "Low Risk: Relatively safe."

def analyze_cvar(cvar):
    if cvar < -0.07:
        return "High Tail Risk: Significant losses beyond VaR."
    elif -0.07 <= cvar < -0.04:
        return "Moderate Tail Risk: Moderate losses beyond VaR."
    else:
        return "Low Tail Risk: Minimal losses beyond VaR."

def analyze_max_drawdown(dd):
    if dd < -0.20:
        return "Severe Drawdown: Major decline experienced."
    elif -0.20 <= dd < -0.10:
        return "Moderate Drawdown: Noticeable decline."
    else:
        return "Minor Drawdown: Stability maintained."

def analyze_hhi(hhi):
    if hhi > 0.6:
        return "High Concentration: Lacks diversification."
    elif 0.3 < hhi <= 0.6:
        return "Moderate Concentration: Some diversification."
    else:
        return "Good Diversification: Well diversified."

def analyze_sharpe(sharpe):
    if sharpe > 1:
        return "Great! Good risk-adjusted returns."
    elif 0.5 < sharpe <= 1:
        return "Average: Acceptable returns for the risk."
    else:
        return "Poor: Insufficient returns for the risk."

def display_metrics_table(metrics, lang):
    metric_display = []
    for key, value in metrics.items():
        display_key = get_translated_text(lang, key)
        if key in ["hhi"]:
            display_value = f"{value:.4f}"
        elif key in ["beta", "alpha"]:
            display_value = f"{value:.2f}"
        elif key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            display_value = f"{value:.2f}"
        else:
            display_value = f"{value:.2%}"
        analysis_func = {
            "var": analyze_var,
            "cvar": analyze_cvar,
            "max_drawdown": analyze_max_drawdown,
            "hhi": analyze_hhi,
            "sharpe_ratio": analyze_sharpe,
            "sortino_ratio": analyze_sharpe,
            "calmar_ratio": analyze_sharpe,
            "beta": analyze_sharpe,
            "alpha": analyze_sharpe
        }.get(key, lambda x: "")
        analysis = analysis_func(value)
        metric_display.append({
            "Metric": display_key,
            "Value": display_value,
            "Analysis": analysis
        })
    metrics_df = pd.DataFrame.from_dict(metric_display)
    st.table(metrics_df.style.set_properties(**{'text-align': 'left', 'padding': '5px'}))

def compare_portfolios(base_metrics, optimized_metrics, lang):
    comparison_data = []
    better_portfolio = ""
    better_metric = ""
    for key in base_metrics.keys():
        base_value = base_metrics[key]
        optimized_value = optimized_metrics[key]
        metric_display = get_translated_text(lang, key)
        if key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha"]:
            if optimized_value > base_value:
                better = "Optimized"
                better_portfolio = "Optimized"
                better_metric = metric_display
            else:
                better = "Base"
                better_portfolio = "Base"
                better_metric = metric_display
        elif key in ["var", "cvar", "max_drawdown", "beta", "hhi"]:
            if optimized_value < base_value:
                better = "Optimized"
                better_portfolio = "Optimized"
                better_metric = metric_display
            else:
                better = "Base"
                better_portfolio = "Base"
                better_metric = metric_display
        else:
            better = "-"
        def format_val(k, v):
            if k in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha"]:
                return f"{v:.2f}"
            elif k in ["var", "cvar", "max_drawdown", "beta", "hhi"]:
                return f"{v:.4f}" if k in ["hhi"] else f"{v:.2%}"
            else:
                return f"{v:.2f}"
        comparison_data.append({
            "Metric": metric_display,
            "Base Portfolio": format_val(key, base_value),
            "Optimized Portfolio": format_val(key, optimized_value),
            "Better": better
        })
    comparison_df = pd.DataFrame(comparison_data)
    def highlight_better(row):
        better = row['Better']
        styles = [''] * len(row)
        if better == "Optimized":
            styles[comparison_df.columns.get_loc("Optimized Portfolio")] = 'background-color: lightgreen'
        elif better == "Base":
            styles[comparison_df.columns.get_loc("Base Portfolio")] = 'background-color: lightgreen'
        return styles
    comparison_df = comparison_df.style.apply(highlight_better, axis=1)
    st.markdown("<h3>📊 Comparison: Sharpe vs Base Portfolio</h3>", unsafe_allow_html=True)
    st.table(comparison_df)
    if better_metric:
        recommendation_text = translations[lang].get("recommendation", "").format(
            better_portfolio=better_portfolio, better_metric=better_metric
        )
        st.markdown(f"<p><strong>Recommendation:</strong> {recommendation_text}</p>", unsafe_allow_html=True)

# ---------------------------
# Streamlit App Main Function
# ---------------------------
def main():
    st.set_page_config(page_title="Portfolio Optimization App", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.header("🌐 Language Selection")
    selected_language = st.sidebar.selectbox("Select Language:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]
    st.title(get_translated_text(lang, "title"))

    # ---------------------------
    # Sidebar: User Inputs and Asset Universe
    # ---------------------------
    st.sidebar.header(get_translated_text(lang, "user_inputs"))
    universe_options = {
        'Tech Giants': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet', 'AMZN - Amazon', 'META - Meta Platforms', 'TSLA - Tesla', 'NVDA - NVIDIA', 'ADBE - Adobe', 'INTC - Intel', 'CSCO - Cisco'],
        'Finance Leaders': ['JPM - JPMorgan Chase', 'BAC - Bank of America', 'WFC - Wells Fargo', 'C - Citigroup', 'GS - Goldman Sachs', 'MS - Morgan Stanley', 'AXP - American Express', 'BLK - BlackRock', 'SCHW - Charles Schwab', 'USB - U.S. Bancorp'],
        'Healthcare Majors': ['JNJ - Johnson & Johnson', 'PFE - Pfizer', 'UNH - UnitedHealth', 'MRK - Merck', 'ABBV - AbbVie', 'ABT - Abbott', 'TMO - Thermo Fisher Scientific', 'MDT - Medtronic', 'DHR - Danaher', 'BMY - Bristol-Myers Squibb'],
        'Cryptocurrencies': ['BTC-USD - Bitcoin', 'ETH-USD - Ethereum', 'ADA-USD - Cardano', 'SOL-USD - Solana'],
        'Custom': []
    }
    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), options=list(universe_options.keys()), index=0)
    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(get_translated_text(lang, "custom_tickers"), value="")
    else:
        selected_universe_assets = st.sidebar.multiselect(get_translated_text(lang, "add_portfolio"), universe_options[universe_choice], default=[])

    # Use Session State to hold portfolio tickers and metrics
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []
    if 'base_portfolio_metrics' not in st.session_state:
        st.session_state['base_portfolio_metrics'] = None
    if 'optimized_portfolio_metrics' not in st.session_state:
        st.session_state['optimized_portfolio_metrics'] = None

    # Add assets to portfolio from preset or custom
    if universe_choice != 'Custom':
        if selected_universe_assets:
            if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
                new_tickers = [extract_ticker(asset) for asset in selected_universe_assets]
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
                st.sidebar.success(get_translated_text(lang, "add_portfolio") + " " + get_translated_text(lang, "my_portfolio"))
    else:
        if st.sidebar.button(get_translated_text(lang, "add_portfolio")):
            if custom_tickers.strip():
                new_tickers = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
                st.session_state['my_portfolio'] = list(set(st.session_state['my_portfolio'] + new_tickers))
                st.sidebar.success(get_translated_text(lang, "add_portfolio") + " " + get_translated_text(lang, "my_portfolio"))

    st.sidebar.subheader(get_translated_text(lang, "my_portfolio"))
    if st.session_state['my_portfolio']:
        st.sidebar.write(", ".join(st.session_state['my_portfolio']))
    else:
        st.sidebar.write(get_translated_text(lang, "no_assets"))

    # ---------------------------
    # Sidebar: Optimization Parameters
    # ---------------------------
    st.sidebar.header(get_translated_text(lang, "optimization_parameters"))
    start_date = st.sidebar.date_input(get_translated_text(lang, "start_date"), value=datetime(2024, 1, 1), max_value=datetime.today())
    def get_last_day_previous_month():
        today = datetime.today()
        first_day_current_month = today.replace(day=1)
        last_day_prev_month = first_day_current_month - pd.Timedelta(days=1)
        return last_day_prev_month
    end_date = st.sidebar.date_input(get_translated_text(lang, "end_date"), value=get_last_day_previous_month(), max_value=datetime.today())
    risk_free_rate = st.sidebar.number_input(get_translated_text(lang, "risk_free_rate"), value=2.0, step=0.1) / 100

    # Investment Strategy Options
    investment_strategy = st.sidebar.radio(get_translated_text(lang, "investment_strategy"),
                                             (get_translated_text(lang, "strategy_risk_free"), get_translated_text(lang, "strategy_profit")))
    if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
        specific_target_return = st.sidebar.slider(get_translated_text(lang, "target_return"), min_value=-5.0, max_value=20.0, value=5.0, step=0.1) / 100
    else:
        specific_target_return = None

    # Option: Use Fixed Expected Returns via Black–Litterman
    use_fixed_returns = st.sidebar.checkbox(get_translated_text(lang, "use_fixed_returns"), value=False)
    fixed_returns_dict = {}
    if use_fixed_returns:
        fixed_returns_dict["Stock"] = st.sidebar.number_input(get_translated_text(lang, "expected_stock"), value=0.09, step=0.01)
        fixed_returns_dict["Crypto"] = st.sidebar.number_input(get_translated_text(lang, "expected_crypto"), value=0.12, step=0.01)
        fixed_returns_dict["Bond"] = st.sidebar.number_input(get_translated_text(lang, "expected_bond"), value=0.05, step=0.01)
        fixed_returns_dict["Derivative"] = st.sidebar.number_input(get_translated_text(lang, "expected_deriv"), value=0.10, step=0.01)

    # LSTM Training and Optimization Buttons
    train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))
    optimize_portfolio = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))
    optimize_sharpe = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"))
    compare_portfolios_btn = st.sidebar.button(get_translated_text(lang, "compare_portfolios"))

    st.header(get_translated_text(lang, "portfolio_analysis"))

    # ---------------------------
    # LSTM Model Training Section
    # ---------------------------
    if train_lstm:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_lstm"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                optimizer.fetch_data()
                X_train, y_train, X_test, y_test, scaler = optimizer.prepare_data_for_lstm()
                model = optimizer.train_lstm_model(X_train, y_train, epochs=10, batch_size=32)
                mae, rmse, r2 = optimizer.evaluate_model(model, scaler, X_test, y_test)
                st.success(get_translated_text(lang, "success_lstm"))
                st.subheader("LSTM Model Evaluation Metrics")
                eval_metrics = {
                    "Mean Absolute Error (MAE)": mae,
                    "Root Mean Squared Error (RMSE)": rmse,
                    "R-squared (R²)": r2
                }
                eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value'])
                st.table(eval_df.style.format({"Value": "{:.4f}"}))
                future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
                future_dates = pd.date_range(end_date, periods=len(future_returns), freq='B').to_pydatetime().tolist()
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Returns': future_returns
                })
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prediction_df['Date'], prediction_df['Predicted Returns'], label="Predicted Returns", color='blue')
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Returns")
                ax.set_title(get_translated_text(lang, "train_lstm"))
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                with st.expander(get_translated_text(lang, "more_info_lstm")):
                    explanation = get_translated_text(lang, "explanation_lstm")
                    st.markdown(explanation)
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An error occurred during LSTM training or prediction.")
                st.error(f"{e}")

    # ---------------------------
    # Portfolio Optimization Section
    # ---------------------------
    if optimize_portfolio:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                # Determine asset classes based on preset universe:
                if universe_choice in ["Tech Giants", "Finance Leaders", "Healthcare Majors"]:
                    asset_classes = ["Stock"] * len(clean_tickers)
                elif universe_choice == "Cryptocurrencies":
                    asset_classes = ["Crypto"] * len(clean_tickers)
                else:  # For Custom, default to "Stock"
                    asset_classes = ["Stock"] * len(clean_tickers)
                # If fixed expected returns are used, assign them per asset class
                if use_fixed_returns:
                    fixed_expected_returns = [fixed_returns_dict.get(ac, 0.09) for ac in asset_classes]
                    # Optionally, adjust using Black–Litterman views:
                    # Set up identity P matrix and Q = fixed_expected_returns vector.
                    n = len(fixed_expected_returns)
                    P = np.eye(n)
                    Q = np.array(fixed_expected_returns)
                    cov = st.session_state.get("covariance")
                    if cov is None:
                        # Fetch data to compute covariance
                        temp_optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                        temp_optimizer.fetch_data()
                        cov = temp_optimizer.returns.cov() * 252
                        st.session_state["covariance"] = cov
                    # Historical prior: computed mean returns from data
                    temp_optimizer.fetch_data()
                    prior = temp_optimizer.returns.mean().values * 252
                    adjusted_returns = black_litterman(prior, cov, P, Q, tau=0.05)
                else:
                    fixed_expected_returns = None
                    asset_classes = None
                    adjusted_returns = None
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate, fixed_expected_returns, asset_classes)
                updated_tickers = optimizer.fetch_data()
                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    if specific_target_return is None:
                        st.error("Please select a target return for Risk Averse strategy.")
                        st.stop()
                    optimal_weights = optimizer.min_volatility(specific_target_return)
                    details = "Details: You selected a 'Risk Averse' strategy, aiming for minimal risk while achieving the specified target return."
                else:
                    optimal_weights = optimizer.optimize_sharpe_ratio()
                    details = "Details: You selected a 'Profit-focused' strategy, aiming for maximum potential returns accepting higher risk."
                portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)
                allocation = pd.DataFrame({
                    "Asset": updated_tickers,
                    "Weight (%)": np.round(optimal_weights * 100, 2)
                })
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)
                target_display = round(specific_target_return*100, 2) if specific_target_return else "N/A"
                st.subheader(get_translated_text(lang, "allocation_title").format(target=target_display))
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))
                metrics = {
                    "var": var_95,
                    "cvar": cvar_95,
                    "max_drawdown": max_dd,
                    "hhi": hhi,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": 0.0,  # Placeholder
                    "calmar_ratio": 0.0,   # Placeholder
                    "beta": 0.0,           # Placeholder
                    "alpha": 0.0           # Placeholder
                }
                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    st.session_state['base_portfolio_metrics'] = metrics
                else:
                    st.session_state['optimized_portfolio_metrics'] = metrics
                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics_table(metrics, lang)
                st.subheader(get_translated_text(lang, "visual_analysis"))
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax1.axis('equal')
                    ax1.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig1)
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    performance_metrics = {
                        "Expected\n Annual Return (%)": portfolio_return * 100,
                        "Annual Volatility\n(Risk) (%)": portfolio_volatility * 100,
                        "Sharpe Ratio": sharpe_ratio
                    }
                    metrics_bar = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax2)
                    ax2.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax2.patches:
                        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2., p.get_height()),
                                     ha='center', va='bottom', fontsize=10)
                    plt.xticks(rotation=0, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig2)
                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                correlation_matrix = optimizer.returns.corr()
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
                plt.title(get_translated_text(lang, "correlation_heatmap"))
                plt.tight_layout()
                st.pyplot(fig3)
                st.success(get_translated_text(lang, "success_optimize"))
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An unexpected error occurred during optimization.")
                st.error(f"{e}")

    # ---------------------------
    # Optimize for Highest Sharpe Ratio Section
    # ---------------------------
    if optimize_sharpe:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = [ticker for ticker in st.session_state['my_portfolio']]
                if universe_choice in ["Tech Giants", "Finance Leaders", "Healthcare Majors"]:
                    asset_classes = ["Stock"] * len(clean_tickers)
                elif universe_choice == "Cryptocurrencies":
                    asset_classes = ["Crypto"] * len(clean_tickers)
                else:
                    asset_classes = ["Stock"] * len(clean_tickers)
                if use_fixed_returns:
                    fixed_expected_returns = [fixed_returns_dict.get(ac, 0.09) for ac in asset_classes]
                else:
                    fixed_expected_returns = None
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate, fixed_expected_returns, asset_classes)
                updated_tickers = optimizer.fetch_data()
                optimal_weights = optimizer.optimize_sharpe_ratio()
                portfolio_return, portfolio_volatility, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)
                allocation = pd.DataFrame({
                    "Asset": updated_tickers,
                    "Weight (%)": np.round(optimal_weights * 100, 2)
                })
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)
                st.subheader("🔑 Optimal Portfolio Allocation (Highest Sharpe Ratio)")
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))
                metrics = {
                    "var": var_95,
                    "cvar": cvar_95,
                    "max_drawdown": max_dd,
                    "hhi": hhi,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "beta": 0.0,
                    "alpha": 0.0
                }
                st.session_state['optimized_portfolio_metrics'] = metrics
                st.subheader(get_translated_text(lang, "performance_metrics"))
                display_metrics_table(metrics, lang)
                st.subheader(get_translated_text(lang, "visual_analysis"))
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax1.axis('equal')
                    ax1.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig1)
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    performance_metrics = {
                        "Expected\n Annual Return (%)": portfolio_return * 100,
                        "Annual Volatility\n(Risk) (%)": portfolio_volatility * 100,
                        "Sharpe Ratio": sharpe_ratio
                    }
                    metrics_bar = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['Value'])
                    sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax2)
                    ax2.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax2.patches:
                        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='bottom', fontsize=10)
                    plt.xticks(rotation=0, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig2)
                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                correlation_matrix = optimizer.returns.corr()
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax3,
                            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
                plt.title(get_translated_text(lang, "correlation_heatmap"))
                plt.tight_layout()
                st.pyplot(fig3)
                st.subheader("📈 Efficient Frontier : Graph loading, please wait...")
                results, weights_record = optimizer.compute_efficient_frontier()
                portfolio_volatility_arr = results[0]
                portfolio_return_arr = results[1]
                sharpe_ratios = results[2]
                max_sharpe_idx = np.argmax(sharpe_ratios)
                max_sharpe_vol = portfolio_volatility_arr[max_sharpe_idx]
                max_sharpe_ret = portfolio_return_arr[max_sharpe_idx]
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                scatter = ax4.scatter(portfolio_volatility_arr, portfolio_return_arr, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.3)
                ax4.scatter(max_sharpe_vol, max_sharpe_ret, c='red', marker='*', s=200, label='Max Sharpe Ratio')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax4.set_xlabel('Annual \n Volatility (Risk)')
                ax4.set_ylabel('Expected Annual Return')
                ax4.set_title('Efficient Frontier')
                ax4.legend()
                plt.tight_layout()
                st.pyplot(fig4)
                st.markdown("**Analysis:** This portfolio offers the highest Sharpe Ratio, meaning it provides the best risk-adjusted return among the sampled portfolios.")
                st.subheader("🔍 Detailed Metrics for Highest Sharpe Ratio Portfolio")
                detailed_metrics = {
                    "Expected Annual Return (%)": max_sharpe_ret * 100,
                    "Annual Volatility\n(Risk) (%)": max_sharpe_vol * 100,
                    "Sharpe Ratio": sharpe_ratios[max_sharpe_idx],
                    "Value at Risk (VaR)": optimizer.value_at_risk(weights_record[max_sharpe_idx], confidence_level=0.95),
                    "Conditional Value at Risk (CVaR)": optimizer.conditional_value_at_risk(weights_record[max_sharpe_idx], confidence_level=0.95),
                    "Maximum Drawdown": optimizer.maximum_drawdown(weights_record[max_sharpe_idx]),
                    "Herfindahl-Hirschman Index (HHI)": optimizer.herfindahl_hirschman_index(weights_record[max_sharpe_idx])
                }
                detailed_metrics_df = pd.DataFrame.from_dict(detailed_metrics, orient='index', columns=['Value'])
                st.table(detailed_metrics_df.style.format({"Value": lambda x: f"{x:.2f}"}))
                st.subheader("📊 Detailed Performance Metrics")
                for key in [
                    "Expected \n Annual Return (%)",
                    "Annual Volatility\n(Risk) (%)",
                    "Sharpe Ratio",
                    "Value at Risk (VaR)",
                    "Conditional Value at Risk (CVaR)",
                    "Maximum Drawdown",
                    "Herfindahl-Hirschman Index (HHI)"
                ]:
                    value = detailed_metrics.get(key, None)
                    if value is not None:
                        display_value = f"{value:.2f}" if key == "Sharpe Ratio" else (f"{value:.2f}%" if "%" in key else f"{value:.4f}")
                        st.markdown(f"**{key}:** {display_value}")
                        if key == "Value at Risk (VaR)":
                            feedback = analyze_var(value)
                        elif key == "Conditional Value at Risk (CVaR)":
                            feedback = analyze_cvar(value)
                        elif key == "Maximum Drawdown":
                            feedback = analyze_max_drawdown(value)
                        elif key == "Herfindahl-Hirschman Index (HHI)":
                            feedback = analyze_hhi(value)
                        elif key == "Sharpe Ratio":
                            feedback = analyze_sharpe(value)
                        else:
                            feedback = ""
                        if feedback:
                            st.markdown(f"**Analysis:** {feedback}")
                st.success(get_translated_text(lang, "explanation_sharpe_button"))
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                logger.exception("An unexpected error occurred during Sharpe Ratio optimization.")
                st.error(f"{e}")

    # ---------------------------
    # Compare Portfolios Section
    # ---------------------------
    if compare_portfolios_btn:
        if st.session_state['base_portfolio_metrics'] is None or st.session_state['optimized_portfolio_metrics'] is None:
            st.error("Please optimize both the base portfolio and the highest Sharpe Ratio portfolio before comparing.")
        else:
            base_metrics = st.session_state['base_portfolio_metrics']
            optimized_metrics = st.session_state['optimized_portfolio_metrics']
            compare_portfolios(base_metrics, optimized_metrics, lang)

if __name__ == "__main__":
    main()
