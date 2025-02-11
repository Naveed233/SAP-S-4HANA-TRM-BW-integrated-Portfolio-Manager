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
import requests
from fpdf import FPDF  # pip install fpdf2
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Portfolio Optimization App",
    layout="wide",
    initial_sidebar_state="expanded",
)

##############################
# Global Translation Dictionaries
##############################
languages = {
    'English': 'en',
    '日本語': 'ja'
}

translations = {
    'en': {
        "title": "Portfolio Optimization with Advanced Features",
        "intro_heading": "Welcome to the Portfolio Optimization App",
        "intro_text": (
            "This project integrates advanced portfolio optimization with SAP technologies:\n"
            "• It retrieves market and risk data via SAP Treasury Position Flows (API_TRSYPOSFLOW_SRV).\n"
            "• It fetches SAP BW/BEx connection and BEx query outline data using the /bwconnections REST APIs.\n"
            "• It performs portfolio optimization using advanced analytics (including LSTM-based future return predictions).\n"
            "• It generates a professional PDF report of the optimization metrics and visualizations.\n"
            "Future improvements will include live SAP BW data integration and enhanced risk models."
        ),
        "instructions_heading": "How to Use This App",
        "instructions_text": (
            "1. Choose your language from the sidebar.\n"
            "2. Select stocks from different sectors (to achieve diversification).\n"
            "3. Choose your date range for historical analysis (to capture sufficient market data).\n"
            "4. Choose your investment strategy:\n"
            "   - Risk-free Investment: Focuses on minimizing risk while achieving stable returns.\n"
            "   - Profit-focused Investment: Aims to maximize returns with higher risk tolerance.\n"
            "5. Train the LSTM model to forecast future returns (helps predict market trends).\n"
            "6. Optimize the portfolio (either by minimizing volatility or maximizing Sharpe ratio) to balance risk and return.\n"
            "7. Compare the resulting portfolios to see which strategy performs best.\n"
            "8. Save the report as a PDF for future reference."
        ),
        "sap_bw_connections": "SAP BW Connections (Future Feature)",
        "fetch_bw_connections": "Fetch BW Connections",
        "bw_connection_details": "BW Connection Details",
        "bex_query_outline": "BEx Query Outline",
        "save_pdf": "Save PDF Report",
        "restart_selection": "Restart selection",
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
        "strategy_risk_free": "Risk-free Investment",
        "strategy_profit": "Profit-focused Investment",
        "target_return": "Select a specific target return (in %)",
        "train_lstm": "Train LSTM Model for Future Returns Prediction",
        "more_info_lstm": "ℹ️ More Information on LSTM",
        "lstm_loading": "LSTM Graph loading, please wait ...",
        "optimize_portfolio": "Optimize Portfolio",
        "optimize_sharpe": "Optimize for Highest Sharpe Ratio",
        "compare_portfolios": "Compare Sharpe vs Base",
        "portfolio_analysis": "🔍 Portfolio Analysis & Optimization Results",
        "success_lstm": "🤖 LSTM model trained successfully!",
        "error_no_assets_lstm": "Please add at least one asset to your portfolio before training the LSTM model.",
        "error_no_assets_opt": "Please add at least one asset to your portfolio before optimization.",
        "error_date": "Start date must be earlier than end date.",
        "allocation_title": "🔑 Optimal Portfolio Allocation (Target Return: {target}%)",
        "allocation_title_no_target": "🔑 Optimal Portfolio Allocation",
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
        "explanation_var": "**Value at Risk (VaR):** The maximum potential loss over a specified period.",
        "explanation_cvar": "**Conditional Value at Risk (CVaR):** The expected loss beyond VaR.",
        "explanation_max_drawdown": "**Maximum Drawdown:** The largest peak-to-trough decline.",
        "explanation_hhi": "**Herfindahl-Hirschman Index (HHI):** A measure of portfolio concentration.",
        "explanation_sharpe_ratio": "**Sharpe Ratio:** A risk-adjusted return measure.",
        "explanation_sortino_ratio": "**Sortino Ratio:** Like Sharpe but penalizes only downside volatility.",
        "explanation_calmar_ratio": "**Calmar Ratio:** Annualized return divided by maximum drawdown (absolute).",
        "explanation_beta": "**Beta:** Sensitivity to market returns.",
        "explanation_alpha": "**Alpha:** Excess return relative to market expectations.",
        "explanation_lstm": "**LSTM Model Explanation:** LSTM is a neural network effective for predicting time-series data such as stock returns.",
        "feedback_sharpe_good": "Great! The Sharpe Ratio is above 1.",
        "feedback_sharpe_average": "Average Sharpe Ratio.",
        "feedback_sharpe_poor": "Poor Sharpe Ratio. Consider adjusting your strategy.",
        "feedback_sortino_good": "Excellent Sortino Ratio!",
        "feedback_sortino_average": "Average Sortino Ratio.",
        "feedback_sortino_poor": "Poor Sortino Ratio.",
        "feedback_calmar_good": "Excellent Calmar Ratio!",
        "feedback_calmar_average": "Good Calmar Ratio.",
        "feedback_calmar_poor": "Poor Calmar Ratio.",
        "feedback_beta_high": "High Beta: Portfolio is more volatile than the market.",
        "feedback_beta_moderate": "Moderate Beta: Comparable to the market.",
        "feedback_beta_low": "Low Beta: Less volatile than the market.",
        "feedback_alpha_positive": "Positive Alpha: Outperforming the market.",
        "feedback_alpha_neutral": "Neutral Alpha.",
        "feedback_alpha_negative": "Negative Alpha: Underperforming the market.",
        "feedback_hhi_high": "High concentration: Poor diversification.",
        "feedback_hhi_moderate": "Moderate concentration: Some diversification.",
        "feedback_hhi_good": "Good diversification.",
        "success_optimize": "Portfolio optimization completed successfully!",
        "explanation_sharpe_button": "**Optimize for Highest Sharpe Ratio:** Maximizes risk-adjusted return.",
        "recommendation": "Based on these metrics, the {better_portfolio} portfolio is recommended for better {better_metric}."
    },
    'ja': {
        "title": "高度な機能を備えたポートフォリオ最適化アプリ",
        "intro_heading": "ポートフォリオ最適化アプリへようこそ",
        "intro_text": (
            "このプロジェクトは、SAPテクノロジーを活用して高度なポートフォリオ最適化を実現します。\n"
            "• SAPトレジャリー・ポジションフロー（API_TRSYPOSFLOW_SRV）を利用して市場およびリスクデータを取得します。\n"
            "• /bwconnections REST APIを使用してSAP BW/BEx接続およびBExクエリ概要を取得します。\n"
            "• LSTMを用いた将来リターン予測を含む高度な解析によりポートフォリオの最適化を行います。\n"
            "• 最適化結果と視覚化をPDFレポートとして生成・保存します。\n"
            "今後、SAP BWのライブデータ連携やリスクモデルの強化を予定しています。"
        ),
        "instructions_heading": "アプリの使い方",
        "instructions_text": (
            "1. サイドバーから言語を選択してください。\n"
            "2. 異なるセクターから株式を選択して分散投資を図ります。\n"
            "3. 過去の分析に必要な十分なデータを取得するため、日付範囲を選択してください。\n"
            "4. 投資戦略を選択してください：\n"
            "   - リスクフリー投資：リスクを最小限に抑え安定したリターンを目指す。\n"
            "   - 利益重視投資：リスク許容度を高めリターンの最大化を狙う。\n"
            "5. LSTMモデルを訓練して将来リターンを予測します（市場動向の予測に有効）。\n"
            "6. ポートフォリオを最適化して、リスクとリターンのバランスを見つけます。\n"
            "7. 異なる戦略の結果を比較して最良の戦略を判断します。\n"
            "8. 最終結果をPDFレポートとして保存します。"
        ),
        "sap_bw_connections": "SAP BW接続（将来実装予定）",
        "fetch_bw_connections": "BW接続を取得",
        "bw_connection_details": "BW接続詳細",
        "bex_query_outline": "BExクエリ概要",
        "save_pdf": "PDFレポートを保存",
        "restart_selection": "選択をリセット",
        "user_inputs": "🔧 ユーザー入力",
        "select_universe": "資産ユニバースを選択してください：",
        "custom_tickers": "株式ティッカーをカンマ区切りで入力してください（例：AAPL, MSFT, TSLA）：",
        "add_portfolio": "マイポートフォリオに追加",
        "my_portfolio": "📁 マイポートフォリオ",
        "no_assets": "まだ資産が追加されていません。",
        "optimization_parameters": "📅 最適化パラメータ",
        "start_date": "開始日",
        "end_date": "終了日",
        "risk_free_rate": "無リスク金利（％）を入力してください：",
        "investment_strategy": "投資戦略を選択してください：",
        "strategy_risk_free": "リスクフリー投資",
        "strategy_profit": "利益重視投資",
        "target_return": "特定の目標リターン（％）を選択してください",
        "train_lstm": "将来リターン予測のためにLSTMモデルを訓練",
        "more_info_lstm": "ℹ️ LSTMに関する詳細情報",
        "lstm_loading": "LSTMグラフを読み込み中です。しばらくお待ちください…",
        "optimize_portfolio": "ポートフォリオを最適化",
        "optimize_sharpe": "シャープレシオ最大化のために最適化",
        "compare_portfolios": "シャープ vs ベースを比較",
        "portfolio_analysis": "🔍 ポートフォリオ分析と最適化結果",
        "success_lstm": "🤖 LSTMモデルの訓練に成功しました！",
        "error_no_assets_lstm": "LSTMモデルの訓練前に、少なくとも1つの資産を追加してください。",
        "error_no_assets_opt": "最適化前に、少なくとも1つの資産を追加してください。",
        "error_date": "開始日は終了日より前でなければなりません。",
        "allocation_title": "🔑 最適なポートフォリオ配分（目標リターン：{target}％）",
        "allocation_title_no_target": "🔑 最適なポートフォリオ配分",
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
        "explanation_var": "**リスク価値 (VaR):** 指定された期間内の最大損失を推定します。",
        "explanation_cvar": "**条件付きリスク価値 (CVaR):** VaRを超える損失の期待値を測定します。",
        "explanation_max_drawdown": "**最大ドローダウン:** ピークからの最大下落率。",
        "explanation_hhi": "**ハーフィンダール・ハーシュマン指数 (HHI):** 投資集中度の指標。",
        "explanation_sharpe_ratio": "**シャープレシオ:** リスク調整後のリターンを示します。",
        "explanation_sortino_ratio": "**ソルティーノレシオ:** 下方リスクにのみペナルティを与えた指標です。",
        "explanation_calmar_ratio": "**カルマーレシオ:** 年率リターン／最大ドローダウン（絶対値）。",
        "explanation_beta": "**ベータ:** 市場との連動性。",
        "explanation_alpha": "**アルファ:** 市場期待を上回る超過リターン。",
        "explanation_lstm": "**LSTMモデルの説明：**\nLSTMは時系列データ（株式リターン等）の予測に有効なニューラルネットワークです。",
        "feedback_sharpe_good": "素晴らしいです！シャープレシオが1以上です。",
        "feedback_sharpe_average": "平均的なシャープレシオです。",
        "feedback_sharpe_poor": "シャープレシオが低いです。戦略の調整を検討してください。",
        "feedback_sortino_good": "優れたソルティーノレシオです！",
        "feedback_sortino_average": "平均的なソルティーノレシオです。",
        "feedback_sortino_poor": "ソルティーノレシオが低いです。",
        "feedback_calmar_good": "優れたカルマーレシオです！",
        "feedback_calmar_average": "良好なカルマーレシオです。",
        "feedback_calmar_poor": "カルマーレシオが低いです。",
        "feedback_beta_high": "高いベータ：市場よりもボラティリティが高いです。",
        "feedback_beta_moderate": "中程度のベータです。",
        "feedback_beta_low": "低いベータ：市場よりもボラティリティが低いです。",
        "feedback_alpha_positive": "プラスのアルファ：市場を上回っています。",
        "feedback_alpha_neutral": "アルファは市場と同程度です。",
        "feedback_alpha_negative": "マイナスのアルファ：市場を下回っています。",
        "feedback_hhi_high": "高い集中度：分散投資が不足しています。",
        "feedback_hhi_moderate": "中程度の集中度：ある程度分散されています。",
        "feedback_hhi_good": "良好な分散投資ができています。",
        "success_optimize": "ポートフォリオの最適化が正常に完了しました！",
        "explanation_sharpe_button": "**シャープレシオ最大化のために最適化：** リスク調整後の最高のリターンを目指します。",
        "recommendation": "Based on these metrics, the {better_portfolio} portfolio is recommended for better {better_metric}."
    }
}

##############################
# SAP BW / BEx Integration Functions
##############################
BW_BASE_URL = "http://<server_name>:6405/biprws/raylight/v1"  # Replace <server_name> with your actual server

def fetch_bw_connections(offset=0, limit=10):
    url = f"{BW_BASE_URL}/bwconnections?offset={offset}&limit={limit}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("BW Connections fetch failed: " + response.text)
            return None
    except Exception as e:
        logger.exception("Exception fetching BW connections.")
        return None

def fetch_bw_connection_details(bwConnectionID):
    url = f"{BW_BASE_URL}/bwconnections/{bwConnectionID}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("BW Connection details fetch failed: " + response.text)
            return None
    except Exception as e:
        logger.exception("Exception fetching BW connection details.")
        return None

def fetch_bex_query_outline(bwConnectionID):
    url = f"{BW_BASE_URL}/bwconnections/{bwConnectionID}/outline"
    try:
        response = requests.put(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("BEx Query outline fetch failed: " + response.text)
            return None
    except Exception as e:
        logger.exception("Exception fetching BEx Query outline.")
        return None

def fetch_bex_query_capabilities(bwConnectionID):
    url = f"{BW_BASE_URL}/bwconnections/{bwConnectionID}/capabilities"
    try:
        response = requests.put(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("BEx Query capabilities fetch failed: " + response.text)
            return None
    except Exception as e:
        logger.exception("Exception fetching BEx Query capabilities.")
        return None

##############################
# PDF Report Generation Function
##############################
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Portfolio Optimization Report", ln=True, align="C")
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def generate_pdf_report(opt_metrics, graph_paths, language="en"):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Portfolio Optimization Report", ln=True, align="C")
    pdf.ln(5)
    pdf.cell(0, 10, "Optimization Metrics:", ln=True)
    for key, value in opt_metrics.items():
        pdf.cell(0, 10, f"{key}: {value:.4f}", ln=True)
    pdf.ln(5)
    if graph_paths:
        pdf.cell(0, 10, "Graphs:", ln=True)
        for path in graph_paths:
            pdf.image(path, w=pdf.epw)
            pdf.ln(10)
    pdf.cell(0, 10, "End of Report", ln=True, align="C")
    return pdf.output(dest="S").encode("latin1")

##############################
# Portfolio Optimizer Class (Including SAP TRM Integration and Additional Metrics)
##############################
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.returns = None
        self.use_sap_api = True

    def fetch_data(self):
        logger.info(f"Fetching data for tickers: {self.tickers}")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
        st.write("Fetched Data Preview:", data.head())
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                data = data.xs('Adj Close', axis=1, level=0)
            elif 'Close' in data.columns.levels[0]:
                data = data.xs('Close', axis=1, level=0)
            else:
                st.error("Neither 'Adj Close' nor 'Close' columns available in multi-index.")
                raise ValueError("Missing columns in multi-index.")
        else:
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                st.error("Neither 'Adj Close' nor 'Close' columns available.")
                raise ValueError("Missing columns.")
        data.dropna(axis=1, how='all', inplace=True)
        if data.empty:
            raise ValueError("No data fetched. Please check tickers and date range.")
        if isinstance(data, pd.DataFrame):
            self.tickers = list(data.columns)
        else:
            self.tickers = [data.name]
            data = pd.DataFrame(data)
        self.returns = data.pct_change().dropna()
        return self.tickers

    def portfolio_stats(self, weights):
        weights = np.array(weights)
        if len(weights) != len(self.tickers):
            raise ValueError("Weights length mismatch.")
        weights = weights / np.sum(weights)
        port_return = np.dot(weights, self.returns.mean()) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (port_return - self.risk_free_rate) / port_vol
        return port_return, port_vol, sharpe_ratio

    def compute_additional_metrics(self, weights):
        port_return, port_vol, sharpe_ratio = self.portfolio_stats(weights)
        daily_returns = self.returns.dot(weights)
        target = self.risk_free_rate / 252
        downside_returns = daily_returns[daily_returns < target]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean((downside_returns - target) ** 2))
        else:
            downside_deviation = 0.0001
        sortino_ratio = (((port_return / 252) - target) / downside_deviation) * np.sqrt(252)
        max_dd = self.maximum_drawdown(weights)
        calmar_ratio = (port_return / 252) / abs(max_dd) if max_dd != 0 else 0
        try:
            benchmark_data = yf.download("SPY", start=self.start_date, end=self.end_date, progress=False)
            benchmark_prices = benchmark_data['Adj Close'] if 'Adj Close' in benchmark_data.columns else benchmark_data['Close']
            benchmark_returns = benchmark_prices.pct_change().dropna()
            common_idx = self.returns.index.intersection(benchmark_returns.index)
            port_daily = self.returns.dot(weights).loc[common_idx]
            bench_daily = benchmark_returns.loc[common_idx]
            covariance = np.cov(port_daily, bench_daily)[0, 1]
            variance = np.var(bench_daily)
            beta = covariance / variance if variance != 0 else 0
            bench_annual_return = bench_daily.mean() * 252
            alpha = port_return - (self.risk_free_rate + beta * (bench_annual_return - self.risk_free_rate))
        except Exception as e:
            beta = 0
            alpha = 0
        return sortino_ratio, calmar_ratio, beta, alpha
        
# --- SAP TRM Integration using Treasury Position Flows API ---
    def fetch_sap_portfolio_risk_metrics(self, weights):
        SAP_API_KEY = "o6aLGqMRUwKu8ispGpYnwLuM46PKKwje"  # Provided API key (store securely)
        sap_api_url = "https://api.sap.com/sap/opu/odata/sap/API_TRSYPOSFLOW_SRV/TreasuryPositionFlows?$format=json"
        headers = {
            "APIKey": SAP_API_KEY,
            "Content-Type": "application/json"
        }
        try:
            response = requests.get(sap_api_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # Dummy risk metrics for demonstration purposes
                return {"VaR": -0.03, "CVaR": -0.04}
            else:
                logger.warning("SAP TRM API call failed: " + response.text)
                return None
        except Exception as e:
            logger.exception("Exception during SAP TRM API call.")
            return None
            
    def value_at_risk(self, weights, confidence_level=0.95):
        if self.use_sap_api:
            sap_data = self.fetch_sap_portfolio_risk_metrics(weights)
            if sap_data and "VaR" in sap_data:
                return sap_data["VaR"]
        portfolio_returns = self.returns.dot(weights)
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    def conditional_value_at_risk(self, weights, confidence_level=0.95):
        if self.use_sap_api:
            sap_data = self.fetch_sap_portfolio_risk_metrics(weights)
            if sap_data and "CVaR" in sap_data:
                return sap_data["CVaR"]
        portfolio_returns = self.returns.dot(weights)
        var = self.value_at_risk(weights, confidence_level)
        return portfolio_returns[portfolio_returns <= var].mean()

    def maximum_drawdown(self, weights):
        portfolio_returns = self.returns.dot(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.cummax()
        return ((cumulative - peak) / peak).min()

    def herfindahl_hirschman_index(self, weights):
        return np.sum(weights ** 2)

    def sharpe_ratio_objective(self, weights):
        _, _, sharpe = self.portfolio_stats(weights)
        return -sharpe

    def optimize_sharpe_ratio(self):
        num_assets = len(self.tickers)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        result = minimize(self.sharpe_ratio_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else initial_weights

    def min_volatility(self, target_return, max_weight=0.3):
        num_assets = len(self.tickers)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: self.portfolio_stats(weights)[0] - target_return}
        )
        bounds = tuple((0, max_weight) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets
        result = minimize(lambda weights: self.portfolio_stats(weights)[1], init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else np.ones(num_assets) / num_assets

    def prepare_data_for_lstm(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.returns.values)
        X, y = [], []
        look_back = 60
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i])
            y.append(scaled_data[i])
        split = int(len(X) * 0.8)
        X_train, X_test = np.array(X[:split]), np.array(X[split:])
        y_train, y_test = np.array(y[:split]), np.array(y[split:])
        if not X_train.size or not y_train.size:
            raise ValueError("Not enough data for training.")
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
            raise ValueError("Not enough data for prediction.")
        last_data = self.returns[-60:].values
        scaled_last_data = scaler.transform(last_data)
        X_test = np.array([scaled_last_data])
        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)
        return predicted[0][:steps] if len(predicted[0]) >= steps else predicted[0]

    def evaluate_model(self, model, scaler, X_test, y_test):
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_inverse = scaler.inverse_transform(y_test)
        mae = mean_absolute_error(y_test_inverse, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions))
        r2 = r2_score(y_test_inverse, predictions)
        return mae, rmse, r2

    def compute_efficient_frontier(self, num_portfolios=200):
        results = np.zeros((4, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            weights_record.append(weights)
            port_return, port_vol, sharpe = self.portfolio_stats(weights)
            var = self.value_at_risk(weights, confidence_level=0.95)
            cvar = self.conditional_value_at_risk(weights, confidence_level=0.95)
            results[0, i] = port_vol
            results[1, i] = port_return
            results[2, i] = sharpe
            results[3, i] = self.herfindahl_hirschman_index(weights)
        return results, weights_record

##############################
# Helper Functions for Display & Comparison
##############################
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
        return "Severe Drawdown: Major decline."
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
        return "Good Diversification."

def analyze_sharpe(sharpe):
    if sharpe > 1:
        return "Great! Portfolio is generating good returns."
    elif 0.5 < sharpe <= 1:
        return "Average. Returns are acceptable."
    else:
        return "Poor. Consider adjusting your strategy."

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
        analysis_text = {
            "var": analyze_var,
            "cvar": analyze_cvar,
            "max_drawdown": analyze_max_drawdown,
            "hhi": analyze_hhi,
            "sharpe_ratio": analyze_sharpe,
            "sortino_ratio": analyze_sharpe,
            "calmar_ratio": analyze_sharpe,
            "beta": analyze_sharpe,
            "alpha": analyze_sharpe
        }.get(key, lambda x: "")(value)
        metric_display.append({
            "Metric": display_key,
            "Value": display_value,
            "Analysis": analysis_text
        })
    metrics_df = pd.DataFrame.from_dict(metric_display)
    st.table(metrics_df.style.set_properties(**{'text-align': 'left', 'padding': '5px'}))

def compare_portfolios(base_metrics, optimized_metrics, lang):
    comparison_data = []
    recommendations = []
    for key in base_metrics.keys():
        base_value = base_metrics[key]
        optimized_value = optimized_metrics[key]
        metric_display = get_translated_text(lang, key)
        if base_value == 0 and optimized_value == 0:
            better = "-"
        else:
            if key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "alpha"]:
                better = "Optimized" if optimized_value > base_value else "Base"
            elif key in ["var", "cvar", "max_drawdown", "beta", "hhi"]:
                better = "Optimized" if optimized_value < base_value else "Base"
            else:
                better = "-"
        if better != "-":
            recommendations.append((metric_display, better))
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
        styles = [''] * len(row)
        if row['Better'] == "Optimized":
            styles[comparison_df.columns.get_loc("Optimized Portfolio")] = 'background-color: lightgreen'
        elif row['Better'] == "Base":
            styles[comparison_df.columns.get_loc("Base Portfolio")] = 'background-color: lightgreen'
        return styles
    comparison_df = comparison_df.style.apply(highlight_better, axis=1)
    st.markdown("<h3>📊 Comparison: Sharpe vs Base Portfolio</h3>", unsafe_allow_html=True)
    st.table(comparison_df)
    if recommendations:
        metric_rec, rec_port = recommendations[0]
        rec_text = f"Based on the {metric_rec} metric, the {rec_port} portfolio is recommended."
    else:
        rec_text = "No clear recommendation; metrics are neutral."
    st.markdown(f"<p><strong>Recommendation:</strong> {rec_text}</p>", unsafe_allow_html=True)

##############################
# Introduction / Notice Section
##############################
if "show_intro" not in st.session_state:
    st.session_state["show_intro"] = True

def show_introduction(lang):
    st.info(f"### {get_translated_text(lang, 'intro_heading')}\n\n{get_translated_text(lang, 'intro_text')}")

def show_instructions(lang):
    st.markdown(f"## {get_translated_text(lang, 'instructions_heading')}")
    st.markdown(get_translated_text(lang, "instructions_text"))

##############################
# Main Streamlit App
##############################
def main():
    # Sidebar language selection
    st.sidebar.header("🌐 Language Selection")
    selected_language = st.sidebar.selectbox("Select Language:", options=list(languages.keys()), index=0)
    lang = languages[selected_language]
    
    # If any action button is pressed, hide introduction/instructions.
    action_pressed = st.sidebar.button(get_translated_text(lang, "train_lstm")) or \
                     st.sidebar.button(get_translated_text(lang, "optimize_portfolio")) or \
                     st.sidebar.button(get_translated_text(lang, "optimize_sharpe")) or \
                     st.sidebar.button(get_translated_text(lang, "compare_portfolios"))
    if action_pressed:
        st.session_state["show_intro"] = False

    if st.session_state["show_intro"]:
        show_introduction(lang)
        show_instructions(lang)
    
    st.title(get_translated_text(lang, "title"))
    
    # SAP BW/BEx Section in Sidebar
    st.sidebar.header(get_translated_text(lang, "sap_bw_connections"))
    if st.sidebar.button(get_translated_text(lang, "fetch_bw_connections")):
        bw_connections = fetch_bw_connections()
        if bw_connections:
            st.sidebar.write(bw_connections)
        else:
            st.sidebar.write("BW Connections could not be fetched.")
        st.sidebar.caption("This feature will be implemented in a future version.")
    
    # Display current portfolio and Restart Selection button
    st.sidebar.subheader(get_translated_text(lang, "my_portfolio"))
    if 'my_portfolio' not in st.session_state:
        st.session_state['my_portfolio'] = []
    if st.session_state['my_portfolio']:
        st.sidebar.write(", ".join(st.session_state['my_portfolio']))
        if st.sidebar.button(get_translated_text(lang, "restart_selection")):
            st.session_state['my_portfolio'] = []
    else:
        st.sidebar.write(get_translated_text(lang, "no_assets"))
    
    # Sidebar for User Inputs
    st.sidebar.header(get_translated_text(lang, "user_inputs"))
    universe_options = {
        'Tech Giants': ['AAPL - Apple', 'MSFT - Microsoft', 'GOOGL - Alphabet', 'AMZN - Amazon', 'META - Meta Platforms', 'TSLA - Tesla', 'NVDA - NVIDIA', 'ADBE - Adobe', 'INTC - Intel', 'CSCO - Cisco'],
        'Finance Leaders': ['JPM - JPMorgan Chase', 'BAC - Bank of America', 'WFC - Wells Fargo', 'C - Citigroup', 'GS - Goldman Sachs', 'MS - Morgan Stanley', 'AXP - American Express', 'BLK - BlackRock', 'SCHW - Charles Schwab', 'USB - U.S. Bancorp'],
        'Healthcare Majors': ['JNJ - Johnson & Johnson', 'PFE - Pfizer', 'UNH - UnitedHealth', 'MRK - Merck', 'ABBV - AbbVie', 'ABT - Abbott', 'TMO - Thermo Fisher Scientific', 'MDT - Medtronic', 'DHR - Danaher', 'BMY - Bristol-Myers Squibb'],
        'Custom': []
    }
    universe_choice = st.sidebar.selectbox(get_translated_text(lang, "select_universe"), options=list(universe_options.keys()), index=0)
    if universe_choice == 'Custom':
        custom_tickers = st.sidebar.text_input(get_translated_text(lang, "custom_tickers"), value="")
    else:
        selected_universe_assets = st.sidebar.multiselect(get_translated_text(lang, "add_portfolio"), universe_options[universe_choice], default=[])
    
    # Add assets to portfolio
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
    
    # Optimization Parameters
    st.sidebar.header(get_translated_text(lang, "optimization_parameters"))
    start_date = st.sidebar.date_input(get_translated_text(lang, "start_date"), value=datetime(2024, 1, 1), max_value=datetime.today())
    def get_last_day_previous_month():
        today = datetime.today()
        first_day = today.replace(day=1)
        return first_day - pd.Timedelta(days=1)
    end_date = st.sidebar.date_input(get_translated_text(lang, "end_date"), value=get_last_day_previous_month(), max_value=datetime.today())
    risk_free_rate = st.sidebar.number_input(get_translated_text(lang, "risk_free_rate"), value=2.0, step=0.1) / 100
    investment_strategy = st.sidebar.radio(get_translated_text(lang, "investment_strategy"), (get_translated_text(lang, "strategy_risk_free"), get_translated_text(lang, "strategy_profit")))
    if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
        specific_target_return = st.sidebar.slider(get_translated_text(lang, "target_return"), min_value=-5.0, max_value=20.0, value=5.0, step=0.1) / 100
        allocation_title = get_translated_text(lang, "allocation_title").format(target=round(specific_target_return * 100, 2))
    else:
        specific_target_return = None
        allocation_title = get_translated_text(lang, "allocation_title_no_target")
    
    # Action Buttons
    train_lstm = st.sidebar.button(get_translated_text(lang, "train_lstm"))
    optimize_portfolio = st.sidebar.button(get_translated_text(lang, "optimize_portfolio"))
    optimize_sharpe = st.sidebar.button(get_translated_text(lang, "optimize_sharpe"))
    compare_portfolios_btn = st.sidebar.button(get_translated_text(lang, "compare_portfolios"))
    save_pdf_report = st.sidebar.button(get_translated_text(lang, "save_pdf"))
    
    st.header(get_translated_text(lang, "portfolio_analysis"))
    
    # --- LSTM Training Section ---
    if train_lstm:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_lstm"))
        else:
            st.info(get_translated_text(lang, "lstm_loading"))
            try:
                clean_tickers = st.session_state['my_portfolio']
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                optimizer.fetch_data()
                X_train, y_train, X_test, y_test, scaler = optimizer.prepare_data_for_lstm()
                model = optimizer.train_lstm_model(X_train, y_train, epochs=10, batch_size=32)
                mae, rmse, r2 = optimizer.evaluate_model(model, scaler, X_test, y_test)
                st.success(get_translated_text(lang, "success_lstm"))
                st.subheader("LSTM Model Evaluation Metrics")
                eval_metrics = {"Mean Absolute Error (MAE)": mae, "Root Mean Squared Error (RMSE)": rmse, "R-squared (R²)": r2}
                st.table(pd.DataFrame.from_dict(eval_metrics, orient='index', columns=['Value']).style.format({"Value": "{:.4f}"}))
                future_returns = optimizer.predict_future_returns(model, scaler, steps=30)
                future_dates = pd.date_range(end_date, periods=len(future_returns), freq='B').to_pydatetime().tolist()
                prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Returns': future_returns})
                fig_lstm, ax_lstm = plt.subplots(figsize=(10, 4))
                ax_lstm.plot(prediction_df['Date'], prediction_df['Predicted Returns'], label="Predicted Returns", color='blue')
                ax_lstm.set_xlabel("Date")
                ax_lstm.set_ylabel("Predicted Returns")
                ax_lstm.set_title(get_translated_text(lang, "train_lstm"))
                ax_lstm.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_lstm)
                st.session_state["fig_lstm"] = fig_lstm
            except Exception as e:
                st.error(str(e))
    
    # --- Optimize Portfolio Section ---
    if optimize_portfolio:
        if not st.session_state['my_portfolio']:
            st.error(get_translated_text(lang, "error_no_assets_opt"))
        elif start_date >= end_date:
            st.error(get_translated_text(lang, "error_date"))
        else:
            try:
                clean_tickers = st.session_state['my_portfolio']
                optimizer = PortfolioOptimizer(clean_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_free_rate)
                updated_tickers = optimizer.fetch_data()
                if investment_strategy == get_translated_text(lang, "strategy_risk_free"):
                    if specific_target_return is None:
                        st.error("Please select a target return for Risk-free Investment strategy.")
                        st.stop()
                    optimal_weights = optimizer.min_volatility(specific_target_return)
                else:
                    optimal_weights = optimizer.optimize_sharpe_ratio()
                port_return, port_vol, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
                var_95 = optimizer.value_at_risk(optimal_weights, confidence_level=0.95)
                cvar_95 = optimizer.conditional_value_at_risk(optimal_weights, confidence_level=0.95)
                max_dd = optimizer.maximum_drawdown(optimal_weights)
                hhi = optimizer.herfindahl_hirschman_index(optimal_weights)
                sortino_ratio, calmar_ratio, beta, alpha = optimizer.compute_additional_metrics(optimal_weights)
                allocation = pd.DataFrame({"Asset": updated_tickers, "Weight (%)": np.round(optimal_weights * 100, 2)})
                allocation = allocation[allocation['Weight (%)'] > 0].reset_index(drop=True)
                st.subheader(allocation_title)
                st.dataframe(allocation.style.format({"Weight (%)": "{:.2f}"}))
                metrics = {
                    "var": var_95,
                    "cvar": cvar_95,
                    "max_drawdown": max_dd,
                    "hhi": hhi,
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio,
                    "beta": beta,
                    "alpha": alpha
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
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                    ax_pie.pie(allocation['Weight (%)'], labels=allocation['Asset'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                    ax_pie.axis('equal')
                    ax_pie.set_title(get_translated_text(lang, "portfolio_composition"))
                    st.pyplot(fig_pie)
                    st.session_state["fig_pie"] = fig_pie
                with col2:
                    fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
                    perf_metrics = {"Expected\n Annual Return (%)": port_return * 100, "Annual Volatility\n(Risk) (%)": port_vol * 100, "Sharpe Ratio": sharpe_ratio}
                    metrics_bar = pd.DataFrame.from_dict(perf_metrics, orient='index', columns=['Value'])
                    sns.barplot(x=metrics_bar.index, y='Value', data=metrics_bar, palette='viridis', ax=ax_bar)
                    ax_bar.set_title(get_translated_text(lang, "portfolio_metrics"))
                    for p in ax_bar.patches:
                        ax_bar.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width()/2., p.get_height()),
                                        ha='center', va='bottom', fontsize=10)
                    plt.xticks(rotation=0, ha='center')
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                st.subheader(get_translated_text(lang, "correlation_heatmap"))
                corr_matrix = optimizer.returns.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='Spectral', linewidths=0.3, ax=ax_corr,
                            cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 8})
                ax_corr.set_title(get_translated_text(lang, "correlation_heatmap"))
                plt.tight_layout()
                st.pyplot(fig_corr)
                st.subheader("📈 Efficient Frontier : Graph loading, please wait ...")
                results, weights_record = optimizer.compute_efficient_frontier(num_portfolios=200)
                vol_arr, ret_arr, sharpe_arr, _ = results
                max_sharpe_idx = np.argmax(sharpe_arr)
                max_sharpe_vol = vol_arr[max_sharpe_idx]
                max_sharpe_ret = ret_arr[max_sharpe_idx]
                fig_ef, ax_ef = plt.subplots(figsize=(10, 6))
                scatter = ax_ef.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', marker='o', s=10, alpha=0.3)
                ax_ef.scatter(max_sharpe_vol, max_sharpe_ret, c='red', marker='*', s=200, label='Max Sharpe Ratio')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax_ef.set_xlabel("Annual Volatility (Risk)")
                ax_ef.set_ylabel("Expected Annual Return")
                ax_ef.set_title("Efficient Frontier")
                ax_ef.legend()
                plt.tight_layout()
                st.pyplot(fig_ef)
                st.session_state["fig_efficient"] = fig_ef
                st.success(get_translated_text(lang, "success_optimize"))
            except Exception as e:
                st.error(str(e))
    
    # --- PDF Report Generation Section ---
    if save_pdf_report:
        opt_metrics = st.session_state.get('optimized_portfolio_metrics', st.session_state.get('base_portfolio_metrics'))
        if opt_metrics is None:
            st.error("No optimization metrics available to generate the report.")
        else:
            graph_paths = []
            # Save efficient frontier, LSTM, and portfolio composition graphs if available
            for key in ["fig_efficient", "fig_lstm", "fig_pie"]:
                if key in st.session_state:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        st.session_state[key].savefig(tmpfile.name, bbox_inches='tight')
                        graph_paths.append(tmpfile.name)
            pdf_bytes = generate_pdf_report(opt_metrics, graph_paths, language=lang)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="Portfolio_Report.pdf", mime="application/pdf")
    
    # --- Compare Portfolios Section ---
    if compare_portfolios_btn:
        if st.session_state['base_portfolio_metrics'] is None or st.session_state['optimized_portfolio_metrics'] is None:
            st.error("Please optimize both the base portfolio and the highest Sharpe Ratio portfolio before comparing.")
        else:
            compare_portfolios(st.session_state['base_portfolio_metrics'], st.session_state['optimized_portfolio_metrics'], lang)

if __name__ == "__main__":
    main()
