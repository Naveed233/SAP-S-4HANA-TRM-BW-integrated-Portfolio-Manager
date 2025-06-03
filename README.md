# SAP-S-4HANA-TRM-BW-integrated-Portfolio-Manager

# Portfolio Optimization App – README

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the App](#running-the-app)
7. [Using the App – Step-by-Step](#using-the-app)
8. [Explanation of Metrics](#explanation-of-metrics)
9. [Customisation & Extensibility](#customisation--extensibility)
10. [Troubleshooting](#troubleshooting)
11. [Roadmap](#roadmap)
12. [License](#license)

---

## Introduction

This Streamlit application provides **end-to-end portfolio construction and analysis** for both real-market data (via Yahoo Finance) and hypothetical “fixed-return” scenarios. It is aimed at investors, students, and researchers who want an approachable interface plus advanced quantitative tools—without writing a single line of code.

---

## Key Features

| Category                   | Capability                                                                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Multi-language UI**      | English & Japanese (easy to extend)                                                                                                             |
| **Asset Modes**            | • *Market Data* – live/historical prices via Yahoo Finance  <br>• *Fixed Returns* – user-defined expected returns & covariance                  |
| **Universe Builder**       | Pre-set “Tech Giants”, “Finance Leaders”, “Healthcare Majors”, “Cryptocurrencies”, or fully custom tickers                                      |
| **Portfolio Optimisation** | • Minimum-volatility portfolio for a target return  <br>• Maximum-Sharpe (risk-adjusted) portfolio  <br>• Position size caps, long-only weights |
| **Risk Metrics**           | VaR, CVaR, Max Drawdown, Herfindahl-Hirschman Index, Sharpe                                                                                     |
| **Visual Analytics**       | Allocation pie chart, performance bar chart, correlation heat-map                                                                               |
| **Machine Learning Stub**  | Hooks for training an LSTM price-forecast model (placeholder ready for future implementation)                                                   |

---

## Architecture at a Glance

```
┌────────┐        ┌─────────────┐
│ Stream │ UI ◀──▶│  Session    │
│  lit   │        │  State      │
└────────┘        └────┬────────┘
                       │ calls
               ┌───────▼────────┐
               │ Business Logic │
               │  (optimiser)   │
               └───────┬────────┘
          fetch & calc │
                       ▼
                 ┌────────┐
                 │ yfinance│ (market)
                 └────────┘
```

All heavy numerics run in **NumPy / CVXPY / SciPy**, ensuring reproducibility and no opaque black-box optimisations.

---

## Prerequisites

| Requirement      | Version / Notes                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| Python           | **3.9+** recommended (3.8–3.12 tested)                                                                 |
| pip              | Latest                                                                                                 |
| System libraries | • `tk` for matplotlib back-end (usually pre-installed) <br>• Suitable C/C++ compiler if CVXPY needs it |

---

## Installation

```bash
# 1. Clone the repository (or copy the app.py file)
git clone https://github.com/your-org/portfolio-optimiser.git
cd portfolio-optimiser

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```text
streamlit
yfinance
numpy
pandas
matplotlib
seaborn
scipy
cvxpy
scikit-learn
tensorflow       # optional – only needed for LSTM section
```

---

## Running the App

```bash
streamlit run app.py
```

The browser should open automatically; if not, visit `http://localhost:8501`.

---

## Using the App

### 1. Sidebar – Language & Mode

1. **Language Selection** – English or 日本語.
2. **Asset Mode**

   * **Market Data** – pull price history from Yahoo; supports arbitrary symbols (e.g., `AAPL`, `BTC-USD`).
   * **Fixed Returns** – specify expected returns for four pre-defined asset classes (Stock, Crypto, Bond, Derivative) and optimise against a synthetic covariance matrix.

### 2. Market Data Workflow

| Step                      | Action                                                                                                                                                                                                    |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Universe**              | Pick a sector list or choose “Custom” and enter comma-separated tickers.                                                                                                                                  |
| **My Portfolio**          | Use *Add to My Portfolio* to commit selections.                                                                                                                                                           |
| **Parameters**            | Set **Start / End** dates, **Risk-free rate**, and choose **Investment Strategy**:<br>• *Risk Averse* – minimise volatility for a chosen target return.<br>• *Profit-focused* – directly maximise Sharpe. |
| **Train LSTM** (optional) | Placeholder button (prints preview; full ML integration ready).                                                                                                                                           |
| **Optimise Portfolio**    | Computes weights, risk metrics, visual charts, and a recommendation.                                                                                                                                      |

### 3. Fixed Returns Workflow

1. Enter expected returns for each asset class.
2. Select optimisation objective: *Risk-Averse* (target return slider) or *Profit-Focused* (global max-Sharpe search).
3. Click **Optimise** – results include allocation, Sharpe, pie chart.

---

## Explanation of Metrics

| Metric           | Formula                            | Interpretation (Higher = Better unless noted)                                             |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| **VaR (95 %)**   | 5th percentile of portfolio return | Potential one-day loss not exceeded 95 % of the time *(negative number; higher is safer)* |
| **CVaR (95 %)**  | Mean loss beyond VaR               | Tail-risk indicator *(negative; closer to 0 is safer)*                                    |
| **Max Drawdown** | Min(Tₜ / peakₜ – 1)                | Worst peak-to-trough drop *(negative)*                                                    |
| **HHI**          | Σ wᵢ²                              | Concentration < 0.3 diversified・> 0.6 concentrated                                        |
| **Sharpe**       | (μ – rₑ) / σ                       | Risk-adjusted performance *(> 1 excellent)*                                               |

Embedded helper functions also return plain-English risk commentary for each statistic.

---

## Customisation & Extensibility

| Aspect               | How to Extend                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Languages**        | Add keys to `translations` dict; UI auto-detects.                                                                   |
| **Universe Lists**   | Edit `universe_options` for curated sectors.                                                                        |
| **Risk Models**      | Override `get_default_covariance()` or plug alternate estimators (e.g., EWMA).                                      |
| **Machine Learning** | Replace LSTM placeholder with your forecasting pipeline—`fetch_data()` already returns a cleaned returns DataFrame. |
| **Constraints**      | Adjust weight bounds or add short-selling, ESG screens in the optimiser classes.                                    |

---

## Troubleshooting

| Symptom                           | Fix                                                                                      |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| **“No data fetched.”**            | Check ticker spelling or Yahoo symbol availability. Crypto tickers must end with `-USD`. |
| **CVXPY “solver error”**          | Install a supported solver (`pip install ecos` or `osqp`).                               |
| **Streamlit cannot open browser** | Manually navigate to the printed local URL.                                              |
| **TensorFlow warnings**           | LSTM section is optional. Comment out TensorFlow import if unneeded.                     |

---

## Roadmap

* **Portfolio-to-portfolio comparison** UI (button placeholder present).
* **Live data refresh & scheduled re-optimisation** via `streamlit-autoreload` or background jobs.
* **Monte-Carlo stress tests** for scenario analysis.
* **Interactive correlation heat-map** in-app (currently in metrics table only).

---

## License

Distributed under the MIT License 
Copyright (c) 2025 Naveed Ahmed Maqbool

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---

*Happy optimising!*
