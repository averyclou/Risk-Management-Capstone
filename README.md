# Wolverine Risk Analytics Dashboard
## Monte Carlo Simulation + Heavy-Tailed Risk Modeling | MADS Capstone Project

## Overview
The Wolverine Risk Analytics Dashboard is an interactive financial-risk analysis environment developed as a capstone project for the University of Michigan Master of Applied Data Science (MADS) program. This application models the behavior of equity returns using Student’s-t heavy-tailed distributions, Monte Carlo simulation, rolling diagnostics, and path‑dependent risk analysis. The dashboard integrates statistical modeling, simulation, and visualization into a unified tool that helps analysts, students, and investors explore equity risk dynamics.

## Key Features
### Core Analytics
- Daily log-return modeling using a fitted Student’s-t distribution  
- Annualized performance metrics: mean return, volatility, Sharpe ratio  
- Risk‑adjusted metrics: Sortino ratio, Calmar ratio, downside deviation, Ulcer index  
- Rolling analytics: rolling Sharpe (63 & 126 days), rolling mean returns  
- Monthly behavior: monthly return time series and heatmap  

### Tail Risk & Distribution Diagnostics
- Value‑at‑Risk (VaR) and Conditional VaR (CVaR) computed from Student’s‑t  
- Distribution diagnostics: skewness, excess kurtosis, Jarque–Bera p‑value  
- Rolling VaR breaches  
- Overlay of empirical return histogram and fitted Student’s‑t PDF  

### Drawdown & Path‑Dependent Risk
- Full drawdown curve with max drawdown, current drawdown, and longest drawdown  
- Drawdown event analysis: depth vs duration  
- Recovery metrics: average and maximum recovery times  
- Ulcer index  

### Monte Carlo Simulation Engine
- Thousands of forward-looking simulated price paths  
- Configurable simulation horizon  
- Controls for visible paths vs total simulations  
- Metrics including median terminal price and approximate 95% CI width  

### Regime Modeling
- Volatility regimes (Low / Medium / High)  
- Return regimes (Up / Flat / Down)  
- Timeline heatmaps for both  

## Repository Structure
```
Risk-Management-Capstone/
├── wolverineRiskDashboardApp.py   # Main Streamlit application
├── umichRiskDashboardApp.py
├── app.py
├── dash_firstdraft.ipynb
├── final_dashboard.ipynb
├── monte_carlo.ipynb
├── umich_dash.ipynb
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository
```
git clone https://github.com/<your-username>/Risk-Management-Capstone.git
cd Risk-Management-Capstone
```

### 2. Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Configure your Alpha Vantage API key
```
export ALPHAVANTAGE_API_KEY="your_api_key_here"
```
Windows:
```
$env:ALPHAVANTAGE_API_KEY="your_api_key_here"
```

## Running the Dashboard
```
streamlit run wolverineRiskDashboardApp.py
```

## Visuals (Placeholders)
Add screenshots to `images/` folder.

![Dashboard Overview](images/dashboard_overview.png)

## App Versions
| Feature / Aspect                 | app.py | umichRiskDashboardApp.py | wolverineRiskDashboardApp.py |
|----------------------------------|--------|---------------------------|-------------------------------|
| Distribution Model               | Student’s-t | Student’s-t | Student’s-t |
| VaR / CVaR                       | Yes | Yes | Yes + rolling breaches |
| Monte Carlo Simulation           | Basic | Enhanced | Full with metrics |
| Rolling Volatility               | Yes | Yes | Yes |
| Rolling Sharpe                   | No | Yes | Yes (short + long) |
| Downside Deviation               | No | Yes | Yes |
| Calmar Ratio                     | No | Yes | Yes |
| Ulcer Index                      | No | Yes | Yes |
| Drawdown Analysis                | Basic | Basic | Full + event analysis |
| Monthly Returns                  | No | Yes | Yes + heatmap |
| Volatility Regimes               | Basic | Yes | Yes + timeline heatmap |
| Return Regimes                   | No | No | Yes |
| UI / Layout                      | Basic | UMich theme | Full analytical environment |

## Background & Related Work
Financial returns often exhibit heavy tails, skewness, and volatility clustering, diverging from Gaussian assumptions. Heavy‑tailed distributions such as the Student’s‑t more accurately capture these behaviors (Aas & Haff, 2006; Cont, 2001). Monte Carlo simulation provides a flexible framework for expressing uncertainty over future price paths, while drawdown‑based and regime‑based diagnostics help contextualize market risk beyond variance alone (Chekhlov, Uryasev, & Zabarankin, 2005; Hamilton, 1989).

### References
- Aas, K., & Haff, I. H. (2006). The generalized hyperbolic skew Student’s t‑distribution. *Journal of Financial Econometrics*.  
- Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). Drawdown measure in portfolio optimization.  
- Cont, R. (2001). Empirical properties of asset returns.  
- Hamilton, J. D. (1989). A new approach to economic analysis of nonstationary time series. *Econometrica*.

## Contributors
- Avery Cloutier — Simulation engine, dashboard development  
- Zara Masood — Metric selection, analytics guidance  
- Jeffrey Prachick — Initial dashboard integration and technical writing  

## License
MIT License
