# Probabilistic Portfolio Strategy Using Hidden Markov Models

## Overview

This project implements a **Hidden Markov Model (HMM)**-based asset allocation strategy over a 20-year horizon (2005–2025).  
The goal is to dynamically allocate capital between **SPY** (U.S. equities) and **GLD** (gold) based on **probabilistic regime detection**.

Two variants are implemented:

-  **Optimized Strategy**: focuses on return maximization using pure HMM inference  
-  **Conservative Strategy**: applies volatility and drawdown filters for robust capital preservation  

Both strategies are trained and validated using a **walk-forward out-of-sample framework**.

---

##  Key Results

| Strategy     | CAGR    | Max Drawdown | Sharpe Ratio |
|--------------|---------|--------------|---------------|
| Optimized    | 11.69%  | -31.03%      | 0.76          |
| Conservative | 10.68%  | -22.88%      | 0.74          |
| SPY TR       | 8.66%   | -58.16%      | 0.54          |

>  Both strategies consistently outperform SPY and SPY Total Return on a risk-adjusted basis.

---

##  Methodology

- Gaussian Mixture Hidden Markov Models (GMMHMM)
- Weekly walk-forward retraining over a 104-week window
- Dynamic risk-based allocation using state Sharpe ratios
- Fallback and override logic (soft and hard filters in conservative variant)