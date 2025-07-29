#pylint: disable=all

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#Load and prepare data
spy = pd.read_csv('../data/SPY.csv')
gld = pd.read_csv('../data/GLD.csv')
spy_tr = pd.read_csv('../data/SP500TR.csv')

for df in [spy, gld, spy_tr]:
    df.columns = [col.capitalize() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

spy_weekly = spy['Close'].resample('W-FRI').last()
spy_tr_weekly = spy_tr['Close'].resample('W-FRI').last()
gld_weekly = gld['Close'].resample('W-FRI').last()

#ULTRA-ENHANCED FEATURE ENGINEERING
def create_ultra_features(spy_weekly, spy_tr_weekly, gld_weekly):
    #Basic returns
    spy_returns = np.log(spy_weekly / spy_weekly.shift(1)).to_frame(name='SPY_Return')
    spy_tr_returns = np.log(spy_tr_weekly / spy_tr_weekly.shift(1)).to_frame(name='SPY_TR_Return')
    gld_returns = np.log(gld_weekly / gld_weekly.shift(1)).to_frame(name='GLD_Return')
    
    combined = spy_returns.join(gld_returns, how='inner')
    combined = combined.join(spy_tr_returns, how='inner')
    
    #ENHANCED VOLATILITY FEATURES
    combined['Vol_2w'] = combined['SPY_Return'].rolling(window=2).std()
    combined['Vol_4w'] = combined['SPY_Return'].rolling(window=4).std()
    combined['Vol_8w'] = combined['SPY_Return'].rolling(window=8).std()
    combined['Vol_12w'] = combined['SPY_Return'].rolling(window=12).std()
    combined['Vol_26w'] = combined['SPY_Return'].rolling(window=26).std()
    
    #VOLATILITY REGIME INDICATORS
    combined['Vol_Regime_ST'] = combined['Vol_2w'] / combined['Vol_8w']  #Short-term
    combined['Vol_Regime_MT'] = combined['Vol_4w'] / combined['Vol_12w']  #Medium-term
    combined['Vol_Regime_LT'] = combined['Vol_8w'] / combined['Vol_26w']  #Long-term
    
    #VOLATILITY PERSISTENCE
    combined['Vol_Persistence'] = combined['Vol_4w'].rolling(window=4).std() / combined['Vol_4w']
    
    #ENHANCED CORRELATION FEATURES
    combined['SPY_GLD_Corr_4w'] = combined['SPY_Return'].rolling(window=4).corr(combined['GLD_Return'])
    combined['SPY_GLD_Corr_12w'] = combined['SPY_Return'].rolling(window=12).corr(combined['GLD_Return'])
    combined['SPY_GLD_Corr_26w'] = combined['SPY_Return'].rolling(window=26).corr(combined['GLD_Return'])
    
    #CORRELATION REGIME CHANGE
    combined['Corr_Change'] = combined['SPY_GLD_Corr_4w'] - combined['SPY_GLD_Corr_12w']
    
    #MOMENTUM SUITE
    combined['SPY_Mom_2w'] = combined['SPY_Return'].rolling(window=2).mean()
    combined['SPY_Mom_4w'] = combined['SPY_Return'].rolling(window=4).mean()
    combined['SPY_Mom_8w'] = combined['SPY_Return'].rolling(window=8).mean()
    combined['SPY_Mom_12w'] = combined['SPY_Return'].rolling(window=12).mean()
    combined['GLD_Mom_2w'] = combined['GLD_Return'].rolling(window=2).mean()
    combined['GLD_Mom_4w'] = combined['GLD_Return'].rolling(window=4).mean()
    combined['GLD_Mom_8w'] = combined['GLD_Return'].rolling(window=8).mean()
    combined['GLD_Mom_12w'] = combined['GLD_Return'].rolling(window=12).mean()
    
    #MOMENTUM CONVERGENCE/DIVERGENCE
    combined['SPY_MACD'] = combined['SPY_Mom_4w'] - combined['SPY_Mom_12w']
    combined['GLD_MACD'] = combined['GLD_Mom_4w'] - combined['GLD_Mom_12w']
    combined['Momentum_Spread'] = combined['SPY_MACD'] - combined['GLD_MACD']
    
    #TAIL RISK INDICATORS
    combined['SPY_Skew_4w'] = combined['SPY_Return'].rolling(window=4).skew()
    combined['SPY_Skew_12w'] = combined['SPY_Return'].rolling(window=12).skew()
    combined['SPY_Kurt_4w'] = combined['SPY_Return'].rolling(window=4).kurt()
    combined['SPY_Kurt_12w'] = combined['SPY_Return'].rolling(window=12).kurt()
    combined['GLD_Skew_4w'] = combined['GLD_Return'].rolling(window=4).skew()
    combined['GLD_Kurt_4w'] = combined['GLD_Return'].rolling(window=4).kurt()
    
    #DRAWDOWN INDICATORS
    spy_cumret = (1 + combined['SPY_Return']).cumprod()
    spy_peak = spy_cumret.cummax()
    combined['SPY_Drawdown'] = (spy_cumret - spy_peak) / spy_peak
    combined['SPY_Underwater'] = (combined['SPY_Drawdown'] < -0.05).astype(int)
    
    #VOLATILITY OF VOLATILITY (VIX PROXY)
    combined['VolOfVol_ST'] = combined['Vol_2w'].rolling(window=4).std()
    combined['VolOfVol_MT'] = combined['Vol_4w'].rolling(window=8).std()
    
    #RELATIVE PERFORMANCE INDICATORS
    combined['SPY_vs_GLD'] = combined['SPY_Return'] - combined['GLD_Return']
    combined['RelPerf_4w'] = combined['SPY_vs_GLD'].rolling(window=4).mean()
    combined['RelPerf_12w'] = combined['SPY_vs_GLD'].rolling(window=12).mean()
    combined['RelPerf_Trend'] = combined['RelPerf_4w'] - combined['RelPerf_12w']
    
    #TREND STRENGTH
    combined['SPY_Trend_4vs12'] = np.where(combined['SPY_Mom_4w'] > combined['SPY_Mom_12w'], 1, -1)
    combined['GLD_Trend_4vs12'] = np.where(combined['GLD_Mom_4w'] > combined['GLD_Mom_12w'], 1, -1)
    combined['Trend_Divergence'] = combined['SPY_Trend_4vs12'] - combined['GLD_Trend_4vs12']
    
    #MEAN REVERSION INDICATORS
    combined['SPY_ZScore'] = (combined['SPY_Return'] - combined['SPY_Return'].rolling(26).mean()) / combined['SPY_Return'].rolling(26).std()
    combined['GLD_ZScore'] = (combined['GLD_Return'] - combined['GLD_Return'].rolling(26).mean()) / combined['GLD_Return'].rolling(26).std()
    
    #REGIME PERSISTENCE INDICATOR
    combined['High_Vol_Regime'] = (combined['Vol_Regime_ST'] > 1.2).astype(int)
    combined['Regime_Persistence'] = combined['High_Vol_Regime'].rolling(window=4).mean()
    
    return combined

combined = create_ultra_features(spy_weekly, spy_tr_weekly, gld_weekly)
combined.dropna(inplace=True)

#ULTRA-ENHANCED STRATEGY
window_size = 156
n_components = 4
n_mix = 2

exception_count = 0
strategy_returns = []

#ADVANCED PORTFOLIO OPTIMIZATION WITH RISK PARITY
def ultra_optimize_portfolio(returns_spy, returns_gld, pi_next, state_analysis, market_conditions):
    """Ultra-advanced portfolio optimization with risk parity and multi-factor adjustment"""
    
    #Base allocation from HMM
    high_risk_state = state_analysis['high_risk_state']
    crisis_state = state_analysis.get('crisis_state', high_risk_state)
    
    #Multi-state probability weighted allocation
    base_gld_weight = 0.0
    for state in range(len(pi_next)):
        if state == high_risk_state:
            base_gld_weight += pi_next[state] * 0.8  #High defensive allocation
        elif state == crisis_state:
            base_gld_weight += pi_next[state] * 0.9  #Ultra defensive
        else:
            base_gld_weight += pi_next[state] * 0.2  #Normal allocation
    
    #VOLATILITY REGIME ADJUSTMENTS
    vol_regime_st = market_conditions['vol_regime_st']
    vol_regime_mt = market_conditions['vol_regime_mt']
    vol_persistence = market_conditions['vol_persistence']
    
    #Strong volatility regime signal
    if vol_regime_st > 1.5 and vol_regime_mt > 1.3:
        vol_adj = 0.3
    elif vol_regime_st > 1.2 and vol_regime_mt > 1.1:
        vol_adj = 0.2
    elif vol_regime_st < 0.8 and vol_regime_mt < 0.9:
        vol_adj = -0.2
    else:
        vol_adj = 0.0
    
    #Volatility persistence adjustment
    if vol_persistence > 0.5:
        vol_adj *= 1.2            # --> Amplify if volatile regime is persistent
    
    #CORRELATION REGIME ADJUSTMENT
    corr_4w = market_conditions['corr_4w']
    corr_change = market_conditions['corr_change']
    
    #Rising correlation = more defensive
    if corr_4w > 0.3 and corr_change > 0.1:
        corr_adj = 0.25
    elif corr_4w > 0.5:
        corr_adj = 0.15
    elif corr_4w < -0.2:
        corr_adj = -0.1     #Negative correlation = less defensive needed
    else:
        corr_adj = 0.0
    
    #MOMENTUM AND TREND ADJUSTMENT
    spy_momentum = market_conditions['spy_momentum']
    gld_momentum = market_conditions['gld_momentum']
    momentum_spread = market_conditions['momentum_spread']
    trend_divergence = market_conditions['trend_divergence']
    
    #Momentum-based adjustment
    if spy_momentum > 0.01 and gld_momentum < 0.005:
        mom_adj = -0.15                                     # --> SPY momentum strong, reduce GLD
    elif spy_momentum < -0.01 and gld_momentum > 0.005:
        mom_adj = 0.15                                      # --> SPY weak, increase GLD
    else:
        mom_adj = momentum_spread * -5                      # --> Gradual adjustment
    
    #Trend divergence adjustment
    if trend_divergence > 1:                                # --> SPY trending up, GLD down
        trend_adj = -0.1
    elif trend_divergence < -1:                             # --> SPY trending down, GLD up
        trend_adj = 0.1
    else:
        trend_adj = 0.0
    
    #TAIL RISK ADJUSTMENT
    spy_skew = market_conditions['spy_skew']
    spy_underwater = market_conditions['spy_underwater']
    
    #Negative skew = tail risk
    if spy_skew < -0.5:
        tail_adj = 0.15
    elif spy_skew < -1.0:
        tail_adj = 0.25
    else:
        tail_adj = 0.0
    
    #Drawdown adjustment
    if spy_underwater:
        dd_adj = 0.2
    else:
        dd_adj = 0.0
    
    #FINAL WEIGHT CALCULATION
    weight_gld = base_gld_weight + vol_adj + corr_adj + mom_adj + trend_adj + tail_adj + dd_adj
    
    #RISK PARITY CONSTRAINT
    spy_vol = returns_spy.std()
    gld_vol = returns_gld.std()
    
    if spy_vol > 0 and gld_vol > 0:
        #Risk parity weights
        rp_weight_spy = (1/spy_vol) / (1/spy_vol + 1/gld_vol)
        rp_weight_gld = (1/gld_vol) / (1/spy_vol + 1/gld_vol)
        
        #Blend with tactical allocation (70% tactical, 30% risk parity)
        weight_gld = 0.7 * weight_gld + 0.3 * rp_weight_gld
    
    #CONSTRAINTS WITH DYNAMIC BOUNDS
    if market_conditions['vol_regime_st'] > 1.5:  #High vol regime
        weight_gld = np.clip(weight_gld, 0.2, 0.9)
    elif market_conditions['vol_regime_st'] < 0.8:  #Low vol regime
        weight_gld = np.clip(weight_gld, 0.05, 0.6)
    else:
        weight_gld = np.clip(weight_gld, 0.1, 0.8)
    
    weight_spy = 1 - weight_gld
    
    return weight_spy, weight_gld, {
        'base_gld': base_gld_weight,
        'vol_adj': vol_adj,
        'corr_adj': corr_adj,
        'mom_adj': mom_adj,
        'tail_adj': tail_adj
    }

#ENHANCED FEATURE SELECTION
ultra_feature_cols = [
    'SPY_Return', 'Vol_Regime_ST', 'Vol_Regime_MT', 'Vol_Persistence',
    'SPY_GLD_Corr_4w', 'Corr_Change', 'SPY_MACD', 'GLD_MACD', 'Momentum_Spread',
    'SPY_Skew_4w', 'SPY_Kurt_4w', 'VolOfVol_ST', 'RelPerf_Trend',
    'SPY_Underwater', 'Regime_Persistence', 'SPY_ZScore'
]

#WALK FORWARD LOOP
for i in range(window_size, len(combined)):
    train_data = combined.iloc[i - window_size:i].copy()
    test_index = combined.index[i]
    
    #Features for HMM
    X_train = train_data[ultra_feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    #Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    try:
        #Ultra-enhanced HMM
        model = GMMHMM(
            n_components=n_components,
            n_mix=n_mix,
            covariance_type='diag',
            n_iter=300,
            tol=1e-6,
            random_state=42
        )
        model.fit(X_train_scaled)
        
        #Enhanced state analysis
        states = model.predict(X_train_scaled)
        train_data['State'] = states
        
        #Multi-criteria state analysis
        state_analysis = {}
        state_returns = train_data.groupby('State')['SPY_Return']
        state_vols = state_returns.std()
        state_means = state_returns.mean()
        state_sharpe = state_means / (state_vols + 1e-8)
        state_skew = state_returns.skew()
        
        #Identify high-risk and crisis states
        high_risk_state = state_sharpe.idxmin()
        crisis_candidates = state_means[state_means < state_means.quantile(0.3)]
        if len(crisis_candidates) > 0:
            crisis_state = crisis_candidates.idxmin()
        else:
            crisis_state = high_risk_state
        
        state_analysis['high_risk_state'] = high_risk_state
        state_analysis['crisis_state'] = crisis_state
        
        #Transition probabilities
        posteriors = model.predict_proba(X_train_scaled)
        pi_t = posteriors[-1]
        pi_next = pi_t @ model.transmat_
        
        #Current market conditions
        market_conditions = {
            'vol_regime_st': combined.loc[test_index, 'Vol_Regime_ST'],
            'vol_regime_mt': combined.loc[test_index, 'Vol_Regime_MT'],
            'vol_persistence': combined.loc[test_index, 'Vol_Persistence'],
            'corr_4w': combined.loc[test_index, 'SPY_GLD_Corr_4w'],
            'corr_change': combined.loc[test_index, 'Corr_Change'],
            'spy_momentum': combined.loc[test_index, 'SPY_MACD'],
            'gld_momentum': combined.loc[test_index, 'GLD_MACD'],
            'momentum_spread': combined.loc[test_index, 'Momentum_Spread'],
            'trend_divergence': combined.loc[test_index, 'Trend_Divergence'],
            'spy_skew': combined.loc[test_index, 'SPY_Skew_4w'],
            'spy_underwater': combined.loc[test_index, 'SPY_Underwater']
        }
        
        #Ultra portfolio optimization
        weight_spy, weight_gld, decomp = ultra_optimize_portfolio(
            train_data['SPY_Return'], 
            train_data['GLD_Return'],
            pi_next,
            state_analysis,
            market_conditions
        )
        
    except Exception as e:
        #Enhanced fallback strategy
        recent_vol = combined.loc[combined.index[i-4:i], 'Vol_4w'].mean()
        historical_vol = combined.loc[combined.index[i-26:i-4], 'Vol_4w'].mean()
        
        if recent_vol > 1.5 * historical_vol:    #High vol regime
            weight_spy, weight_gld = 0.3, 0.7
        elif recent_vol < 0.7 * historical_vol:  #Low vol regime
            weight_spy, weight_gld = 0.8, 0.2
        else:                                    #Normal regime with momentum
            recent_spy_perf = combined.loc[combined.index[i-4:i], 'SPY_Return'].mean()
            recent_gld_perf = combined.loc[combined.index[i-4:i], 'GLD_Return'].mean()
            
            if recent_spy_perf > recent_gld_perf:
                weight_spy, weight_gld = 0.7, 0.3
            else:
                weight_spy, weight_gld = 0.4, 0.6
                
        exception_count += 1
    
    #Calculate strategy return
    r_spy = combined.loc[test_index, 'SPY_Return']
    r_gld = combined.loc[test_index, 'GLD_Return']
    
    #Enhanced return calculation with rebalancing cost
    strategy_return = np.log(weight_spy * np.exp(r_spy) + weight_gld * np.exp(r_gld))
    
    #Add small transaction cost for rebalancing
    if i > window_size:
        prev_weight_gld = strategy_returns[-1][4]
        rebalancing_cost = 0.001 * abs(weight_gld - prev_weight_gld)
        strategy_return -= rebalancing_cost
    
    strategy_returns.append((test_index, strategy_return, r_spy, r_gld, weight_gld, weight_spy))

#RESULTS ANALYSIS
strategy_df = pd.DataFrame(strategy_returns, 
                          columns=['Date', 'Strategy_Return', 'SPY_Return', 'GLD_Return', 'GLD_Weight', 'SPY_Weight'])
strategy_df.set_index('Date', inplace=True)

#CUMULATIVE RETURNS
strategy_df['Cumulative_Strategy'] = (1 + strategy_df['Strategy_Return']).cumprod()
strategy_df['Cumulative_SPY'] = (1 + strategy_df['SPY_Return']).cumprod()
strategy_df['Cumulative_GLD'] = (1 + strategy_df['GLD_Return']).cumprod()

#Benchmarks
strategy_df['Mixed_Return'] = np.log(0.5 * np.exp(strategy_df['SPY_Return']) + 0.5 * np.exp(strategy_df['GLD_Return']))
strategy_df['Cumulative_Mixed'] = (1 + strategy_df['Mixed_Return']).cumprod()

strategy_df['Tactical_Return'] = np.log(0.65 * np.exp(strategy_df['SPY_Return']) + 0.35 * np.exp(strategy_df['GLD_Return']))
strategy_df['Cumulative_Tactical'] = (1 + strategy_df['Tactical_Return']).cumprod()

#Risk parity benchmark
spy_vol = strategy_df['SPY_Return'].std()
gld_vol = strategy_df['GLD_Return'].std()
rp_weight_spy = (1/spy_vol) / (1/spy_vol + 1/gld_vol)
rp_weight_gld = 1 - rp_weight_spy

strategy_df['RiskParity_Return'] = np.log(rp_weight_spy * np.exp(strategy_df['SPY_Return']) + rp_weight_gld * np.exp(strategy_df['GLD_Return']))
strategy_df['Cumulative_RiskParity'] = (1 + strategy_df['RiskParity_Return']).cumprod()

#PLOTTING
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

#Cumulative returns
strategy_df[['Cumulative_Strategy', 'Cumulative_SPY', 'Cumulative_GLD', 'Cumulative_Mixed', 'Cumulative_Tactical', 'Cumulative_RiskParity']].plot(ax=axes[0])
axes[0].set_title("Ultra-Enhanced HMM Strategy: Multi-Factor Risk Allocation")
axes[0].set_ylabel("Cumulative Return")
axes[0].grid(True)
axes[0].legend()

#Allocation over time
strategy_df[['SPY_Weight', 'GLD_Weight']].plot(ax=axes[1], kind='area', alpha=0.7)
axes[1].set_title("Dynamic Asset Allocation")
axes[1].set_ylabel("Weight")
axes[1].grid(True)

#Rolling Sharpe ratio
rolling_sharpe = strategy_df['Strategy_Return'].rolling(52).mean() / strategy_df['Strategy_Return'].rolling(52).std() * np.sqrt(52)
spy_rolling_sharpe = strategy_df['SPY_Return'].rolling(52).mean() / strategy_df['SPY_Return'].rolling(52).std() * np.sqrt(52)

pd.DataFrame({'Strategy': rolling_sharpe, 'SPY': spy_rolling_sharpe}).plot(ax=axes[2])
axes[2].set_title("Rolling 1-Year Sharpe Ratio")
axes[2].set_ylabel("Sharpe Ratio")
axes[2].grid(True)

plt.tight_layout()
plt.show()

#ADVANCED STATISTICS
def compute_hedge_fund_stats(returns, freq='W'):
    periods_per_year = {'D': 252, 'W': 52, 'M': 12}[freq]
    
    #Basic metrics
    cumulative_return = (1 + returns).prod()
    n_periods = len(returns)
    years = n_periods / periods_per_year
    cagr = cumulative_return ** (1 / years) - 1
    
    #Drawdown analysis
    rolling = (1 + returns).cumprod()
    peak = rolling.cummax()
    drawdown = (rolling - peak) / peak
    max_dd = drawdown.min()
    avg_dd = drawdown[drawdown < 0].mean()
    
    #Recovery time
    underwater = drawdown < -0.05
    if underwater.any():
        recovery_periods = []
        in_drawdown = False
        start_dd = None
        for i, is_dd in enumerate(underwater):
            if is_dd and not in_drawdown:
                in_drawdown = True
                start_dd = i
            elif not is_dd and in_drawdown:
                in_drawdown = False
                recovery_periods.append(i - start_dd)
        avg_recovery = np.mean(recovery_periods) if recovery_periods else 0
    else:
        avg_recovery = 0
    
    #Risk metrics
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    #Downside metrics
    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = returns.mean() / downside_dev * np.sqrt(periods_per_year) if len(downside_returns) > 0 else np.nan
    
    #Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    
    #Tail metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    #Ulcer Index
    ulcer_index = np.sqrt((drawdown ** 2).mean())
    
    #Information ratio (vs zero)
    info_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    return {
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_dd,
        'Average Drawdown': avg_dd,
        'Ulcer Index': ulcer_index,
        'Average Recovery (weeks)': avg_recovery,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95,
        'Win Rate': (returns > 0).mean(),
        'Information Ratio': info_ratio
    }

#Calculate statistics
stats_strategy = compute_hedge_fund_stats(strategy_df['Strategy_Return'], freq='W')
stats_spy = compute_hedge_fund_stats(strategy_df['SPY_Return'], freq='W')
stats_gld = compute_hedge_fund_stats(strategy_df['GLD_Return'], freq='W')
stats_mixed = compute_hedge_fund_stats(strategy_df['Mixed_Return'], freq='W')
stats_tactical = compute_hedge_fund_stats(strategy_df['Tactical_Return'], freq='W')
stats_rp = compute_hedge_fund_stats(strategy_df['RiskParity_Return'], freq='W')

report = pd.DataFrame(
    [stats_strategy, stats_spy, stats_gld, stats_mixed, stats_tactical, stats_rp],
    index=['Ultra Strategy', 'SPY', 'GLD', '50/50', 'Tactical 65/35', 'Risk Parity']
)

print("=== ULTRA-ENHANCED HEDGE FUND STRATEGY REPORT ===")
print(report.round(4))
print(f"\nModel exceptions: {exception_count}")
print(f"Exception rate: {exception_count/len(strategy_returns):.2%}")
print(f"Average GLD allocation: {strategy_df['GLD_Weight'].mean():.2%}")
print(f"GLD allocation std: {strategy_df['GLD_Weight'].std():.2%}")
print(f"GLD allocation range: {strategy_df['GLD_Weight'].min():.2%} - {strategy_df['GLD_Weight'].max():.2%}")

#CRISIS PERFORMANCE ANALYSIS
print("\n=== CRISIS PERFORMANCE ANALYSIS ===")
crisis_periods = {
    '2008 Financial Crisis': ('2008-01-01', '2009-03-31'),
    'COVID-19 Crash': ('2020-01-01', '2020-05-31'),
    'Recent Volatility': ('2022-01-01', '2022-12-31')
}

for crisis_name, (start, end) in crisis_periods.items():
    mask = (strategy_df.index >= start) & (strategy_df.index <= end)
    if mask.any():
        crisis_strategy = strategy_df.loc[mask, 'Strategy_Return'].mean() * 52
        crisis_spy = strategy_df.loc[mask, 'SPY_Return'].mean() * 52
        crisis_gld_weight = strategy_df.loc[mask, 'GLD_Weight'].mean()
        
        print(f"{crisis_name}:")
        print(f"  Strategy return: {crisis_strategy:.2%}")
        print(f"  SPY return: {crisis_spy:.2%}")
        print(f"  Outperformance: {crisis_strategy - crisis_spy:.2%}")
        print(f"  Average GLD allocation: {crisis_gld_weight:.2%}")
        print()

#REGIME DETECTION ANALYSIS
print("\n=== REGIME DETECTION PERFORMANCE ===")
strategy_df['Vol_Regime'] = combined.loc[strategy_df.index, 'Vol_Regime_ST']
strategy_df['High_Vol'] = strategy_df['Vol_Regime'] > 1.2

high_vol_periods = strategy_df[strategy_df['High_Vol']]
low_vol_periods = strategy_df[~strategy_df['High_Vol']]

if len(high_vol_periods) > 0:
    print("High Volatility Periods:")
    print(f"  Strategy CAGR: {high_vol_periods['Strategy_Return'].mean() * 52:.2%}")
    print(f"  SPY CAGR: {high_vol_periods['SPY_Return'].mean() * 52:.2%}")
    print(f"  Average GLD allocation: {high_vol_periods['GLD_Weight'].mean():.2%}")
    print(f"  Periods: {len(high_vol_periods)}")

if len(low_vol_periods) > 0:
    print("\nLow Volatility Periods:")
    print(f"  Strategy CAGR: {low_vol_periods['Strategy_Return'].mean() * 52:.2%}")
    print(f"  SPY CAGR: {low_vol_periods['SPY_Return'].mean() * 52:.2%}")
    print(f"  Average GLD allocation: {low_vol_periods['GLD_Weight'].mean():.2%}")
    print(f"  Periods: {len(low_vol_periods)}")

#REBALANCING ANALYSIS
print("\n=== REBALANCING ANALYSIS ===")
strategy_df['GLD_Weight_Change'] = strategy_df['GLD_Weight'].diff().abs()
strategy_df['Rebalancing_Cost'] = strategy_df['GLD_Weight_Change'] * 0.001

total_rebalancing_cost = strategy_df['Rebalancing_Cost'].sum()
avg_rebalancing_cost = strategy_df['Rebalancing_Cost'].mean()
turnover_rate = strategy_df['GLD_Weight_Change'].mean()

print(f"Total rebalancing cost: {total_rebalancing_cost:.4f}")
print(f"Average weekly rebalancing cost: {avg_rebalancing_cost:.4f}")
print(f"Average turnover rate: {turnover_rate:.2%}")
print(f"Annualized turnover: {turnover_rate * 52:.2%}")

#FACTOR ATTRIBUTION ANALYSIS
print("\n=== FACTOR ATTRIBUTION ANALYSIS ===")
#Decompose returns into factors
strategy_df['Beta_SPY'] = strategy_df['SPY_Weight']
strategy_df['Beta_GLD'] = strategy_df['GLD_Weight']

#Rolling beta analysis
rolling_window = 26
strategy_df['Rolling_Beta_SPY'] = strategy_df['Beta_SPY'].rolling(rolling_window).mean()
strategy_df['Rolling_Beta_GLD'] = strategy_df['Beta_GLD'].rolling(rolling_window).mean()

#Calculate factor contributions
strategy_df['SPY_Contribution'] = strategy_df['SPY_Weight'] * strategy_df['SPY_Return']
strategy_df['GLD_Contribution'] = strategy_df['GLD_Weight'] * strategy_df['GLD_Return']

spy_contribution = strategy_df['SPY_Contribution'].sum()
gld_contribution = strategy_df['GLD_Contribution'].sum()
total_contribution = spy_contribution + gld_contribution

print(f"SPY factor contribution: {spy_contribution:.4f} ({spy_contribution/total_contribution:.1%})")
print(f"GLD factor contribution: {gld_contribution:.4f} ({gld_contribution/total_contribution:.1%})")

#MARKET TIMING ANALYSIS
print("\n=== MARKET TIMING ANALYSIS ===")
#Timing skill measurement
strategy_df['Market_Direction'] = np.where(strategy_df['SPY_Return'] > 0, 1, -1)
strategy_df['Position_Direction'] = np.where(strategy_df['SPY_Weight'] > 0.5, 1, -1)
strategy_df['Timing_Correct'] = strategy_df['Market_Direction'] == strategy_df['Position_Direction']

timing_accuracy = strategy_df['Timing_Correct'].mean()
print(f"Market timing accuracy: {timing_accuracy:.2%}")

#Calculate timing value (vs static allocation)
static_return = 0.6 * strategy_df['SPY_Return'] + 0.4 * strategy_df['GLD_Return']
timing_value = strategy_df['Strategy_Return'] - static_return
timing_value_annual = timing_value.mean() * 52

print(f"Timing value (annualized): {timing_value_annual:.2%}")

#RISK BUDGETING ANALYSIS
print("\n=== RISK BUDGETING ANALYSIS ===")
#Calculate risk contributions
spy_vol = strategy_df['SPY_Return'].std()
gld_vol = strategy_df['GLD_Return'].std()
corr_spy_gld = strategy_df['SPY_Return'].corr(strategy_df['GLD_Return'])

#Portfolio volatility components
avg_spy_weight = strategy_df['SPY_Weight'].mean()
avg_gld_weight = strategy_df['GLD_Weight'].mean()

spy_risk_contrib = (avg_spy_weight**2 * spy_vol**2) / (avg_spy_weight**2 * spy_vol**2 + avg_gld_weight**2 * gld_vol**2 + 2*avg_spy_weight*avg_gld_weight*corr_spy_gld*spy_vol*gld_vol)
gld_risk_contrib = (avg_gld_weight**2 * gld_vol**2) / (avg_spy_weight**2 * spy_vol**2 + avg_gld_weight**2 * gld_vol**2 + 2*avg_spy_weight*avg_gld_weight*corr_spy_gld*spy_vol*gld_vol)

print(f"SPY risk contribution: {spy_risk_contrib:.1%}")
print(f"GLD risk contribution: {gld_risk_contrib:.1%}")
print(f"SPY weight: {avg_spy_weight:.1%}")
print(f"GLD weight: {avg_gld_weight:.1%}")

#ADVANCED OPTIMIZATION SUGGESTIONS
print("\n=== OPTIMIZATION SUGGESTIONS ===")
print("1. MODEL STABILITY:")
print(f"   - Exception rate: {exception_count/len(strategy_returns):.1%} (target: <5%)")
print("   - Consider ensemble of models for robustness")
print("   - Add model confidence scoring")

print("\n2. RISK MANAGEMENT:")
print(f"   - Current max drawdown: {stats_strategy['Max Drawdown']:.1%}")
print("   - Consider volatility targeting")
print("   - Add position sizing based on Kelly criterion")

print("\n3. TRANSACTION COSTS:")
print(f"   - Current turnover: {turnover_rate * 52:.1%}")
print("   - Consider rebalancing thresholds")
print("   - Optimize rebalancing frequency")

print("\n4. FEATURE ENGINEERING:")
print("   - Add macro indicators (VIX, yield curve)")
print("   - Include sentiment indicators")
print("   - Consider alternative risk assets")

#ENHANCED BACKTESTING METRICS
print("\n=== ENHANCED BACKTESTING VALIDATION ===")

#Out-of-sample periods
split_date = '2018-01-01'
is_mask = strategy_df.index < split_date
oos_mask = strategy_df.index >= split_date

if is_mask.any() and oos_mask.any():
    is_performance = compute_hedge_fund_stats(strategy_df.loc[is_mask, 'Strategy_Return'], freq='W')
    oos_performance = compute_hedge_fund_stats(strategy_df.loc[oos_mask, 'Strategy_Return'], freq='W')
    
    print(f"In-sample CAGR: {is_performance['CAGR']:.2%}")
    print(f"Out-of-sample CAGR: {oos_performance['CAGR']:.2%}")
    print(f"Performance consistency: {abs(is_performance['CAGR'] - oos_performance['CAGR']):.2%}")

#FINAL RECOMMENDATIONS
print("\n=== IMPLEMENTATION RECOMMENDATIONS ===")
print("IMMEDIATE IMPROVEMENTS:")
print("   1. Reduce model exceptions with better preprocessing")
print("   2. Add confidence intervals for allocations")
print("   3. Implement dynamic rebalancing thresholds")
print("   4. Add macro regime overlay")

print("ADVANCED ENHANCEMENTS:")
print("   1. Multi-timeframe HMM (daily + weekly)")
print("   2. Ensemble of different models")
print("   3. Options overlay for downside protection")
print("   4. Alternative risk assets (commodities, crypto)")

print("MONITORING DASHBOARD:")
print("   1. Real-time regime probabilities")
print("   2. Risk attribution breakdown")
print("   3. Performance attribution")
print("   4. Model confidence scoring")

#SAVE RESULTS
strategy_df.to_csv('ultra_enhanced_hmm_results.csv')
report.to_csv('ultra_enhanced_hmm_performance.csv')

print(f"Results saved to CSV files")
print(f"Final Performance: {stats_strategy['CAGR']:.2%} CAGR, {stats_strategy['Sharpe Ratio']:.2f} Sharpe")
print(f"Alpha vs SPY: {stats_strategy['CAGR'] - stats_spy['CAGR']:.2%}")
print(f"Max DD: {stats_strategy['Max Drawdown']:.1%} vs SPY {stats_spy['Max Drawdown']:.1%}")