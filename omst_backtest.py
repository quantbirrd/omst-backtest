import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# 1. DATA ACQUISITION & PARAMETERS
# Following Section 7.2: Target = NEM, Sector = GDX, Macro Driver = GLD
target_ticker = "NEM"
sector_ticker = "GDX"
macro_ticker  = "GLD"

start_date = "2023-01-01"
end_date   = "2026-05-25"  # Current Date

print("Downloading historical market data...")
data = yf.download([target_ticker, sector_ticker, macro_ticker], start=start_date, end=end_date)

# Clean multi-index columns from yfinance
close = data['Close']
volume = data['Volume']

df = pd.DataFrame({
    'target_close': close[target_ticker],
    'target_vol': volume[target_ticker],
    'sector_close': close[sector_ticker],
    'macro_close': close[macro_ticker]
}).dropna()

# Calculate daily log returns
df['target_ret'] = np.log(df['target_close'] / df['target_close'].shift(1))
df['sector_ret'] = np.log(df['sector_close'] / df['sector_close'].shift(1))
df = df.dropna()

# Parameters
k = 20         # Fractal Periodicity window (roughly 1 trading month)
W = 60         # Macro Normalization window (roughly 3 trading months)
w1 = 0.5       # Weight for Smoothness
w2 = 0.5       # Weight for Liquidity

# Initialize output columns
df['OMSE_raw'] = np.nan
df['FIX'] = np.nan
df['C_t'] = np.nan
df['PS_prior'] = np.nan
df['OMSE_posterior'] = np.nan


# 2. RUNNING THE OMST MATHEMATICAL BLENDER
print("Running OMST engine...")

# Step A: Pre-calculate Amihud Illiquidity (Eq 10)
# Amihud Ratio = |Return| / Dollar Volume
df['amihud'] = np.abs(df['target_ret']) / (df['target_vol'] * df['target_close'])

# Step B: Main Loop for Fractal Lookbacks
for t in range(max(k, W), len(df)):

    # Pillar 1: Directional Smoothness (Eq 1)
    price_window = df['target_close'].iloc[t-k:t].values
    # Normalize window to start at 1.0 for scale-invariance
    p_norm = price_window / price_window[0]
    
    # Calculate discrete arc length components
    dx = 1.0 / k
    dy = np.diff(p_norm)
    
    actual_path = np.sum(np.sqrt(dx**2 + dy**2))
    straight_line = np.sqrt(1.0**2 + (p_norm[-1] - 1.0)**2)
    
    # Smoothness ratio S_t
    S_t = straight_line / actual_path if actual_path > 0 else 0
    

    # Pillar 2: Relative Liquidity (Eq 10)
    # Z-score normalization of Amihud over window W
    rolling_amihud = df['amihud'].iloc[t-W:t]
    mean_ami = rolling_amihud.mean()
    std_ami = rolling_amihud.std() if rolling_amihud.std() > 0 else 1e-8
    
    Z_t = (df['amihud'].iloc[t] - mean_ami) / std_ami
    
    # Map Z-score to [0,1] using Normal CDF. 
    # High Z (high illiquidity) -> Low Liquidity L_t
    L_t = norm.cdf(-Z_t)
    
    # Pillar 3: Blended Raw OMSE (Eq 2)
    
    OMSE_raw = (w1 * S_t + w2 * L_t)
    df.iloc[t, df.columns.get_loc('OMSE_raw')] = OMSE_raw
    
    
    # Pillar 4: Ecosystem Convergence (Eq 11)
    
    ret_target = df['target_ret'].iloc[t-W:t]
    ret_sector = df['sector_ret'].iloc[t-W:t]
    C_t = ret_target.corr(ret_sector)
    df.iloc[t, df.columns.get_loc('C_t')] = C_t

# 3. BAYESIAN UPDATE & PREDISPOSITION STATE

# Transition probabilities for the prior (Eq 5)
p11 = 0.80  # Probability of remaining in Optimal State
p01 = 0.15  # Probability of jumping to Optimal State

# We initialize yesterday's posterior at 50%
posterior_t_minus_1 = 0.50

for t in range(max(k, W), len(df)):
    OMSE_raw = df['OMSE_raw'].iloc[t]
    if np.isnan(OMSE_raw):
        continue
        
    # Calculate Predispositioned State Prior (Eq 5)
    PS_prior = p11 * posterior_t_minus_1 + p01 * (1.0 - posterior_t_minus_1)
    df.iloc[t, df.columns.get_loc('PS_prior')] = PS_prior
    
    # Likelihood functions (Eq 6) modeled via normal distributions
    # Optimal State favors high OMSE (mean=0.80); Chaotic favors low (mean=0.45)
    L_optimal = norm.pdf(OMSE_raw, loc=0.80, scale=0.15)
    L_chaotic = norm.pdf(OMSE_raw, loc=0.45, scale=0.25)
    
    # Posterior calculation (Eq 6)
    numerator = L_optimal * PS_prior
    denominator = (L_optimal * PS_prior) + (L_chaotic * (1.0 - PS_prior))
    
    posterior_t = numerator / denominator if denominator > 0 else 0.5
    df.iloc[t, df.columns.get_loc('OMSE_posterior')] = posterior_t
    
    # Update state for next period
    posterior_t_minus_1 = posterior_t

# 4. FLUCTUATION INDEX (FIX) (Eq 7 & 8)

# Calculate FIX dynamically over a rolling window W
for t in range(max(k, W) + W, len(df)):
    rolling_posterior = df['OMSE_posterior'].iloc[t-W:t].dropna()
    if len(rolling_posterior) < W:
        continue
        
    mu_omse = rolling_posterior.mean()
    sigma_omse = rolling_posterior.std()
    
    q1 = rolling_posterior.quantile(0.25)
    q3 = rolling_posterior.quantile(0.75)
    iqr_omse = q3 - q1
    
    # Eq 7
    denom = sigma_omse * iqr_omse
    FIX_score = mu_omse / denom if denom > 0 else 0.0
    df.iloc[t, df.columns.get_loc('FIX')] = FIX_score

# Drop initial empty rows for analysis
df_clean = df.dropna().copy()

# 5. VISUALIZATION & ANALYSIS

print("Generating analysis charts...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel 1: Target Price Chart
ax1.plot(df_clean.index, df_clean['target_close'], color='black', label=f'{target_ticker} Close Price')
ax1.set_title(f"Optimal Market State Theory (OMST) - Analytical Backtest ({target_ticker})", fontsize=14, fontweight='bold')
ax1.set_ylabel("Price ($)")
ax1.legend(loc='upper left')
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 2: OMSE (Raw vs Bayesian Posterior)
ax2.plot(df_clean.index, df_clean['OMSE_raw'] * 100, color='gray', alpha=0.5, label='Raw Blended OMSE %')
ax2.plot(df_clean.index, df_clean['OMSE_posterior'] * 100, color='blue', label='Bayesian Posterior State Probability (PS-Filtered)')
ax2.axhline(85, color='green', linestyle=':', label='Optimal Boundary (85%)')
ax2.set_ylabel("Probability / Score (%)")
ax2.legend(loc='upper left')
ax2.grid(True, linestyle="--", alpha=0.5)

# Panel 3: Fluctuation Index (FIX)
ax3.plot(df_clean.index, df_clean['FIX'], color='purple', label='Fluctuation Index (FIX)')
ax3.axhline(10.0, color='green', linestyle='--', alpha=0.7, label='Sustained Stability Limit (FIX = 10.0)')
ax3.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Chaos Boundary (FIX = 1.0)')
ax3.set_ylabel("FIX Index Score")
ax3.set_xlabel("Date")
ax3.legend(loc='upper left')
ax3.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print("\n--- Backtest Summary ---")
print(f"Total Trading Days Analyzed: {len(df_clean)}")
print(f"Average Ecosystem Convergence (C_t): {df_clean['C_t'].mean():.2f}")
print(f"Sustained Stability Days (FIX >= 10): {len(df_clean[df_clean['FIX'] >= 10.0])}")
print(f"Highly Volatile Days (FIX < 1): {len(df_clean[df_clean['FIX'] < 1.0])}")