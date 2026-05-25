import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# 1. DATA ACQUISITION & PARAMETERS
# ==========================================
# Testing the S&P 500 (SPY) Control Asset
target_ticker = "SPY"
sector_ticker = "XLK"  # Technology Sector ETF
macro_ticker  = "SPY"

# We use a longer history to accommodate the 252-day baseline
start_date = "2022-01-01"
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
k = 20           # Fractal Periodicity window (roughly 1 trading month)
W = 60           # Fluctuation Index (FIX) evaluation window
baseline_W = 252 # --- THE COUPLING FIX --- 1 trading year baseline for Z-score normalization
w1 = 0.5         # Weight for Smoothness
w2 = 0.5         # Weight for Liquidity

# Initialize output columns
df['OMSE_raw'] = np.nan
df['FIX'] = np.nan
df['C_t'] = np.nan
df['PS_prior'] = np.nan
df['OMSE_posterior'] = np.nan

# ==========================================
# 2. RUNNING THE OMST MATHEMATICAL BLENDER
# ==========================================
print("Running OMST engine...")

# Step A: Pre-calculate Amihud Illiquidity (Eq 10)
df['amihud'] = np.abs(df['target_ret']) / (df['target_vol'] * df['target_close'])

# Smooth Amihud over k-period to filter high-frequency noise
df['amihud_smooth'] = df['amihud'].rolling(window=k).mean()

# Step B: Main Loop for Fractal Lookbacks (Starts after baseline_W to allow Z-score calculation)
for t in range(baseline_W, len(df)):

    # --------------------------------------
    # Pillar 1: Directional Smoothness (Eq 1)
    # --------------------------------------
    price_window = df['target_close'].iloc[t-k:t].values
    p_norm = price_window / price_window[0]
    
    # Calculate discrete arc length components
    dy = np.diff(p_norm)
    dx = 1.0 / len(dy)  # Standardizes total time step to 1.0
    
    actual_path = np.sum(np.sqrt(dx**2 + dy**2))
    straight_line = np.sqrt(1.0**2 + (p_norm[-1] - 1.0)**2)
    
    # Smoothness ratio S_t
    S_t = straight_line / actual_path if actual_path > 0 else 0
    
    # --------------------------------------
    # Pillar 2: Relative Liquidity (Eq 10)
    # --------------------------------------
    # Z-score normalization of Amihud over the larger 252-day baseline_W
    rolling_amihud = df['amihud_smooth'].iloc[t-baseline_W:t]
    mean_ami = rolling_amihud.mean()
    std_ami = rolling_amihud.std() if rolling_amihud.std() > 0 else 1e-8
    
    Z_t = (df['amihud_smooth'].iloc[t] - mean_ami) / std_ami
    
    # Map Z-score to [0,1] using Normal CDF.
    L_t = norm.cdf(-Z_t)
    
    # --------------------------------------
    # Pillar 3: Blended Raw OMSE (Eq 2)
    # --------------------------------------
    OMSE_raw = (w1 * S_t + w2 * L_t)
    df.iloc[t, df.columns.get_loc('OMSE_raw')] = OMSE_raw
    
    # --------------------------------------
    # Pillar 4: Ecosystem Convergence (Eq 11)
    # --------------------------------------
    ret_target = df['target_ret'].iloc[t-W:t]
    ret_sector = df['sector_ret'].iloc[t-W:t]
    C_t = ret_target.corr(ret_sector)
    df.iloc[t, df.columns.get_loc('C_t')] = C_t

# ==========================================
# 3. BAYESIAN UPDATE & PREDISPOSITION STATE
# ==========================================
p11 = 0.80  # Probability of remaining in Optimal State
p01 = 0.15  # Probability of jumping to Optimal State
posterior_t_minus_1 = 0.50

for t in range(baseline_W, len(df)):
    OMSE_raw = df['OMSE_raw'].iloc[t]
    if np.isnan(OMSE_raw):
        continue
        
    PS_prior = p11 * posterior_t_minus_1 + p01 * (1.0 - posterior_t_minus_1)
    df.iloc[t, df.columns.get_loc('PS_prior')] = PS_prior
    
    L_optimal = norm.pdf(OMSE_raw, loc=0.80, scale=0.15)
    L_chaotic = norm.pdf(OMSE_raw, loc=0.45, scale=0.25)
    
    numerator = L_optimal * PS_prior
    denominator = (L_optimal * PS_prior) + (L_chaotic * (1.0 - PS_prior))
    
    posterior_t = numerator / denominator if denominator > 0 else 0.5
    df.iloc[t, df.columns.get_loc('OMSE_posterior')] = posterior_t
    posterior_t_minus_1 = posterior_t

# ==========================================
# 4. FLUCTUATION INDEX (FIX) (Eq 7 & 8)
# ==========================================
# Calculate FIX dynamically over a rolling window W
for t in range(baseline_W + W, len(df)):
    rolling_omse = df['OMSE_raw'].iloc[t-W:t].dropna()
    if len(rolling_omse) < W:
        continue
    
    # Scale decimal OMSE to percentage space (0 - 100)
    rolling_omse_pct = rolling_omse * 100.0
        
    mu_omse = rolling_omse_pct.mean()
    sigma_omse = rolling_omse_pct.std()
    
    q1 = rolling_omse_pct.quantile(0.25)
    q3 = rolling_omse_pct.quantile(0.75)
    iqr_omse = q3 - q1
    
    denom = sigma_omse * iqr_omse
    FIX_score = mu_omse / denom if denom > 0 else 0.0
    df.iloc[t, df.columns.get_loc('FIX')] = FIX_score

# ==========================================
# 5. VISUALIZATION & ANALYSIS
# ==========================================
df_clean = df.dropna(subset=['OMSE_posterior', 'FIX'])

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