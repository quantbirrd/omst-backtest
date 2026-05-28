# Optimal Market State Theory (OMST) Backtest

This is the Python backtesting engine for the research paper: **Optimal Market State Theory (OMST): A Fractal Approach to Regime Classification**. 

OMST acts as a "Universal Translator" for market health, evaluating the physical properties of an asset: directional smoothness and liquidity depth.

The working paper is hosted on SSRN: [Insert Link]

## How It Works

The engine calculates the model's core metrics over rolling windows, perfectly matching the mathematical specifications of the paper:

### 1. Directional Smoothness ($S_t$)
We define the theoretical directional smoothness of a continuous price curve $P(t)$ over the interval $[0, T]$ as the ratio of net displacement to total arc length (Equation 1):

$$S_t = \frac{|P(T) - P(0)|}{\int_{0}^{T} \sqrt{1 + [P'(t)]^2} \, dt}$$

*Implementation Note:* In continuous-time stochastic calculus, a purely fractal price path has infinite total variation (arc length). In our discrete-time Python engine, we implement this as a path-efficiency ratio, comparing the straight-line Euclidean distance of the trend to the actual discrete path length traveled over the $k$-period window. This ensures $S_t \le 1.0$.

### 2. Relative Liquidity ($L_t$)
To remove size bias, raw liquidity metrics ($x_t$, the Amihud illiquidity ratio) are normalized using a rolling Z-score over a historical baseline window $W_{\text{baseline}}$ set to 252 trading days (one year) (Equation 11):

$$Z_t = \frac{x_t - \mu_{W_{\text{baseline}}}(x)}{\sigma_{W_{\text{baseline}}}(x)}$$

This decoupled baseline prevents the rolling Z-score from forcing artificial variance in our shorter evaluation window. The normalized Z-score is mapped to a $[0, 1]$ scale using a standard normal Cumulative Distribution Function (CDF) to define Liquidity Depth ($L_t = \Phi(-Z_t)$).

The raw blended OMSE percentage is then calculated as (Equation 2):

$$OMSE\% = \left( w_1 \cdot S_t + w_2 \cdot L_t \right) \times 100\%$$

### 3. Predispositioned State ($PS_t$)
To give the model "memory," yesterday's state probability acts as the transition probability or Bayesian prior ($PS_t$) (Equation 5):

$$PS_t = P(State_t = 1 \mid State_{t-1})$$

We then use Bayes' Theorem to continuously update this state probability given today's incoming data $\mathcal{D}_t$ (Equation 6):

$$P(State_t = 1 \mid \mathcal{D}_t) = \frac{P(\mathcal{D}_t \mid State_t = 1) \cdot PS_t}{\sum_{s \in \{0,1\}} P(\mathcal{D}_t \mid State_s) \cdot P(State_s)}$$

This Bayesian anchor filters out temporary daily data spikes.

### 4. Fluctuation Index (FIX)
The stability of the optimal state over time is measured by the Fluctuation Index ($FIX$), which divides the rolling mean efficiency by the product of its standard deviation and Interquartile Range ($IQR$) (Equation 7):

$$FIX = \frac{\mu_{OMSE}}{\sigma_{OMSE} \cdot IQR_{OMSE}}$$

We formally classify the output using the following mathematical piecewise function (Equation 9):

$$FIX_{\text{Regime}} = 
\begin{cases} 
    \text{Sustained Stability} & \text{if } FIX \ge 10.0 \\
    \text{Moderate Stability} & \text{if } 1.0 \le FIX < 10.0 \\
    \text{High-Noise Volatility} & \text{if } FIX < 1.0
\end{cases}$$

Applying the continuous limit as standard deviation approaches zero, the score approaches its upper limit of infinity (Equation 10):

$$\lim_{\sigma_{OMSE} \to 0} FIX = \infty$$

### 5. Ecosystem Convergence ($C_t$)
To validate the trend's macro strength, we calculate the Ecosystem Convergence ($C_t$) as the rolling Pearson correlation ($\rho$) of returns between our target asset ($R_a$) and its sector index ($R_s$) over the evaluation window $W$ (Equation 12):

$$C_t = \rho_{W}(R_a, R_s) = \frac{\text{Cov}(R_a, R_s)}{\sigma(R_a)\sigma(R_s)}$$

---

## Setup and Parameters

### Dependencies:
```bash
pip install yfinance pandas numpy matplotlib scipy
```

### Parameters:
*   `k = 20` (Fractal Periodicity lookback window - Eq 3)
*   `W = 60` (FIX evaluation and correlation window - Eq 12)
*   `baseline_W = 252` (Historical baseline window for Z-score normalization - Eq 11)
*   `w1 = 0.5` / `w2 = 0.5` (Weights for Smoothness and Liquidity - Eq 2)

### Run:
```bash
python omst_backtest.py
```

---

## Disclaimer
This is an academic research project and a proof-of-concept. It is not commercial trading software, nor does it provide investment advice. The backtest assumes a frictionless market (zero transaction fees, zero execution slippage, and perfect close execution), which does not exist in real-world trading. This engine serves as a macro regime-filtering model rather than a standalone high-frequency signal.

## AI Disclosure
I believe in full research transparency. The core economic hypotheses and logic are entirely my own. I used AI as an assistant to help write and debug the Python code, typeset the equations in LaTeX, and proofread the final paper.
