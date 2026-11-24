import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Quant Market Terminal", layout="wide")
st.title("Quant Market Terminal")

# Sidebar controls
st.sidebar.header("Terminal Controls")
ticker = st.sidebar.text_input("Primary Ticker", value="AAPL").upper()
benchmark = st.sidebar.text_input("Benchmark Ticker", value="SPY").upper()
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2024, 1, 1))
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.25, step=0.25) / 100

st.sidebar.divider()
st.sidebar.subheader("Monte Carlo Settings")
sim_days = st.sidebar.slider("Forecast Horizon (Trading Days)", 21, 252, 63)
sim_paths = st.sidebar.slider("Simulated Paths", 50, 500, 150)

@st.cache_data(ttl=300)
def fetch_history(sym, start):
    data = yf.download(sym, start=str(start), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [c.lower() for c in data.columns]
    return data

# Standard normal PDF & CDF without external dependencies
def norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

def norm_cdf(x):
    if isinstance(x, np.ndarray):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

def black_scholes_greeks(S, K, T, r, sigma):
    T = max(T, 1e-6)
    sigma = max(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    put_price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    delta_call = norm_cdf(d1)
    delta_put = delta_call - 1.0
    gamma = norm_pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * norm_pdf(d1) * np.sqrt(T)) / 100.0  # per 1% vol change
    theta_call = (-(S * norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm_cdf(d2)) / 365.0
    theta_put = (-(S * norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm_cdf(-d2)) / 365.0
    rho_call = (K * T * np.exp(-r * T) * norm_cdf(d2)) / 100.0
    rho_put = (-K * T * np.exp(-r * T) * norm_cdf(-d2)) / 100.0

    return {
        "call": call_price, "put": put_price,
        "delta_call": delta_call, "delta_put": delta_put,
        "gamma": gamma, "vega": vega,
        "theta_call": theta_call, "theta_put": theta_put,
        "rho_call": rho_call, "rho_put": rho_put,
    }

if ticker:
    try:
        with st.spinner(f"Loading market data & analytics for {ticker}..."):
            df = fetch_history(ticker, start_date)
            bench_df = fetch_history(benchmark, start_date)

        # Core quantitative metrics
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["rolling_vol"] = df["log_ret"].rolling(21).std() * np.sqrt(252)
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        ann_ret = df["log_ret"].mean() * 252
        ann_vol = df["log_ret"].std() * np.sqrt(252)
        sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
        rolling_max = df["close"].cummax()
        max_dd = ((df["close"] - rolling_max) / rolling_max).min()
        latest_spot = float(df["close"].iloc[-1])

        # Top KPI banner
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Close", f"${latest_spot:.2f}", f"{df['close'].pct_change().iloc[-1]*100:.2f}%")
        c2.metric("Annualized Return", f"{ann_ret*100:.2f}%")
        c3.metric("Annualized Volatility", f"{ann_vol*100:.2f}%")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c5.metric("Max Drawdown", f"{max_dd*100:.2f}%")

        # 8-Tab Quantitative Terminal Workspace
        tab_news, tab_chart, tab_options, tab_bs, tab_mc, tab_risk, tab_macro, tab_fund = st.tabs([
            "Live News Feed",
            "Price & Alpha",
            "Options Chain",
            "Black-Scholes & Greeks",
            "Monte Carlo",
            "Correlation & VaR Risk",
            "Macro Pulse",
            "Fundamentals"
        ])

        # TAB 1: LIVE FINANCIAL NEWS
        with tab_news:
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                st.subheader(f"{ticker} Company Headlines")
                try:
                    comp_news = yf.Ticker(ticker).news or []
                    for item in comp_news[:12]:
                        content = item.get("content", item)
                        title = content.get("title", "No title")
                        url = (content.get("canonicalUrl") or {}).get("url") or content.get("link", "")
                        pub_date = str(content.get("pubDate", ""))[:16]
                        st.markdown(f"**[{title}]({url})**  \n`{pub_date}`")
                        st.divider()
                except Exception as e:
                    st.info(f"Company news unavailable: {e}")

            with col_n2:
                st.subheader("Global Market Wire (S&P 500)")
                try:
                    spy_news = yf.Ticker("SPY").news or []
                    for item in spy_news[:12]:
                        content = item.get("content", item)
                        title = content.get("title", "No title")
                        url = (content.get("canonicalUrl") or {}).get("url") or content.get("link", "")
                        pub_date = str(content.get("pubDate", ""))[:16]
                        st.markdown(f"**[{title}]({url})**  \n`{pub_date}`")
                        st.divider()
                except Exception as e:
                    st.info(f"Market wire unavailable: {e}")

        # TAB 2: PRICE, VOLATILITY & BENCHMARK
        with tab_chart:
            st.subheader(f"{ticker} Price Action & Moving Averages")
            st.line_chart(df[["close", "sma_20", "sma_50"]])

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.subheader(f"Cumulative Return: {ticker} vs {benchmark}")
                comp = pd.DataFrame({
                    ticker: (1 + df["close"].pct_change()).cumprod() - 1,
                    benchmark: (1 + bench_df["close"].pct_change()).cumprod() - 1
                }).dropna()
                st.line_chart(comp)
            with col_v2:
                st.subheader("21-Day Rolling Annualized Volatility")
                st.line_chart(df["rolling_vol"])

        # TAB 3: OPTIONS CHAIN & IMPLIED VOLATILITY
        with tab_options:
            st.subheader(f"{ticker} Live Options Chain (Calls & Puts)")
            try:
                tk = yf.Ticker(ticker)
                expiries = tk.options
                selected_exp = st.selectbox("Select Expiration Date", expiries)
                chain = tk.option_chain(selected_exp)

                opt_type = st.radio("Option Type", ["Calls", "Puts"], horizontal=True)
                filtered_opt = chain.calls if opt_type == "Calls" else chain.puts

                cols_to_show = [c for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"] if c in filtered_opt.columns]
                st.dataframe(filtered_opt[cols_to_show], use_container_width=True)

                if "impliedVolatility" in filtered_opt.columns and "strike" in filtered_opt.columns:
                    st.subheader("Volatility Smile / Skew Across Strikes")
                    smile_df = filtered_opt[["strike", "impliedVolatility"]].dropna().set_index("strike")
                    st.line_chart(smile_df)
            except Exception as e:
                st.warning(f"Options data unavailable for {ticker}: {e}")

        # TAB 4: BLACK-SCHOLES OPTION PRICING & GREEKS
        with tab_bs:
            st.subheader(f"Black-Scholes-Merton European Option Pricer ({ticker})")
            b1, b2, b3 = st.columns(3)
            bs_strike = b1.number_input("Strike Price (K)", value=round(latest_spot, 2), step=1.0)
            bs_days = b2.slider("Days to Expiration (DTE)", 1, 365, 30)
            default_vol = float(ann_vol * 100) if not np.isnan(ann_vol) else 25.0
            bs_vol = b3.number_input("Volatility σ (%)", value=round(default_vol, 2), step=1.0) / 100.0

            T_years = bs_days / 365.0
            greeks = black_scholes_greeks(latest_spot, bs_strike, T_years, rf_rate, bs_vol)

            p1, p2 = st.columns(2)
            p1.metric("Theoretical Call Price", f"${greeks['call']:.2f}")
            p2.metric("Theoretical Put Price", f"${greeks['put']:.2f}")

            greeks_table = pd.DataFrame({
                "Greek": ["Delta (Δ)", "Gamma (Γ)", "Vega (ν / 1% vol)", "Theta (Θ / day)", "Rho (ρ / 1% rate)"],
                "Call": [
                    round(greeks["delta_call"], 4),
                    round(greeks["gamma"], 4),
                    round(greeks["vega"], 4),
                    round(greeks["theta_call"], 4),
                    round(greeks["rho_call"], 4),
                ],
                "Put": [
                    round(greeks["delta_put"], 4),
                    round(greeks["gamma"], 4),
                    round(greeks["vega"], 4),
                    round(greeks["theta_put"], 4),
                    round(greeks["rho_put"], 4),
                ]
            })
            st.dataframe(greeks_table, use_container_width=True, hide_index=True)

            # Spot sensitivity curve
            spot_grid = np.linspace(latest_spot * 0.75, latest_spot * 1.25, 60)
            curve_data = [black_scholes_greeks(s, bs_strike, T_years, rf_rate, bs_vol) for s in spot_grid]
            curve_df = pd.DataFrame({
                "Spot Price": spot_grid,
                "Call Value": [x["call"] for x in curve_data],
                "Put Value": [x["put"] for x in curve_data],
            }).set_index("Spot Price")
            st.subheader("Option Price Sensitivity Across Spot Prices")
            st.line_chart(curve_df)

        # TAB 5: MONTE CARLO GBM SIMULATION
        with tab_mc:
            st.subheader(f"Geometric Brownian Motion Simulation ({sim_days} Days, {sim_paths} Paths)")
            mu = df["log_ret"].mean()
            sigma = df["log_ret"].std()

            rng = np.random.default_rng(42)
            dt = 1
            shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=(sim_days, sim_paths))
            price_paths = latest_spot * np.exp(np.cumsum(shocks, axis=0))
            mc_df = pd.DataFrame(price_paths)

            st.line_chart(mc_df.iloc[:, :40])

            final_prices = price_paths[-1, :]
            m1, m2, m3 = st.columns(3)
            m1.metric("Median Simulated Price", f"${np.median(final_prices):.2f}")
            m2.metric("95% VaR (5th Percentile)", f"${np.percentile(final_prices, 5):.2f}")
            m3.metric("95th Percentile Upside", f"${np.percentile(final_prices, 95):.2f}")

        # TAB 6: MULTI-ASSET CORRELATION & VAR RISK PANEL
        with tab_risk:
            st.subheader("Multi-Asset Portfolio Risk & Correlation Matrix")
            r_col1, r_col2 = st.columns([2, 1])
            raw_tickers = r_col1.text_input("Portfolio Tickers (comma-separated)", value=f"{ticker}, MSFT, NVDA, SPY, GLD")
            port_value = r_col2.number_input("Portfolio Notional ($)", value=100000, step=5000)

            symbols = [s.strip().upper() for s in raw_tickers.split(",") if s.strip()]
            if len(symbols) >= 2:
                close_dict = {}
                for sym in symbols:
                    try:
                        hist = fetch_history(sym, start_date)
                        if not hist.empty and "close" in hist.columns:
                            close_dict[sym] = hist["close"]
                    except Exception:
                        pass

                if len(close_dict) >= 2:
                    prices_df = pd.DataFrame(close_dict).dropna()
                    rets_df = prices_df.pct_change().dropna()

                    n_assets = len(rets_df.columns)
                    weights = np.array([1.0 / n_assets] * n_assets)
                    port_daily_ret = rets_df.values @ weights

                    port_ann_ret = np.mean(port_daily_ret) * 252
                    port_ann_vol = np.std(port_daily_ret) * np.sqrt(252)
                    port_sharpe = (port_ann_ret - rf_rate) / port_ann_vol if port_ann_vol > 0 else 0

                    # Historical & Parametric 1-Day VaR and CVaR (Expected Shortfall)
                    var_95_hist = -np.percentile(port_daily_ret, 5) * port_value
                    var_99_hist = -np.percentile(port_daily_ret, 1) * port_value
                    cvar_95 = -port_daily_ret[port_daily_ret <= np.percentile(port_daily_ret, 5)].mean() * port_value
                    var_95_param = -(np.mean(port_daily_ret) - 1.64485 * np.std(port_daily_ret)) * port_value

                    rk1, rk2, rk3, rk4 = st.columns(4)
                    rk1.metric("Equal-Weight Ann. Return", f"{port_ann_ret*100:.2f}%")
                    rk2.metric("Portfolio Ann. Volatility", f"{port_ann_vol*100:.2f}%")
                    rk3.metric("Portfolio Sharpe Ratio", f"{port_sharpe:.2f}")
                    rk4.metric("1-Day 95% Hist. VaR", f"${var_95_hist:,.2f}")

                    rk5, rk6, rk7 = st.columns(3)
                    rk5.metric("1-Day 95% Parametric VaR", f"${var_95_param:,.2f}")
                    rk6.metric("1-Day 99% Hist. VaR", f"${var_99_hist:,.2f}")
                    rk7.metric("1-Day 95% CVaR (Expected Shortfall)", f"${cvar_95:,.2f}")

                    st.subheader("Pairwise Daily Return Correlation Matrix")
                    corr_matrix = rets_df.corr().round(3)
                    st.dataframe(corr_matrix, use_container_width=True)

                    st.subheader("Cumulative Equal-Weight Portfolio Growth ($)")
                    port_cum = (1 + pd.Series(port_daily_ret, index=rets_df.index)).cumprod() * port_value
                    st.line_chart(port_cum.rename("Portfolio Value ($)"))
                else:
                    st.warning("Could not load enough valid tickers to compute correlation.")
            else:
                st.info("Enter at least two comma-separated tickers.")

        # TAB 7: MACRO & CROSS-ASSET WATCHLIST
        with tab_macro:
            st.subheader("Cross-Asset Pulse (Equities, Volatility, Rates, Commodities, FX)")
            macro_tickers = {
                "S&P 500 (SPY)": "SPY",
                "Nasdaq 100 (QQQ)": "QQQ",
                "VIX Volatility (^VIX)": "^VIX",
                "US 10Y Yield (^TNX)": "^TNX",
                "Gold (GC=F)": "GC=F",
                "WTI Crude Oil (CL=F)": "CL=F",
                "GBP/USD (GBPUSD=X)": "GBPUSD=X",
                "Bitcoin (BTC-USD)": "BTC-USD"
            }
            macro_rows = []
            for label, sym in macro_tickers.items():
                try:
                    m_df = fetch_history(sym, datetime.date.today() - datetime.timedelta(days=10))
                    last_val = m_df["close"].iloc[-1]
                    prev_val = m_df["close"].iloc[-2]
                    chg = ((last_val - prev_val) / prev_val) * 100
                    macro_rows.append({"Asset": label, "Ticker": sym, "Last": round(last_val, 3), "1D Change (%)": round(chg, 2)})
                except Exception:
                    pass
            if macro_rows:
                st.dataframe(pd.DataFrame(macro_rows), use_container_width=True)

        # TAB 8: FUNDAMENTALS & RAW DATA
        with tab_fund:
            st.subheader(f"{ticker} Company Profile & Key Stats")
            try:
                info = yf.Ticker(ticker).info
                key_fields = {k: info.get(k) for k in ["shortName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "beta", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"] if k in info}
                st.dataframe(pd.DataFrame.from_dict(key_fields, orient="index", columns=["Value"]), use_container_width=True)
            except Exception:
                st.info("Profile data unavailable.")
            st.subheader("Historical OHLCV & Quantitative Series")
            st.dataframe(df.tail(25), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")