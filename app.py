import os
os.environ["OPENBB_AUTO_BUILD"] = "false"
import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quant Market Terminal", layout="wide")
st.title("Quant Market Terminal")

@st.cache_resource
def load_openbb():
    import openbb
    openbb.build()
    from openbb import obb
    return obb

obb = load_openbb()

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

if ticker:
    try:
        with st.spinner(f"Loading market data & analytics for {ticker}..."):
            df = obb.equity.price.historical(
                symbol=ticker, start_date=str(start_date), provider="yfinance"
            ).to_dataframe()
            bench_df = obb.equity.price.historical(
                symbol=benchmark, start_date=str(start_date), provider="yfinance"
            ).to_dataframe()

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

        # Top KPI banner
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Close", f"${df['close'].iloc[-1]:.2f}", f"{df['close'].pct_change().iloc[-1]*100:.2f}%")
        c2.metric("Annualized Return", f"{ann_ret*100:.2f}%")
        c3.metric("Annualized Volatility", f"{ann_vol*100:.2f}%")
        c4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c5.metric("Max Drawdown", f"{max_dd*100:.2f}%")

        # 6-Tab Bloomberg-style workspace
        tab_news, tab_chart, tab_options, tab_mc, tab_macro, tab_fund = st.tabs([
            "Live News Feed",
            "Price, Volatility & Alpha",
            "Options & Implied Vol",
            "Monte Carlo Simulation",
            "Macro & Cross-Asset",
            "Fundamentals & Data"
        ])

        # TAB 1: LIVE FINANCIAL NEWS
        with tab_news:
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                st.subheader(f"{ticker} Company Headlines")
                try:
                    comp_news = obb.news.company(symbol=ticker, limit=12, provider="yfinance").to_dataframe()
                    for _, row in comp_news.iterrows():
                        title = row.get("title", "No title")
                        url = row.get("url", "")
                        date_str = str(row.get("date", ""))[:16]
                        st.markdown(f"**[{title}]({url})**  \n`{date_str}`")
                        st.divider()
                except Exception as e:
                    st.info(f"Company news unavailable: {e}")

            with col_n2:
                st.subheader("Global Market Wire")
                try:
                    world_news = obb.news.world(limit=12, provider="yfinance").to_dataframe()
                    for _, row in world_news.iterrows():
                        title = row.get("title", "No title")
                        url = row.get("url", "")
                        date_str = str(row.get("date", ""))[:16]
                        st.markdown(f"**[{title}]({url})**  \n`{date_str}`")
                        st.divider()
                except Exception:
                    spy_news = obb.news.company(symbol="SPY", limit=12, provider="yfinance").to_dataframe()
                    for _, row in spy_news.iterrows():
                        title = row.get("title", "No title")
                        url = row.get("url", "")
                        date_str = str(row.get("date", ""))[:16]
                        st.markdown(f"**[{title}]({url})**  \n`{date_str}`")
                        st.divider()

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
                chains = obb.derivatives.options.chains(symbol=ticker, provider="yfinance").to_dataframe()
                expiries = sorted(chains["expiration"].astype(str).unique())
                selected_exp = st.selectbox("Select Expiration Date", expiries)
                sub_chain = chains[chains["expiration"].astype(str) == selected_exp]

                opt_type = st.radio("Option Type", ["call", "put"], horizontal=True)
                filtered_opt = sub_chain[sub_chain["option_type"] == opt_type]

                cols_to_show = [c for c in ["strike", "last_trade_price", "bid", "ask", "volume", "open_interest", "implied_volatility"] if c in filtered_opt.columns]
                st.dataframe(filtered_opt[cols_to_show], use_container_width=True)

                if "implied_volatility" in filtered_opt.columns and "strike" in filtered_opt.columns:
                    st.subheader("Volatility Smile / Skew Across Strikes")
                    smile_df = filtered_opt[["strike", "implied_volatility"]].dropna().set_index("strike")
                    st.line_chart(smile_df)
            except Exception as e:
                st.warning(f"Options data unavailable for {ticker}: {e}")

        # TAB 4: MONTE CARLO GBM SIMULATION
        with tab_mc:
            st.subheader(f"Geometric Brownian Motion Simulation ({sim_days} Days, {sim_paths} Paths)")
            last_price = df["close"].iloc[-1]
            mu = df["log_ret"].mean()
            sigma = df["log_ret"].std()

            rng = np.random.default_rng(42)
            dt = 1
            shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=(sim_days, sim_paths))
            price_paths = last_price * np.exp(np.cumsum(shocks, axis=0))
            mc_df = pd.DataFrame(price_paths)

            st.line_chart(mc_df.iloc[:, :40])

            final_prices = price_paths[-1, :]
            m1, m2, m3 = st.columns(3)
            m1.metric("Median Simulated Price", f"${np.median(final_prices):.2f}")
            m2.metric("95% VaR (5th Percentile)", f"${np.percentile(final_prices, 5):.2f}")
            m3.metric("95th Percentile Upside", f"${np.percentile(final_prices, 95):.2f}")

        # TAB 5: MACRO & CROSS-ASSET WATCHLIST
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
                    m_df = obb.equity.price.historical(symbol=sym, start_date=str(datetime.date.today() - datetime.timedelta(days=10)), provider="yfinance").to_dataframe()
                    last_val = m_df["close"].iloc[-1]
                    prev_val = m_df["close"].iloc[-2]
                    chg = ((last_val - prev_val) / prev_val) * 100
                    macro_rows.append({"Asset": label, "Ticker": sym, "Last": round(last_val, 3), "1D Change (%)": round(chg, 2)})
                except Exception:
                    pass
            if macro_rows:
                st.dataframe(pd.DataFrame(macro_rows), use_container_width=True)

        # TAB 6: FUNDAMENTALS & RAW DATA
        with tab_fund:
            st.subheader(f"{ticker} Company Profile & Key Stats")
            try:
                profile = obb.equity.profile(symbol=ticker, provider="yfinance").to_dataframe()
                st.dataframe(profile.T, use_container_width=True)
            except Exception:
                st.info("Profile data unavailable.")
            st.subheader("Historical OHLCV & Quantitative Series")
            st.dataframe(df.tail(25), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")