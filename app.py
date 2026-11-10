from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SignalForge Strategy Lab", page_icon="ðŸ“ˆ", layout="wide")

ROOT = Path(__file__).resolve().parent
DEFAULT_CANDLES = ROOT / "data" / "raw" / "btc_usdt_1m.csv"
DEFAULT_FUNDING = ROOT / "data" / "raw" / "btc_usdt_funding.csv"

st.markdown("""
<style>
.stApp { background: #0b1020; color: #e8edf7; }
[data-testid="stSidebar"] { background: #11182b; }
div[data-testid="stMetric"] { background: #121b30; border: 1px solid #283650; padding: 14px; border-radius: 12px; }
h1, h2, h3 { letter-spacing: -0.03em; }
</style>
""", unsafe_allow_html=True)

st.title("SignalForge Strategy Lab")
st.caption("BTCUSDT Â· Historical strategy research Â· No live orders are placed")
st.warning(
    "Research tool only. Results depend on data quality, assumptions, and selected rules. "
    "This dashboard does not establish that a strategy will be profitable in live markets."
)

def load_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candle CSV not found: {path}")
    df = pd.read_csv(path)
    needed = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Candle CSV missing columns: {missing}")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce", format="mixed")
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce", format="mixed")
    else:
        df["close_time"] = df["open_time"] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=needed + ["close_time"]).sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("At least 100 valid candles are required.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Found non-positive OHLC prices.")
    return df

def load_funding(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=["fundingTime", "fundingRate", "markPrice"]), "Funding CSV not found; funding costs will be zero and this must be disclosed."
    f = pd.read_csv(path)
    required = ["fundingTime", "fundingRate", "markPrice"]
    if any(c not in f.columns for c in required):
        return pd.DataFrame(columns=required), "Funding file is missing required columns; funding costs will be zero."
    f["fundingTime"] = pd.to_datetime(f["fundingTime"], utc=True, errors="coerce", format="mixed")
    for c in ["fundingRate", "markPrice"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.dropna(subset=required).sort_values("fundingTime").reset_index(drop=True)
    return f, None if not f.empty else "No valid funding rows; funding costs will be zero."

def add_indicators(df):
    d = df.copy()
    d["ema_fast"] = d["close"].ewm(span=12, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=26, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    d["rsi"] = d["rsi"].fillna(50)
    d["sma_20"] = d["close"].rolling(20).mean()
    d["return_1"] = d["close"].pct_change()
    d["return_1_pct"] = d["return_1"] * 100
    d["volume_sma_20"] = d["volume"].rolling(20).mean()
    d["volume_ratio"] = d["volume"] / d["volume_sma_20"].replace(0, np.nan)
    d["ema_9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema_21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["macd"] = d["close"].ewm(span=12, adjust=False).mean() - d["close"].ewm(span=26, adjust=False).mean()
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    prev_close = d["close"].shift(1)
    true_range = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["atr_14"] = true_range.rolling(14).mean()
    d["bb_mid"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std
    return d

RULE_INDICATORS = {
    "Close": "close",
    "EMA 9": "ema_9",
    "EMA 12": "ema_fast",
    "EMA 21": "ema_21",
    "EMA 26": "ema_slow",
    "SMA 20": "sma_20",
    "RSI 14": "rsi",
    "MACD": "macd",
    "MACD signal": "macd_signal",
    "ATR 14": "atr_14",
    "Bollinger upper": "bb_upper",
    "Bollinger middle": "bb_mid",
    "Bollinger lower": "bb_lower",
    "Volume": "volume",
    "Volume ratio": "volume_ratio",
    "1-candle return (%)": "return_1_pct",
}

RULE_OPERATORS = ["greater than", "greater/equal", "less than", "less/equal", "crosses above", "crosses below"]

def add_rule_condition_ui(label, prefix):
    st.markdown(f"**{label}**")
    c1, c2, c3 = st.columns([1.2, 1.1, 1])
    indicator_label = c1.selectbox("Indicator", list(RULE_INDICATORS), key=f"{prefix}_indicator")
    operator = c2.selectbox("Operator", RULE_OPERATORS, key=f"{prefix}_operator")
    default_value = 30.0 if indicator_label == "RSI 14" else (1.0 if indicator_label == "Volume ratio" else 0.0)
    value = c3.number_input("Threshold", value=default_value, key=f"{prefix}_threshold", format="%.6f")
    return (RULE_INDICATORS[indicator_label], operator, float(value))

def evaluate_rule_condition(df, rule):
    col, operator, threshold = rule
    current = pd.to_numeric(df[col], errors="coerce")
    previous = current.shift(1)
    if operator == "greater than":
        result = current > threshold
    elif operator == "greater/equal":
        result = current >= threshold
    elif operator == "less than":
        result = current < threshold
    elif operator == "less/equal":
        result = current <= threshold
    elif operator == "crosses above":
        result = (current > threshold) & (previous <= threshold)
    else:
        result = (current < threshold) & (previous >= threshold)
    return result.fillna(False)

def combine_rule_conditions(df, first, connector, second):
    a = evaluate_rule_condition(df, first)
    b = evaluate_rule_condition(df, second)
    return (a & b) if connector == "AND" else (a | b)

def make_signals(df, entry_mode, trade_direction, rsi_long_max, rsi_short_min,
                 custom_long=None, custom_short=None):
    d = df.copy()
    if entry_mode == "EMA crossover":
        cross_up = (d["ema_fast"] > d["ema_slow"]) & (d["ema_fast"].shift(1) <= d["ema_slow"].shift(1))
        cross_down = (d["ema_fast"] < d["ema_slow"]) & (d["ema_fast"].shift(1) >= d["ema_slow"].shift(1))
        long_sig = cross_up
        short_sig = cross_down
    elif entry_mode == "RSI reversal":
        long_sig = (d["rsi"] > 30) & (d["rsi"].shift(1) <= 30)
        short_sig = (d["rsi"] < 70) & (d["rsi"].shift(1) >= 70)
    elif entry_mode == "EMA trend + RSI filter":
        long_sig = (d["ema_fast"] > d["ema_slow"]) & (d["rsi"] <= rsi_long_max)
        short_sig = (d["ema_fast"] < d["ema_slow"]) & (d["rsi"] >= rsi_short_min)
    elif entry_mode == "Custom rule builder":
        if custom_long is None or custom_short is None:
            raise ValueError("Custom entry conditions are required.")
        long_sig = combine_rule_conditions(d, *custom_long)
        short_sig = combine_rule_conditions(d, *custom_short)
    else:
        cross_up = (
            (d["close"] > d["sma_20"])
            & (d["close"].shift(1) <= d["sma_20"].shift(1))
        )
        cross_down = (
            (d["close"] < d["sma_20"])
            & (d["close"].shift(1) >= d["sma_20"].shift(1))
        )
        long_sig = cross_up
        short_sig = cross_down
    if trade_direction == "Long only":
        short_sig = pd.Series(False, index=d.index)
    elif trade_direction == "Short only":
        long_sig = pd.Series(False, index=d.index)
    d["long_signal"] = long_sig.fillna(False)
    d["short_signal"] = short_sig.fillna(False)
    return d

def run_backtest(d, funding, start_capital, position_pct, fee_rate, slippage_rate,
                 stop_pct, target_pct, max_hold, allow_reverse,
                 custom_exit_enabled=False):
    start = d["open_time"].min()
    end = d["open_time"].max()
    work = d.loc[(d["open_time"] >= start) & (d["open_time"] <= end)].reset_index(drop=True)
    equity = float(start_capital)
    peak = equity
    max_dd = 0.0
    trades = []
    equity_points = [{"time": work.loc[0, "open_time"], "equity": equity}]
    i = 1
    while i < len(work) - 1:
        row = work.iloc[i]
        side = 1 if bool(row["long_signal"]) else (-1 if bool(row["short_signal"]) else 0)
        if side == 0:
            i += 1
            continue
        # Signal is known at candle i close; enter at candle i+1 open.
        entry_i = i + 1
        entry_row = work.iloc[entry_i]
        entry_market = float(entry_row["open"])
        entry_fill = entry_market * (1 + slippage_rate * side)
        notional = equity * position_pct
        if notional <= 0 or entry_fill <= 0:
            i += 1
            continue
        qty = notional / entry_fill
        stop_price = entry_fill * (1 - stop_pct * side) if stop_pct > 0 else None
        target_price = entry_fill * (1 + target_pct * side) if target_pct > 0 else None
        exit_i = min(entry_i + max_hold - 1, len(work) - 1)
        exit_reason = "Max holding period"
        exit_market = float(work.iloc[exit_i]["close"])
        actual_exit_i = exit_i
        exit_execution_time = work.iloc[exit_i]["close_time"]
        for j in range(entry_i, exit_i + 1):
            bar = work.iloc[j]
            lo, hi = float(bar["low"]), float(bar["high"])
            stop_hit = stop_price is not None and ((side == 1 and lo <= stop_price) or (side == -1 and hi >= stop_price))
            target_hit = target_price is not None and ((side == 1 and hi >= target_price) or (side == -1 and lo <= target_price))
            # Conservative intrabar assumption: if both levels hit in one candle, stop is assumed first.
            if stop_hit:
                exit_market = float(stop_price)
                actual_exit_i = j
                # OHLC data cannot reveal the exact intrabar execution timestamp;
                # use this candle's close timestamp as a conservative bookkeeping proxy.
                exit_execution_time = bar["close_time"]
                exit_reason = "Stop-loss" if not target_hit else "Stop-loss (both levels touched)"
                break
            if target_hit:
                exit_market = float(target_price)
                actual_exit_i = j
                exit_execution_time = bar["close_time"]
                exit_reason = "Take-profit"
                break
            # Signals are known at bar close and are filled at the next candle open.
            opposite = (side == 1 and bool(bar["short_signal"])) or (side == -1 and bool(bar["long_signal"]))
            custom_exit = custom_exit_enabled and (
                (side == 1 and bool(bar.get("exit_long_signal", False))) or
                (side == -1 and bool(bar.get("exit_short_signal", False)))
            )
            if (allow_reverse and opposite) or custom_exit:
                if j < len(work) - 1:
                    actual_exit_i = j + 1
                    exit_market = float(work.iloc[actual_exit_i]["open"])
                    exit_execution_time = work.iloc[actual_exit_i]["open_time"]
                    exit_reason = "Custom exit rule" if custom_exit else "Opposite signal"
                    break
        exit_row = work.iloc[actual_exit_i]
        exit_fill = exit_market * (1 - slippage_rate * side)
        exit_notional = qty * exit_fill
        entry_fee = notional * fee_rate
        exit_fee = exit_notional * fee_rate
        # Separate raw market P&L from simulated slippage so cost attribution is auditable.
        gross_market_pnl = qty * (exit_market - entry_market) * side
        pnl_after_slippage_before_fees = qty * (exit_fill - entry_fill) * side
        slippage_cost = gross_market_pnl - pnl_after_slippage_before_fees
        funding_cost = 0.0
        funding_count = 0
        if not funding.empty:
            events = funding.loc[
                (funding["fundingTime"] > entry_row["open_time"]) &
                (funding["fundingTime"] <= exit_execution_time)
            ]
            if not events.empty:
                for _, ev in events.iterrows():
                    # Positive funding: longs pay, shorts receive. Use historical mark price.
                    funding_cost += side * qty * float(ev["markPrice"]) * float(ev["fundingRate"])
                    funding_count += 1
        net_pnl = pnl_after_slippage_before_fees - entry_fee - exit_fee - funding_cost
        equity_before = equity
        equity += net_pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        trades.append({
            "side": "LONG" if side == 1 else "SHORT",
            "signal_time": row["open_time"],
            "entry_time": entry_row["open_time"],
            "exit_time": exit_execution_time,
            "entry_price": entry_fill,
            "exit_price": exit_fill,
            "position_notional": notional,
            "gross_market_pnl_before_slippage": gross_market_pnl,
            "slippage_cost": slippage_cost,
            "pnl_after_slippage_before_fees": pnl_after_slippage_before_fees,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "funding_cost": funding_cost,
            "funding_events": funding_count,
            "net_pnl": net_pnl,
            "return_on_starting_equity_pct": net_pnl / start_capital * 100,
            "equity_before": equity_before,
            "equity_after": equity,
            "exit_reason": exit_reason,
        })
        equity_points.append({"time": exit_execution_time, "equity": equity})
        i = max(actual_exit_i + 1, i + 1)
    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_points).drop_duplicates("time").sort_values("time")
    return trades_df, eq_df, equity, max_dd

with st.sidebar:
    st.header("Data")
    candle_path_text = st.text_input("Candle CSV path", str(DEFAULT_CANDLES))
    funding_path_text = st.text_input("Funding CSV path", str(DEFAULT_FUNDING))
    st.caption("Paths default to the project's existing raw-data files.")
    st.header("Strategy")
    entry_mode = st.selectbox("Entry rule", ["EMA crossover", "RSI reversal", "EMA trend + RSI filter", "Close vs 20-period SMA", "Custom rule builder"])
    direction = st.selectbox("Trade direction", ["Long and short", "Long only", "Short only"])
    rsi_long_max = st.slider("RSI maximum for long trend entries", 10, 60, 40)
    rsi_short_min = st.slider("RSI minimum for short trend entries", 40, 90, 60)

    custom_long = custom_short = None
    if entry_mode == "Custom rule builder":
        st.markdown("---")
        st.subheader("Custom entries")
        st.caption("Each side combines two threshold conditions. Cross rules mean crossing a numeric threshold between candles.")
        with st.expander("Long entry conditions", expanded=True):
            long_rule_1 = add_rule_condition_ui("Condition 1", "entry_long_1")
            long_connector = st.selectbox("Combine with", ["AND", "OR"], key="entry_long_connector")
            long_rule_2 = add_rule_condition_ui("Condition 2", "entry_long_2")
        with st.expander("Short entry conditions", expanded=True):
            short_rule_1 = add_rule_condition_ui("Condition 1", "entry_short_1")
            short_connector = st.selectbox("Combine with", ["AND", "OR"], key="entry_short_connector")
            short_rule_2 = add_rule_condition_ui("Condition 2", "entry_short_2")
        custom_long = (long_rule_1, long_connector, long_rule_2)
        custom_short = (short_rule_1, short_connector, short_rule_2)

    st.header("Exit rules")
    stop_pct = st.number_input("Stop-loss (%) â€” 0 disables", min_value=0.0, max_value=20.0, value=1.0, step=0.25) / 100
    target_pct = st.number_input("Take-profit (%) â€” 0 disables", min_value=0.0, max_value=50.0, value=2.0, step=0.25) / 100
    max_hold = st.number_input("Maximum holding period (candles)", min_value=1, max_value=1440, value=30, step=1)
    allow_reverse = st.checkbox("Exit on opposite entry signal", value=True)
    custom_exit_enabled = st.checkbox("Enable custom indicator exit rules", value=False)
    custom_exit_long = custom_exit_short = None
    if custom_exit_enabled:
        with st.expander("Custom exit conditions", expanded=True):
            st.caption("Each exit condition is checked at candle close and filled at the next candle open.")
            custom_exit_long = add_rule_condition_ui("Exit LONG when condition is true", "exit_long")
            custom_exit_short = add_rule_condition_ui("Exit SHORT when condition is true", "exit_short")
    st.header("Execution assumptions")
    starting_capital = st.number_input("Starting equity (USDT)", min_value=10.0, value=1000.0, step=100.0)
    position_pct = st.slider("Position notional (% of current equity)", 1, 100, 10) / 100
    fee_rate = st.number_input("Taker fee per side (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.01) / 100
    slippage_rate = st.number_input("Slippage per side (%)", min_value=0.0, max_value=2.0, value=0.02, step=0.01) / 100
    run_clicked = st.button("Run backtest", type="primary", use_container_width=True)

if not run_clicked:
    st.info("Choose your strategy and execution assumptions in the sidebar, then click **Run backtest**.")
    st.markdown("### Included strategy templates")
    c1, c2, c3 = st.columns(3)
    c1.subheader("EMA crossover")
    c1.write("Long when EMA 12 crosses above EMA 26; short when it crosses below.")
    c2.subheader("RSI reversal")
    c2.write("Long when RSI crosses back above 30; short when it crosses below 70.")
    c3.subheader("Trend + RSI")
    c3.write("Trade with EMA trend, filtered by the RSI levels you choose.")
    st.caption("Custom entries support two indicator-threshold conditions per direction with AND/OR logic. Custom exits support one threshold condition per direction. Arbitrary uploaded Python strategies are not executed.")
    st.stop()

try:
    candle_path = Path(candle_path_text).expanduser()
    funding_path = Path(funding_path_text).expanduser()
    candles = load_candles(candle_path)
    funding, funding_warning = load_funding(funding_path)
    if funding_warning:
        st.warning(funding_warning)
    data = add_indicators(candles)
    data = make_signals(data, entry_mode, direction, rsi_long_max, rsi_short_min,
                        custom_long=custom_long, custom_short=custom_short)
    if custom_exit_enabled:
        data["exit_long_signal"] = evaluate_rule_condition(data, custom_exit_long)
        data["exit_short_signal"] = evaluate_rule_condition(data, custom_exit_short)
    else:
        data["exit_long_signal"] = False
        data["exit_short_signal"] = False
    trades, equity_curve, final_equity, max_drawdown = run_backtest(
        data, funding, starting_capital, position_pct, fee_rate, slippage_rate,
        stop_pct, target_pct, int(max_hold), allow_reverse, custom_exit_enabled
    )
except Exception as exc:
    st.error(f"Could not run backtest: {exc}")
    st.stop()

net_pnl = final_equity - starting_capital
net_return = net_pnl / starting_capital * 100
n_trades = len(trades)
wins = int((trades["net_pnl"] > 0).sum()) if n_trades else 0
losses = int((trades["net_pnl"] <= 0).sum()) if n_trades else 0
win_rate = wins / n_trades * 100 if n_trades else 0.0
gross_wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum() if n_trades else 0.0
gross_losses = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum() if n_trades else 0.0
profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)
avg_win = trades.loc[trades["net_pnl"] > 0, "net_pnl"].mean() if wins else 0.0
avg_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].mean() if losses else 0.0

st.subheader("Backtest results")
st.caption(f"Strategy: {entry_mode} Â· Direction: {direction} Â· Data: {candles['open_time'].min()} to {candles['open_time'].max()} Â· {len(candles):,} candles")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net P&L", f"{net_pnl:,.2f} USDT", f"{net_return:.2f}%")
m2.metric("Ending equity", f"{final_equity:,.2f} USDT")
m3.metric("Trades", f"{n_trades:,}")
m4.metric("Win rate", f"{win_rate:.1f}%")
m5.metric("Max drawdown", f"{max_drawdown * 100:.2f}%")

m6, m7, m8, m9 = st.columns(4)
m6.metric("Profit factor", "âˆž" if math.isinf(profit_factor) else f"{profit_factor:.2f}")
m7.metric("Average win", f"{avg_win:,.2f} USDT")
m8.metric("Average loss", f"{avg_loss:,.2f} USDT")
m9.metric("Funding cost", f"{trades['funding_cost'].sum():,.4f} USDT" if n_trades else "0.0000 USDT")

tab1, tab2, tab3, tab4 = st.tabs(["Equity curve", "Trade log", "Diagnostics", "Assumptions"])
with tab1:
    if not equity_curve.empty:
        chart_df = equity_curve.set_index("time")[["equity"]]
        st.line_chart(chart_df, height=320)
    else:
        st.info("No completed trades yet; no equity curve is available.")
with tab2:
    if trades.empty:
        st.info("No trades were generated with these rules and settings.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download trades CSV",
            data=trades.to_csv(index=False).encode("utf-8"),
            file_name="signalforge_strategy_trades.csv",
            mime="text/csv",
        )
with tab3:
    if trades.empty:
        st.info("No completed trades to diagnose yet.")
    else:
        total_entry_fees = float(trades["entry_fee"].sum())
        total_exit_fees = float(trades["exit_fee"].sum())
        total_fees = total_entry_fees + total_exit_fees
        total_slippage = float(trades["slippage_cost"].sum())
        total_funding = float(trades["funding_cost"].sum())
        market_pnl = float(trades["gross_market_pnl_before_slippage"].sum())
        diag_cols = st.columns(5)
        diag_cols[0].metric("Market P&L before slippage", f"{market_pnl:,.2f} USDT")
        diag_cols[1].metric("Simulated slippage", f"{total_slippage:,.2f} USDT")
        diag_cols[2].metric("Entry + exit fees", f"{total_fees:,.2f} USDT")
        diag_cols[3].metric("Funding cost", f"{total_funding:,.4f} USDT")
        diag_cols[4].metric("Net P&L", f"{float(trades['net_pnl'].sum()):,.2f} USDT")
        st.markdown("#### Performance by exit reason")
        reason_summary = trades.groupby("exit_reason", dropna=False).agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda x: (x > 0).mean() * 100),
            fees=("entry_fee", "sum"),
        )
        exit_fees_by_reason = trades.groupby("exit_reason")["exit_fee"].sum()
        reason_summary["fees"] = reason_summary["fees"] + exit_fees_by_reason
        reason_summary["slippage_cost"] = trades.groupby("exit_reason")["slippage_cost"].sum()
        reason_summary["funding_cost"] = trades.groupby("exit_reason")["funding_cost"].sum()
        st.dataframe(reason_summary.reset_index().round(4), use_container_width=True, hide_index=True)
        st.markdown("#### Performance by direction")
        side_summary = trades.groupby("side").agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            win_rate_pct=("net_pnl", lambda x: (x > 0).mean() * 100),
        )
        st.dataframe(side_summary.reset_index().round(4), use_container_width=True, hide_index=True)
        st.caption("Market P&L is calculated using unadjusted candle prices. Slippage is separately estimated from the simulated entry and exit fills. Fees and funding are then deducted to produce net P&L.")
with tab4:
    st.write({
        "Entry execution": "Next candle open after signal candle closes",
        "Stop/target intrabar ambiguity": "If both are touched in the same candle, stop-loss is assumed first",
        "Fees": f"{fee_rate*100:.4f}% per side",
        "Slippage": f"{slippage_rate*100:.4f}% per side; separately reported in trade log",
        "Cost attribution": "Raw market P&L minus modeled slippage, entry/exit fees, and funding equals net P&L",
        "Funding": "Historical events between entry and exit; positive funding charged to longs and credited to shorts",
        "Leverage/margin/liquidation": "Not modeled",
        "Market impact/order-book depth": "Not modeled",
        "Strategy type": "Built-in templates or visual indicator-threshold rules; arbitrary uploaded Python is not executed",
        "Custom exit rules": bool(custom_exit_enabled),
        "Exit timestamp": "Recorded at the estimated execution time; opposite/custom signals fill at the next candle open",
    })
    st.caption("Do not interpret this backtest as a guarantee of future results. Check for data gaps and test on an untouched out-of-sample period before making decisions.")

st.subheader("Strategy settings")
st.json({
    "entry_rule": entry_mode,
    "direction": direction,
    "stop_loss_pct": round(stop_pct * 100, 4),
    "take_profit_pct": round(target_pct * 100, 4),
    "max_hold_candles": int(max_hold),
    "starting_equity": starting_capital,
    "position_fraction": position_pct,
    "fee_per_side_pct": fee_rate * 100,
    "slippage_per_side_pct": slippage_rate * 100,
})
summary = {
    "strategy": entry_mode,
    "direction": direction,
    "data_start": str(candles["open_time"].min()),
    "data_end": str(candles["open_time"].max()),
    "candles": int(len(candles)),
    "trades": int(n_trades),
    "wins": int(wins),
    "losses": int(losses),
    "win_rate_pct": float(win_rate),
    "starting_equity": float(starting_capital),
    "ending_equity": float(final_equity),
    "net_pnl": float(net_pnl),
    "net_return_pct": float(net_return),
    "max_drawdown_pct": float(max_drawdown * 100),
    "profit_factor": None if math.isinf(profit_factor) else float(profit_factor),
    "fee_per_side_pct": float(fee_rate * 100),
    "slippage_per_side_pct": float(slippage_rate * 100),
    "gross_market_pnl_before_slippage": float(trades["gross_market_pnl_before_slippage"].sum()) if n_trades else 0.0,
    "slippage_cost": float(trades["slippage_cost"].sum()) if n_trades else 0.0,
    "entry_fees": float(trades["entry_fee"].sum()) if n_trades else 0.0,
    "exit_fees": float(trades["exit_fee"].sum()) if n_trades else 0.0,
    "funding_cost": float(trades["funding_cost"].sum()) if n_trades else 0.0,
    "custom_exit_rules_enabled": bool(custom_exit_enabled),
    "limitations": ["No leverage/margin/liquidation", "No order-book spread/depth or market impact", "Custom rules limited to supported indicator threshold conditions"]
}
st.download_button(
    "Download summary JSON",
    data=pd.Series(summary).to_json(indent=2),
    file_name="signalforge_strategy_summary.json",
    mime="application/json",
)


st.subheader("Chronological robustness check")
st.caption(
    "The same selected rules and execution assumptions are tested independently across three consecutive "
    "time segments. This is a stability check, not parameter optimization or proof of future profitability. "
    "Each segment starts with the configured starting equity; open trades are closed at that segment's end."
)

segment_rows = []
segment_edges = np.linspace(0, len(data), 4, dtype=int)
for segment_idx in range(3):
    seg_start = int(segment_edges[segment_idx])
    seg_end = int(segment_edges[segment_idx + 1])
    segment = data.iloc[seg_start:seg_end].copy().reset_index(drop=True)
    if len(segment) < 3:
        continue
    seg_start_time = segment["open_time"].min()
    seg_end_time = segment["open_time"].max()
    segment_funding = funding.loc[
        (funding["fundingTime"] > seg_start_time) &
        (funding["fundingTime"] <= segment["close_time"].max())
    ].copy() if not funding.empty else funding.copy()

    seg_trades, seg_equity_curve, seg_final_equity, seg_max_dd = run_backtest(
        segment,
        segment_funding,
        starting_capital,
        position_pct,
        fee_rate,
        slippage_rate,
        stop_pct,
        target_pct,
        int(max_hold),
        allow_reverse,
        custom_exit_enabled,
    )
    seg_count = len(seg_trades)
    seg_wins = int((seg_trades["net_pnl"] > 0).sum()) if seg_count else 0
    seg_gross_wins = float(seg_trades.loc[seg_trades["net_pnl"] > 0, "net_pnl"].sum()) if seg_count else 0.0
    seg_gross_losses = float(-seg_trades.loc[seg_trades["net_pnl"] < 0, "net_pnl"].sum()) if seg_count else 0.0
    seg_pf = seg_gross_wins / seg_gross_losses if seg_gross_losses > 0 else (float("inf") if seg_gross_wins > 0 else 0.0)
    seg_net_pnl = float(seg_final_equity - starting_capital)
    segment_rows.append({
        "segment": f"Segment {segment_idx + 1}",
        "start_utc": str(seg_start_time),
        "end_utc": str(seg_end_time),
        "candles": int(len(segment)),
        "trades": int(seg_count),
        "net_pnl_usdt": round(seg_net_pnl, 4),
        "net_return_pct": round(seg_net_pnl / starting_capital * 100, 4) if starting_capital else 0.0,
        "win_rate_pct": round(seg_wins / seg_count * 100, 2) if seg_count else 0.0,
        "profit_factor": round(seg_pf, 4) if math.isfinite(seg_pf) else "infinity",
        "max_drawdown_pct": round(seg_max_dd * 100, 4),
    })

if segment_rows:
    segment_report = pd.DataFrame(segment_rows)
    st.dataframe(segment_report, use_container_width=True, hide_index=True)
    positive_segments = sum(1 for row in segment_rows if row["net_pnl_usdt"] > 0)
    st.metric("Segments with positive net P&L", f"{positive_segments} / {len(segment_rows)}")
    st.download_button(
        "Download chronological check CSV",
        data=segment_report.to_csv(index=False).encode("utf-8"),
        file_name="signalforge_chronological_check.csv",
        mime="text/csv",
    )
else:
    st.info("Not enough candles to create the chronological segment report.")
