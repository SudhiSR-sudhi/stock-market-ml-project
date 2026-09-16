import gdown
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

file_id = "https://drive.google.com/file/d/1JdE3hyHG_X1SSeRcnUWc_kVAdwf3SuMw/view?usp=drive_link"

gdown.download(
    f"https://drive.google.com/file/d/1JdE3hyHG_X1SSeRcnUWc_kVAdwf3SuMw/view?usp=drive_link",
    "rf_model.pkl",
    quiet=False
)

model = joblib.load("rf_model.pkl")

print("Model loaded successfully")
# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Reliance Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(90deg, #0b3d2e 0%, #14532d 45%, #ff9933 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.3rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { color: #e8f5e9; margin: 0.3rem 0 0 0; font-size: 0.92rem; }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    section[data-testid="stSidebar"] { background-color: #0b3d2e; }
    section[data-testid="stSidebar"] * { color: #f1f1f1 !important; }

    .stButton>button {
        background-color: #14532d;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { background-color: #0b3d2e; color: #ffe0b3; }

    .result-card {
        background: #f4faf6;
        border-left: 6px solid #14532d;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-top: 1rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0b3d2e;
        border-left: 5px solid #ff9933;
        padding-left: 0.6rem;
        margin: 1.4rem 0 0.6rem 0;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f6f2;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.1rem;
        font-weight: 600;
        color: #0b3d2e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #14532d !important;
        color: #ffffff !important;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        height: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .info-card h4 { margin: 0.6rem 0 0.3rem 0; color: #0b3d2e; }
    .info-card p { margin: 0; color: #555; font-size: 0.9rem; line-height: 1.45; }

    .hero-tagline {
        font-size: 1.05rem;
        color: #333;
        line-height: 1.6;
        margin: 0.6rem 0 1.4rem 0;
    }

    .stat-pill {
        display: inline-block;
        background: #e8f5e9;
        color: #0b3d2e;
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }

    .footer-note { text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# LOAD MODEL, SCALER, METRICS, HISTORICAL DATA
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None
    return model, scaler, metrics


@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("historical_reliance.csv", parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma50"] = df["close"].rolling(50).mean()
        df["daily_return"] = df["close"].pct_change() * 100
        # Bollinger Bands (20-day, 2 std)
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        # RSI (14)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        # Rolling annualised volatility (20-day)
        df["volatility_20"] = df["daily_return"].rolling(20).std() * np.sqrt(252)
        return df
    except FileNotFoundError:
        return None


try:
    model, scaler, metrics = load_model_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python prepare_data.py` then "
        "`python train_model.py --csv historical_reliance.csv` first to "
        "generate `rf_model.pkl`, `scaler.pkl`, and `metrics.json`."
    )
    st.stop()

hist_df = load_historical_data()

DEFAULT_FEATURE_ORDER = ["open", "high", "low", "volume", "oi", "prev_close", "gap"]
FEATURE_LABELS = {
    "open": "Open", "high": "High", "low": "Low", "volume": "Volume",
    "oi": "Open Interest", "prev_close": "Previous Close",
    "gap": "Opening Gap",
}


def resolve_feature_order(model, scaler, metrics):
    """Figure out the exact column order the model/scaler were trained on.

    Getting this wrong is the #1 cause of wildly-off predictions (e.g. a
    predicted close of ₹4,000+ when every input is around ₹2,450): scikit-
    learn's StandardScaler applies (value - mean_i) / std_i per COLUMN
    POSITION, not by name. If the app builds its input vector in a
    different order than train_model.py used when calling scaler.fit(),
    every value gets normalized against the wrong feature's statistics,
    and the model then extrapolates on nonsense input.

    Priority:
      1. scaler.feature_names_in_ — scikit-learn records this automatically
         if the scaler was fit on a pandas DataFrame (the normal way). This
         is ground truth straight from the trained artifact.
      2. model.feature_names_in_ — same idea, if the model itself records it.
      3. metrics.json's "features" list, if train_model.py wrote one.
      4. A hard-coded guess (DEFAULT_FEATURE_ORDER) — last resort, and the
         one most likely to silently disagree with how the model was
         actually trained.
    """
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_), "the scaler's recorded feature names (most reliable)"
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_), "the model's recorded feature names"
    if metrics and "features" in metrics:
        return metrics["features"], "metrics.json"
    return DEFAULT_FEATURE_ORDER, "a hard-coded default guess (⚠️ verify this matches train_model.py)"


FEATURE_ORDER, FEATURE_ORDER_SOURCE = resolve_feature_order(model, scaler, metrics)

n_expected = getattr(scaler, "n_features_in_", len(FEATURE_ORDER))
if n_expected != len(FEATURE_ORDER):
    st.error(
        f"Feature mismatch: the saved scaler was trained on **{n_expected} feature(s)**, "
        f"but this app is configured for **{len(FEATURE_ORDER)}** "
        f"({', '.join(FEATURE_ORDER)}).\n\n"
        "This happens when `rf_model.pkl` / `scaler.pkl` / `metrics.json` came from "
        "different training runs. Fix it by regenerating all three together:\n\n"
        "```\npython prepare_data.py\npython train_model.py --csv historical_reliance.csv\n```"
    )
    st.stop()

if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []


LIVE_TICKER = "RELIANCE.NS"  # NSE ticker suffix used by Yahoo Finance


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_snapshot(ticker: str = LIVE_TICKER):
    """Pull today's live/intraday data from Yahoo Finance.

    Cached for 60 seconds so a burst of reruns/users doesn't hammer the API.
    Returns None (never raises) if data can't be fetched — e.g. market is
    closed with no session data yet, no internet access, or Yahoo is
    rate-limiting — so the caller can fall back gracefully.
    """
    if not YFINANCE_AVAILABLE:
        return None
    try:
        tk = yf.Ticker(ticker)

        intraday = tk.history(period="1d", interval="5m")
        if intraday is None or intraday.empty:
            # Market likely closed right now — fall back to the last few
            # daily sessions so there's still something to plot.
            intraday = tk.history(period="5d", interval="15m")
        if intraday is None or intraday.empty:
            return None

        intraday = intraday.reset_index()
        intraday.rename(columns={intraday.columns[0]: "datetime"}, inplace=True)

        try:
            prev_close = float(tk.fast_info["previousClose"])
        except Exception:
            daily = tk.history(period="5d", interval="1d")
            prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float(intraday["Close"].iloc[0])

        current_price = float(intraday["Close"].iloc[-1])
        day_high = float(intraday["High"].max())
        day_low = float(intraday["Low"].min())
        day_volume = float(intraday["Volume"].sum())

        return {
            "current_price": current_price,
            "prev_close": prev_close,
            "day_high": day_high,
            "day_low": day_low,
            "volume": day_volume,
            "intraday": intraday,
            "fetched_at": pd.Timestamp.now().strftime("%d %b %Y, %I:%M:%S %p"),
        }
    except Exception:
        return None


@st.cache_data
def get_test_predictions():
    """Reproduce the same time-respecting train/test split used in training,
    so we can show honest actual-vs-predicted performance on unseen data."""
    if hist_df is None:
        return None
    df = hist_df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["gap"] = df["open"] - df["prev_close"]
    df = df.dropna(subset=FEATURE_ORDER + ["close"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    if test_df.empty:
        return None

    X_test = test_df.reindex(columns=FEATURE_ORDER).copy()
    for i, f in enumerate(FEATURE_ORDER):
        if f not in test_df.columns or not pd.api.types.is_numeric_dtype(test_df[f]):
            X_test[f] = float(scaler.mean_[i]) if hasattr(scaler, "mean_") else 0.0
    X_test_scaled = scaler.transform(X_test.values)
    test_df["predicted_close"] = model.predict(X_test_scaled)

    tree_preds = np.stack([t.predict(X_test_scaled) for t in model.estimators_], axis=0)
    test_df["pred_std"] = tree_preds.std(axis=0)
    return test_df


@st.cache_data
def get_history_with_features():
    """hist_df augmented with prev_close/gap, used to look up a historical
    day's actual feature values + actual close for the Predict-tab
    validation picker."""
    if hist_df is None:
        return None
    df = hist_df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["gap"] = df["open"] - df["prev_close"]
    return df.dropna(subset=["prev_close"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>📈 Reliance Industries (RELIANCE) — Stock Analytics &amp; Price Predictor</h1>
        <p>Random Forest Regression on NSE OHLCV &amp; Open Interest data, with historical market charts</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.header("📘 About this Project")
st.sidebar.info(
    "A **Random Forest Regressor** predicts RELIANCE's closing price from "
    "engineered features: OHLC, volume, open interest, previous close, "
    "and opening gap."
)
if metrics:
    st.sidebar.markdown("### 🧪 Model Performance (test set)")
    st.sidebar.metric("R² Score", f"{metrics['r2']:.4f}")
    st.sidebar.metric("RMSE (₹)", f"{metrics['rmse']:,.2f}")
    st.sidebar.metric("MAE (₹)", f"{metrics['mae']:,.2f}")
    st.sidebar.caption(f"Trained on {metrics['n_train']:,} rows · Trees: {metrics['n_estimators']} · Max depth: {metrics['max_depth']}")
st.sidebar.markdown("---")
st.sidebar.write("🇮🇳 Model scope: NSE — RELIANCE (Reliance Industries Ltd.)")
st.sidebar.caption("⚠️ For educational purposes only. Not investment advice.")

st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Feature order in use"):
    st.write(" → ".join(FEATURE_ORDER))
    if "hard-coded" in FEATURE_ORDER_SOURCE:
        st.warning(f"Source: {FEATURE_ORDER_SOURCE}")
    else:
        st.caption(f"Source: {FEATURE_ORDER_SOURCE}")
    st.caption(
        "This MUST match the exact column order used when `scaler.fit()` was "
        "called in train_model.py, or predictions will be badly wrong."
    )



# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_home, tab_predict, tab_live, tab_market, tab_insights, tab_log = st.tabs(
    ["🏠 Home", "🔮 Predict", "🔴 Live Market", "📊 Market Overview", "🌲 Model Insights", "🕒 Prediction Log"]
)

# ========================================================================
# TAB 0 — HOME
# ========================================================================
with tab_home:
    st.image("assets/hero_banner.png", width='stretch')

    st.markdown(
        """
        <p class="hero-tagline">
        The <b>Indian stock market</b> is one of the fastest-growing markets in the world,
        anchored by two major exchanges — the <b>NSE</b> (National Stock Exchange) and the
        <b>BSE</b> (Bombay Stock Exchange, Asia's oldest) — and regulated by <b>SEBI</b>
        (Securities and Exchange Board of India). This app focuses on one of its most
        closely watched constituents: <b>Reliance Industries Ltd. (RELIANCE)</b>, the
        largest company by market capitalisation on the NSE and a heavyweight member of
        the Nifty 50 index.
        </p>
        """,
        unsafe_allow_html=True,
    )

    span_badges = ""
    if hist_df is not None:
        span_badges = f"""
        <span class="stat-pill">📅 {hist_df['date'].min().strftime('%b %Y')} – {hist_df['date'].max().strftime('%b %Y')}</span>
        <span class="stat-pill">📈 All-time High ₹{hist_df['high'].max():,.0f}</span>
        <span class="stat-pill">📉 All-time Low ₹{hist_df['low'].min():,.0f}</span>
        <span class="stat-pill">🔁 {len(hist_df):,} trading days</span>
        """
    st.markdown(span_badges, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧭 Know the Market</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image("assets/icon_index.png", width=90)
        st.markdown(
            "<div class='info-card'><h4>Major Indices</h4>"
            "<p><b>Nifty 50</b> (NSE) and <b>Sensex</b> (BSE) track the 50 and 30 largest, "
            "most liquid Indian companies respectively — the two benchmarks most often "
            "quoted as \"the market.\"</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.image("assets/icon_growth.png", width=90)
        st.markdown(
            "<div class='info-card'><h4>Trading Hours</h4>"
            "<p>Normal trading runs <b>9:15 AM – 3:30 PM IST</b>, Monday–Friday, preceded "
            "by a 15-minute pre-open session used for orderly price discovery.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.image("assets/icon_shield.png", width=90)
        st.markdown(
            "<div class='info-card'><h4>Regulator</h4>"
            "<p><b>SEBI</b> oversees exchanges, brokers, and listed companies, enforcing "
            "disclosure norms and investor-protection rules across the market.</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.image("assets/icon_bulb.png", width=90)
        st.markdown(
            "<div class='info-card'><h4>Why Reliance?</h4>"
            "<p>Spanning energy, retail, and telecom (Jio), Reliance is consistently the "
            "single largest weight in the Nifty 50 — a useful bellwether stock.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">🗺️ How This App Is Organised</div>', unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    o1.markdown(
        "<div class='info-card'><h4>🔮 Predict</h4><p>Enter today's Open, High, Low, "
        "Volume, Open Interest, and yesterday's Close to get a Random Forest–predicted "
        "closing price, with a visual range and confidence gauge.</p></div>",
        unsafe_allow_html=True,
    )
    o2.markdown(
        "<div class='info-card'><h4>📊 Market Overview</h4><p>Explore historical price "
        "action with candlesticks, moving averages, Bollinger Bands, RSI, volatility, "
        "and the biggest single-day movers.</p></div>",
        unsafe_allow_html=True,
    )
    o3.markdown(
        "<div class='info-card'><h4>🌲 Model Insights</h4><p>See what drives the model's "
        "predictions, how accurate it is on unseen data, and how confident each "
        "prediction really is.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 Head to the **🔮 Predict** tab above to try the model with your own market data.")

    st.caption(
        "⚠️ Educational project only. Figures and charts here are for learning purposes "
        "and do not constitute investment advice."
    )

# ========================================================================
# TAB 1 — PREDICT
# ========================================================================
with tab_predict:
    st.subheader("Enter Today's Market Data")

    hist_with_feats = get_history_with_features()

    # ------------------------------------------------------------------
    # OPTIONAL: auto-fill inputs from a real historical day, so you can
    # sanity-check the model by comparing its prediction to what actually
    # happened. A well-trained model will land CLOSE to the actual close,
    # not exactly on it — see the note below for why exact equality would
    # actually mean something is wrong (data leakage or memorization).
    # ------------------------------------------------------------------
    selected_actual_close = None
    if hist_with_feats is not None and not hist_with_feats.empty:
        with st.expander("📅 Validate against a real historical day (optional)"):
            st.caption(
                "Pick a past trading day to auto-fill the inputs below with its "
                "real Open/High/Low/Volume/OI/Previous Close, then hit Predict "
                "to compare the model's output against what actually happened. "
                "Expect the prediction to be **close, not identical** — a model "
                "that reproduces the actual close exactly would be memorizing "
                "or leaking data, not genuinely predicting."
            )
            date_options = hist_with_feats["date"].dt.strftime("%Y-%m-%d").tolist()
            picked_date = st.selectbox(
                "Historical date", options=list(reversed(date_options)), index=0,
                key="hist_date_picker",
            )
            if st.button("⬇️ Load this day's data into the form"):
                row = hist_with_feats.loc[
                    hist_with_feats["date"].dt.strftime("%Y-%m-%d") == picked_date
                ].iloc[0]
                st.session_state["open_input"] = float(row["open"])
                st.session_state["high_input"] = float(row["high"])
                st.session_state["low_input"] = float(row["low"])
                st.session_state["prev_close_input"] = float(row["prev_close"])
                st.session_state["volume_input"] = int(row["volume"])
                st.session_state["oi_input"] = int(row["oi"])
                st.session_state["validation_actual_close"] = float(row["close"])
                st.session_state["validation_date"] = picked_date
                st.rerun()

    if "validation_actual_close" in st.session_state:
        selected_actual_close = st.session_state["validation_actual_close"]
        st.info(
            f"📌 Loaded **{st.session_state.get('validation_date')}** — actual close that "
            f"day was **₹{selected_actual_close:,.2f}**. Inputs below are pre-filled from "
            "that day; hit Predict to compare."
        )

    default_prev_close = float(hist_df["close"].iloc[-1]) if hist_df is not None else 2440.0

    col1, col2 = st.columns(2)
    with col1:
        open_price = st.number_input("Open Price (₹)", min_value=0.0, max_value=10000.0,
                                      step=0.5, key="open_input",
                                      value=st.session_state.get("open_input", 2450.00))
        high_price = st.number_input("High Price (₹)", min_value=0.0, max_value=10000.0,
                                      step=0.5, key="high_input",
                                      value=st.session_state.get("high_input", 2475.00))
        low_price = st.number_input("Low Price (₹)", min_value=0.0, max_value=10000.0,
                                     step=0.5, key="low_input",
                                     value=st.session_state.get("low_input", 2430.00))
        prev_close = st.number_input("Previous Day's Close (₹)", min_value=0.0, max_value=10000.0,
                                      step=0.5, key="prev_close_input",
                                      value=st.session_state.get("prev_close_input", default_prev_close))
    with col2:
        volume = st.number_input("Volume (shares traded)", min_value=0, max_value=500_000_000,
                                  step=100_000, key="volume_input",
                                  value=st.session_state.get("volume_input", 8_000_000))
        oi = st.number_input("Open Interest (OI)", min_value=0, max_value=500_000_000,
                              step=100_000, key="oi_input",
                              value=st.session_state.get("oi_input", 45_000_000))

    if high_price < max(open_price, low_price) or low_price > min(open_price, high_price):
        st.warning("Check your inputs: High should be ≥ Open/Low, and Low should be ≤ Open/High.")

    gap = open_price - prev_close

    st.metric("Opening Gap (₹)", f"{gap:,.2f}")

    if st.button("🔍 Predict Closing Price"):
        feature_values = {
            "open": open_price, "high": high_price, "low": low_price,
            "volume": volume, "oi": oi, "prev_close": prev_close,
            "gap": gap,
        }
        missing_features = [f for f in FEATURE_ORDER if f not in feature_values]
        if missing_features:
            if hasattr(scaler, "mean_"):
                # Auto-fill features the form doesn't collect (e.g. `symbol`) with
                # their trained mean. Safe here because such columns are constant
                # across a single-stock dataset and carry no predictive signal —
                # StandardScaler will map the mean to 0 for that column either way.
                for i, f in enumerate(FEATURE_ORDER):
                    if f in missing_features:
                        feature_values[f] = float(scaler.mean_[i])
                st.caption(
                    f"ℹ️ Auto-filled `{', '.join(missing_features)}` using the trained "
                    "average, since this form doesn't collect it and it's constant for "
                    "a single-stock model."
                )
            else:
                st.error(
                    f"The trained model expects feature(s) {missing_features} that this "
                    "form doesn't collect, and the scaler has no recorded mean to safely "
                    "fill them with. Retrain with the current `train_model.py` so the "
                    "feature set matches the app, or update the input form to match your "
                    "custom feature list."
                )
                st.stop()

        input_data = np.array([[feature_values[f] for f in FEATURE_ORDER]])
        scaled_input = scaler.transform(input_data)
        predicted_close = model.predict(scaled_input)[0]

        tree_preds = np.array([t.predict(scaled_input)[0] for t in model.estimators_])
        pred_std = tree_preds.std()

        change = predicted_close - open_price
        pct_change = (change / open_price) * 100 if open_price else 0
        change_vs_prev = predicted_close - prev_close
        pct_change_vs_prev = (change_vs_prev / prev_close) * 100 if prev_close else 0

        # ------------------------------------------------------------
        # Plausibility check. A single day's close almost never strays far
        # outside that day's own Open/High/Low/PrevClose band. If it does,
        # that's a strong signal of a feature-order/scale mismatch between
        # training and inference (see the sidebar's "Feature order in use"),
        # not a real market move — so we say so loudly instead of quietly
        # presenting a bad number as if it were a confident forecast.
        # ------------------------------------------------------------
        band_low = min(open_price, low_price, prev_close) * 0.85
        band_high = max(open_price, high_price, prev_close) * 1.15
        implausible = not (band_low <= predicted_close <= band_high)

        if implausible:
            # Fallback: a transparent formula-based estimate, used only because
            # the trained model's output failed the plausibility check above.
            # This is NOT the Random Forest — it's a simple weighted average of
            # today's own inputs, so it always lands in a sane range by
            # construction. Retrain train_model.py (excluding `symbol` from its
            # feature set) to fix the real model and remove the need for this.
            fallback_close = (
                0.4 * open_price + 0.25 * high_price + 0.25 * low_price + 0.1 * prev_close
            )
            st.warning(
                f"⚠️ The trained model's prediction (₹{predicted_close:,.2f}) fell far "
                f"outside today's plausible range (₹{band_low:,.2f}–₹{band_high:,.2f}), "
                "so it's being replaced below with a simple formula-based estimate. "
                "This is **not** the Random Forest model — it's a placeholder until "
                "`rf_model.pkl` / `scaler.pkl` are retrained correctly (see the "
                "**🔧 Feature order in use** panel)."
            )
            predicted_close = fallback_close
            pred_std = 0.0

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"🏦 Predicted Closing Price: **₹{predicted_close:,.2f}** ± ₹{pred_std:,.2f} (1 std across trees)")
        st.markdown("</div>", unsafe_allow_html=True)

        # If this prediction was run against a loaded historical day, show
        # Predicted vs Actual side-by-side — this is the honest version of
        # "check it against the real number" that doesn't fake exactness.
        if selected_actual_close is not None:
            err = predicted_close - selected_actual_close
            pct_err = (err / selected_actual_close) * 100 if selected_actual_close else 0
            v1, v2, v3 = st.columns(3)
            v1.metric("Predicted Close (₹)", f"{predicted_close:,.2f}")
            v2.metric("Actual Close (₹)", f"{selected_actual_close:,.2f}")
            v3.metric("Prediction Error", f"{err:,.2f}", delta=f"{pct_err:.2f}%")
            st.caption(
                "A small non-zero error here is expected and healthy — it shows the model "
                "is genuinely estimating rather than having memorized or leaked the answer. "
                "The sidebar's R² / RMSE / MAE summarize this error across many such days."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Close (₹)", f"{predicted_close:,.2f}")
        m2.metric("vs Open", f"{change:,.2f}", delta=f"{pct_change:.2f}%")
        m3.metric("vs Previous Close", f"{change_vs_prev:,.2f}", delta=f"{pct_change_vs_prev:.2f}%")
        m4.metric("Predicted Range Position", f"{((predicted_close-low_price)/(high_price-low_price)*100 if high_price>low_price else 50):.0f}%")

        if predicted_close > open_price:
            st.info("📈 Model expects the stock to close **higher** than the opening price.")
        elif predicted_close < open_price:
            st.info("📉 Model expects the stock to close **lower** than the opening price.")
        else:
            st.info("➡️ Model expects the stock to close roughly **flat**.")

        st.markdown('<div class="section-title">📊 Today\'s Range vs Prediction</div>', unsafe_allow_html=True)
        fig_ohlc = go.Figure()
        fig_ohlc.add_trace(go.Candlestick(
            x=["Today"], open=[open_price], high=[high_price], low=[low_price], close=[predicted_close],
            increasing_line_color="#14532d", decreasing_line_color="#c0392b", name="OHLC",
        ))
        fig_ohlc.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig_ohlc, width='stretch')

        st.markdown('<div class="section-title">🎯 Predicted Close Within Today\'s Range</div>', unsafe_allow_html=True)
        gauge_min = min(low_price, predicted_close) * 0.998
        gauge_max = max(high_price, predicted_close) * 1.002
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=predicted_close,
            delta={"reference": open_price, "increasing": {"color": "#14532d"}, "decreasing": {"color": "#c0392b"}},
            number={"prefix": "₹", "valueformat": ",.2f"},
            gauge={
                "axis": {"range": [gauge_min, gauge_max]},
                "bar": {"color": "#ff9933"},
                "steps": [
                    {"range": [gauge_min, open_price], "color": "#fdecea"},
                    {"range": [open_price, gauge_max], "color": "#e8f5e9"},
                ],
                "threshold": {"line": {"color": "#0b3d2e", "width": 4}, "thickness": 0.8, "value": open_price},
            },
        ))
        fig_gauge.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(fig_gauge, width='stretch')

        st.session_state.prediction_log.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "open": open_price, "high": high_price, "low": low_price,
            "prev_close": prev_close, "volume": volume, "oi": oi,
            "predicted_close": round(float(predicted_close), 2),
            "uncertainty": round(float(pred_std), 2),
            "actual_close": round(float(selected_actual_close), 2) if selected_actual_close is not None else None,
        })
        st.caption("✅ Saved to Prediction Log tab.")

# ========================================================================
# TAB 2 — LIVE MARKET
# ========================================================================
with tab_live:
    st.subheader("🔴 Live Market Snapshot — RELIANCE.NS")

    if not YFINANCE_AVAILABLE:
        st.warning(
            "`yfinance` isn't installed, so live data can't be shown. Add "
            "`yfinance` to `requirements.txt` and reinstall to enable this tab."
        )
    else:
        refresh_col, _ = st.columns([1, 5])
        with refresh_col:
            if st.button("🔄 Refresh live data"):
                fetch_live_snapshot.clear()

        live = fetch_live_snapshot()

        if live is None:
            st.info(
                "Couldn't fetch live data right now — the market may be closed, "
                "or the data provider is temporarily unreachable. Try the "
                "**🔄 Refresh live data** button above, or check back during "
                "market hours (9:15 AM – 3:30 PM IST, Mon–Fri)."
            )
        else:
            change = live["current_price"] - live["prev_close"]
            pct_change = (change / live["prev_close"]) * 100 if live["prev_close"] else 0

            lc1, lc2, lc3, lc4, lc5 = st.columns(5)
            lc1.metric("Current Price (₹)", f"{live['current_price']:,.2f}",
                       delta=f"{change:,.2f} ({pct_change:.2f}%)")
            lc2.metric("Previous Close (₹)", f"{live['prev_close']:,.2f}")
            lc3.metric("Day High (₹)", f"{live['day_high']:,.2f}")
            lc4.metric("Day Low (₹)", f"{live['day_low']:,.2f}")
            lc5.metric("Volume (today)", f"{live['volume']:,.0f}")
            st.caption(
                f"Last updated: {live['fetched_at']} · Source: Yahoo Finance (yfinance) · "
                "Data auto-refreshes at most once every 60 seconds."
            )

            li1, li2 = st.columns([2, 1])
            with li1:
                fig_intraday = go.Figure()
                fig_intraday.add_trace(go.Scatter(
                    x=live["intraday"]["datetime"], y=live["intraday"]["Close"],
                    line=dict(color="#14532d", width=2), fill="tozeroy",
                    fillcolor="rgba(20,83,45,0.08)", name="Price",
                ))
                fig_intraday.add_hline(y=live["prev_close"], line_dash="dash",
                                        line_color="#ff9933", annotation_text="Prev Close")
                fig_intraday.update_layout(
                    template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10),
                    title="Today's Intraday Price", yaxis_title="Price (₹)",
                )
                st.plotly_chart(fig_intraday, width='stretch')
            with li2:
                gauge_min = min(live["day_low"], live["prev_close"]) * 0.999
                gauge_max = max(live["day_high"], live["prev_close"]) * 1.001
                fig_live_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=live["current_price"],
                    delta={"reference": live["prev_close"],
                           "increasing": {"color": "#14532d"}, "decreasing": {"color": "#c0392b"}},
                    number={"prefix": "₹", "valueformat": ",.2f"},
                    gauge={
                        "axis": {"range": [gauge_min, gauge_max]},
                        "bar": {"color": "#ff9933"},
                        "steps": [
                            {"range": [gauge_min, live["prev_close"]], "color": "#fdecea"},
                            {"range": [live["prev_close"], gauge_max], "color": "#e8f5e9"},
                        ],
                        "threshold": {"line": {"color": "#0b3d2e", "width": 4}, "thickness": 0.8,
                                      "value": live["prev_close"]},
                    },
                ))
                fig_live_gauge.update_layout(
                    template="plotly_white", height=340, margin=dict(l=20, r=20, t=40, b=10),
                    title="Price vs Previous Close",
                )
                st.plotly_chart(fig_live_gauge, width='stretch')

            if hist_df is not None:
                last_ma20 = hist_df["ma20"].iloc[-1]
                last_ma50 = hist_df["ma50"].iloc[-1]
                if pd.notna(last_ma20) and pd.notna(last_ma50):
                    st.markdown('<div class="section-title">📍 Current Price vs Moving Averages</div>',
                                unsafe_allow_html=True)
                    compare_vals = [live["current_price"], last_ma20, last_ma50]
                    fig_ma_compare = go.Figure(go.Bar(
                        x=["Current Price", "20-day MA", "50-day MA"],
                        y=compare_vals,
                        marker_color=["#ff9933", "#14532d", "#1f6feb"],
                        text=[f"₹{v:,.2f}" for v in compare_vals],
                        textposition="outside",
                    ))
                    fig_ma_compare.update_layout(
                        template="plotly_white", height=320, margin=dict(l=10, r=10, t=20, b=10),
                    )
                    st.plotly_chart(fig_ma_compare, width='stretch')
                    if live["current_price"] > last_ma20 > last_ma50:
                        st.success("📈 Price is above both moving averages — short and medium-term trend look bullish.")
                    elif live["current_price"] < last_ma20 < last_ma50:
                        st.error("📉 Price is below both moving averages — short and medium-term trend look bearish.")
                    else:
                        st.info("➡️ Price is mixed relative to its moving averages — no clear trend alignment.")
                else:
                    st.caption(
                        "Not enough historical rows to compute 20/50-day moving averages yet, "
                        "so the comparison chart above is skipped."
                    )
            else:
                st.caption(
                    "No `historical_reliance.csv` found, so the current-price-vs-moving-average "
                    "comparison is skipped — that chart needs historical data to compute MA20/MA50."
                )

# ========================================================================
# TAB 3 — MARKET OVERVIEW (historical charts)
# ========================================================================
with tab_market:
    if hist_df is None:
        st.warning(
            "No historical data found. Run `python prepare_data.py` to generate "
            "`historical_reliance.csv`, or drop your own file with the same name "
            "and columns (date, open, high, low, close, volume, oi) next to app.py."
        )
    else:
        st.subheader("RELIANCE — Historical Price Action")

        min_d, max_d = hist_df["date"].min().date(), hist_df["date"].max().date()
        date_range = st.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d))
        view = hist_df[(hist_df["date"].dt.date >= date_range[0]) & (hist_df["date"].dt.date <= date_range[1])].copy()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Latest Close (₹)", f"{view['close'].iloc[-1]:,.2f}")
        k2.metric("Period High (₹)", f"{view['high'].max():,.2f}")
        k3.metric("Period Low (₹)", f"{view['low'].min():,.2f}")
        k4.metric("Avg Volume", f"{view['volume'].mean():,.0f}")
        period_return = (view['close'].iloc[-1] / view['close'].iloc[0] - 1) * 100
        k5.metric("Period Return", f"{period_return:.2f}%")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
            vertical_spacing=0.04,
        )
        fig.add_trace(go.Scatter(x=view["date"], y=view["bb_upper"], name="Bollinger Upper",
                                  line=dict(color="rgba(11,61,46,0.25)", width=1),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=view["date"], y=view["bb_lower"], name="Bollinger Band",
                                  line=dict(color="rgba(11,61,46,0.25)", width=1),
                                  fill="tonexty", fillcolor="rgba(11,61,46,0.06)"), row=1, col=1)
        fig.add_trace(go.Candlestick(
            x=view["date"], open=view["open"], high=view["high"], low=view["low"], close=view["close"],
            increasing_line_color="#14532d", decreasing_line_color="#c0392b", name="Price",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=view["date"], y=view["ma20"], name="MA 20", line=dict(color="#ff9933", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=view["date"], y=view["ma50"], name="MA 50", line=dict(color="#1f6feb", width=1.5)), row=1, col=1)

        vol_colors = np.where(view["close"] >= view["open"], "#14532d", "#c0392b")
        fig.add_trace(go.Bar(x=view["date"], y=view["volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)

        fig.update_layout(
            template="plotly_white", height=580, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        st.plotly_chart(fig, width='stretch')
        st.caption("Shaded band = Bollinger Bands (20-day MA ± 2 std) — a common measure of relative price extremes.")

        st.markdown('<div class="section-title">⚡ Momentum — RSI (14-day)</div>', unsafe_allow_html=True)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=view["date"], y=view["rsi_14"], line=dict(color="#0b3d2e", width=1.8), name="RSI 14"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#c0392b", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#14532d", annotation_text="Oversold (30)")
        fig_rsi.update_layout(template="plotly_white", height=260, margin=dict(l=10, r=10, t=20, b=10),
                               yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=view["date"], y=view["volatility_20"], line=dict(color="#c0392b"), fill="tozeroy",
                                          fillcolor="rgba(192,57,43,0.08)", name="Volatility"))
            fig_vol.update_layout(title="Annualised 20-day Volatility (%)", template="plotly_white",
                                   height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_vol, width='stretch')
        with c2:
            monthly = view.set_index("date")["close"].resample("ME").last().pct_change().dropna() * 100
            colors = ["#14532d" if v >= 0 else "#c0392b" for v in monthly.values]
            fig_month = go.Figure(go.Bar(x=monthly.index.strftime("%b %Y"), y=monthly.values, marker_color=colors))
            fig_month.update_layout(title="Monthly Returns (%)", template="plotly_white",
                                     height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_month, width='stretch')

        st.markdown('<div class="section-title">🚀 Biggest Single-Day Movers</div>', unsafe_allow_html=True)
        movers = view.dropna(subset=["daily_return"]).copy()
        top_gainers = movers.nlargest(5, "daily_return")[["date", "close", "daily_return"]]
        top_losers = movers.nsmallest(5, "daily_return")[["date", "close", "daily_return"]]
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**📈 Top Gainers**")
            st.dataframe(
                top_gainers.assign(date=top_gainers["date"].dt.strftime("%d %b %Y")).rename(
                    columns={"close": "Close (₹)", "daily_return": "Return (%)"}
                ).round(2), width='stretch', hide_index=True,
            )
        with g2:
            st.markdown("**📉 Top Losers**")
            st.dataframe(
                top_losers.assign(date=top_losers["date"].dt.strftime("%d %b %Y")).rename(
                    columns={"close": "Close (₹)", "daily_return": "Return (%)"}
                ).round(2), width='stretch', hide_index=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            fig_ret = go.Figure(go.Histogram(x=view["daily_return"].dropna(), nbinsx=40, marker_color="#14532d"))
            fig_ret.update_layout(title="Daily Return Distribution (%)", template="plotly_white",
                                   height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_ret, width='stretch')
        with c4:
            fig_oi = go.Figure()
            fig_oi.add_trace(go.Scatter(x=view["date"], y=view["oi"], line=dict(color="#0b3d2e"), name="Open Interest"))
            fig_oi.update_layout(title="Open Interest Over Time", template="plotly_white",
                                  height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_oi, width='stretch')

# ========================================================================
# TAB 4 — MODEL INSIGHTS
# ========================================================================
with tab_insights:
    st.subheader("🌲 How the Random Forest Works")
    e1, e2, e3 = st.columns(3)
    e1.markdown(
        "<div class='info-card'><h4>1. Many Trees</h4><p>The model builds "
        f"{metrics['n_estimators'] if metrics else '300'} independent decision trees, "
        "each trained on a random subset of the historical data.</p></div>",
        unsafe_allow_html=True,
    )
    e2.markdown(
        "<div class='info-card'><h4>2. Each Tree Votes</h4><p>Every tree makes its own "
        "closing-price estimate by following its own learned decision rules on your "
        "inputs.</p></div>",
        unsafe_allow_html=True,
    )
    e3.markdown(
        "<div class='info-card'><h4>3. Average = Prediction</h4><p>The forest's final "
        "prediction is the average across all trees — the spread between trees is what "
        "powers the uncertainty (±) shown on each prediction.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">🎯 Feature Importance</div>', unsafe_allow_html=True)
    importances = model.feature_importances_
    labels = [FEATURE_LABELS.get(f, f.replace("_", " ").title()) for f in FEATURE_ORDER]
    order = np.argsort(importances)

    fi1, fi2 = st.columns(2)
    with fi1:
        fig_imp = go.Figure(go.Bar(
            x=importances[order], y=[labels[i] for i in order], orientation="h",
            marker=dict(color=importances[order], colorscale=[[0, "#e8f5e9"], [1, "#14532d"]]),
        ))
        fig_imp.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10),
                               xaxis_title="Relative importance", title="By Feature")
        st.plotly_chart(fig_imp, width='stretch')
    with fi2:
        sorted_desc = importances[np.argsort(importances)[::-1]]
        cum_imp = np.cumsum(sorted_desc)
        sorted_labels = [labels[i] for i in np.argsort(importances)[::-1]]
        fig_cum = go.Figure(go.Scatter(x=sorted_labels, y=cum_imp * 100, mode="lines+markers",
                                        line=dict(color="#ff9933", width=2.5), marker=dict(size=8)))
        fig_cum.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10),
                               yaxis_title="Cumulative importance (%)", title="Cumulative")
        st.plotly_chart(fig_cum, width='stretch')

    if metrics:
        st.markdown('<div class="section-title">🧪 Test-Set Performance</div>', unsafe_allow_html=True)
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("R² Score", f"{metrics['r2']:.4f}")
        mcol2.metric("RMSE (₹)", f"{metrics['rmse']:,.2f}")
        mcol3.metric("MAE (₹)", f"{metrics['mae']:,.2f}")

    test_df = get_test_predictions()
    if test_df is not None:
        st.markdown('<div class="section-title">🔬 Actual vs. Predicted (Held-Out Test Data)</div>', unsafe_allow_html=True)
        ap1, ap2 = st.columns(2)
        with ap1:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=test_df["close"], y=test_df["predicted_close"], mode="markers",
                marker=dict(color="#14532d", size=7, opacity=0.65), name="Predictions",
            ))
            lo = min(test_df["close"].min(), test_df["predicted_close"].min())
            hi = max(test_df["close"].max(), test_df["predicted_close"].max())
            fig_scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                              line=dict(color="#ff9933", dash="dash"), name="Perfect prediction"))
            fig_scatter.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10),
                                       xaxis_title="Actual Close (₹)", yaxis_title="Predicted Close (₹)",
                                       title="Predicted vs Actual")
            st.plotly_chart(fig_scatter, width='stretch')
        with ap2:
            residuals = test_df["close"] - test_df["predicted_close"]
            fig_resid = go.Figure(go.Scatter(
                x=test_df["predicted_close"], y=residuals, mode="markers",
                marker=dict(color="#c0392b", size=7, opacity=0.65),
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color="#0b3d2e")
            fig_resid.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10),
                                     xaxis_title="Predicted Close (₹)", yaxis_title="Residual (Actual − Predicted)",
                                     title="Residuals")
            st.plotly_chart(fig_resid, width='stretch')

        st.markdown('<div class="section-title">📏 Prediction Confidence Over Time</div>', unsafe_allow_html=True)
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=test_df["date"], y=test_df["predicted_close"] + test_df["pred_std"],
            line=dict(width=0), showlegend=False,
        ))
        fig_conf.add_trace(go.Scatter(
            x=test_df["date"], y=test_df["predicted_close"] - test_df["pred_std"],
            line=dict(width=0), fill="tonexty", fillcolor="rgba(255,153,51,0.2)",
            name="±1 std (tree spread)",
        ))
        fig_conf.add_trace(go.Scatter(x=test_df["date"], y=test_df["predicted_close"],
                                       line=dict(color="#ff9933", width=2), name="Predicted"))
        fig_conf.add_trace(go.Scatter(x=test_df["date"], y=test_df["close"],
                                       line=dict(color="#14532d", width=2), name="Actual"))
        fig_conf.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=20, b=10),
                                yaxis_title="Close (₹)")
        st.plotly_chart(fig_conf, width='stretch')
        st.caption(
            "The shaded band shows how much individual trees in the forest disagree with "
            "each other — a wider band means the model is less confident for that day."
        )

    if hist_df is not None:
        st.markdown('<div class="section-title">🔗 Feature Correlation with Closing Price</div>', unsafe_allow_html=True)
        corr_df = hist_df.copy()
        corr_df["prev_close"] = corr_df["close"].shift(1)
        corr_df["gap"] = corr_df["open"] - corr_df["prev_close"]
        corr_cols = ["open", "high", "low", "volume", "oi", "prev_close", "gap", "close"]
        corr = corr_df[corr_cols].dropna().corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#c0392b"], [0.5, "#ffffff"], [1, "#14532d"]],
            zmin=-1, zmax=1, colorbar=dict(title="corr"),
        ))
        fig_corr.update_layout(template="plotly_white", height=440, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_corr, width='stretch')

# ========================================================================
# TAB 5 — PREDICTION LOG
# ========================================================================
with tab_log:
    st.subheader("Your Prediction History (this session)")
    if not st.session_state.prediction_log:
        st.info("No predictions yet — make one in the **Predict** tab and it'll show up here.")
    else:
        log_df = pd.DataFrame(st.session_state.prediction_log)
        st.dataframe(log_df, width='stretch')

        fig_log = go.Figure()
        if "uncertainty" in log_df.columns:
            fig_log.add_trace(go.Scatter(
                x=log_df["timestamp"], y=log_df["predicted_close"] + log_df["uncertainty"],
                line=dict(width=0), showlegend=False,
            ))
            fig_log.add_trace(go.Scatter(
                x=log_df["timestamp"], y=log_df["predicted_close"] - log_df["uncertainty"],
                line=dict(width=0), fill="tonexty", fillcolor="rgba(20,83,45,0.12)",
                name="±1 std",
            ))
        fig_log.add_trace(go.Scatter(
            x=log_df["timestamp"], y=log_df["predicted_close"],
            mode="lines+markers", line=dict(color="#14532d"), name="Predicted Close",
        ))
        if "actual_close" in log_df.columns and log_df["actual_close"].notna().any():
            fig_log.add_trace(go.Scatter(
                x=log_df["timestamp"], y=log_df["actual_close"],
                mode="lines+markers", line=dict(color="#c0392b", dash="dot"), name="Actual Close (when known)",
            ))
        fig_log.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10),
                               title="Predicted Close Over Your Session")
        st.plotly_chart(fig_log, width='stretch')

        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download prediction log (CSV)", data=csv_bytes,
                            file_name="reliance_prediction_log.csv", mime="text/csv")

        if st.button("🗑️ Clear log"):
            st.session_state.prediction_log = []
            st.rerun()

st.markdown(
    """
    <div class="footer-note">
        Built with Streamlit, scikit-learn, Plotly &amp; joblib · Random Forest Regressor ·
        Data scope: NSE RELIANCE · For educational purposes only — not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
