# src/features.py
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import ta  # type: ignore
from loguru import logger  # type: ignore
from src.config import FeatureCfg, MtaCfg, InterMarketCfg, PriceActionCfg  # Import FeatureConfig from src.config

_TF_DELTA = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}

# Columns that carry an absolute price level (or a cumulative sum that depends on where the window starts). They stay in `X`: the
# stops and trailing read `atr_14`, and the contextual bandit reads `macd_diff`. The model gets the relative versions only (C5).
RAW_LEVEL_COLUMNS = frozenset({
    "macd", "macd_signal", "macd_diff", "momentum_5", "momentum_10", "atr_14", "vol_ma_20", "obv",
    "donchian_h", "donchian_l", "donchian_m", "ema_fast", "ema_slow", "bb_high", "bb_low", "macd_x_adx",
})


def is_model_column(name: str) -> bool:
    if name in RAW_LEVEL_COLUMNS:
        return False
    return not (name.startswith("mta_ema_") and not name.startswith("mta_ema_dist_"))   # the raw higher-timeframe ema is a price


def model_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """The columns the models are trained and asked on: `X` without the absolute price levels."""
    return X[[c for c in X.columns if is_model_column(c)]]


EMA_DIST_SPAN = 200


def ema_distance(close: pd.Series, span: int = EMA_DIST_SPAN) -> pd.Series:
    """(close - ema) / close, the risk context's `dist_from_ema_200`. It is not a model column, so it changes no saved model.
    The ema uses bars <= t only; it needs `span` bars and an unconverged start is NaN, so give it the whole history."""
    ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
    return ((close - ema) / close).replace([np.inf, -np.inf], np.nan)


def ema_distance_at(close: pd.Series, when, span: int = EMA_DIST_SPAN) -> float:
    """`ema_distance` at the bar stamped `when`; 0.0 when it is missing or not finite."""
    value = ema_distance(close, span).get(when, np.nan)
    if isinstance(value, pd.Series):   # a repeated stamp: the last one
        value = value.iloc[-1]
    return float(value) if np.isfinite(value) else 0.0


def add_contextual_features(df: pd.DataFrame, mta_df: pd.DataFrame = None, inter_market_df: pd.DataFrame = None, mta_cfg: "MtaCfg" = None, im_cfg: "InterMarketCfg" = None, relative: bool = False) -> pd.DataFrame:
    """
    Adds contextual features from higher timeframes (MTA) and other markets. `relative` also adds the MTA ema as a distance from
    the higher-timeframe close, which is what the model reads (C5); the research scripts keep their columns with the default.
    """
    if mta_df is not None and mta_cfg and mta_cfg.enabled:
        logger.debug(f"Adding MTA features from timeframe {mta_cfg.timeframe}...")
        # Calculate indicators on MTA dataframe
        mta_ema = ta.trend.ema_indicator(mta_df["close"], window=mta_cfg.ema_period)
        mta_rsi = ta.momentum.rsi(mta_df["close"], window=mta_cfg.rsi_period)

        # Create a dataframe for these features
        mta_features = pd.DataFrame(index=mta_df.index)
        mta_features[f'mta_ema_{mta_cfg.ema_period}'] = mta_ema
        mta_features[f'mta_rsi_{mta_cfg.rsi_period}'] = mta_rsi
        if relative:
            mta_features[f'mta_ema_dist_{mta_cfg.ema_period}'] = (mta_df["close"] - mta_ema) / mta_df["close"]   # the model's version of the ema

        # MT5 indexes bars by OPEN time, but a bar's close is only known one bar-length later.
        # Shift to availability time and as-of join so no bar sees its own higher-timeframe bar's unfinished close.
        mta_features.index = mta_features.index + pd.Timedelta(_TF_DELTA[mta_cfg.timeframe])
        df = pd.merge_asof(df.sort_index(), mta_features.sort_index(), left_index=True, right_index=True, direction="backward")

    if inter_market_df is not None and im_cfg and im_cfg.enabled:
        logger.debug(f"Adding Inter-Market features from symbol {im_cfg.symbol}...")
        im_features = pd.DataFrame(index=inter_market_df.index)
        for lag in im_cfg.roc_lags:
            im_features[f'im_{im_cfg.symbol}_roc_{lag}'] = inter_market_df["close"].pct_change(lag)

        # Align with the primary dataframe
        df = pd.merge(df, im_features, left_index=True, right_index=True, how='left')
        df.ffill(inplace=True)

    return df


def build_static_features(df: pd.DataFrame, symbol: str = None, pa_cfg: "PriceActionCfg" = None) -> pd.DataFrame:
    """
    Builds features that do not depend on tunable hyperparameters.
    These can be calculated once and cached.
    """
    logger.debug(f"[{symbol}] Building static features...")
    X = pd.DataFrame(index=df.index)

    # --- MACD ---
    macd = ta.trend.MACD(df["close"])
    X["macd"] = macd.macd()
    X["macd_signal"] = macd.macd_signal()
    X["macd_diff"] = macd.macd_diff()

    # --- Fixed Momentum ---
    X["momentum_5"] = df["close"] - df["close"].shift(5)
    X["momentum_10"] = df["close"] - df["close"].shift(10)

    # --- Fixed Volatility ---
    X["atr_14"] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    X["volatility_10"] = df["close"].pct_change().rolling(10).std()
    X["volatility_20"] = df["close"].pct_change().rolling(20).std()

    # --- Volume Features (if available) ---
    if "volume" in df.columns:
        X["vol_ma_20"] = df["volume"].rolling(20).mean()
        X["vol_ratio"] = df["volume"] / (X["vol_ma_20"] + 1e-10)

    # --- Fractal Features ---
    # Confirmed 5-bar fractals: the centre bar is t-2, so the pattern is only known at bar t (no look-ahead).
    hi_c, lo_c = df["high"].shift(2), df["low"].shift(2)
    X["fractal_up"] = ((hi_c > df["high"].shift(4)) & (hi_c > df["high"].shift(3)) & (hi_c > df["high"].shift(1)) & (hi_c > df["high"])).astype(int)
    X["fractal_down"] = ((lo_c < df["low"].shift(4)) & (lo_c < df["low"].shift(3)) & (lo_c < df["low"].shift(1)) & (lo_c < df["low"])).astype(int)

    # --- Rolling Statistics ---
    X["ret_skew_10"] = df["close"].pct_change().rolling(10).skew()
    X["ret_kurt_10"] = df["close"].pct_change().rolling(10).kurt()

    # --- Time-based Features ---
    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # --- Advanced Price Action Features ---
    if pa_cfg and pa_cfg.enabled:
        # Distance from 'Home Base' MA
        home_base_ma = ta.trend.ema_indicator(df["close"], window=pa_cfg.home_base_ma_period)
        X[f'dist_from_ema_{pa_cfg.home_base_ma_period}'] = (df["close"] - home_base_ma) / home_base_ma

        # Time since N-bar high/low
        rolling_high = df["high"].rolling(window=pa_cfg.swing_lookback).max()
        rolling_low = df["low"].rolling(window=pa_cfg.swing_lookback).min()

        is_new_high = df["high"] == rolling_high
        is_new_low = df["low"] == rolling_low

        # Cumulatively count bars since the last event
        X['bars_since_high'] = is_new_high.cumsum().groupby((is_new_high).cumsum()).cumcount()
        X['bars_since_low'] = is_new_low.cumsum().groupby((is_new_low).cumsum()).cumcount()

    # --- New Momentum Indicators ---
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    X["stoch"] = stoch.stoch()
    X["stoch_signal"] = stoch.stoch_signal()
    X["willr"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()

    # --- New Volume Indicators ---
    if "volume" in df.columns:
        X["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        X["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=20).chaikin_money_flow()

    # --- New Volatility Indicators ---
    donchian = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"], window=20)
    X["donchian_h"] = donchian.donchian_channel_hband()
    X["donchian_l"] = donchian.donchian_channel_lband()
    X["donchian_m"] = donchian.donchian_channel_mband()

    return X


def build_dynamic_features(df: pd.DataFrame, static_features: pd.DataFrame, cfg: FeatureCfg, symbol: str = None) -> pd.DataFrame:
    """
    Builds features that depend on tunable hyperparameters, using pre-calculated static features.
    """
    X = static_features.copy()

    try:
        # --- Price/Momentum Features (Dynamic) ---
        X["rsi"] = ta.momentum.rsi(df["close"], window=cfg.rsi_period)
        X["ema_fast"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_fast)
        X["ema_slow"] = ta.trend.ema_indicator(df["close"], window=cfg.ema_slow)
        X["ema_diff"] = (X["ema_fast"] - X["ema_slow"]) / df["close"]

        for l in cfg.roc_lags:
            X[f"ret_{l}"] = df["close"].pct_change(l)

        # --- Volatility Features (Dynamic) ---
        bb = ta.volatility.BollingerBands(df["close"], window=cfg.window_vol, window_dev=2)
        X["bb_high"] = bb.bollinger_hband()
        X["bb_low"] = bb.bollinger_lband()
        X["bb_width"] = (X["bb_high"] - X["bb_low"]) / df["close"]

        # --- Regime Detection & Mean-Reversion Features (Dynamic) ---
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=cfg.adx_period)
        X["adx"] = adx_indicator.adx()
        X["is_trending"] = (X["adx"] > cfg.adx_trend_thresh).astype(int)

        X["rsi_ob"] = (X["rsi"] > cfg.rsi_ob_level).astype(int)
        X["rsi_os"] = (X["rsi"] < cfg.rsi_os_level).astype(int)

        X["bb_touch_upper"] = (df["high"] >= X["bb_high"]).astype(int)
        X["bb_touch_lower"] = (df["low"] <= X["bb_low"]).astype(int)

        # --- Interaction Features ---
        X["rsi_x_adx"] = X["rsi"] * X["adx"]
        X["macd_x_adx"] = X["macd"] * X["adx"]
        X["ema_diff_x_adx"] = X["ema_diff"] * X["adx"]

        # --- handle NaNs and infs ---
        X = X.replace([np.inf, -np.inf], np.nan).ffill()   # no bfill: it would give the first rows values from later bars

    except Exception as e:
        logger.exception(f"[{symbol}] Error building dynamic features: {e}")
        raise

    return X


def add_relative_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Adds the relative versions of the absolute-level columns (C5), computed from the raw columns of `X` and the bars `df`
    (bar t only uses bars <= t). The raw columns stay: stops and trailing read `atr_14`, the contextual bandit reads
    `macd_diff`; `model_matrix` keeps them away from the model. Called by `build_features` (the main pipeline) and the tuner; the
    H1 bot and the research scripts build their own matrix with the builders above and are left as they were."""
    X = X.copy()
    close = df["close"]
    for name in ("macd", "macd_signal", "macd_diff"):
        if name in X:
            X[f"{name}_rel"] = X[name] / close
    for name in ("momentum_5", "momentum_10"):
        if name in X:
            X[f"{name}_rel"] = X[name] / close
    if "atr_14" in X:
        X["atr_rel"] = X["atr_14"] / close
    if "obv" in X and "volume" in df.columns:
        # the change over 20 bars per unit of volume: the cumulative level depends on where the window starts, the change does not
        X["obv_chg_20"] = (X["obv"] - X["obv"].shift(20)) / (df["volume"].rolling(20).sum() + 1e-10)
    if "donchian_h" in X:
        X["donchian_h_rel"] = (X["donchian_h"] - close) / close
        X["donchian_l_rel"] = (close - X["donchian_l"]) / close
        X["donchian_m_rel"] = (X["donchian_m"] - close) / close
    for name in ("ema_fast", "ema_slow"):
        if name in X:
            X[f"{name}_dist"] = (close - X[name]) / close
    if "bb_high" in X:
        X["bb_pos"] = (close - X["bb_low"]) / (X["bb_high"] - X["bb_low"] + 1e-12)
    if "macd_rel" in X and "adx" in X:
        X["macd_rel_x_adx"] = X["macd_rel"] * X["adx"]
    return X


from src.config import Cfg  # Import Cfg


def build_features(df: pd.DataFrame, feature_cfg: FeatureCfg, main_cfg: Cfg, symbol: str = None, mta_df: pd.DataFrame = None, inter_market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Original build_features function, now delegates to static and dynamic builders.
    This remains for compatibility with other scripts that may use it directly.
    """
    pa_cfg = getattr(getattr(main_cfg, 'context_features', None), 'price_action', None)
    mta_cfg = getattr(main_cfg.context_features, 'mta', None)
    im_cfg = getattr(main_cfg.context_features, 'inter_market', None)

    static_X = build_static_features(df, symbol, pa_cfg=pa_cfg)
    dynamic_X = build_dynamic_features(df, static_X, cfg=feature_cfg, symbol=symbol)

    dynamic_X = add_relative_features(dynamic_X, df)

    # Add contextual features
    dynamic_X = add_contextual_features(dynamic_X, mta_df=mta_df, inter_market_df=inter_market_df, mta_cfg=mta_cfg, im_cfg=im_cfg, relative=True)

    return dynamic_X


def make_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = (fwd > 0).astype(int)
    return y.loc[df.index]
