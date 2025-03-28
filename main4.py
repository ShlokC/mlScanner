import os
import time
import logging
import pickle
import datetime
import threading
import gc
import ccxt
import queue
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ccxt.base.errors import RequestTimeout, ExchangeError
from sklearn.preprocessing import RobustScaler
from queue import Queue

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ------------------ Global Parameters and Globals ------------------

AMOUNT_USD = 2
MAX_OPEN_TRADES = 2
WINDOW_SIZE = 5
TIMEFRAME = '3m'
LIMIT = 300
THRESHOLD = 0.005
PREDICTION_HORIZON = 3
SLEEP_INTERVAL = 20
K = 10  # Number of nearest neighbors for prediction
M = 10000  # Maximum number of historical sequences to store

market_historical_data = {}  # key: market_id, value: (X_hist_flatten, y_hist, timestamps)
market_scalers = {}  # key: market_id, value: scaler
historical_training_queue = Queue(maxsize=5)
training_in_progress = set()
training_lock = threading.Lock()
market_leverages = {}  # Dictionary to store leverage per market_id
leverage_options = [50, 20 , 10, 8]  # List of leverage values to try

# ------------------ Helper Functions ------------------

def get_historic_csv_filename(market_id):
    return f"{market_id}_historic.csv"

def get_scaler_filename(market_id):
    return f"scaler_{market_id}_ws{WINDOW_SIZE}.pkl"

def set_leverage_with_fallback(exchange, market_id, leverage_options):
    """Attempts to set leverage for a market_id with fallback."""
    for leverage in leverage_options:
        try:
            exchange.set_leverage(leverage, market_id)
            logging.info(f"Successfully set leverage to {leverage} for {market_id}")
            return leverage
        except Exception as e:
            logging.warning(f"Failed to set leverage to {leverage} for {market_id}: {e}")
    logging.error(f"Failed to set any leverage for {market_id}")
    return None

def get_leverage_for_market(exchange, market_id):
    """Retrieves or sets the leverage for a market_id."""
    if market_id in market_leverages:
        return market_leverages[market_id]
    else:
        leverage = set_leverage_with_fallback(exchange, market_id, leverage_options)
        if leverage is not None:
            market_leverages[market_id] = leverage
        return leverage

def preprocess_data(df, window_size=WINDOW_SIZE, lags=[1, 2, 3], scaler=None, fit_scaler=True):
    """
    Preprocesses candlestick data into sequences for prediction.
    Returns X (sequences), y (future closes), scaler, timestamps, and scaled data.
    """
    try:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Dataframe missing required column: {col}")
        df = df.dropna(subset=['close'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)

        # Technical indicators
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df['BB_PCT'] = bollinger['BBP_20_2.0']
        df['OBV'] = ta.obv(df['close'], df['volume'])
        df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])

        # Binary features
        df['EMA_20_Cross_EMA_50'] = np.where(
            (df['EMA_20'] > df['EMA_50']) & (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1)), 1,
            np.where((df['EMA_20'] < df['EMA_50']) & (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1)), -1, 0)
        )
        df['EMA_20_GT_EMA_50'] = (df['EMA_20'] > df['EMA_50']).astype(int)
        stoch_rsi = ta.stochrsi(df['close'])
        df['StochRSI_K'] = stoch_rsi['STOCHRSIk_14_14_3_3']
        df['StochRSI_D'] = stoch_rsi['STOCHRSId_14_14_3_3']
        df['StochRSI_Cross_D'] = np.where(
            (df['StochRSI_K'] > df['StochRSI_D']) & (df['StochRSI_K'].shift(1) <= df['StochRSI_D'].shift(1)), 1,
            np.where((df['StochRSI_K'] < df['StochRSI_D']) & (df['StochRSI_K'].shift(1) >= df['StochRSI_D'].shift(1)), -1, 0)
        )
        df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
        df['BB_Overextended'] = ((df['BB_PCT'] > 1) | (df['BB_PCT'] < 0)).astype(int)

        df.dropna(inplace=True)

        features = [
            'close', 'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'ATR', 'BB_PCT', 'OBV', 'CMF',
            'EMA_20_Cross_EMA_50', 'EMA_20_GT_EMA_50', 'StochRSI_Cross_D', 'RSI_Overbought', 'RSI_Oversold', 'BB_Overextended'
        ]

        lagged_dfs = {lag: df[features].shift(lag).add_suffix(f'_lag{lag}') for lag in lags}
        df_lagged = pd.concat([df[features]] + list(lagged_dfs.values()), axis=1)
        df_lagged.dropna(inplace=True)

        all_features = df_lagged.columns.tolist()
        data = df_lagged.values
        if scaler is None or fit_scaler or (scaler is not None and scaler.scale_.shape[0] != data.shape[1]):
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = scaler.transform(data)
        X, y = [], []
        target_index = all_features.index("close")
        for i in range(window_size, len(data_scaled) - PREDICTION_HORIZON):
            X.append(data_scaled[i - window_size:i])
            future_close_prices = data_scaled[i:i + PREDICTION_HORIZON, target_index]
            y.append(future_close_prices)
        timestamps = df['timestamp'].iloc[window_size:len(df) - PREDICTION_HORIZON].values
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler, timestamps
    except Exception as e:
        logging.exception(f"Error in preprocessing data: {e}")
        return np.array([]), np.array([]), None, None

def inverse_close(normalized_close, scaler):
    """Inverse transforms scaled close prices."""
    try:
        dummy = np.zeros((len(normalized_close), len(scaler.scale_)))
        dummy[:, 3] = normalized_close  # 'close' is at index 3
        inv = scaler.inverse_transform(dummy)
        return inv[:, 3]
    except Exception as e:
        logging.exception(f"Error in inverse transforming 'close': {e}")
        return normalized_close

# ------------------ Training and Prediction Functions ------------------

def historical_training_worker():
    """Processes historical data to populate market_historical_data."""
    while True:
        try:
            task = historical_training_queue.get()
            if task is None:
                break
            market_id = task['market_id']
            historic_csv = get_historic_csv_filename(market_id)
            if not os.path.exists(historic_csv):
                logging.warning(f"No historical data for {market_id}. Skipping.")
                historical_training_queue.task_done()
                continue
            df_market = pd.read_csv(historic_csv)
            df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])
            X, y, scaler, timestamps = preprocess_data(df_market, window_size=WINDOW_SIZE, fit_scaler=True)
            if X.size == 0:
                logging.warning(f"Not enough data for {market_id}. Skipping.")
                historical_training_queue.task_done()
                continue
            X_flatten = [x.flatten() for x in X]
            market_historical_data[market_id] = (X_flatten, y, timestamps)
            market_scalers[market_id] = scaler
            with open(get_scaler_filename(market_id), 'wb') as f:
                pickle.dump(scaler, f)
            logging.info(f"Historical data processed for {market_id}")
            with training_lock:
                training_in_progress.remove(market_id)
            historical_training_queue.task_done()
        except Exception as e:
            logging.exception(f"Error in historical training worker for task {task}: {e}")
            with training_lock:
                if market_id in training_in_progress:
                    training_in_progress.discard(market_id)
            historical_training_queue.task_done()

def predict_trend(market_id, X_new_flat):
    """Predicts future close prices using K-nearest neighbors."""
    if market_id not in market_historical_data:
        logging.warning(f"No historical data for {market_id}. Cannot predict.")
        return None
    X_hist_flatten, y_hist, _ = market_historical_data[market_id]
    if len(X_hist_flatten) == 0:
        logging.warning(f"No historical sequences for {market_id}. Cannot predict.")
        return None
    distances = [np.linalg.norm(x - X_new_flat) for x in X_hist_flatten]
    sorted_indices = np.argsort(distances)
    k = min(K, len(sorted_indices))
    pred_y = np.mean(y_hist[sorted_indices[:k]], axis=0)
    return pred_y

# ------------------ Market and Data Fetching Functions ------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30),
       retry=retry_if_exception_type((RequestTimeout, ExchangeError)), reraise=True)
def fetch_active_symbols(exchange):
    try:
        markets = exchange.load_markets()
        # Filter for USDT-settled, active markets, excluding ETH and BTC pairs
        active_markets = [
            (symbol, market['id'], market.get('type', ''))
            for symbol, market in markets.items()
            if market.get("settle") == "USDT" and market.get("active")
            and "ETH" not in market['id']
            and "BTC" not in market['id']
        ]
        swap_markets = [(symbol, market_id) for symbol, market_id, m_type in active_markets if m_type == 'swap']
        future_markets = [(symbol, market_id) for symbol, market_id, m_type in active_markets if m_type == 'future']

        # Dictionary to store metrics: (abs_price_change, volume_increase)
        market_metrics = {}

        # Fetch OHLCV data for swap markets (last 15 minutes, 1m candles)
        if swap_markets:
            exchange.options['defaultType'] = 'swap'
            for symbol, market_id in swap_markets:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=15)
                    if len(ohlcv) < 15:
                        logging.warning(f"Insufficient data for {symbol} (swap): {len(ohlcv)} candles. Skipping.")
                        continue
                    # Extract first and last candles
                    close_first = float(ohlcv[0][4])  # Close price at start
                    close_last = float(ohlcv[-1][4])  # Close price at end
                    volume_first = float(ohlcv[0][5])  # Volume at start
                    volume_last = float(ohlcv[-1][5])  # Volume at end

                    # Calculate absolute price change and volume increase
                    abs_price_change = abs((close_last - close_first) / close_first) if close_first != 0 else 0.0
                    volume_increase = (volume_last - volume_first) / volume_first if volume_first != 0 else 0.0

                    # Only include markets with volume increase
                    if volume_increase > 0:
                        market_metrics[symbol] = (abs_price_change, volume_increase)
                    # else:
                    #     logging.info(f"Excluded {symbol} (swap): Volume decreased ({volume_increase:.4%}).")
                except Exception as e:
                    logging.warning(f"Error fetching OHLCV for {symbol} (swap): {e}. Skipping.")
                    continue

        # Fetch OHLCV data for future markets (last 15 minutes, 1m candles)
        if future_markets:
            exchange.options['defaultType'] = 'future'
            for symbol, market_id in future_markets:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=15)
                    if len(ohlcv) < 15:
                        logging.warning(f"Insufficient data for {symbol} (future): {len(ohlcv)} candles. Skipping.")
                        continue
                    close_first = float(ohlcv[0][4])
                    close_last = float(ohlcv[-1][4])
                    volume_first = float(ohlcv[0][5])
                    volume_last = float(ohlcv[-1][5])

                    abs_price_change = abs((close_last - close_first) / close_first) if close_first != 0 else 0.0
                    volume_increase = (volume_last - volume_first) / volume_first if volume_first != 0 else 0.0

                    if volume_increase > 0:
                        market_metrics[symbol] = (abs_price_change, volume_increase)
                    # else:
                    #     logging.info(f"Excluded {symbol} (future): Volume decreased ({volume_increase:.4%}).")
                except Exception as e:
                    logging.warning(f"Error fetching OHLCV for {symbol} (future): {e}. Skipping.")
                    continue

        # Reset default type to swap
        exchange.options['defaultType'] = 'swap'

        # If no metrics were calculated, return empty list
        if not market_metrics:
            logging.error("No valid markets with volume increase in last 15 mins.")
            return []

        # Sort markets: primary by abs_price_change, secondary by volume_increase
        sorted_symbols = sorted(
            market_metrics.keys(),
            key=lambda x: (market_metrics[x][0], market_metrics[x][1]),
            reverse=True
        )
        sorted_active_markets = [
            (symbol, next(market_id for s, market_id, _ in active_markets if s == symbol))
            for symbol in sorted_symbols
        ]

        # Log top 5 for debugging
        logging.info("Top 5 markets by abs % price change (primary) and % volume increase (secondary, last 15 mins):")
        for symbol in sorted_symbols[:5]:
            abs_price_chg, volume_inc = market_metrics[symbol]
            logging.info(f"{symbol}: Abs Price Change {abs_price_chg:.4%}, Volume Increase {volume_inc:.4%}")

        return sorted_active_markets

    except Exception as e:
        logging.exception(f"Error loading markets or calculating metrics: {e}")
        return []

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30),
       retry=retry_if_exception_type((RequestTimeout, ExchangeError)), reraise=True)
def fetch_open_positions(exchange):
    """Fetches current open positions."""
    open_positions_map = {}
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            amt = float(pos['info'].get('positionAmt', 0))
            if amt != 0 and 'symbol' in pos:
                market_info = exchange.market(pos['symbol'])
                m_id = market_info['id']
                open_positions_map[m_id] = pos
        return open_positions_map
    except Exception as e:
        logging.exception(f"Error fetching positions: {e}")
        return {}

def fetch_and_store_historic_data_threaded():
    """Fetches and stores historical data for active markets."""
    logging.info("Historical data download thread started.")
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    active_markets = fetch_active_symbols(exchange)
    for unified_symbol, market_id in active_markets:
        csv_filename = get_historic_csv_filename(market_id)
        if not os.path.exists(csv_filename):
            fetch_last_year_data(market_id, timeframe=TIMEFRAME, csv_filename=csv_filename)
    logging.info("Historical data download thread finished.")
    gc.collect()

def fetch_last_year_data(market_id, timeframe=TIMEFRAME, csv_filename=None):
    """Fetches one year of historical data."""
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 30000
    })
    now = exchange.milliseconds()
    one_year_ago = now - 365 * 24 * 60 * 60 * 1000
    all_ohlcv = []
    since = one_year_ago
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        except Exception as e:
            logging.exception(f"Error fetching historical data for {market_id}: {e}")
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    if csv_filename:
        df.to_csv(csv_filename, index=False)
        logging.info(f"Saved historical data to {csv_filename}")
    return df

# ------------------ Trading Functions ------------------

def get_order_quantity(current_price, leverage):
    """Calculates order quantity."""
    try:
        quantity = (AMOUNT_USD * leverage) / current_price
        notional = quantity * current_price
        min_notional = 19
        if notional < min_notional:
            quantity = min_notional / current_price
        return quantity
    except Exception as e:
        logging.exception(f"Error calculating order quantity: {e}")
        return 0

def place_order(exchange, market_id, side, quantity, current_price, stop_loss_price=None, take_profit_price=None):
    """Places a market order with optional stop-loss and take-profit."""
    params = {}
    if stop_loss_price:
        params['stopLoss'] = {'price': stop_loss_price, 'type': 'MARKET'}
    if take_profit_price:
        params['takeProfit'] = {'price': take_profit_price, 'type': 'MARKET', 'timeInForce': 'GTE-GTC'}
    try:
        order = exchange.create_order(market_id, type='market', side=side, amount=quantity, params=params)
        logging.info(f"Order placed for {market_id}: {order}")
        return order
    except Exception as e:
        logging.exception(f"Failed to place order for {market_id}: {e}")
        return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30),
       retry=retry_if_exception_type((RequestTimeout, ExchangeError)), reraise=True)
def fetch_binance_data(market_id, timeframe=TIMEFRAME, limit=LIMIT):
    """Fetches live OHLCV data."""
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 30000
    })
    ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.dropna(subset=['close'], inplace=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

# ------------------ Main Trading Loop ------------------

def continuous_loop(exchange):
    """Main trading loop with KNN-based prediction."""
    while True:
        try:
            logging.info("=== Starting new trading cycle ===")
            active_markets = fetch_active_symbols(exchange)
            if not active_markets:
                logging.error("No active markets found.")
                time.sleep(SLEEP_INTERVAL)
                continue
            open_positions_map = fetch_open_positions(exchange)
            for unified_symbol, market_id in active_markets:
                if len(open_positions_map) >= MAX_OPEN_TRADES and market_id not in open_positions_map:
                    continue
                if market_id not in market_historical_data or market_id not in market_scalers:
                    scaler_file = get_scaler_filename(market_id)
                    if os.path.exists(scaler_file):
                        with open(scaler_file, 'rb') as f:
                            market_scalers[market_id] = pickle.load(f)
                        with training_lock:
                            if market_id not in training_in_progress:
                                historical_training_queue.put({'market_id': market_id}, block=False)
                                training_in_progress.add(market_id)
                                logging.info(f"Enqueued historical training for {market_id}")
                    continue
                scaler = market_scalers[market_id]
                df_live = fetch_binance_data(market_id, timeframe=TIMEFRAME, limit=LIMIT)
                if df_live.empty:
                    continue
                
                # Preprocess the live data to include all features
                X_live, y_live, _, timestamps_live = preprocess_data(df_live, window_size=WINDOW_SIZE, scaler=scaler, fit_scaler=False)
                if X_live.size == 0:
                    logging.warning(f"Not enough processed live data for {market_id}. Skipping...")
                    continue

                # Use the last window from X_live for prediction
                X_new = X_live[-1]  # Shape: (WINDOW_SIZE, num_features)
                X_new_flat = X_new.flatten()

                # Predict using KNN
                pred_sequence_norm = predict_trend(market_id, X_new_flat)
                if pred_sequence_norm is None:
                    continue
                predicted_close_sequence = inverse_close(pred_sequence_norm, scaler)
                last_close_norm = X_new[-1, 3]  # 'close' at index 3 in the feature list
                last_close = inverse_close(np.array([last_close_norm]), scaler)[0]
                predicted_close_horizon = predicted_close_sequence[-1]
                pct_change_horizon = (predicted_close_horizon - last_close) / last_close

                # Trend determination
                volatility = df_live['close'].pct_change().dropna().tail(20).std() * np.sqrt(252) if not df_live.empty else 0.01
                if np.isnan(volatility) or volatility == 0:
                    volatility = 0.01
                base_threshold = max(0.002, volatility)
                dynamic_threshold = base_threshold
                if pct_change_horizon > dynamic_threshold:
                    trend_direction = "UPWARD"
                    trend_confidence = min(1.0, pct_change_horizon / (2 * dynamic_threshold))
                elif pct_change_horizon < -dynamic_threshold:
                    trend_direction = "DOWNWARD"
                    trend_confidence = min(1.0, -pct_change_horizon / (2 * dynamic_threshold))
                else:
                    trend_direction = "FLAT"
                    trend_confidence = 0.0
                logging.info(f"{market_id} - Trend Direction: {trend_direction} (Confidence: {trend_confidence:.2%}), Pct Change Horizon: {pct_change_horizon:.4f}")

                # Trading logic
                if market_id not in globals().get('cumulative_pct_change', {}):
                    cumulative_pct_change = {}
                    globals()['cumulative_pct_change'] = cumulative_pct_change
                if market_id not in cumulative_pct_change:
                    cumulative_pct_change[market_id] = 0.0
                decay_factor = 0.9
                cumulative_pct_change[market_id] = (cumulative_pct_change[market_id] * decay_factor) + pct_change_horizon
                signal = "HOLD"
                strong_signal_threshold = dynamic_threshold * 1.2
                cumulative_threshold = dynamic_threshold * 2.0
                if (trend_direction == "UPWARD" and
                    (pct_change_horizon > strong_signal_threshold or cumulative_pct_change[market_id] > cumulative_threshold) and
                    trend_confidence > 0.9):
                    signal = "SELL"
                    cumulative_pct_change[market_id] = 0.0
                elif (trend_direction == "DOWNWARD" and
                      (abs(pct_change_horizon) > strong_signal_threshold or abs(cumulative_pct_change[market_id]) > cumulative_threshold) and
                      trend_confidence > 0.9):
                    signal = "BUY"
                    cumulative_pct_change[market_id] = 0.0
                logging.info(f"{market_id} - Signal: {signal}")

                adverse_threshold = -0.005
                exit_executed = False
                if market_id in open_positions_map:
                    pos = open_positions_map[market_id]
                    current_amt = float(pos['info'].get('positionAmt', 0))
                    current_signal = "BUY" if current_amt > 0 else "SELL"
                    last_close_live = df_live['close'].iloc[-1]
                    entry_price = float(pos['info'].get('entryPrice', last_close_live))
                    pct_change_since_entry = (last_close_live - entry_price) / entry_price if current_signal == "BUY" else (entry_price - last_close_live) / entry_price
                    if pct_change_since_entry < adverse_threshold or (signal in ["BUY", "SELL"] and signal != current_signal):
                        exit_side = "sell" if current_signal == "BUY" else "buy"
                        exit_quantity = abs(current_amt)
                        leverage = get_leverage_for_market(exchange, market_id)
                        if leverage:
                            exit_order = place_order(exchange, market_id, exit_side, exit_quantity, last_close_live)
                            if exit_order:
                                logging.info(f"{market_id} - Exit order executed.")
                                open_positions_map.pop(market_id, None)
                                exit_executed = True
                if not exit_executed and market_id not in open_positions_map and signal in ["BUY", "SELL"]:
                    leverage = get_leverage_for_market(exchange, market_id)
                    if leverage:
                        last_close_live = df_live['close'].iloc[-1]
                        quantity = get_order_quantity(last_close_live, leverage)
                        stop_loss_price = last_close_live * (0.995 if signal == "BUY" else 1.005)
                        take_profit_price = last_close_live * (1.01 if signal == "BUY" else 0.99)
                        entry_order = place_order(exchange, market_id, signal.lower(), quantity, last_close_live, stop_loss_price, take_profit_price)
                        if entry_order:
                            logging.info(f"{market_id} - Entry order executed.")
                            open_positions_map[market_id] = {'symbol': market_id,
                                                            'info': {'positionAmt': str(quantity if signal == "BUY" else -quantity),
                                                                     'entryPrice': str(last_close_live)}}

                # Update historical data
                if market_id in market_historical_data:
                    X_hist_flatten, y_hist, timestamps_hist = market_historical_data[market_id]
                    max_timestamp = max(timestamps_hist) if timestamps_hist.size > 0 else pd.Timestamp.min
                    new_indices = [i for i, ts in enumerate(timestamps_live) if ts > max_timestamp]
                    if new_indices:
                        X_live_flatten = [X_live[i].flatten() for i in new_indices]
                        y_live_new = y_live[new_indices]
                        timestamps_live_new = timestamps_live[new_indices]
                        X_hist_flatten.extend(X_live_flatten)
                        y_hist = np.vstack([y_hist, y_live_new]) if y_hist.size > 0 else y_live_new
                        timestamps_hist = np.append(timestamps_hist, timestamps_live_new)
                        if len(X_hist_flatten) > M:
                            X_hist_flatten = X_hist_flatten[-M:]
                            y_hist = y_hist[-M:]
                            timestamps_hist = timestamps_hist[-M:]
                        market_historical_data[market_id] = (X_hist_flatten, y_hist, timestamps_hist)
                gc.collect()
            time.sleep(SLEEP_INTERVAL)
        except Exception as e:
            logging.exception(f"Error in trading cycle: {e}")
            time.sleep(SLEEP_INTERVAL)

# ------------------ Main ------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    data_thread = threading.Thread(target=fetch_and_store_historic_data_threaded, daemon=True)
    data_thread.start()
    historical_training_thread = threading.Thread(target=historical_training_worker, daemon=True)
    historical_training_thread.start()
    live_thread = threading.Thread(target=continuous_loop, args=(exchange,), daemon=True)
    live_thread.start()
    while True:
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()