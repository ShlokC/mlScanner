import os
import time
import logging
import pickle
import datetime
import threading
import gc
import ccxt
import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ccxt.base.errors import RequestTimeout, ExchangeError
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dense, Dropout, Bidirectional, Attention, Add, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from queue import Queue  # <-- Import Queue for our training tasks
from scipy import stats
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
# ------------------ Global Parameters and Globals ------------------

AMOUNT_USD = 2
DEFAULT_LEVERAGE = 20
MAX_OPEN_TRADES = 2
WINDOW_SIZE = 5
TIMEFRAME = '3m'
LIMIT = 300
THRESHOLD = 0.005  # Lowered from 0.005 to 0.2% minimum threshold
PREDICTION_HORIZON = 1  # Reduced from 5 for more responsive signals
SLEEP_INTERVAL = 20

PERFORMANCE_CHECK_INTERVAL = 10
ERROR_THRESHOLD = 0.01

NUM_MC_SAMPLES = 5
FIXED_THRESHOLD = 0.01

DEFAULT_MODEL_PARAMS = {
    'lstm_units1': 64,
    'lstm_units2': 32,
    'lstm_units3': 16,
    'dropout_rate': 0.3,
    'attention_heads': 4,
    'cnn_filters': 64,     # New parameter
    'cnn_kernel_size': 3   # New parameter
}

# Instead of global baap_model/scaler, we use per-market dictionaries.
market_models = {}      # key: market_id, value: model
market_scalers = {}     # key: market_id, value: scaler
market_model_params = {}  # key: market_id, value: model parameters
market_locks = {}  # Add this at the global scope
# This file now stores a dict with keys: 'completed_markets' (list) and 'historical_training_completed' (bool)
BAAP_TRAINED_FLAG_FILENAME = f"cnn_model{WINDOW_SIZE}_trained_flag.pkl"

historical_training_completed = False

# ------------------ Helper Functions for Unique Filenames ------------------

def get_model_filename(market_id):
    return f"model_cnn_{market_id}_ws{WINDOW_SIZE}.keras"

def get_scaler_filename(market_id):
    return f"scaler_cnn_{market_id}_ws{WINDOW_SIZE}.pkl"

def get_historic_csv_filename(market_id):
    return f"{market_id}_historic.csv"

# ------------------ Training Queue Setup ------------------
# Create a queue with a maxsize of 1 to avoid duplicate training tasks.
training_queue = Queue(maxsize=1)

def training_worker():
    """Worker thread that processes training tasks from the training_queue."""
    while True:
        try:
            # This will block until a task is available.
            task = training_queue.get()
            if task is None:
                break  # Optional: to allow graceful shutdown
            market_id, X_live, y_live, scaler_live = task
            train_market_model_in_background(market_id, market_models.get(market_id), X_live, y_live, scaler_live)
        except Exception as e:
            logging.exception(f"Error in training worker: {e}")
        finally:
            training_queue.task_done()

# ------------------ Helper Functions for Historic Data ------------------

def fetch_last_year_data(market_id, timeframe=TIMEFRAME, csv_filename=None):
    try:
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
                last_timestamp = ohlcv[-1][0]
                if last_timestamp == since:
                    break
                since = last_timestamp + 1
                if last_timestamp >= now:
                    break
            except Exception as e:
                logging.exception(f"Error fetching historical data for {market_id}: {e}")
                break
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Compute features directly without X, y split for CSV purposes
        df_processed = df.copy()
        df_processed['EMA_20'] = ta.ema(df['close'], length=20)
        df_processed['EMA_50'] = ta.ema(df['close'], length=50)
        df_processed['RSI_14'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df_processed['MACD'] = macd['MACD_12_26_9']
        df_processed['MACD_Signal'] = macd['MACDs_12_26_9']
        df_processed['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df_processed['BB_PCT'] = bollinger['BBP_20_2.0']
        df_processed['OBV'] = ta.obv(df['close'], df['volume'])
        df_processed['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
        df_processed['EMA_20_Cross_EMA_50'] = np.where(
            (df_processed['EMA_20'] > df_processed['EMA_50']) & (df_processed['EMA_20'].shift(1) <= df_processed['EMA_50'].shift(1)), 1,
            np.where((df_processed['EMA_20'] < df_processed['EMA_50']) & (df_processed['EMA_20'].shift(1) >= df_processed['EMA_50'].shift(1)), -1, 0)
        )
        df_processed['EMA_20_GT_EMA_50'] = (df_processed['EMA_20'] > df_processed['EMA_50']).astype(int)
        stoch_rsi = ta.stochrsi(df['close'])
        df_processed['StochRSI_Cross_D'] = np.where(
            (stoch_rsi['STOCHRSIk_14_14_3_3'] > stoch_rsi['STOCHRSId_14_14_3_3']) & 
            (stoch_rsi['STOCHRSIk_14_14_3_3'].shift(1) <= stoch_rsi['STOCHRSId_14_14_3_3'].shift(1)), 1,
            np.where((stoch_rsi['STOCHRSIk_14_14_3_3'] < stoch_rsi['STOCHRSId_14_14_3_3']) & 
                     (stoch_rsi['STOCHRSIk_14_14_3_3'].shift(1) >= stoch_rsi['STOCHRSId_14_14_3_3'].shift(1)), -1, 0)
        )
        df_processed['RSI_Overbought'] = (df_processed['RSI_14'] > 70).astype(int)
        df_processed['RSI_Oversold'] = (df_processed['RSI_14'] < 30).astype(int)
        df_processed['BB_Overextended'] = ((df_processed['BB_PCT'] > 1) | (df_processed['BB_PCT'] < 0)).astype(int)
        df_processed.dropna(inplace=True)

        if csv_filename:
            df_processed.to_csv(csv_filename, index=False)
            logging.info(f"Saved processed historic data to {csv_filename}")
        return df  # Return raw OHLCV for downstream use
    except Exception as e:
        logging.exception(f"Failed to fetch last year data for {market_id}: {e}")
        return pd.DataFrame()
# ------------------ Data Fetching Thread Function ------------------

def fetch_and_store_historic_data_threaded():
    logging.info("Historical data download thread started.")
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    active_markets = fetch_active_symbols(exchange)
    if not active_markets:
        logging.error("No active markets found, cannot fetch historical data.")
        return

    for unified_symbol, market_id in active_markets:
        csv_filename = get_historic_csv_filename(market_id)
        if not os.path.exists(csv_filename):
            logging.info(f"Downloading historical data for {market_id} to {csv_filename}...")
            fetch_last_year_data(market_id, timeframe=TIMEFRAME, csv_filename=csv_filename)
        else:
            logging.info(f"Historical data already exists for {market_id} at {csv_filename}. Skipping download.")
    logging.info("Historical data download thread finished.")
    gc.collect()

# ------------------ Data Generator Function with Optional Weighting ------------------

def data_generator(X, y, batch_size, use_weights=True, decay=0.95):  # Adjusted decay from 0.99 to 0.95
    num_samples = len(X)
    while True:
        indices = np.arange(num_samples)
        if use_weights:
            # Steeper decay: newer samples have much higher weight
            weights = np.array([decay ** (num_samples - 1 - i) for i in range(num_samples)], dtype=np.float32)
            weights /= weights.sum()  # Normalize weights to sum to 1
            indices = np.random.choice(indices, size=num_samples, p=weights, replace=True)

        for i in range(0, num_samples, batch_size):
            end_i = min(i + batch_size, num_samples)
            batch_indices = indices[i:end_i]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            if use_weights:
                batch_weights = weights[batch_indices]
                yield batch_X, batch_y, batch_weights
            else:
                yield batch_X, batch_y

# ------------------ Model Training Thread Function (Incremental Saving with Resume) ------------------

def train_market_model_in_background(market_id, model, X_live, y_live, scaler_live):
    """Retrains the market model using live data with an efficient tf.data pipeline."""
    try:
        # Clear TensorFlow session and graph to prevent state corruption
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        # Load progress to check last training time
        progress = {
            'completed_markets': [],
            'historical_training_completed': False,
            'training_times': {}
        }
        if os.path.exists(BAAP_TRAINED_FLAG_FILENAME):
            with open(BAAP_TRAINED_FLAG_FILENAME, 'rb') as f:
                loaded_progress = pickle.load(f)
                progress.update(loaded_progress)

        # Check if retraining is needed
        current_time = datetime.datetime.now()
        last_train_time = progress['training_times'].get(market_id)
        # if last_train_time and (current_time - last_train_time).total_seconds() < 4 * 60 * 60:
        #     logging.info(f"Skipping retraining for {market_id}; last trained at {last_train_time}, less than 24 hours ago.")
        #     return

        # Verify model is valid
        if model is None:
            logging.error(f"Model for {market_id} is None. Cannot retrain.")
            return
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_live, y_live, test_size=0.2, shuffle=False)
        batch_size = 32

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4, verbose=1),
        ]

        # Train with timing
        logging.info(f"Starting background retraining for {market_id}...")
        start_time = time.time()
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        logging.info(f"Background retraining for {market_id} completed in {training_time:.2f} seconds.")

        # Evaluate
        evaluation = model.evaluate(X_live, y_live, verbose=0)
        logging.info(f"Market {market_id} - Background training loss: {evaluation[0]:.4f}")

        # Save model and scaler
        model.save(get_model_filename(market_id))
        with open(get_scaler_filename(market_id), 'wb') as f:
            pickle.dump(scaler_live, f)
        market_models[market_id] = model
        market_scalers[market_id] = scaler_live

        # Update training time
        progress['training_times'][market_id] = current_time
        with open(BAAP_TRAINED_FLAG_FILENAME, 'wb') as f:
            pickle.dump(progress, f)
        logging.info(f"Updated training time for {market_id} in {BAAP_TRAINED_FLAG_FILENAME}")
    except Exception as e:
        logging.exception(f"Error in background training for {market_id}: {e}")
    finally:
        gc.collect()


def train_market_models_threaded():
    global historical_training_completed
    logging.info("Market models training thread started (incremental).")
    exchange = ccxt.binanceusdm()
    active_markets = fetch_active_symbols(exchange)
    if not active_markets:
        logging.error("No active markets found, cannot train market models.")
        return

    # Load progress, including training times
    progress = {
        'completed_markets': [],
        'historical_training_completed': False,
        'training_times': {}  # New field for timestamps
    }
    if os.path.exists(BAAP_TRAINED_FLAG_FILENAME):
        try:
            with open(BAAP_TRAINED_FLAG_FILENAME, 'rb') as f:
                loaded_progress = pickle.load(f)
                progress.update(loaded_progress)  # Merge with defaults
            logging.info(f"Loaded training progress: {progress}")
        except Exception as e:
            logging.error(f"Error loading training progress: {e}. Starting from scratch.")

    for unified_symbol, market_id in active_markets:
        if market_id in progress['completed_markets']:
            logging.info(f"Market {market_id} already trained historically. Skipping initial training.")
            continue

        historic_csv = get_historic_csv_filename(market_id)
        if not os.path.exists(historic_csv):
            logging.warning(f"No historical data for {market_id}. Skipping.")
            continue

        df_market = pd.read_csv(historic_csv)
        df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])

        logging.info(f"Preprocessing historical data for {market_id}...")
        X, y, scaler_local = preprocess_data(df_market.copy(), window_size=WINDOW_SIZE)
        if X.size == 0:
            logging.warning(f"Not enough data for {market_id}. Skipping.")
            continue

        logging.info(f"Splitting data and building/loading model for {market_id}...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model_file = get_model_filename(market_id)
        scaler_file = get_scaler_filename(market_id)

        if os.path.exists(model_file):
            try:
                model_local = tf.keras.models.load_model(model_file)
                if model_local.output_shape[-1] != PREDICTION_HORIZON:
                    logging.info(f"Model for {market_id} has incorrect output shape, rebuilding.")
                    K.clear_session()
                    model_local = build_model(input_shape, DEFAULT_MODEL_PARAMS)
                else:
                    logging.info(f"Loaded existing model for {market_id}.")
            except Exception as e:
                logging.error(f"Error loading model for {market_id}: {e}. Re-initializing.")
                K.clear_session()
                model_local = build_model(input_shape, DEFAULT_MODEL_PARAMS)
        else:
            K.clear_session()
            model_local = build_model(input_shape, DEFAULT_MODEL_PARAMS)
        market_model_params[market_id] = DEFAULT_MODEL_PARAMS.copy()

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4, verbose=1)
        ]
        batch_size = 32
        train_gen = data_generator(X_train, y_train, batch_size, use_weights=True)
        val_gen = data_generator(X_val, y_val, batch_size)
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)

        logging.info(f"Training model for {market_id}...")
        start_time = time.time()
        model_local.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        logging.info(f"Training for {market_id} completed in {training_time:.2f} seconds.")

        logging.info(f"Saving model and scaler for {market_id}...")
        model_local.save(model_file)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler_local, f)
        market_models[market_id] = model_local
        market_scalers[market_id] = scaler_local

        # Update training time in progress
        progress['training_times'][market_id] = datetime.datetime.now()
        with open(BAAP_TRAINED_FLAG_FILENAME, 'wb') as f:
            pickle.dump(progress, f)
            f.flush()
            os.fsync(f.fileno())
        logging.info(f"Updated training time for {market_id} in {BAAP_TRAINED_FLAG_FILENAME}")

        del X, y, X_train, X_val, y_train, y_val, train_gen, val_gen
        gc.collect()
        logging.info(f"Model trained and saved for {market_id}.")

        progress['completed_markets'].append(market_id)
        with open(BAAP_TRAINED_FLAG_FILENAME, 'wb') as f:
            pickle.dump(progress, f)
            f.flush()
            os.fsync(f.fileno())
        logging.info(f"Training progress updated for {market_id}")

    progress['historical_training_completed'] = True
    with open(BAAP_TRAINED_FLAG_FILENAME, 'wb') as f:
        pickle.dump(progress, f)
    logging.info(f"Historical training completed flag saved")
    historical_training_completed = True
    logging.info("Market models training thread finished (incremental).")

# ------------------ Market and Data Fetching Functions ------------------
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=4, max=30),
       retry=retry_if_exception_type((RequestTimeout, ExchangeError)),
       reraise=True)
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

# ------------------ Trading Functions ------------------

@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=4, max=30),
       retry=retry_if_exception_type((RequestTimeout, ExchangeError)),
       reraise=True)
def fetch_open_positions(exchange):
    open_positions_map = {}
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            amt = float(pos['info'].get('positionAmt', 0))
            if amt != 0 and 'symbol' in pos:
                try:
                    market_info = exchange.market(pos['symbol'])
                    m_id = market_info['id']
                    open_positions_map[m_id] = pos
                except Exception as conv_e:
                    logging.warning(f"Conversion error for {pos['symbol']}: {conv_e}")
        return open_positions_map
    except Exception as e:
        logging.exception(f"Error fetching positions: {e}")
        return {}

def get_order_quantity(current_price):
    try:
        # Compute initial quantity based on desired notional (AMOUNT_USD * DEFAULT_LEVERAGE)
        quantity = (AMOUNT_USD * DEFAULT_LEVERAGE) / current_price
        # Calculate notional value of this order
        notional = quantity * current_price
        # If notional is below the minimum threshold, adjust the quantity accordingly
        min_notional = 19  # minimum 5 USDT as required by Binance
        if notional < min_notional:
            quantity = min_notional / current_price
        return quantity
    except Exception as e:
        logging.exception(f"Error calculating order quantity: {e}")
        return 0

def place_order(exchange, market_id, side, quantity, leverage, current_price, stop_loss_price=None, take_profit_price=None):
    # List of leverage values to try, starting with the default
    leverage_options = [75, 50, 25, 20, 8]
    applied_leverage = None

    # Try setting leverage with fallbacks
    for lev in leverage_options:
        try:
            exchange.set_leverage(lev, market_id)
            applied_leverage = lev
            logging.info(f"Successfully set leverage to {lev}x for {market_id}")
            break  # Exit loop if successful
        except Exception as e:
            logging.warning(f"Failed to set leverage to {lev}x for {market_id}: {e}")
            if lev == leverage_options[-1]:  # Last attempt failed
                logging.error(f"All leverage attempts failed for {market_id}. Cannot place order.")
                return None
            continue  # Try next lower leverage

    # If no leverage was applied, exit
    if applied_leverage is None:
        return None
    params = {'leverage': leverage}

    if stop_loss_price:
        params['stopLoss'] = {'price': stop_loss_price, 'type': 'MARKET'}
    if take_profit_price:
        params['takeProfit'] = {'price': take_profit_price, 'type': 'MARKET'}

    # If no take profit price is provided, calculate it so that profit is $0.50
    if take_profit_price is None:
        if side.lower() == 'buy':
            take_profit_price = current_price + (0.8 * current_price) / (AMOUNT_USD * leverage)
        elif side.lower() == 'sell':
            take_profit_price = current_price - (0.8 * current_price) / (AMOUNT_USD * leverage)
    take_profit_side = 'sell' if side == 'buy' else 'buy'
    takeProfitParams = {'stopPrice': take_profit_price , 'timeInForce': 'GTE_GTC'}
    price = None

    try:
        order = exchange.create_order(
            market_id,
            type='market',
            side=side,
            amount=quantity,
            params=params
        )
        logging.info(f"Order placed for {market_id}: {order}")
        takeProfitOrder = exchange.create_order(market_id, 'TAKE_PROFIT_MARKET', take_profit_side, quantity, price, takeProfitParams)
        logging.info(f"Profit Order placed for {market_id}: {takeProfitOrder}")
        return order
    except Exception as e:
        logging.exception(f"Failed to place order for {market_id}: {e}")
        return None

@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=4, max=30),
       retry=retry_if_exception_type((RequestTimeout, ExchangeError)),
       reraise=True)
def fetch_binance_data(market_id, timeframe=TIMEFRAME, limit=LIMIT):
    exchange = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 30000
    })
    try:
        ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=limit)
    except Exception as e:
        logging.exception(f"Error fetching OHLCV for {market_id}: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.dropna(subset=['close'], inplace=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

def preprocess_data(df, window_size=WINDOW_SIZE, lags=[1, 2, 3], scaler=None, fit_scaler=True):
    try:
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Dataframe missing required column: {col}")
        df = df.dropna(subset=['close'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'])

        # Continuous features
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

        # Feature list
        features = [
            'close', 'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'ATR', 'BB_PCT', 'OBV', 'CMF',
            'EMA_20_Cross_EMA_50', 'EMA_20_GT_EMA_50', 'StochRSI_Cross_D', 'RSI_Overbought', 'RSI_Oversold', 'BB_Overextended'
        ]

        # Compute lagged features
        lagged_dfs = {lag: df[features].shift(lag).add_suffix(f'_lag{lag}') for lag in lags}
        df_lagged = pd.concat([df[features]] + list(lagged_dfs.values()), axis=1)
        df_lagged.dropna(inplace=True)

        # Scaling and sequence creation
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
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

    except Exception as e:
        logging.exception(f"Error in preprocessing data: {e}")
        return np.array([]), np.array([]), None

def inverse_close(normalized_close, scaler):
    try:
        dummy = np.zeros((len(normalized_close), len(scaler.scale_)))
        dummy[:, 0] = normalized_close  # 'close' is at index 0
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]  # Return the inverse transformed 'close'
    except Exception as e:
        logging.exception(f"Error in inverse transforming 'close': {e}")
        return normalized_close

# ------------------ Model Building Functions ------------------

def build_model_with_params(model_params, input_shape):
    try:
        # Define the input layer with the given shape (window_size, number_of_features)
        inputs = Input(shape=input_shape)
        
        # Add a Conv1D layer to extract local features
        x = Conv1D(
            filters=64,          # Number of filters (can experiment with 32, 64, 128)
            kernel_size=2,       # Size of the convolutional window (can experiment with 2, 3, 4)
            padding='same',      # Ensures output length equals input length
            activation='relu'    # ReLU activation for feature extraction
        )(inputs)
        
        # Add batch normalization to stabilize training
        x = BatchNormalization()(x)
        
        # First Bidirectional LSTM layer with return sequences
        x = Bidirectional(LSTM(
            model_params.get('lstm_units1', 64),
            return_sequences=True,
            recurrent_dropout=0.1  # Slight regularization
        ))(x)
        x = Dropout(model_params.get('dropout_rate', 0.3))(x)
        
        # Multi-head attention layer
        attention = Attention(use_scale=True)([x, x])
        
        # Second Bidirectional LSTM layer with return sequences
        x = Bidirectional(LSTM(
            model_params.get('lstm_units2', 32),
            return_sequences=True,
            recurrent_dropout=0.1
        ))(attention)
        x = Dropout(model_params.get('dropout_rate', 0.3))(x)
        
        # Third Bidirectional LSTM layer without return sequences
        x = Bidirectional(LSTM(
            model_params.get('lstm_units3', 16),
            return_sequences=False
        ))(x)
        x = Dropout(model_params.get('dropout_rate', 0.3))(x)
        
        # Output layer with linear activation for regression
        outputs = Dense(PREDICTION_HORIZON, activation='linear')(x)
        
        # Create and compile the model
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',         # Mean squared error for regression
            metrics=['mae']     # Mean absolute error for monitoring
        )
        return model
    except Exception as e:
        logging.exception(f"Error building model with params {model_params}: {e}")
        return None
    
def build_model(input_shape, model_params=DEFAULT_MODEL_PARAMS):
    return build_model_with_params(model_params, input_shape)

# ------------------ Self-Improvement Functions ------------------

def self_improve_model(current_model, X, y, scaler, current_params):
    logging.info("Starting self-improvement for LSTM model")
    X_train_candidate, X_val_candidate, y_train_candidate, y_val_candidate = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    candidate_params = [
        # Increase capacity
        {**current_params, 'lstm_units1': current_params.get('lstm_units1', 64) + 16},
        {**current_params, 'lstm_units2': current_params.get('lstm_units2', 32) + 8},
        {**current_params, 'lstm_units3': current_params.get('lstm_units3', 16) + 8},
        # Adjust dropout
        {**current_params, 'dropout_rate': max(0.1, current_params.get('dropout_rate', 0.3) - 0.1)},
        {**current_params, 'dropout_rate': min(0.5, current_params.get('dropout_rate', 0.3) + 0.1)},
        # Decrease capacity
        {**current_params, 'lstm_units1': max(32, current_params.get('lstm_units1', 64) - 16)},
        {**current_params, 'lstm_units2': max(16, current_params.get('lstm_units2', 32) - 8)}
    ]
    
    best_loss = np.inf
    best_model = current_model
    best_params = current_params
    input_shape = X.shape[1:]
    
    for i, params in enumerate(candidate_params):
        try:
            logging.info(f"Evaluating candidate model {i+1} with params: {params}")
            candidate_model = build_model_with_params(params, input_shape)
            if candidate_model is None:
                continue
                
            # Enhanced training with more robust callbacks
            callbacks = [
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=0
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    min_delta=1e-4,
                    restore_best_weights=True,
                    verbose=0
                )
            ]
            
            candidate_model.fit(
                X_train_candidate, y_train_candidate,
                epochs=5,
                batch_size=32,
                verbose=0,
                validation_data=(X_val_candidate, y_val_candidate),
                callbacks=callbacks
            )
            loss = candidate_model.evaluate(X_val_candidate, y_val_candidate, verbose=0)[0]
            logging.info(f"Candidate model {i+1} achieved validation loss {loss:.4f}")
            
            if loss < best_loss:
                best_loss = loss
                best_model = candidate_model
                best_params = params
        except Exception as e:
            logging.exception(f"Candidate params {params} failed due to error: {e}")
    
    if best_loss == np.inf:
        logging.warning("All candidate models failed. Retaining current model")
        return current_model, current_params
        
    logging.info(f"Best candidate achieved validation loss {best_loss:.4f} with params {best_params}")
    return best_model, best_params

# ------------------ Meta-Learner Functions ------------------

def build_meta_learner(input_dim):
    try:
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=input_dim))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        logging.exception(f"Error building meta-learner with input_dim {input_dim}: {e}")
        return None

# For simplicity the meta-learner remains global
baap_meta_learner = None
BAAP_META_FILENAME = "baap_meta_model.keras"

# ------------------ Continuous Trading Loop ------------------

def continuous_loop(exchange):
    global baap_meta_learner, historical_training_completed

    # Initialize the meta-learner (remains common for all markets)
    if os.path.exists(BAAP_META_FILENAME):
        try:
            baap_meta_learner_local = tf.keras.models.load_model(BAAP_META_FILENAME)
            baap_meta_learner = baap_meta_learner_local
        except Exception as e:
            logging.warning(f"Failed to load baap_meta_learner: {e}")
            baap_meta_learner = build_meta_learner(NUM_MC_SAMPLES * PREDICTION_HORIZON)
    else:
        baap_meta_learner = build_meta_learner(NUM_MC_SAMPLES * PREDICTION_HORIZON)

    # Load training progress
    progress = {
        'completed_markets': [],
        'historical_training_completed': False,
        'training_times': {}
    }
    if os.path.exists(BAAP_TRAINED_FLAG_FILENAME):
        try:
            with open(BAAP_TRAINED_FLAG_FILENAME, 'rb') as f:
                loaded_progress = pickle.load(f)
                progress.update(loaded_progress)
            historical_training_completed = progress['historical_training_completed']
            logging.info(f"Loaded training progress: {progress}")
        except Exception as e:
            logging.error(f"Error loading training progress: {e}")

    while True:
        try:
            logging.info("=== Starting new trading cycle ===")
            active_markets = fetch_active_symbols(exchange)
            if not active_markets:
                logging.error("No active markets found.")
                time.sleep(SLEEP_INTERVAL)
                continue

            open_positions_map = fetch_open_positions(exchange)
            current_open_trades = len(open_positions_map)

            for unified_symbol, market_id in active_markets:
                try:
                    current_open_trades = len(open_positions_map)  # Recalculate for each market
                    logging.info(f"Processing {market_id}: current open trades = {current_open_trades}")
                    # Skip trading new markets if max trades reached and no position exists
                    if current_open_trades >= MAX_OPEN_TRADES and market_id not in open_positions_map:
                        logging.info(f"{market_id} - Max open trades ({MAX_OPEN_TRADES}) reached. Skipping new trade.")
                        continue

                    df_live = fetch_binance_data(market_id, timeframe=TIMEFRAME, limit=LIMIT)
                    if df_live.empty or len(df_live) < WINDOW_SIZE + PREDICTION_HORIZON:
                        logging.warning(f"Not enough live data for {market_id}. Skipping...")
                        continue

                    # Ensure the latest candle is closed
                    now = pd.Timestamp.now(tz='UTC')
                    timeframe_minutes = int(TIMEFRAME.rstrip('m'))
                    candle_duration = pd.Timedelta(minutes=timeframe_minutes)
                    if df_live['timestamp'].iloc[-1] + candle_duration > now:
                        df_live = df_live.iloc[:-1]
                    if df_live.empty:
                        logging.warning(f"Live data for {market_id} has no closed candles. Skipping...")
                        continue

                    # Load or initialize model and scaler
                    model_file = get_model_filename(market_id)
                    scaler_file = get_scaler_filename(market_id)
                    model_exists = os.path.exists(model_file) and os.path.exists(scaler_file)

                    if market_id in market_models and market_id in market_scalers:
                        model = market_models[market_id]
                        scaler = market_scalers[market_id]
                    elif model_exists:
                        try:
                            model = tf.keras.models.load_model(model_file)
                            with open(scaler_file, 'rb') as f:
                                scaler = pickle.load(f)
                            market_models[market_id] = model
                            market_scalers[market_id] = scaler
                            logging.info(f"Loaded existing model and scaler for {market_id}.")
                        except Exception as e:
                            logging.error(f"Error loading model/scaler for {market_id}: {e}")
                            continue
                    else:
                        # Only train new model if below MAX_OPEN_TRADES
                        if current_open_trades >= MAX_OPEN_TRADES:
                            logging.info(f"{market_id} - No model exists, but max trades reached. Skipping training.")
                            continue

                        # Fetch historical data if not already present
                        historic_csv = get_historic_csv_filename(market_id)
                        if not os.path.exists(historic_csv):
                            logging.info(f"Fetching historical data for {market_id}...")
                            fetch_last_year_data(market_id, timeframe=TIMEFRAME, csv_filename=historic_csv)
                        if not os.path.exists(historic_csv):
                            logging.warning(f"Failed to fetch historical data for {market_id}. Skipping...")
                            continue

                        df_historic = pd.read_csv(historic_csv)
                        df_historic['timestamp'] = pd.to_datetime(df_historic['timestamp'])
                        X_hist, y_hist, scaler = preprocess_data(df_historic, window_size=WINDOW_SIZE)
                        if X_hist.size == 0:
                            logging.warning(f"Not enough historical data for {market_id}. Skipping...")
                            continue

                        # Initialize and train new model
                        X_train, X_val, y_train, y_val = train_test_split(X_hist, y_hist, test_size=0.2, shuffle=False)
                        input_shape = (X_train.shape[1], X_train.shape[2])
                        model = build_model(input_shape, DEFAULT_MODEL_PARAMS)
                        if model is None:
                            logging.error(f"Failed to build model for {market_id}. Skipping...")
                            continue

                        callbacks = [
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
                            EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4)
                        ]
                        batch_size = 32
                        train_gen = data_generator(X_train, y_train, batch_size, use_weights=True)
                        val_gen = data_generator(X_val, y_val, batch_size)
                        steps_per_epoch = max(1, len(X_train) // batch_size)
                        validation_steps = max(1, len(X_val) // batch_size)

                        logging.info(f"Training new model for {market_id}...")
                        model.fit(
                            train_gen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=10,
                            validation_data=val_gen,
                            validation_steps=validation_steps,
                            callbacks=callbacks,
                            verbose=1
                        )

                        model.save(model_file)
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(scaler, f)
                        market_models[market_id] = model
                        market_scalers[market_id] = scaler
                        market_model_params[market_id] = DEFAULT_MODEL_PARAMS.copy()

                        progress['completed_markets'].append(market_id)
                        progress['training_times'][market_id] = datetime.datetime.now()
                        with open(BAAP_TRAINED_FLAG_FILENAME, 'wb') as f:
                            pickle.dump(progress, f)
                        logging.info(f"New model trained and saved for {market_id}.")

                    # Preprocess live data with existing or new scaler
                    X_live, y_live, scaler_live = preprocess_data(df_live, window_size=WINDOW_SIZE, scaler=scaler, fit_scaler=False)
                    if X_live.size == 0:
                        logging.warning(f"Not enough processed live data for {market_id}. Skipping...")
                        continue

                    # Prediction and trading logic (unchanged from original)
                    last_window = X_live[-1].reshape(1, X_live.shape[1], X_live.shape[2])
                    candidate_preds_sequences = []
                    for _ in range(NUM_MC_SAMPLES):
                        pred_sequence_norm = model(last_window, training=True).numpy().flatten()
                        candidate_preds_sequences.append(pred_sequence_norm)
                    if not candidate_preds_sequences:
                        continue

                    candidate_vectors = np.array(candidate_preds_sequences)
                    meta_learner_input = candidate_vectors.reshape(1, -1)
                    final_pred_sequence_norm = baap_meta_learner.predict(meta_learner_input, verbose=0).flatten()
                    if np.isnan(final_pred_sequence_norm).any():
                        final_pred_sequence_norm = np.mean(candidate_vectors, axis=0)

                    last_close = df_live['close'].iloc[-1]
                    predicted_close_sequence = inverse_close(final_pred_sequence_norm, scaler)
                    predicted_close_horizon = predicted_close_sequence[-1]
                    pct_change_horizon = (predicted_close_horizon - last_close) / last_close

                    recent_returns = df_live['close'].pct_change().dropna().tail(20)
                    volatility = recent_returns.std() * np.sqrt(252) if not recent_returns.empty else 0.01
                    dynamic_threshold = max(0.002, volatility)

                    if pct_change_horizon > dynamic_threshold:
                        trend_direction = "UPWARD"
                        trend_confidence = min(1.0, pct_change_horizon / (2 * dynamic_threshold))
                    elif pct_change_horizon < -dynamic_threshold:
                        trend_direction = "DOWNWARD"
                        trend_confidence = min(1.0, -pct_change_horizon / (2 * dynamic_threshold))
                    else:
                        trend_direction = "FLAT"
                        trend_confidence = 0.0

                    if market_id not in globals().get('cumulative_pct_change', {}):
                        cumulative_pct_change = {}
                        globals()['cumulative_pct_change'] = cumulative_pct_change
                    if market_id not in cumulative_pct_change:
                        cumulative_pct_change[market_id] = 0.0

                    decay_factor = 0.5
                    cumulative_pct_change[market_id] = (cumulative_pct_change[market_id] * decay_factor) + pct_change_horizon
                    signal = "HOLD"
                    strong_signal_threshold = dynamic_threshold * 1.2
                    cumulative_threshold = dynamic_threshold * 2.0
                    if (trend_direction == "UPWARD" and
                        (pct_change_horizon > strong_signal_threshold or cumulative_pct_change[market_id] > cumulative_threshold) and
                        trend_confidence > 0.9):
                        signal = "BUY"
                        cumulative_pct_change[market_id] = 0.0
                    elif (trend_direction == "DOWNWARD" and
                        (abs(pct_change_horizon) > strong_signal_threshold or abs(cumulative_pct_change[market_id]) > cumulative_threshold) and
                        trend_confidence > 0.9):
                        signal = "SELL"
                        cumulative_pct_change[market_id] = 0.0

                    # Position management (unchanged)
                    adverse_threshold = -0.005
                    exit_executed = False
                    if market_id in open_positions_map:
                        pos = open_positions_map[market_id]
                        current_amt = float(pos['info'].get('positionAmt', 0))
                        current_signal = "BUY" if current_amt > 0 else "SELL"
                        entry_price = float(pos['info'].get('entryPrice', last_close))
                        pct_change_since_entry = (last_close - entry_price) / entry_price if current_signal == "BUY" else (entry_price - last_close) / entry_price

                        if pct_change_since_entry < adverse_threshold or (signal in ["BUY", "SELL"] and signal != current_signal):
                            exit_side = "sell" if current_signal == "BUY" else "buy"
                            exit_quantity = abs(current_amt)
                            exit_order = place_order(exchange, market_id, exit_side, exit_quantity, DEFAULT_LEVERAGE, last_close)
                            if exit_order:
                                open_positions_map.pop(market_id, None)
                                exit_executed = True

                    if not exit_executed and market_id not in open_positions_map and signal in ["BUY", "SELL"]:
                        quantity = get_order_quantity(last_close)
                        stop_loss_price = last_close * (0.995 if signal == "BUY" else 1.005)
                        take_profit_price = last_close * (1.01 if signal == "BUY" else 0.99)
                        entry_order = place_order(exchange, market_id, signal.lower(), quantity, DEFAULT_LEVERAGE, last_close,
                                                stop_loss_price, take_profit_price)
                        if entry_order:
                            open_positions_map[market_id] = {
                                'symbol': market_id,
                                'info': {'positionAmt': str(quantity if signal == "BUY" else -quantity),
                                        'entryPrice': str(last_close)}
                            }

                    # Enqueue retraining for existing models (not new ones if max trades reached)
                    if market_id in open_positions_map and (training_queue.empty() or np.random.random() < 0.2):
                        training_queue.put((market_id, X_live, y_live, scaler), block=False)
                        logging.info(f"Enqueued retraining task for {market_id}.")

                    K.clear_session()
                except Exception as market_e:
                    logging.exception(f"Error processing market {market_id}: {market_e}")
                    continue

            # Update historical training flag if all active markets have models
            if all(market_id in progress['completed_markets'] for _, market_id in active_markets):
                progress['historical_training_completed'] = True
                with open(BAAP_TRAINED_FLAG_FILENAME, 'wb') as f:
                    pickle.dump(progress, f)
                historical_training_completed = True

            time.sleep(SLEEP_INTERVAL)
            gc.collect()
        except Exception as cycle_e:
            logging.exception(f"Unexpected error in trading cycle: {cycle_e}")
            time.sleep(SLEEP_INTERVAL)
# ------------------ Main ------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        exchange = ccxt.binanceusdm({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    except Exception as e:
        logging.exception(f"Failed to initialize exchange: {e}")
        return

    # Start historical data fetching thread
    data_thread = threading.Thread(target=fetch_and_store_historic_data_threaded, daemon=True)
    data_thread.start()
    logging.info("Started historical data download thread.")

    # Start training worker thread
    training_thread = threading.Thread(target=training_worker, daemon=True)
    training_thread.start()
    logging.info("Training worker thread started.")

    # Run continuous loop in main thread
    continuous_loop(exchange)  # No separate thread, runs in main to avoid daemon issues

if __name__ == '__main__':
    main()