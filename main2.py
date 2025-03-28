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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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
MAX_OPEN_TRADES = 2
WINDOW_SIZE = 5
TIMEFRAME = '3m'
LIMIT = 300
THRESHOLD = 0.005  # Lowered from 0.005 to 0.2% minimum threshold
PREDICTION_HORIZON = 3  # Reduced from 5 for more responsive signals
SLEEP_INTERVAL = 20

PERFORMANCE_CHECK_INTERVAL = 10
ERROR_THRESHOLD = 0.01

NUM_MC_SAMPLES = 5
FIXED_THRESHOLD = 0.01

DEFAULT_MODEL_PARAMS = {
    'lstm_units1': 64,    # Increased from 50 for better feature extraction
    'lstm_units2': 32,    # Increased from 25
    'lstm_units3': 16,    # New layer for deeper temporal processing
    'dropout_rate': 0.3,
    'attention_heads': 4  # New parameter for multi-head attention
}

# Instead of global baap_model/scaler, we use per-market dictionaries.
market_models = {}      # key: market_id, value: model
market_scalers = {}     # key: market_id, value: scaler
market_model_params = {}  # key: market_id, value: model parameters
market_locks = {}  # Add this at the global scope
# Add these at the top of the file, after imports and before other global parameters
market_leverages = {}  # Dictionary to store leverage per market_id
leverage_options = [20, 10,8]  # List of leverage values to try
# This file now stores a dict with keys: 'completed_markets' (list) and 'historical_training_completed' (bool)
BAAP_TRAINED_FLAG_FILENAME = "baap_model_trained_flag.pkl"

historical_training_completed = False

# Conditional decorator based on TensorFlow version
if hasattr(tf.keras, 'saving'):
    register_decorator = tf.keras.saving.register_keras_serializable(package="CustomLayers")
else:
    register_decorator = tf.keras.utils.register_keras_serializable(package="CustomLayers")

@register_decorator
class LearnableBiasedAttention(Attention):
    def __init__(self, sequence_length, **kwargs):
        super(LearnableBiasedAttention, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.bias = self.add_weight(
            shape=(sequence_length,),
            initializer='zeros',
            trainable=True,
            name='attention_bias'
        )

    def _calculate_scores(self, query, key):
        scores = super(LearnableBiasedAttention, self)._calculate_scores(query, key)
        bias = tf.expand_dims(self.bias, axis=0)
        scores += bias
        return scores

    def get_config(self):
        config = super(LearnableBiasedAttention, self).get_config()
        config.update({'sequence_length': self.sequence_length})
        return config
# ------------------ Helper Functions for Unique Filenames ------------------
def set_leverage_with_fallback(exchange, market_id, leverage_options):
    """Attempts to set leverage for a market_id, starting with the highest value and falling back."""
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
        return market_leverages[market_id]  # Use stored leverage if available
    else:
        leverage = set_leverage_with_fallback(exchange, market_id, leverage_options)
        if leverage is not None:
            market_leverages[market_id] = leverage  # Store successful leverage
        return leverage
def get_model_filename(market_id):
    return f"model_{market_id}_ws{WINDOW_SIZE}.keras"

def get_scaler_filename(market_id):
    return f"scaler_{market_id}_ws{WINDOW_SIZE}.pkl"

def get_historic_csv_filename(market_id):
    return f"{market_id}_historic.csv"

# ------------------ Training Queue Setup ------------------
# Create a queue with a maxsize of 1 to avoid duplicate training tasks.
# Global variables
# training_queue = queue.Queue(maxsize=5)
historical_training_queue = queue.Queue(maxsize=5)  # For initial historical training
retraining_queue = queue.Queue(maxsize=5)          # For periodic retraining
training_in_progress = set()                        # Shared set to track ongoing training
training_lock = threading.Lock()                    # Ensures thread-safe access to the set




def self_improve_model_hyperopt(current_model, X, y, scaler, current_params):
    """
    Automated hyperparameter tuning using Hyperopt for the LSTM model.
    Splits the provided data, defines an objective function for Hyperopt,
    and returns the best model and parameters found.
    """
    logging.info("Starting automated hyperparameter tuning with Hyperopt for LSTM model")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    input_shape = X.shape[1:]  # (window_size, num_features)
    
    def objective(params):
        # Clear TensorFlow state to prevent session/graph corruption
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        
        # Build a candidate model with sampled hyperparameters
        trial_params = {
            "lstm_units1": int(params["lstm_units1"]),
            "lstm_units2": int(params["lstm_units2"]),
            "lstm_units3": int(params["lstm_units3"]),
            "dropout_rate": params["dropout_rate"]
        }
        learning_rate = params["learning_rate"]
        epochs = int(params["epochs"])
        
        model = build_model_with_params(trial_params, input_shape, sequence_length=WINDOW_SIZE)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse', metrics=['mae'])
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4,
                                   restore_best_weights=True, verbose=0)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks)
        # Return validation loss
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        return {'loss': val_loss, 'status': STATUS_OK}
    
    # Define hyperparameter search space
    space = {
        "lstm_units1": hp.quniform("lstm_units1", 32, 128, 16),
        "lstm_units2": hp.quniform("lstm_units2", 16, 64, 8),
        "lstm_units3": hp.quniform("lstm_units3", 16, 32, 8),
        "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
        "epochs": hp.quniform("epochs", 5, 15, 1)
    }
    
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    # Convert best values to appropriate types
    best_params = {
        "lstm_units1": int(best["lstm_units1"]),
        "lstm_units2": int(best["lstm_units2"]),
        "lstm_units3": int(best["lstm_units3"]),
        "dropout_rate": best["dropout_rate"],
        "learning_rate": best["learning_rate"],
        "epochs": int(best["epochs"])
    }
    logging.info(f"Hyperopt found best parameters: {best_params}")
    
    # Clear state again before building the final model
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    # Rebuild the best model with the best hyperparameters
    best_model = build_model_with_params(best_params, input_shape, sequence_length=WINDOW_SIZE)
    best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
                       loss='mse', metrics=['mae'])
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=10, batch_size=32, verbose=0, callbacks=[
                       EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-4,
                                     restore_best_weights=True, verbose=0)
                   ])
    return best_model, best_params

def historical_training_worker():
    """Worker thread that processes historical training tasks."""
    while True:
        try:
            task = historical_training_queue.get()
            if task is None:  # Shutdown signal
                break
            market_id = task['market_id']
            historic_csv = get_historic_csv_filename(market_id)
            if not os.path.exists(historic_csv):
                logging.warning(f"No historical data for {market_id}. Skipping.")
                historical_training_queue.task_done()
                continue
            df_market = pd.read_csv(historic_csv)
            df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])
            X, y, scaler = preprocess_data(df_market, window_size=WINDOW_SIZE, fit_scaler=True)
            if X.size == 0:
                logging.warning(f"Not enough data for {market_id}. Skipping.")
                historical_training_queue.task_done()
                continue

            input_shape = (X.shape[1], X.shape[2])
            model = build_model(input_shape, sequence_length=WINDOW_SIZE, model_params=DEFAULT_MODEL_PARAMS)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
                EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4, verbose=1)
            ]
            logging.info(f"Starting historical training for {market_id}...")
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=callbacks, verbose=1)
            model.save(get_model_filename(market_id))
            with open(get_scaler_filename(market_id), 'wb') as f:
                pickle.dump(scaler, f)
            market_models[market_id] = model
            market_scalers[market_id] = scaler
            logging.info(f"Historical training completed for {market_id}")

            with training_lock:
                training_in_progress.remove(market_id)
            historical_training_queue.task_done()
        except Exception as e:
            logging.exception(f"Error in historical training worker for task {task}: {e}")
            with training_lock:
                if market_id in training_in_progress:
                    training_in_progress.discard(market_id)
            historical_training_queue.task_done()

# Updated training_worker function using Hyperopt for live retraining:
def retraining_worker():
    """Worker thread that processes retraining tasks with live data."""
    while True:
        try:
            task = retraining_queue.get()
            if task is None:  # Shutdown signal
                break
            market_id = task['market_id']
            X_live = task['X']
            y_live = task['y']
            scaler_live = task['scaler']
            if market_id not in market_models:
                logging.warning(f"Model for {market_id} not found. Cannot retrain.")
                retraining_queue.task_done()
                continue
            current_model = market_models[market_id]
            current_params = market_model_params.get(market_id, DEFAULT_MODEL_PARAMS.copy())
            
            # Perform retraining with Hyperopt
            improved_model, best_params = self_improve_model_hyperopt(current_model, X_live, y_live, scaler_live, current_params)
            
            # Update global dictionaries
            market_models[market_id] = improved_model
            market_model_params[market_id] = best_params
            logging.info(f"Retraining with Hyperopt tuning completed for {market_id}")

            with training_lock:
                training_in_progress.remove(market_id)
            retraining_queue.task_done()
        except Exception as e:
            logging.exception(f"Error in retraining worker for task {task}: {e}")
            with training_lock:
                if market_id in training_in_progress:
                    training_in_progress.discard(market_id)
            retraining_queue.task_done()

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

def data_generator(X, y, batch_size, use_weights=True, decay=0.9):  # Adjusted decay from 0.99 to 0.95
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
        # if last_train_time and (current_time - last_train_time).total_seconds() < 24 * 60 * 60:
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
                    model_local = build_model(input_shape, sequence_length=WINDOW_SIZE, model_params=DEFAULT_MODEL_PARAMS)
                else:
                    logging.info(f"Loaded existing model for {market_id}.")
            except Exception as e:
                logging.error(f"Error loading model for {market_id}: {e}. Re-initializing.")
                K.clear_session()
                model_local = build_model(input_shape, sequence_length=WINDOW_SIZE, model_params=DEFAULT_MODEL_PARAMS)
        else:
            K.clear_session()
            model_local = build_model(input_shape, sequence_length=WINDOW_SIZE, model_params=DEFAULT_MODEL_PARAMS)
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

def get_order_quantity(current_price, leverage):
    """Calculates order quantity based on current price and leverage."""
    try:
        quantity = (AMOUNT_USD * leverage) / current_price
        notional = quantity * current_price
        min_notional = 19  # Minimum 5 USDT as required by Binance
        if notional < min_notional:
            quantity = min_notional / current_price
        return quantity
    except Exception as e:
        logging.exception(f"Error calculating order quantity: {e}")
        return 0

def place_order(exchange, market_id, side, quantity, current_price, stop_loss_price=None, take_profit_price=None):
    # try:
    #     # Set leverage for the symbol explicitly
    #     exchange.set_leverage(leverage, market_id)
    # except Exception as e:
    #     logging.exception(f"Failed to set leverage for {market_id}: {e}")
    #     return None
    params = {}

    if stop_loss_price:
        params['stopLoss'] = {'price': stop_loss_price, 'type': 'MARKET'}
    if take_profit_price:
        params['takeProfit'] = {'price': take_profit_price, 'type': 'MARKET'}

    # If no take profit price is provided, calculate it so that profit is $0.50
    # if take_profit_price is None:
    #     if side.lower() == 'buy':
    #         take_profit_price = current_price + (0.8 * current_price) / (AMOUNT_USD * leverage)
    #     elif side.lower() == 'sell':
    #         take_profit_price = current_price - (0.8 * current_price) / (AMOUNT_USD * leverage)
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
        # takeProfitOrder = exchange.create_order(market_id, 'TAKE_PROFIT_MARKET', take_profit_side, quantity, price, takeProfitParams)
        # logging.info(f"Profit Order placed for {market_id}: {takeProfitOrder}")
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
        dummy[:, 3] = normalized_close
        inv = scaler.inverse_transform(dummy)
        return inv[:, 3]
    except Exception as e:
        logging.exception(f"Error in inverse transforming 'close': {e}")
        return normalized_close

# ------------------ Model Building Functions ------------------

def build_model_with_params(model_params, input_shape, sequence_length):
    try:
        inputs = Input(shape=input_shape)
        
        # First LSTM layer with return sequences
        x = LSTM(
            model_params.get('lstm_units1', 64),
            return_sequences=True,
            recurrent_dropout=0.1
        )(inputs)
        x = Dropout(model_params.get('dropout_rate', 0.3))(x)
        
        # Replace standard Attention with LearnableBiasedAttention
        attention = LearnableBiasedAttention(sequence_length=sequence_length)([x, x])
        
        # Second LSTM layer with return sequences
        x = LSTM(
            model_params.get('lstm_units2', 32),
            return_sequences=True,
            recurrent_dropout=0.1
        )(attention)
        x = Dropout(model_params.get('dropout_rate', 0.3))(x)
        
        # Third LSTM layer without return sequences
        x = LSTM(
            model_params.get('lstm_units3', 16),
            return_sequences=False
        )(x)
        x = Dropout(model_params.get('dropout_rate', 0.3))(x)
        
        # Output layer
        outputs = Dense(PREDICTION_HORIZON, activation='tanh')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    except Exception as e:
        logging.exception(f"Error building model with params {model_params}: {e}")
        return None

def build_model(input_shape, sequence_length, model_params=DEFAULT_MODEL_PARAMS):
    return build_model_with_params(model_params, input_shape, sequence_length)
    

# ------------------ Self-Improvement Functions ------------------

# ------------------ Automated Hyperparameter Tuning with Optuna ------------------

def self_improve_model(current_model, X, y, scaler, current_params):
    """
    Automated hyperparameter tuning for the LSTM model using Optuna.
    This function splits the provided data, defines an objective function for Optuna,
    and returns the best model and parameters found.
    """
    logging.info("Starting automated hyperparameter tuning with Optuna for LSTM model")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    input_shape = X.shape[1:]  # (window_size, num_features)
    
    def objective(trial):
        tf.keras.backend.clear_session()
        # Define hyperparameters to tune
        lstm_units1 = trial.suggest_int("lstm_units1", 32, 128, step=16)
        lstm_units2 = trial.suggest_int("lstm_units2", 16, 64, step=8)
        lstm_units3 = trial.suggest_int("lstm_units3", 16, 32, step=8)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 5, 15)
        
        # Build model with the sampled hyperparameters
        trial_params = {
            "lstm_units1": lstm_units1,
            "lstm_units2": lstm_units2,
            "lstm_units3": lstm_units3,
            "dropout_rate": dropout_rate
        }
        model = build_model_with_params(trial_params, input_shape, sequence_length=WINDOW_SIZE)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        
        # Train the model
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4, restore_best_weights=True, verbose=0)]
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks)
        # Evaluate and return validation loss
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params
    best_loss = study.best_value
    logging.info(f"Optuna found best parameters: {best_params} with loss: {best_loss:.4f}")
    
    # Rebuild the model using the best hyperparameters
    best_model = build_model_with_params(best_params, input_shape, sequence_length=WINDOW_SIZE)
    best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params.get("learning_rate", 0.001)), loss='mse', metrics=['mae'])
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0, callbacks=[
         EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-4, restore_best_weights=True, verbose=0)
    ])
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
    global baap_meta_learner
    
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

    while True:
        try:
            logging.info("=== Starting new trading cycle ===")
            try:
                active_markets = fetch_active_symbols(exchange)
            except Exception as e:
                logging.exception(f"Failed to fetch active symbols: {e}")
                time.sleep(SLEEP_INTERVAL)
                continue
            if not active_markets:
                logging.error("No active markets found.")
                time.sleep(SLEEP_INTERVAL)
                continue
            try:
                open_positions_map = fetch_open_positions(exchange)
            except Exception as e:
                logging.exception(f"Failed to fetch open positions: {e}")
                open_positions_map = {}
            for unified_symbol, market_id in active_markets:
                try:
                    if len(open_positions_map) >= MAX_OPEN_TRADES and market_id not in open_positions_map:
                        continue
                    if market_id not in market_models or market_id not in market_scalers:
                        model_file = get_model_filename(market_id)
                        scaler_file = get_scaler_filename(market_id)
                        if os.path.exists(model_file) and os.path.exists(scaler_file):
                            try:
                                model = tf.keras.models.load_model(model_file)
                                with open(scaler_file, 'rb') as f:
                                    scaler = pickle.load(f)
                                market_models[market_id] = model
                                market_scalers[market_id] = scaler
                            except Exception as e:
                                logging.error(f"Error loading model/scaler for {market_id}: {e}")
                                continue
                        else:
                            # Enqueue historical training if conditions allow
                            with training_lock:
                                if (market_id not in training_in_progress and 
                                    len(open_positions_map) < MAX_OPEN_TRADES):
                                    try:
                                        historical_training_queue.put(
                                            {'market_id': market_id, 'is_historical': True},
                                            block=False
                                        )
                                        training_in_progress.add(market_id)
                                        logging.info(f"Enqueued historical training task for {market_id}")
                                    except queue.Full:
                                        logging.warning(f"Training queue full. Cannot enqueue {market_id}")
                            continue
                    else:
                        model = market_models[market_id]
                        scaler = market_scalers[market_id]
                    # Fetch and process live data
                    df_live = fetch_binance_data(market_id, timeframe=TIMEFRAME, limit=LIMIT)
                    df_live.dropna(inplace=True)
                    if df_live.empty:
                        logging.warning(f"Not enough live data for {market_id}. Skipping...")
                        continue
                    # now = pd.Timestamp.now(tz='UTC')
                    # try:
                    #     timeframe_minutes = int(TIMEFRAME.rstrip('m'))
                    # except Exception:
                    #     timeframe_minutes = 3
                    # candle_duration = pd.Timedelta(minutes=timeframe_minutes)
                    # if df_live['timestamp'].iloc[-1].tzinfo is None:
                    #     df_live['timestamp'] = df_live['timestamp'].dt.tz_localize('UTC')
                    # if df_live['timestamp'].iloc[-1] + candle_duration > now:
                    #     df_live = df_live.iloc[:-1]
                    if df_live.empty:
                        logging.warning(f"Live data for {market_id} has no closed candles. Skipping...")
                        continue
                    df_live['close_delta'] = df_live['close'].diff()
                    df_live['volume_delta'] = df_live['volume'].diff()

                    # Load or retrieve the model and scaler for this market
                    if market_id not in market_models or market_id not in market_scalers:
                        model_file = get_model_filename(market_id)
                        scaler_file = get_scaler_filename(market_id)
                        if os.path.exists(model_file) and os.path.exists(scaler_file):
                            try:
                                model = tf.keras.models.load_model(model_file)
                                with open(scaler_file, 'rb') as f:
                                    scaler = pickle.load(f)
                                market_models[market_id] = model
                                market_scalers[market_id] = scaler
                            except Exception as e:
                                logging.error(f"Error loading model/scaler for market {market_id}: {e}")
                                continue
                        else:
                            logging.error(f"Model or scaler for market {market_id} not found. Skipping market.")
                            continue
                    else:
                        model = market_models[market_id]
                        scaler = market_scalers[market_id]

                    X_live, y_live, scaler_live = preprocess_data(df_live, window_size=WINDOW_SIZE, scaler=scaler, fit_scaler=False)
                    if X_live.size == 0:
                        logging.warning(f"Not enough processed live data for {market_id}. Skipping...")
                        continue

                    last_window = X_live[-1].reshape(1, X_live.shape[1], X_live.shape[2])
                    candidate_preds_sequences = []
                    K.clear_session()
                    tf.compat.v1.reset_default_graph()  # Reset TensorFlow graph
                    for _ in range(NUM_MC_SAMPLES):
                        try:
                            # Enable dropout during prediction for MC sampling by setting training=True
                            pred_sequence_norm = model.predict(last_window, verbose=0)
                            candidate_preds_sequences.append(pred_sequence_norm.flatten())
                        except Exception as e:
                            logging.exception(f"Error during candidate prediction for {market_id}: {e}")
                    if not candidate_preds_sequences:
                        logging.warning(f"No candidate predictions for {market_id}. Skipping...")
                        continue
                    candidate_vectors = np.array(candidate_preds_sequences)
                    try:
                        meta_learner_input = candidate_vectors.reshape(1, -1)
                        final_pred_sequence_norm = baap_meta_learner.predict(meta_learner_input).flatten()
                    except Exception as e:
                        logging.exception(f"Meta-learner prediction error for {market_id}: {e}")
                        final_pred_sequence_norm = np.mean(candidate_vectors, axis=0)
                    if np.isnan(final_pred_sequence_norm).any() or np.all(final_pred_sequence_norm == 0):
                        final_pred_sequence_norm = np.mean(candidate_vectors, axis=0)
                    try:
                        last_close_norm = X_live[-1, -1, 3]
                        last_close = inverse_close(np.array([last_close_norm]), scaler)[0]
                        #last_close = df_live['close'].iloc[-1]
                        predicted_close_sequence = inverse_close(final_pred_sequence_norm, scaler)
                        # Calculate mean of predicted close sequence
                        # mean_predicted_close = np.mean(predicted_close_sequence)
                        # mean_pct_change = (mean_predicted_close - last_close) / last_close
                        # Use the predicted close at the end of the horizon
                        predicted_close_horizon = predicted_close_sequence[-1]
                        pct_change_horizon = (predicted_close_horizon - last_close) / last_close
                        # Calculate volatility
                        recent_returns = df_live['close'].pct_change().dropna().tail(5)
                        volatility = recent_returns.std() * np.sqrt(252) if not recent_returns.empty else 0.01
                        if np.isnan(volatility) or volatility == 0:
                            volatility = 0.01
                        # Set base_threshold to max(0.005, volatility)
                        base_threshold = max(0.002, volatility)
                        dynamic_threshold = base_threshold
                        # Determine trend direction and confidence
                        # Determine trend direction and confidence
                        signal = "HOLD"
                        if pct_change_horizon > dynamic_threshold:
                            signal = "BUY"
                        elif pct_change_horizon < -dynamic_threshold:
                            signal = "SELL"
                            trend_confidence = 0.0
                    except Exception as e:
                        logging.exception(f"Error in inverse transforming predictions for {market_id}: {e}")
                        continue
                    # logging.info(f"{market_id} - Trend Direction: {trend_direction} (Confidence: {trend_confidence:.2%}), "
                    #          f"Pct Change Horizon: {pct_change_horizon:.4f}, Volatility: {volatility:.4f}, Dynamic Threshold: {dynamic_threshold:.4f}")
                    # (Cumulative change logic remains unchanged)
                    if market_id not in globals().get('cumulative_pct_change', {}):
                        cumulative_pct_change = {}
                        globals()['cumulative_pct_change'] = cumulative_pct_change
                    if market_id not in cumulative_pct_change:
                        cumulative_pct_change[market_id] = 0.0

                    # Update cumulative change with decay
                    decay_factor = 0.9
                    cumulative_pct_change[market_id] = (cumulative_pct_change[market_id] * decay_factor) + pct_change_horizon
                    # signal = "HOLD"
                    # strong_signal_threshold = dynamic_threshold * 1.2
                    # cumulative_threshold = dynamic_threshold * 2.0
                    # if (trend_direction == "UPWARD" and
                    #     (pct_change_horizon > strong_signal_threshold or cumulative_pct_change[market_id] > cumulative_threshold) and
                    #     trend_confidence > 0.9):
                    #     signal = "SELL"
                    #     cumulative_pct_change[market_id] = 0.0
                    # elif (trend_direction == "DOWNWARD" and
                    #     (abs(pct_change_horizon) > strong_signal_threshold or abs(cumulative_pct_change[market_id]) > cumulative_threshold) and
                    #     trend_confidence > 0.9):
                    #     signal = "BUY"
                    #     cumulative_pct_change[market_id] = 0.0
                    logging.info(f"{market_id} - Signal: {signal}")
                    adverse_threshold = -0.005  # Increased to 0.5% to match stop loss
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
                            if leverage is None:
                                logging.error(f"Cannot determine leverage for {market_id}. Skipping exit order.")
                                continue
                            logging.info(f"{market_id} - Exiting position ({current_signal}) with quantity {exit_quantity:.4f}.")
                            exit_order = place_order(exchange, market_id, exit_side, exit_quantity, last_close_live)
                            if exit_order:
                                logging.info(f"{market_id} - Exit order executed.")
                                open_positions_map.pop(market_id, None)
                                exit_executed = True
                            else:
                                logging.warning(f"{market_id} - Exit order failed.")
                    if not exit_executed and market_id not in open_positions_map and signal in ["BUY", "SELL"]:
                        leverage = get_leverage_for_market(exchange, market_id)
                        if leverage is None:
                            logging.error(f"Cannot determine leverage for {market_id}. Skipping entry order.")
                            continue
                        
                        last_close_live = df_live['close'].iloc[-1] 
                        quantity = get_order_quantity(last_close_live, leverage)                       
                        stop_loss_price = last_close_live * (0.995 if signal == "BUY" else 1.005)
                        take_profit_price = last_close_live * (1.01 if signal == "BUY" else 0.99)
                        logging.info(f"{market_id} - Entering new position: signal={signal}, qty={quantity:.4f}, SL={stop_loss_price:.4f}, TP={take_profit_price:.4f}.")
                        entry_order = place_order(exchange, market_id, signal.lower(), quantity, last_close_live, stop_loss_price, take_profit_price)
                        if entry_order:
                            logging.info(f"{market_id} - Entry order executed.")
                            open_positions_map[market_id] = {'symbol': market_id,
                                                            'info': {'positionAmt': str(quantity if signal == "BUY" else -quantity),
                                                                    'entryPrice': str(last_close_live)}}
                        else:
                            logging.warning(f"{market_id} - Entry order failed.")
                    # Enqueue a training task if none is already queued.
                    if (len(open_positions_map) < MAX_OPEN_TRADES and 
                        market_id not in training_in_progress):
                        with training_lock:
                            if market_id not in training_in_progress:
                                try:
                                    retraining_queue.put(
                                        {'market_id': market_id, 'X': X_live, 'y': y_live, 'scaler': scaler_live},
                                        block=False
                                    )
                                    training_in_progress.add(market_id)
                                    logging.info(f"Enqueued retraining task for {market_id}")
                                except queue.Full:
                                    logging.warning(f"Retraining queue full. Cannot enqueue {market_id}")
                    else:
                        logging.info("Training task already queued; skipping enqueue for now.")
                    logging.info(f"{market_id} - Signal processed using its model.\n")
                    K.clear_session()
                except Exception as market_e:
                    logging.exception(f"Error processing market {market_id}: {market_e}")
                    continue
            logging.info("Cycle complete for all active markets. Waiting for next cycle...\n")
            gc.collect()
            time.sleep(SLEEP_INTERVAL)
        except Exception as cycle_e:
            logging.exception(f"Unexpected error in trading cycle: {cycle_e}")
            time.sleep(SLEEP_INTERVAL)
            K.clear_session()
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

    global historical_training_completed
    historical_training_completed = False

    data_thread = threading.Thread(target=fetch_and_store_historic_data_threaded, daemon=True)
    data_thread.start()
    logging.info("Started historical data download thread.")

    # Start both training worker threads
    historical_training_thread = threading.Thread(target=historical_training_worker, daemon=True)
    historical_training_thread.start()
    logging.info("Historical training worker thread started.")

    retraining_thread = threading.Thread(target=retraining_worker, daemon=True)
    retraining_thread.start()
    logging.info("Retraining worker thread started.")

    live_thread = threading.Thread(target=continuous_loop, args=(exchange,), daemon=True)
    live_thread.start()
    logging.info("Live trading loop started in main thread.")

    while True:
        time.sleep(SLEEP_INTERVAL)
        gc.collect()

if __name__ == '__main__':
    main()