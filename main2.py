
import os
import time
import logging
import threading
import json
import queue
import ccxt # Still needed for exchange interaction (orders, initial fetch, OI)
import pandas as pd
import numpy as np
from collections import defaultdict, deque, Counter
from OrderBookOIScalpingIndicator import EnhancedOrderBookVolumeWrapper # Assuming this file exists and is compatible
import asyncio
import sys # Import sys to check the platform
# Import python-binance
from binance import ThreadedWebsocketManager, Client # Use Client for REST calls

# Configuration
SLEEP_INTERVAL = 2  # seconds - Main loop check interval (much shorter now)
MAX_OPEN_TRADES = 3
AMOUNT_USD = 1
TIMEFRAME = '3m' # Kline interval for websockets
LIMIT = 100 # History length for indicators
LEVERAGE_OPTIONS = [30, 25, 15, 8]
WEBSOCKET_DEPTH_LEVELS = 20 # Order book depth
WEBSOCKET_UPDATE_SPEED = '1000ms' # Or '100ms' for faster updates, more resource intensive

# Anti-cycling settings
MINIMUM_HOLD_MINUTES = 2
SYMBOL_COOLDOWN_MINUTES = 15

# --- WebSocket Data Storage ---
# Use dictionaries protected by locks to store the latest data from websockets
latest_kline_data = {}      # {symbol: deque(maxlen=LIMIT+10)} # Store raw kline list/dict
latest_depth_data = {}      # {symbol: {'bids': {price: qty}, 'asks': {price: qty}, 'lastUpdateId': id}}
latest_ticker_data = {}     # {symbol: ticker_dict}
last_ws_update_time = {}    # {symbol: timestamp} # Track freshness

kline_lock = threading.Lock()
depth_lock = threading.Lock()
ticker_lock = threading.Lock()
ws_update_time_lock = threading.Lock()
monitored_symbols = set()
# ----------------------------

# Thread synchronization (existing)
symbol_cooldowns = {}
position_entry_times = {}
cooldown_lock = threading.Lock()
consecutive_stop_losses = {}
consecutive_stop_loss_lock = threading.Lock()
position_details = {}
position_details_lock = threading.Lock()

# Global metrics (existing)
global_metrics = {
    'patterns_detected': 0, 'patterns_below_threshold': 0, 'insufficient_tf_confirmation': 0,
    'failed_sequence_check': 0, 'failed_reversal_check': 0, 'failed_risk_reward': 0,
    'failed_proximity_check': 0, 'order_placement_errors': 0, 'successful_entries': 0,
    'last_reset': time.time()
}

# Logger setup (existing)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories (existing)
PARAMS_DIR = 'optimized_params'
os.makedirs(PARAMS_DIR, exist_ok=True)
TRADE_LOG_DIR = 'trade_logs'
os.makedirs(TRADE_LOG_DIR, exist_ok=True)
PATTERN_STATS_DIR = 'pattern_stats'
os.makedirs(PATTERN_STATS_DIR, exist_ok=True)

# OHLCV cache is no longer needed for primary data, keep for potential fallbacks or OI polling context
ohlcv_cache = defaultdict(lambda: {'data': None, 'timestamp': 0})
CACHE_TTL = 60 # Reduced TTL as it's less critical now
cache_lock = threading.Lock()

# Indicator instances (existing)
symbol_indicators = {}
indicator_lock = threading.Lock()

# Default parameters (existing) - Update potentially if indicator changes
DEFAULT_PARAMS = {
    'ob_depth': WEBSOCKET_DEPTH_LEVELS, # Align with WS depth
    'ob_update_interval': 1.0 if WEBSOCKET_UPDATE_SPEED == '1000ms' else 0.1, # Align with WS speed
    'min_wall_size': 20, 'imbalance_threshold': 1.8,
    'oi_window': 24, 'oi_change_threshold': 1.0,
    'atr_window': 3, 'vol_multiplier': 1.2, 'min_vol': 0.0005,
    'risk_reward_ratio': 1.5, 'profit_target_pct': 0.3, 'stop_loss_pct': 0.2,
    'liquidation_oi_threshold': 15.0
}

# File writer queue (existing)
file_update_queue = queue.Queue()
file_writer_stop_event = threading.Event()

# --- WebSocket Callback Functions ---

def process_kline_message(msg):
    """Handles incoming kline messages."""
    try:
        if msg.get('e') == 'error':
            logger.error(f"Kline WS Error: {msg.get('m')}")
            return
        if 'k' not in msg:
            logger.debug(f"Ignoring non-kline message: {msg.get('e')}")
            return

        kline = msg['k']
        symbol = kline['s']
        is_closed = kline['x']
        start_time = kline['t']
        
        # Structure the data similarly to fetch_ohlcv output but as a dict
        kline_data = {
            'timestamp': pd.to_datetime(start_time, unit='ms', utc=True), # Pandas Timestamp for consistency
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'is_closed': is_closed # Add flag to know if candle is finished
        }

        with kline_lock:
            if symbol not in latest_kline_data:
                 # Initialize deque when first message arrives
                 # We fetch historical data initially, so deque should exist
                 # If not, maybe log an error or initialize here cautiously
                 logger.warning(f"Received kline for {symbol} but no deque initialized. Initial fetch might have failed.")
                 latest_kline_data[symbol] = deque(maxlen=LIMIT + 10) # Add buffer

            current_deque = latest_kline_data[symbol]

            if len(current_deque) > 0:
                last_candle_in_deque = current_deque[-1]
                # If the new kline has the same start time, update the last element
                if last_candle_in_deque['timestamp'].timestamp() * 1000 == start_time:
                    current_deque[-1] = kline_data # Replace the last (ongoing) candle
                # If the new kline is a *new* candle (different start time)
                else:
                    # Only append if it's truly the next candle sequentially
                    # (Handle potential duplicates or out-of-order messages)
                    time_diff = (kline_data['timestamp'] - last_candle_in_deque['timestamp']).total_seconds()
                    expected_diff = ccxt.Exchange.parse_timeframe(TIMEFRAME)
                    if time_diff >= expected_diff: # Allow for slight timing variations
                         current_deque.append(kline_data)
                    else:
                        logger.warning(f"Ignoring potentially out-of-order/duplicate kline for {symbol}. Timestamps: {last_candle_in_deque['timestamp']} -> {kline_data['timestamp']}")

            else:
                # Deque is empty (should only happen briefly at startup before history is loaded)
                current_deque.append(kline_data)


        with ws_update_time_lock:
            last_ws_update_time[symbol] = time.time()
            
        # logger.debug(f"Processed kline for {symbol}: {'Closed' if is_closed else 'Ongoing'} @ {kline_data['close']}")

    except Exception as e:
        logger.error(f"Error processing kline message: {e}\nMessage: {msg}", exc_info=False)


# Helper to manage depth updates efficiently
def update_depth_cache(symbol, bids, asks, last_update_id):
     with depth_lock:
        if symbol not in latest_depth_data:
             # Initialize structure if not present (e.g., on first diff after snapshot)
             latest_depth_data[symbol] = {'bids': {}, 'asks': {}, 'lastUpdateId': 0}

        # Apply bid updates
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0:
                if price in latest_depth_data[symbol]['bids']:
                    del latest_depth_data[symbol]['bids'][price]
            else:
                latest_depth_data[symbol]['bids'][price] = qty

        # Apply ask updates
        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0:
                if price in latest_depth_data[symbol]['asks']:
                    del latest_depth_data[symbol]['asks'][price]
            else:
                latest_depth_data[symbol]['asks'][price] = qty

        # Keep bids sorted high to low, asks low to high (optional but good practice)
        latest_depth_data[symbol]['bids'] = dict(sorted(latest_depth_data[symbol]['bids'].items(), reverse=True))
        latest_depth_data[symbol]['asks'] = dict(sorted(latest_depth_data[symbol]['asks'].items()))

        # Update the last update ID
        latest_depth_data[symbol]['lastUpdateId'] = last_update_id

def process_depth_message(msg):
    """Handles incoming depth messages (diffs)."""
    try:
        if msg.get('e') == 'error':
            logger.error(f"Depth WS Error: {msg.get('m')}")
            return
        if msg.get('e') != 'depthUpdate':
             logger.debug(f"Ignoring non-depthUpdate message: {msg.get('e')}")
             return

        symbol = msg['s']
        first_update_id = msg['U']
        final_update_id = msg['u']
        bids_update = msg['b']
        asks_update = msg['a']

        with depth_lock:
            # Check if we have a cache and if the update aligns
            if symbol in latest_depth_data:
                cached_last_update_id = latest_depth_data[symbol].get('lastUpdateId', 0)

                # Standard Binance logic: Drop events where msg['U'] <= lastUpdateId
                # Process first event where msg['U'] <= lastUpdateId + 1 and msg['u'] >= lastUpdateId + 1
                if final_update_id <= cached_last_update_id:
                     # logger.debug(f"Depth update for {symbol} too old (u={final_update_id} <= cache={cached_last_update_id}). Skipping.")
                     return # Ignore old update

                if first_update_id > cached_last_update_id + 1:
                    # Gap detected! Need to resync.
                    logger.warning(f"Depth cache gap detected for {symbol} (U={first_update_id} > cache={cached_last_update_id}+1). Clearing cache, resync needed.")
                    # Clear cache - A resync mechanism should ideally fetch a new snapshot
                    # For now, we just clear and wait for the next update that fits.
                    # A robust solution would trigger a snapshot fetch here.
                    del latest_depth_data[symbol]
                    # Maybe need to signal main loop to avoid using this symbol's depth until resynced
                    return

                # If we are here, the update is okay to apply (U <= L+1 and u >= L+1)
                update_depth_cache(symbol, bids_update, asks_update, final_update_id)
                # logger.debug(f"Updated depth cache for {symbol} via diff (u={final_update_id})")

            else:
                # No cache yet, likely waiting for initial snapshot fetch to complete.
                # We can't apply diffs without a base.
                logger.debug(f"Received depth diff for {symbol} but cache not initialized. Waiting for snapshot.")
                # Log when the update *would* have been applied if cache existed
                # This helps debug if snapshot fetch is slow/failing
                # Store the last attempted update ID?

        with ws_update_time_lock:
            last_ws_update_time[symbol] = time.time()

    except Exception as e:
        logger.error(f"Error processing depth message: {e}\nMessage: {msg}", exc_info=False)


def process_ticker_message(msg):
    """Handles incoming ticker messages."""
    try:
        if msg.get('e') == 'error':
            logger.error(f"Ticker WS Error: {msg.get('m')}")
            return
        if msg.get('e') != '24hrTicker':
            logger.debug(f"Ignoring non-ticker message: {msg.get('e')}")
            return

        symbol = msg['s']
        # Store relevant parts of the ticker info
        ticker_info = {
            'last_price': float(msg['c']),
            'price_change_percent': float(msg['P']),
            'high': float(msg['h']),
            'low': float(msg['l']),
            'volume': float(msg['v']),
            'quote_volume': float(msg['q']),
            'timestamp': time.time() # Add timestamp of reception
        }

        with ticker_lock:
            latest_ticker_data[symbol] = ticker_info

        with ws_update_time_lock:
             # May not need separate update time if ticker_info includes it
             # But good for a central "last seen" check
            last_ws_update_time[symbol] = time.time()

        # logger.debug(f"Processed ticker for {symbol}: Price={ticker_info['last_price']}")

    except Exception as e:
        logger.error(f"Error processing ticker message: {e}\nMessage: {msg}", exc_info=False)

# --- End WebSocket Callbacks ---


# --- Existing Functions (Modified where needed) ---

# File Writer Worker (Unchanged, uses queue)
def file_writer_worker():
    """Worker thread that processes file update requests from a queue."""
    logger.info("File writer worker thread started.")
    while not file_writer_stop_event.is_set():
        try:
            try:
                task = file_update_queue.get(timeout=1)
            except queue.Empty:
                continue

            file_type = task.get('type')
            data = task.get('data')
            symbol = task.get('symbol')

            if not file_type or data is None:
                logger.warning(f"Invalid task received in file writer queue: {task}")
                file_update_queue.task_done()
                continue

            if file_type == 'trade_log':
                try:
                    date_str = time.strftime("%Y%m%d", time.localtime())
                    filename = f"{symbol.replace('/', '_').replace(':', '_')}_{date_str}_trades.csv"
                    filepath = os.path.join(TRADE_LOG_DIR, filename)
                    file_exists = os.path.isfile(filepath)
                    df_to_log = pd.DataFrame([data])
                    df_to_log.to_csv(filepath, mode='a', header=not file_exists, index=False)
                except Exception as e:
                    logger.error(f"File writer error processing trade_log for {symbol}: {e}", exc_info=False)

            elif file_type == 'params':
                try:
                    filename = symbol.replace('/', '_').replace(':', '_') + '.json'
                    filepath = os.path.join(PARAMS_DIR, filename)
                    temp_filepath = filepath + '.tmp'
                    with open(temp_filepath, 'w') as f:
                        json.dump(data, f, indent=4)
                    if os.path.exists(temp_filepath): os.replace(temp_filepath, filepath)
                except Exception as e:
                    logger.error(f"File writer error processing params for {symbol}: {e}", exc_info=False)

            elif file_type == 'pattern_stats':
                 try:
                    if not symbol:
                        logger.error("Missing symbol for pattern_stats save request.")
                        file_update_queue.task_done()
                        continue
                    filename = f"{symbol.replace('/', '_').replace(':', '_')}_ob_oi_pattern_stats.json"
                    filepath = os.path.join(PATTERN_STATS_DIR, filename)
                    temp_filepath = filepath + '.tmp'
                    with open(temp_filepath, 'w') as f:
                         json.dump(data, f, indent=4)
                    if os.path.exists(temp_filepath): os.replace(temp_filepath, filepath)
                 except Exception as e:
                    logger.error(f"File writer error processing pattern_stats for {symbol}: {e}", exc_info=True)
            else:
                logger.warning(f"Unknown file type '{file_type}' received in file writer queue.")

            file_update_queue.task_done()

        except Exception as e:
            logger.error(f"Unexpected error in file_writer_worker loop: {e}", exc_info=True)
            # Avoid busy-waiting on error
            if not file_writer_stop_event.is_set():
                 time.sleep(1)


    logger.info("File writer worker thread stopped.")


# Metrics, Logging, Params (Largely Unchanged)
def log_trade_metrics(reason, increment=True):
    """Increment and log metrics on trade filtering."""
    global global_metrics
    if increment and reason in global_metrics: global_metrics[reason] += 1
    current_time = time.time()
    if current_time - global_metrics.get('last_reset', 0) > 86400:
        report_metrics()
        for key in global_metrics:
            if key != 'last_reset': global_metrics[key] = 0
        global_metrics['last_reset'] = current_time
        logger.info("Trade metrics reset")

def report_metrics():
    """Generate a report of current trade filtering metrics."""
    global global_metrics
    patterns_detected = global_metrics.get('patterns_detected', 0)
    if patterns_detected > 0:
        success_rate = global_metrics.get('successful_entries', 0) / patterns_detected * 100
        report = f"\n{'='*50}\nTRADE FILTERING METRICS REPORT\n{'='*50}\n\n"
        report += f"Total Patterns Detected: {patterns_detected}\n"
        report += f"Successful Trade Entries: {global_metrics.get('successful_entries', 0)} ({success_rate:.2f}%)\n\n"
        report += "REJECTION REASONS:\n"
        # Simplified report generation
        for i, (key, value) in enumerate(global_metrics.items()):
             if key not in ['patterns_detected', 'successful_entries', 'last_reset']:
                 pct = value / patterns_detected * 100 if patterns_detected else 0
                 report += f"{i+1}. {key.replace('_', ' ').title()}: {value} ({pct:.2f}%)\n"
        report += f"\n{'='*50}\n"
        logger.info(report)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        report_path = os.path.join(TRADE_LOG_DIR, f"trade_metrics_{timestamp}.txt")
        try:
            with open(report_path, 'w') as f: f.write(report)
            logger.info(f"Trade metrics report saved to {report_path}")
        except Exception as e:
             logger.error(f"Failed to save metrics report: {e}")
    else:
        logger.info("No patterns detected since last reset, no metrics to report")

def log_trade_funnel():
     """Periodically log the trade funnel to track conversion rates."""
     patterns_detected = global_metrics.get('patterns_detected', 0)
     if patterns_detected == 0:
         logger.info("No patterns detected yet, cannot calculate trade funnel")
         return

     # Simplified calculation
     steps = {
         'Detected': patterns_detected,
         'Passed Threshold': patterns_detected - global_metrics.get('patterns_below_threshold', 0),
         'Passed TF Confirm': (patterns_detected - global_metrics.get('patterns_below_threshold', 0)) - global_metrics.get('insufficient_tf_confirmation', 0),
         'Passed Sequence': ((patterns_detected - global_metrics.get('patterns_below_threshold', 0)) - global_metrics.get('insufficient_tf_confirmation', 0)) - global_metrics.get('failed_sequence_check', 0),
         # ... add other steps similarly ...
         'Passed Proximity': (((patterns_detected - global_metrics.get('patterns_below_threshold', 0)) - global_metrics.get('insufficient_tf_confirmation', 0)) - global_metrics.get('failed_sequence_check', 0) - global_metrics.get('failed_reversal_check', 0) - global_metrics.get('failed_risk_reward', 0)), # Example calculation up to RR
         'Successful Orders': global_metrics.get('successful_entries', 0)
     }
     # Calculate step after proximity check correctly
     passed_rr = steps['Passed Proximity'] # This is actually passed RR check based on above calc
     steps['Passed Proximity Check'] = passed_rr - global_metrics.get('failed_proximity_check', 0) # Now calculate based on proximity failures
     steps['Successful Orders'] = global_metrics.get('successful_entries', 0)


     logger.info("\n" + "-" * 40 + "\nTRADE FUNNEL ANALYSIS\n" + "-" * 40)
     last_step_count = patterns_detected
     for name, count in steps.items():
         if count < 0: count = 0 # Ensure count doesn't go negative
         rate = count / last_step_count * 100 if last_step_count > 0 else 0
         logger.info(f"{name}: {count} ({rate:.1f}%)")
         # Update last_step_count for the *next* percentage calculation relative to the previous step
         # Only update if the current step represents a filtering stage (not the final success)
         if name != 'Successful Orders':
             last_step_count = count # Use the count of the current step as the base for the next step's %

     overall_conversion = steps['Successful Orders'] / patterns_detected * 100 if patterns_detected > 0 else 0
     logger.info("-" * 40 + f"\nOVERALL CONVERSION RATE: {overall_conversion:.2f}%\n" + "-" * 40)


def log_trade(symbol, trade_info, force_write=False):
    """ Queue trade details for logging. (Unchanged logic, uses queue) """
    try:
        trade_info['symbol'] = symbol
        if 'timestamp' not in trade_info: trade_info['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if 'position_status' not in trade_info: trade_info['position_status'] = 'Closed' if trade_info.get('exit_reason') else 'Open'

        essential_cols = ['symbol', 'timestamp', 'position_status', 'position_type', 'entry_price', 'exit_price', 'stop_loss', 'target', 'entry_reason', 'exit_reason', 'profit_pct', 'leverage', 'signal_strength', 'risk_reward']
        log_data = {col: trade_info.get(col) for col in essential_cols}

        if log_data.get('exit_price') is not None and log_data.get('entry_price') is not None and log_data['entry_price'] != 0:
             pnl_mult = 1 if log_data.get('position_type') == 'long' else -1
             log_data['profit_pct'] = ((log_data['exit_price'] - log_data['entry_price']) / log_data['entry_price']) * pnl_mult * 100

        task = {'type': 'trade_log', 'symbol': symbol, 'data': log_data}
        file_update_queue.put(task)

        status_msg = f"Status: {log_data.get('position_status', 'N/A')}"
        pnl_msg = f"PnL: {log_data.get('profit_pct', 0):.2f}%" if log_data.get('profit_pct') is not None else ""
        reason_msg = f"Entry: {log_data.get('entry_reason', 'N/A')}" + (f" -> Exit: {log_data.get('exit_reason')}" if log_data.get('exit_reason') else "")
        logger.info(f"TRADE QUEUED: {symbol} | {reason_msg} | {status_msg} | {pnl_msg}")

        # Update pattern statistics (Unchanged logic)
        if log_data.get('position_status') == 'Closed' and log_data.get('signal_type'):
            try:
                signal_type = log_data['signal_type']
                profit_pct = log_data.get('profit_pct', 0)
                win = profit_pct > 0 if profit_pct is not None else False
                with indicator_lock:
                    if symbol in symbol_indicators:
                        symbol_indicators[symbol].update_pattern_trade_result(signal_type, profit_pct, win=win)
            except Exception as e:
                logger.warning(f"Could not update pattern statistics after queuing trade log: {e}")
        return True
    except Exception as e:
        logger.exception(f"Error queuing trade log for {symbol}: {e}")
        return False

def save_params(symbol, params):
    """ Queue parameters for saving. (Unchanged logic, uses queue) """
    try:
        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        serializable_params = {}
        # Basic serialization handling (adjust if complex objects are in params)
        for k, v in params.items():
            if k in ['file_queue', 'symbol']: continue # Don't save these
            try:
                 # Attempt to JSON serialize; if fails, convert or skip
                 json.dumps({k: v}) # Test serialization
                 serializable_params[k] = v
            except TypeError:
                 if isinstance(v, (np.integer, np.floating)): serializable_params[k] = v.item()
                 elif isinstance(v, np.ndarray): serializable_params[k] = v.tolist()
                 else: logger.debug(f"Skipping non-serializable param {k} type {type(v)} for {symbol}")

        task = {'type': 'params', 'symbol': clean_symbol, 'data': serializable_params}
        file_update_queue.put(task)
        logger.info(f"Queued saving parameters for {symbol}")
        filepath = os.path.join(PARAMS_DIR, clean_symbol + '.json')
        return filepath
    except Exception as e:
         logger.exception(f"Error queueing params for {symbol}: {e}")
         return None

def load_params(symbol):
    """Load parameters for a symbol. (Unchanged)"""
    filename = symbol.replace('/', '_').replace(':', '_') + '.json'
    filepath = os.path.join(PARAMS_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except Exception as e: logger.exception(f"Error loading parameters for {symbol}: {e}")
    return None


# --- Data Fetching / Retrieval ---

def fetch_initial_klines(exchange, market_id, timeframe=TIMEFRAME, limit=LIMIT):
    """ Fetch initial historical klines using ccxt """
    try:
        # Fetch slightly more to ensure limit after potential processing
        ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=limit + 5)
        if not ohlcv:
            logger.warning(f"fetch_initial_klines received no data for {market_id}")
            return []

        # Convert to list of dicts matching WS structure (for consistency)
        kline_list = []
        for candle in ohlcv:
            ts, O, H, L, C, V = candle
            # Assume fetched candles are closed
            kline_list.append({
                 'timestamp': pd.to_datetime(ts, unit='ms', utc=True),
                 'open': float(O), 'high': float(H), 'low': float(L), 'close': float(C), 'volume': float(V),
                 'is_closed': True
            })

        # Return only the last 'limit' candles
        return kline_list[-limit:]
    except ccxt.RateLimitExceeded as e:
         logger.warning(f"Rate limit hit fetching initial klines for {market_id}: {e}. Retrying might be needed.")
         time.sleep(10) # Basic wait
         return None # Signal failure or retry needed
    except Exception as e:
        logger.exception(f"Failed to fetch initial klines for {market_id}: {e}")
        return None # Indicate error

def get_kline_df(symbol):
    """ Gets the current kline data as a DataFrame from the deque """
    with kline_lock:
        if symbol not in latest_kline_data or not latest_kline_data[symbol]:
             return pd.DataFrame() # Return empty DF if no data

        # Convert deque of dicts to DataFrame
        df = pd.DataFrame(list(latest_kline_data[symbol]))
        if df.empty:
            return df

        # Set timestamp as index
        if 'timestamp' in df.columns:
             df.set_index('timestamp', inplace=True)
        else:
             logger.error(f"Timestamp column missing in kline data for {symbol}")
             return pd.DataFrame()

        # Ensure correct data types (might be redundant if process_kline_message handles it)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove the current incomplete candle for indicator calculations if needed
        # Assuming indicator wants only closed candles. Check indicator requirements.
        # Or the indicator needs to handle the last candle appropriately.
        # Let's return the full DF including the potentially incomplete last candle for now.
        # The indicator logic might need adjustment if it assumes all candles are closed.
        # df_closed = df[df['is_closed'] == True] # Option 1: Only closed
        # return df_closed.drop(columns=['is_closed'])

        return df # Option 2: Return all, indicator handles it


def fetch_initial_order_book(exchange, symbol, limit=1000):
    """ Fetches the initial order book snapshot using ccxt """
    try:
        # Fetch a deep snapshot (limit 1000 is max for Binance REST)
        # This is needed to correctly initialize the state before applying WS diffs
        ob = exchange.fetch_order_book(symbol, limit=limit)
        last_update_id = ob.get('nonce', 0) # Binance uses 'nonce' for lastUpdateId in REST response

        if not last_update_id:
             logger.warning(f"Order book snapshot for {symbol} missing 'nonce' (lastUpdateId). Depth sync may fail.")

        # Store in the format used by our WS cache
        with depth_lock:
            latest_depth_data[symbol] = {
                'bids': dict(sorted([(float(p), float(q)) for p, q in ob['bids']], reverse=True)),
                'asks': dict(sorted([(float(p), float(q)) for p, q in ob['asks']])),
                'lastUpdateId': last_update_id
            }
        logger.info(f"Fetched initial order book snapshot for {symbol} up to updateId {last_update_id}")
        return True
    except ccxt.RateLimitExceeded as e:
         logger.warning(f"Rate limit hit fetching initial order book for {symbol}: {e}.")
         time.sleep(10)
         return False
    except Exception as e:
        logger.exception(f"Failed to fetch initial order book for {symbol}: {e}")
        return False

def get_order_book(symbol):
     """ Gets the current order book from the cache """
     with depth_lock:
         # Return a copy to prevent modification issues outside the lock
         # Convert back to list format expected by ccxt/indicator?
         # The indicator might need adjustment to accept the dict format,
         # or we convert it here. Let's convert it.
         ob_data = latest_depth_data.get(symbol)
         if ob_data:
             # Convert dict back to list of lists [price, quantity]
             # Ensure sorting is maintained
             bids_list = [[p, q] for p, q in sorted(ob_data['bids'].items(), key=lambda item: item[0], reverse=True)]
             asks_list = [[p, q] for p, q in sorted(ob_data['asks'].items(), key=lambda item: item[0])]
             # Limit depth if needed, though WS already limits incoming stream
             bids_list = bids_list[:WEBSOCKET_DEPTH_LEVELS]
             asks_list = asks_list[:WEBSOCKET_DEPTH_LEVELS]

             # Return in ccxt-like format
             return {
                 'bids': bids_list,
                 'asks': asks_list,
                 'timestamp': None, # WS doesn't provide per-update timestamp easily
                 'datetime': None,
                 'nonce': ob_data['lastUpdateId'] # lastUpdateId from WS
             }
         else:
             return None # No data available

def get_current_price(symbol):
     """ Gets the latest price from the ticker stream """
     with ticker_lock:
         ticker = latest_ticker_data.get(symbol)
         if ticker:
             return ticker.get('last_price')
     return None # No price available

def fetch_open_interest(exchange, symbol):
    """ Fetch open interest data (Remains polling via REST). """
    # This function remains largely the same as it relies on polling
    # We might add caching here if needed, but the original didn't have it explicitly
    # Use the existing cache mechanism for this?
    cache_key = f"oi_{symbol}"
    current_time = time.time()

    with cache_lock: # Reuse the ohlcv cache lock or create a dedicated OI cache lock
        cached = ohlcv_cache.get(cache_key)
        if cached and current_time - cached.get('timestamp', 0) < CACHE_TTL:
            # logger.debug(f"Using cached OI for {symbol}")
            return cached.get('data')

    # Fetch new data
    try:
        # Assuming exchange object 'exchange' is the ccxt instance
        if hasattr(exchange, 'fetch_open_interest'):
            # Use fetch_open_interest if available (standard ccxt way)
            oi_data = exchange.fetch_open_interest(symbol)
            oi_value = oi_data.get('openInterestAmount') # Or 'openInterestValue' depending on needs
            if oi_value is None and 'info' in oi_data:
                # Fallback to info dict if standard field is missing (exchange specific)
                oi_value = float(oi_data['info'].get('openInterest', 0)) # Common field name
            if oi_value is None:
                 logger.warning(f"Could not extract Open Interest value for {symbol} from: {oi_data}")
                 return None

            logger.debug(f"Fetched Open Interest for {symbol}: {oi_value}")

            # Update cache
            with cache_lock:
                ohlcv_cache[cache_key] = {'data': float(oi_value), 'timestamp': current_time}
            return float(oi_value)

        else:
             logger.warning(f"Exchange object does not support fetch_open_interest for {symbol}.")
             # Fallback simulation (REMOVE FOR PRODUCTION)
             # current_price = get_current_price(symbol) or 1 # Get price from WS if possible
             # simulated_oi = current_price * 10000 * (1 + np.random.normal(0, 0.05))
             # logger.debug(f"Using simulated open interest for {symbol}: {simulated_oi}")
             # return simulated_oi
             return None # No real data

    except ccxt.NotSupported:
         logger.warning(f"fetch_open_interest not supported by exchange for {symbol}")
         return None
    except ccxt.RateLimitExceeded as e:
         logger.warning(f"Rate limit hit fetching OI for {symbol}: {e}")
         # Return stale cache data if available?
         with cache_lock:
             cached = ohlcv_cache.get(cache_key)
             if cached: return cached.get('data')
         return None
    except Exception as e:
        logger.error(f"Error fetching open interest for {symbol}: {e}", exc_info=False)
        return None

# --- Indicator Management (Modified) ---

def get_indicator(symbol, exchange, force_new=False):
    """ Get or create indicator. Now relies less on internal fetching. """
    global file_update_queue
    indicator_instance = None

    with indicator_lock:
        if symbol in symbol_indicators and not force_new:
            indicator_instance = symbol_indicators[symbol]
        else:
            params = load_params(symbol) if not force_new else None
            if params:
                params_with_queue = params.copy()
                params_with_queue['file_queue'] = file_update_queue
                params_with_queue['symbol'] = symbol
                try:
                    # Ensure indicator is initialized correctly without relying on fetch methods
                    # It should primarily accept dataframes/series/values as input now
                    symbol_indicators[symbol] = EnhancedOrderBookVolumeWrapper(**params_with_queue)
                    indicator_instance = symbol_indicators[symbol]
                    # logger.info(f"Loaded existing parameters for {symbol}")
                except Exception as e:
                     logger.error(f"Error creating indicator with loaded params for {symbol}: {e}")
                     force_new = True
            
            if force_new or indicator_instance is None:
                default_params_copy = DEFAULT_PARAMS.copy()
                default_params_copy['file_queue'] = file_update_queue
                default_params_copy['symbol'] = symbol
                if 'BTC' in symbol: default_params_copy['min_wall_size'] = 50
                elif 'ETH' in symbol: default_params_copy['min_wall_size'] = 30
                else: default_params_copy['min_wall_size'] = 20
                
                try:
                    symbol_indicators[symbol] = EnhancedOrderBookVolumeWrapper(**default_params_copy)
                    indicator_instance = symbol_indicators[symbol]
                    logger.info(f"Created new indicator with default parameters for {symbol}")
                    save_params(symbol, default_params_copy) # Queue saving
                except Exception as e:
                    logger.critical(f"CRITICAL: Failed to create default indicator for {symbol}: {e}")
                    return None

    return indicator_instance

# --- Exchange and Position Management (Largely Unchanged logic, but uses ccxt) ---

def create_exchange():
    """Create and return a CCXT exchange object."""
    # Ensure API keys are loaded correctly from environment
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    if not api_key or not secret_key:
        logger.critical("Binance API Key or Secret Key not found in environment variables.")
        raise ValueError("Missing API Credentials")

    return ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 30000
    })

# get_position_details, is_in_cooldown, can_exit_position remain the same

def get_position_details(symbol, entry_price, position_type, position_entry_times):
    """Retrieve position details. (Unchanged)"""
    try:
        with position_details_lock:
            if symbol in position_details: return position_details[symbol].copy()
        # Default values (unchanged)
        default_details = { 'entry_price': entry_price, 'stop_loss': entry_price * (0.997 if position_type == 'long' else 1.003), 'target': entry_price * (1.006 if position_type == 'long' else 0.994), 'position_type': position_type, 'entry_reason': 'Unknown', 'probability': 0.5, 'entry_time': position_entry_times.get(symbol, time.time()), 'highest_reached': entry_price if position_type == 'long' else None, 'lowest_reached': entry_price if position_type == 'short' else None, }
        with position_details_lock:
            if symbol not in position_details: position_details[symbol] = default_details
            return position_details[symbol].copy()
    except Exception as e:
        logger.warning(f"Error retrieving position details for {symbol}: {e}")
        # Fallback default (unchanged)
        return { 'entry_price': entry_price, 'stop_loss': entry_price * (0.997 if position_type == 'long' else 1.003), 'target': entry_price * (1.006 if position_type == 'long' else 0.994), 'position_type': position_type, 'entry_time': time.time() - 3600, 'highest_reached': entry_price if position_type == 'long' else None, 'lowest_reached': entry_price if position_type == 'short' else None, }


def is_in_cooldown(symbol):
    """Check if a symbol is in cooldown period. (Unchanged)"""
    with cooldown_lock:
        cooldown_until = symbol_cooldowns.get(symbol)
        if cooldown_until and time.time() < cooldown_until:
            # remaining = int((cooldown_until - time.time()) / 60)
            # logger.debug(f"{symbol} in cooldown for {remaining} more minutes.")
            return True
        elif cooldown_until:
             del symbol_cooldowns[symbol] # Cooldown expired
    return False

def can_exit_position(symbol):
    """Check if a position has been held long enough. (Unchanged)"""
    with cooldown_lock:
        entry_time = position_entry_times.get(symbol)
        if entry_time:
            hold_time_minutes = (time.time() - entry_time) / 60
            if hold_time_minutes < MINIMUM_HOLD_MINUTES:
                # remaining = int(MINIMUM_HOLD_MINUTES - hold_time_minutes)
                # logger.info(f"{symbol} position held for {hold_time_minutes:.1f} mins, minimum hold is {MINIMUM_HOLD_MINUTES} mins.")
                return False
    return True


def fetch_active_symbols(exchange, num_symbols=50):
    """Fetch top N active trading symbols using ccxt."""
    try:
        # Use ccxt's fetch_tickers
        tickers = exchange.fetch_tickers()
        if not tickers:
            logger.error("Failed to fetch tickers.")
            return []

        # Load markets to filter by USDT perpetual swaps
        markets = exchange.load_markets()
        
        # Filter symbols: USDT settled, perpetual swap, has ticker info, doesn't contain BTC/ETH
        # Note: ccxt symbol format might be 'BTC/USDT:USDT' for perpetuals
        filtered_symbols = []
        for symbol, ticker_data in tickers.items():
             market_info = markets.get(symbol)
             # Check if market_info exists and meets criteria
             if (market_info and
                 market_info.get('settle') == 'USDT' and
                 market_info.get('swap') and # Checks if it's a swap
                 market_info.get('linear') and # Often used for USDT margined
                 'USDT' in symbol and # Ensure USDT pair
                 'quoteVolume' in ticker_data and ticker_data['quoteVolume'] is not None): # Has volume data
                 # Check base currency isn't BTC or ETH
                 base_currency = market_info.get('base')
                 if base_currency and base_currency not in ['BTC', 'ETH']:
                      # Use the symbol format expected by the rest of the bot (e.g., 'ADA/USDT:USDT')
                      # The indicator and order placement need the correct ccxt symbol.
                      # However, websockets might use a different format (e.g., 'ADAUSDT'). We need consistency.
                      # Let's assume ccxt format is primary for now.
                      filtered_symbols.append(symbol)


        # Sort by quote volume
        top_symbols = sorted(
            filtered_symbols,
            key=lambda s: tickers[s].get('quoteVolume', 0),
            reverse=True
        )

        logger.info(f"Fetched {len(filtered_symbols)} active USDT perpetuals (excluding BTC/ETH). Top {num_symbols} by volume selected.")
        return top_symbols[:num_symbols]

    except ccxt.NetworkError as e:
         logger.error(f"Network error fetching active symbols: {e}")
         return []
    except Exception as e:
        logger.exception(f"Error fetching active symbols: {e}")
        return []

def get_leverage_for_market(exchange, market_id, leverage_options=LEVERAGE_OPTIONS):
    """Set and get leverage using ccxt."""
    # ccxt uses set_leverage(leverage, symbol)
    try:
        # First, try to fetch current leverage if API supports it (often not standard)
        # If not, iterate and set
        for leverage in leverage_options:
            try:
                # logger.debug(f"Attempting to set leverage {leverage} for {market_id}")
                exchange.set_leverage(leverage, market_id)
                # Verify? Some exchanges return confirmation, others don't reliably
                # We assume success if no exception is thrown
                logger.info(f"Set leverage to {leverage} for {market_id}")
                return leverage
            except ccxt.ExchangeError as e:
                 # Handle specific errors like "leverage not modified" which might be okay
                 if "leverage not modified" in str(e).lower():
                      logger.info(f"Leverage already set to {leverage} for {market_id} (or cannot be modified).")
                      # How to confirm the *actual* current leverage? fetch_position might show it.
                      # For now, assume this means the leverage is acceptable. Return it.
                      # Re-fetch position after setting might be needed for verification.
                      return leverage
                 logger.warning(f"Could not set leverage {leverage} for {market_id}: {e}")
                 continue # Try next lower leverage
            except Exception as e:
                 logger.warning(f"Non-ExchangeError setting leverage {leverage} for {market_id}: {e}")
                 continue

        logger.error(f"Failed to set any leverage from {leverage_options} for {market_id}")
        return None # Failed to set any leverage
    except Exception as e:
        logger.exception(f"General error setting leverage for {market_id}: {e}")
        return None

def fetch_open_positions(exchange):
    """Fetch open positions using ccxt."""
    try:
        # Ensure market data is loaded for mapping symbols if needed
        if not exchange.markets:
            exchange.load_markets()

        # Fetch positions (may need symbol=None or specific symbols depending on needs)
        # Passing no symbol usually fetches all positions for the account type
        positions = exchange.fetch_positions(symbols=None) # Fetch all relevant positions

        # Filter for positions with non-zero size
        open_positions = {}
        for pos in positions:
            # ccxt standard format includes 'contracts', 'contractSize', 'side', 'unrealizedPnl' etc.
            # Check for non-zero position size. 'contracts' is often the field.
            position_size = pos.get('contracts', 0.0)
            if position_size is None: position_size = 0.0 # Handle None case

            # Also check 'info' dict for exchange-specific amount if 'contracts' is zero/None
            if abs(position_size) < 1e-9 and 'info' in pos: # Precision check
                position_size = float(pos['info'].get('positionAmt', 0.0)) # Common Binance field

            if abs(position_size) > 1e-9: # Use a small threshold for floating point
                 symbol = pos.get('symbol')
                 if symbol:
                     # Store in a format consistent with original structure if needed
                     # Or just store the whole ccxt position object
                     open_positions[symbol] = {
                         'symbol': symbol,
                         'size': position_size, # Store the determined size
                         'side': 'long' if position_size > 0 else 'short',
                         'entry_price': pos.get('entryPrice'),
                         'leverage': pos.get('leverage'),
                         'ccxt_object': pos # Store the full object for later use
                         # Keep original nested 'info' structure if place_order/exit relies on it heavily
                         # 'info': {'positionAmt': str(position_size), 'entryPrice': str(pos.get('entryPrice'))}
                     }
                 else:
                      logger.warning(f"Position found with non-zero size but missing symbol: {pos}")

        # logger.debug(f"Fetched {len(open_positions)} open positions.")
        return open_positions

    except ccxt.NetworkError as e:
         logger.error(f"Network error fetching positions: {e}")
         return {}
    except ccxt.ExchangeError as e:
         logger.error(f"Exchange error fetching positions: {e}")
         return {}
    except Exception as e:
        logger.exception(f"Unexpected error fetching positions: {e}")
        return {}

# Order placement and quantity calculation (Largely unchanged, uses ccxt)
def place_order(exchange, market_id, side, quantity, price, leverage=None, exit_order=False,
               stop_loss=None, target_price=None, place_target_order=True, skip_cooldown=False):
    """Place an order using ccxt with SL/TP handling."""
    # Use ccxt's create_order and parameter conventions
    order_type = 'MARKET' # Use market orders for entry/exit
    params = {}
    order_placed_successfully = False
    market_order_info = None # To store result from create_order

    if exit_order:
        params['reduceOnly'] = True

    try:
        # Pre-flight checks (Unchanged logic)
        if not exit_order and place_target_order:
            if stop_loss is None or target_price is None:
                 logger.error(f"Cannot place entry for {market_id}: Missing SL/TP.")
                 return None
            # Basic validation (prices make sense relative to current 'price')
            min_dist = price * 0.0005 # 0.05% minimum distance
            if side == 'buy': # Long
                if stop_loss >= price - min_dist: logger.error(f"Invalid SL {stop_loss} for {market_id} LONG @ {price}. Too close or above entry."); return None
                if target_price <= price + min_dist: logger.error(f"Invalid TP {target_price} for {market_id} LONG @ {price}. Too close or below entry."); return None
            else: # Short
                if stop_loss <= price + min_dist: logger.error(f"Invalid SL {stop_loss} for {market_id} SHORT @ {price}. Too close or below entry."); return None
                if target_price >= price - min_dist: logger.error(f"Invalid TP {target_price} for {market_id} SHORT @ {price}. Too close or above entry."); return None


        # Use price_to_precision and amount_to_precision
        market = exchange.markets[market_id]
        precise_quantity = exchange.amount_to_precision(market_id, quantity)
        
        # Check against minimum amount and cost
        min_amount = market['limits']['amount']['min']
        min_cost = market['limits']['cost']['min']
        
        if float(precise_quantity) < min_amount:
             logger.warning(f"Quantity {precise_quantity} for {market_id} below minimum {min_amount}. Adjusting...")
             precise_quantity = exchange.amount_to_precision(market_id, min_amount) # Adjust to min amount
             # Recalculate cost with adjusted quantity
             current_cost = float(precise_quantity) * price
        else:
             current_cost = float(precise_quantity) * price

        if min_cost is not None and current_cost < min_cost:
             logger.error(f"Order cost {current_cost:.2f} for {market_id} below minimum {min_cost:.2f}. Cannot place order.")
             # Attempt to increase quantity to meet min_cost? Risky, changes position size. Abort for safety.
             return None


        logger.info(f"Attempting {side.upper()} {'exit' if exit_order else 'entry'} order: {market_id}, Type: {order_type}, Qty: {precise_quantity}")

        # Place the main market order
        market_order = exchange.create_order(market_id, order_type, side, precise_quantity, price=None, params=params)
        # Price = None for market orders

        market_order_info = market_order # Store the response
        order_id = market_order.get('id', 'N/A')
        avg_price = market_order.get('average', price) # Use average fill price if available

        logger.info(f"Market order {order_id} placed for {market_id}: {side} {precise_quantity}. Avg Price: {avg_price}")
        order_placed_successfully = True

        # Post-entry/exit logic (Unchanged logic flow)
        if not exit_order:
            with cooldown_lock: position_entry_times[market_id] = time.time()
            logger.info(f"Position entry time recorded for {market_id}")
            if place_target_order:
                time.sleep(1.5) # Allow market order to fill
                # Pass the potentially filled average price to SL/TP placement if needed?
                # But SL/TP triggers are based on levels, not entry price directly.
                place_sl_tp_orders(exchange, market_id, side, precise_quantity, stop_loss, target_price)
        else: # Exit order placed
            if not skip_cooldown:
                with cooldown_lock:
                    cooldown_end_time = time.time() + SYMBOL_COOLDOWN_MINUTES * 60
                    symbol_cooldowns[market_id] = cooldown_end_time
                    if market_id in position_entry_times: del position_entry_times[market_id]
                logger.info(f"Setting cooldown for {market_id} until: {time.strftime('%H:%M:%S', time.localtime(cooldown_end_time))}")
            else:
                logger.info(f"Skipping cooldown for {market_id}")

        return market_order_info # Return the order details dict

    except ccxt.InsufficientFunds as e:
         log_trade_metrics('order_placement_errors')
         logger.error(f"Insufficient funds to place {side} order for {market_id}: {e}")
         # Clean up local state if entry failed
         if not exit_order and not order_placed_successfully:
             with position_details_lock:
                 if market_id in position_details: del position_details[market_id]
             with cooldown_lock:
                  if market_id in position_entry_times: del position_entry_times[market_id]
         return None
    except ccxt.ExchangeError as e:
        log_trade_metrics('order_placement_errors')
        logger.error(f"Exchange error placing order for {market_id}: {e}")
        if not exit_order and not order_placed_successfully:
             with position_details_lock:
                 if market_id in position_details: del position_details[market_id]
             with cooldown_lock:
                  if market_id in position_entry_times: del position_entry_times[market_id]
        return None
    except Exception as e:
        log_trade_metrics('order_placement_errors')
        logger.exception(f"Unexpected error placing order for {market_id}: {e}")
        if not exit_order and not order_placed_successfully:
             with position_details_lock:
                 if market_id in position_details: del position_details[market_id]
             with cooldown_lock:
                  if market_id in position_entry_times: del position_entry_times[market_id]
        return None

def place_sl_tp_orders(exchange, market_id, entry_side, quantity, stop_loss, target_price, retries=2):
    """Place SL/TP orders using ccxt unified stop methods."""
    if stop_loss is None or target_price is None:
        logger.error(f"Missing SL ({stop_loss}) or TP ({target_price}) for {market_id}.")
        return False

    inverted_side = 'sell' if entry_side == 'buy' else 'buy'
    precise_quantity = exchange.amount_to_precision(market_id, quantity)
    precise_sl = exchange.price_to_precision(market_id, stop_loss)
    precise_tp = exchange.price_to_precision(market_id, target_price)

    sl_order_id = None
    tp_order_id = None
    success = False

    # Ensure SL/TP prices are valid relative to each other
    if entry_side == 'buy' and float(precise_sl) >= float(precise_tp):
         logger.error(f"Invalid SL/TP for {market_id} LONG: SL {precise_sl} >= TP {precise_tp}. Aborting SL/TP placement.")
         return False
    if entry_side == 'sell' and float(precise_sl) <= float(precise_tp):
         logger.error(f"Invalid SL/TP for {market_id} SHORT: SL {precise_sl} <= TP {precise_tp}. Aborting SL/TP placement.")
         return False


    # Use create_order with 'STOP_MARKET' and 'TAKE_PROFIT_MARKET' types
    # Specify stopPrice in params
    try:
        # Place Stop Loss (STOP_MARKET)
        sl_params = {
            'stopPrice': precise_sl,
            'reduceOnly': True,
            # 'timeInForce': 'GTE_GTC', # May not be needed/supported for STOP_MARKET with stopPrice
        }
        logger.info(f"Placing SL Order: {market_id}, Type: STOP_MARKET, Side: {inverted_side}, Qty: {precise_quantity}, StopPx: {precise_sl}")
        sl_order = exchange.create_order(market_id, 'STOP_MARKET', inverted_side, precise_quantity, price=None, params=sl_params)
        sl_order_id = sl_order.get('id')
        logger.info(f"SL order {sl_order_id} placed for {market_id} at stop {precise_sl}")

        time.sleep(0.5) # Brief pause

        # Place Take Profit (TAKE_PROFIT_MARKET)
        tp_params = {
            'stopPrice': precise_tp, # stopPrice acts as the trigger for take profit market
            'reduceOnly': True,
            # 'timeInForce': 'GTE_GTC',
        }
        logger.info(f"Placing TP Order: {market_id}, Type: TAKE_PROFIT_MARKET, Side: {inverted_side}, Qty: {precise_quantity}, StopPx: {precise_tp}")
        tp_order = exchange.create_order(market_id, 'TAKE_PROFIT_MARKET', inverted_side, precise_quantity, price=None, params=tp_params)
        tp_order_id = tp_order.get('id')
        logger.info(f"TP order {tp_order_id} placed for {market_id} at trigger {precise_tp}")

        success = True

    except ccxt.ExchangeError as e:
         # Attempt to cancel the one that might have succeeded if the other failed
         logger.error(f"Exchange error placing SL/TP for {market_id}: {e}")
         if sl_order_id and not tp_order_id: # SL placed, TP failed
             logger.warning(f"TP order failed for {market_id}, attempting to cancel SL order {sl_order_id}")
             try: exchange.cancel_order(sl_order_id, market_id)
             except Exception as cancel_e: logger.error(f"Failed to cancel SL order {sl_order_id}: {cancel_e}")
         elif not sl_order_id and tp_order_id: # SL failed, TP placed (less likely sequence)
              logger.warning(f"SL order failed for {market_id}, attempting to cancel TP order {tp_order_id}")
              try: exchange.cancel_order(tp_order_id, market_id)
              except Exception as cancel_e: logger.error(f"Failed to cancel TP order {tp_order_id}: {cancel_e}")
         # Log metric if placing failed
         if not success: log_trade_metrics('order_placement_errors')
         return False # Indicate failure

    except Exception as e:
        logger.exception(f"Unexpected error placing SL/TP orders for {market_id}: {e}")
        # Add cancellation logic here too if needed
        if not success: log_trade_metrics('order_placement_errors')
        return False

    # Store order IDs if successful
    if success:
         with position_details_lock:
             if market_id in position_details:
                 position_details[market_id]['sl_order_id'] = sl_order_id
                 position_details[market_id]['tp_order_id'] = tp_order_id
                 logger.debug(f"Stored SL/TP order IDs ({sl_order_id}, {tp_order_id}) for {market_id}")
         return True
    else:
         return False


def get_order_quantity(exchange, market_id, price, leverage):
    """Calculate order quantity using ccxt market info. (Unchanged logic)"""
    try:
        market = exchange.markets[market_id]
        min_notional = market['limits']['cost']['min'] if market['limits'].get('cost') else None
        min_quantity = market['limits']['amount']['min'] if market['limits'].get('amount') else 0.0

        # Calculate initial quantity
        quantity = (AMOUNT_USD * leverage) / price

        # Adjust for minimum notional if applicable
        if min_notional is not None:
            notional_value = quantity * price
            if notional_value < min_notional:
                quantity = (min_notional / price) * 1.01 # Add 1% buffer
                logger.info(f"Adjusted quantity for {market_id} to meet min notional {min_notional}. New Qty: ~{quantity}")

        # Ensure minimum quantity is met
        if quantity < min_quantity:
             logger.info(f"Adjusted quantity for {market_id} from {quantity} to meet min amount {min_quantity}.")
             quantity = min_quantity

        # Return the calculated quantity (precision applied in place_order)
        return quantity

    except KeyError as e:
         logger.error(f"Market data error for {market_id}: Missing key {e}. Cannot calculate quantity.")
         return 0.0
    except Exception as e:
        logger.exception(f"Error calculating quantity for {market_id}: {e}")
        return 0.0

# is_market_suitable can be adapted to use WS data (e.g. ticker volume, kline range)
def is_market_suitable(symbol):
    """ Check market suitability using data from WS streams (ticker, klines) """
    try:
        # Get data from WS caches
        with ticker_lock:
             ticker = latest_ticker_data.get(symbol)
        df = get_kline_df(symbol) # Gets DataFrame from deque

        if ticker is None or df.empty or len(df) < 20:
            # logger.debug(f"Insufficient data for suitability check: {symbol} (Ticker: {ticker is not None}, Candles: {len(df)})")
            return False # Need recent ticker and sufficient klines

        # Use last N klines for volatility check (e.g., last 30 minutes on 1m or equiv on 3m)
        # If using 3m klines, maybe use last 10-15 candles (30-45 mins)
        recent_klines = df.tail(15)
        if len(recent_klines) < 5: return False # Need at least a few recent candles

        mean_close = recent_klines['close'].mean()
        if mean_close <= 0 or pd.isna(mean_close):
            logger.warning(f"{symbol} has invalid price data in recent klines.")
            return False

        # 1. Price Range Check (using recent klines High/Low)
        price_range_pct = ((recent_klines['high'].max() - recent_klines['low'].min()) / mean_close) * 100
        min_range, max_range = 0.2, 5.0 # Existing thresholds
        if not (min_range <= price_range_pct <= max_range):
             # logger.debug(f"Market {symbol} unsuitable: Price range {price_range_pct:.2f}% outside [{min_range}%, {max_range}%]")
             return False

        # 2. Volume Check (using ticker 24h volume or recent kline volume)
        # Option A: Use 24h Quote Volume from Ticker (simpler)
        quote_volume_24h = ticker.get('quote_volume')
        min_quote_volume = 500000 # Example threshold: $500k daily volume
        if quote_volume_24h is None or quote_volume_24h < min_quote_volume:
            # logger.debug(f"Market {symbol} unsuitable: 24h Quote Volume {quote_volume_24h} < {min_quote_volume}")
            return False

        # Option B: Volume stability from recent klines (like original)
        # volume_data = recent_klines['volume'].replace(0, np.nan).dropna()
        # if len(volume_data) < 5: return False
        # volume_mean = volume_data.mean()
        # volume_std = volume_data.std()
        # if volume_mean <= 1e-9: return False
        # volume_stability = volume_std / volume_mean if volume_mean > 0 else float('inf')
        # max_vol_stability = 1.5
        # if volume_stability > max_vol_stability:
        #     logger.debug(f"Market {symbol} unsuitable: Volume stability {volume_stability:.2f} > {max_vol_stability}")
        #     return False

        # If all checks pass
        # logger.debug(f"Market {symbol} suitable.")
        return True

    except Exception as e:
        logger.warning(f"Error checking market suitability for {symbol}: {e}", exc_info=False)
        return False

def sync_positions_with_exchange(exchange):
    """Synchronize local position tracking with exchange. (Unchanged logic)"""
    logger.info("Synchronizing positions with exchange...")
    try:
        exchange_positions = fetch_open_positions(exchange) # Uses updated fetch_open_positions
        exchange_symbols = set(exchange_positions.keys())

        with position_details_lock: local_symbols = set(position_details.keys())

        closed_locally = local_symbols - exchange_symbols
        for symbol in closed_locally:
            logger.warning(f"Position for {symbol} closed on exchange but still tracked locally. Removing.")
            with position_details_lock:
                 if symbol in position_details: del position_details[symbol]
            with cooldown_lock: # Also clear related cooldown/entry times
                 if symbol in symbol_cooldowns: del symbol_cooldowns[symbol]
                 if symbol in position_entry_times: del position_entry_times[symbol]


        new_on_exchange = exchange_symbols - local_symbols
        for symbol in new_on_exchange:
            logger.warning(f"Position for {symbol} found on exchange but not tracked locally. Adding 'Recovery' entry.")
            position = exchange_positions[symbol]
            entry_price = position.get('entry_price')
            position_size = position.get('size')

            if entry_price is None or position_size is None:
                 logger.error(f"Cannot recover position {symbol}: Missing entry price or size from exchange data.")
                 continue

            position_type = 'long' if position_size > 0 else 'short'

            # Create default details for recovery
            with position_details_lock:
                position_details[symbol] = {
                    'entry_price': entry_price,
                    'stop_loss': entry_price * (0.98 if position_type == 'long' else 1.02), # Wider default SL
                    'target': entry_price * (1.04 if position_type == 'long' else 0.96), # Wider default TP
                    'position_type': position_type,
                    'entry_reason': 'Recovery', 'probability': 0.5,
                    'entry_time': time.time() - 3600, # Assume 1 hour ago
                    'highest_reached': entry_price if position_type == 'long' else None,
                    'lowest_reached': entry_price if position_type == 'short' else None,
                    'signal_type': 'Recovery',
                    # Ensure leverage is stored if available
                    'leverage': position.get('leverage')
                }

            # Log the recovered position entry
            recovery_log = {
                 'entry_price': entry_price, 'stop_loss': position_details[symbol]['stop_loss'],
                 'target': position_details[symbol]['target'], 'entry_reason': 'Recovery',
                 'position_type': position_type, 'position_status': 'Open (Recovery)',
                 'leverage': position.get('leverage')
            }
            log_trade(symbol, recovery_log) # Queue log

        logger.info(f"Position synchronization complete. Exchange: {len(exchange_symbols)}, Local: {len(position_details)}")

    except Exception as e:
        logger.exception(f"Error synchronizing positions: {e}")


# --- Main Trading Loop (Refactored for WebSocket Data) ---

def check_ws_data_freshness(symbol, max_staleness=60):
     """ Check if WS data for symbol is recent """
     with ws_update_time_lock:
         last_update = last_ws_update_time.get(symbol, 0)
     if time.time() - last_update > max_staleness:
         logger.warning(f"WebSocket data for {symbol} is stale (last update {time.time() - last_update:.0f}s ago).")
         return False
     return True

def continuous_loop(exchange):
    """ Main loop using WebSocket data and periodic OI polling. """
    last_metrics_time = time.time()
    last_oi_fetch_time = 0
    oi_fetch_interval = 45 # Seconds between OI fetches

    # Track symbols being actively monitored by WS
    monitored_symbols = set()
    with kline_lock: monitored_symbols.update(latest_kline_data.keys())
    with depth_lock: monitored_symbols.update(latest_depth_data.keys())
    with ticker_lock: monitored_symbols.update(latest_ticker_data.keys())


    while not file_writer_stop_event.is_set(): # Check stop event
        try:
            current_time = time.time()

            # --- Periodic Tasks ---
            if current_time - last_metrics_time > 3600: # Hourly metrics
                log_trade_funnel()
                # report_metrics() # Optionally report full metrics hourly too
                last_metrics_time = current_time

            # Fetch Open Interest for all monitored symbols periodically
            if current_time - last_oi_fetch_time > oi_fetch_interval:
                logger.info(f"Fetching Open Interest for {len(monitored_symbols)} symbols...")
                # Create a temporary set to avoid issues if monitored_symbols changes during iteration
                symbols_to_fetch_oi = list(monitored_symbols) 
                for symbol in symbols_to_fetch_oi:
                    # Fetch OI (already includes caching)
                    _ = fetch_open_interest(exchange, symbol) 
                    time.sleep(0.1) # Small delay between OI fetches
                last_oi_fetch_time = current_time
                logger.info("Open Interest fetch cycle complete.")


            # --- Position Management ---
            # Get current positions (relatively frequent check)
            open_positions = fetch_open_positions(exchange)
            open_symbols = list(open_positions.keys())

            # Determine entry mode
            entry_only_mode = len(open_positions) >= MAX_OPEN_TRADES

            # --- Process Existing Positions for Exit ---
            symbols_to_remove_from_processing = set() # Symbols exited in this cycle
            for symbol in open_symbols:
                if symbol not in monitored_symbols:
                    logger.warning(f"Position exists for {symbol}, but it's not actively monitored by WebSockets. Cannot manage exit.")
                    # Potentially start WS streams for this symbol? Or close manually?
                    continue

                if not check_ws_data_freshness(symbol): continue # Skip if data stale

                try:
                    # Minimum hold time check
                    if not can_exit_position(symbol): continue

                    # Get position details from exchange fetch result
                    position = open_positions[symbol] # Already fetched above
                    position_amt = position['size'] # Use size from fetch_positions
                    entry_price_from_exchange = position['entry_price']

                    if abs(position_amt) < 1e-9 or entry_price_from_exchange is None:
                         logger.warning(f"Invalid position data for {symbol} during exit check (Size: {position_amt}, EP: {entry_price_from_exchange}). Skipping.")
                         # Maybe trigger sync or cleanup
                         continue

                    # Get local details
                    with position_details_lock:
                         if symbol not in position_details:
                             logger.error(f"Position details missing locally for open position {symbol}! Triggering sync/recovery might be needed.")
                             # Attempt a quick sync for this symbol? Risky in loop. Skip for now.
                             continue
                         pos_details = position_details[symbol].copy()

                    # Verify/update entry price (use exchange as source of truth)
                    local_entry_price = pos_details.get('entry_price')
                    if local_entry_price is None or abs(local_entry_price - entry_price_from_exchange) / entry_price_from_exchange > 0.001:
                         logger.warning(f"Updating local entry price for {symbol}: {local_entry_price} -> {entry_price_from_exchange}")
                         pos_details['entry_price'] = entry_price_from_exchange
                         with position_details_lock:
                             if symbol in position_details: position_details[symbol]['entry_price'] = entry_price_from_exchange


                    # Safely extract details needed for exit check
                    entry_price = pos_details['entry_price']
                    stop_loss = pos_details.get('stop_loss')
                    target = pos_details.get('target')
                    position_type = pos_details.get('position_type')
                    trailing_activated = pos_details.get('trailing_activated', False)
                    highest_reached = pos_details.get('highest_reached')
                    lowest_reached = pos_details.get('lowest_reached')
                    signal_type_reason = pos_details.get('signal_type', 'unknown') # Entry reason for stats

                    if not all([entry_price, stop_loss, target, position_type]):
                        logger.error(f"Incomplete position details for {symbol} exit check. Skipping.")
                        continue

                    # Get required market data from WS caches
                    current_price = get_current_price(symbol)
                    order_book = get_order_book(symbol) # Gets ccxt-like format {bids:[], asks:[], ...}
                    open_interest = fetch_open_interest(exchange, symbol) # Get last polled/cached OI

                    if current_price is None or order_book is None:
                        logger.warning(f"Missing price or order book data from WS for {symbol}. Cannot check exit.")
                        continue
                    # OI is allowed to be None if fetch failed, indicator should handle it

                    # Get indicator instance
                    indicator = get_indicator(symbol, exchange)
                    if indicator is None: continue

                    # --- Perform Exit Check ---
                    exit_result = indicator.check_exit_conditions(
                        current_price=current_price,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        position_type=position_type,
                        trailing_activated=trailing_activated,
                        highest_reached=highest_reached,
                        lowest_reached=lowest_reached,
                        order_book=order_book, # Pass ccxt-like OB format
                        open_interest=open_interest # Pass potentially None OI
                    )

                    # Update local state based on result (trailing SL, etc.)
                    new_stop_loss_from_trail = exit_result.get('new_stop_loss', stop_loss)
                    stop_loss_updated = False
                    with position_details_lock:
                        if symbol in position_details: # Still exists?
                            position_details[symbol]['highest_reached'] = exit_result.get('highest_reached')
                            position_details[symbol]['lowest_reached'] = exit_result.get('lowest_reached')
                            position_details[symbol]['trailing_activated'] = exit_result.get('trailing_activated', trailing_activated)
                            if new_stop_loss_from_trail != stop_loss:
                                position_details[symbol]['stop_loss'] = new_stop_loss_from_trail
                                logger.info(f"Trailing Stop Updated: {symbol} {position_type} -> {new_stop_loss_from_trail:.{indicator.indicator.price_decimals(current_price)}f}")
                                stop_loss = new_stop_loss_from_trail # Update for current loop logic
                                stop_loss_updated = True
                                # **** IMPORTANT: If SL is trailed, we need to CANCEL existing SL/TP orders and place NEW ones ****
                                # This part was missing in the original logic too.
                                # Adding basic cancellation/replacement here.
                                old_sl_id = pos_details.get('sl_order_id')
                                old_tp_id = pos_details.get('tp_order_id')
                                logger.info(f"Trailing stop moved for {symbol}. Attempting to replace SL/TP orders.")
                                # Cancel existing orders first (best effort)
                                try:
                                     if old_sl_id: exchange.cancel_order(old_sl_id, symbol)
                                     if old_tp_id: exchange.cancel_order(old_tp_id, symbol)
                                     logger.info(f"Cancelled previous SL/TP orders ({old_sl_id}, {old_tp_id}) for {symbol}")
                                except Exception as cancel_e:
                                     logger.error(f"Failed to cancel old SL/TP for {symbol} during trailing update: {cancel_e}")
                                # Place new SL/TP orders with the updated stop loss
                                time.sleep(1.0) # Wait after cancel
                                # Quantity needs to be the current position amount
                                current_quantity = abs(position_amt)
                                place_sl_tp_orders(exchange, symbol, position_type, current_quantity, new_stop_loss_from_trail, target)


                    # --- Check Exit Trigger ---
                    if exit_result.get('exit_triggered', False):
                        quantity_to_exit = abs(position_amt)
                        exit_side = 'sell' if position_type == 'long' else 'buy'
                        exit_reason = exit_result.get('exit_reason', 'Unknown')
                        profit_pct = exit_result.get('profit_pct', 0)

                        logger.info(f"EXIT TRIGGERED: {symbol} {position_type} | Reason: {exit_reason} | PnL: {profit_pct:.2f}%")

                        # ** Cancel existing SL/TP orders BEFORE placing market exit order **
                        sl_order_id = pos_details.get('sl_order_id')
                        tp_order_id = pos_details.get('tp_order_id')
                        cancelled_bracket = True # Assume success unless error
                        try:
                             logger.info(f"Cancelling bracket orders ({sl_order_id}, {tp_order_id}) before market exit for {symbol}.")
                             if sl_order_id: exchange.cancel_order(sl_order_id, symbol)
                             if tp_order_id: exchange.cancel_order(tp_order_id, symbol)
                             logger.info(f"Successfully cancelled bracket orders for {symbol}.")
                             # Clear IDs from local state immediately after cancellation attempt
                             with position_details_lock:
                                  if symbol in position_details:
                                       position_details[symbol]['sl_order_id'] = None
                                       position_details[symbol]['tp_order_id'] = None
                        except Exception as cancel_e:
                             logger.error(f"Failed to cancel bracket orders for {symbol} before exit: {cancel_e}. Proceeding with exit anyway.")
                             cancelled_bracket = False # Log failure


                        # Place the market exit order
                        exit_order_info = place_order(
                            exchange, symbol, exit_side, quantity_to_exit, current_price,
                            exit_order=True, # Sets reduceOnly
                            place_target_order=False # Do not place SL/TP on an exit order
                        )

                        if exit_order_info:
                            logger.info(f"Exit order placed successfully for {symbol}.")
                            # Log trade closure details
                            exit_price = exit_order_info.get('average', current_price) # Use actual fill price if possible
                            # Recalculate final PnL based on actual exit price
                            if entry_price != 0:
                                 pnl_mult = 1 if position_type == 'long' else -1
                                 final_profit_pct = ((exit_price - entry_price) / entry_price) * pnl_mult * 100
                            else: final_profit_pct = 0

                            exit_log = pos_details.copy()
                            exit_log.update({
                                'exit_price': exit_price,
                                'exit_reason': exit_reason,
                                'profit_pct': final_profit_pct,
                                'position_status': 'Closed',
                                'exit_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                'stop_loss': stop_loss # Log the final SL level (potentially trailed)
                            })
                            # Clean up fields irrelevant for closed trade log
                            keys_to_remove = ['highest_reached', 'lowest_reached', 'trailing_activated', 'entry_time', 'probability', 'sl_order_id', 'tp_order_id'] # Keep entry details, reasons etc.
                            for key in keys_to_remove: exit_log.pop(key, None)

                            log_trade(symbol, exit_log) # Queue the log

                            # Update pattern stats
                            if signal_type_reason != 'unknown' and indicator:
                                try:
                                    indicator.update_pattern_trade_result(signal_type_reason, final_profit_pct, win=(final_profit_pct > 0))
                                except Exception as stat_e: logger.error(f"Error updating pattern stats for {symbol} on exit: {stat_e}")

                            # Clean up local state *after* logging
                            with position_details_lock:
                                if symbol in position_details: del position_details[symbol]
                            with cooldown_lock: # Also clear entry time if present
                                 if symbol in position_entry_times: del position_entry_times[symbol]
                            symbols_to_remove_from_processing.add(symbol) # Avoid re-entry in this cycle

                        else: # Exit order failed
                             logger.error(f"Failed to place exit order for {symbol}. Position remains open.")
                             # Did we fail to cancel brackets? If so, they might still trigger.
                             if not cancelled_bracket:
                                 logger.warning(f" SL/TP orders ({sl_order_id}, {tp_order_id}) might still be active for {symbol} after failed market exit!")
                             # Consider retry logic or alert

                    # else: logger.debug(f"No exit condition met for {symbol}")


                except ccxt.NetworkError as ne: logger.warning(f"Network error on exit for {symbol}: {ne}. Retrying next cycle.")
                except ccxt.ExchangeError as ee: logger.error(f"Exchange error on exit for {symbol}: {ee}. Skipping.")
                except Exception as e: logger.exception(f"Unexpected error processing exit for {symbol}: {e}")


            # --- Process symbols for potential new entries ---
            # Re-fetch open positions count after exits
            active_open_positions_count = len(fetch_open_positions(exchange))
            if active_open_positions_count >= MAX_OPEN_TRADES:
                # logger.debug("Max open trades reached. Skipping new entries.")
                pass # Skip entry loop
            else:
                # Iterate through symbols that have WS data
                symbols_to_scan = list(monitored_symbols) # Use a copy
                for symbol in symbols_to_scan:
                    # Check eligibility for entry
                    if symbol in open_positions or \
                       symbol in symbols_to_remove_from_processing or \
                       is_in_cooldown(symbol):
                        continue

                    if not check_ws_data_freshness(symbol): continue # Skip stale data

                    try:
                        # Market suitability check (using WS data)
                        if not is_market_suitable(symbol):
                            # logger.debug(f"Market {symbol} unsuitable for entry.")
                            continue

                        # Get data for indicator
                        df = get_kline_df(symbol) # Gets DataFrame (incl. potentially partial last candle)
                        order_book = get_order_book(symbol) # Gets ccxt-like format
                        open_interest = fetch_open_interest(exchange, symbol) # Last fetched/cached OI
                        current_price = get_current_price(symbol) # Latest price

                        if df.empty or len(df) < 20 or order_book is None or current_price is None:
                             logger.warning(f"Insufficient data for entry signal check: {symbol} (Candles:{len(df)}, OB:{order_book is not None}, Price:{current_price is not None})")
                             continue

                        # Get indicator
                        indicator = get_indicator(symbol, exchange)
                        if indicator is None: continue

                        # --- Generate Signals ---
                        # The indicator needs to accept DataFrame/Series/Dicts now
                        # Ensure the indicator's compute_signals method is adapted
                        # to potentially handle a non-closed last candle if needed.
                        buy_signals, sell_signals, _, signal_info = indicator.compute_signals(
                            df['open'], df['high'], df['low'], df['close'], # Pass series
                            order_book=order_book,       # Pass OB dict
                            open_interest=open_interest  # Pass OI value
                        )
                        # Note: compute_signals might need access to the full DF including the last candle
                        # for calculating things like current ATR based on the incomplete candle's range.

                        if signal_info is None or signal_info.empty:
                             # logger.debug(f"No signal info generated for {symbol}")
                             continue

                        # Check signal on the latest candle data point available
                        # signal_info index should align with df index
                        latest_idx = signal_info.index[-1]

                        # --- Process Signal ---
                        signal_found = None
                        if buy_signals.loc[latest_idx]: signal_found = 'buy'
                        elif sell_signals.loc[latest_idx]: signal_found = 'sell'

                        if signal_found:
                            side = signal_found
                            position_type = 'long' if side == 'buy' else 'short'

                            # Extract details from signal_info
                            reason = signal_info['reason'].loc[latest_idx]
                            strength = signal_info['strength'].loc[latest_idx]
                            probability = signal_info['probability'].loc[latest_idx]
                            stop_loss = signal_info['stop_loss'].loc[latest_idx]
                            target = signal_info['target'].loc[latest_idx]
                            sl_basis = signal_info['sl_basis'].loc[latest_idx]
                            tp_basis = signal_info['tp_basis'].loc[latest_idx]
                            risk_reward = signal_info['risk_reward'].loc[latest_idx]

                            log_trade_metrics('patterns_detected')

                            # Validation (Signal Strength, SL/TP validity, R/R)
                            signal_threshold = 30
                            required_rr = 1.2
                            if strength < signal_threshold: log_trade_metrics('patterns_below_threshold'); continue
                            if pd.isna(stop_loss) or pd.isna(target) or stop_loss == 0 or target == 0 or risk_reward < required_rr: log_trade_metrics('failed_risk_reward'); logger.warning(f"Skipping {symbol} {side.upper()}: Invalid SL/TP/RR. SL={stop_loss}, TP={target}, RR={risk_reward:.2f}"); continue
                            # Check SL/TP relative to current price
                            if side == 'buy' and (stop_loss >= current_price or target <= current_price): log_trade_metrics('failed_risk_reward'); logger.warning(f"Skipping {symbol} BUY: SL {stop_loss} or TP {target} invalid relative to Price {current_price}"); continue
                            if side == 'sell' and (stop_loss <= current_price or target >= current_price): log_trade_metrics('failed_risk_reward'); logger.warning(f"Skipping {symbol} SELL: SL {stop_loss} or TP {target} invalid relative to Price {current_price}"); continue

                            # Prepare for order placement
                            leverage = get_leverage_for_market(exchange, symbol)
                            if not leverage: continue
                            quantity = get_order_quantity(exchange, symbol, current_price, leverage)
                            if quantity <= 0: logger.error(f"Calculated quantity is zero or negative for {symbol}. Skipping."); continue

                            # Store details *before* placing order
                            entry_time_ts = time.time()
                            with position_details_lock:
                                position_details[symbol] = {
                                    'entry_price': current_price, # Tentative entry price
                                    'stop_loss': stop_loss, 'target': target, 'position_type': position_type,
                                    'entry_reason': reason, 'probability': probability, 'entry_time': entry_time_ts,
                                    'highest_reached': current_price if position_type == 'long' else None,
                                    'lowest_reached': current_price if position_type == 'short' else None,
                                    'signal_type': reason, 'signal_strength': strength, 'risk_reward': risk_reward,
                                    'sl_basis': sl_basis, 'tp_basis': tp_basis, 'leverage': leverage,
                                    'sl_order_id': None, 'tp_order_id': None # Initialize SL/TP order IDs
                                }

                            logger.info(f"Attempting {side.upper()} ENTRY: {symbol} | Reason: {reason} | Strength: {strength:.1f}% | RR: {risk_reward:.2f}")

                            # Place ENTRY order (Market) + SL/TP (Stop Market/Take Profit Market)
                            order_info = place_order(
                                exchange, symbol, side, quantity, current_price,
                                leverage=leverage, exit_order=False,
                                stop_loss=stop_loss, target_price=target, place_target_order=True # place_order now handles SL/TP placement
                            )

                            if order_info:
                                log_trade_metrics('successful_entries')
                                executed_price = order_info.get('average', current_price) # Use actual fill price

                                # Update local state with actual entry price if significantly different
                                if abs(executed_price - current_price) / current_price > 0.0005: # 0.05% diff
                                    logger.info(f"Updating entry price for {symbol} to executed avg: {executed_price}")
                                    with position_details_lock:
                                        if symbol in position_details: position_details[symbol]['entry_price'] = executed_price
                                else:
                                    executed_price = current_price # Assume intended price if very close

                                # Log the successful entry
                                price_decimals = indicator.indicator.price_decimals(executed_price) if indicator and hasattr(indicator, 'indicator') else 6 # Get decimals for formatting
                                logger.info(f"Opened {side.upper()} position for {symbol}. "
                                            f"ExecPx: {executed_price:.{price_decimals}f} | "
                                            f"Strength: {strength:.1f}% | R/R: {risk_reward:.2f} | "
                                            f"SL: {stop_loss:.{price_decimals}f} ({sl_basis}) | "
                                            f"TP: {target:.{price_decimals}f} ({tp_basis})")

                                entry_log = {
                                    'symbol': symbol, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry_time_ts)),
                                    'position_status': 'Open', 'position_type': position_type, 'entry_price': executed_price,
                                    'exit_price': None, 'stop_loss': stop_loss, 'target': target,
                                    'entry_reason': reason, 'exit_reason': None, 'profit_pct': None,
                                    'leverage': leverage, 'signal_strength': strength, 'risk_reward': risk_reward,
                                    'signal_type': reason, 'sl_basis': sl_basis, 'tp_basis': tp_basis, 'probability': probability
                                }
                                log_trade(symbol, entry_log) # Queue the log

                                time.sleep(SLEEP_INTERVAL / 2) # Small pause
                                # Break if only one entry per cycle is desired
                                # break # Uncomment to allow only one new entry per loop cycle
                                # Check if max trades reached after this entry
                                if len(fetch_open_positions(exchange)) >= MAX_OPEN_TRADES:
                                    logger.info("Max open trades reached after entry. Stopping entry scan for this cycle.")
                                    break # Stop scanning for new entries in this cycle

                            else: # Order placement failed (place_order handles logging error)
                                log_trade_metrics('order_placement_errors')
                                # Clean up local state added before attempting order
                                with position_details_lock:
                                    if symbol in position_details and position_details[symbol]['entry_time'] == entry_time_ts:
                                        del position_details[symbol]
                                # No cooldown needed as no position was opened/closed

                        # else: logger.debug(f"No entry signal for {symbol} on latest data.")


                    except ccxt.NetworkError as ne: logger.warning(f"Network error processing entry for {symbol}: {ne}")
                    except ccxt.ExchangeError as ee: logger.error(f"Exchange error processing entry for {symbol}: {ee}")
                    except Exception as e: logger.exception(f"Unexpected error processing entry check for {symbol}: {e}")


            # --- Main Loop Sleep ---
            # logger.debug(f"Main loop cycle finished. Sleeping for {SLEEP_INTERVAL} seconds.")
            time.sleep(SLEEP_INTERVAL)

        except ccxt.RateLimitExceeded as e: logger.warning(f"Rate limit exceeded in main loop: {e}. Sleeping longer."); time.sleep(60)
        except ccxt.NetworkError as e: logger.warning(f"Network error in main loop: {e}. Retrying soon."); time.sleep(20)
        except ccxt.ExchangeError as e: logger.error(f"Exchange error in main loop: {e}. Sleeping."); time.sleep(SLEEP_INTERVAL * 2)
        except KeyboardInterrupt:
             logger.info("KeyboardInterrupt received. Stopping file writer and exiting.")
             file_writer_stop_event.set()
             break # Exit the while loop
        except Exception as e:
            logger.exception(f"CRITICAL error in main trading loop: {e}")
            time.sleep(SLEEP_INTERVAL * 5) # Longer sleep on critical errors


# --- Initialization and Startup ---
def main():
    """ Main function to initialize and start the bot. """
    logger.info("Starting trading bot...")
     # --- ADD THIS SNIPPET ---
    # Set asyncio event loop policy for Windows compatibility with aiodns/aiohttp
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("Set asyncio event loop policy to WindowsSelectorEventLoopPolicy for compatibility.")
        except Exception as e:
             logger.error(f"Failed to set asyncio event loop policy: {e}. WebSocket stability might be affected.")
    # --- END SNIPPET ---

    # Create CCXT exchange instance (for REST API calls)
    exchange = create_exchange()
    try:
        exchange.load_markets()
        logger.info("Markets loaded via CCXT.")
    except Exception as e:
        logger.critical(f"Failed to load markets: {e}. Exiting.")
        return

    # Start File Writer Thread
    file_writer_thread = threading.Thread(target=file_writer_worker, daemon=True)
    file_writer_thread.start()
    logger.info("Started file writer thread.")

    # Initial position synchronization
    sync_positions_with_exchange(exchange)

    # Determine initial symbols to monitor
    # Fetch active symbols OR use a predefined list
    initial_symbols = fetch_active_symbols(exchange, num_symbols=25) # Monitor top 25 initially
    if not initial_symbols:
         logger.critical("No active symbols found to monitor. Exiting.")
         file_writer_stop_event.set() # Signal writer thread to stop
         file_writer_thread.join(timeout=5) # Wait briefly for writer
         return
    logger.info(f"Monitoring initial symbols: {initial_symbols}")


    # Initialize WebSocket Manager
    twm = ThreadedWebsocketManager(
        api_key=os.getenv('BINANCE_API_KEY'),
        api_secret=os.getenv('BINANCE_SECRET_KEY'),
        # Use testnet URL if needed: tld='com' for mainnet, check docs for testnet
        # tld='com' # or 'us' etc. based on exchange
    )
    twm.start() # Start the manager thread

    # --- Start WebSocket Streams for Initial Symbols ---
    active_streams = set() # Keep track of running streams

    # Convert ccxt symbols (e.g., 'ADA/USDT:USDT') to websocket format (e.g., 'ADAUSDT')
    def symbol_to_ws(ccxt_symbol):
        # Basic conversion, might need adjustment based on exact naming conventions
        return ccxt_symbol.split(':')[0].replace('/', '')

    for symbol_ccxt in initial_symbols:
        symbol_ws = symbol_to_ws(symbol_ccxt)
        logger.info(f"Starting WebSocket streams for {symbol_ccxt} (WS: {symbol_ws})")

        try:
            # 1. Fetch Initial Klines (REST) & Initialize Deque
            initial_klines_list = fetch_initial_klines(exchange, symbol_ccxt, timeframe=TIMEFRAME, limit=LIMIT)
            if initial_klines_list is not None: # Check for fetch error
                with kline_lock:
                    latest_kline_data[symbol_ccxt] = deque(initial_klines_list, maxlen=LIMIT + 10)
                logger.info(f"Fetched initial {len(initial_klines_list)} klines for {symbol_ccxt}")
            else:
                logger.error(f"Could not fetch initial klines for {symbol_ccxt}. Indicator might not work.")
                # Continue trying to start streams, but log the issue

            # 2. Start Kline Stream (WS)
            kline_stream_name = twm.start_kline_socket(callback=process_kline_message, symbol=symbol_ws, interval=TIMEFRAME)
            active_streams.add(kline_stream_name)
            logger.debug(f"Kline stream started: {kline_stream_name}")
            time.sleep(0.2) # Small delay between starting streams

            # 3. Fetch Initial Order Book Snapshot (REST) - CRITICAL: Do this *before* processing diffs
            if not fetch_initial_order_book(exchange, symbol_ccxt):
                 logger.error(f"Could not fetch initial order book for {symbol_ccxt}. Depth stream may be inaccurate initially.")
            time.sleep(0.5) # Wait for snapshot fetch and potential initial processing

            # 4. Start Depth Stream (WS) - Use DIFF stream
            # Construct the specific stream name for the differential depth stream
            depth_stream_name_path = f"{symbol_ws.lower()}@depth@{WEBSOCKET_UPDATE_SPEED}"
            logger.info(f"Constructed depth stream path: {depth_stream_name_path}")

            # Use start_multiplex_socket to start custom stream paths
            # Pass the callback and a list containing the stream path(s)
            # The return value is an identifier for the multiplex connection itself.
            multiplex_stream_id = twm.start_multiplex_socket(callback=process_depth_message, streams=[depth_stream_name_path])

            # Add the *path* to active_streams for tracking purposes,
            # as the multiplex_stream_id refers to the connection, not the specific stream path within it.
            active_streams.add(depth_stream_name_path)
            logger.debug(f"Depth stream added via multiplex socket: Path={depth_stream_name_path}, ConnectionID={multiplex_stream_id}")
            time.sleep(0.2)
            # 5. Start Ticker Stream (WS)
            ticker_stream_name = twm.start_symbol_ticker_socket(callback=process_ticker_message, symbol=symbol_ws)
            active_streams.add(ticker_stream_name)
            logger.debug(f"Ticker stream started: {ticker_stream_name}")
            time.sleep(0.2)

            # 6. Initialize Indicator (after initial data is potentially available)
            _ = get_indicator(symbol_ccxt, exchange)

            # 7. Initial OI Fetch (optional, loop handles periodic)
            _ = fetch_open_interest(exchange, symbol_ccxt)

            # Mark symbol as monitored (using ccxt symbol as the key)
            with ws_update_time_lock: last_ws_update_time[symbol_ccxt] = time.time() # Mark as initially updated
            monitored_symbols.add(symbol_ccxt) # Add to the set used by main loop


        except Exception as e:
            logger.error(f"Failed to start streams or fetch initial data for {symbol_ccxt}: {e}", exc_info=True)
            # Clean up any partial data for this symbol?
            with kline_lock: latest_kline_data.pop(symbol_ccxt, None)
            with depth_lock: latest_depth_data.pop(symbol_ccxt, None)
            with ticker_lock: latest_ticker_data.pop(symbol_ccxt, None)
            with ws_update_time_lock: last_ws_update_time.pop(symbol_ccxt, None)
            monitored_symbols.discard(symbol_ccxt)


    logger.info(f"WebSocket streams started for {len(monitored_symbols)} symbols.")

    # Start the main trading loop thread
    trading_thread = threading.Thread(target=continuous_loop, args=(exchange,), daemon=True)
    trading_thread.start()
    logger.info("Started main trading loop thread.")

    # Keep main thread alive to manage WebSocket connection and allow graceful shutdown
    try:
        while trading_thread.is_alive():
             trading_thread.join(timeout=1.0) # Wait for trading thread
    except KeyboardInterrupt:
        logger.info("Shutdown signal received in main thread.")
    finally:
        logger.info("Stopping WebSocket manager...")
        twm.stop() # Stop WebSocket manager gracefully
        logger.info("Signaling file writer thread to stop...")
        file_writer_stop_event.set()
        file_writer_thread.join(timeout=5) # Wait for writer thread
        logger.info("Trading bot shutdown complete.")


if __name__ == "__main__":
    # Load environment variables (e.g., from .env file if using python-dotenv)
    # from dotenv import load_dotenv
    # load_dotenv()
    main()

# --- END OF main_websocket.py ---