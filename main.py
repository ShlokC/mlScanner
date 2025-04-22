import os
import time
import logging
import threading
import ccxt

import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
from collections import defaultdict
from MomentumShortIndicator import MomentumShortIndicator, create_momentum_short_signal_generator

# Configuration
SLEEP_INTERVAL = 1  # seconds
MAX_OPEN_TRADES = 7
AMOUNT_USD = 1
TIMEFRAME = '3m'
MOMENTUM_TIMEFRAME = '5m'  # 5-minute timeframe for momentum strategy
LIMIT = 100
MOMENTUM_LIMIT = 1000  # Need more candles for momentum strategy to detect 20% gain over days
LEVERAGE_OPTIONS = [10, 8]
ENABLE_AUTO_REVERSALS = True  # Enable auto-reversals for momentum strategy 
# Anti-cycling settings
MINIMUM_HOLD_MINUTES = 1  # Minimum time to hold a position regardless of signals (in minutes)
SYMBOL_COOLDOWN_MINUTES = 15  # Time to wait before re-entering after exit (in minutes)

# Thread synchronization
symbol_cooldowns = {}  # {symbol: timestamp_when_cooldown_ends}
position_entry_times = {}  # {symbol: timestamp_when_position_opened}
cooldown_lock = threading.Lock()  # Lock for accessing cooldown data
consecutive_stop_losses = {}  # {symbol: count}
consecutive_stop_loss_lock = threading.Lock()  # Lock for thread safety
# Position details dictionary for enhanced trade tracking
position_details = {}  # {symbol: {'entry_price', 'stop_loss', 'target', 'position_type', 'entry_reason', 'probability', 'entry_time'}}
position_details_lock = threading.Lock()  # Lock for accessing position details

# Global metrics to track trade filtering
global_metrics = {
    'patterns_detected': 0,
    'patterns_below_threshold': 0,
    'insufficient_tf_confirmation': 0,
    'failed_sequence_check': 0,
    'failed_reversal_check': 0,
    'failed_risk_reward': 0,
    'failed_proximity_check': 0,
    'order_placement_errors': 0,
    'successful_entries': 0,
    'last_reset': time.time()
}

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cache for OHLCV data with TTL
ohlcv_cache = defaultdict(lambda: {'data': None, 'timestamp': 0})
CACHE_TTL = 200  # 5 minutes
cache_lock = threading.Lock()  # Lock for accessing cache data

# Dictionary to store indicator instances for each symbol
symbol_indicators = {}
indicator_lock = threading.Lock()  # Lock for accessing indicator data

# Dictionary to store momentum short indicators
momentum_indicators = {}
momentum_lock = threading.Lock()  # Lock for accessing momentum indicators

# Base parameters with fallback defaults
BASE_PARAMS = {
    # Order book parameters
    'ob_depth': 15,
    'min_wall_size': 20,  # Minimum size multiplier for significant walls
    'imbalance_threshold': 1.8,
    
    # Volatility parameters
    'atr_window': 3,
    'vol_multiplier': 1.2,
    
    # Risk parameters
    'risk_reward_ratio': 1.5,
    'profit_target_pct': 0.3,
    'stop_loss_pct': 0.2,
    
    # Trailing stop parameters
    'trailing_start_pct': 0.3,
    'trailing_distance_pct': 0.15,
    
    # Strategy parameters
    'strategy_preference': 'auto',  # 'breakout', 'fade', or 'auto'
    'breakout_confirmation_threshold': 70,  # Score threshold for breakout signals
    'fade_confirmation_threshold': 65,  # Score threshold for fade signals
}

# Parameters for momentum short strategy
MOMENTUM_PARAMS = {
    'lookback_days': 1,          # Check gains over 1 day (24 hours) instead of 3 days
    'min_price_gain': 20.0,      # At least 20% gain instead of 20%
    'kama_period': 40,           # 40 * 5min = 24 hours
    'supertrend_factor': 2.0,    # Standard multiplier
    'supertrend_length': 7,      # Standard length
    'sl_buffer_pct': 6.0         # Stop loss 2% above kama
}

def fetch_extended_historical_data(exchange, symbol, timeframe=MOMENTUM_TIMEFRAME, max_candles=2980):
    """
    Fetch extended historical OHLCV data without caching, using multiple API calls.
    Gets data in chunks of ~1490 candles, then combines them to provide deeper history.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        timeframe: Candle timeframe (default: MOMENTUM_TIMEFRAME)
        max_candles: Maximum candles to fetch in total
        
    Returns:
        DataFrame: Processed OHLCV data with indicators
    """
    try:
        #logger.info(f"Fetching extended historical data for {symbol} on {timeframe} timeframe")
        
        # First batch - most recent candles (up to 1490)
        first_batch_limit = min(1490, max_candles)
        first_batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=first_batch_limit)
        
        if not first_batch or len(first_batch) < 100:  # Need at least some reasonable amount of data
            logger.warning(f"Insufficient data returned for {symbol}: only {len(first_batch) if first_batch else 0} candles")
            return None
            
        # Convert to DataFrame
        df1 = pd.DataFrame(first_batch, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='ms', utc=True)
        
        # If we need more candles, get a second batch
        if len(first_batch) >= 1000 and max_candles > 1490:
            try:
                # Get the earliest timestamp from the first batch
                earliest_ts = int(df1['timestamp'].min().timestamp() * 1000)
                
                # Second batch - older candles before the earliest timestamp in first batch
                second_batch = exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    limit=1490,
                    since=earliest_ts - (1490 * ccxt.Exchange.parse_timeframe(timeframe) * 1000)  # Go back 1490 candles
                )
                
                if second_batch and len(second_batch) > 10:
                    df2 = pd.DataFrame(second_batch, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms', utc=True)
                    
                    # Combine both batches
                    df_combined = pd.concat([df2, df1])
                    # logger.info(f"Combined {len(df2)} older candles with {len(df1)} recent candles for {symbol}")
                else:
                    df_combined = df1
                    logger.info(f"Second batch returned insufficient data for {symbol}, using only first batch")
            except Exception as e:
                logger.warning(f"Error fetching second batch for {symbol}: {e}")
                df_combined = df1
        else:
            df_combined = df1
            
        # Process the combined data
        df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='first')
        df_combined.set_index('timestamp', inplace=True)
        df_combined = df_combined.sort_index()
        
        # Ensure numeric values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
        # Drop any rows with NaN values
        df_combined = df_combined.dropna()
        
        # Calculate required indicators using pandas-ta
        try:
            # 1. kama (Exponential Moving Average)
            kama_period = 40  # 12 hours on 5-minute timeframe
            kama_result = df_combined.ta.kama(length=kama_period)
            if kama_result is not None and not kama_result.empty:
                df_combined[f"kama_{kama_period}"] = kama_result
            else:
                logger.warning(f"Failed to calculate kama for {symbol} - got empty result")
                # Fallback: calculate kama using pandas
                df_combined[f"kama_{kama_period}"] = df_combined['close'].ewm(span=kama_period, adjust=False).mean()
        
            # 2. Supertrend
            supertrend_length = 7
            supertrend_factor = 2.0
            supertrend_result = df_combined.ta.supertrend(length=supertrend_length, multiplier=supertrend_factor)
            if supertrend_result is not None and not isinstance(supertrend_result, type(None)) and not supertrend_result.empty:
                for col in supertrend_result.columns:
                    df_combined[col] = supertrend_result[col]
        except Exception as e:
            logger.warning(f"Error calculating indicators for {symbol}: {e}")
            # Provide fallback kama calculation if the ta method fails
            df_combined[f"kama_{kama_period}"] = df_combined['close'].ewm(span=kama_period, adjust=False).mean()
        
        # Verify we have the kama column
        kama_column = f"kama_{kama_period}"
        if kama_column not in df_combined.columns:
            logger.warning(f"kama column missing for {symbol} after indicator calculation - adding fallback")
            df_combined[kama_column] = df_combined['close'].ewm(span=kama_period, adjust=False).mean()
        
        # Final check to ensure the DataFrame has at least some data
        if df_combined.empty:
            logger.warning(f"Empty DataFrame after processing for {symbol}")
            return None
            
        return df_combined
        
    except Exception as e:
        logger.exception(f"Error fetching extended historical data for {symbol}: {e}")
        return None
# Calculate dynamic parameters based on historical data
def calculate_dynamic_parameters(symbol, exchange, ohlcv_data=None):
    """
    Calculate dynamic parameters based on market conditions and historical data.
    
    Args:
        symbol: Trading pair symbol
        exchange: Exchange instance
        ohlcv_data: Optional pre-fetched OHLCV data
        
    Returns:
        dict: Parameters adjusted to current market conditions
    """
    try:
        # Start with base parameters
        params = BASE_PARAMS.copy()
        
        # Fetch OHLCV data if not provided
        if ohlcv_data is None or ohlcv_data.empty:
            # Get more data for better volatility calculations
            df = fetch_binance_data(exchange, symbol, timeframe=TIMEFRAME, limit=200)
            if df is None or len(df) < 20:
                logger.warning(f"Insufficient historical data for {symbol}, using default parameters")
                return adjust_for_asset_class(symbol, params)
        else:
            df = ohlcv_data
            
        # Calculate recent volatility using ATR (Average True Range)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # Calculate volume characteristics
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_std_20'] = df['volume'].rolling(20).std()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Calculate price characteristics
        df['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['avg_price_range_20'] = df['price_range_pct'].rolling(20).mean()
        
        # Calculate momentum
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        
        # Get latest market stats
        if len(df) > 20:
            latest_close = df['close'].iloc[-1]
            latest_atr = df['atr_14'].iloc[-1]
            atr_pct = latest_atr / latest_close * 100  # ATR as percentage of price
            avg_volume = df['volume'].iloc[-20:].mean()
            volume_std = df['volume'].iloc[-20:].std()
            volume_volatility = volume_std / avg_volume if avg_volume > 0 else 1
            mean_price_range = df['avg_price_range_20'].iloc[-1]
            latest_rsi = df['rsi_14'].iloc[-1]
            
            # Adjust order book parameters based on volatility and volume
            params['ob_depth'] = max(10, min(25, int(15 * (1 + 0.5 * volume_volatility))))
            
            # Adjust wall size based on volume volatility
            # Higher volume volatility requires larger walls to filter noise
            base_wall_size = 20
            if 'BTC' in symbol:
                base_wall_size = 50
            elif 'ETH' in symbol:
                base_wall_size = 20
                
            params['min_wall_size'] = max(15, min(60, int(base_wall_size * (1 + 0.3 * volume_volatility))))
            
            # Adjust imbalance threshold based on price range
            # More volatile markets need higher imbalance to confirm direction
            params['imbalance_threshold'] = max(1.3, min(2.5, 1.8 * (1 + 0.2 * (mean_price_range / 2))))
            
            # Adjust ATR window based on recent volatility patterns
            params['atr_window'] = max(2, min(5, 3 + (1 if volume_volatility > 1.5 else 0)))
            
            # Adjust vol multiplier based on price range
            params['vol_multiplier'] = max(1.0, min(1.5, 1.2 * (1 + 0.15 * (mean_price_range / 3))))
            
            # Adjust risk parameters based on ATR percentage
            # For more volatile markets, we need larger stop distances but better R:R
            params['risk_reward_ratio'] = max(1.3, min(2.0, 1.5 * (1 + 0.2 * (atr_pct / 1.5))))
            
            # Adjust profit target and stop loss percentages based on ATR
            base_target = 0.3
            base_stop = 0.2
            volatility_factor = atr_pct / 1.0  # Normalize to typical 1% ATR
            
            params['profit_target_pct'] = max(0.2, min(0.5, base_target * volatility_factor))
            params['stop_loss_pct'] = max(0.15, min(0.3, base_stop * volatility_factor))
            
            # Adjust trailing parameters based on price characteristics
            params['trailing_start_pct'] = max(0.2, min(0.4, 0.3 * volatility_factor))
            params['trailing_distance_pct'] = max(0.1, min(0.25, 0.15 * volatility_factor))
            
            # Adjust strategy preference based on momentum indicators
            if latest_rsi > 70:
                # Overbought conditions - prefer fade signals
                params['strategy_preference'] = 'fade'
                params['fade_confirmation_threshold'] = 60  # Lower threshold in strong conditions
            elif latest_rsi < 20:
                # Oversold conditions - prefer breakout signals (potential reversal)
                params['strategy_preference'] = 'breakout'
                params['breakout_confirmation_threshold'] = 65  # Lower threshold in strong conditions
            else:
                # Neutral conditions - auto strategy
                params['strategy_preference'] = 'auto'
                params['breakout_confirmation_threshold'] = 70
                params['fade_confirmation_threshold'] = 65
                
            logger.info(f"Calculated dynamic parameters for {symbol} based on ATR: {atr_pct:.2f}%, Vol volatility: {volume_volatility:.2f}")
        
        return params
    
    except Exception as e:
        logger.warning(f"Error calculating dynamic parameters for {symbol}: {e}")
        # Fallback to default adjusted for asset class
        return adjust_for_asset_class(symbol, BASE_PARAMS.copy())

def adjust_for_asset_class(symbol, params):
    """Adjust parameters based on asset type (BTC, ETH, or altcoin)"""
    if 'BTC' in symbol:
        params['min_wall_size'] = 50
    elif 'ETH' in symbol:
        params['min_wall_size'] = 20
    else:
        params['min_wall_size'] = 20
    return params

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    delta = delta.dropna()
    
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = down.abs()
    
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

def log_trade_metrics(reason, increment=True):
    """Increment metrics on trade filtering (console only, no file writing)."""
    global global_metrics
    
    # Initialize if doesn't exist
    if 'patterns_detected' not in global_metrics:
        global_metrics = {
            'patterns_detected': 0,
            'patterns_below_threshold': 0,
            'insufficient_tf_confirmation': 0,
            'failed_sequence_check': 0,
            'failed_reversal_check': 0,
            'failed_risk_reward': 0,
            'failed_proximity_check': 0,
            'order_placement_errors': 0,
            'successful_entries': 0,
            'last_reset': time.time()
        }
    
    # Increment the appropriate counter
    if increment and reason in global_metrics:
        global_metrics[reason] += 1
    
    # Reset metrics periodically (every 24 hours)
    current_time = time.time()
    if current_time - global_metrics.get('last_reset', 0) > 86400:
        # Reset all counters except last_reset
        for key in global_metrics:
            if key != 'last_reset':
                global_metrics[key] = 0
        global_metrics['last_reset'] = current_time
        logger.info("Trade metrics reset")



def get_momentum_indicator(symbol, exchange, force_new=False):
    """Get or create momentum short indicator for a symbol."""
    indicator_instance = None

    with momentum_lock:
        if symbol in momentum_indicators and not force_new:
            indicator_instance = momentum_indicators[symbol]
        else:
            # Use base momentum parameters with symbol
            params = MOMENTUM_PARAMS.copy()
            params['symbol'] = symbol
            
            try:
                momentum_indicators[symbol] = create_momentum_short_signal_generator(**params)
                indicator_instance = momentum_indicators[symbol]
                # logger.info(f"Created momentum short indicator for {symbol}")
                
                # Log the key parameters
                # log_msg = f"Momentum Parameters: Lookback={params['lookback_days']} days, "
                # log_msg += f"Min Gain={params['min_price_gain']}%, "
                # log_msg += f"kama Period={params['kama_period']}, "
                # log_msg += f"SL Buffer={params['sl_buffer_pct']}%"
                # logger.info(log_msg)
                
            except Exception as e:
                logger.critical(f"CRITICAL: Failed to create momentum indicator for {symbol}: {e}")
                return None

    return indicator_instance

def fetch_binance_data(exchange, market_id, timeframe=TIMEFRAME, limit=LIMIT, include_current=True):
    """
    Fetch OHLCV data from Binance with caching and proper timestamp handling.
    """
    try:
        # Generate cache key
        cache_key = f"{market_id}_{timeframe}_{limit}_{include_current}"
        current_time = time.time()
        
        # Check cache with thread safety
        with cache_lock:
            if (cache_key in ohlcv_cache and 
                ohlcv_cache[cache_key]['data'] is not None and 
                current_time - ohlcv_cache[cache_key]['timestamp'] < CACHE_TTL):
                logger.debug(f"Using cached data for {market_id}, age: {current_time - ohlcv_cache[cache_key]['timestamp']:.1f}s")
                return ohlcv_cache[cache_key]['data']
        
        # Fetch new data - increase the limit to ensure we have enough after processing
        actual_fetch_limit = limit * 2  # Double the requested limit to account for potential losses
        ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=actual_fetch_limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Check for duplicate timestamps
        duplicates = df['timestamp'].duplicated()
        if duplicates.any():
            logger.debug(f"Removed {duplicates.sum()} duplicate timestamps for {market_id}")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Now set the timestamp as index after removing duplicates
        df.set_index('timestamp', inplace=True)
        
        # Convert columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Verify the index is still unique after all processing
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        
        # FIX: Improve current candle identification by checking timestamp directly
        current_candle_timestamp = None
        if len(df) > 0:
            tf_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
            current_time_ms = int(current_time * 1000)
            current_candle_start = current_time_ms - (current_time_ms % tf_ms)
            
            # Mark candles with is_current_candle flag
            df['is_current_candle'] = False
            
            # Convert index to timestamp in milliseconds
            df['timestamp_ms'] = df.index.astype(np.int64) // 10**6
            
            # Find the current candle
            current_candle_mask = df['timestamp_ms'] >= current_candle_start
            if current_candle_mask.any():
                df.loc[current_candle_mask, 'is_current_candle'] = True
                current_candle_idx = df[current_candle_mask].index[0]
                current_candle_timestamp = pd.Timestamp(current_candle_start, unit='ms', tz='UTC')
                logger.debug(f"Current candle for {market_id} identified at {current_candle_timestamp}")
            
            # Clean up temporary column
            df.drop('timestamp_ms', axis=1, inplace=True)
        
        # If requested, remove the most recent (potentially incomplete) candle
        if not include_current and 'is_current_candle' in df.columns:
            prev_len = len(df)
            df = df[~df['is_current_candle']]
            if len(df) < prev_len:
                logger.debug(f"Removed current candle for {market_id} (include_current=False)")
        
        # Return only the requested number of candles (from the end)
        if len(df) > limit:
            df = df.iloc[-limit:]
        
        # Update cache with thread safety
        with cache_lock:
            ohlcv_cache[cache_key] = {
                'data': df,
                'timestamp': current_time
            }
        
        logger.debug(f"Fetched {len(df)} candles for {market_id}, current candle included: {include_current}")
        return df
    except Exception as e:
        logger.exception(f"Failed to fetch data for {market_id}: {e}")
        return pd.DataFrame()

def fetch_order_book(exchange, symbol, limit=20):
    """Fetch order book data."""
    try:
        order_book = exchange.fetch_order_book(symbol, limit=limit)
        return order_book
    except Exception as e:
        logger.warning(f"Error fetching order book for {symbol}: {e}")
        return None

def fetch_open_interest(exchange, symbol):
    """
    Fetch open interest data from the 'info' object in the API response.
    """
    try:
        # Try to fetch from exchange if futures market
        if hasattr(exchange, 'fetch_open_interest'):
            response = exchange.fetch_open_interest(symbol)
            
            # The openInterest value is inside the 'info' object
            if 'info' in response and 'openInterest' in response['info']:
                oi_value = float(response['info']['openInterest'])
                logger.debug(f"Retrieved open interest for {symbol}: {oi_value}")
                return oi_value
            else:
                logger.debug(f"Open interest data structure for {symbol} was not as expected: {response}")
        
        # Fallback simulation for testing
        current_price = exchange.fetch_ticker(symbol)['last']
        simulated_oi = current_price * 10000 * (1 + np.random.normal(0, 0.05))
        logger.debug(f"Using simulated open interest for {symbol}: {simulated_oi}")
        return simulated_oi
    except Exception as e:
        logger.warning(f"Error fetching open interest for {symbol}: {e}")
        return None

def create_exchange():
    """Create and return a CCXT exchange object for Binance futures."""
    return ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 30000
    })

def get_position_details(symbol, entry_price, position_type, position_entry_times):
    """Retrieve position details."""
    try:
        with position_details_lock:
            if symbol in position_details:
                return position_details[symbol].copy()
        
        # Default values
        default_details = {
            'entry_price': entry_price,
            'stop_loss': entry_price * (0.997 if position_type == 'long' else 1.003),
            'target': entry_price * (1.006 if position_type == 'long' else 0.994),
            'position_type': position_type,
            'entry_reason': 'Unknown',
            'probability': 0.5,
            'entry_time': position_entry_times.get(symbol, time.time()),
            'highest_reached': entry_price if position_type == 'long' else None,
            'lowest_reached': entry_price if position_type == 'short' else None,
        }
        
        with position_details_lock:
            if symbol not in position_details:
                position_details[symbol] = default_details
            return position_details[symbol].copy()
    
    except Exception as e:
        logger.warning(f"Error retrieving position details for {symbol}: {e}")
        return {
            'entry_price': entry_price,
            'stop_loss': entry_price * (0.997 if position_type == 'long' else 1.003),
            'target': entry_price * (1.006 if position_type == 'long' else 0.994),
            'position_type': position_type,
            'entry_time': time.time() - 3600,
            'highest_reached': entry_price if position_type == 'long' else None,
            'lowest_reached': entry_price if position_type == 'short' else None,
        }


def can_exit_position(symbol):
    """Check if a position has been held long enough to consider exiting."""
    with cooldown_lock:
        if symbol in position_entry_times:
            entry_time = position_entry_times[symbol]
            current_time = time.time()
            hold_time_minutes = (current_time - entry_time) / 60
            
            if hold_time_minutes < MINIMUM_HOLD_MINUTES:
                rkamaining = int(MINIMUM_HOLD_MINUTES - hold_time_minutes)
                logger.info(f"{symbol} position held for {hold_time_minutes:.1f} minutes, must hold for {rkamaining} more minutes")
                return False
    
    return True

def fetch_active_symbols(exchange, sort_type='gainers', top_count=50):
    """
    Fetch active trading symbols sorted by price change in last 24hrs using 5min rolling candles.
    
    Args:
        exchange: CCXT exchange instance
        sort_type: Type of sorting - 'gainers' or 'both'
        top_count: Number of symbols to return for each category
        
    Returns:
        dict or list: When sort_type='both', returns a dict with 'gainers' and 'losers' lists.
                     Otherwise returns a single list of symbols.
    """
    try:
        # First get ticker data to pre-filter markets
        ticker_data = exchange.fetch_tickers()
        markets = exchange.load_markets()
        
        # Filter markets that are USDT-settled swaps, excluding BTC and ETH
        active_markets = [
            symbol for symbol, market in markets.items()
            if market.get('settle') == 'USDT' and market.get('swap') and 'BTC' not in symbol and 'ETH' not in symbol
        ]
        
        # Pre-filter by some minimal volume to avoid processing very illiquid markets
        active_markets_with_volume = [
            symbol for symbol in active_markets 
            if symbol in ticker_data and ticker_data[symbol].get('quoteVolume', 0) > 50000  # Minimum $50K volume
        ]
        
        # Dictionary to store price change metrics
        price_changes = {}
        
        # Get the current time for accurate calculation
        now = int(time.time() * 1000)  # Current time in milliseconds
        
        # Look back 24 hours
        twenty_four_hours_ago = now - (24 * 60 * 60 * 1000)  # 24 hours ago in milliseconds
        
        # Limit processing to top 100 by volume initially to avoid excessive API calls
        pre_filtered_symbols = sorted(
            active_markets_with_volume,
            key=lambda x: ticker_data[x].get('quoteVolume', 0),
            reverse=True
        )[:100]  # Pre-limit to top 100 by volume
        
        logger.info(f"Calculating 24hr rolling price changes for {len(pre_filtered_symbols)} markets")
        
        # For each market, calculate price change
        for symbol in pre_filtered_symbols:
            try:
                # Add slight delay to avoid rate limits
                time.sleep(0.1)
                
                # Fetch 5-minute candles for the last 24 hours
                # We need approximately 40 candles for 24 hours (24 * 12), but we'll ask for 300 to be safe
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', since=twenty_four_hours_ago, limit=300)
                
                if not ohlcv or len(ohlcv) < 12:  # Need enough candles for meaningful calculation (at least 1 hour)
                    continue
                
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Ensure we have data from the last 24 hours
                df = df[df['timestamp'] >= twenty_four_hours_ago]
                
                if len(df) < 12:  # Double-check we have enough data after filtering
                    continue
                
                # Calculate rolling price with 5-minute candles
                df['rolling_close'] = df['close'].rolling(window=12).mean()  # 1-hour rolling average
                df = df.dropna()  # Remove NaN values at the beginning
                
                if len(df) < 12:  # Need enough data after dropping NaNs
                    continue
                
                # Calculate percentage change in price from first to last of the 24-hour period
                first_price = df['close'].iloc[0]
                last_price = df['close'].iloc[-1]
                
                if first_price <= 0:  # Avoid division by zero
                    continue
                    
                price_change_pct = ((last_price - first_price) / first_price) * 100
                
                # Also calculate the maximum gain and maximum drop in the period
                min_price = df['low'].min()
                max_price = df['high'].max()
                
                max_gain_pct = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
                max_drop_pct = ((max_price - min_price) / max_price) * 100 if max_price > 0 else 0
                
                # Store the results
                price_changes[symbol] = {
                    'price_change_pct': price_change_pct,
                    'max_gain_pct': max_gain_pct,
                    'max_drop_pct': max_drop_pct,
                    'first_price': first_price,
                    'last_price': last_price,
                    'min_price': min_price,
                    'max_price': max_price
                }
                
            except Exception as e:
                logger.debug(f"Error calculating price change for {symbol}: {e}")
                continue
        
        # Sort symbols by price change percentage - gainers (highest first)
        gainers = sorted(
            price_changes.keys(),
            key=lambda x: price_changes[x]['price_change_pct'],
            reverse=True
        )
        
        # Take top N from each
        top_gainers = gainers[:top_count]
        
        # Log the top symbols
        # if logger.isEnabledFor(logging.INFO):
        #     logger.info("Top gainers:")
        #     for i, symbol in enumerate(top_gainers[:10]):  # Log just top 10 for brevity
        #         info = price_changes[symbol]
        #         logger.info(f"{i+1}. {symbol}: 24hr Price Change: {info['price_change_pct']:.2f}%, "
        #                    f"Max Gain: {info['max_gain_pct']:.2f}%")
                           
        # Return based on requested sort type
        if sort_type == 'both':
            return {'gainers': top_gainers}
        else:
            return top_gainers  # Default to gainers for backward compatibility
    
    except Exception as e:
        logger.exception(f"Error fetching active symbols: {e}")
        if sort_type == 'both':
            return {'gainers': []}
        else:
            return []
# Function to properly identify momentum short candidates
# In main.py:

def find_momentum_candidates(exchange, sort_type='gainers', limit=20):
    """
    Find coins that have significant price movements in the last 24 hours.
    For shorts: 20%+ gains
    
    Returns dictionaries mapping symbols to their signals along with a list of prioritized symbols.
    """
    try:
        # Use updated fetch_active_symbols to get coins sorted by gains
        active_symbols_dict = fetch_active_symbols(exchange, sort_type=sort_type)
        
        # Determine which symbol lists to process
        if sort_type == 'both':
            symbols_to_process = {'gainers': active_symbols_dict['gainers']}
        else:
            # Default for backward compatibility
            symbols_to_process = {'gainers': active_symbols_dict}
        
        # Results dictionaries to track movements and store signals
        all_short_candidates = {}
        stored_signals = {}
        
        for category, symbol_list in symbols_to_process.items():
            logger.info(f"Analyzing {len(symbol_list)} {category} for significant price movements...")
            
            for symbol in symbol_list:
                try:
                    # Add slight delay to avoid rate limits
                    time.sleep(0.2)
                    
                    # Fetch extended historical data with indicators
                    df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                    
                    if df_historical is None or len(df_historical) < 40:  # Need at least 24h of data
                        logger.debug(f"Insufficient history for {symbol}. Skipping.")
                        continue
                    
                    # Get the momentum indicator for this symbol
                    indicator = get_momentum_indicator(symbol, exchange)
                    if indicator is None:
                        logger.error(f"Failed to get momentum indicator for {symbol}. Skipping.")
                        continue
                    
                    # Update the indicator with historical data
                    indicator.update_price_data(df_historical)
                    
                    # Generate signal using the MomentumShortIndicator's generate_signal method
                    # This reuses all the logic already implemented there
                    signal = indicator.generate_signal()
                    
                    # Store the complete signal - we'll use this for all details
                    stored_signals[symbol] = signal.copy()
                    
                    # Extract all the short-related information from the signal
                    short_conditions = signal.get('short_conditions_met', {})
                    
                    # Only process if we have the expected condition data
                    if short_conditions:
                        # Check if this coin had significant gain in the last 24 hours
                        has_gain = short_conditions.get('price_gain_met', False)
                        
                        if not has_gain:
                            continue  # Skip coins without significant gain (20%)
                        
                        # Extract key information from the signal and conditions
                        gain_pct = float(short_conditions.get('price_gain_pct', '0%').rstrip('%')) if short_conditions.get('price_gain_pct', 'N/A') != 'N/A' else 0
                        low_price = short_conditions.get('low_price')
                        high_price = short_conditions.get('high_price')
                        drawdown_pct = short_conditions.get('drawdown_pct', 0)
                        
                        has_kama_crossunder = short_conditions.get('kama_crossunder_met', False)
                        is_first_crossunder = short_conditions.get('is_first_crossunder', False)
                        is_recent_crossunder = short_conditions.get('is_recent_crossunder', False)
                        crossunder_age = short_conditions.get('crossunder_age_candles', 0)
                        minutes_ago = short_conditions.get('crossunder_minutes_ago', 0)
                        
                        is_below_kama = short_conditions.get('is_below_kama', False)
                        is_supertrend_bearish = short_conditions.get('supertrend_bearish_met', False)
                        
                        current_price = signal.get('price')
                        current_kama = signal.get('kama_value')
                        
                        # Calculate kama_price_diff_pct
                        kama_price_diff_pct = 0
                        if current_kama is not None and current_price > 0:
                            kama_price_diff_pct = ((current_kama - current_price) / current_price) * 100
                        
                        # Store candidate information
                        all_short_candidates[symbol] = {
                            'gain_pct': gain_pct,
                            'low_price': low_price,
                            'high_price': high_price,
                            'current_price': current_price,
                            'drawdown_pct': drawdown_pct,
                            'is_below_kama': is_below_kama,
                            'has_kama_crossunder': has_kama_crossunder,
                            'is_first_crossunder': is_first_crossunder,
                            'is_recent_crossunder': is_recent_crossunder,
                            'crossunder_age': crossunder_age,
                            'crossunder_minutes_ago': minutes_ago,
                            'supertrend_bearish': is_supertrend_bearish,
                            'signal_generated': signal['signal'] == 'sell',
                            'stop_loss': signal['stop_loss'],
                            'kama_value': current_kama,
                            'kama_price_diff_pct': kama_price_diff_pct
                        }
                
                except Exception as e:
                    logger.info(f"Error analyzing {symbol} for momentum signals: {e}")
                    continue
        
        # Log the candidates found
        logger.info(f"Found {len(all_short_candidates)} coins with 20%+ gains in a 24-hour period for SHORT opportunities")
        
        # Prioritize SHORT candidates
        short_ready_symbols = [s for s in all_short_candidates.keys() if all_short_candidates[s]['signal_generated']]
        
        recently_crossed_short_symbols = [
            s for s in all_short_candidates.keys() 
            if all_short_candidates[s]['has_kama_crossunder'] and 
            all_short_candidates[s]['is_recent_crossunder'] and
            all_short_candidates[s]['supertrend_bearish'] and
            all_short_candidates[s]['is_below_kama'] and
            s not in short_ready_symbols
        ]
        
        sorted_recently_crossed_short = sorted(
            recently_crossed_short_symbols,
            key=lambda x: (
                all_short_candidates[x]['crossunder_age'],  # Sort by age of crossunder (newer first)
                -all_short_candidates[x]['drawdown_pct'],
                -all_short_candidates[x]['gain_pct']
            )
        )
        
        sorted_short_ready = sorted(
            short_ready_symbols,
            key=lambda x: (
                0 if all_short_candidates[x]['has_kama_crossunder'] else 1,
                all_short_candidates[x]['crossunder_minutes_ago'] if all_short_candidates[x]['has_kama_crossunder'] else 999,
                -all_short_candidates[x]['drawdown_pct']
            )
        )
        
        # Combine all short groups with priority order
        sorted_short_symbols = sorted_recently_crossed_short + sorted_short_ready
        
        # Add remaining short symbols based on kama crossunder status
        remaining_short_symbols = set(all_short_candidates.keys()) - set(sorted_short_symbols)
        
        sorted_remaining_short = sorted(
            remaining_short_symbols,
            key=lambda x: (
                0 if all_short_candidates[x]['has_kama_crossunder'] else 1,
                0 if all_short_candidates[x]['is_recent_crossunder'] else 1,
                all_short_candidates[x]['crossunder_age'] if all_short_candidates[x]['has_kama_crossunder'] else 999,
                -all_short_candidates[x]['drawdown_pct'],
                -all_short_candidates[x]['gain_pct']
            )
        )
        
        sorted_short_symbols.extend(sorted_remaining_short)
        
        # Log top candidates
        if len(sorted_short_symbols) > 0:
            logger.info("Top momentum short candidates (20%+ gain in 24h, crossunder within 20 candles):")
            for i, symbol in enumerate(sorted_short_symbols[:10]):
                data = all_short_candidates[symbol]
                status_emoji = "🟢" if data['signal_generated'] else ("🔥" if data['is_recent_crossunder'] else "⬇️")
                logger.info(f"{i+1}. {symbol}: {data['gain_pct']:.2f}% gain, {data['drawdown_pct']:.2f}% down from high, "
                       f"crossunder age: {data['crossunder_age']} candles, "
                       f"{status_emoji} {'READY' if data['signal_generated'] else ('RECENT CROSSUNDER' if data['is_recent_crossunder'] else 'Potential')}")
        
        # Check if we have any valid momentum signals
        valid_short_signals = {s: sig for s, sig in stored_signals.items() if sig['signal'] == 'sell'}
        
        logger.info(f"Found {len(valid_short_signals)} valid 'sell' signals for SHORT trades")
        
        # Return results based on requested sort_type
        if sort_type == 'both':
            return {
                'short': {
                    'sorted_symbols': sorted_short_symbols[:limit],
                    'signals': stored_signals
                }
            }
        else:
            # Default to returning short candidates for backward compatibility
            return sorted_short_symbols[:limit], stored_signals
    
    except Exception as e:
        logger.exception(f"Error finding momentum candidates: {e}")
        if sort_type == 'both':
            return {'short': {'sorted_symbols': [], 'signals': {}}}
        else:
            return [], {}
def get_leverage_for_market(exchange, market_id, leverage_options=LEVERAGE_OPTIONS):
    """Set and get leverage for a market."""
    try:
        for leverage in leverage_options:
            try:
                exchange.set_leverage(leverage, market_id)
                return leverage
            except Exception:
                continue
        logger.error(f"Failed to set leverage for {market_id}")
        return None
    except Exception as e:
        logger.exception(f"Error in leverage setting for {market_id}: {e}")
        return None

def fetch_open_positions(exchange):
    """Fetch open positions from the exchange."""
    try:
        positions = exchange.fetch_positions()
        return {
            pos['symbol']: {
                'symbol': pos['symbol'],
                'info': {
                    'positionAmt': str(float(pos['info'].get('positionAmt', 0))),
                    'entryPrice': pos['info'].get('entryPrice', '0')
                }
            }
            for pos in positions if abs(float(pos['info'].get('positionAmt', 0))) > 0
        }
    except Exception as e:
        logger.exception(f"Error fetching positions: {e}")
        return {}

def place_order(exchange, market_id, side, quantity, price, leverage=None, reduceOnly=False):
    """
    Simplified order placement function that just places a market order.
    Properly handles the reduceOnly parameter with fallback for errors.
    """
    params = {}
    
    # Add reduceOnly parameter if specified
    if reduceOnly:
        params['reduceOnly'] = True

    try:
        # Get current market price
        ticker_data = exchange.fetch_ticker(market_id)
        latest_market_price = ticker_data['last']
        
        # Log order details
        logger.info(f"Placing {side.upper()} order for {market_id} - Qty: {quantity:.8f} at Market Price: {latest_market_price:.6f}, reduceOnly: {reduceOnly}")
        
        # Place a simple market order
        try:
            market_order = exchange.create_order(market_id, 'MARKET', side, quantity, params=params)
        except ccxt.InvalidOrder as e:
            # Check for reduceOnly rejection error
            if 'ReduceOnly Order is rejected' in str(e) and reduceOnly:
                logger.warning(f"ReduceOnly order rejected for {market_id}. Attempting without reduceOnly parameter.")
                # Try again without reduceOnly
                params = {}  # Remove the reduceOnly parameter
                market_order = exchange.create_order(market_id, 'MARKET', side, quantity, params=params)
            else:
                # Re-raise other InvalidOrder errors
                raise
        
        # Log success
        logger.info(f"Order successfully placed for {market_id}: {side.upper()} {quantity:.8f} @ {latest_market_price:.6f}, ID: {market_order.get('id', 'N/A')}")
        
        # Record entry time for minimum hold time tracking
        if not reduceOnly:
            with cooldown_lock:
                position_entry_times[market_id] = time.time()
                logger.info(f"Updated entry time for {market_id}")
        
        return market_order

    except Exception as e:
        logger.exception(f"Error placing order for {market_id}: {e}")
        return None

def place_sl_tp_orders(exchange, market_id, entry_side, quantity, stop_loss, target_price, retries=2):
    """Place stop-loss and take-profit orders with enhanced validation."""
    # Validate SL/TP values before proceeding
    if stop_loss is None:
        logger.error(f"Missing SL ({stop_loss}) for {market_id}, cannot place.")
        return False

    # Get current market price for validation
    try:
        current_price = exchange.fetch_ticker(market_id)['last']
        logger.info(f"Validating SL/TP for {market_id} at current price: {current_price}")
        
        # Get position details to verify against entry price
        with position_details_lock:
            if market_id in position_details:
                entry_price = position_details[market_id].get('entry_price', current_price)
            else:
                entry_price = current_price
        
        # Ensure SL is correctly positioned relative to current price AND entry price
        if entry_side == 'buy':  # LONG
            if stop_loss >= current_price:
                logger.error(f"Invalid SL for {market_id} LONG: SL {stop_loss} >= Current {current_price}")
                stop_loss = current_price * 0.99  # Set to 1% below
                logger.info(f"Corrected SL to {stop_loss}")
                
            if target_price and target_price <= current_price:
                logger.error(f"Invalid TP for {market_id} LONG: TP {target_price} <= Current {current_price}")
                target_price = current_price * 1.015  # Set to 1.5% above
                logger.info(f"Corrected TP to {target_price}")
                
        else:  # SHORT
            # IMPROVED: For shorts, ensure SL is above current price
            if stop_loss <= current_price:
                logger.error(f"Invalid SL for {market_id} SHORT: SL {stop_loss} <= Current {current_price}")
                stop_loss = current_price * 1.01  # Set to 1% above
                logger.info(f"Corrected SL to {stop_loss}")
            
            # Make sure SL is also above entry price
            if stop_loss <= entry_price:
                logger.error(f"Invalid SL for {market_id} SHORT: SL {stop_loss} <= Entry {entry_price}")
                stop_loss = entry_price * 1.005  # Set to 0.5% above entry
                logger.info(f"Corrected SL to {stop_loss} (above entry price)")
                
            if target_price and target_price >= current_price:
                logger.error(f"Invalid TP for {market_id} SHORT: TP {target_price} >= Current {current_price}")
                target_price = current_price * 0.985  # Set to 1.5% below
                logger.info(f"Corrected TP to {target_price}")
    except Exception as e:
        logger.warning(f"Error fetching current price for SL/TP validation: {e}")
        # Continue with the original values but log the issue

    inverted_side = 'sell' if entry_side == 'buy' else 'buy'

    try:
        # Place Stop Loss
        sl_params = {
            'stopPrice': exchange.price_to_precision(market_id, stop_loss),
            'reduceOnly': True,
            'timeInForce': 'GTE_GTC',
        }
        logger.info(f"Placing SL Order for {market_id}: {inverted_side.upper()} Qty:{quantity:.8f} @ Stop:{sl_params['stopPrice']}")
        sl_order = exchange.create_order(
            market_id, 'STOP_MARKET', inverted_side, quantity, None, params=sl_params
        )
        logger.info(f"SL order {sl_order.get('id', 'N/A')} placed for {market_id} at {sl_params['stopPrice']}")

        # Wait briefly
        time.sleep(0.5)

        # Place Take Profit if needed
        if target_price is not None:
            tp_params = {
                'stopPrice': exchange.price_to_precision(market_id, target_price),
                'reduceOnly': True,
                'timeInForce': 'GTE_GTC',
            }
            logger.info(f"Placing TP Order for {market_id}: {inverted_side.upper()} Qty:{quantity:.8f} @ Stop:{tp_params['stopPrice']}")
            tp_order = exchange.create_order(
                market_id, 'TAKE_PROFIT_MARKET', inverted_side, quantity, None, params=tp_params
            )
            logger.info(f"TP order {tp_order.get('id', 'N/A')} placed for {market_id} at {tp_params['stopPrice']}")

            # Store order IDs
            with position_details_lock:
                if market_id in position_details:
                    position_details[market_id]['sl_order_id'] = sl_order.get('id')
                    position_details[market_id]['tp_order_id'] = tp_order.get('id')
        else:
            # Store just the SL order ID
            with position_details_lock:
                if market_id in position_details:
                    position_details[market_id]['sl_order_id'] = sl_order.get('id')
                    position_details[market_id]['tp_order_id'] = None
                    
        return True

    except Exception as e:
        logger.error(f"Error placing SL/TP orders for {market_id}: {e}")
        return False

def get_order_quantity(exchange, market_id, price, leverage):
    """Calculate order quantity based on AMOUNT_USD and leverage."""
    try:
        market = exchange.markets[market_id]
        min_notional = market['limits']['cost']['min']
        min_quantity = market['limits']['amount']['min']
        quantity = (AMOUNT_USD * leverage) / price
        notional = quantity * price
        if notional < min_notional:
            quantity = min_notional / price
        return max(quantity, min_quantity)
    except Exception as e:
        logger.exception(f"Error calculating quantity for {market_id}: {e}")
        return 0.0



def sync_positions_with_exchange(exchange):
    """Synchronize local position tracking with exchange positions."""
    # logger.info("Synchronizing positions with exchange...")
    
    try:
        # Get current positions from exchange
        exchange_positions = fetch_open_positions(exchange)
        exchange_symbols = set(exchange_positions.keys())
        
        # Get locally tracked positions
        with position_details_lock:
            local_symbols = set(position_details.keys())
        
        # Check for positions that exist locally but not on exchange
        closed_positions = local_symbols - exchange_symbols
        for symbol in closed_positions:
            logger.info(f"Position for {symbol} found in local tracking but not on exchange - marking as closed")
            
            with position_details_lock:
                if symbol in position_details:
                    del position_details[symbol]
        
        # Check for positions that exist on exchange but not locally
        new_positions = exchange_symbols - local_symbols
        for symbol in new_positions:
            logger.info(f"Position for {symbol} found on exchange but not in local tracking - adding to tracking")
            
            position = exchange_positions[symbol]
            position_amt = float(position['info']['positionAmt'])
            entry_price = float(position['info']['entryPrice'])
            position_type = 'long' if position_amt > 0 else 'short'
            
            # Create default trade details
            with position_details_lock:
                position_details[symbol] = {
                    'entry_price': entry_price,
                    'stop_loss': entry_price * (0.98 if position_type == 'long' else 1.02),
                    'target': entry_price * (1.04 if position_type == 'long' else 0.96),
                    'position_type': position_type,
                    'entry_reason': 'Recovery',
                    'probability': 0.5,
                    'entry_time': time.time() - 3600,  # Assume 1 hour ago
                    'highest_reached': entry_price if position_type == 'long' else None,
                    'lowest_reached': entry_price if position_type == 'short' else None,
                }
            
            logger.info(f"Recovered {position_type} position for {symbol} at entry price {entry_price}")
        
        # logger.info(f"Position synchronization complete. Found {len(exchange_symbols)} active positions on exchange.")
                  
    except Exception as e:
        logger.exception(f"Error synchronizing positions: {e}")
def update_momentum_stop_losses(exchange, update_all_positions=False, update_all_shorts=None):
    """
    Update reference stop losses for momentum positions based on entry prices.
    No actual orders are placed - this just updates the reference values in position_details.
    
    Args:
        exchange: The exchange instance
        update_all_positions: If True, update all positions regardless of signal_type
        update_all_shorts: Legacy parameter, use update_all_positions instead
    """
    # Handle backward compatibility
    if update_all_shorts is not None:
        update_all_positions = update_all_shorts
    try:
        # Get open positions from exchange
        open_positions = fetch_open_positions(exchange)
        
        if not open_positions:
            return
            
        # logger.info(f"Updating entry-price-based reference stop levels for {len(open_positions)} open positions")
        
        for symbol in open_positions:
            try:
                # Skip if not in our position tracking
                with position_details_lock:
                    if symbol not in position_details:
                        continue
                    
                    # Get position details
                    pos_details = position_details[symbol].copy()
                
                # Get position type and entry price
                position_type = pos_details.get('position_type')
                entry_price = pos_details.get('entry_price')
                
                if position_type not in ['long', 'short'] or entry_price is None or entry_price <= 0:
                    logger.warning(f"Invalid position details for {symbol}: type={position_type}, entry_price={entry_price}. Skipping.")
                    continue
                
                # Then check if it's a momentum trade or we're updating all positions
                if not update_all_positions:
                    # Check multiple fields to identify momentum trades
                    signal_type = pos_details.get('signal_type', '')
                    entry_reason = pos_details.get('entry_reason', '')
                    
                    # Skip if not identified as a momentum trade
                    is_momentum_trade = (
                        signal_type.startswith('momentum_') or 
                        'momentum' in entry_reason.lower()
                    )
                    
                    if not is_momentum_trade:
                        logger.debug(f"Skipping {symbol}: Not identified as momentum trade. "
                                   f"signal_type={signal_type}, entry_reason={entry_reason}")
                        continue
                
                # Get the momentum indicator for this symbol
                indicator = get_momentum_indicator(symbol, exchange)
                if not indicator:
                    logger.warning(f"Cannot update SL for {symbol}: Failed to get momentum indicator")
                    continue
                
                # Fetch latest price
                current_price = exchange.fetch_ticker(symbol)['last']
                
                # Calculate the new reference stop loss with buffer - different for long and short
                if position_type == 'long':
                    # For long positions, stop loss is below entry price by buffer percentage
                    new_stop_loss = entry_price * (1 - indicator.sl_buffer_pct / 100)
                else:  # short
                    # For short positions, stop loss is above entry price by buffer percentage
                    new_stop_loss = entry_price * (1 + indicator.sl_buffer_pct / 100)
                
                # Get current stop loss from position details
                current_stop_loss = pos_details.get('stop_loss')
                
                # Update the reference stop loss if needed
                if current_stop_loss is None or abs(current_stop_loss - new_stop_loss) / new_stop_loss > 0.001:  # 0.1% change threshold
                    sl_direction = "below entry" if position_type == "long" else "above entry"
                    logger.info(f"Updating reference SL for {symbol} {position_type}: {current_stop_loss:.6f} -> {new_stop_loss:.6f} "
                             f"(Entry: {entry_price:.6f}, Buffer: {indicator.sl_buffer_pct}%, {sl_direction})")
                    
                    # Update in position details
                    with position_details_lock:
                        if symbol in position_details:
                            position_details[symbol]['stop_loss'] = new_stop_loss
                
                # Special case warnings based on position type
                if position_type == 'long':
                    # For longs, warn if price is close to dropping below stop loss
                    is_close_to_sl = current_price < new_stop_loss * 1.05
                    is_close_to_exit = current_price < entry_price * 0.95
                    
                    if is_close_to_sl and is_close_to_exit:
                        logger.warning(f"{symbol} LONG price {current_price:.6f} is approaching exit conditions: "
                                      f"Entry: {entry_price:.6f}, SL({indicator.sl_buffer_pct}% below entry): {new_stop_loss:.6f}")
                else:  # short
                    # For shorts, warn if price is close to rising above stop loss
                    is_close_to_sl = current_price > new_stop_loss * 0.95
                    is_close_to_exit = current_price > entry_price * 1.05
                    
                    if is_close_to_sl and is_close_to_exit:
                        logger.warning(f"{symbol} SHORT price {current_price:.6f} is approaching exit conditions: "
                                      f"Entry: {entry_price:.6f}, SL({indicator.sl_buffer_pct}% above entry): {new_stop_loss:.6f}")
                
            except Exception as e:
                logger.exception(f"Error updating SL reference for {symbol}: {e}")
                continue
                
    except Exception as e:
        logger.exception(f"Error in update_momentum_stop_losses: {e}")
def momentum_trading_loop(exchange):
    """
    Momentum trading loop using extended historical data.
    Handles short trading opportunities.
    Uses pre-generated signals to avoid redundant calculations.
    """
    while True:
        try:
            # Check for position exits and handle existing positions
            open_positions = fetch_open_positions(exchange)
            
            # Synchronize local position tracking with exchange positions
            # This will automatically clean up any positions that were closed by SL/TP orders
            sync_positions_with_exchange(exchange)
            
            # Check if we can look for new entries
            open_positions = fetch_open_positions(exchange)  # Refresh after sync
            if len(open_positions) >= MAX_OPEN_TRADES:
                # logger.info(f"At maximum number of open trades ({len(open_positions)}/{MAX_OPEN_TRADES}). Skipping new entries.")
                time.sleep(SLEEP_INTERVAL * 3)
                continue  # Skip to next loop iteration, don't look for new entries
            
            # Find coins with significant price movements (only gainers)
            momentum_candidates = find_momentum_candidates(exchange, sort_type='gainers', limit=20)
            
            # Extract short candidates
            if isinstance(momentum_candidates, tuple):
                # Handle the case where find_momentum_candidates returns a tuple
                short_candidates, stored_signals = momentum_candidates
            else:
                # Handle the case where it returns a dictionary 
                short_candidates = momentum_candidates['short']['sorted_symbols']
                stored_signals = momentum_candidates['short']['signals']
            
            logger.info(f"Checking {len(short_candidates)} momentum short candidates for entry signals...")
            
            entry_placed = False  # Flag to track if we've placed an entry
            
            # Process short candidates
            for symbol in short_candidates:
                # Skip if already in position
                if symbol in open_positions:
                    continue
                
                try:
                    # Use the pre-generated signal if available
                    if symbol in stored_signals:
                        signal = stored_signals[symbol]
                    else:
                        # Fallback to generating a new signal if somehow not pre-generated
                        df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                        
                        if df_historical is None or len(df_historical) < 1000:
                            logger.warning(f"Insufficient historical data for {symbol}. Skipping.")
                            continue
                        
                        indicator = get_momentum_indicator(symbol, exchange)
                        if indicator is None:
                            logger.error(f"Failed to get momentum indicator for {symbol}. Skipping.")
                            continue
                        
                        indicator.update_price_data(df_historical)
                        signal = indicator.generate_signal()
                        logger.warning(f"Had to re-generate signal for {symbol} - not found in pre-generated signals")
                    
                    # Look for valid SHORT signals only
                    if signal['signal'] == 'sell':
                        current_price = signal['price']
                        has_kama_crossunder = signal.get('has_kama_crossunder', False)
                        crossunder_age = signal.get('crossunder_age', 0)
                        minutes_ago = signal.get('crossunder_minutes_ago', 0)
                        
                        logger.info(f"MOMENTUM SHORT SIGNAL: {symbol} - {signal['reason']}")
                        
                        leverage = get_leverage_for_market(exchange, symbol)
                        if not leverage:
                            logger.error(f"Failed to set leverage for {symbol}. Skipping.")
                            continue
                            
                        quantity = get_order_quantity(exchange, symbol, current_price, leverage)
                        
                        if quantity <= 0:
                            logger.warning(f"Invalid quantity calculated for {symbol}. Skipping.")
                            continue
                            
                        current_kama = signal.get('kama_value')
                        
                        # Calculate fixed SL and TP based on entry price and leverage
                        # For short, SL is 60% loss of position value
                        stop_loss = current_price * (1 + 0.6/leverage)
                        # Target is 100% gain
                        target_price = current_price * (1 - 1/leverage)
                            
                        with position_details_lock:
                            position_details[symbol] = {
                                'entry_price': current_price,
                                'stop_loss': stop_loss,
                                'target': target_price,
                                'position_type': 'short',
                                'entry_reason': f"Momentum Short: {signal['reason']}",
                                'probability': signal.get('probability', 0.7),
                                'entry_time': time.time(),
                                'highest_reached': None,
                                'lowest_reached': current_price,
                                'signal_type': 'momentum_short',
                                'signal_strength': 70,
                                'kama_value': current_kama,
                                'crossunder_age': crossunder_age,
                                'crossunder_minutes_ago': minutes_ago,
                                'leverage': leverage
                            }
                            
                        logger.info(f"Placing momentum short for {symbol} at {current_price} "
                                 f"(crossunder was {crossunder_age} candles / {minutes_ago} minutes ago)")
                        
                        order = place_order(
                            exchange, symbol, 'sell', quantity, current_price, leverage=leverage
                        )
                        
                        if order:
                            logger.info(f"Opened MOMENTUM SHORT position for {symbol}")
                            
                            executed_price = order.get('average', current_price)
                            with position_details_lock:
                                if symbol in position_details:
                                    position_details[symbol]['entry_price'] = executed_price
                                    # Update SL and TP with executed price
                                    position_details[symbol]['stop_loss'] = executed_price * (1 + 0.6/leverage)
                                    position_details[symbol]['target'] = executed_price * (1 - 1/leverage)
                                    stop_loss = position_details[symbol]['stop_loss']
                                    target_price = position_details[symbol]['target']
                            
                            logger.info(f"Position opened for {symbol} with fixed SL at {stop_loss} and TP at {target_price}")
                            
                            # Place SL and TP orders
                            # place_sl_tp_orders(
                            #     exchange, 
                            #     symbol, 
                            #     'sell',  # Entry side was sell for short
                            #     quantity, 
                            #     stop_loss, 
                            #     target_price
                            # )
                            
                            entry_placed = True
                            
                            open_positions = fetch_open_positions(exchange)
                            if len(open_positions) >= MAX_OPEN_TRADES:
                                logger.info(f"Reached maximum number of open trades ({len(open_positions)}/{MAX_OPEN_TRADES}). Stopping entry processing.")
                                break
                            
                            break
                        else:
                            logger.error(f"Failed to open momentum short for {symbol}")
                            with position_details_lock:
                                if symbol in position_details:
                                    del position_details[symbol]
                
                except Exception as e:
                    logger.exception(f"Error processing momentum short for {symbol}: {e}")
            
            if not entry_placed:
                logger.info("No suitable momentum candidates found for entry")
            
            # Sleep between checks
            time.sleep(SLEEP_INTERVAL * 3)
            
        except Exception as e:
            logger.exception(f"Error in momentum trading loop: {e}")
            time.sleep(SLEEP_INTERVAL * 5)

def main():
    """Main function to start the trading bot."""
    exchange = create_exchange()
    # Update configuration constants - MINIMUM_HOLD_MINUTES is all we need
    global MINIMUM_HOLD_MINUTES
    MINIMUM_HOLD_MINUTES = 2  # Minimum 2 minutes hold time
    # Recovery mode - check for any missed exits
    sync_positions_with_exchange(exchange)
    
    # Start the momentum short strategy loop in a separate thread
    momentum_thread = threading.Thread(target=momentum_trading_loop, args=(exchange,), daemon=True)
    momentum_thread.start()
    logger.info("Started momentum short trading thread")
    
    # Keep the main thread alive
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()