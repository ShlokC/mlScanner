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
SLEEP_INTERVAL = 2  # seconds
MAX_OPEN_TRADES = 2
AMOUNT_USD = 1
TIMEFRAME = '3m'
MOMENTUM_TIMEFRAME = '5m'  # 5-minute timeframe for momentum strategy
LIMIT = 100
MOMENTUM_LIMIT = 1000  # Need more candles for momentum strategy to detect 20% gain over days
LEVERAGE_OPTIONS = [30, 25, 15, 8]

# Anti-cycling settings
MINIMUM_HOLD_MINUTES = 2  # Minimum time to hold a position regardless of signals (in minutes)
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
CACHE_TTL = 300  # 5 minutes
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
    'lookback_days': 3,          # Check gains over 3 days
    'min_price_gain': 20.0,      # At least 20% gain
    'hma_period': 144,           # 288 * 5min = 24 hours
    'supertrend_factor': 1.0,    # Standard multiplier
    'supertrend_length': 7,     # Standard length
    'sl_buffer_pct': 2.0         # Stop loss 1% above hma
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
        # 1. Hull Moving Average (HMA)
        hma_period = 144  # 12 hours on 5-minute timeframe
        df_combined.ta.hma(length=hma_period, append=True)
        hma_col = f"HMA_{hma_period}"
        
        # 2. Supertrend
        supertrend_length = 7
        supertrend_factor = 1.0
        df_combined.ta.supertrend(length=supertrend_length, multiplier=supertrend_factor, append=True)
        
        #logger.info(f"Successfully fetched and processed {len(df_combined)} candles for {symbol} with indicators")
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
            if df is None or len(df) < 30:
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
                base_wall_size = 30
                
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
            elif latest_rsi < 30:
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
        params['min_wall_size'] = 30
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
                # log_msg += f"hma Period={params['hma_period']}, "
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

def is_in_cooldown(symbol):
    """Check if a symbol is in cooldown period after a recent trade."""
    # with cooldown_lock:
    #     if symbol in symbol_cooldowns:
    #         cooldown_until = symbol_cooldowns[symbol]
    #         current_time = time.time()
            
    #         if current_time < cooldown_until:
    #             rhmaining = int((cooldown_until - current_time) / 60)
    #             return True
    #         else:
    #             # Cooldown expired
    #             del symbol_cooldowns[symbol]
    
    return False

def can_exit_position(symbol):
    """Check if a position has been held long enough to consider exiting."""
    with cooldown_lock:
        if symbol in position_entry_times:
            entry_time = position_entry_times[symbol]
            current_time = time.time()
            hold_time_minutes = (current_time - entry_time) / 60
            
            if hold_time_minutes < MINIMUM_HOLD_MINUTES:
                remaining = int(MINIMUM_HOLD_MINUTES - hold_time_minutes)
                logger.info(f"{symbol} position held for {hold_time_minutes:.1f} minutes, must hold for {remaining} more minutes")
                return False
    
    return True

def fetch_active_symbols(exchange):
    """Fetch active trading symbols sorted by price gain in last 1hr using 5min rolling candles."""
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
        one_hour_ago = now - (60 * 60 * 1000)  # One hour ago in milliseconds
        
        # Limit processing to top 100 by volume initially to avoid excessive API calls
        pre_filtered_symbols = sorted(
            active_markets_with_volume,
            key=lambda x: ticker_data[x].get('quoteVolume', 0),
            reverse=True
        )[:100]  # Pre-limit to top 100 by volume
        
        logger.info(f"Calculating 1hr rolling price gains for {len(pre_filtered_symbols)} markets")
        
        # For each market, calculate price change
        for symbol in pre_filtered_symbols:
            try:
                # Add slight delay to avoid rate limits
                time.sleep(0.1)
                
                # Fetch 5-minute candles for the last hour (plus buffer)
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', since=one_hour_ago, limit=20)
                
                if not ohlcv or len(ohlcv) < 6:  # Need enough candles for meaningful calculation
                    continue
                
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Ensure we have data from the last hour
                df = df[df['timestamp'] >= one_hour_ago]
                
                if len(df) < 6:  # Double-check we have enough data after filtering
                    continue
                
                # Calculate rolling price with 5-minute candles
                df['rolling_close'] = df['close'].rolling(window=3).mean()
                df = df.dropna()  # Remove NaN values at the beginning
                
                if len(df) < 3:  # Need enough data after dropping NaNs
                    continue
                
                # Calculate percentage change in price from first to last of the hour
                first_price = df['close'].iloc[0]
                last_price = df['close'].iloc[-1]
                
                if first_price <= 0:  # Avoid division by zero
                    continue
                    
                price_change_pct = ((last_price - first_price) / first_price) * 100
                
                # Store the result
                price_changes[symbol] = price_change_pct
                
            except Exception as e:
                logger.debug(f"Error calculating price change for {symbol}: {e}")
                continue
        
        # Sort symbols by price change percentage (highest first)
        sorted_symbols = sorted(
            price_changes.keys(),
            key=lambda x: price_changes[x],
            reverse=True
        )
        
        # Take top 50
        top_symbols = sorted_symbols[:50]
        
        # Log the top symbols with their price change percentage
        for i, symbol in enumerate(top_symbols[:10]):  # Log just top 10 for brevity
            logger.info(f"{i+1}. {symbol}: 1hr Price Change: {price_changes[symbol]:.2f}%")
        
        return top_symbols
    
    except Exception as e:
        logger.exception(f"Error fetching active symbols: {e}")
        return []
# Function to properly identify momentum short candidates
# In main.py:

def find_momentum_short_candidates(exchange, limit=30):
    """
    Find coins that have gained 20%+ in a rolling window.
    Uses extended historical data fetched in two iterations (up to ~3000 candles).
    """
    try:
        # Use fetch_active_symbols to get coins sorted by 1hr gains
        active_symbols = fetch_active_symbols(exchange)
        
        # Results dictionary to track gains
        all_candidates = {}
        
        logger.info(f"Analyzing {len(active_symbols)} markets from fetch_active_symbols for rolling gains...")
        
        # Process each symbol to find rolling gains
        for symbol in active_symbols:
            try:
                # Add slight delay to avoid rate limits
                time.sleep(0.2)
                
                # Fetch extended historical data with indicators
                df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                
                if df_historical is None or len(df_historical) < 1000:  # Need sufficient data
                    logger.debug(f"Insufficient history for {symbol}. Skipping.")
                    continue
                
                # Get the momentum indicator for this symbol
                indicator = get_momentum_indicator(symbol, exchange)
                if indicator is None:
                    logger.error(f"Failed to get momentum indicator for {symbol}. Skipping.")
                    continue
                
                # Update the indicator with historical data
                indicator.update_price_data(df_historical)
                
                # Check if this coin had significant gain in the last 1-3 days
                has_gain, gain_pct, low_price, high_price, low_idx, high_idx = indicator.check_price_gain(df_historical)
                
                if not has_gain:
                    continue  # Skip coins without significant gain
                
                # Calculate current drawdown from the high
                current_price = df_historical['close'].iloc[-1]
                drawdown_pct = ((high_price - current_price) / high_price) * 100 if high_price and high_price > 0 else 0
                
                # Calculate how long ago the high occurred (in minutes)
                high_minutes_ago = 0
                if high_idx is not None and isinstance(high_idx, pd.Timestamp):
                    current_time = df_historical.index[-1]
                    if isinstance(current_time, pd.Timestamp):
                        time_diff = current_time - high_idx
                        high_minutes_ago = int(time_diff.total_seconds() / 60)
                
                # Generate signal using the precalculated indicators
                signal = indicator.generate_signal()
                # Extract crossunder information from the signal response
                has_hma_crossunder = signal.get('has_hma_crossunder', False)
                crossunder_age = signal.get('crossunder_age', 0)
                minutes_ago = signal.get('crossunder_minutes_ago', 0)
                is_first_crossunder = signal.get('is_first_crossunder', False)
                # Get HMA and Supertrend values from pandas-ta calculated columns
                hma_column = [col for col in df_historical.columns if col.startswith('HMA_')]
                supertrend_dir_column = [col for col in df_historical.columns if col.startswith('SUPERTd_')]
                
                is_below_hma = False
                is_supertrend_bearish = False
                
                if hma_column:
                    current_hma = df_historical[hma_column[0]].iloc[-1]
                    is_below_hma = current_price < current_hma
                
                if supertrend_dir_column:
                    current_supertrend_dir = df_historical[supertrend_dir_column[0]].iloc[-1]
                    is_supertrend_bearish = current_supertrend_dir == -1  # -1 means bearish
                
                # Use the updated check_hma_crossunder method
                # # The indicator will use its internal price_history that was updated earlier
                # has_hma_crossunder, crossunder_age, minutes_ago, is_first_crossunder = indicator.check_hma_crossunder(
                #     df_historical,  # Pass the DataFrame explicitly
                #     hma_values=None  # Let the method find HMA values in the DataFrame
                # )
                
                # Store candidate information
                all_candidates[symbol] = {
                    'gain_pct': gain_pct,
                    'low_price': low_price,
                    'high_price': high_price,
                    'high_minutes_ago': high_minutes_ago,
                    'current_price': current_price,
                    'drawdown_pct': drawdown_pct,
                    'is_below_hma': is_below_hma,
                    'has_hma_crossunder': has_hma_crossunder,
                    'is_first_crossunder': is_first_crossunder,
                    'crossunder_age': crossunder_age,
                    'crossunder_minutes_ago': minutes_ago,
                    'supertrend_bearish': is_supertrend_bearish,
                    'signal_generated': signal['signal'] == 'sell',
                    'stop_loss': signal['stop_loss'],
                    'hma_value': current_hma if hma_column else None,
                    'hma_price_diff_pct': ((current_hma - current_price) / current_price) * 100 if current_hma and current_price > 0 else 0
                }
                
                # Log info about this candidate
                hma_status = "BELOW hma" if all_candidates[symbol]['is_below_hma'] else "ABOVE hma"
                crossunder_text = f", CROSSUNDER {all_candidates[symbol]['crossunder_minutes_ago']} mins ago" if all_candidates[symbol]['has_hma_crossunder'] else ""
                first_cross_text = ", FIRST CROSSUNDER" if all_candidates[symbol]['is_first_crossunder'] else ""
                supertrend_text = ", SUPERTREND BEARISH" if all_candidates[symbol]['supertrend_bearish'] else ""
                
                logger.debug(f"{symbol}: {gain_pct:.2f}% gain from {low_price:.6f} to {high_price:.6f}, "
                          f"now {drawdown_pct:.2f}% down from high, "
                          f"{hma_status}{crossunder_text}{first_cross_text}{supertrend_text}")
            
            except Exception as e:
                logger.info(f"Error analyzing {symbol} for momentum shorts: {e}")
                continue
        
        # Log the candidates found
        logger.info(f"Found {len(all_candidates)} coins with 20%+ gains in a 1-3 day period")
        
        # Prioritize candidates as before
        # First prioritize coins with valid sell signals
        signal_ready_symbols = [s for s in all_candidates.keys() if all_candidates[s]['signal_generated']]
        
        # Prioritize coins that have JUST crossed below hma (within the last 15 minutes)
        just_crossed_symbols = [
            s for s in all_candidates.keys() 
            if all_candidates[s]['has_hma_crossunder'] and 
            all_candidates[s]['is_first_crossunder'] and
            all_candidates[s]['crossunder_minutes_ago'] <= 15 and
            all_candidates[s]['supertrend_bearish'] and
            all_candidates[s]['is_below_hma'] and
            s not in signal_ready_symbols
        ]
        
        # Sort these categories as before
        sorted_just_crossed = sorted(
            just_crossed_symbols,
            key=lambda x: (
                -all_candidates[x]['drawdown_pct'],
                -all_candidates[x]['gain_pct']
            )
        )
        
        sorted_signal_ready = sorted(
            signal_ready_symbols,
            key=lambda x: (
                0 if all_candidates[x]['has_hma_crossunder'] else 1,
                all_candidates[x]['crossunder_minutes_ago'] if all_candidates[x]['has_hma_crossunder'] else 999,
                -all_candidates[x]['drawdown_pct']
            )
        )
        
        # Combine all groups with priority order
        sorted_symbols = sorted_just_crossed + sorted_signal_ready
        
        # Add remaining symbols based on HMA crossunder status
        remaining_symbols = set(all_candidates.keys()) - set(sorted_symbols)
        
        # Sort remaining symbols by crossunder status
        sorted_remaining = sorted(
            remaining_symbols,
            key=lambda x: (
                0 if all_candidates[x]['has_hma_crossunder'] else 1,
                0 if all_candidates[x]['is_first_crossunder'] else 1,
                all_candidates[x]['crossunder_minutes_ago'] if all_candidates[x]['has_hma_crossunder'] else 999,
                -all_candidates[x]['drawdown_pct'],
                -all_candidates[x]['gain_pct']
            )
        )
        
        sorted_symbols.extend(sorted_remaining)
        
        # Log top candidates
        logger.info("Top momentum short candidates:")
        for i, symbol in enumerate(sorted_symbols[:10]):
            data = all_candidates[symbol]
            status_emoji = "🟢" if data['signal_generated'] else ("🔥" if data['is_first_crossunder'] and data['crossunder_minutes_ago'] <= 15 else "⬇️")
            logger.info(f"{i+1}. {symbol}: {data['gain_pct']:.2f}% gain, {data['drawdown_pct']:.2f}% down from high, "
                       f"{status_emoji} {'READY' if data['signal_generated'] else ('FIRST CROSSUNDER' if data['is_first_crossunder'] else 'Potential')}")
        
        # Return the sorted symbols (limiting to the requested number)
        return sorted_symbols[:limit]
    
    except Exception as e:
        logger.exception(f"Error finding momentum short candidates: {e}")
        return []
    
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

def is_market_suitable(symbol, exchange):
    """Check if the market has suitable conditions for trading."""
    try:
        # Get recent volatility
        df = fetch_binance_data(exchange, symbol, timeframe='1m', limit=50)
        if len(df) < 20:
            logger.warning(f"{symbol} has insufficient candle data: {len(df)} candles")
            return False
            
        # Calculate volatility
        mean_close = df['close'].mean()
        if mean_close <= 0 or pd.isna(mean_close):
            logger.warning(f"{symbol} has invalid price data - mean close: {mean_close}")
            return False
            
        price_range_pct = ((df['high'].max() - df['low'].min()) / mean_close) * 100
        
        # Check volume
        volume_data = df['volume'].replace(0, np.nan).dropna()
        if len(volume_data) < 10 or volume_data.sum() < 1e-10:
            logger.warning(f"{symbol} has insufficient volume data ({len(volume_data)} valid points)")
            return False
            
        volume_mean = volume_data.mean()
        volume_std = volume_data.std()
        
        if volume_mean <= 1e-10:
            logger.warning(f"{symbol} has near-zero mean volume: {volume_mean}")
            return False
            
        volume_stability = volume_std / volume_mean
        
        # Market is suitable if:
        # 1. Price range is between 0.2% and 5% in the last 30 minutes
        # 2. Volume is relatively stable (std/mean < 1.5)
        is_suitable = (0.2 <= price_range_pct <= 5.0) and (volume_stability < 1.5)
        
        return is_suitable
    except Exception as e:
        logger.warning(f"Error checking market suitability for {symbol}: {e}")
        return False

def sync_positions_with_exchange(exchange):
    """Synchronize local position tracking with exchange positions."""
    logger.info("Synchronizing positions with exchange...")
    
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
        
        logger.info(f"Position synchronization complete. Found {len(exchange_symbols)} active positions on exchange.")
                  
    except Exception as e:
        logger.exception(f"Error synchronizing positions: {e}")
def update_momentum_stop_losses(exchange, update_all_shorts=False):
    """
    Update reference stop losses for momentum short positions based on current HMA values.
    No actual orders are placed - this just updates the reference values in position_details.
    
    Args:
        exchange: The exchange instance
        update_all_shorts: If True, update all short positions regardless of signal_type
    """
    try:
        # Get open positions from exchange
        open_positions = fetch_open_positions(exchange)
        
        if not open_positions:
            return
            
        logger.info(f"Updating HMA-based reference stop levels for {len(open_positions)} open positions")
        
        for symbol in open_positions:
            try:
                # Skip if not in our position tracking
                with position_details_lock:
                    if symbol not in position_details:
                        continue
                    
                    # Get position details
                    pos_details = position_details[symbol].copy()
                
                # First check if it's a short position
                position_type = pos_details.get('position_type')
                if position_type != 'short':
                    continue
                
                # Then check if it's a momentum short or we're updating all shorts
                if not update_all_shorts:
                    # Check multiple fields to identify momentum shorts
                    signal_type = pos_details.get('signal_type')
                    entry_reason = pos_details.get('entry_reason', '')
                    
                    # Skip if not identified as a momentum short
                    is_momentum_short = (
                        signal_type == 'momentum_short' or 
                        'momentum short' in entry_reason.lower() or
                        'momentum_short' in entry_reason.lower()
                    )
                    
                    if not is_momentum_short:
                        logger.debug(f"Skipping {symbol}: Not identified as momentum short. "
                                   f"signal_type={signal_type}, entry_reason={entry_reason}")
                        continue
                
                # Get the momentum indicator for this symbol
                indicator = get_momentum_indicator(symbol, exchange)
                if not indicator:
                    logger.warning(f"Cannot update SL for {symbol}: Failed to get momentum indicator")
                    continue
                
                # Fetch latest 5-minute candles
                df_5m = fetch_binance_data(exchange, symbol, timeframe=MOMENTUM_TIMEFRAME, limit=300)
                if df_5m is None or len(df_5m) < indicator.hma_period:
                    logger.warning(f"Cannot update SL for {symbol}: Insufficient candle data")
                    continue
                
                # Update the indicator with latest price data
                indicator.update_price_data(df_5m)
                
                # Calculate the latest HMA value
                hma_series = df_5m.ta.hma(length=indicator.hma_period)
                if hma_series.empty:
                    logger.warning(f"Cannot update SL for {symbol}: Failed to calculate HMA")
                    continue
                
                current_hma = hma_series.iloc[-1]
                previous_hma = pos_details.get('hma_value')
                
                # Calculate the new reference stop loss with buffer
                new_stop_loss = current_hma * (1 + indicator.sl_buffer_pct / 100)
                
                # Get current stop loss from position details
                current_stop_loss = pos_details.get('stop_loss')
                entry_price = pos_details.get('entry_price')
                
                # Get current price
                current_price = df_5m['close'].iloc[-1]
                
                # Check if HMA has changed significantly
                should_update = False
                
                # Check if the HMA value changed significantly
                if previous_hma is not None:
                    hma_change_pct = abs((current_hma - previous_hma) / previous_hma) * 100
                    # Update if HMA changed by more than 0.2%
                    if hma_change_pct > 0.2:
                        should_update = True
                else:
                    # No previous HMA value, always update
                    should_update = True
                
                # Update the reference stop loss if needed
                if should_update:
                    logger.info(f"Updating reference SL for {symbol} short: {current_stop_loss:.6f} -> {new_stop_loss:.6f} "
                              f"(HMA: {current_hma:.6f}, Buffer: {indicator.sl_buffer_pct}%, Entry: {entry_price:.6f})")
                    
                    # Update in position details
                    with position_details_lock:
                        if symbol in position_details:
                            position_details[symbol]['stop_loss'] = new_stop_loss
                            position_details[symbol]['hma_value'] = current_hma
                
                # Special case: if price has moved up close to exit conditions, log a warning
                is_close_to_entry = current_price > entry_price * 0.95
                is_close_to_hma_threshold = current_price > new_stop_loss * 0.95
                
                if is_close_to_entry and is_close_to_hma_threshold:
                    logger.warning(f"{symbol} price {current_price:.6f} is approaching exit conditions: "
                                  f"Entry: {entry_price:.6f}, HMA+buffer: {new_stop_loss:.6f}")
                
            except Exception as e:
                logger.exception(f"Error updating SL reference for {symbol}: {e}")
                continue
                
    except Exception as e:
        logger.exception(f"Error in update_momentum_stop_losses: {e}")
def momentum_short_trading_loop(exchange):
    """
    Momentum short trading loop using extended historical data.
    Fetches fresh data in two iterations for deeper historical context.
    """
    while True:
        try:
            # Check for position exits and handle existing positions
            open_positions = fetch_open_positions(exchange)
            
            # Periodically update reference stop losses for momentum short positions
            try:
                update_momentum_stop_losses(exchange, update_all_shorts=False)
            except Exception as e:
                logger.exception(f"Error updating stop losses: {e}")
                
            # Process position exits and potential reversals
            for symbol in open_positions:
                try:
                    # Get position details from exchange
                    position = open_positions[symbol]
                    position_amt = float(position['info'].get('positionAmt', 0))
                    
                    # Skip if position amount is negligible
                    if abs(position_amt) < 1e-9:
                        logger.warning(f"Position amount for {symbol} is zero. Skipping.")
                        continue
                    
                    # Determine current position direction
                    current_position_type = 'long' if position_amt > 0 else 'short'
                    
                    # Get local position details
                    with position_details_lock:
                        if symbol not in position_details:
                            logger.warning(f"Position details not found locally for {symbol}. Creating default.")
                            # Create default position details if missing
                            entry_price = float(position['info'].get('entryPrice', 0))
                            
                            # Get extended historical data with indicators
                            df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                            if df_historical is None or len(df_historical) < 1000:  # Need sufficient data
                                logger.warning(f"Insufficient historical data for {symbol}. Skipping.")
                                continue
                            
                            # Get the momentum indicator for this symbol to access HMA calculation
                            indicator = get_momentum_indicator(symbol, exchange)
                            if indicator is None:
                                logger.error(f"Failed to get momentum indicator for {symbol}. Skipping.")
                                continue
                            
                            # Update the indicator with the historical data
                            indicator.update_price_data(df_historical)
                            
                            # Get HMA column
                            hma_column = [col for col in df_historical.columns if col.startswith('HMA_')]
                            if not hma_column:
                                logger.warning(f"HMA column not found in historical data for {symbol}. Skipping.")
                                continue
                            
                            current_hma = df_historical[hma_column[0]].iloc[-1]
                            
                            # HMA-based stop loss - using the indicator's sl_buffer_pct
                            if current_position_type == 'long':
                                # For long, SL is below HMA
                                stop_loss = current_hma * (1 - indicator.sl_buffer_pct / 100)
                            else:
                                # For short, SL is above HMA
                                stop_loss = current_hma * (1 + indicator.sl_buffer_pct / 100)
                            
                            position_details[symbol] = {
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'position_type': current_position_type,
                                'entry_reason': 'Recovery',
                                'entry_time': time.time() - 3600,  # Assume 1hr ago
                                'highest_reached': entry_price if current_position_type == 'long' else None,
                                'lowest_reached': entry_price if current_position_type == 'short' else None,
                                'hma_value': current_hma
                            }
                        
                        pos_details = position_details[symbol].copy()
                    
                    # Get fresh extended historical data
                    df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                    
                    # Skip if we don't have data
                    if df_historical is None or len(df_historical) < 1000:
                        logger.warning(f"No valid historical data for {symbol}. Skipping.")
                        continue
                    
                    # Get the momentum indicator for this symbol
                    indicator = get_momentum_indicator(symbol, exchange)
                    if indicator is None:
                        logger.warning(f"Failed to get momentum indicator for {symbol}. Using existing SL.")
                        continue
                    
                    # Update the indicator with historical data
                    indicator.update_price_data(df_historical)
                    
                    # Get current HMA value
                    hma_column = [col for col in df_historical.columns if col.startswith('HMA_')]
                    if not hma_column:
                        logger.warning(f"HMA column not found in historical data for {symbol}. Skipping.")
                        continue
                    
                    current_hma = df_historical[hma_column[0]].iloc[-1]
                    
                    # Update HMA value in position details
                    with position_details_lock:
                        if symbol in position_details:
                            position_details[symbol]['hma_value'] = current_hma
                            
                            # Dynamically calculate exit levels based on current HMA
                            if current_position_type == 'long':
                                # For long, exit below HMA with buffer
                                updated_stop_loss = current_hma * (1 - indicator.sl_buffer_pct / 100)
                            else:
                                # For short, exit above HMA with buffer
                                updated_stop_loss = current_hma * (1 + indicator.sl_buffer_pct / 100)
                            
                            position_details[symbol]['stop_loss'] = updated_stop_loss
                            
                    # Get the updated position details
                    with position_details_lock:
                        pos_details = position_details[symbol].copy()
                    
                    # Extract details
                    entry_price = pos_details.get('entry_price')
                    stop_loss = pos_details.get('stop_loss')
                    position_type = pos_details.get('position_type')
                    hma_value = pos_details.get('hma_value')
                    
                    # Get exit signal from the indicator
                    exit_signal = indicator.generate_signal(
                        current_position=position_type,
                        entry_price=entry_price,
                        stop_loss=stop_loss
                    )
                    
                    # Check if exit is triggered according to the indicator
                    exit_triggered = exit_signal.get('exit_triggered', False)
                    execute_reversal = exit_signal.get('execute_reversal', False)
                    
                    # Log current position status
                    logger.debug(f"{symbol} {position_type}: Entry: {entry_price}, HMA+buffer: {stop_loss}, HMA: {hma_value}")
                    
                    # Execute exit and potential reversal if triggered
                    if exit_triggered and can_exit_position(symbol):
                        exit_reason = exit_signal.get('reason', 'Exit conditions met')
                        current_price = exit_signal.get('price', stop_loss)
                        
                        logger.info(f"EXIT TRIGGERED for {symbol} {position_type} | Reason: {exit_reason} | Price: {current_price}, HMA+buffer: {stop_loss}")
                        
                        # Validate the position still exists and get the current amount
                        try:
                            current_positions = exchange.fetch_positions([symbol])
                            if not current_positions or len(current_positions) == 0:
                                logger.warning(f"Position for {symbol} no longer exists. Cleaning up local tracking.")
                                with position_details_lock:
                                    if symbol in position_details:
                                        del position_details[symbol]
                                continue
                                
                            current_position = current_positions[0]
                            current_position_amt = float(current_position['info'].get('positionAmt', 0))
                            
                            # Check if position amount is zero or negligible
                            if abs(current_position_amt) < 1e-9:
                                logger.warning(f"Position amount for {symbol} is zero. Cleaning up local tracking.")
                                with position_details_lock:
                                    if symbol in position_details:
                                        del position_details[symbol]
                                continue
                                
                            # Verify position direction matches what we think it is
                            current_position_direction = 'long' if current_position_amt > 0 else 'short'
                            if current_position_direction != position_type:
                                logger.warning(f"Position direction mismatch for {symbol}: Expected {position_type}, found {current_position_direction}. Updating.")
                                position_type = current_position_direction
                                
                            # Use the actual position amount from the exchange
                            quantity = abs(current_position_amt)
                            
                        except Exception as e:
                            logger.error(f"Error validating position for {symbol}: {e}")
                            # Continue with the amount we have locally, but log the issue
                            quantity = abs(position_amt)
                            logger.warning(f"Using local position amount for {symbol}: {quantity}")
                        
                        # Determine exit side and potential reversal side
                        exit_side = 'buy' if position_type == 'short' else 'sell'
                        reversal_side = 'buy' if position_type == 'short' else 'sell'  # Same as exit_side for simplicity
                        new_position_type = 'long' if position_type == 'short' else 'short'
                        
                        # Set leverage for the exit and potential reversal
                        leverage = get_leverage_for_market(exchange, symbol)
                        if not leverage:
                            logger.error(f"Failed to set leverage for {symbol}. Using default.")
                            leverage = 10  # Default fallback
                        
                        # Step 1: Exit the current position
                        logger.info(f"Exiting {position_type} position for {symbol} - Qty: {quantity}")
                        exit_order = place_order(
                            exchange, symbol, exit_side, quantity, current_price, 
                            leverage=leverage, reduceOnly=True
                        )
                        
                        if not exit_order:
                            logger.error(f"Failed to exit {position_type} position for {symbol}. Cannot proceed with reversal.")
                            continue
                        
                        # Brief pause to ensure exit is processed
                        time.sleep(1)
                        
                        # Only reverse position if the indicator says to do so (typically only for shorts)
                        if execute_reversal:
                            # Step 2: Enter new position in the opposite direction
                            logger.info(f"Reversing to {new_position_type} position for {symbol} - Qty: {quantity}")
                            
                            # Verify the position is closed before opening a new one
                            try:
                                current_positions = exchange.fetch_positions([symbol])
                                if current_positions and len(current_positions) > 0:
                                    current_pos_amt = float(current_positions[0]['info'].get('positionAmt', 0))
                                    if abs(current_pos_amt) > 1e-9:
                                        logger.warning(f"Position for {symbol} still exists with amount {current_pos_amt}. Cannot reverse yet.")
                                        
                                        # Try once more to close it
                                        close_side = 'buy' if current_pos_amt < 0 else 'sell'
                                        close_amt = abs(current_pos_amt)
                                        logger.info(f"Attempting forced close for {symbol}: {close_side} {close_amt}")
                                        close_order = place_order(exchange, symbol, close_side, close_amt, current_price, reduceOnly=True)
                                        if close_order:
                                            logger.info(f"Forced close order placed for {symbol}")
                                            time.sleep(1)
                                        else:
                                            logger.error(f"Failed to force close {symbol}. Skipping reversal.")
                                            continue
                            except Exception as e:
                                logger.error(f"Error verifying position closure for {symbol}: {e}")
                            
                            # Now attempt to open the new position
                            entry_order = place_order(
                                exchange, symbol, reversal_side, quantity, current_price,
                                leverage=leverage, reduceOnly=False
                            )
                            
                            if not entry_order:
                                logger.error(f"Failed to enter {new_position_type} position for {symbol} after successful exit.")
                                
                                # Clean up position details since we exited but failed to re-enter
                                with position_details_lock:
                                    if symbol in position_details:
                                        del position_details[symbol]
                                continue
                            
                            # Position successfully reversed - update position details
                            executed_price = entry_order.get('average', current_price)
                            
                            # Get fresh historical data for updated HMA
                            new_df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                            hma_column = [col for col in new_df_historical.columns if col.startswith('HMA_')] if new_df_historical is not None else []
                            if new_df_historical is not None and hma_column:
                                current_hma = new_df_historical[hma_column[0]].iloc[-1]
                            
                                # Calculate new stop loss based on HMA
                                if new_position_type == 'long':
                                    # For long, SL is below HMA
                                    sl_price = current_hma * (1 - indicator.sl_buffer_pct / 100)
                                else:
                                    # For short, SL is above HMA
                                    sl_price = current_hma * (1 + indicator.sl_buffer_pct / 100)
                            else:
                                # Fallback if we can't get fresh data
                                sl_price = executed_price * (0.97 if new_position_type == 'long' else 1.03)
                                current_hma = None
                            
                            # Update position details
                            with position_details_lock:
                                position_details[symbol] = {
                                    'entry_price': executed_price,
                                    'stop_loss': sl_price,
                                    'position_type': new_position_type,
                                    'entry_reason': f"Position Reversal (from {position_type})",
                                    'entry_time': time.time(),
                                    'highest_reached': executed_price if new_position_type == 'long' else None,
                                    'lowest_reached': executed_price if new_position_type == 'short' else None,
                                    'signal_type': 'reversal',
                                    'signal_strength': 60,
                                    'leverage': leverage,
                                    'hma_value': current_hma
                                }
                            
                            logger.info(f"Successfully reversed {symbol} from {position_type} to {new_position_type} at {executed_price}. Reference SL: {sl_price}, HMA: {current_hma}")
                        else:
                            # If reversal not indicated, just clean up
                            logger.info(f"Exited {position_type} position for {symbol} without reversal (indicator decision).")
                            
                            # Clean up position details
                            with position_details_lock:
                                if symbol in position_details:
                                    del position_details[symbol]
                
                except Exception as e:
                    logger.exception(f"Error processing position exit for {symbol}: {e}")
            
            # Check if we can look for new entries
            open_positions = fetch_open_positions(exchange)  # Refresh after processing exits
            if len(open_positions) >= MAX_OPEN_TRADES:
                logger.info(f"At maximum number of open trades ({len(open_positions)}/{MAX_OPEN_TRADES}). Skipping new entries.")
                time.sleep(SLEEP_INTERVAL * 3)
                continue  # Skip to next loop iteration, don't look for new entries
            
            # Find coins that have gained 20%+ in a 1-3 day period
            momentum_candidates = find_momentum_short_candidates(exchange, limit=30)
            
            logger.info(f"Checking {len(momentum_candidates)} momentum candidates for entry signals...")
            
            entry_placed = False  # Flag to track if we've placed an entry
            
            for symbol in momentum_candidates:
                # Skip if already in position
                if symbol in open_positions:
                    continue
                
                try:
                    # Get extended historical data with indicators
                    df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                    
                    if df_historical is None or len(df_historical) < 1000:
                        logger.warning(f"Insufficient historical data for {symbol}. Skipping.")
                        continue
                    
                    # Get the momentum indicator for this symbol
                    indicator = get_momentum_indicator(symbol, exchange)
                    if indicator is None:
                        logger.error(f"Failed to get momentum indicator for {symbol}. Skipping.")
                        continue
                    
                    # Update the indicator with historical data
                    indicator.update_price_data(df_historical)
                    
                    # Generate signal - now the indicator contains all the logic
                    signal = indicator.generate_signal()
                    
                    # Look for valid SHORT signals only
                    if signal['signal'] == 'sell':
                        is_first_time = signal.get('is_first_crossunder', False)
                        crossunder_age = signal.get('crossunder_age', 999)
                        minutes_ago = signal.get('crossunder_minutes_ago', 0)
                        
                        # Get HMA value from pre-calculated data
                        hma_column = [col for col in df_historical.columns if col.startswith('HMA_')]
                        current_hma = df_historical[hma_column[0]].iloc[-1] if hma_column else signal.get('hma_value')
                        
                        # Log crossunder details for debugging
                        logger.info(f"{symbol}: Crossunder detected {crossunder_age} candles ago ({minutes_ago} minutes ago), First crossunder: {is_first_time}")
                        
                        # We only enter on signals generated by the indicator
                        logger.info(f"MOMENTUM SHORT SIGNAL: {symbol} - {signal['reason']}")
                        
                        # Verify we have a valid stop loss
                        if signal['stop_loss'] is None or not isinstance(signal['stop_loss'], (int, float)) or signal['stop_loss'] <= 0:
                            logger.error(f"Invalid stop loss for {symbol}: {signal['stop_loss']}. Skipping.")
                            continue
                            
                        # Calculate position size with leverage
                        leverage = get_leverage_for_market(exchange, symbol)
                        if not leverage:
                            logger.error(f"Failed to set leverage for {symbol}. Skipping.")
                            continue
                            
                        current_price = signal['price']
                        quantity = get_order_quantity(exchange, symbol, current_price, leverage)
                        
                        if quantity <= 0:
                            logger.warning(f"Invalid quantity calculated for {symbol}. Skipping.")
                            continue
                            
                        # Store position details before placing order
                        with position_details_lock:
                            position_details[symbol] = {
                                'entry_price': current_price,
                                'stop_loss': signal['stop_loss'],
                                'position_type': 'short',
                                'entry_reason': f"Momentum Short: {signal['reason']}",
                                'probability': signal.get('probability', 0.7),
                                'entry_time': time.time(),
                                'highest_reached': None,
                                'lowest_reached': current_price,
                                'signal_type': 'momentum_short',
                                'signal_strength': 70,
                                'hma_value': current_hma,
                                'crossunder_age': crossunder_age,
                                'crossunder_minutes_ago': minutes_ago
                            }
                            
                        # Place order (simple market order)
                        logger.info(f"Placing momentum short for {symbol} at {current_price} (crossunder was {crossunder_age} candles / {minutes_ago} minutes ago)")
                        
                        order = place_order(
                            exchange, symbol, 'sell', quantity, current_price, leverage=leverage
                        )
                        
                        if order:
                            logger.info(f"Opened MOMENTUM SHORT position for {symbol}")
                            
                            # Update entry price if available
                            executed_price = order.get('average', current_price)
                            with position_details_lock:
                                if symbol in position_details:
                                    position_details[symbol]['entry_price'] = executed_price
                            
                            # DO NOT place static SL order - we will check dynamically
                            # Instead, just log the current HMA and potential exit level for reference
                            sl_level = signal['stop_loss']
                            logger.info(f"Position opened without static SL order. Will dynamically check for exit conditions.")
                            logger.info(f"Current HMA: {current_hma}, Entry: {executed_price}, Reference exit level: {sl_level}")
                            
                            entry_placed = True  # Mark that we've placed an entry
                            
                            # IMPORTANT: Check if we've reached MAX_OPEN_TRADES
                            open_positions = fetch_open_positions(exchange)
                            if len(open_positions) >= MAX_OPEN_TRADES:
                                logger.info(f"Reached maximum number of open trades ({len(open_positions)}/{MAX_OPEN_TRADES}). Stopping entry processing.")
                                break  # Exit the loop to avoid exceeding MAX_OPEN_TRADES
                            
                            # Otherwise, break after one trade even if we haven't reached the limit
                            break
                        else:
                            logger.error(f"Failed to open momentum short for {symbol}")
                            # Clean up position details if order failed
                            with position_details_lock:
                                if symbol in position_details:
                                    del position_details[symbol]
                            
                except Exception as e:
                    logger.exception(f"Error processing momentum short for {symbol}: {e}")
            
            if not entry_placed:
                logger.info("No suitable momentum short candidates found for entry")
            
            # Sleep between checks
            time.sleep(SLEEP_INTERVAL * 3)
            
        except Exception as e:
            logger.exception(f"Error in momentum short loop: {e}")
            time.sleep(SLEEP_INTERVAL * 5)
# def continuous_loop(exchange):
#     """
#     Main trading loop that processes market data, generates signals, and manages positions.
#     Updated to use OrderBookOIScalpingIndicator with refined entry/exit.
#     """
#     last_metrics_time = time.time()

#     while True:
#         try:
#             current_time = time.time()

#             # Reset metrics periodically
#             if current_time - last_metrics_time > 3600:  # Every hour
#                 last_metrics_time = current_time

#             # Fetch active trading symbols
#             active_markets = fetch_active_symbols(exchange)[:20] # Process top 20 liquid markets

#             # Get current positions from the API
#             open_positions = fetch_open_positions(exchange)
#             open_symbols = list(open_positions.keys())

#             # Determine if we're in "entry only" mode
#             entry_only_mode = len(open_positions) >= MAX_OPEN_TRADES

#             # Determine markets to process for potential new entries
#             markets_to_process_for_entry = []
#             if not entry_only_mode:
#                 markets_to_process_for_entry = active_markets

#             # --- Process existing positions for exit ---
#             symbols_to_remove_from_processing = set() # Avoid processing symbols just exited
#             for symbol in open_symbols:
#                 if symbol in symbols_to_remove_from_processing:
#                     continue # Skip if just exited in this loop iteration

#                 try:
#                     # Check if we can exit (minimum hold time)
#                     if not can_exit_position(symbol):
#                         continue

#                     # Get position details from exchange data
#                     position = open_positions[symbol]
#                     position_amt = float(position['info'].get('positionAmt', 0))
#                     # Ensure position still exists (amount could be zero due to race condition)
#                     if abs(position_amt) < 1e-9:
#                          logger.warning(f"Position amount for {symbol} is zero during exit check. Might be closed already.")
#                          # Clean up local state if inconsistent
#                          with position_details_lock:
#                              if symbol in position_details: del position_details[symbol]
#                          continue

#                     entry_price_from_exchange = float(position['info'].get('entryPrice', '0')) # Use exchange entry price as primary source

#                     # Get local position details, potentially correcting entry price if needed
#                     with position_details_lock:
#                          if symbol not in position_details:
#                              logger.warning(f"Position details not found locally for {symbol} during exit check. Attempting recovery or skip.")
#                              # Optionally attempt recovery here, or simply skip
#                              continue
#                          pos_details = position_details[symbol].copy() # Work with a copy

#                     # Verify/Update entry price from exchange data if significantly different
#                     local_entry_price = pos_details.get('entry_price')
#                     if local_entry_price is None or abs(local_entry_price - entry_price_from_exchange) / entry_price_from_exchange > 0.001: # 0.1% difference threshold
#                          logger.warning(f"Updating local entry price for {symbol} from {local_entry_price} to exchange value {entry_price_from_exchange}")
#                          pos_details['entry_price'] = entry_price_from_exchange
#                          with position_details_lock:
#                             if symbol in position_details: # Check again before writing
#                                 position_details[symbol]['entry_price'] = entry_price_from_exchange


#                     # Extract other details safely using .get()
#                     entry_price = pos_details.get('entry_price') # Use the potentially updated entry price
#                     stop_loss = pos_details.get('stop_loss')
#                     target = pos_details.get('target')
#                     position_type = pos_details.get('position_type')
#                     trailing_activated = pos_details.get('trailing_activated', False)
#                     highest_reached = pos_details.get('highest_reached')
#                     lowest_reached = pos_details.get('lowest_reached')

#                     # Basic validation of required details
#                     if not all([entry_price, stop_loss, position_type]):
#                         logger.error(f"Incomplete position details for {symbol} during exit check (EP:{entry_price}, SL:{stop_loss}, Type:{position_type}). Skipping.")
#                         continue

#                     # Fetch current price
#                     ticker = exchange.fetch_ticker(symbol)
#                     current_price = ticker['last']
                    
#                     # Check for stop loss or take profit hit (simplified version for this code update)
#                     exit_triggered = False
#                     exit_reason = None
                    
#                     if position_type == 'long':
#                         if current_price <= stop_loss:
#                             exit_triggered = True
#                             exit_reason = 'Stop Loss Hit'
#                         elif target is not None and current_price >= target:
#                             exit_triggered = True
#                             exit_reason = 'Take Profit Hit'
#                     else:  # short
#                         if current_price >= stop_loss:
#                             exit_triggered = True
#                             exit_reason = 'Stop Loss Hit'
#                         elif target is not None and current_price <= target:
#                             exit_triggered = True
#                             exit_reason = 'Take Profit Hit'
                    
#                     # Execute exit if triggered
#                     if exit_triggered:
#                         # Calculate exit quantity (absolute value of current position)
#                         quantity = abs(position_amt)
#                         # Determine exit side
#                         exit_side = 'sell' if position_type == 'long' else 'buy'
                        
#                         logger.info(f"EXIT TRIGGERED for {symbol} {position_type} | Reason: {exit_reason}")
                        
#                         # Place exit order
#                         exit_order = place_order(
#                             exchange, symbol, exit_side, quantity, current_price, exit_order=True
#                         )
                        
#                         if exit_order:
#                             logger.info(f"Exit order placed successfully for {symbol}")
#                             # Clean up position details
#                             with position_details_lock:
#                                 if symbol in position_details:
#                                     del position_details[symbol]
                            
#                             # Add to set to prevent re-processing in entry loop
#                             symbols_to_remove_from_processing.add(symbol)
#                         else:
#                             logger.error(f"Failed to place exit order for {symbol}. Position rhmains open.")

#                 except ccxt.NetworkError as ne:
#                      logger.warning(f"Network error processing exit for {symbol}: {ne}. Will retry.")
#                      time.sleep(5) # Wait before potentially retrying symbol
#                 except ccxt.ExchangeError as ee:
#                      logger.error(f"Exchange error processing exit for {symbol}: {ee}. Skipping symbol for now.")
#                 except Exception as e:
#                     logger.exception(f"Unexpected error processing exit for {symbol}: {e}")


#             # --- Process markets for potential new entries ---
#             active_open_positions = fetch_open_positions(exchange) # Fetch again after exit processing
#             if len(active_open_positions) >= MAX_OPEN_TRADES:
#                 markets_to_process_for_entry = [] # Skip entry scan

#             for symbol in markets_to_process_for_entry:
#                 # Skip symbols just exited, already in position, or in cooldown
#                 if symbol in symbols_to_remove_from_processing or symbol in active_open_positions or is_in_cooldown(symbol):
#                     continue

#                 try:
#                     # Fetch price data (OHLCV)
#                     df = fetch_binance_data(exchange, symbol, timeframe=TIMEFRAME, limit=100)
#                     if df is None or len(df) < 20: # Need enough data for indicator
#                         logger.warning(f"Insufficient OHLCV data for {symbol} ({len(df) if df is not None else 0} candles). Skipping.")
#                         continue
#                     latest_price = df['close'].iloc[-1]

#                     # Fetch order book and open interest data
#                     order_book = fetch_order_book(exchange, symbol)
#                     open_interest = fetch_open_interest(exchange, symbol)
#                     if not order_book:
#                         logger.warning(f"Failed to fetch order book for {symbol}. Skipping entry check.")
#                         continue

#                     # Create or get indicator instance
#                     indicator = get_indicator(symbol, exchange)
#                     if indicator is None:
#                         logger.error(f"Failed to get indicator for {symbol}. Skipping entry check.")
#                         continue

#                     # Compute signals using the indicator's compatibility method
#                     # This method internally calls generate_signals and updates OB/OI/ATR
#                     buy_signals, sell_signals, _, signal_info = indicator.compute_signals(
#                         df['open'], df['high'], df['low'], df['close'],
#                         order_book=order_book,
#                         open_interest=open_interest
#                     )

#                     # Check for signals on the latest candle
#                     if signal_info is None or signal_info.empty:
#                          logger.debug(f"No signal info returned for {symbol}.")
#                          continue

#                     latest_idx = signal_info.index[-1] # Use index from signal_info DataFrame

#                     # --- Process BUY Signal ---
#                     if buy_signals.loc[latest_idx]:
#                         side = 'buy'
#                         position_type = 'long'
#                         # Extract details from signal_info
#                         reason = signal_info['reason'].loc[latest_idx]
#                         strength = signal_info['strength'].loc[latest_idx]
#                         probability = signal_info['probability'].loc[latest_idx]
#                         stop_loss = signal_info['stop_loss'].loc[latest_idx]
#                         target = signal_info['target'].loc[latest_idx]
#                         sl_basis = signal_info['sl_basis'].loc[latest_idx]
#                         tp_basis = signal_info['tp_basis'].loc[latest_idx]
#                         risk_reward = signal_info['risk_reward'].loc[latest_idx]
                        
#                         strategy = signal_info['strategy'].loc[latest_idx] if 'strategy' in signal_info.columns else 'unknown'
                        
#                         # Log detection with strategy
#                         log_trade_metrics('patterns_detected')

#                         # Validate signal strength
#                         signal_threshold = 30 # Use a consistent threshold
#                         if strength < signal_threshold:
#                             log_trade_metrics('patterns_below_threshold')
#                             continue

#                         # Validate SL/TP/RR before proceeding
#                         required_rr = 1.2 # Minimum acceptable R/R
#                         if pd.isna(stop_loss) or pd.isna(target) or stop_loss == 0 or target == 0 or \
#                            stop_loss >= latest_price or target <= latest_price or risk_reward < required_rr:
#                             logger.warning(f"Skipping {symbol} BUY: Invalid SL/TP/RR. SL={stop_loss}, TP={target}, Price={latest_price}, RR={risk_reward:.2f}")
#                             log_trade_metrics('failed_risk_reward')
#                             continue

#                         # Get leverage and quantity
#                         leverage = get_leverage_for_market(exchange, symbol)
#                         if not leverage: continue
#                         quantity = get_order_quantity(exchange, symbol, latest_price, leverage)
#                         if quantity <= 0: continue

#                         # Store details before placing order
#                         with position_details_lock:
#                             position_details[symbol] = {
#                                 'entry_price': latest_price, # Use latest price as intended entry
#                                 'stop_loss': stop_loss,
#                                 'target': target,
#                                 'position_type': position_type,
#                                 'entry_reason': reason,
#                                 'probability': probability,
#                                 'entry_time': time.time(),
#                                 'highest_reached': latest_price, # Initial high for long
#                                 'lowest_reached': None,
#                                 'signal_type': reason, # Store reason for stats mapping on close
#                                 'signal_strength': strength,
#                                 'risk_reward': risk_reward,
#                                 'sl_basis': sl_basis,
#                                 'tp_basis': tp_basis,
#                                 'leverage': leverage # Store leverage used
#                             }

#                         logger.info(f"Attempting {side.upper()} ENTRY for {symbol} | Reason: {reason} | Strength: {strength:.1f}%")

#                         # Place entry order with SL/TP
#                         order = place_order(
#                             exchange, symbol, side, quantity, latest_price,
#                             leverage=leverage, exit_order=False,
#                             stop_loss=stop_loss, target_price=target, place_target_order=True
#                         )

#                         if order:
#                             log_trade_metrics('successful_entries')
#                             # Use actual executed price if available, else latest_price
#                             executed_price = order.get('average', latest_price)
#                             # Update entry price in local state if different
#                             if abs(executed_price - latest_price) / latest_price > 0.0005: # 0.05% diff
#                                 logger.info(f"Updating entry price for {symbol} to executed avg: {executed_price}")
#                                 with position_details_lock:
#                                     if symbol in position_details:
#                                         position_details[symbol]['entry_price'] = executed_price
#                             else:
#                                 executed_price = latest_price # Keep intended price if very close

#                             logger.info(f"Opened {side.upper()} position for {symbol}. "
#                                         f"ExecPx: {executed_price} | "
#                                         f"Strength: {strength:.1f}% | R/R: {risk_reward:.2f} | "
#                                         f"SL: {stop_loss} ({sl_basis}) | "
#                                         f"TP: {target} ({tp_basis})")

#                             time.sleep(SLEEP_INTERVAL / 2) # Small pause after entry
#                             break # Process only one new entry per main loop cycle

#                         else: # Order placement failed
#                             log_trade_metrics('order_placement_errors')
#                             logger.error(f"Failed to place {side.upper()} entry order for {symbol}.")
#                             # Clean up local state if order failed
#                             with position_details_lock:
#                                 if symbol in position_details: del position_details[symbol]


#                     # --- Process SELL Signal ---
#                     elif sell_signals.loc[latest_idx]:
#                         side = 'sell'
#                         position_type = 'short'
#                         # Extract details directly from the latest row of signal_info
#                         reason = signal_info['reason'].loc[latest_idx]
#                         strength = signal_info['strength'].loc[latest_idx]
#                         probability = signal_info['probability'].loc[latest_idx]
#                         stop_loss = signal_info['stop_loss'].loc[latest_idx]
#                         target = signal_info['target'].loc[latest_idx]
#                         sl_basis = signal_info['sl_basis'].loc[latest_idx]
#                         tp_basis = signal_info['tp_basis'].loc[latest_idx]
#                         risk_reward = signal_info['risk_reward'].loc[latest_idx]

#                         # Log detection
#                         log_trade_metrics('patterns_detected')

#                         # Validate signal strength
#                         signal_threshold = 30
#                         if strength < signal_threshold:
#                             log_trade_metrics('patterns_below_threshold')
#                             continue

#                         # Validate SL/TP/RR before proceeding
#                         required_rr = 1.2
#                         if pd.isna(stop_loss) or pd.isna(target) or stop_loss == 0 or target == 0 or \
#                            stop_loss <= latest_price or target >= latest_price or risk_reward < required_rr:
#                             logger.warning(f"Skipping {symbol} SELL: Invalid SL/TP/RR. SL={stop_loss}, TP={target}, Price={latest_price}, RR={risk_reward:.2f}")
#                             log_trade_metrics('failed_risk_reward')
#                             continue

#                         # Get leverage and quantity
#                         leverage = get_leverage_for_market(exchange, symbol)
#                         if not leverage: continue
#                         quantity = get_order_quantity(exchange, symbol, latest_price, leverage)
#                         if quantity <= 0: continue

#                         # Store details before placing order
#                         with position_details_lock:
#                             position_details[symbol] = {
#                                 'entry_price': latest_price,
#                                 'stop_loss': stop_loss,
#                                 'target': target,
#                                 'position_type': position_type,
#                                 'entry_reason': reason,
#                                 'probability': probability,
#                                 'entry_time': time.time(),
#                                 'highest_reached': None,
#                                 'lowest_reached': latest_price, # Initial low for short
#                                 'signal_type': reason, # Store reason for stats mapping
#                                 'signal_strength': strength,
#                                 'risk_reward': risk_reward,
#                                 'sl_basis': sl_basis,
#                                 'tp_basis': tp_basis,
#                                 'leverage': leverage
#                             }

#                         logger.info(f"Attempting {side.upper()} ENTRY for {symbol} | Reason: {reason} | Strength: {strength:.1f}%")

#                         # Place entry order with SL/TP
#                         order = place_order(
#                             exchange, symbol, side, quantity, latest_price,
#                             leverage=leverage, exit_order=False,
#                             stop_loss=stop_loss, target_price=target, place_target_order=True
#                         )

#                         if order:
#                             log_trade_metrics('successful_entries')
#                             executed_price = order.get('average', latest_price)
#                             if abs(executed_price - latest_price) / latest_price > 0.0005:
#                                 logger.info(f"Updating entry price for {symbol} to executed avg: {executed_price}")
#                                 with position_details_lock:
#                                     if symbol in position_details:
#                                         position_details[symbol]['entry_price'] = executed_price
#                             else:
#                                 executed_price = latest_price

#                             logger.info(f"Opened {side.upper()} position for {symbol}. "
#                                         f"ExecPx: {executed_price} | "
#                                         f"Strength: {strength:.1f}% | R/R: {risk_reward:.2f} | "
#                                         f"SL: {stop_loss} ({sl_basis}) | "
#                                         f"TP: {target} ({tp_basis})")

#                             time.sleep(SLEEP_INTERVAL / 2)
#                             break # Process only one new entry per main loop cycle

#                         else: # Order placement failed
#                             log_trade_metrics('order_placement_errors')
#                             logger.error(f"Failed to place {side.upper()} entry order for {symbol}.")
#                             with position_details_lock:
#                                 if symbol in position_details: del position_details[symbol]


#                 except ccxt.NetworkError as ne:
#                      logger.warning(f"Network error processing entry for {symbol}: {ne}. Skipping symbol.")
#                 except ccxt.ExchangeError as ee:
#                      logger.error(f"Exchange error processing entry for {symbol}: {ee}. Skipping symbol.")
#                 except Exception as e:
#                     logger.exception(f"Unexpected error processing entry check for {symbol}: {e}")

#             # Main loop sleep
#             time.sleep(SLEEP_INTERVAL)

#         except ccxt.RateLimitExceeded as e:
#             logger.warning(f"Rate limit exceeded: {e}. Sleeping for 60 seconds.")
#             time.sleep(60)
#         except ccxt.NetworkError as e:
#             logger.warning(f"Main loop network error: {e}. Retrying in 30 seconds.")
#             time.sleep(30)
#         except ccxt.ExchangeError as e:
#             # Handle specific exchange errors if necessary (e.g., maintenance)
#             logger.error(f"Main loop exchange error: {e}. Sleeping.")
#             time.sleep(SLEEP_INTERVAL * 2)
#         except Exception as e:
#             logger.exception(f"Critical error in main trading loop: {e}")
#             # Implement more robust error handling or restart mechanism if needed
#             time.sleep(SLEEP_INTERVAL * 5) # Longer sleep on critical unknown errors

def main():
    """Main function to start the trading bot."""
    exchange = create_exchange()
    # Update configuration constants - MINIMUM_HOLD_MINUTES is all we need
    global MINIMUM_HOLD_MINUTES
    MINIMUM_HOLD_MINUTES = 2  # Minimum 2 minutes hold time
    # Recovery mode - check for any missed exits
    sync_positions_with_exchange(exchange)
    
    # Start the main trading loop
    # trading_thread = threading.Thread(target=continuous_loop, args=(exchange,), daemon=True)
    # trading_thread.start()
    # logger.info("Started main trading thread")
    
    # Start the momentum short strategy loop in a separate thread
    momentum_thread = threading.Thread(target=momentum_short_trading_loop, args=(exchange,), daemon=True)
    momentum_thread.start()
    logger.info("Started momentum short trading thread")
    
    # Keep the main thread alive
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()