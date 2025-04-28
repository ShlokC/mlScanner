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
MAX_OPEN_TRADES = 5  # Maximum number of open trades at any time
BASE_AMOUNT_USD = 1 # Starting position size
MAX_AMOUNT_USD = 3.0  # Maximum position size
AMOUNT_INCREMENT = 1  # Position size increments
TIMEFRAME = '3m'
MOMENTUM_TIMEFRAME = '5m'  # 5-minute timeframe for momentum strategy
LIMIT = 100
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

# Minimized position tracking - only keep max profit tracking
max_profit_tracking = {}  # {symbol: max_profit_pct}
max_profit_lock = threading.Lock()  # Lock for max profit tracking

# NEW: Performance tracking for position sizing
trade_performance = {}  # {symbol: [list of completed trade results]}
performance_lock = threading.Lock()  # Lock for accessing performance data

# NEW: Pyramid tracking
pyramid_details = {}  # {symbol: {'count': 0, 'entries': []}}
pyramid_lock = threading.Lock()  # Lock for pyramid data

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

# NEW: Configuration for pyramiding
PYRAMID_CONFIG = {
    'max_entries': 3,              # Maximum number of pyramid entries per position
    'profit_thresholds': [15, 30, 45],  # Profit thresholds for adding positions (%)
    'size_multipliers': [1.0, 1.5, 2.0]  # Size multipliers for each pyramid entry
}

# NEW: Configuration for reversal handling
REVERSAL_CONFIG = {
    'min_profit_for_reversal': 5,  # Minimum profit % to consider reversal
    'reversal_size_multiplier': 1.5,  # Multiplier for position size on reversal
    'max_reversal_size': 3.0,      # Maximum position size for reversal entries
    'min_reversal_strength': 60    # Minimum reversal strength to trigger reversal (0-100)
}

def calculate_position_profit(entry_price, current_price, position_type, leverage=1):
    """
    Calculate actual position profit percentage including leverage effect.
    
    Args:
        entry_price: Original entry price
        current_price: Current price
        position_type: 'long' or 'short'
        leverage: Position leverage multiplier
        
    Returns:
        float: Position profit percentage with leverage applied
    """
    # Calculate raw price change percentage
    if position_type == 'long':
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
    else:  # short
        price_change_pct = ((entry_price - current_price) / entry_price) * 100
        
    # Apply leverage multiplier to get actual position profit
    position_profit_pct = price_change_pct * leverage
    
    return position_profit_pct

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

def fetch_open_positions(exchange):
    """
    Enhanced fetch_open_positions that extracts all relevant position data from the exchange.
    Properly calculates fixed 6% SL and 20% TP based on price change only, regardless of leverage.
    """
    try:
        positions = exchange.fetch_positions()
        result = {}
        
        for pos in positions:
            # Filter only positions with non-zero amount
            position_amt = float(pos['info'].get('positionAmt', 0))
            if abs(position_amt) > 0:
                symbol = pos['symbol']
                entry_price = float(pos['info'].get('entryPrice', 0))
                
                # Get position information directly from exchange
                leverage = float(pos['info'].get('leverage', 10))
                position_type = 'long' if position_amt > 0 else 'short'
                
                # Get position open time - try to extract or use a fallback
                entry_time = None
                try:
                    # Try to get updateTime from position info (when position was last modified)
                    update_time = float(pos['info'].get('updateTime', 0))
                    if update_time > 0:
                        entry_time = update_time / 1000  # Convert to seconds
                    
                    # Or try to get time from position_entry_times if available
                    if not entry_time and symbol in position_entry_times:
                        entry_time = position_entry_times[symbol]
                except Exception:
                    pass
                
                # Use current time as fallback if we couldn't get entry time
                if not entry_time:
                    entry_time = time.time() - 3600  # Assume 1 hour old as a reasonable fallback
                
                # Calculate position size in USD (initialMargin on Binance)
                position_size_usd = float(pos['info'].get('initialMargin', 0))
                
                # If initialMargin is not available or zero, calculate it from notional value and leverage
                if position_size_usd <= 0:
                    notional_value = abs(position_amt * entry_price)
                    position_size_usd = notional_value / leverage
                
                # Get current market price
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # FIXED: Calculate stop_loss with EXACT 6% price change (not dividing by leverage)
                if position_type == 'long':
                    stop_loss = entry_price * 0.94  # Fixed 6% price decrease
                else:  # short
                    stop_loss = entry_price * 1.06  # Fixed 6% price increase
                
                # FIXED: Calculate target with EXACT 20% price change (not dividing by leverage)
                if position_type == 'long':
                    target_price = entry_price * 1.2  # Fixed 20% price increase
                else:  # short
                    target_price = entry_price * 0.8  # Fixed 20% price decrease
                
                # Calculate current profit percentage with leverage
                if position_type == 'long':
                    price_change_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # short
                    price_change_pct = ((entry_price - current_price) / entry_price) * 100
                current_profit_pct = price_change_pct * leverage
                
                # Get or initialize pyramid data
                # Calculate pyramid count directly from position size
                pyramid_count = int(round(position_size_usd / AMOUNT_INCREMENT))
                # Cap at 0 minimum (no negative values)
                pyramid_count = max(0, pyramid_count)
                pyramid_entries = []
                with pyramid_lock:
                    if symbol in pyramid_details:                        
                        pyramid_entries = pyramid_details[symbol].get('entries', [])
                    else:
                        # Initialize pyramid tracking for new positions
                        pyramid_details[symbol] = {'count': 0, 'entries': []}
                
                # Get max profit tracking if available
                max_profit = current_profit_pct
                with max_profit_lock:
                    if symbol in max_profit_tracking:
                        stored_max_profit = max_profit_tracking[symbol]
                        if stored_max_profit > current_profit_pct:
                            max_profit = stored_max_profit
                        else:
                            # Update if current profit is higher than stored max
                            max_profit_tracking[symbol] = current_profit_pct
                    else:
                        # Initialize with current profit
                        max_profit_tracking[symbol] = current_profit_pct
                
                # Calculate highest_reached and lowest_reached based on position type and price
                highest_reached = None
                lowest_reached = None
                if position_type == 'long':
                    highest_reached = current_price  # Default to current price
                else:  # short
                    lowest_reached = current_price   # Default to current price
                
                # Calculate average entry price if we have pyramid entries
                avg_entry_price = entry_price
                original_entry_price = entry_price
                
                # Combine all data into one comprehensive position info structure
                result[symbol] = {
                    'symbol': symbol,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'quantity': abs(position_amt),
                    'position_size_usd': position_size_usd,
                    'leverage': leverage,
                    'stop_loss': stop_loss,
                    'target': target_price,
                    'entry_time': entry_time,
                    'current_profit_pct': current_profit_pct,
                    'max_profit_pct': max_profit,
                    'pyramid_count': pyramid_count,
                    'pyramid_entries': pyramid_entries,
                    'avg_entry_price': avg_entry_price,
                    'original_entry_price': original_entry_price,
                    'highest_reached': highest_reached,
                    'lowest_reached': lowest_reached
                }
        
        return result
    except Exception as e:
        logger.exception(f"Error fetching positions: {e}")
        return {}

def check_for_pyramid_opportunity(exchange, symbol):
    """
    Check if we should add to an existing position via pyramiding.
    Enforces a minimum 2% price change between pyramid entries.
    """
    try:
        # Get open positions directly from exchange
        open_positions = fetch_open_positions(exchange)
        
        # Skip if position doesn't exist on exchange
        if symbol not in open_positions:
            return False, 0, "No active position"

        # Get position info directly from exchange data
        position_info = open_positions[symbol]
        position_type = position_info['position_type']
        entry_price = position_info['entry_price'] 
        current_total_size = position_info['position_size_usd']
        leverage = position_info['leverage']
        current_price = position_info['current_price']
        
        # Get pyramid details from our tracking
        pyramid_count = position_info['pyramid_count']
        pyramid_entries = position_info['pyramid_entries']

        # Check if we've reached maximum pyramid entries
        if pyramid_count >= PYRAMID_CONFIG['max_entries']:
            return False, 0, f"Maximum pyramid count reached ({pyramid_count})"
        
        # Check if position already at max size
        if current_total_size >= MAX_AMOUNT_USD:
            #logger.info(f"Cannot pyramid {symbol}: Position size ${current_total_size:.2f} already at or exceeds max ${MAX_AMOUNT_USD:.2f}")
            return False, 0, f"Position already at maximum size (${current_total_size:.2f})"
        # Check if we're already very close to max size (within 0.5 USD or 90%, whichever is smaller)
        threshold = min(0.5, MAX_AMOUNT_USD * 0.1)  # Either $0.5 or 10% of max, whichever is smaller
        if (MAX_AMOUNT_USD - current_total_size) <= threshold:
            logger.info(f"Position size ${current_total_size:.2f} is already close enough to maximum ${MAX_AMOUNT_USD:.2f}. Considering pyramid complete.")
            return False, 0, f"Position already effectively at maximum size (${current_total_size:.2f})"
        # Get the previous pyramid level price or use original entry price
        if pyramid_entries and len(pyramid_entries) > 0:
            previous_level_price = pyramid_entries[-1].get('price', entry_price)
        else:
            previous_level_price = entry_price
            
        # Calculate the absolute price change since previous entry
        price_change_pct = 0
        
        if position_type == 'short':
            # For SHORT, we need price to move down by at least 2% from previous level
            price_change_pct = ((previous_level_price - current_price) / previous_level_price) * 100
            # Check if price has moved enough (minimum 2% price decrease)
            if price_change_pct < 2.0:
                return False, 0, f"Insufficient price decrease for pyramid ({price_change_pct:.2f}% < 2.0%)"
                
            # Calculate whether price has moved enough to warrant a new pyramid level
            next_target_price = previous_level_price * 0.98  # 2% lower than previous entry
            
            # Only pyramid if price is at or below the target price
            if current_price > next_target_price:
                return False, 0, f"Current price ({current_price:.6f}) above target ({next_target_price:.6f}) for next pyramid level"
                
            # We've passed the checks, proceed with pyramid entry
            reason = f"Short Pyramid Level {pyramid_count + 1}: Price {current_price:.6f} <= Target {next_target_price:.6f} " 
            reason += f"(Price decrease: {price_change_pct:.2f}% >= Required: 2.00%)"
                
        elif position_type == 'long':
            # For LONG, we need price to move up by at least 2% from previous level
            price_change_pct = ((current_price - previous_level_price) / previous_level_price) * 100
            # Check if price has moved enough (minimum 2% price increase)
            if price_change_pct < 2.0:
                return False, 0, f"Insufficient price increase for pyramid ({price_change_pct:.2f}% < 2.0%)"
                
            # Calculate whether price has moved enough to warrant a new pyramid level
            next_target_price = previous_level_price * 1.02  # 2% higher than previous entry
            
            # Only pyramid if price is at or above the target price
            if current_price < next_target_price:
                return False, 0, f"Current price ({current_price:.6f}) below target ({next_target_price:.6f}) for next pyramid level"
                
            # We've passed the checks, proceed with pyramid entry
            reason = f"Long Pyramid Level {pyramid_count + 1}: Price {current_price:.6f} >= Target {next_target_price:.6f} " 
            reason += f"(Price increase: {price_change_pct:.2f}% >= Required: 2.00%)"
        else:
            return False, 0, f"Unknown position type '{position_type}'"

        # Always use fixed AMOUNT_INCREMENT for each pyramid
        size_to_add = AMOUNT_INCREMENT
        
        # Ensure we don't exceed max size
        capped_add_size = max(0.0, min(size_to_add, MAX_AMOUNT_USD - current_total_size))
        
        return True, capped_add_size, reason

    except Exception as e:
        logger.error(f"Error checking pyramid opportunity for {symbol}: {e}", exc_info=True)
        return False, 0, f"Error: {str(e)}"

def check_for_pyramid_entries(exchange):
    """
    Check for pyramid entry opportunities in open positions.
    Uses exchange data as the source of truth.
    """
    try:
        # Get open positions directly from exchange
        open_positions = fetch_open_positions(exchange)

        if not open_positions:
            return

        for symbol in open_positions.keys():
            try:
                # Check if we can pyramid this position
                should_pyramid, size_to_add, reason = check_for_pyramid_opportunity(exchange, symbol)

                if not should_pyramid:
                    continue # Skip to next symbol

                # Get position details directly from exchange data
                position_info = open_positions[symbol]
                position_type = position_info['position_type']
                leverage = position_info['leverage']
                current_price = position_info['current_price']
                
                # Determine entry side ('buy' for long, 'sell' for short)
                entry_side = 'buy' if position_type == 'long' else 'sell'

                # Calculate quantity based on size_to_add
                quantity = get_order_quantity(exchange, symbol, current_price, leverage, size_to_add)

                if quantity <= 0:
                    logger.warning(f"Invalid quantity ({quantity}) calculated for {symbol} pyramid entry with add size {size_to_add:.2f} USD.")
                    continue

                # Log pyramid details before placing order
                logger.info(f"ADDING PYRAMID POSITION for {symbol} {position_type.upper()}: {reason}")
                logger.info(f"Current position size: ${position_info['position_size_usd']:.2f}, Adding: ${size_to_add:.2f} (Qty: {quantity:.8f})")

                # Place the pyramid order (adds to the existing position)
                pyramid_order = place_order(
                    exchange,
                    symbol,
                    entry_side,
                    quantity,
                    current_price,
                    leverage=leverage
                )

                if pyramid_order:
                    # Use executed price if available, otherwise market price estimate
                    executed_price = pyramid_order.get('average', current_price) if pyramid_order.get('average') else current_price
                    logger.info(f"Successfully submitted pyramid add order for {symbol} at ~{executed_price:.6f}")

                    # Update pyramid tracking count and entry details
                    with pyramid_lock:
                        if symbol not in pyramid_details:
                            pyramid_details[symbol] = {'count': 0, 'entries': []}

                        # Add the new entry info
                        pyramid_details[symbol]['count'] += 1
                        pyramid_details[symbol]['entries'].append({
                            'price': executed_price,
                            'added_size_usd': size_to_add,
                            'quantity': quantity,
                            'time': time.time()
                        })

                else:
                    logger.error(f"Failed to place pyramid add order for {symbol}")

            except Exception as e:
                logger.exception(f"Error checking/placing pyramid entry for {symbol}: {e}")
                continue

    except Exception as e:
        logger.exception(f"Error in check_for_pyramid_entries main loop: {e}")

def check_for_position_exits(exchange):
    """
    Check for position exits using hybrid approach combining KAMA trend and price action.
    Protects against large adverse price movements within candles.
    """
    try:
        # Get open positions directly from exchange
        open_positions = fetch_open_positions(exchange)
        
        if not open_positions:
            return
            
        logger.info(f"Checking {len(open_positions)} open positions for exit conditions")
        
        for symbol in open_positions.keys():
            try:
                # Get position details from exchange data
                position_info = open_positions[symbol]
                position_type = position_info['position_type']
                entry_price = position_info['entry_price']
                avg_entry_price = position_info.get('avg_entry_price', entry_price)
                current_price = position_info['current_price']
                current_profit_pct = position_info['current_profit_pct']
                max_profit_pct = position_info['max_profit_pct']
                pyramid_count = position_info['pyramid_count']
                pyramid_entries = position_info['pyramid_entries']
                quantity = position_info['quantity']
                stop_loss = position_info['stop_loss']
                leverage = position_info['leverage']
                entry_time = position_info['entry_time']
                
                # ENFORCE MINIMUM HOLD PERIOD (5 minutes)
                current_time = time.time()
                hold_time_minutes = (current_time - entry_time) / 60
                
                # Skip exit check if minimum hold time not met
                if hold_time_minutes < 5:
                    remaining_minutes = int(5 - hold_time_minutes)
                    logger.info(f"Skipping exit check for {symbol}: Position held for {hold_time_minutes:.1f} minutes, " 
                               f"minimum required: 5 minutes (wait {remaining_minutes} more minutes)")
                    continue
                
                # Get inverted side for exit orders
                inverted_side = 'buy' if position_type == 'short' else 'sell'
                
                # REGULAR EXIT LOGIC - Use signal data properly
                try:
                    # Get indicator and generate signal ONCE
                    indicator = get_momentum_indicator(symbol, exchange)
                    if not indicator:
                        continue
                        
                    df = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                    if df is None or len(df) < 100:
                        continue
                        
                    indicator.update_price_data(df)
                    
                    # IMPORTANT FIX: Generate signal ONCE with current position info
                    signal = indicator.generate_signal(
                        current_position=position_type, 
                        entry_price=entry_price, 
                        stop_loss=stop_loss
                    )
                    
                    # Update max profit tracking
                    with max_profit_lock:
                        if symbol not in max_profit_tracking or current_profit_pct > max_profit_tracking[symbol]:
                            max_profit_tracking[symbol] = current_profit_pct
                            max_profit_pct = current_profit_pct
                    
                    # Calculate profit retracement
                    profit_retracement = max_profit_pct - current_profit_pct
                    retracement_ratio = profit_retracement / max_profit_pct if max_profit_pct > 0 else 0
                    
                    # Get reversal info directly from signal
                    is_reversal = signal.get('is_reversal', False)
                    reversal_strength = signal.get('reversal_strength', 0)
                    reversal_reason = signal.get('reversal_reason', "")
                    
                    # FIX: Use exit signal data directly instead of recalculating
                    exit_triggered = signal.get('exit_triggered', False)
                    exit_reason = signal.get('reason', "")
                    should_reverse = signal.get('execute_reversal', False)
                    
                    # HYBRID APPROACH: Combine KAMA trend with price action safeguards
                    kama_value = None
                    is_open_above_kama = False
                    is_high_above_kama = False
                    is_volatile_candle = False
                    
                    # Get KAMA value
                    kama_columns = [col for col in df.columns if col.startswith('kama_')]
                    if kama_columns and len(df) > 0:
                        kama_column = kama_columns[0]
                        kama_value = df[kama_column].iloc[-1]
                        
                        if kama_value is not None and len(df) > 0:
                            # Get current candle data
                            current_open = df['open'].iloc[-1]
                            current_high = df['high'].iloc[-1]
                            current_low = df['low'].iloc[-1]
                            current_close = df['close'].iloc[-1]
                            
                            # Calculate percentage differences
                            exit_gap_percent = ((current_open - kama_value) / kama_value) * 100
                            high_gap_percent = ((current_high - kama_value) / kama_value) * 100
                            
                            # Calculate candle volatility
                            candle_range = current_high - current_low
                            volatility_percent = (candle_range / current_low) * 100
                            
                            # Check conditions
                            is_open_above_kama = exit_gap_percent >= 1.0  # Original condition
                            is_high_above_kama = high_gap_percent >= 2.5  # New: high is significantly above KAMA
                            is_volatile_candle = volatility_percent >= 3.0  # New: candle shows high volatility
                            
                            # For short positions, implement hybrid exit strategy
                            if position_type == 'short':
                                # Exit if open is above KAMA (original condition)
                                if is_open_above_kama:
                                    exit_triggered = True
                                    exit_reason = f"SHORT exit: Open price ({current_open:.6f}) is {exit_gap_percent:.2f}% above KAMA ({kama_value:.6f})"
                                
                                # OR exit if high is significantly above KAMA even if open wasn't
                                elif is_high_above_kama:
                                    exit_triggered = True
                                    exit_reason = f"SHORT exit: High price ({current_high:.6f}) reached {high_gap_percent:.2f}% above KAMA ({kama_value:.6f})"
                                
                                # OR exit if candle shows extreme volatility (potential large adverse move)
                                elif is_volatile_candle and current_close > current_open:
                                    exit_triggered = True 
                                    exit_reason = f"SHORT exit: Volatile candle detected ({volatility_percent:.2f}% range) with bullish close"
                                
                                # Log detailed analysis even if not exiting
                                if not exit_triggered and (is_high_above_kama or is_volatile_candle):
                                    logger.info(f"CAUTION for {symbol}: High nearly breached KAMA ({high_gap_percent:.2f}%) " 
                                              f"with {volatility_percent:.2f}% candle volatility")
                            
                            # For long positions, implement mirror hybrid exit strategy  
                            elif position_type == 'long':
                                # Calculate downside gaps
                                low_gap_percent = ((kama_value - current_low) / kama_value) * 100
                                open_below_gap = ((kama_value - current_open) / kama_value) * 100
                                
                                # Exit if open is below KAMA by threshold
                                is_open_below_kama = open_below_gap >= 1.0
                                is_low_below_kama = low_gap_percent >= 2.5
                                
                                if is_open_below_kama:
                                    exit_triggered = True
                                    exit_reason = f"LONG exit: Open price ({current_open:.6f}) is {open_below_gap:.2f}% below KAMA ({kama_value:.6f})"
                                
                                # OR exit if low is significantly below KAMA
                                elif is_low_below_kama:
                                    exit_triggered = True
                                    exit_reason = f"LONG exit: Low price ({current_low:.6f}) reached {low_gap_percent:.2f}% below KAMA ({kama_value:.6f})"
                                
                                # OR exit if candle shows extreme volatility with bearish close
                                elif is_volatile_candle and current_close < current_open:
                                    exit_triggered = True
                                    exit_reason = f"LONG exit: Volatile candle detected ({volatility_percent:.2f}% range) with bearish close"
                                
                                # Log detailed analysis even if not exiting
                                if not exit_triggered and (is_low_below_kama or is_volatile_candle):
                                    logger.info(f"CAUTION for {symbol}: Low nearly breached KAMA ({low_gap_percent:.2f}%) " 
                                              f"with {volatility_percent:.2f}% candle volatility")
                    
                    # If signal doesn't contain exit info, check additional conditions
                    if not exit_triggered:
                        # Check profit retracement
                        if max_profit_pct > 20:
                            retracement_threshold = 0.4 - (0.05 * pyramid_count)
                            retracement_threshold = max(0.25, retracement_threshold)
                            
                            if retracement_ratio > retracement_threshold:
                                exit_triggered = True
                                exit_reason = f"Profit retracement: {profit_retracement:.1f}% ({retracement_ratio:.2f}) "
                                exit_reason += f"from max {max_profit_pct:.1f}% [Threshold: {retracement_threshold:.2f}]"
                                should_reverse = is_reversal and reversal_strength > 50
                        
                        # Check strong reversal signal
                        elif is_reversal and reversal_strength >= 70:
                            exit_triggered = True
                            exit_reason = f"Strong reversal signal detected ({reversal_strength}): {reversal_reason}"
                            should_reverse = True   
                    
                    # Check fixed stop loss (6% price change)
                    if current_price >= stop_loss and position_type == 'short':
                        exit_triggered = True
                        exit_reason = f"Price {current_price} hit stop loss at {stop_loss} (6% above entry)"
                    elif current_price <= stop_loss and position_type == 'long':
                        exit_triggered = True
                        exit_reason = f"Price {current_price} hit stop loss at {stop_loss} (6% below entry)"
                    
                    # Process the exit if needed
                    if exit_triggered:
                        logger.info(f"EXIT SIGNAL for {symbol} {position_type.upper()}: {exit_reason}")
                        
                        # Close position
                        exit_order = place_order(
                            exchange, 
                            symbol, 
                            inverted_side, 
                            quantity, 
                            current_price, 
                            reduceOnly=True
                        )
                        
                        if exit_order:
                            logger.info(f"Successfully exited {symbol} {position_type.upper()} position. "
                                      f"Profit: {current_profit_pct:.2f}% [with {leverage}x leverage]")
                            
                            # Record trade result
                            record_trade_result(
                                symbol,
                                avg_entry_price,
                                current_price,
                                position_type,
                                position_info['entry_time'],
                                leverage=leverage
                            )
                            
                            # Handle reversal if applicable
                            if should_reverse and ENABLE_AUTO_REVERSALS:
                                logger.info(f"Attempting position reversal for {symbol} based on {exit_reason}")
                                handle_position_reversal(
                                    exchange,
                                    symbol,
                                    reversal_strength,
                                    position_type,
                                    current_profit_pct
                                )
                            else:
                                # Clean up tracking
                                with pyramid_lock:
                                    if symbol in pyramid_details:
                                        del pyramid_details[symbol]
                                        
                                with max_profit_lock:
                                    if symbol in max_profit_tracking:
                                        del max_profit_tracking[symbol]
                        else:
                            logger.error(f"Failed to exit position for {symbol}")
                            
                except Exception as e:
                    logger.exception(f"Error in regular exit checks for {symbol}: {e}")
            
            except Exception as e:
                logger.exception(f"Error checking exit for {symbol}: {e}")
                continue
    
    except Exception as e:
        logger.exception(f"Error in check_for_position_exits: {e}")
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

def calculate_position_size(symbol, base_size=BASE_AMOUNT_USD, max_size=MAX_AMOUNT_USD, increment=AMOUNT_INCREMENT):
    """
    Calculate position size based on recent performance
    """
    position_size = base_size
    
    with performance_lock:
        if symbol in trade_performance and trade_performance[symbol]:
            # Get the last 5 trades (or less if fewer exist)
            recent_trades = trade_performance[symbol][-5:]
            
            # Count profitable trades
            profitable_trades = sum(1 for trade in recent_trades if trade.get('profit_pct', 0) > 0)
            
            # Adjust position size based on number of profitable trades
            position_size = min(base_size + (increment * profitable_trades), max_size)
            
            # Additional adjustment based on last trade's performance
            if recent_trades and 'profit_pct' in recent_trades[-1]:
                last_profit = recent_trades[-1]['profit_pct']
                
                # Increase size after strong win
                if last_profit > 15:
                    position_size = min(position_size + increment, max_size)
                    logger.info(f"Increasing position size for {symbol} due to strong last trade: {last_profit:.1f}%")
                # Decrease size after loss
                elif last_profit < 0 and profitable_trades < 2:
                    position_size = max(base_size, position_size - increment)
                    logger.info(f"Decreasing position size for {symbol} due to recent loss: {last_profit:.1f}%")
    
    logger.info(f"Calculated position size for {symbol}: ${position_size} based on performance")
    return position_size

def record_trade_result(symbol, entry_price, exit_price, position_type, entry_time, exit_time=None, leverage=10):
    """
    Record the results of a completed trade for performance tracking.
    Entry price can be the average entry price for pyramided positions.
    """
    if exit_time is None:
        exit_time = time.time()
        
    # Calculate profit WITH LEVERAGE ADJUSTMENT
    profit_pct = calculate_position_profit(entry_price, exit_price, position_type, leverage)
    
    # Record trade details
    trade_result = {
        'symbol': symbol,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'position_type': position_type,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'duration_minutes': (exit_time - entry_time) / 60,
        'profit_pct': profit_pct,
        'is_profitable': profit_pct > 0,
        'leverage': leverage
    }
    
    # Save to performance history
    with performance_lock:
        if symbol not in trade_performance:
            trade_performance[symbol] = []
        trade_performance[symbol].append(trade_result)
    
    logger.info(f"Recorded trade result for {symbol}: {profit_pct:.2f}% profit over {trade_result['duration_minutes']:.1f} minutes [with {leverage}x leverage]")
    return trade_result

def close_partial_position(exchange, symbol, side, quantity, current_price, reduceOnly=True):
    """
    Close a portion of an open position.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        side: Order side ('buy' or 'sell')
        quantity: Position quantity to close
        current_price: Current market price (for logging)
        reduceOnly: Whether to use reduceOnly flag
        
    Returns:
        order object or None on failure
    """
    try:
        logger.info(f"Closing partial position for {symbol}: {side.upper()} {quantity:.8f} @ ~{current_price:.6f}")
        
        # Place the order
        order = place_order(
            exchange,
            symbol,
            side,
            quantity,
            current_price,
            reduceOnly=reduceOnly
        )
        
        if order:
            logger.info(f"Partial position close successful for {symbol}, order ID: {order.get('id', 'N/A')}")
            return order
        else:
            logger.error(f"Failed to close partial position for {symbol}")
            return None
            
    except Exception as e:
        logger.exception(f"Error closing partial position for {symbol}: {e}")
        return None

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

def fetch_active_symbols(exchange, sort_type='gainers', top_count=50):
    """
    Fetch active trading symbols sorted by price change in last 24hrs using 5min rolling candles.
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
        
        # logger.info(f"Calculating 24hr rolling price changes for {len(pre_filtered_symbols)} markets")
        
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

def get_order_quantity(exchange, market_id, price, leverage, position_size=None):
    """Calculate order quantity based on position size and leverage."""
    try:
        # Use position_size if provided, otherwise use BASE_AMOUNT_USD
        amount_usd = position_size if position_size is not None else BASE_AMOUNT_USD
        
        market = exchange.markets[market_id]
        min_notional = market['limits']['cost']['min']
        min_quantity = market['limits']['amount']['min']
        quantity = (amount_usd * leverage) / price
        notional = quantity * price
        
        # Ensure minimum notional value
        if notional < min_notional:
            quantity = min_notional / price
        
        # Log calculation details
        logger.info(f"Calculated order quantity for {market_id}: {quantity:.8f} (${amount_usd} × {leverage} leverage)")
        
        return max(quantity, min_quantity)
    except Exception as e:
        logger.exception(f"Error calculating quantity for {market_id}: {e}")
        return 0.0

def place_order(exchange, market_id, side, quantity, price, leverage=None, reduceOnly=False):
    """
    Simplified order placement function that places a market order.
    Ensures isolated margin mode is set before placing the order.
    Properly handles the reduceOnly parameter with fallback for errors.
    """
    params = {}
    
    # Add reduceOnly parameter if specified
    if reduceOnly:
        params['reduceOnly'] = True

    try:
        # First, ensure the margin mode is set to "Isolated"
        try:
            # Check current margin mode
            position_info = exchange.fetch_positions([market_id])
            if position_info and len(position_info) > 0:
                current_margin_mode = position_info[0].get('marginMode')
                
                # If not in isolated mode, set it to isolated
                if current_margin_mode != 'isolated':
                    exchange.set_margin_mode('isolated', market_id)
                    # Small delay to ensure the change takes effect
                    time.sleep(0.5)
            else:
                # If no position info available, set to isolated mode anyway
                exchange.set_margin_mode('isolated', market_id)
                time.sleep(0.5)
                
        except Exception as e:
            logger.warning(f"Error checking/setting margin mode for {market_id}: {e}")
            # Continue with order placement as some exchanges may set isolated mode automatically

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

def place_tp_only_order(exchange, market_id, entry_side, quantity, leverage=10, retries=2):
    """
    Place a take-profit order with a fixed 20% price change profit target.
    FIXED to ensure exactly 20% price change (not 2%).
    """
    try:
        # Get position information from exchange
        open_positions = fetch_open_positions(exchange)
        
        if market_id in open_positions:
            position_info = open_positions[market_id]
            entry_price = position_info['entry_price']
        else:
            # Fallback if position not found
            current_price = exchange.fetch_ticker(market_id)['last']
            entry_price = current_price
            
        logger.info(f"Setting TP for {market_id} with entry price: {entry_price}")
        
        # FIXED: Calculate target price with EXACTLY 20% price change
        if entry_side == 'buy':  # LONG
            # For longs: Entry + 20% price change
            target_price = entry_price * 1.20  # Fixed 20% price increase - VERIFIED
        else:  # SHORT
            # For shorts: Entry - 20% price change
            target_price = entry_price * 0.80  # Fixed 20% price decrease - VERIFIED
        
        # Ensure formatting is correct - often precision issues can cause problems
        formatted_entry = float(f"{entry_price:.6f}")
        formatted_target = float(f"{target_price:.6f}")
        
        # Double-check the calculation to ensure it's really 20%
        if entry_side == 'buy':
            actual_pct = ((formatted_target - formatted_entry) / formatted_entry) * 100
        else:
            actual_pct = ((formatted_entry - formatted_target) / formatted_entry) * 100
            
        logger.info(f"TP target for {market_id}: {formatted_target} ({actual_pct:.2f}% price change from entry {formatted_entry})")
        
        # Sanity check - if the calculation is way off, log a warning
        if abs(actual_pct - 20.0) > 0.5:  # Should be very close to 20%
            logger.warning(f"TP calculation error! Target {actual_pct:.2f}% is not 20% from entry")
    
    except Exception as e:
        logger.warning(f"Error calculating TP price for {market_id}: {e}")
        return False

    # For TP orders, we want the opposite side of our entry
    inverted_side = 'sell' if entry_side == 'buy' else 'buy'

    try:
        # Place Take Profit order
        tp_params = {
            'stopPrice': exchange.price_to_precision(market_id, target_price),
            'reduceOnly': True,
            'timeInForce': 'GTE_GTC',
        }
        
        logger.info(f"Placing TP Order (EXACTLY 20% price change) for {market_id}: {inverted_side.upper()} "
                   f"Qty:{quantity:.8f} @ Stop:{tp_params['stopPrice']}")
        
        tp_order = exchange.create_order(
            market_id, 'TAKE_PROFIT_MARKET', inverted_side, quantity, None, params=tp_params
        )
        
        logger.info(f"TP order {tp_order.get('id', 'N/A')} placed for {market_id} at {tp_params['stopPrice']} (20% price change target)")
                
        return True

    except Exception as e:
        # Handle retries if configured
        if retries > 0:
            logger.warning(f"Error placing TP order for {market_id}, retrying ({retries} attempts left): {e}")
            time.sleep(1)  # Wait before retry
            return place_tp_only_order(exchange, market_id, entry_side, quantity, leverage, retries-1)
        else:
            logger.error(f"Failed to place TP order for {market_id} after all retry attempts: {e}")
            return False

def handle_position_reversal(exchange, symbol, reversal_strength, current_position_type, profit_pct=0):
    """
    Handle position reversal using exchange data instead of position_details.
    """
    try:
        # Check if reversal meets minimum criteria
        if reversal_strength < REVERSAL_CONFIG['min_reversal_strength']:
            logger.info(f"Reversal signal for {symbol} too weak ({reversal_strength}) to trigger automatic reversal")
            return False
            
        # Check if we have minimum profit for reversal
        if profit_pct < REVERSAL_CONFIG['min_profit_for_reversal']:
            logger.info(f"Insufficient profit ({profit_pct:.1f}%) for {symbol} reversal. Minimum: {REVERSAL_CONFIG['min_profit_for_reversal']}%")
            return False
        
        # Get position details from exchange
        open_positions = fetch_open_positions(exchange)
        if symbol not in open_positions:
            logger.warning(f"Cannot reverse position for {symbol}: No open position found")
            return False
            
        position_info = open_positions[symbol]
        entry_price = position_info['entry_price']
        current_size = position_info['position_size_usd']
        current_price = position_info['current_price']
        leverage = position_info['leverage']
        
        # Determine new position type and entry side
        new_position_type = 'long' if current_position_type == 'short' else 'short'
        new_entry_side = 'buy' if new_position_type == 'long' else 'sell'
        
        # Calculate reversal position size
        reversal_size = min(current_size * REVERSAL_CONFIG['reversal_size_multiplier'], MAX_AMOUNT_USD)
        
        # FIX: Generate a proper signal for the new position direction to get appropriate SL
        indicator = get_momentum_indicator(symbol, exchange)
        if not indicator:
            return False
            
        df = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
        if df is None or len(df) < 100:
            return False
            
        indicator.update_price_data(df)
        
        # Generate signal for the new position direction
        signal = indicator.generate_signal()
        
        # Get stop loss from signal if available for the intended direction
        if new_position_type == 'long' and signal['signal'] == 'buy':
            stop_loss = signal['stop_loss']
        elif new_position_type == 'short' and signal['signal'] == 'sell':
            stop_loss = signal['stop_loss']
        else:
            # Fallback if signal doesn't match intended direction
            stop_loss = current_price * (1 - 0.06/leverage if new_position_type == 'long' else 1 + 0.06/leverage)
            
        # Calculate quantity
        quantity = get_order_quantity(exchange, symbol, current_price, leverage, reversal_size)
        if quantity <= 0:
            logger.warning(f"Invalid quantity calculated for {symbol} reversal")
            return False
            
        # Log reversal details
        logger.info(f"EXECUTING REVERSAL for {symbol}: {current_position_type.upper()} → {new_position_type.upper()} "
                   f"with {reversal_strength} strength at {current_price}. "
                   f"Size: ${current_size} → ${reversal_size}, Leverage: {leverage}x")
        
        # Record completion of previous trade
        record_trade_result(
            symbol, 
            entry_price, 
            current_price, 
            current_position_type, 
            position_info['entry_time'],
            leverage=leverage
        )
        
        # Place order for new position
        order = place_order(exchange, symbol, new_entry_side, quantity, current_price, leverage)
        
        if not order:
            logger.error(f"Failed to execute reversal order for {symbol}")
            return False
            
        # Calculate stop loss for the new position
        indicator = get_momentum_indicator(symbol, exchange)
        if indicator:
            df = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
            if df is not None and len(df) > 0:
                indicator.update_price_data(df)
                stop_loss = indicator.determine_stop_loss(df, position_type=new_position_type, entry_price=current_price)
            else:
                # Fallback SL calculation
                stop_loss = current_price * (1 - 0.06/leverage if new_position_type == 'long' else 1 + 0.06/leverage)
        else:
            # Fallback SL calculation
            stop_loss = current_price * (1 - 0.06/leverage if new_position_type == 'long' else 1 + 0.06/leverage)
        
        # Reset pyramid tracking for this symbol
        with pyramid_lock:
            pyramid_details[symbol] = {'count': 0, 'entries': []}
            
        # Reset max profit tracking
        with max_profit_lock:
            max_profit_tracking[symbol] = 0
            
        # Place TP order for new position
        place_tp_only_order(
            exchange, 
            symbol, 
            new_entry_side, 
            quantity, 
            leverage
        )
        
        logger.info(f"Successfully reversed position for {symbol} to {new_position_type.upper()} with ${reversal_size}")
        return True
        
    except Exception as e:
        logger.exception(f"Error handling position reversal for {symbol}: {e}")
        return False

def sync_positions_with_exchange(exchange):
    """
    Synchronize minimal tracking data (pyramid details and max profit) with the exchange.
    Relying on exchange as the source of truth.
    """
    try:
        # Get current positions from exchange
        exchange_positions = fetch_open_positions(exchange)
        exchange_symbols = set(exchange_positions.keys())
        
        # Clean up tracking for closed positions
        with pyramid_lock:
            tracked_symbols = list(pyramid_details.keys())
            for symbol in tracked_symbols:
                if symbol not in exchange_symbols:
                    # Position closed - clean up tracking
                    del pyramid_details[symbol]
        
        with max_profit_lock:
            tracked_symbols = list(max_profit_tracking.keys())
            for symbol in tracked_symbols:
                if symbol not in exchange_symbols:
                    # Position closed - clean up tracking
                    del max_profit_tracking[symbol]
        
    except Exception as e:
        logger.exception(f"Error in sync_positions_with_exchange: {e}")

def momentum_trading_loop(exchange):
    """
    Enhanced momentum trading loop relying on exchange as the source of truth.
    """
    while True:
        try:
            # Sync positions with exchange
            sync_positions_with_exchange(exchange)
            
            # Check for position exits including reversals
            check_for_position_exits(exchange)
            
            # Check for pyramid opportunities
            check_for_pyramid_entries(exchange)
            
            # Check current positions after the above operations
            open_positions = fetch_open_positions(exchange)
            
            # Check if we can look for new entries
            if len(open_positions) >= MAX_OPEN_TRADES:
                logger.info(f"At maximum number of open trades ({len(open_positions)}/{MAX_OPEN_TRADES}). Skipping new entries.")
                time.sleep(SLEEP_INTERVAL * 3)
                continue  # Skip to next loop iteration, don't look for new entries
            
            # Find coins with significant price movements
            momentum_candidates = find_momentum_candidates(exchange, sort_type='both', limit=20)
            
            # Extract short candidates
            if isinstance(momentum_candidates, tuple):
                # Handle the case where find_momentum_candidates returns a tuple
                short_candidates, stored_signals = momentum_candidates
                long_candidates = []  # No long candidates in this format
            else:
                # Handle the case where it returns a dictionary 
                short_candidates = momentum_candidates['short']['sorted_symbols']
                stored_signals = momentum_candidates['short']['signals']
                long_candidates = []  # No specific long candidates yet
            
            logger.info(f"Checking {len(short_candidates)} momentum candidates for entry signals...")
            
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
                        
                        # Calculate position size based on performance history
                        position_size = calculate_position_size(symbol)
                        
                        leverage = get_leverage_for_market(exchange, symbol)
                        if not leverage:
                            logger.error(f"Failed to set leverage for {symbol}. Skipping.")
                            continue
                            
                        quantity = get_order_quantity(exchange, symbol, current_price, leverage, position_size)
                        
                        if quantity <= 0:
                            logger.warning(f"Invalid quantity calculated for {symbol}. Skipping.")
                            continue
                            
                        # Initialize tracking data for max profit
                        with max_profit_lock:
                            max_profit_tracking[symbol] = 0
                            
                        # Initialize pyramid tracking
                        with pyramid_lock:
                            pyramid_details[symbol] = {'count': 0, 'entries': []}
                            
                        logger.info(f"Placing momentum short for {symbol} at {current_price} "
                                 f"with position size ${position_size} and {leverage}x leverage "
                                 f"(crossunder was {crossunder_age} candles / {minutes_ago} minutes ago)")
                        
                        order = place_order(
                            exchange, symbol, 'sell', quantity, current_price, leverage=leverage
                        )
                        
                        if order:
                            logger.info(f"Opened MOMENTUM SHORT position for {symbol}")
                            
                            # Record entry time for cooldown checks
                            with cooldown_lock:
                                position_entry_times[symbol] = time.time()
                            
                            # Place only TP order (not SL since we use dynamic management)
                            place_tp_only_order(
                                exchange, 
                                symbol, 
                                'sell',  # Entry side was sell for short
                                quantity, 
                                leverage
                            )
                            
                            entry_placed = True
                            
                            open_positions = fetch_open_positions(exchange)
                            if len(open_positions) >= MAX_OPEN_TRADES:
                                logger.info(f"Reached maximum number of open trades ({len(open_positions)}/{MAX_OPEN_TRADES}). Stopping entry processing.")
                                break
                            
                            break
                        else:
                            logger.error(f"Failed to open momentum short for {symbol}")
                            # Clean up tracking data if order failed
                            with pyramid_lock:
                                if symbol in pyramid_details:
                                    del pyramid_details[symbol]
                                    
                            with max_profit_lock:
                                if symbol in max_profit_tracking:
                                    del max_profit_tracking[symbol]
                
                except Exception as e:
                    logger.exception(f"Error processing momentum short for {symbol}: {e}")
            
            # Process long candidates if we still have room for trades
            if not entry_placed and len(open_positions) < MAX_OPEN_TRADES:
                # For now, we'll check for long signals in the stored signals
                long_signals = {s: sig for s, sig in stored_signals.items() 
                               if sig['signal'] == 'buy' and s not in open_positions}
                
                if long_signals:
                    logger.info(f"Found {len(long_signals)} potential long entries to process")
                    
                    for symbol, signal in long_signals.items():
                        try:
                            current_price = signal['price']
                            position_size = calculate_position_size(symbol)
                            
                            leverage = get_leverage_for_market(exchange, symbol)
                            if not leverage:
                                continue
                                
                            quantity = get_order_quantity(exchange, symbol, current_price, leverage, position_size)
                            if quantity <= 0:
                                continue
                                
                            stop_loss = signal['stop_loss']
                            
                            logger.info(f"MOMENTUM LONG SIGNAL: {symbol} - {signal['reason']}")
                            
                            # Initialize tracking for max profit
                            with max_profit_lock:
                                max_profit_tracking[symbol] = 0
                                
                            # Initialize pyramid tracking
                            with pyramid_lock:
                                pyramid_details[symbol] = {'count': 0, 'entries': []}
                                
                            order = place_order(
                                exchange, symbol, 'buy', quantity, current_price, leverage=leverage
                            )
                            
                            if order:
                                logger.info(f"Opened MOMENTUM LONG position for {symbol}")
                                
                                # Record entry time for cooldown checks
                                with cooldown_lock:
                                    position_entry_times[symbol] = time.time()
                                
                                # Place only TP order for this position
                                place_tp_only_order(
                                    exchange, 
                                    symbol, 
                                    'buy',  # Entry side was buy for long
                                    quantity, 
                                    leverage
                                )
                                
                                entry_placed = True
                                break
                            else:
                                logger.error(f"Failed to open momentum long for {symbol}")
                                # Clean up tracking if order failed
                                with pyramid_lock:
                                    if symbol in pyramid_details:
                                        del pyramid_details[symbol]
                                
                                with max_profit_lock:
                                    if symbol in max_profit_tracking:
                                        del max_profit_tracking[symbol]
                                        
                        except Exception as e:
                            logger.exception(f"Error processing momentum long for {symbol}: {e}")
            
            if not entry_placed:
                logger.info("No suitable momentum candidates found for entry")
            
            # Sleep between checks
            time.sleep(SLEEP_INTERVAL * 3)
            
        except Exception as e:
            logger.exception(f"Error in momentum trading loop: {e}")
            time.sleep(SLEEP_INTERVAL * 5)

def main():
    """Main function to start the trading bot with enhanced features."""
    exchange = create_exchange()
    
    # Update configuration constants for minimum hold time
    global MINIMUM_HOLD_MINUTES
    MINIMUM_HOLD_MINUTES = 2  # Minimum 2 minutes hold time
    
    # Recovery mode - check for any missed exits
    sync_positions_with_exchange(exchange)
    
    # Start the momentum trading loop in a separate thread
    momentum_thread = threading.Thread(target=momentum_trading_loop, args=(exchange,), daemon=True)
    momentum_thread.start()
    logger.info("Started enhanced momentum trading thread with exchange-based state management")
    
    # Keep the main thread alive
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()