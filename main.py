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
ENABLE_HEDGE_MODE = True
MAX_OPEN_TRADES = 2  # Maximum number of open trades at any time
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
# Configuration for hybrid exit strategy
HYBRID_EXIT_CONFIG = {
    'open_kama_threshold': 1.0,      # Exit when open is 1% above/below KAMA
    'extreme_kama_threshold': 2,    # Exit when high/low is 2.5% beyond KAMA
    'volatility_threshold': 3.0,      # Exit on candles with 3%+ range
    'enable_hybrid_exits': True,      # Master switch to enable/disable hybrid exits
    'enable_volatility_exits': True,  # Enable exits based on candle volatility
    'log_near_threshold_warnings': True  # Log warnings for near-threshold conditions
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
                logger.info(f"Failed to calculate kama for {symbol} - got empty result")
                # Fallback: calculate kama using pandas
                # df_combined[f"kama_{kama_period}"] = df_combined['close'].ewm(span=kama_period, adjust=False).mean()
        
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
    Improved to properly handle hedge mode positions (LONG and SHORT on same symbol).
    Calculates fixed 6% SL and 20% TP based on price change only, regardless of leverage.
    """
    try:
        positions = exchange.fetch_positions()
        result = {}
        
        # Process positions by symbol and position side
        for pos in positions:
            # Filter only positions with non-zero amount
            position_amt = float(pos['info'].get('positionAmt', 0))
            if abs(position_amt) <= 0:
                continue
                
            symbol = pos['symbol']
            entry_price = float(pos['info'].get('entryPrice', 0))
            
            # Get position side explicitly from exchange data for hedge mode support
            # In hedge mode, positionSide will be 'LONG' or 'SHORT'
            # In one-way mode, it will typically be 'BOTH'
            position_side = pos['info'].get('positionSide', 'BOTH')
            
            # Determine position type based on amount and position side
            if position_side == 'LONG' or (position_side == 'BOTH' and position_amt > 0):
                position_type = 'long'
            elif position_side == 'SHORT' or (position_side == 'BOTH' and position_amt < 0):
                position_type = 'short'
            else:
                # Skip if we can't determine position type
                logger.warning(f"Skipping position with unknown type: {pos['info']}")
                continue
            
            # Get position information directly from exchange
            leverage = float(pos['info'].get('leverage', 10))
            
            # Handle position keys differently in hedge mode
            position_key = symbol
            if ENABLE_HEDGE_MODE and position_side != 'BOTH':
                # In hedge mode, use symbol_LONG or symbol_SHORT as key
                position_key = f"{symbol}_{position_side}"
            
            # Get position open time - try to extract or use a fallback
            entry_time = None
            try:
                # Try to get updateTime from position info (when position was last modified)
                update_time = float(pos['info'].get('updateTime', 0))
                if update_time > 0:
                    entry_time = update_time / 1000  # Convert to seconds
                
                # Or try to get time from position_entry_times if available
                if not entry_time and position_key in position_entry_times:
                    entry_time = position_entry_times[position_key]
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
            
            # Get max profit tracking
            with max_profit_lock:
                max_profit_pct_tracked = max_profit_tracking.get(position_key, 0)
                
                # Update max_profit_pct_tracked if current profit is higher
                if current_profit_pct > max_profit_pct_tracked:
                    max_profit_tracking[position_key] = current_profit_pct
                    max_profit_pct_tracked = current_profit_pct # Use the updated value immediately
            
            # Get pyramid data with position_key for hedge mode support
            with pyramid_lock:
                pyramid_count = 0
                pyramid_entries = []
                
                if position_key in pyramid_details:
                    pyramid_details_data = pyramid_details[position_key]
                    pyramid_count = pyramid_details_data.get('count', 0)
                    pyramid_entries = pyramid_details_data.get('entries', [])
                else:
                    # Initialize pyramid tracking for new positions
                    pyramid_details[position_key] = {'count': 0, 'entries': []}
                    
                # Fallback: Calculate pyramid count from position size if tracking is empty
                if pyramid_count == 0 and position_size_usd > BASE_AMOUNT_USD:
                    pyramid_count = int(round(position_size_usd / AMOUNT_INCREMENT))
                    pyramid_count = max(0, pyramid_count)
            
            # Calculate average entry price if we have pyramid entries
            avg_entry_price = entry_price
            original_entry_price = entry_price
            
            # If we have pyramid entries with valid price data, calculate weighted average
            if pyramid_entries and len(pyramid_entries) > 0:
                try:
                    total_qty = abs(position_amt)
                    sum_price_qty = entry_price * total_qty
                    
                    # Note: We're not actually recalculating average entry here
                    # because the exchange already provides it in entry_price
                    # This is just for completeness if you wanted to implement it
                    
                    # Identify original entry price (first entry)
                    if pyramid_entries[0].get('price', 0) > 0:
                        original_entry_price = pyramid_entries[0]['price']
                except Exception as e:
                    logger.warning(f"Error calculating avg entry price for {position_key}: {e}")
            
            # Combine all data into one comprehensive position info structure
            result[position_key] = {
                'symbol': symbol,
                'position_type': position_type,
                'position_side': position_side,  # Store actual exchange position side
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': abs(position_amt),
                'position_size_usd': position_size_usd,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'target': target_price,
                'entry_time': entry_time,
                'current_profit_pct': current_profit_pct,
                'max_profit_pct': max_profit_pct_tracked,
                'pyramid_count': pyramid_count,
                'pyramid_entries': pyramid_entries,
                'avg_entry_price': avg_entry_price,
                'original_entry_price': original_entry_price,
                'hedge_mode': ENABLE_HEDGE_MODE
            }
        
        return result
    except Exception as e:
        logger.exception(f"Error fetching positions: {e}")
        return {}

def check_for_pyramid_opportunity(exchange, symbol):
    """
    Check if we should add to an existing position via pyramiding.
    FIXED: Now properly handles hedge mode where positions have _LONG/_SHORT suffixes.
    """
    try:
        # Get open positions directly from exchange
        open_positions = fetch_open_positions(exchange)
        
        # FIXED: Handle hedge mode position key lookup
        position_key = symbol
        position_info = None
        
        if ENABLE_HEDGE_MODE:
            # In hedge mode, positions have _LONG or _SHORT suffixes
            long_key = f"{symbol}_LONG"
            short_key = f"{symbol}_SHORT"
            
            if long_key in open_positions:
                position_key = long_key
                position_info = open_positions[long_key]
            elif short_key in open_positions:
                position_key = short_key
                position_info = open_positions[short_key]
        else:
            # In one-way mode, use plain symbol
            if symbol in open_positions:
                position_info = open_positions[symbol]
        
        # Skip if no position found
        if position_info is None:
            return False, 0, "No active position"
        
        # Get position details
        position_type = position_info['position_type']
        entry_price = position_info['entry_price'] 
        current_total_size = position_info['position_size_usd']
        leverage = position_info['leverage']
        current_price = position_info['current_price']
        pyramid_entries = position_info['pyramid_entries']
        pyramid_count = position_info['pyramid_count']

        # Check if we've reached maximum pyramid entries
        if pyramid_count >= PYRAMID_CONFIG['max_entries']:
            return False, 0, f"Maximum pyramid count reached ({pyramid_count})"
        
        # Check if position already at max size
        if current_total_size >= MAX_AMOUNT_USD:
            return False, 0, f"Position already at maximum size (${current_total_size:.2f})"
            
        # Check if we're already very close to max size
        threshold = min(0.5, MAX_AMOUNT_USD * 0.1)
        if (MAX_AMOUNT_USD - current_total_size) <= threshold:
            return False, 0, f"Position already effectively at maximum size (${current_total_size:.2f})"
            
        # Get the previous pyramid level price or use original entry price
        if pyramid_entries and len(pyramid_entries) > 0:
            previous_level_price = pyramid_entries[-1].get('price', entry_price)
        else:
            previous_level_price = entry_price
            
        # Calculate dynamic pyramid threshold (simplified to 2%)
        pyramid_threshold = 2.0  # Fixed 2% threshold
        
        if position_type == 'short':
            # For SHORT, check if price moved down 2% from previous level
            price_change_pct = ((previous_level_price - current_price) / previous_level_price) * 100
            if price_change_pct < pyramid_threshold:
                return False, 0, f"Price decrease {price_change_pct:.2f}% < threshold {pyramid_threshold:.2f}%"
                
            reason = f"Short Pyramid Level {pyramid_count + 1}: Price decrease {price_change_pct:.2f}%"
                
        elif position_type == 'long':
            # For LONG, check if price moved up 2% from previous level
            price_change_pct = ((current_price - previous_level_price) / previous_level_price) * 100
            if price_change_pct < pyramid_threshold:
                return False, 0, f"Price increase {price_change_pct:.2f}% < threshold {pyramid_threshold:.2f}%"
                
            reason = f"Long Pyramid Level {pyramid_count + 1}: Price increase {price_change_pct:.2f}%"
        else:
            return False, 0, f"Unknown position type '{position_type}'"

        # Calculate pyramid size: Pₙ = P₀ × 0.5^n
        initial_size = BASE_AMOUNT_USD
        size_to_add = calculate_pyramid_size(initial_size, pyramid_count)
        
        # Ensure we don't exceed max size
        capped_add_size = max(0.0, min(size_to_add, MAX_AMOUNT_USD - current_total_size))
        
        # If addition too small, skip
        if capped_add_size < 0.5:
            return False, 0, f"Calculated pyramid size (${capped_add_size:.2f}) too small"
            
        reason += f", Adding: ${capped_add_size:.2f}"
        
        return True, capped_add_size, reason

    except Exception as e:
        logger.error(f"Error checking pyramid opportunity for {symbol}: {e}", exc_info=True)
        return False, 0, f"Error: {str(e)}"
def calculate_risk_optimized_tp(exchange, symbol, position_type, entry_price, stop_loss):
    """
    Calculate take-profit level ensuring minimum risk-reward ratio of 2:1.
    With additional validation and logging.
    """
    try:
        # Safety check for invalid inputs
        if entry_price <= 0 or stop_loss <= 0:
            logger.error(f"Invalid entry_price ({entry_price}) or stop_loss ({stop_loss}) for {symbol}")
            return entry_price * (1.2 if position_type == 'long' else 0.8)  # Fallback to 20%
            
        # Calculate risk
        if position_type == 'long':
            if stop_loss >= entry_price:
                logger.error(f"Stop loss ({stop_loss}) above entry price ({entry_price}) for LONG position")
                stop_loss = entry_price * 0.94  # Fallback to 6% below entry
                
            risk = entry_price - stop_loss
            # Target should be at least 2x risk above entry (framework rule)
            min_reward = risk * 2
            target_price = entry_price + min_reward
        else:  # short
            if stop_loss <= entry_price:
                logger.error(f"Stop loss ({stop_loss}) below entry price ({entry_price}) for SHORT position")
                stop_loss = entry_price * 1.06  # Fallback to 6% above entry
                
            risk = stop_loss - entry_price
            # Target should be at least 2x risk below entry (framework rule)
            min_reward = risk * 2
            target_price = entry_price - min_reward
        
        # Log base calculation
        logger.info(f"Base TP calculation for {symbol} {position_type}: Entry {entry_price:.6f}, Stop {stop_loss:.6f}, " +
                   f"Risk {risk:.6f}, Min Reward {min_reward:.6f}, Base Target {target_price:.6f}")
        
        # Calculate dynamic TP based on price change
        df = fetch_extended_historical_data(exchange, symbol, TIMEFRAME)
        if df is None or len(df) < 20:
            # If no dynamic calculation possible, use the minimum risk-reward TP
            logger.info(f"Using base risk-reward TP for {symbol}: {target_price:.6f}")
            return target_price
            
        # Calculate recent price change percentage (last 24 hours)
        candle_count = min(480, len(df) - 1)
        start_price = df['close'].iloc[-candle_count]
        end_price = df['close'].iloc[-1]
        
        # Calculate absolute percentage change
        price_change_24h_pct = abs((end_price - start_price) / start_price) * 100
        
        # Calculate volatility-based TP percentage
        tp_pct = price_change_24h_pct * 1.5
        
        # Clamp TP percentage between 5% and 30%
        tp_pct = max(5.0, min(30.0, tp_pct))
        
        # Calculate price-based target
        dynamic_target = entry_price * (1 + tp_pct/100) if position_type == 'long' else entry_price * (1 - tp_pct/100)
        
        # Log dynamic calculation
        logger.info(f"Dynamic TP calculation for {symbol}: 24h Change {price_change_24h_pct:.2f}%, " +
                   f"TP% {tp_pct:.2f}%, Dynamic Target {dynamic_target:.6f}")
        
        # Use the target that gives better reward while maintaining minimum risk-reward
        if position_type == 'long':
            # For longs, use the farther target (higher price)
            final_target = max(target_price, dynamic_target)
        else:  # short
            # For shorts, use the farther target (lower price)
            final_target = min(target_price, dynamic_target)
        
        # Calculate actual risk-reward ratio for logging
        if position_type == 'long':
            actual_reward = final_target - entry_price
            actual_ratio = actual_reward / risk if risk > 0 else 0
        else:
            actual_reward = entry_price - final_target
            actual_ratio = actual_reward / risk if risk > 0 else 0
            
        logger.info(f"Final TP calculation for {symbol}: Risk ${risk:.6f}, Reward ${actual_reward:.6f}, " 
                   f"Ratio {actual_ratio:.2f}, Target {final_target:.6f}")
        
        # Verify target is reasonable (not too far from current price)
        current_price = df['close'].iloc[-1]
        if position_type == 'long' and final_target > current_price * 1.5:
            logger.warning(f"TP target for {symbol} LONG seems excessive: {final_target:.6f} (50%+ above current price {current_price:.6f})")
        elif position_type == 'short' and final_target < current_price * 0.5:
            logger.warning(f"TP target for {symbol} SHORT seems excessive: {final_target:.6f} (50%+ below current price {current_price:.6f})")
        
        return final_target
        
    except Exception as e:
        logger.error(f"Error calculating risk-optimized TP: {e}")
        # Default to 20% profit target as fallback
        return entry_price * (1.2 if position_type == 'long' else 0.8)
    
def check_for_hedge_opportunity(exchange, symbol, position_info):
    """
    Check if we should hedge a position based on 3% retracement from low.
    Hedge size Hₙ = Current Position Exposure / 2
    """
    try:
        position_type = position_info['position_type']
        entry_price = position_info['entry_price']
        current_price = position_info['current_price']
        current_exposure = position_info['position_size_usd']
        entry_time = position_info['entry_time']
        
        # Skip if no valid entry time
        if not entry_time or entry_time <= 0:
            return False, 0, "No valid entry time"
        
        # Skip if current exposure exceeds MAX_AMOUNT_USD
        if current_exposure >= MAX_AMOUNT_USD:
            return False, 0, f"Position already at maximum size (${current_exposure:.2f})"
        
        # Get historical data since entry
        df = fetch_extended_historical_data(exchange, symbol, TIMEFRAME)
        if df is None or len(df) < 5:
            return False, 0, "Insufficient historical data"
        
        # Filter data since entry
        entry_timestamp = pd.Timestamp(entry_time, unit='s', tz='UTC')
        df_since_entry = df[df.index > entry_timestamp]
        
        if len(df_since_entry) < 3:
            return False, 0, "Insufficient data since entry"
        
        # Calculate retracement from low (for SHORT positions)
        if position_type == 'short':
            lowest_price = df_since_entry['low'].min()
            retracement_pct = ((current_price - lowest_price) / lowest_price) * 100
        else:
            # For LONG positions, calculate retracement from high
            highest_price = df_since_entry['high'].max()
            retracement_pct = ((highest_price - current_price) / highest_price) * 100
        
        # Check if retracement meets 3% threshold
        if retracement_pct < 3.0:
            return False, 0, f"Retracement ({retracement_pct:.2f}%) below 3% threshold"
        
        # Check for existing hedges
        all_positions = fetch_open_positions(exchange)
        opposite_side = 'LONG' if position_type == 'short' else 'SHORT'
        
        hedge_count = 0
        for pos_key, pos in all_positions.items():
            if pos['symbol'] == symbol and pos['position_type'] != position_type:
                hedge_count += 1
        
        # Skip if maximum hedge count reached
        if hedge_count >= 3:
            return False, 0, "Maximum hedge count reached"
        
        # Calculate hedge size: Hₙ = Current Exposure / 2
        hedge_size = current_exposure / 2
        
        # Ensure hedge size respects MAX_AMOUNT_USD
        total_exposure = sum(pos['position_size_usd'] for pos in all_positions.values() if pos['symbol'] == symbol)
        if total_exposure + hedge_size > MAX_AMOUNT_USD:
            hedge_size = max(0, MAX_AMOUNT_USD - total_exposure)
        
        # Ensure minimum viable hedge size
        if hedge_size < BASE_AMOUNT_USD / 2:
            return False, 0, f"Calculated hedge size (${hedge_size:.2f}) too small"
        
        reason = f"{position_type.capitalize()} Hedge: {retracement_pct:.2f}% retracement from {'low' if position_type == 'short' else 'high'} since entry, Hedge size ${hedge_size:.2f}"
        
        return True, hedge_size, reason
        
    except Exception as e:
        logger.error(f"Error checking hedge opportunity for {symbol}: {e}")
        return False, 0, f"Error: {str(e)}"
def calculate_dynamic_hedge_size(exchange, symbol, position_type, current_exposure):
    """Calculate hedge size based on retracement severity and speed."""
    try:
        # Get recent price data
        df = fetch_extended_historical_data(exchange, symbol, TIMEFRAME)
        if df is None or len(df) < 20:
            return BASE_AMOUNT_USD  # Default size if insufficient data
            
        # Calculate retracement percentage
        recent_candles = min(10, len(df) - 1)
        price_series = df['close'].iloc[-recent_candles:].values
        
        if position_type == 'short':
            # For shorts, we're concerned with price increases
            lowest = np.min(price_series[:-1])  # Exclude current price
            current = price_series[-1]
            retracement_pct = ((current - lowest) / lowest) * 100 if lowest > 0 else 0
        else:  # long
            # For longs, we're concerned with price decreases
            highest = np.max(price_series[:-1])  # Exclude current price
            current = price_series[-1]
            retracement_pct = ((highest - current) / highest) * 100 if highest > 0 else 0
        
        # Calculate time to retrace (in minutes)
        # Assuming 3-min candles, estimate conservatively
        time_to_retrace = recent_candles * 3  # 3 minutes per candle for 3m timeframe
        
        # Calculate hedge ratio using the formula
        # H_n = (E/2) × (1 + R/T)
        hedge_ratio = (1 + (retracement_pct / max(1, time_to_retrace)))
        hedge_size = (current_exposure / 2) * hedge_ratio
        
        # Clamp hedge size between 1x and 3x of BASE_AMOUNT_USD
        min_hedge = BASE_AMOUNT_USD
        max_hedge = BASE_AMOUNT_USD * 3
        
        return max(min_hedge, min(max_hedge, hedge_size))
        
    except Exception as e:
        logger.error(f"Error calculating dynamic hedge size: {e}")
        return BASE_AMOUNT_USD  # Default to base amount on error
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

        for position_key in open_positions.keys():
            try:
                # CRITICAL FIX: Extract the base symbol without position side suffix
                symbol = position_key
                if '_LONG' in position_key or '_SHORT' in position_key:
                    symbol = position_key.split('_')[0]
                
                # Check if we can pyramid this position
                should_pyramid, size_to_add, reason = check_for_pyramid_opportunity(exchange, symbol)

                if not should_pyramid:
                    continue # Skip to next symbol

                # Get position details directly from exchange data
                position_info = open_positions[position_key]
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
                # Pass clean symbol and explicit position side
                pyramid_order = place_order(
                    exchange,
                    symbol,  # Clean symbol without suffix
                    entry_side,
                    quantity,
                    current_price,
                    leverage=leverage,
                    positionSide=position_type.upper()  # Explicitly set position side based on position type
                )
                
                if pyramid_order:
                    # Use executed price if available, otherwise market price estimate
                    executed_price = pyramid_order.get('average', current_price) if pyramid_order.get('average') else current_price
                    logger.info(f"Successfully submitted pyramid add order for {symbol} at ~{executed_price:.6f}")

                    # Update pyramid tracking count and entry details
                    with pyramid_lock:
                        if position_key not in pyramid_details:
                            pyramid_details[position_key] = {'count': 0, 'entries': []}

                        # Add the new entry info
                        pyramid_details[position_key]['count'] += 1
                        pyramid_details[position_key]['entries'].append({
                            'price': executed_price,
                            'added_size_usd': size_to_add,
                            'quantity': quantity,
                            'time': time.time()
                        })

                else:
                    logger.error(f"Failed to place pyramid add order for {position_key}")

            except Exception as e:
                logger.exception(f"Error checking/placing pyramid entry for {position_key}: {e}")
                continue

    except Exception as e:
        logger.exception(f"Error in check_for_pyramid_entries main loop: {e}")
def calculate_adjustment_factor(exchange, symbol, position_type):
    """Calculate position size adjustment factor based on trend strength."""
    try:
        # Get price data
        df = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
        if df is None or len(df) < 20:
            return 1.0  # Default factor if insufficient data
            
        # Calculate momentum score (0-10 scale)
        recent_candles = 20
        closes = df['close'].iloc[-recent_candles:].values
        
        # Calculate price change direction in recent candles
        direction_changes = np.diff(np.sign(np.diff(closes)))
        streak_length = 0
        for i in range(len(direction_changes) - 1, -1, -1):
            if direction_changes[i] == 0:  # No direction change
                streak_length += 1
            else:
                break
                
        # Calculate momentum score based on streak and volatility
        momentum_score = min(10, streak_length / 2)
        
        # If price making new highs/lows, increase the factor
        if position_type == 'long' and closes[-1] > np.max(closes[:-1]):
            adjustment_factor = 1 + (momentum_score / 10)
        elif position_type == 'short' and closes[-1] < np.min(closes[:-1]):
            adjustment_factor = 1 + (momentum_score / 10)
        else:
            # Calculate retracement severity (0-10 scale)
            if position_type == 'long':
                highest = np.max(closes)
                retracement_pct = ((highest - closes[-1]) / highest) * 100
            else:  # short
                lowest = np.min(closes)
                retracement_pct = ((closes[-1] - lowest) / lowest) * 100
                
            reversal_severity = min(10, retracement_pct * 2)
            adjustment_factor = 1 - (reversal_severity / 10)
            
        return max(0.5, min(2.0, adjustment_factor))  # Clamp between 0.5 and 2.0
        
    except Exception as e:
        logger.error(f"Error calculating adjustment factor: {e}")
        return 1.0  # Default to neutral factor on error
def calculate_pyramid_size(initial_size, pyramid_count):
    """
    Calculate pyramid size based on the formula: Pₙ = Pₙ₋₁ × 0.5
    
    Args:
        initial_size: Initial position size (P₀)
        pyramid_count: Current pyramid count (n)
        
    Returns:
        float: Size for the next pyramid addition
    """
    return initial_size * (0.5 ** pyramid_count)

def calculate_hedge_size(current_exposure, hedge_count, remaining_exposure=None):
    """
    Calculate hedge size based on the sequence:
    H₁ = Current Exposure / 2
    H₂ = Remaining Exposure / 2
    H₃ = Full Flip
    
    Args:
        current_exposure: Current position exposure
        hedge_count: Current hedge count (1, 2, or 3)
        remaining_exposure: Exposure after previous hedges (required for hedge_count > 1)
        
    Returns:
        float: Size for the next hedge
    """
    if hedge_count == 1:
        return min(current_exposure / 2, MAX_AMOUNT_USD)
    elif hedge_count == 2:
        if remaining_exposure is None:
            raise ValueError("remaining_exposure must be provided for hedge_count > 1")
        return min(remaining_exposure / 2, MAX_AMOUNT_USD)
    elif hedge_count == 3:
        # Full flip - close remaining and open opposite direction with equal size
        if remaining_exposure is None:
            raise ValueError("remaining_exposure must be provided for hedge_count > 1")
        return remaining_exposure  # Full remaining exposure
    else:
        raise ValueError(f"Invalid hedge_count: {hedge_count}. Must be 1, 2, or 3.")
def check_risk_reward_ratio(entry_price, stop_loss, target_price, position_type, min_ratio=2.0):
    """
    Check if the risk-reward ratio meets the minimum requirement: Target Reward ≥ min_ratio × Risk
    
    Args:
        entry_price: Position entry price
        stop_loss: Stop loss price
        target_price: Target price for taking profit
        position_type: 'long' or 'short'
        min_ratio: Minimum risk-reward ratio (default: 2.0)
        
    Returns:
        tuple: (meets_requirement, actual_ratio)
    """
    if position_type == 'long':
        reward = target_price - entry_price
        risk = entry_price - stop_loss
    else:  # short
        reward = entry_price - target_price
        risk = stop_loss - entry_price
        
    # Avoid division by zero
    if risk <= 0:
        return False, 0.0
        
    actual_ratio = reward / risk
    meets_requirement = actual_ratio >= min_ratio
    
    return meets_requirement, actual_ratio
    
def check_for_position_exits(exchange):
    """
    Enhanced position management with 3% retracement reduction logic.
    FIXED: Now properly sets reduceOnly=True for position reductions.
    """
    try:
        # Get open positions directly from exchange
        open_positions = fetch_open_positions(exchange)
        
        if not open_positions:
            return
            
        for position_key, position_info in open_positions.items():
            try:
                # Get position details from exchange data
                symbol = position_info['symbol']
                position_type = position_info['position_type']
                position_side = position_info.get('position_side', 'BOTH')
                entry_price = position_info['entry_price']
                current_price = position_info['current_price']
                quantity = position_info['quantity']
                position_size_usd = position_info['position_size_usd']
                leverage = position_info['leverage']
                entry_time = position_info['entry_time']
                
                # --- 8% MAX LOSS EXIT LOGIC ---
                if position_type == 'short':
                    price_change_pct = ((current_price - entry_price) / entry_price) * 100
                    unfavorable_move = price_change_pct > 0
                else:
                    price_change_pct = ((entry_price - current_price) / entry_price) * 100
                    unfavorable_move = price_change_pct > 0
                
                MAX_PRICE_CHANGE_PCT = 8.0
                
                if unfavorable_move and abs(price_change_pct) >= MAX_PRICE_CHANGE_PCT:
                    exit_side = 'buy' if position_type == 'short' else 'sell'
                    
                    logger.info(f"MAX PRICE CHANGE EXIT: {symbol} {position_type.upper()} - " +
                              f"Price moved {price_change_pct:.2f}% against entry.")
                    
                    # FIXED: Use reduceOnly=True for closing positions
                    exit_order = place_order(
                        exchange, symbol, exit_side, quantity, current_price,
                        reduceOnly=True, positionSide=position_side if ENABLE_HEDGE_MODE else None
                    )
                    
                    if exit_order:
                        logger.info(f"Successfully closed {symbol} {position_type.upper()} position due to 8% price change.")
                        record_trade_result(symbol, entry_price, current_price, position_type, entry_time, leverage=leverage)
                        continue
                
                # --- 3% RETRACEMENT REDUCTION LOGIC ---
                if not entry_time or entry_time <= 0:
                    logger.warning(f"No valid entry time for {symbol}. Skipping retracement check.")
                    continue
                
                # Get historical data since entry
                df = fetch_extended_historical_data(exchange, symbol, TIMEFRAME)
                if df is None or len(df) < 5:
                    logger.debug(f"Insufficient historical data for {symbol} retracement check")
                    continue
                
                # Filter data to only include candles AFTER entry time
                entry_timestamp = pd.Timestamp(entry_time, unit='s', tz='UTC')
                df_since_entry = df[df.index > entry_timestamp]
                
                if len(df_since_entry) < 3:  # Need some data after entry
                    logger.debug(f"Insufficient data since entry for {symbol}")
                    continue
                
                # Calculate retracement based on position type
                reduce_position = False
                reduction_reason = ""
                
                if position_type == 'short':
                    # For SHORT: Find lowest price since entry
                    lowest_price = df_since_entry['low'].min()
                    
                    # Calculate 3% above the low
                    retracement_threshold = lowest_price * 1.03
                    
                    # Current retracement percentage
                    retracement_pct = ((current_price - lowest_price) / lowest_price) * 100
                    
                    # Check if current price is above threshold (3% retracement)
                    if current_price >= retracement_threshold and retracement_pct >= 3.0:
                        reduce_position = True
                        reduction_reason = f"Price {current_price:.6f} retraced {retracement_pct:.2f}% from low {lowest_price:.6f}"
                
                elif position_type == 'long':
                    # For LONG: Find highest price since entry
                    highest_price = df_since_entry['high'].max()
                    
                    # Calculate 3% below the high
                    retracement_threshold = highest_price * 0.97
                    
                    # Current retracement percentage
                    retracement_pct = ((highest_price - current_price) / highest_price) * 100
                    
                    # Check if current price is below threshold (3% retracement)
                    if current_price <= retracement_threshold and retracement_pct >= 3.0:
                        reduce_position = True
                        reduction_reason = f"Price {current_price:.6f} retraced {retracement_pct:.2f}% from high {highest_price:.6f}"
                
                # Execute reduction if needed
                if reduce_position and position_size_usd > BASE_AMOUNT_USD/2:
                    # Check if position size is around $1 - if so, exit completely
                    if 0.9 <= position_size_usd <= 1.1:
                        logger.info(f"CLOSING ENTIRE {position_type.upper()} {symbol}: Position size ${position_size_usd:.2f} is around $1. {reduction_reason}")
                        
                        exit_side = 'buy' if position_type == 'short' else 'sell'
                        # FIXED: Use reduceOnly=True for closing positions
                        close_order = place_order(
                            exchange, symbol, exit_side, quantity, current_price,
                            reduceOnly=True, positionSide=position_side if ENABLE_HEDGE_MODE else None
                        )
                        
                        if close_order:
                            logger.info(f"SUCCESS: Closed entire {position_type.upper()} {symbol} position (${position_size_usd:.2f})")
                            record_trade_result(symbol, entry_price, current_price, position_type, entry_time, leverage=leverage)
                            continue
                    else:
                        # Reduce by BASE_AMOUNT_USD
                        logger.info(f"REDUCING {position_type.upper()} {symbol}: {reduction_reason}")
                        
                        reduction_size_usd = BASE_AMOUNT_USD
                        reduction_qty = get_order_quantity(exchange, symbol, current_price, leverage, reduction_size_usd)
                        reduction_qty = min(reduction_qty, quantity)
                        
                        if reduction_qty > 0:
                            exit_side = 'buy' if position_type == 'short' else 'sell'
                            # FIXED: Use reduceOnly=True for reducing positions
                            reduce_order = place_order(
                                exchange, symbol, exit_side, reduction_qty, current_price,
                                reduceOnly=True, positionSide=position_side if ENABLE_HEDGE_MODE else None
                            )
                            
                            if reduce_order:
                                logger.info(f"SUCCESS: Reduced {position_type.upper()} {symbol} by {reduction_qty:.8f} (${reduction_size_usd:.2f})")
                            else:
                                logger.error(f"Failed to place reduction order for {position_type.upper()} {symbol}")
                
                # HEDGE LOGIC (keep existing)
                should_hedge, hedge_size, hedge_reason = check_for_hedge_opportunity(exchange, symbol, position_info)
                
                if should_hedge and hedge_size > 0:
                    hedge_side = 'buy' if position_type == 'short' else 'sell'
                    
                    logger.info(f"HEDGING {symbol} {position_type.upper()}: {hedge_reason}")
                    
                    hedge_qty = get_order_quantity(exchange, symbol, current_price, leverage, hedge_size)
                    
                    if hedge_qty > 0:
                        opposite_position_side = 'LONG' if position_type == 'short' else 'SHORT'
                        
                        hedge_order = place_order(
                            exchange,
                            symbol,
                            hedge_side,
                            hedge_qty,
                            current_price,
                            leverage=leverage,
                            positionSide=opposite_position_side if ENABLE_HEDGE_MODE else None
                        )
                        
                        if hedge_order:
                            logger.info(f"Successfully placed hedge for {symbol} {position_type.upper()} with ${hedge_size:.2f}")
                
            except KeyError as ke:
                logger.error(f"KeyError processing position for {position_key}: {ke}")
            except Exception as e:
                logger.exception(f"Error processing position for {position_key}: {e}")
                continue
    
    except Exception as e:
        logger.exception(f"Critical error in check_for_position_exits main loop: {e}")

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

def close_partial_position(exchange, symbol, side, quantity, current_price, reduceOnly=True, positionSide=None):
    """
    Close a portion of an open position with hedge mode support.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        side: Order side ('buy' or 'sell')
        quantity: Position quantity to close
        current_price: Current market price (for logging)
        reduceOnly: Whether to use reduceOnly flag
        positionSide: Position side to close ('LONG' or 'SHORT') - required for hedge mode
        
    Returns:
        order object or None on failure
    """
    try:
        # For hedge mode, determine position side if not provided
        if ENABLE_HEDGE_MODE and not positionSide:
            # In hedge mode, when closing:
            # - 'buy' side typically closes a 'SHORT' position
            # - 'sell' side typically closes a 'LONG' position
            positionSide = 'SHORT' if side == 'buy' else 'LONG'
        
        logger.info(f"Closing partial position for {symbol}: {side.upper()} {quantity:.8f} @ ~{current_price:.6f}" + 
                  (f" (positionSide: {positionSide})" if ENABLE_HEDGE_MODE else ""))
        
        # Place the order with position side for hedge mode
        order = place_order(
            exchange,
            symbol,
            side,
            quantity,
            current_price,
            reduceOnly=reduceOnly,
            positionSide=positionSide
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
        # if len(sorted_short_symbols) > 0:
        #     logger.info("Top momentum short candidates (20%+ gain in 24h, crossunder within 20 candles):")
        #     for i, symbol in enumerate(sorted_short_symbols[:10]):
        #         data = all_short_candidates[symbol]
        #         status_emoji = "🟢" if data['signal_generated'] else ("🔥" if data['is_recent_crossunder'] else "⬇️")
        #         logger.info(f"{i+1}. {symbol}: {data['gain_pct']:.2f}% gain, {data['drawdown_pct']:.2f}% down from high, "
        #                f"crossunder age: {data['crossunder_age']} candles, "
        #                f"{status_emoji} {'READY' if data['signal_generated'] else ('RECENT CROSSUNDER' if data['is_recent_crossunder'] else 'Potential')}")
        
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
        # FIXED: Strip position side suffix (_LONG or _SHORT) from market_id if present
        if isinstance(market_id, str) and ('_LONG' in market_id or '_SHORT' in market_id):
            # Extract the base symbol by removing the position side suffix
            base_market_id = market_id.split('_')[0]
            logger.info(f"Stripped position side suffix from market ID: {market_id} -> {base_market_id}")
            market_id = base_market_id
        
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

def place_order(exchange, market_id, side, quantity, price, leverage=None, reduceOnly=False, positionSide=None):
    """Enhanced order placement function with proper Binance hedge mode handling."""
    params = {}
    
    # Ensure we're using a clean market_id without position suffixes
    if isinstance(market_id, str):
        if '_LONG' in market_id:
            market_id = market_id.split('_LONG')[0]
        elif '_SHORT' in market_id:
            market_id = market_id.split('_SHORT')[0]
    
    # Handle hedge mode and positionSide
    if ENABLE_HEDGE_MODE:
        # In hedge mode, we need positionSide but NOT reduceOnly
        if not positionSide:
            positionSide = 'LONG' if side == 'buy' else 'SHORT'
        params['positionSide'] = positionSide
        
        # CRITICAL: Do NOT add reduceOnly in hedge mode
        # Binance automatically understands we're targeting a specific position
        # when positionSide is specified
        
    else:
        # In one-way mode, we can use reduceOnly but NOT positionSide
        if reduceOnly:
            params['reduceOnly'] = True
        # positionSide is not allowed in one-way mode

    try:
        # Skip MAX_AMOUNT_USD check for reduction orders
        if not reduceOnly:  # Only check for new positions or increasing existing ones
            # Get current positions
            open_positions = fetch_open_positions(exchange)
            
            # Calculate current exposure for this symbol
            current_exposure = 0
            position_key = market_id
            
            # In hedge mode, use position-specific key
            if ENABLE_HEDGE_MODE and positionSide:
                position_key = f"{market_id}_{positionSide}"
                
            # Check if we already have a position
            if position_key in open_positions:
                current_exposure = open_positions[position_key]['position_size_usd']
            
            # Calculate order size in USD
            ticker_data = exchange.fetch_ticker(market_id)
            order_price = ticker_data['last']
            order_notional = quantity * order_price
            
            # With leverage, the position size is notional / leverage
            if leverage:
                order_size_usd = order_notional / leverage
            else:
                # Try to get leverage from existing position
                if position_key in open_positions:
                    leverage_value = open_positions[position_key]['leverage']
                    order_size_usd = order_notional / leverage_value
                else:
                    # Default to 10x leverage if not specified
                    order_size_usd = order_notional / 10
            
            # Calculate total position size after this order
            total_size_after_order = current_exposure + order_size_usd
            
            # Check if it would exceed MAX_AMOUNT_USD
            if total_size_after_order > MAX_AMOUNT_USD:
                logger.warning(f"Order for {market_id} would exceed MAX_AMOUNT_USD (${MAX_AMOUNT_USD}). " +
                             f"Current: ${current_exposure:.2f}, Adding: ${order_size_usd:.2f}, " +
                             f"Total: ${total_size_after_order:.2f}")
                
                # Adjust the quantity to respect the maximum
                if current_exposure < MAX_AMOUNT_USD:
                    allowed_addition = MAX_AMOUNT_USD - current_exposure
                    adjusted_quantity = (allowed_addition * leverage) / order_price
                    
                    # Only proceed if the adjusted quantity is meaningful
                    if adjusted_quantity > 0:
                        logger.info(f"Adjusting order quantity from {quantity:.8f} to {adjusted_quantity:.8f} " +
                                  f"to respect MAX_AMOUNT_USD (${MAX_AMOUNT_USD})")
                        quantity = adjusted_quantity
                    else:
                        logger.error(f"Cannot place order: Position size ${current_exposure:.2f} " +
                                   f"already at or near maximum ${MAX_AMOUNT_USD:.2f}")
                        return None
                else:
                    logger.error(f"Cannot place order: Position size ${current_exposure:.2f} " +
                               f"already exceeds maximum ${MAX_AMOUNT_USD:.2f}")
                    return None
        else:
            # For reduction orders, log that we're bypassing the check
            logger.debug(f"Bypassing MAX_AMOUNT_USD check for reduction order: {market_id} {side} {quantity:.8f}")

        # Proceed with hedge mode and margin mode checks
        if ENABLE_HEDGE_MODE:
            try:
                account_info = exchange.fapiPrivateGetAccount()
                current_mode = account_info.get('dualSidePosition', False)
                if not current_mode:
                    exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
                    logger.info("Switched to hedge mode for account")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error checking/setting hedge mode: {e}")
        
        # Ensure isolated margin mode
        try:
            position_info = exchange.fetch_positions([market_id])
            if position_info and len(position_info) > 0:
                current_margin_mode = position_info[0].get('marginMode')
                if current_margin_mode != 'isolated':
                    exchange.set_margin_mode('isolated', market_id)
                    time.sleep(0.5)
            else:
                exchange.set_margin_mode('isolated', market_id)
                time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Error checking/setting margin mode for {market_id}: {e}")

        # Get current market price
        ticker_data = exchange.fetch_ticker(market_id)
        latest_market_price = ticker_data['last']
        
        # Log order details
        mode_info = f" (positionSide: {positionSide})" if ENABLE_HEDGE_MODE else f" (reduceOnly: {reduceOnly})"
        logger.info(f"Placing {side.upper()} order for {market_id} - Qty: {quantity:.8f} at ~{latest_market_price:.6f}{mode_info}")
        
        # Place market order
        try:
            market_order = exchange.create_order(market_id, 'MARKET', side, quantity, params=params)
        except ccxt.InvalidOrder as e:
            error_msg = str(e)
            
            # Handle various Binance errors
            if 'ReduceOnly Order is rejected' in error_msg and 'reduceOnly' in params:
                logger.warning(f"ReduceOnly order rejected for {market_id}. Retrying without reduceOnly parameter.")
                params.pop('reduceOnly', None)
                market_order = exchange.create_order(market_id, 'MARKET', side, quantity, params=params)
            elif 'Parameter \'reduceonly\' sent when not required' in error_msg:
                logger.warning(f"reduceOnly parameter not required for {market_id}. Retrying without it.")
                params.pop('reduceOnly', None)
                market_order = exchange.create_order(market_id, 'MARKET', side, quantity, params=params)
            else:
                raise
        
        # Log success
        logger.info(f"Order successfully placed for {market_id}: {side.upper()} {quantity:.8f} @ {latest_market_price:.6f}, ID: {market_order.get('id', 'N/A')}")
        
        # Record entry time if not reducing
        if not reduceOnly:
            position_key = market_id
            if ENABLE_HEDGE_MODE and positionSide:
                position_key = f"{market_id}_{positionSide}"
                
            with cooldown_lock:
                position_entry_times[position_key] = time.time()
        
        return market_order

    except Exception as e:
        logger.exception(f"Error placing order for {market_id}: {e}")
        return None
def check_for_hedged_position(open_positions, symbol, position_side=None):
    """
    Check if a symbol has an open position in the specified side when in hedge mode.
    This helps determine if we already have a hedged position in the opposite direction.
    
    Args:
        open_positions: Dictionary of open positions from fetch_open_positions
        symbol: Trading pair symbol to check
        position_side: Optional - specific position side to check ('LONG' or 'SHORT')
        
    Returns:
        tuple: (has_position, position_info) where position_info is the position data if found
    """
    # If hedge mode is disabled, just check if the symbol exists in positions
    if not ENABLE_HEDGE_MODE:
        if symbol in open_positions:
            return True, open_positions[symbol]
        return False, None
    
    # In hedge mode, we need to check with position key format
    if position_side:
        # Check for specific position side (symbol_LONG or symbol_SHORT)
        position_key = f"{symbol}_{position_side}"
        if position_key in open_positions:
            return True, open_positions[position_key]
    else:
        # Check for any position in this symbol (either side)
        for key, position in open_positions.items():
            if position['symbol'] == symbol:
                return True, position
    
    return False, None
def calculate_volatility_adjusted_tp(exchange, symbol, position_type, entry_price):
    """Calculate volatility-adjusted take-profit level."""
    try:
        # Get price data
        df = fetch_extended_historical_data(exchange, symbol, TIMEFRAME)
        if df is None or len(df) < 20:
            # Default to standard 20% if insufficient data
            return entry_price * (1.2 if position_type == 'long' else 0.8)
            
        # Calculate 20-period rolling volatility (standard deviation of returns)
        returns = df['close'].pct_change().iloc[-20:].dropna().values
        volatility = np.std(returns) * 100  # Convert to percentage
        
        # Adjust for timeframe - scale up for smaller timeframes
        if TIMEFRAME == '1m':
            volatility = volatility * 2
        elif TIMEFRAME == '3m':
            volatility = volatility * 1.5
            
        # Apply multiplier to volatility to get TP percentage
        k = 1.5  # Multiplier
        tp_pct = volatility * k
        
        # Clamp TP percentage between 5% and 30%
        tp_pct = max(5.0, min(30.0, tp_pct))
        
        logger.info(f"Volatility for {symbol}: {volatility:.2f}%, Adjusted TP: {tp_pct:.2f}%")
        
        # Calculate target price based on position type and entry price
        if position_type == 'long':
            target_price = entry_price * (1 + tp_pct/100)
        else:  # short
            target_price = entry_price * (1 - tp_pct/100)
            
        return target_price
        
    except Exception as e:
        logger.error(f"Error calculating volatility-adjusted TP: {e}")
        # Default to standard 20% on error
        return entry_price * (1.2 if position_type == 'long' else 0.8)
def place_tp_only_order(exchange, market_id, entry_side, quantity, leverage=10, retries=2):
    """
    Place a take-profit order with risk-optimized profit target.
    Implements the framework rule: Target Reward ≥ 2 × Risk per Layer
    """
    try:
        # Clean up market_id if it has position suffix
        if isinstance(market_id, str):
            if '_LONG' in market_id:
                market_id = market_id.split('_LONG')[0]
                logger.info(f"Cleaned up market ID: {market_id}")
            elif '_SHORT' in market_id:
                market_id = market_id.split('_SHORT')[0]
                logger.info(f"Cleaned up market ID: {market_id}")
            
        # Get position information from exchange
        open_positions = fetch_open_positions(exchange)
        
        # Look for the position using both the regular symbol and potential position-side-specific keys
        position_info = None
        if market_id in open_positions:
            position_info = open_positions[market_id]
        else:
            # Try with position side suffixes if in hedge mode
            if ENABLE_HEDGE_MODE:
                long_key = f"{market_id}_LONG"
                short_key = f"{market_id}_SHORT"
                if entry_side == 'buy' and long_key in open_positions:
                    position_info = open_positions[long_key]
                elif entry_side == 'sell' and short_key in open_positions:
                    position_info = open_positions[short_key]
        
        # Get entry price - from position or fallback to current price
        if position_info:
            entry_price = position_info['entry_price']
        else:
            # Fallback if position not found
            current_price = exchange.fetch_ticker(market_id)['last']
            entry_price = current_price
            
        logger.info(f"Setting TP for {market_id} with entry price: {entry_price}")
        
        # Determine position type and get stop loss
        position_type = 'long' if entry_side == 'buy' else 'short'
        
        # Get indicator to determine stop loss
        indicator = get_momentum_indicator(market_id, exchange)
        df = fetch_extended_historical_data(exchange, market_id, MOMENTUM_TIMEFRAME)
        
        # Calculate stop loss
        if indicator and df is not None and len(df) > 20:
            stop_loss = indicator.determine_stop_loss(df, position_type=position_type, entry_price=entry_price)
        else:
            # Fallback calculation if indicator not available
            stop_loss = entry_price * (1 - 0.06/leverage if position_type == 'long' else 1 + 0.06/leverage)
        
        # FRAMEWORK IMPLEMENTATION: Calculate risk-optimized target
        target_price = calculate_risk_optimized_tp(
            exchange, market_id, position_type, entry_price, stop_loss
        )
        
        # Ensure formatting is correct
        formatted_entry = float(f"{entry_price:.6f}")
        formatted_target = float(f"{target_price:.6f}")
        
        # Verify calculation
        if position_type == 'long':
            actual_pct = ((formatted_target - formatted_entry) / formatted_entry) * 100
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        else:
            actual_pct = ((formatted_entry - formatted_target) / formatted_entry) * 100
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        
        # Calculate risk-reward ratio
        risk_reward_ratio = reward / risk if risk > 0 else 0
            
        logger.info(f"Risk-optimized TP target for {market_id}: {formatted_target} ({actual_pct:.2f}% price change, " +
                    f"Risk-Reward {risk_reward_ratio:.2f}:1)")
    
    except Exception as e:
        logger.warning(f"Error calculating TP price for {market_id}: {e}")
        return False

    # For TP orders, we want the opposite side of our entry
    inverted_side = 'sell' if entry_side == 'buy' else 'buy'
    
    try:
        # Set up TP order parameters
        tp_params = {
            'stopPrice': exchange.price_to_precision(market_id, target_price),
            'timeInForce': 'GTE_GTC',
        }
        
        # IMPORTANT: Handle reduceOnly vs positionSide properly
        if ENABLE_HEDGE_MODE:
            # In hedge mode, specify positionSide but NOT reduceOnly (they conflict)
            positionSide = 'LONG' if entry_side == 'buy' else 'SHORT'
            tp_params['positionSide'] = positionSide
            # Do NOT add reduceOnly in hedge mode for TP orders
            hedge_info = f" (positionSide: {positionSide})"
        else:
            # In one-way mode, use reduceOnly
            tp_params['reduceOnly'] = True
            hedge_info = ""
        
        logger.info(f"Placing Risk-Optimized TP Order for {market_id}: {inverted_side.upper()} "
                   f"Qty:{quantity:.8f} @ Stop:{tp_params['stopPrice']}{hedge_info} (Risk-Reward {risk_reward_ratio:.2f}:1)")
        
        tp_order = exchange.create_order(
            market_id, 'TAKE_PROFIT_MARKET', inverted_side, quantity, None, params=tp_params
        )
        
        logger.info(f"TP order {tp_order.get('id', 'N/A')} placed for {market_id} at {tp_params['stopPrice']}")
                
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
        order = place_order(
            exchange, 
            symbol, 
            new_entry_side, 
            quantity, 
            current_price, 
            leverage=leverage,
            positionSide=new_position_type.upper()  # Specify new position type
        )
        
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
    Enhanced momentum trading loop implementing the mathematical framework.
    - Validates entries with minimum risk-reward of 2:1
    - Uses pyramid sizing based on Pₙ = Pₙ₋₁ × 0.5
    - Implements dynamic hedging based on retracement
    """
    while True:
        try:
            # Sync positions with exchange
            sync_positions_with_exchange(exchange)
            
            # Check for position exits including hedging and reversals
            check_for_position_exits(exchange)
            
            # Check for pyramid opportunities
            check_for_pyramid_entries(exchange)
            
            # Check current positions after the above operations
            open_positions = fetch_open_positions(exchange)
            
            # Count only SHORT positions for MAX_OPEN_TRADES check
            short_positions = {key: pos for key, pos in open_positions.items() 
                              if pos['position_type'] == 'short'}
            
            # Check if we can look for new SHORT entries
            if len(short_positions) >= MAX_OPEN_TRADES:
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
                        stop_loss = signal['stop_loss']
                        
                        # FRAMEWORK IMPLEMENTATION: Check risk-reward ratio
                        target_price = calculate_risk_optimized_tp(
                            exchange, symbol, 'short', current_price, stop_loss
                        )
                        
                        meets_requirement, actual_ratio = check_risk_reward_ratio(
                            current_price, stop_loss, target_price, 'short', min_ratio=2.0
                        )
                        
                        if not meets_requirement:
                            logger.info(f"Skipping {symbol} SHORT: Risk-reward ratio ({actual_ratio:.2f}) below minimum requirement (2.0)")
                            continue
                        
                        logger.info(f"MOMENTUM SHORT SIGNAL: {symbol} - {signal['reason']} with R:R {actual_ratio:.2f}")
                        
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
                                 f"(Risk-Reward: {actual_ratio:.2f}:1)")
                        
                        order = place_order(
                                exchange, 
                                symbol, 
                                'sell', 
                                quantity, 
                                current_price, 
                                leverage=leverage,
                                positionSide='SHORT'  # Explicitly set short position side
                            )
                        
                        if order:
                            logger.info(f"Opened MOMENTUM SHORT position for {symbol}")
                            
                            # Record entry time for cooldown checks
                            with cooldown_lock:
                                position_entry_times[symbol] = time.time()
                            
                            # Place risk-optimized TP order
                            place_tp_only_order(
                                exchange, 
                                symbol, 
                                'sell',  # Entry side was sell for short
                                quantity, 
                                leverage
                            )
                            
                            entry_placed = True
                            
                            # Refresh open positions and count only SHORT positions for limit check
                            open_positions = fetch_open_positions(exchange)
                            short_positions = {key: pos for key, pos in open_positions.items() 
                                              if pos['position_type'] == 'short'}
                            
                            if len(short_positions) >= MAX_OPEN_TRADES:
                                logger.info(f"Reached maximum number of SHORT trades ({len(short_positions)}/{MAX_OPEN_TRADES}). "
                                          f"Stopping entry processing.")
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
            
            # Process long candidates without checking MAX_OPEN_TRADES limit
            # We allow unlimited LONG positions now
            if not entry_placed:  # Only try LONG entries if no SHORT entries were placed
                # For now, we'll check for long signals in the stored signals
                long_signals = {s: sig for s, sig in stored_signals.items() 
                               if sig['signal'] == 'buy' and s not in open_positions}
                
                if long_signals:
                    logger.info(f"Found {len(long_signals)} potential long entries to process")
                    
                    for symbol, signal in long_signals.items():
                        try:
                            current_price = signal['price']
                            stop_loss = signal['stop_loss']
                            
                            # FRAMEWORK IMPLEMENTATION: Check risk-reward ratio
                            target_price = calculate_risk_optimized_tp(
                                exchange, symbol, 'long', current_price, stop_loss
                            )
                            
                            meets_requirement, actual_ratio = check_risk_reward_ratio(
                                current_price, stop_loss, target_price, 'long', min_ratio=2.0
                            )
                            
                            if not meets_requirement:
                                logger.info(f"Skipping {symbol} LONG: Risk-reward ratio ({actual_ratio:.2f}) below minimum requirement (2.0)")
                                continue
                            
                            position_size = calculate_position_size(symbol)
                            
                            leverage = get_leverage_for_market(exchange, symbol)
                            if not leverage:
                                continue
                                
                            quantity = get_order_quantity(exchange, symbol, current_price, leverage, position_size)
                            if quantity <= 0:
                                continue
                                
                            logger.info(f"MOMENTUM LONG SIGNAL: {symbol} - {signal['reason']} with R:R {actual_ratio:.2f}")
                            
                            # Initialize tracking for max profit
                            with max_profit_lock:
                                max_profit_tracking[symbol] = 0
                                
                            # Initialize pyramid tracking
                            with pyramid_lock:
                                pyramid_details[symbol] = {'count': 0, 'entries': []}
                                
                            order = place_order(
                                        exchange, 
                                        symbol, 
                                        'buy', 
                                        quantity, 
                                        current_price, 
                                        leverage=leverage,
                                        positionSide='LONG'  # Explicitly set long position side
                                    )
                            
                            if order:
                                logger.info(f"Opened MOMENTUM LONG position for {symbol}")
                                
                                # Record entry time for cooldown checks
                                with cooldown_lock:
                                    position_entry_times[symbol] = time.time()
                                
                                # Place risk-optimized TP order
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
    
    # Log updated trading settings
    logger.info(f"Starting momentum trading bot with MAX_OPEN_TRADES={MAX_OPEN_TRADES} (SHORT positions only)")
    logger.info(f"LONG positions will not count against the MAX_OPEN_TRADES limit")
    
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