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
AMOUNT_INCREMENT = 0.5  # Position size increments
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
# Position details dictionary for enhanced trade tracking
position_details = {}  # {symbol: {'entry_price', 'stop_loss', 'target', 'position_type', 'entry_reason', 'probability', 'entry_time'}}
position_details_lock = threading.Lock()  # Lock for accessing position details

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


# NEW: Method for calculating position size based on performance
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

# NEW: Function to check for pyramid opportunity
# MODIFIED: Function to check for pyramid opportunity
def check_for_pyramid_opportunity(exchange, symbol):
    """
    Check if we should add to an existing position via pyramiding.
    Uses ADAPTIVE logic based on absolute price movements from high/low
    instead of percentage-based thresholds.

    Returns:
        tuple: (should_pyramid, size_to_add, reason)
    """
    try:
        with position_details_lock:
            if symbol not in position_details:
                return False, 0, "No active position"

            position_info = position_details[symbol].copy()

        # Get pyramid details
        with pyramid_lock:
            pyramid_info = pyramid_details.get(symbol, {'count': 0, 'entries': []})
            pyramid_count = pyramid_info.get('count', 0)

        # Check if we've reached maximum pyramid entries
        if pyramid_count >= PYRAMID_CONFIG['max_entries']:
            return False, 0, f"Maximum pyramid count reached ({pyramid_count})"

        # Get position details
        position_type = position_info.get('position_type')
        entry_price = position_info.get('entry_price')
        current_total_size = position_info.get('position_size', BASE_AMOUNT_USD)
        leverage = position_info.get('leverage', 10)

        # Get current position value from exchange position data
        current_position_value = current_total_size  # Default fallback
        try:
            open_positions = fetch_open_positions(exchange)
            if symbol in open_positions:
                pos_data = open_positions[symbol]['info']
                current_position_value = float(pos_data.get('initialMargin', current_total_size))
                logger.debug(f"Position values for {symbol}: Exchange reports initialMargin={current_position_value}, "
                           f"tracking has position_size={current_total_size}")
        except Exception as e:
            logger.warning(f"Error getting position value from exchange for {symbol}: {e}")
        
        # Use the higher of our tracked value and exchange-reported value for safety
        current_total_size = max(current_total_size, current_position_value)
        
        # Check if current position already at max size
        if current_total_size >= MAX_AMOUNT_USD:
            logger.info(f"Cannot pyramid {symbol}: Position size ${current_total_size:.2f} already at or exceeds max ${MAX_AMOUNT_USD:.2f}")
            return False, 0, f"Position already at maximum size (${current_total_size:.2f})"

        # Get the momentum indicator instance
        indicator = get_momentum_indicator(symbol, exchange)
        if not indicator:
            logger.warning(f"Cannot check pyramid for {symbol}: Failed to get momentum indicator")
            return False, 0, "Failed to get indicator"

        # Fetch fresh data for indicator analysis
        df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
        if df_historical is None or len(df_historical) < max(indicator.kama_period, 100):
             logger.warning(f"Insufficient historical data for {symbol} pyramid check.")
             return False, 0, "Insufficient historical data"

        # Get current price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']

        # Minimum remaining potential threshold (10%)
        MIN_REMAINING_POTENTIAL = 10.0
        
        # Number of pyramid levels to use for calculations
        TOTAL_PYRAMID_LEVELS = PYRAMID_CONFIG['max_entries']
        
        potential_pyramid_level = -1
        target_entry_price_for_level = -1.0
        size_to_add = AMOUNT_INCREMENT  # MODIFIED: Use fixed AMOUNT_INCREMENT
        reason_for_entry = "No adaptive pyramid opportunity detected"

        # Logic for SHORT positions
        if position_type == 'short':
            logger.debug(f"Checking adaptive pyramid for SHORT {symbol}")

            # Get the most recent significant gain details
            has_gain, gain_pct, low_price_gain, high_price, low_idx_gain, high_idx_gain = indicator.check_price_gain(df_historical)

            if not has_gain or high_price is None or gain_pct <= 0:
                logger.debug(f"No significant gain basis found for {symbol} SHORT pyramid.")
                return False, 0, "No significant gain basis for pyramid"

            # Calculate current drawdown from high price
            drawdown_pct = ((high_price - current_price) / high_price) * 100

            # Calculate remaining potential downside percentage
            remaining_potential_pct = gain_pct - drawdown_pct

            # Check minimum remaining potential
            if remaining_potential_pct < MIN_REMAINING_POTENTIAL:
                logger.debug(f"Skipping SHORT pyramid for {symbol}: Remaining potential {remaining_potential_pct:.2f}% < {MIN_REMAINING_POTENTIAL}% (Gain: {gain_pct:.2f}%, Drawdown: {drawdown_pct:.2f}%)")
                return False, 0, f"Remaining potential too low ({remaining_potential_pct:.1f}%)"

            # MODIFIED: Calculate pyramid levels dynamically 
            # Each level is equally spaced between 0 and (gain_pct - MIN_REMAINING_POTENTIAL)
            usable_range_pct = gain_pct - MIN_REMAINING_POTENTIAL
            level_size_pct = usable_range_pct / TOTAL_PYRAMID_LEVELS
            
            # Check which pyramid level the current price is at
            for i in range(pyramid_count, TOTAL_PYRAMID_LEVELS):
                # Calculate target drawdown percentage for this level
                target_drawdown_pct = level_size_pct * (i + 1)
                
                # Calculate the target entry price (price drops by target_drawdown_pct from high_price)
                target_entry_price = high_price * (1 - target_drawdown_pct / 100.0)

                # Check if current price meets or is below this target entry price
                if current_price <= target_entry_price:
                    logger.info(f"Adaptive Short Pyramid Trigger Check for {symbol} Level {i+1}: "
                               f"Current Price {current_price:.6f} <= Target Price {target_entry_price:.6f} "
                               f"(Target Drop: {target_drawdown_pct:.2f}% of total range {usable_range_pct:.2f}%)")
                    
                    # Store the *first* level triggered in this check cycle
                    if potential_pyramid_level == -1:
                        potential_pyramid_level = i
                        target_entry_price_for_level = target_entry_price
                        reason_for_entry = (f"Adaptive Short Pyramid Level {potential_pyramid_level + 1}: Price <= {target_entry_price_for_level:.6f} "
                                           f"(Target Drop: {target_drawdown_pct:.2f}% of total range {usable_range_pct:.2f}%)")
                    # Continue checking subsequent levels in case multiple triggered
                else:
                    logger.debug(f"Adaptive Short Pyramid Condition NOT MET for {symbol} Level {i+1}: "
                               f"Current Price {current_price:.6f} > Target Price {target_entry_price:.6f}")
                    # Stop checking further levels for shorts if one isn't met
                    break

        # Logic for LONG positions
        elif position_type == 'long':
            logger.debug(f"Checking adaptive pyramid for LONG {symbol}")

            # Get the most recent significant decrease details
            has_decrease, decrease_pct, high_price_dec, low_price, high_idx_dec, low_idx_dec = indicator.check_price_decrease(df_historical)

            if not has_decrease or low_price is None or decrease_pct <= 0:
                logger.debug(f"No significant decrease basis found for {symbol} LONG pyramid.")
                return False, 0, "No significant decrease basis for pyramid"

            # Calculate current recovery from the low price
            recovery_pct = ((current_price - low_price) / low_price) * 100 if low_price > 0 else 0

            # Calculate remaining potential upside percentage
            remaining_potential_pct = decrease_pct - recovery_pct

            # Check minimum remaining potential
            if remaining_potential_pct < MIN_REMAINING_POTENTIAL:
                logger.debug(f"Skipping LONG pyramid for {symbol}: Remaining potential {remaining_potential_pct:.2f}% < {MIN_REMAINING_POTENTIAL}% (Decrease: {decrease_pct:.2f}%, Recovery: {recovery_pct:.2f}%)")
                return False, 0, f"Remaining potential too low ({remaining_potential_pct:.1f}%)"

            # MODIFIED: Calculate pyramid levels dynamically
            # Each level is equally spaced between 0 and (decrease_pct - MIN_REMAINING_POTENTIAL)
            usable_range_pct = decrease_pct - MIN_REMAINING_POTENTIAL
            level_size_pct = usable_range_pct / TOTAL_PYRAMID_LEVELS
            
            # Check which pyramid level the current price is at
            for i in range(pyramid_count, TOTAL_PYRAMID_LEVELS):
                # Calculate target recovery percentage for this level
                target_recovery_pct = level_size_pct * (i + 1)
                
                # Calculate the target entry price (price recovers by target_recovery_pct from low_price)
                target_entry_price = low_price * (1 + target_recovery_pct / 100.0)

                # Check if current price meets or is above this target entry price
                if current_price >= target_entry_price:
                    logger.info(f"Adaptive Long Pyramid Trigger Check for {symbol} Level {i+1}: "
                               f"Current Price {current_price:.6f} >= Target Price {target_entry_price:.6f} "
                               f"(Target Recovery: {target_recovery_pct:.2f}% of total range {usable_range_pct:.2f}%)")
                    
                    # Store the *first* level triggered in this check cycle
                    if potential_pyramid_level == -1:
                        potential_pyramid_level = i
                        target_entry_price_for_level = target_entry_price
                        reason_for_entry = (f"Adaptive Long Pyramid Level {potential_pyramid_level + 1}: Price >= {target_entry_price_for_level:.6f} "
                                           f"(Target Recovery: {target_recovery_pct:.2f}% of total range {usable_range_pct:.2f}%)")
                    # Continue checking subsequent levels in case multiple triggered
                else:
                    logger.debug(f"Adaptive Long Pyramid Condition NOT MET for {symbol} Level {i+1}: "
                               f"Current Price {current_price:.6f} < Target Price {target_entry_price:.6f}")
                    # Stop checking further levels for longs if one isn't met
                    break

        else: # Handle unknown position type
             logger.warning(f"Unknown position type '{position_type}' for {symbol} in pyramid check.")
             return False, 0, "Unknown position type"

        # If a pyramid level's condition was met (either short or long)
        if potential_pyramid_level != -1:
            # MODIFIED: Always use fixed AMOUNT_INCREMENT for size_to_add
            add_size_raw = AMOUNT_INCREMENT
            
            # Ensure we respect MAX_AMOUNT_USD using the more accurate current_total_size
            capped_add_size = max(0.0, min(add_size_raw, MAX_AMOUNT_USD - current_total_size))

            # Avoid adding trivially small amounts
            if capped_add_size < (AMOUNT_INCREMENT / 2.0):
                 logger.info(f"Pyramid add size for {symbol} is too small ({capped_add_size:.2f} USD), skipping.")
                 return False, 0, f"Calculated add size {capped_add_size:.2f} too small"

            logger.info(f"Adaptive Pyramid Size Calculation: Adding fixed amount {AMOUNT_INCREMENT}, "
                       f"CurrentTotal={current_total_size:.2f}, "
                       f"MaxTotal={MAX_AMOUNT_USD:.2f}, CappedAdd={capped_add_size:.2f}")

            # Return True, the calculated size TO ADD, and the reason
            return True, capped_add_size, reason_for_entry
        else:
            # No levels triggered for the relevant position type
            return False, 0, reason_for_entry # Return the reason why no levels were triggered

    except Exception as e:
        logger.error(f"Error checking pyramid opportunity for {symbol}: {e}", exc_info=True)
        return False, 0, f"Error: {str(e)}"
    
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

# NEW: Function to record trade results for performance tracking
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
            'position_size': BASE_AMOUNT_USD,  # Default position size
            'max_profit_pct': 0
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
            'position_size': BASE_AMOUNT_USD,
            'max_profit_pct': 0
        }


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
    """Fetch open positions from the exchange with margin information."""
    try:
        positions = exchange.fetch_positions()
        result = {}
        
        for pos in positions:
            # Filter only positions with non-zero amount
            position_amt = float(pos['info'].get('positionAmt', 0))
            if abs(position_amt) > 0:
                symbol = pos['symbol']
                entry_price = float(pos['info'].get('entryPrice', 0))
                
                # Get initialMargin if available or calculate it
                initial_margin = pos.get('initialMargin', None)
                leverage = pos.get('leverage', 10)  # Default to 10x if not available
                
                # Calculate notional value (position size in USD)
                notional_value = abs(position_amt * entry_price)
                
                # If initialMargin is not available, estimate it
                if initial_margin is None or initial_margin == 0:
                    # Calculate based on notional value and leverage
                    initial_margin = notional_value / leverage
                
                # Get current market price for current value calculation
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    current_value = abs(position_amt * current_price)
                except Exception:
                    # Fallback to entry price if can't get current price
                    current_price = entry_price
                    current_value = notional_value
                
                # Add all info to the position record
                result[symbol] = {
                    'symbol': symbol,
                    'info': {
                        'positionAmt': str(position_amt),
                        'entryPrice': str(entry_price),
                        'initialMargin': str(initial_margin),
                        'leverage': str(leverage),
                        'notionalValue': str(notional_value),
                        'currentValue': str(current_value)
                    }
                }
        
        return result
    except Exception as e:
        logger.exception(f"Error fetching positions: {e}")
        return {}

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
                    # logger.info(f"Changing margin mode for {market_id} from {current_margin_mode} to isolated")
                    exchange.set_margin_mode('isolated', market_id)
                    # Small delay to ensure the change takes effect
                    time.sleep(0.5)
            else:
                # If no position info available, set to isolated mode anyway
                # logger.info(f"Setting margin mode for {market_id} to isolated")
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
    Place only a take-profit order with pyramid-adaptive profit target.
    
    Args:
        exchange: CCXT exchange instance
        market_id: Trading pair symbol
        entry_side: Side of the entry order ('buy' or 'sell')
        quantity: Position size
        leverage: Position leverage (default: 10)
        retries: Number of retry attempts if order fails
        
    Returns:
        bool: Success or failure
    """
    # Get current market price and position details
    try:
        current_price = exchange.fetch_ticker(market_id)['last']
        logger.info(f"Setting TP for {market_id} at current price: {current_price}")
        
        # Get position details to verify against entry price
        with position_details_lock:
            if market_id in position_details:
                entry_price = position_details[market_id].get('entry_price', current_price)
                position_leverage = position_details[market_id].get('leverage', leverage)
            else:
                entry_price = current_price
                position_leverage = leverage
        
        # Get pyramid level for adaptive TP
        with pyramid_lock:
            pyramid_info = pyramid_details.get(market_id, {'count': 0, 'entries': []})
            pyramid_count = pyramid_info.get('count', 0)
            
        # Adjust TP target based on pyramid level - more aggressive with deeper pyramids
        base_profit_multiplier = 2.0  # 200% profit target baseline
        
        # Scale down profit target as pyramid count increases (more conservative)
        if pyramid_count >= PYRAMID_CONFIG['max_entries']:
            # At max pyramid level, target 150% profit instead of 200%
            profit_multiplier = base_profit_multiplier * 0.75
        elif pyramid_count > 0:
            # Scale down by 0.1x for each pyramid level
            profit_multiplier = base_profit_multiplier * (1.0 - 0.1 * pyramid_count)
        else:
            profit_multiplier = base_profit_multiplier
            
        # Calculate target price with adjusted multiplier
        if entry_side == 'buy':  # LONG
            # For longs: TP at adjusted profit % = entry + profit_multiplier * position value
            target_price = entry_price * (1 + profit_multiplier/position_leverage)
            
            # Validate TP is above current price 
            if target_price <= current_price:
                logger.warning(f"Invalid TP for {market_id} LONG: TP {target_price} <= Current {current_price}")
                # Set a reasonable TP at 10% above current price if validation fails
                target_price = current_price * 1.10
                logger.info(f"Corrected TP to {target_price}")
                
        else:  # SHORT
            # For shorts: TP at adjusted profit % = entry - profit_multiplier * position value
            target_price = entry_price * (1 - profit_multiplier/position_leverage)
            
            # Validate TP is below current price for shorts
            if target_price >= current_price:
                logger.warning(f"Invalid TP for {market_id} SHORT: TP {target_price} >= Current {current_price}")
                # Set a reasonable TP at 10% below current price if validation fails
                target_price = current_price * 0.90
                logger.info(f"Corrected TP to {target_price}")
    
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
        
        logger.info(f"Placing TP Order ({profit_multiplier*100:.0f}% profit) for {market_id}: {inverted_side.upper()} "
                   f"Qty:{quantity:.8f} @ Stop:{tp_params['stopPrice']}")
        
        tp_order = exchange.create_order(
            market_id, 'TAKE_PROFIT_MARKET', inverted_side, quantity, None, params=tp_params
        )
        
        logger.info(f"TP order {tp_order.get('id', 'N/A')} placed for {market_id} at {tp_params['stopPrice']} ({profit_multiplier*100:.0f}% profit target)")

        # Store only the TP order ID
        with position_details_lock:
            if market_id in position_details:
                position_details[market_id]['tp_order_id'] = tp_order.get('id')
                position_details[market_id]['target'] = target_price
                
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

# NEW: Enhanced order quantity calculation considering position size
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

# NEW: Function to handle position reversals
def handle_position_reversal(exchange, symbol, reversal_strength, current_position_type, profit_pct=0):
    """
    Handle position reversal when a reversal signal is detected.
    Closes existing position and opens new one in opposite direction.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        reversal_strength: 0-100 strength score of reversal signal
        current_position_type: 'long' or 'short'
        profit_pct: Current profit percentage (already leverage-adjusted)
        
    Returns:
        bool: Success or failure
    """
    try:
        # Check if reversal meets minimum criteria
        if reversal_strength < REVERSAL_CONFIG['min_reversal_strength']:
            logger.info(f"Reversal signal for {symbol} too weak ({reversal_strength}) to trigger automatic reversal")
            return False
            
        # Check if we have minimum profit for reversal
        # profit_pct is already leverage-adjusted from the calling function
        if profit_pct < REVERSAL_CONFIG['min_profit_for_reversal']:
            logger.info(f"Insufficient profit ({profit_pct:.1f}%) for {symbol} reversal. Minimum: {REVERSAL_CONFIG['min_profit_for_reversal']}%")
            return False
        
        # Fetch current position details
        with position_details_lock:
            if symbol not in position_details:
                logger.warning(f"Cannot reverse position for {symbol}: No position details found")
                return False
                
            position_info = position_details[symbol].copy()
        
        # Get position details
        entry_price = position_info.get('entry_price')
        current_size = position_info.get('position_size', BASE_AMOUNT_USD)
        
        # Get latest market price
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Determine new position type
        new_position_type = 'long' if current_position_type == 'short' else 'short'
        
        # Determine new entry side
        new_entry_side = 'buy' if new_position_type == 'long' else 'sell'
        
        # Calculate reversal position size (larger than original when profitable)
        reversal_size = min(current_size * REVERSAL_CONFIG['reversal_size_multiplier'], MAX_AMOUNT_USD)
        
        # Get leverage
        leverage = position_info.get('leverage', 10)  # Use existing leverage if available
        if not leverage:
            leverage = get_leverage_for_market(exchange, symbol)
        if not leverage:
            logger.error(f"Failed to set leverage for {symbol} reversal")
            return False
            
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
            position_info.get('entry_time', time.time() - 3600),
            leverage=leverage  # Pass leverage for proper profit calculation
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
        
        # Calculate target price (2x risk)
        if new_position_type == 'long':
            sl_distance = current_price - stop_loss
            target_price = current_price + (sl_distance * 2)
        else:  # short
            sl_distance = stop_loss - current_price
            target_price = current_price - (sl_distance * 2)
            
        # Reset pyramid count for this symbol
        with pyramid_lock:
            pyramid_details[symbol] = {'count': 0, 'entries': []}
            
        # Update position details
        with position_details_lock:
            position_details[symbol] = {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target': target_price,
                'position_type': new_position_type,
                'entry_reason': f"Reversal from {current_position_type} with strength {reversal_strength}",
                'entry_time': time.time(),
                'position_size': reversal_size,
                'leverage': leverage,
                'highest_reached': current_price if new_position_type == 'long' else None,
                'lowest_reached': current_price if new_position_type == 'short' else None,
                'max_profit_pct': 0
            }
            
        # Place TP order for new position (skip SL order since we use dynamic management)
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
                    # Record trade result before removing from tracking
                    current_price = exchange.fetch_ticker(symbol)['last']
                    position_info = position_details[symbol]
                    
                    record_trade_result(
                        symbol,
                        position_info.get('entry_price'),
                        current_price,
                        position_info.get('position_type'),
                        position_info.get('entry_time')
                    )
                    
                    # Remove from tracking
                    del position_details[symbol]
            
            # Reset pyramid count for this symbol
            with pyramid_lock:
                if symbol in pyramid_details:
                    del pyramid_details[symbol]
        
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
                    'position_size': BASE_AMOUNT_USD,  # Default size
                    'max_profit_pct': 0
                }
            
            # Initialize pyramid tracking for this symbol
            with pyramid_lock:
                pyramid_details[symbol] = {'count': 0, 'entries': []}
            
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
            
        # logger.info(f"Updating entry-price-based reference stop levels for {len(open_positions)} openpositions")
        
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

# NEW: Function to check for and handle position exits
# Updated check_for_position_exits with dynamic, market-adaptive thresholds
def check_for_position_exits(exchange):
    """
    Check for position exits that mirror pyramid entry logic.
    Reduces position at the same percentage levels used for pyramid entries.
    NO time restrictions for scalping strategies.
    """
    try:
        # Get open positions
        open_positions = fetch_open_positions(exchange)
        
        if not open_positions:
            return
            
        logger.info(f"Checking {len(open_positions)} open positions for exit conditions")
        
        for symbol in open_positions.keys():
            try:
                # Skip if not in our position tracking
                with position_details_lock:
                    if symbol not in position_details:
                        continue
                        
                    position_info = position_details[symbol].copy()
                
                # Get position details
                position_type = position_info.get('position_type')
                entry_price = position_info.get('entry_price')
                avg_entry_price = position_info.get('avg_entry_price', entry_price)
                original_entry_price = position_info.get('original_entry_price', entry_price)
                position_size = position_info.get('position_size', BASE_AMOUNT_USD)
                leverage = position_info.get('leverage', 10)
                
                # Get pyramid details 
                with pyramid_lock:
                    pyramid_info = pyramid_details.get(symbol, {'count': 0, 'entries': []})
                    pyramid_count = pyramid_info.get('count', 0)
                    pyramid_entries = pyramid_info.get('entries', [])
                
                # Get current price
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Get position quantity from exchange
                pos_info = open_positions[symbol]
                pos_amt = float(pos_info['info']['positionAmt'])
                total_quantity = abs(pos_amt)
                
                # Get inverted side for exit orders
                inverted_side = 'buy' if position_type == 'short' else 'sell'
                
                # =====================================================================
                # PART 1: MIRROR PYRAMID LOGIC FOR POSITION REDUCTION
                # =====================================================================
                
                # Only process if we have pyramid entries to potentially exit
                if pyramid_count > 0 and len(pyramid_entries) > 0:
                    try:
                        # Get the most recent pyramid entry
                        last_entry = pyramid_entries[-1]
                        last_entry_price = last_entry.get('price', 0)
                        last_entry_quantity = last_entry.get('quantity', 0)
                        
                        # Get the movement thresholds that were used for pyramiding
                        # These are the same thresholds we'll use for exit, creating symmetry
                        target_move_capture_pcts = PYRAMID_CONFIG['profit_thresholds']  # e.g., [15, 30, 45]
                        
                        # Get the indicator instance (needed for gain/decrease calculations)
                        indicator = get_momentum_indicator(symbol, exchange)
                        if indicator is None:
                            logger.warning(f"Cannot check mirror exit for {symbol}: Failed to get momentum indicator")
                            continue
                            
                        # Get historical data for gain/decrease calculation
                        df_historical = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                        if df_historical is None or len(df_historical) < max(indicator.kama_period, 100):
                            logger.warning(f"Insufficient historical data for {symbol} mirror exit check.")
                            continue
                        
                        # Check for exit opportunity based on position type
                        exit_triggered = False
                        exit_level_index = -1
                        
                        if position_type == 'short':
                            # For shorts: Check if price has moved back up toward high price
                            # This mirrors the logic used in check_for_pyramid_opportunity but in reverse
                            
                            # Get gain info from indicator
                            has_gain, gain_pct, low_price_gain, high_price, low_idx_gain, high_idx_gain = indicator.check_price_gain(df_historical)
                            
                            if has_gain and high_price is not None and high_price > 0:
                                # Calculate how much price has recovered from the low (retraced the downmove)
                                # For shorts, we exit if price retraces back upward
                                current_recovery_pct = ((current_price - last_entry_price) / (high_price - last_entry_price)) * 100 if (high_price - last_entry_price) > 0 else 0
                                
                                # Check which pyramid threshold it has crossed
                                for i, threshold_pct in enumerate(target_move_capture_pcts):
                                    # Skip checking levels above our current pyramid count
                                    if i >= pyramid_count:
                                        break
                                        
                                    # For the most recent pyramid level (highest index), we're looking for ANY significant retrace
                                    if i == pyramid_count - 1:
                                        # For the most recent entry, use a fixed threshold
                                        if current_recovery_pct >= 30:  # 30% retrace of the move
                                            exit_triggered = True
                                            exit_level_index = i
                                            logger.info(f"SHORT MIRROR EXIT for {symbol}: Price recovered {current_recovery_pct:.2f}% "
                                                      f"from last entry at {last_entry_price} toward high {high_price}")
                                            break
                                    # For earlier pyramid levels, we use the exact pyramid thresholds
                                    elif current_recovery_pct >= threshold_pct:
                                        exit_triggered = True
                                        exit_level_index = i
                                        logger.info(f"SHORT MIRROR EXIT for {symbol}: Price recovered {current_recovery_pct:.2f}% "
                                                  f"matching pyramid threshold {threshold_pct}%")
                                        break
                            
                        elif position_type == 'long':
                            # For longs: Check if price has moved back down toward low price
                            # This mirrors the logic used in check_for_pyramid_opportunity but in reverse
                            
                            # Get decrease info from indicator
                            has_decrease, decrease_pct, high_price_dec, low_price, high_idx_dec, low_idx_dec = indicator.check_price_decrease(df_historical)
                            
                            if has_decrease and low_price is not None and low_price > 0:
                                # Calculate how much price has retraced from the high (retraced the upmove)
                                # For longs, we exit if price retraces back downward
                                current_retrace_pct = ((last_entry_price - current_price) / (last_entry_price - low_price)) * 100 if (last_entry_price - low_price) > 0 else 0
                                
                                # Check which pyramid threshold it has crossed
                                for i, threshold_pct in enumerate(target_move_capture_pcts):
                                    # Skip checking levels above our current pyramid count
                                    if i >= pyramid_count:
                                        break
                                        
                                    # For the most recent pyramid level (highest index), we're looking for ANY significant retrace
                                    if i == pyramid_count - 1:
                                        # For the most recent entry, use a fixed threshold
                                        if current_retrace_pct >= 30:  # 30% retrace of the move
                                            exit_triggered = True
                                            exit_level_index = i
                                            logger.info(f"LONG MIRROR EXIT for {symbol}: Price retraced {current_retrace_pct:.2f}% "
                                                      f"from last entry at {last_entry_price} toward low {low_price}")
                                            break
                                    # For earlier pyramid levels, we use the exact pyramid thresholds
                                    elif current_retrace_pct >= threshold_pct:
                                        exit_triggered = True
                                        exit_level_index = i
                                        logger.info(f"LONG MIRROR EXIT for {symbol}: Price retraced {current_retrace_pct:.2f}% "
                                                  f"matching pyramid threshold {threshold_pct}%")
                                        break
                        
                        # Execute the exit if triggered
                        if exit_triggered and exit_level_index >= 0 and exit_level_index < len(pyramid_entries):
                            # Get the entry we're exiting
                            exit_entry = pyramid_entries[exit_level_index]
                            exit_quantity = exit_entry.get('quantity', 0)
                            
                            if exit_quantity > 0:
                                logger.info(f"Executing mirror exit for {symbol} at pyramid level {exit_level_index + 1}")
                                
                                # Limit the exit quantity to ensure it's not more than the total
                                partial_exit_qty = min(exit_quantity, total_quantity * 0.9)
                                
                                # Close quantity equal to the pyramid entry
                                partial_exit_order = place_order(
                                    exchange,
                                    symbol,
                                    inverted_side,
                                    partial_exit_qty,
                                    current_price,
                                    reduceOnly=True
                                )
                                
                                if partial_exit_order:
                                    logger.info(f"Successfully executed mirror exit for {symbol}: "
                                              f"Closed {partial_exit_qty:.8f} of {total_quantity:.8f} total")
                                    
                                    # Update pyramid info - remove the exited level and all above it
                                    with pyramid_lock:
                                        if symbol in pyramid_details and pyramid_details[symbol]['count'] > 0:
                                            # Remove the appropriate entries
                                            pyramid_details[symbol]['entries'] = pyramid_details[symbol]['entries'][:exit_level_index]
                                            pyramid_details[symbol]['count'] = len(pyramid_details[symbol]['entries'])
                                            logger.info(f"Updated pyramid count for {symbol} to {pyramid_details[symbol]['count']}")
                                    
                                    # Update position details
                                    with position_details_lock:
                                        if symbol in position_details:
                                            # Recalculate position size after exit
                                            remaining_size = position_size
                                            for i in range(exit_level_index, len(pyramid_entries)):
                                                remaining_size -= pyramid_entries[i].get('added_size_usd', 0)
                                            
                                            position_details[symbol]['position_size'] = max(0, remaining_size)
                                            logger.info(f"Updated position size for {symbol} to ${max(0, remaining_size):.2f}")
                                            
                                            # Reset to entry price before the exited pyramid levels
                                            if exit_level_index == 0:
                                                # If exiting first level, reset to original entry
                                                position_details[symbol]['avg_entry_price'] = original_entry_price
                                            else:
                                                # Otherwise use the price from the previous level
                                                prev_level_price = pyramid_entries[exit_level_index - 1].get('price', original_entry_price)
                                                position_details[symbol]['avg_entry_price'] = prev_level_price
                                    
                                    # Skip other exit checks this cycle - we've already taken action
                                    continue
                                else:
                                    logger.error(f"Failed to execute mirror exit order for {symbol}")
                    
                    except Exception as e:
                        logger.exception(f"Error in mirror exit logic for {symbol}: {e}")
                        # Continue to regular exit checks even if mirror exit fails
                
                # =====================================================================
                # PART 2: REGULAR FULL EXIT LOGIC - Only if we get here
                # =====================================================================
                
                try:
                    # Get indicator for this symbol
                    indicator = get_momentum_indicator(symbol, exchange)
                    if not indicator:
                        logger.warning(f"Cannot check exit for {symbol}: Failed to get momentum indicator")
                        continue
                        
                    # Get historical data
                    df = fetch_extended_historical_data(exchange, symbol, MOMENTUM_TIMEFRAME)
                    if df is None or len(df) < 100:
                        logger.warning(f"Insufficient data to check exit for {symbol}")
                        continue
                        
                    # Update indicator with latest data
                    indicator.update_price_data(df)
                    
                    # Calculate current profit percentage WITH LEVERAGE ADJUSTMENT
                    current_profit_pct = calculate_position_profit(avg_entry_price, current_price, position_type, leverage)
                    
                    # Update max profit if applicable
                    max_profit_pct = position_info.get('max_profit_pct', 0)
                    if current_profit_pct > max_profit_pct:
                        with position_details_lock:
                            if symbol in position_details:
                                position_details[symbol]['max_profit_pct'] = current_profit_pct
                        max_profit_pct = current_profit_pct
                    
                    # Check for significant profit retracement
                    profit_retracement = max_profit_pct - current_profit_pct
                    retracement_ratio = profit_retracement / max_profit_pct if max_profit_pct > 0 else 0
                    
                    # Check for reversal signals and generate exit signal
                    is_reversal, reversal_strength, reversal_reason = indicator.detect_reversal(df, position_type)
                    stop_loss = position_info.get('stop_loss')
                    exit_signal = indicator.generate_exit_signal(df, position_type, avg_entry_price, stop_loss)
                    
                    # Check exit conditions
                    exit_triggered = False
                    exit_reason = None
                    should_reverse = False
                    
                    # 1. Check standard exit signal
                    if exit_signal['exit_triggered']:
                        exit_triggered = True
                        exit_reason = exit_signal['reason']
                        should_reverse = exit_signal.get('execute_reversal', False)
                    
                    # 2. Check profit retracement (with reduced threshold for pyramid positions)
                    # More pyramids = stronger conviction = more aggressive profit protection
                    elif max_profit_pct > 20:
                        # Base threshold at 40%, but reduce for each pyramid level
                        retracement_threshold = 0.4 - (0.05 * pyramid_count)  # 40%, 35%, 30%, 25%
                        retracement_threshold = max(0.25, retracement_threshold)  # Don't go below 25%
                        
                        if retracement_ratio > retracement_threshold:
                            exit_triggered = True
                            exit_reason = f"Profit retracement: {profit_retracement:.1f}% ({retracement_ratio:.2f}) "
                            exit_reason += f"from max {max_profit_pct:.1f}% [Threshold: {retracement_threshold:.2f}]"
                            should_reverse = is_reversal and reversal_strength > 50
                    
                    # 3. Check strong reversal signal
                    elif is_reversal and reversal_strength >= 70:
                        exit_triggered = True
                        exit_reason = f"Strong reversal signal detected ({reversal_strength}): {reversal_reason}"
                        should_reverse = True
                    
                    # Process the exit if needed
                    if exit_triggered:
                        logger.info(f"EXIT SIGNAL for {symbol} {position_type.upper()}: {exit_reason}")
                        
                        # Close position
                        exit_order = place_order(
                            exchange, 
                            symbol, 
                            inverted_side, 
                            total_quantity, 
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
                                position_info.get('entry_time'),
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
                                with position_details_lock:
                                    if symbol in position_details:
                                        del position_details[symbol]
                                        
                                with pyramid_lock:
                                    if symbol in pyramid_details:
                                        del pyramid_details[symbol]
                        else:
                            logger.error(f"Failed to exit position for {symbol}")
                            
                except Exception as e:
                    logger.exception(f"Error in regular exit checks for {symbol}: {e}")
                    # Continue to next symbol
            
            except Exception as e:
                logger.exception(f"Error checking exit for {symbol}: {e}")
                continue
    
    except Exception as e:
        logger.exception(f"Error in check_for_position_exits: {e}")
# NEW: Function to check for pyramid opportunities
def check_for_pyramid_entries(exchange):
    """Check for pyramid entry opportunities in profitable positions with proper position size limits."""
    try:
        # Get open positions
        open_positions = fetch_open_positions(exchange)

        if not open_positions:
            return

        # logger.info(f"Checking {len(open_positions)} open positions for pyramid opportunities")

        for symbol in open_positions.keys():
            try:
                # Check if we can pyramid this position using potentially modified logic
                should_pyramid, size_to_add, reason = check_for_pyramid_opportunity(exchange, symbol)

                if not should_pyramid:
                    continue # Skip to next symbol

                # Get position details needed for placing order
                with position_details_lock:
                    if symbol not in position_details:
                        logger.warning(f"Position details disappeared for {symbol} during pyramid check")
                        continue

                    position_info = position_details[symbol].copy()

                position_type = position_info.get('position_type')
                leverage = position_info.get('leverage', 10) # Use stored or default leverage
                current_total_size = position_info.get('position_size', BASE_AMOUNT_USD) # Get current total size before adding
                entry_price = position_info.get('entry_price')

                # FIXED: Double-check that adding won't exceed MAX_AMOUNT_USD
                if current_total_size + size_to_add > MAX_AMOUNT_USD:
                    logger.warning(f"Pyramid add for {symbol} would exceed MAX_AMOUNT_USD. "
                                   f"Current: ${current_total_size:.2f}, Add: ${size_to_add:.2f}, Max: ${MAX_AMOUNT_USD:.2f}")
                    
                    # Calculate how much we can actually add (if any)
                    allowable_add = max(0, MAX_AMOUNT_USD - current_total_size)
                    if allowable_add < AMOUNT_INCREMENT / 2:
                        logger.info(f"Skipping pyramid for {symbol}: only ${allowable_add:.2f} available to add (below minimum)")
                        continue
                    
                    # Adjust size_to_add to stay within limits
                    logger.info(f"Adjusting pyramid add size for {symbol} from ${size_to_add:.2f} to ${allowable_add:.2f}")
                    size_to_add = allowable_add

                # Determine entry side ('buy' for long, 'sell' for short)
                entry_side = 'buy' if position_type == 'long' else 'sell'

                # Get current price for order placement
                current_price = exchange.fetch_ticker(symbol)['last']

                # Calculate quantity based on the *size_to_add* returned by the check function
                quantity = get_order_quantity(exchange, symbol, current_price, leverage, size_to_add) # Use size_to_add here

                if quantity <= 0:
                    logger.warning(f"Invalid quantity ({quantity}) calculated for {symbol} pyramid entry with add size {size_to_add:.2f} USD.")
                    continue

                # Log pyramid details before placing order
                logger.info(f"ADDING PYRAMID POSITION for {symbol} {position_type.upper()}: {reason}")
                logger.info(f"Current position size: ${current_total_size:.2f}, Adding: ${size_to_add:.2f} (Qty: {quantity:.8f})")

                # Place the pyramid order (adds to the existing position)
                pyramid_order = place_order(
                    exchange,
                    symbol,
                    entry_side,
                    quantity,
                    current_price, # Market order, price is indicative
                    leverage=leverage
                    # reduceOnly should NOT be True for adding to a position
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
                            'added_size_usd': size_to_add, # Store the USD size added
                            'quantity': quantity,       # Store quantity added
                            'time': time.time()
                        })
                        new_pyramid_count = pyramid_details[symbol]['count']
                        
                        # Get all pyramid entries for weighted average calculation
                        pyramid_entries = pyramid_details[symbol]['entries']

                    # Calculate weighted average entry price
                    # Get position details from exchange for accurate total size
                    total_quantity = abs(float(open_positions[symbol]['info']['positionAmt']))
                    
                    # Initialize calculation variables
                    weighted_sum = entry_price * (total_quantity - quantity)  # Original position value
                    weighted_sum += executed_price * quantity  # New position value
                    
                    # Calculate weighted average
                    avg_entry_price = weighted_sum / total_quantity if total_quantity > 0 else entry_price
                    
                    # Update the total position size in position tracking
                    with position_details_lock:
                        if symbol in position_details:
                            # Update size by adding the actual size added
                            new_total_size = current_total_size + size_to_add
                            position_details[symbol]['position_size'] = new_total_size
                            # Save average entry price
                            position_details[symbol]['avg_entry_price'] = avg_entry_price
                            # Keep original entry for reference
                            if 'original_entry_price' not in position_details[symbol]:
                                position_details[symbol]['original_entry_price'] = entry_price
                                
                            logger.info(f"Updated position size for {symbol} to ${new_total_size:.2f} (Pyramid Level: {new_pyramid_count})")
                            logger.info(f"Updated average entry price for {symbol} to {avg_entry_price:.6f} (Original: {entry_price:.6f})")

                else:
                    logger.error(f"Failed to place pyramid add order for {symbol}")

            except Exception as e:
                logger.exception(f"Error checking/placing pyramid entry for {symbol}: {e}")
                continue # Continue to the next symbol

    except Exception as e:
        logger.exception(f"Error in check_for_pyramid_entries main loop: {e}")

def momentum_trading_loop(exchange):
    """
    Enhanced momentum trading loop with position sizing, pyramiding, and reversal handling.
    Handles both short and long trading opportunities.
    """
    while True:
        try:
            # Sync positions with exchange first
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
            
            # Find coins with significant price movements (both gainers and losers)
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
                            
                        current_kama = signal.get('kama_value')
                        
                        # Calculate fixed SL and TP based on entry price and leverage
                        # For short, SL is 6% price move divided by leverage (60% position value with 10x)
                        stop_loss = current_price * (1 + 0.06/leverage)
                        # Target is 10% price move divided by leverage (100% position value with 10x)
                        target_price = current_price * (1 - 0.1/leverage)
                            
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
                                'leverage': leverage,
                                'position_size': position_size,
                                'max_profit_pct': 0
                            }
                            
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
                            
                            executed_price = order.get('average', current_price)
                            with position_details_lock:
                                if symbol in position_details:
                                    position_details[symbol]['entry_price'] = executed_price
                                    # Update SL and TP with executed price
                                    position_details[symbol]['stop_loss'] = executed_price * (1 + 0.06/leverage)
                                    position_details[symbol]['target'] = executed_price * (1 - 0.1/leverage)
                                    stop_loss = position_details[symbol]['stop_loss']
                                    target_price = position_details[symbol]['target']
                            
                            logger.info(f"Position opened for {symbol} with TP at {target_price} (10% price move / ~100% profit with {leverage}x leverage)")
                            
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
                            with position_details_lock:
                                if symbol in position_details:
                                    del position_details[symbol]
                                    
                            with pyramid_lock:
                                if symbol in pyramid_details:
                                    del pyramid_details[symbol]
                
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
                            # Target is 10% price move with leverage (100% profit with 10x leverage)
                            target_price = current_price * (1 + 0.1/leverage)
                            
                            logger.info(f"MOMENTUM LONG SIGNAL: {symbol} - {signal['reason']}")
                            
                            with position_details_lock:
                                position_details[symbol] = {
                                    'entry_price': current_price,
                                    'stop_loss': stop_loss,
                                    'target': target_price,
                                    'position_type': 'long',
                                    'entry_reason': f"Momentum Long: {signal['reason']}",
                                    'probability': signal.get('probability', 0.7),
                                    'entry_time': time.time(),
                                    'highest_reached': current_price,
                                    'lowest_reached': None,
                                    'signal_type': 'momentum_long',
                                    'signal_strength': 70,
                                    'leverage': leverage,
                                    'position_size': position_size,
                                    'max_profit_pct': 0
                                }
                                
                            # Initialize pyramid tracking
                            with pyramid_lock:
                                pyramid_details[symbol] = {'count': 0, 'entries': []}
                                
                            order = place_order(
                                exchange, symbol, 'buy', quantity, current_price, leverage=leverage
                            )
                            
                            if order:
                                logger.info(f"Opened MOMENTUM LONG position for {symbol}")
                                
                                executed_price = order.get('average', current_price)
                                with position_details_lock:
                                    if symbol in position_details:
                                        position_details[symbol]['entry_price'] = executed_price
                                        # Update SL with executed price
                                        position_details[symbol]['stop_loss'] = executed_price * (1 - 0.06/leverage)
                                        position_details[symbol]['target'] = executed_price * (1 + 0.1/leverage)
                                        stop_loss = position_details[symbol]['stop_loss']
                                        target_price = position_details[symbol]['target']
                                
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
                                with position_details_lock:
                                    if symbol in position_details:
                                        del position_details[symbol]
                                with pyramid_lock:
                                    if symbol in pyramid_details:
                                        del pyramid_details[symbol]
                                        
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
    logger.info("Started enhanced momentum trading thread with dynamic position sizing and pyramiding")
    
    # Keep the main thread alive
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()