import os
import time
import logging
import threading
import json
import queue
import ccxt
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from collections import defaultdict, deque, Counter
from custom_indicator import EnhancedScalpingWrapper

# Configuration
SLEEP_INTERVAL = 10  # seconds
MAX_OPEN_TRADES = 2
AMOUNT_USD = 1
TIMEFRAME = '3m'
LIMIT = 100
LEVERAGE_OPTIONS = [30, 25, 15, 8]
OPTIMIZATION_INTERVAL = 86400 * 3  # Optimize parameters every 3 days

# Anti-cycling settings
MINIMUM_HOLD_MINUTES = 2  # Minimum time to hold a position regardless of signals (in minutes)
SYMBOL_COOLDOWN_MINUTES = 15  # Time to wait before re-entering after exit (in minutes)

# Optimization settings
MAX_EVALS = 15
BACKTEST_CANDLES = 500
OPTIMIZATION_THREAD_SLEEP = 20  # Sleep time between optimization attempts (in seconds)

# Thread synchronization
optimization_queue = queue.Queue()  # Thread-safe queue for optimization tasks
optimization_results = {}  # Store optimization results
optimization_lock = threading.Lock()  # Lock for accessing optimization results

# Track recently traded symbols with cooldown timers
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
# Suppress Hyperopt internal logs
logging.getLogger('hyperopt').setLevel(logging.WARNING)

# Directory for storing optimized parameters
PARAMS_DIR = 'optimized_params'
os.makedirs(PARAMS_DIR, exist_ok=True)

# Directory for trade logs
TRADE_LOG_DIR = 'trade_logs'
os.makedirs(TRADE_LOG_DIR, exist_ok=True)

# Directory for pattern statistics
PATTERN_STATS_DIR = 'pattern_stats'
os.makedirs(PATTERN_STATS_DIR, exist_ok=True)

# Cache for OHLCV data with TTL
ohlcv_cache = defaultdict(lambda: {'data': None, 'timestamp': 0})
CACHE_TTL = 300  # 5 minutes
cache_lock = threading.Lock()  # Lock for accessing cache data

# Dictionary to store indicator instances for each symbol
symbol_indicators = {}
# Track when we last optimized for each symbol
last_optimization = {}
indicator_lock = threading.Lock()  # Lock for accessing indicator data

# Default parameters when optimization is not available
DEFAULT_PARAMS = {
    # Core volatility parameters
    'atr_window': 3,  # Reduced from 4 to 3
    'vol_multiplier': 0.8,  # Reduced from 1.0 to 0.8
    'min_vol': 0.0008,  # Reduced from 0.001 to 0.0008
    
    # Risk/reward parameters
    'risk_reward_ratio': 1.4,  # Reduced from 1.8 to 1.4 for quicker profits
    'min_stop_atr': 0.2,  # Reduced from 0.3 to 0.2
    'max_stop_atr': 0.6,  # Reduced from 0.8 to 0.6
    
    # Pattern detection parameters
    'pattern_min_reliability': 30.0,  # REDUCED from 50.0 to 30.0 to allow more trades
    'stoch_threshold': 60.0,  # Reduced from 70.0 to 60.0
    'smooth_period': 8,  # Reduced from 20 to 8
    'min_pattern_count': 5,  # Reduced from 10 to 5
    'pattern_stats_file': "global_pattern_stats.json",
    
    # NEW: Trend analysis parameters
    'trend_period': 4,  # Reduced from 20 to 4
    'trend_angle_threshold': 0.05,  # Reduced from 5.0 to 0.05 (matches your data showing 0.05 angles)
    'trend_strength_multiplier': 1.3,  # Increased to 1.3
    'trend_lookback': 2  # Reduced from 5 to 2
}

def log_trade_metrics(reason, increment=True):
    """Increment and log metrics on trade filtering."""
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
        # Generate a report before resetting
        report_metrics()
        # Reset all counters except last_reset
        for key in global_metrics:
            if key != 'last_reset':
                global_metrics[key] = 0
        global_metrics['last_reset'] = current_time
        logger.info("Trade metrics reset")

def report_metrics():
    """Generate a report of current trade filtering metrics."""
    global global_metrics
    
    patterns_detected = global_metrics.get('patterns_detected', 0)
    
    if patterns_detected > 0:
        # Calculate percentages
        below_threshold_pct = global_metrics.get('patterns_below_threshold', 0) / patterns_detected * 100
        tf_confirm_pct = global_metrics.get('insufficient_tf_confirmation', 0) / patterns_detected * 100
        sequence_pct = global_metrics.get('failed_sequence_check', 0) / patterns_detected * 100
        reversal_pct = global_metrics.get('failed_reversal_check', 0) / patterns_detected * 100
        risk_reward_pct = global_metrics.get('failed_risk_reward', 0) / patterns_detected * 100
        proximity_pct = global_metrics.get('failed_proximity_check', 0) / patterns_detected * 100
        order_errors_pct = global_metrics.get('order_placement_errors', 0) / patterns_detected * 100
        
        # Calculate success rate
        success_rate = global_metrics.get('successful_entries', 0) / patterns_detected * 100
        
        # Build report
        report = "\n" + "=" * 50 + "\n"
        report += "TRADE FILTERING METRICS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Total Patterns Detected: {patterns_detected}\n"
        report += f"Successful Trade Entries: {global_metrics.get('successful_entries', 0)} ({success_rate:.2f}%)\n\n"
        
        report += "REJECTION REASONS:\n"
        report += f"1. Pattern Below Threshold: {global_metrics.get('patterns_below_threshold', 0)} ({below_threshold_pct:.2f}%)\n"
        report += f"2. Insufficient Timeframe Confirmation: {global_metrics.get('insufficient_tf_confirmation', 0)} ({tf_confirm_pct:.2f}%)\n"
        report += f"3. Failed Sequence Check: {global_metrics.get('failed_sequence_check', 0)} ({sequence_pct:.2f}%)\n"
        report += f"4. Failed Reversal Check: {global_metrics.get('failed_reversal_check', 0)} ({reversal_pct:.2f}%)\n"
        report += f"5. Failed Risk/Reward Check: {global_metrics.get('failed_risk_reward', 0)} ({risk_reward_pct:.2f}%)\n"
        report += f"6. Failed Proximity Check: {global_metrics.get('failed_proximity_check', 0)} ({proximity_pct:.2f}%)\n"
        report += f"7. Order Placement Errors: {global_metrics.get('order_placement_errors', 0)} ({order_errors_pct:.2f}%)\n"
        
        report += "\n" + "=" * 50 + "\n"
        
        # Log the report
        logger.info(report)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        report_path = os.path.join(TRADE_LOG_DIR, f"trade_metrics_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Trade metrics report saved to {report_path}")
    else:
        logger.info("No patterns detected since last reset, no metrics to report")

def log_trade_funnel():
    """Periodically log the trade funnel to track conversion rates."""
    patterns_detected = global_metrics.get('patterns_detected', 0)
    
    if patterns_detected == 0:
        logger.info("No patterns detected yet, cannot calculate trade funnel")
        return
    
    # Calculate funnel steps
    passed_threshold = patterns_detected - global_metrics.get('patterns_below_threshold', 0)
    passed_tf_confirm = passed_threshold - global_metrics.get('insufficient_tf_confirmation', 0)
    passed_sequence = passed_tf_confirm - global_metrics.get('failed_sequence_check', 0)
    passed_reversal = passed_sequence - global_metrics.get('failed_reversal_check', 0)
    passed_risk_reward = passed_reversal - global_metrics.get('failed_risk_reward', 0)
    passed_proximity = passed_risk_reward - global_metrics.get('failed_proximity_check', 0)
    successful_orders = global_metrics.get('successful_entries', 0)
    
    # Calculate conversion rates
    threshold_rate = passed_threshold / patterns_detected * 100 if patterns_detected > 0 else 0
    tf_confirm_rate = passed_tf_confirm / passed_threshold * 100 if passed_threshold > 0 else 0
    sequence_rate = passed_sequence / passed_tf_confirm * 100 if passed_tf_confirm > 0 else 0
    reversal_rate = passed_reversal / passed_sequence * 100 if passed_sequence > 0 else 0
    risk_reward_rate = passed_risk_reward / passed_reversal * 100 if passed_reversal > 0 else 0
    proximity_rate = passed_proximity / passed_risk_reward * 100 if passed_risk_reward > 0 else 0
    order_success_rate = successful_orders / passed_proximity * 100 if passed_proximity > 0 else 0
    
    # Overall funnel conversion
    overall_conversion = successful_orders / patterns_detected * 100 if patterns_detected > 0 else 0
    
    # Log the funnel
    logger.info("\n" + "-" * 40)
    logger.info("TRADE FUNNEL ANALYSIS")
    logger.info("-" * 40)
    logger.info(f"Patterns Detected: {patterns_detected} (100%)")
    logger.info(f"→ Passed Threshold: {passed_threshold} ({threshold_rate:.1f}%)")
    logger.info(f"  → Passed TF Confirmation: {passed_tf_confirm} ({tf_confirm_rate:.1f}%)")
    logger.info(f"    → Passed Sequence Check: {passed_sequence} ({sequence_rate:.1f}%)")
    logger.info(f"      → Passed Reversal Check: {passed_reversal} ({reversal_rate:.1f}%)")
    logger.info(f"        → Passed Risk/Reward: {passed_risk_reward} ({risk_reward_rate:.1f}%)")
    logger.info(f"          → Passed Proximity Check: {passed_proximity} ({proximity_rate:.1f}%)")
    logger.info(f"            → Successful Orders: {successful_orders} ({order_success_rate:.1f}%)")
    logger.info("-" * 40)
    logger.info(f"OVERALL CONVERSION RATE: {overall_conversion:.2f}%")
    logger.info("-" * 40)

# Replace the detect_swing_sr_levels function with this improved version
def detect_swing_sr_levels(exchange, symbol, trend_direction="flat"):
    """
    Detect support and resistance levels using price swing points.
    Improved to use recent price action and ensure levels are valid for SL/TP placement.
    
    Args:
        exchange: Exchange object
        symbol: Trading pair symbol
        trend_direction: Current trend direction ('bullish', 'bearish', 'flat')
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Get data from multiple timeframes for confirmation
    # Adjusted to use more recent data points for better accuracy
    timeframes = {
        '1m': 120,   # Very recent levels
        '3m': 40,    # Recent levels
        '5m': 24,    # Medium-term levels
        '15m': 12     # Major levels
    }
    
    # Store support and resistance levels
    all_supports = []
    all_resistances = []
    all_recent_lows = []   # Track all recent lows for fallback
    all_recent_highs = []  # Track all recent highs for fallback
    current_price = None
    
    # Analyze each timeframe
    for tf, limit in timeframes.items():
        try:
            # Fetch data for this timeframe
            df = fetch_binance_data(exchange, symbol, timeframe=tf, limit=limit)
            if len(df) < 20:  # Need at least 20 candles
                continue
                
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Store the most recent high and low for reference
            recent_high = df['high'].iloc[-1]
            recent_low = df['low'].iloc[-1]
            
            # Calculate recent price range and volatility
            price_range_pct = (df['high'].max() - df['low'].min()) / current_price * 100
            price_swings = df['high'].max() - df['low'].min()
            
            # Track all significant highs and lows for potential SL/TP placements
            all_recent_highs.extend(df['high'].iloc[-limit:].values)
            all_recent_lows.extend(df['low'].iloc[-limit:].values)
            
            # Find swing highs and lows using a simple rolling window
            # Adjusted window size based on timeframe (smaller for shorter timeframes)
            if tf == '1m':
                window_size = 3
            elif tf == '3m':
                window_size = 4
            elif tf == '5m':
                window_size = 5
            else:  # 15m or higher
                window_size = 6
            
            # Create columns for swing points
            df['swing_high'] = False
            df['swing_low'] = False
            
            # Identify swing points
            for i in range(window_size, len(df) - window_size):
                # Check for swing high
                if df['high'].iloc[i] == df['high'].iloc[i-window_size:i+window_size+1].max():
                    df.loc[df.index[i], 'swing_high'] = True
                
                # Check for swing low
                if df['low'].iloc[i] == df['low'].iloc[i-window_size:i+window_size+1].min():
                    df.loc[df.index[i], 'swing_low'] = True
            
            # Extract swing highs and lows
            swing_highs = df[df['swing_high']]['high'].values
            swing_lows = df[df['swing_low']]['low'].values
            
            # Log the detected swing points
            logger.info(f"Detected {len(swing_highs)} swing highs and {len(swing_lows)} swing lows on {tf} for {symbol}")
            
            # Add all swing highs as potential resistance - don't filter by current price yet
            for level in swing_highs:
                all_resistances.append({
                    'level': float(level),
                    'source': tf,
                    'type': 'swing_high'
                })
            
            # Add all swing lows as potential support - don't filter by current price yet
            for level in swing_lows:
                all_supports.append({
                    'level': float(level),
                    'source': tf,
                    'type': 'swing_low'
                })
            
            # Also add recent high/low from each timeframe
            if recent_high > current_price:
                all_resistances.append({
                    'level': float(recent_high),
                    'source': f'{tf}_recent',
                    'type': 'recent_high'
                })
            
            if recent_low < current_price:
                all_supports.append({
                    'level': float(recent_low),
                    'source': f'{tf}_recent',
                    'type': 'recent_low'
                })
            
            # Add significant price levels from recent candles (e.g., round numbers, previous day high/low)
            # Calculate significant levels based on average price
            avg_price = df['close'].mean()
            
            # Look for round numbers near the current price
            price_digits = len(str(int(current_price)))
            round_factor = 10 ** (price_digits - 2)  # For round numbers
            
            # Generate potentially significant round numbers
            for mult in range(int(avg_price / round_factor) - 5, int(avg_price / round_factor) + 6):
                round_level = mult * round_factor
                
                # If level is near current price but not too close
                if 0.7 < round_level / current_price < 1.3:
                    if round_level > current_price:
                        all_resistances.append({
                            'level': float(round_level),
                            'source': f'{tf}_round',
                            'type': 'round_number'
                        })
                    else:
                        all_supports.append({
                            'level': float(round_level),
                            'source': f'{tf}_round', 
                            'type': 'round_number'
                        })
            
        except Exception as e:
            logger.warning(f"Error processing {tf} for {symbol}: {e}")
    
    if current_price is None:
        raise ValueError(f"Could not determine current price for {symbol}")
    
    # Now filter levels relative to current price
    supports_below_price = [s for s in all_supports if s['level'] < current_price]
    resistances_above_price = [r for r in all_resistances if r['level'] > current_price]
    
    # Log raw detection counts
    logger.info(f"Raw detection for {symbol}: {len(supports_below_price)} supports below price, "
               f"{len(resistances_above_price)} resistances above price")
    
    # Group nearby levels into zones
    support_zones = group_levels_into_zones(supports_below_price, tolerance_pct=0.005)
    resistance_zones = group_levels_into_zones(resistances_above_price, tolerance_pct=0.005)
    
    # Sort zones by strength (number of confirmations)
    support_zones.sort(key=lambda x: x['strength'], reverse=True)
    resistance_zones.sort(key=lambda x: x['strength'], reverse=True)
    
    # Calculate stop loss and take profit levels
    sl_level, tp_level = calculate_sl_tp_from_zones(
        support_zones,
        resistance_zones,
        current_price,
        trend_direction
    )
    
    # If SL/TP weren't properly determined, log an error
    if sl_level is None or tp_level is None:
        logger.error(f"Failed to calculate valid SL/TP levels for {symbol}")
    
    # Calculate risk/reward ratio
    risk = abs(current_price - sl_level) / current_price if sl_level is not None else 0
    reward = abs(tp_level - current_price) / current_price if tp_level is not None else 0
    risk_reward = reward / risk if risk > 0 else 0
    
    # Add raw price data for backup calculations if needed
    return {
        'support_zones': support_zones,
        'resistance_zones': resistance_zones,
        'current_price': current_price,
        'stop_loss': sl_level,
        'target_price': tp_level,
        'risk_reward': risk_reward,
        'raw_highs': all_recent_highs,  # Add raw data for better calculations if needed
        'raw_lows': all_recent_lows
    }

def group_levels_into_zones(levels, tolerance_pct=0.005):
    """
    Group nearby price levels into zones.
    
    Args:
        levels: List of price levels
        tolerance_pct: Percentage tolerance for considering levels as the same zone
        
    Returns:
        List of zone dictionaries with center price and strength
    """
    if not levels:
        return []
    
    # Sort levels by price
    sorted_levels = sorted(levels, key=lambda x: x['level'])
    
    # Group nearby levels
    zones = []
    current_zone_levels = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        # Check if this level is close to the current zone
        zone_center = sum(l['level'] for l in current_zone_levels) / len(current_zone_levels)
        
        # Calculate tolerance in absolute terms
        tolerance = zone_center * tolerance_pct
        
        if abs(level['level'] - zone_center) <= tolerance:
            # Add to current zone
            current_zone_levels.append(level)
        else:
            # Process current zone and start a new one
            if current_zone_levels:
                zones.append(create_zone_from_levels(current_zone_levels))
            current_zone_levels = [level]
    
    # Process the last zone
    if current_zone_levels:
        zones.append(create_zone_from_levels(current_zone_levels))
    
    return zones

def create_zone_from_levels(levels):
    """Create a zone dictionary from a list of levels."""
    # Calculate zone center (average of levels)
    center = sum(level['level'] for level in levels) / len(levels)
    
    # Get all sources and types
    sources = [level['source'] for level in levels]
    types = [level['type'] for level in levels]
    
    # Calculate zone strength based on number of levels and diversity of sources
    strength = len(levels) * (len(set(sources)) / len(sources) + 0.5)
    
    return {
        'center': center,
        'strength': strength,
        'levels': [level['level'] for level in levels],
        'sources': sources,
        'types': types,
        'confirmations': len(set(sources))
    }

# Replace the calculate_sl_tp_from_zones function with this improved version
def calculate_sl_tp_from_zones(support_zones, resistance_zones, current_price, trend_direction, leverage=None):
    """
    Calculate stop loss and take profit levels based strictly on support and resistance zones.
    Using zone edges for more accurate placement - lower edge of resistance for buy targets,
    upper edge of support for sell targets.
    
    Args:
        support_zones: List of support zones
        resistance_zones: List of resistance zones
        current_price: Current market price
        trend_direction: Current trend direction ('bullish', 'bearish', 'flat')
        leverage: Position leverage (needed for liquidation calculation)
        
    Returns:
        Tuple of (stop_loss, take_profit)
    """
    # Minimum required distance for stop-loss as percentage
    min_stop_pct = 0.008  # Minimum 0.8% stop distance
    
    # Find nearest support/resistance zones based on current price and trend
    sl_level = None
    tp_level = None
    risk_reward = 1.5  # Target risk/reward ratio
    
    # Helper function to get zone edges based on levels
    def get_zone_edges(zone):
        """Return the upper and lower edges of a zone based on its levels or center"""
        if 'levels' in zone and zone['levels']:
            # If we have individual levels in the zone, use min/max
            lower_edge = min(zone['levels'])
            upper_edge = max(zone['levels'])
            return lower_edge, upper_edge
        elif 'center' in zone:
            # If we only have center, estimate edges with a small spread
            spread = zone['center'] * 0.002  # 0.2% spread
            return zone['center'] - spread, zone['center'] + spread
        else:
            # Fallback
            return zone['center'], zone['center']
    
    if trend_direction == "bullish":  # LONG position
        # For longs, stop loss below support, take profit at resistance
        
        # Find supports below current price for stop loss
        valid_supports = [zone for zone in support_zones if zone['center'] < current_price]
        if valid_supports:
            # Sort by strength and proximity (prioritize stronger zones)
            valid_supports.sort(key=lambda z: (-z['strength'], current_price - z['center']))
            
            # Check each support zone, starting with strongest
            for support_zone in valid_supports:
                # Get the upper edge of the support zone for stop placement
                lower_edge, upper_edge = get_zone_edges(support_zone)
                
                # Place stop loss slightly below the lower edge of support zone
                potential_sl = lower_edge * 0.998  # 0.2% below lower edge
                
                # Check if stop distance is at least the minimum
                stop_distance_pct = (current_price - potential_sl) / current_price
                if stop_distance_pct >= min_stop_pct:
                    sl_level = potential_sl
                    logger.info(f"Using support zone at {support_zone['center']:.6f} (lower edge: {lower_edge:.6f}) for SL at {sl_level:.6f} ({stop_distance_pct:.2%} from price)")
                    break
        
        # If no valid support zone found, find closest price level that gives minimum stop distance
        if sl_level is None:
            sl_level = current_price * (1 - min_stop_pct)
            logger.info(f"No suitable support zone found, using minimum stop distance at {sl_level:.6f} ({min_stop_pct:.2%} from price)")
        
        # Calculate risk amount
        risk = current_price - sl_level
        
        # Find resistances above current price for take profit
        valid_resistances = [zone for zone in resistance_zones if zone['center'] > current_price]
        if valid_resistances:
            # Sort by strength and proximity (prioritize stronger zones)
            valid_resistances.sort(key=lambda z: (-z['strength'], z['center'] - current_price))
            
            # Check each resistance zone for suitable target
            for resistance_zone in valid_resistances:
                # Get the lower edge of the resistance zone for target placement
                lower_edge, upper_edge = get_zone_edges(resistance_zone)
                
                # Use the LOWER edge of resistance as target for long positions
                potential_tp = lower_edge
                
                # Calculate reward and R/R ratio
                reward = potential_tp - current_price
                current_rr = reward / risk if risk > 0 else 0
                
                # Use this zone if it provides at least our target R/R
                if current_rr >= risk_reward:
                    tp_level = potential_tp
                    logger.info(f"Using resistance zone at {resistance_zone['center']:.6f} (lower edge: {lower_edge:.6f}) for TP at {tp_level:.6f} (R/R: {current_rr:.2f})")
                    break
        
        # If no suitable resistance found, create target based on risk/reward
        if tp_level is None:
            tp_level = current_price + (risk * risk_reward)
            logger.info(f"No suitable resistance zone found, setting TP at {tp_level:.6f} ({risk_reward}× risk)")
    
    else:  # SHORT position (bearish or flat trend)
        # For shorts, stop loss above resistance, take profit at support
        
        # Find resistances above current price for stop loss
        valid_resistances = [zone for zone in resistance_zones if zone['center'] > current_price]
        if valid_resistances:
            # Sort by strength and proximity (prioritize stronger zones)
            valid_resistances.sort(key=lambda z: (-z['strength'], z['center'] - current_price))
            
            # Check each resistance zone, starting with strongest
            for resistance_zone in valid_resistances:
                # Get the upper edge of the resistance zone for stop placement
                lower_edge, upper_edge = get_zone_edges(resistance_zone)
                
                # Place stop loss slightly above the upper edge of resistance zone
                potential_sl = upper_edge * 1.002  # 0.2% above upper edge
                
                # Check if stop distance is at least the minimum
                stop_distance_pct = (potential_sl - current_price) / current_price
                if stop_distance_pct >= min_stop_pct:
                    sl_level = potential_sl
                    logger.info(f"Using resistance zone at {resistance_zone['center']:.6f} (upper edge: {upper_edge:.6f}) for SL at {sl_level:.6f} ({stop_distance_pct:.2%} from price)")
                    break
        
        # If no valid resistance zone found, find closest price level that gives minimum stop distance
        if sl_level is None:
            sl_level = current_price * (1 + min_stop_pct)
            logger.info(f"No suitable resistance zone found, using minimum stop distance at {sl_level:.6f} ({min_stop_pct:.2%} from price)")
        
        # Calculate risk amount
        risk = sl_level - current_price
        
        # Find supports below current price for take profit
        valid_supports = [zone for zone in support_zones if zone['center'] < current_price]
        if valid_supports:
            # Sort by strength and proximity (prioritize stronger zones)
            valid_supports.sort(key=lambda z: (-z['strength'], current_price - z['center']))
            
            # Check each support zone for suitable target
            for support_zone in valid_supports:
                # Get the upper edge of the support zone for target placement
                lower_edge, upper_edge = get_zone_edges(support_zone)
                
                # Use the UPPER edge of support as target for short positions
                potential_tp = upper_edge
                
                # Calculate reward and R/R ratio
                reward = current_price - potential_tp
                current_rr = reward / risk if risk > 0 else 0
                
                # Use this zone if it provides at least our target R/R
                if current_rr >= risk_reward:
                    tp_level = potential_tp
                    logger.info(f"Using support zone at {support_zone['center']:.6f} (upper edge: {upper_edge:.6f}) for TP at {tp_level:.6f} (R/R: {current_rr:.2f})")
                    break
        
        # If no suitable support found, create target based on risk/reward
        if tp_level is None:
            tp_level = current_price - (risk * risk_reward)
            logger.info(f"No suitable support zone found, setting TP at {tp_level:.6f} ({risk_reward}× risk)")
    
    # Calculate final risk/reward ratio for logging
    final_risk = abs(current_price - sl_level) / current_price
    final_reward = abs(tp_level - current_price) / current_price
    final_rr = final_reward / final_risk if final_risk > 0 else 0
    
    logger.info(f"Final SL/TP for {trend_direction} trade at {current_price:.6f}: " +
               f"SL={sl_level:.6f} ({(abs(current_price - sl_level) / current_price * 100):.2f}% away), " +
               f"TP={tp_level:.6f} ({(abs(tp_level - current_price) / current_price * 100):.2f}% away), " +
               f"R/R={final_rr:.2f}")
    
    # Check for liquidation price if leverage is provided
    if leverage is not None and leverage > 1:
        # Binance typically uses 0.8% maintenance margin for most pairs
        maintenance_margin = 0.008
        
        if trend_direction == "bullish":  # LONG position
            # Liquidation price for a long is below entry
            liquidation_price = current_price * (1 - (1 / leverage) + maintenance_margin)
            
            # Add safety buffer to avoid getting too close to liquidation
            safe_buffer = (current_price - liquidation_price) * 0.25  # 25% buffer
            min_safe_level = liquidation_price + safe_buffer
            
            logger.info(f"LONG position - Entry: {current_price:.6f}, Liquidation: {liquidation_price:.6f}")
            
            # Ensure SL is above liquidation price plus buffer
            if sl_level is not None and sl_level <= min_safe_level:
                old_sl = sl_level
                sl_level = min_safe_level
                logger.warning(f"Adjusted SL from {old_sl:.6f} to {sl_level:.6f} to avoid liquidation risk")
                
                # Recalculate TP to maintain risk/reward ratio
                new_risk = current_price - sl_level
                tp_level = current_price + (new_risk * risk_reward)
                logger.info(f"Recalculated TP to {tp_level:.6f} to maintain {risk_reward:.1f} R/R ratio")
            
        else:  # SHORT position
            # Liquidation price for a short is above entry
            liquidation_price = current_price * (1 + (1 / leverage) - maintenance_margin)
            
            # Add safety buffer to avoid getting too close to liquidation
            safe_buffer = (liquidation_price - current_price) * 0.25  # 25% buffer
            max_safe_level = liquidation_price - safe_buffer
            
            logger.info(f"SHORT position - Entry: {current_price:.6f}, Liquidation: {liquidation_price:.6f}")
            
            # Ensure SL is below liquidation price minus buffer
            if sl_level is not None and sl_level >= max_safe_level:
                old_sl = sl_level
                sl_level = max_safe_level
                logger.warning(f"Adjusted SL from {old_sl:.6f} to {sl_level:.6f} to avoid liquidation risk")
                
                # Recalculate TP to maintain risk/reward ratio
                new_risk = sl_level - current_price
                tp_level = current_price - (new_risk * risk_reward)
                logger.info(f"Recalculated TP to {tp_level:.6f} to maintain {risk_reward:.1f} R/R ratio")
    
    return sl_level, tp_level

def calculate_sl_tp_from_recent_swings(support_zones, resistance_zones, current_price, trend_direction):
    """
    Calculate SL/TP based on recent price swings when S/R zones aren't available.
    Uses the most recent swing levels from available data.
    
    Args:
        support_zones: List of support zones (may be empty but contains raw source data)
        resistance_zones: List of resistance zones (may be empty but contains raw source data) 
        current_price: Current market price
        trend_direction: Current trend direction ('bullish', 'bearish', 'flat')
    
    Returns:
        Tuple of (stop_loss, take_profit)
    """
    # Default fallback distances for SL/TP
    sl_distance_pct = 0.015  # 1.5% for SL
    tp_distance_pct = 0.03   # 3% for TP
    
    # Try to extract recent swing levels from the raw data in the zones
    recent_highs = []
    recent_lows = []
    
    # Extract levels from support zones
    for zone in support_zones:
        if 'levels' in zone:
            recent_lows.extend(zone['levels'])
    
    # Extract levels from resistance zones
    for zone in resistance_zones:
        if 'levels' in zone:
            recent_highs.extend(zone['levels'])
    
    # If we have swing data, use it
    if recent_lows and recent_highs:
        logger.info(f"Found {len(recent_lows)} support levels and {len(recent_highs)} resistance levels")
        
        # Filter levels that are too close to current price
        min_distance_pct = 0.008  # 0.8% minimum
        
        valid_lows = [l for l in recent_lows if (current_price - l) / current_price >= min_distance_pct]
        valid_highs = [h for h in recent_highs if (h - current_price) / current_price >= min_distance_pct]
        
        if trend_direction == "bullish":
            # For LONG positions
            if valid_lows:
                # Sort by distance (closest first)
                valid_lows.sort(key=lambda x: current_price - x)
                sl_level = valid_lows[0] * 0.998  # Add small buffer (0.2%)
            else:
                sl_level = current_price * (1 - sl_distance_pct)
            
            if valid_highs:
                # For TP, use a high level but not too far
                valid_highs.sort(key=lambda x: x - current_price)
                # Use closest high that gives decent R/R
                for high in valid_highs:
                    reward = (high - current_price) / current_price
                    risk = (current_price - sl_level) / current_price
                    if reward / risk >= 1.5:
                        tp_level = high
                        break
                else:
                    # If no good level found, use default
                    tp_level = current_price * (1 + tp_distance_pct)
            else:
                tp_level = current_price * (1 + tp_distance_pct)
        else:
            # For SHORT positions
            if valid_highs:
                valid_highs.sort(key=lambda x: x - current_price)
                sl_level = valid_highs[0] * 1.002  # Add small buffer (0.2%)
            else:
                sl_level = current_price * (1 + sl_distance_pct)
            
            if valid_lows:
                valid_lows.sort(key=lambda x: current_price - x)
                # Find a low that gives good R/R
                for low in valid_lows:
                    reward = (current_price - low) / current_price
                    risk = (sl_level - current_price) / current_price
                    if reward / risk >= 1.5:
                        tp_level = low
                        break
                else:
                    tp_level = current_price * (1 - tp_distance_pct)
            else:
                tp_level = current_price * (1 - tp_distance_pct)
    else:
        # No swing data available, use percentage-based defaults
        logger.warning("No swing data available, using percentage-based defaults for SL/TP")
        
        if trend_direction == "bullish":
            sl_level = current_price * (1 - sl_distance_pct)
            tp_level = current_price * (1 + tp_distance_pct)
        else:
            sl_level = current_price * (1 + sl_distance_pct)
            tp_level = current_price * (1 - tp_distance_pct)
    
    # Log the selected levels
    if trend_direction == "bullish":
        logger.info(f"LONG SL/TP from recent swings: SL={sl_level:.6f} ({(current_price - sl_level) / current_price * 100:.2f}% away), "
                   f"TP={tp_level:.6f} ({(tp_level - current_price) / current_price * 100:.2f}% away)")
    else:
        logger.info(f"SHORT SL/TP from recent swings: SL={sl_level:.6f} ({(sl_level - current_price) / current_price * 100:.2f}% away), "
                   f"TP={tp_level:.6f} ({(current_price - tp_level) / current_price * 100:.2f}% away)")
    
    return sl_level, tp_level

def save_optimized_params(symbol, params):
    """Save optimized parameters for a symbol to a JSON file."""
    # Clean the symbol for filename
    filename = symbol.replace('/', '_').replace(':', '_') + '.json'
    filepath = os.path.join(PARAMS_DIR, filename)
    
    # Ensure all keys are serializable to JSON
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            serializable_params[k] = v
        else:
            # Convert numpy/pandas types to Python native types
            try:
                if hasattr(v, 'item'):
                    serializable_params[k] = v.item()  # For numpy scalars
                else:
                    serializable_params[k] = float(v)  # Last resort, convert to float
            except:
                logger.warning(f"Skipping non-serializable parameter {k} when saving")
    
    # Save to a temporary file first to prevent corruption if interrupted
    temp_filepath = filepath + '.tmp'
    with open(temp_filepath, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    
    # Move the temporary file to the final destination
    if os.path.exists(temp_filepath):
        os.replace(temp_filepath, filepath)
        # logger.info(f"Saved optimized parameters for {symbol} to {filepath}")
    
    return filepath

def load_optimized_params(symbol):
    """Load optimized parameters for a symbol with safe type conversion."""
    # Clean the symbol for filename
    filename = symbol.replace('/', '_').replace(':', '_') + '.json'
    filepath = os.path.join(PARAMS_DIR, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            # Filter parameters to only include ones used by current indicator
            valid_params = [
                'atr_window', 'vol_multiplier', 'min_vol',
                'pattern_min_reliability', 'stoch_threshold', 'smooth_period', 'min_pattern_count',
                'risk_reward_ratio', 'min_stop_atr', 'max_stop_atr', 
                'reversal_lookback', 'level_lookback', 'signal_quality_threshold',
                'trend_period', 'trend_angle_threshold', 'trend_strength_multiplier', 'trend_lookback'
            ]
            
            # Keep only valid parameters, ensure proper types
            filtered_params = {}
            for k, v in params.items():
                if k in valid_params:
                    # Convert parameters that should be integers
                    if k in ['atr_window', 'smooth_period', 'min_pattern_count', 
                             'trend_period', 'trend_lookback', 'reversal_lookback', 'level_lookback']:
                        try:
                            filtered_params[k] = int(float(v))
                        except (ValueError, TypeError):
                            # Use default value if conversion fails
                            filtered_params[k] = DEFAULT_PARAMS[k]
                            logger.warning(f"Invalid {k} value in params file: {v}, using default: {filtered_params[k]}")
                    else:
                        # Keep other parameters as is
                        filtered_params[k] = v
            
            # Ensure pattern_min_reliability is not too high - IMPORTANT CHANGE: CAP AT 40%
            if 'pattern_min_reliability' in filtered_params:
                if filtered_params['pattern_min_reliability'] > 40.0:
                    logger.info(f"Capping pattern_min_reliability from {filtered_params['pattern_min_reliability']} to 40.0")
                    filtered_params['pattern_min_reliability'] = 40.0
            
            # Add pattern_stats_file which is always needed
            filtered_params['pattern_stats_file'] = f"{symbol.replace('/', '_').replace(':', '_')}_pattern_stats.json"
            
            # Check if trend parameters are missing and add defaults if needed
            trend_params = ['trend_period', 'trend_angle_threshold', 'trend_strength_multiplier', 'trend_lookback']
            for param in trend_params:
                if param not in filtered_params:
                    filtered_params[param] = DEFAULT_PARAMS[param]
                    # logger.info(f"Added default {param}: {filtered_params[param]} to {symbol} configuration")
            
            # Verify key integer parameters
            for param in ['atr_window', 'smooth_period', 'min_pattern_count', 'trend_period', 'trend_lookback']:
                value = filtered_params.get(param)
                if not isinstance(value, int):
                    try:
                        filtered_params[param] = int(float(value))
                    except (ValueError, TypeError):
                        filtered_params[param] = int(DEFAULT_PARAMS[param])
                        logger.warning(f"Fixed non-integer {param}: {value} to {filtered_params[param]}")
                        
            return filtered_params
        except Exception as e:
            logger.exception(f"Error loading optimized parameters for {symbol}: {e}")
    
    return None

def log_trade(symbol, trade_info, force_write=False):
    """
    Log completed trade details to a CSV file with comprehensive information.
    
    Args:
        symbol: Trading pair symbol
        trade_info: Dictionary containing trade details
        force_write: If True, write to log even if the trade hasn't officially closed
    """
    try:
        # Create filename based on symbol and date
        date_str = time.strftime("%Y%m%d", time.localtime())
        filename = f"{date_str}_trades.csv"
        filepath = os.path.join(TRADE_LOG_DIR, filename)
        
        # Check if file exists to write header
        file_exists = os.path.isfile(filepath)
        
        # Add symbol to trade info
        trade_info['symbol'] = symbol
        
        # Add timestamp if not already present
        if 'timestamp' not in trade_info:
            trade_info['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Add position status if not specified
        if 'position_status' not in trade_info:
            # Default to closed for completed trades, open for others
            trade_info['position_status'] = 'Closed' if trade_info.get('exit_reason') else 'Open'
        
        # Calculate additional metrics
        if 'entry_price' in trade_info and 'stop_loss' in trade_info and 'target' in trade_info:
            if trade_info.get('position_type') == 'long':
                trade_info['stop_distance_pct'] = ((trade_info['entry_price'] - trade_info['stop_loss']) / 
                                                  trade_info['entry_price'] * 100)
                trade_info['target_distance_pct'] = ((trade_info['target'] - trade_info['entry_price']) / 
                                                    trade_info['entry_price'] * 100)
            else:  # short
                trade_info['stop_distance_pct'] = ((trade_info['stop_loss'] - trade_info['entry_price']) / 
                                                  trade_info['entry_price'] * 100)
                trade_info['target_distance_pct'] = ((trade_info['entry_price'] - trade_info['target']) / 
                                                    trade_info['entry_price'] * 100)
        
        # Write to CSV
        df = pd.DataFrame([trade_info])
        df.to_csv(filepath, mode='a', header=not file_exists, index=False)
        
        status_msg = f"[{trade_info['position_status']}]" if 'position_status' in trade_info else ""
        trend_info = f", Trend: {trade_info.get('trend_direction', 'Unknown')}" if 'trend_direction' in trade_info else ""
        
        logger.info(f"Trade logged for {symbol}: {trade_info.get('entry_reason', 'Unknown')} → "
                   f"{trade_info.get('exit_reason', 'Open')}, {status_msg} "
                   f"PnL: {trade_info.get('profit_pct', 0):.2f}%{trend_info}, "
                   f"Risk/Reward: {trade_info.get('risk_reward', 0):.2f}")
        
        # Update pattern statistics if trade closed with a pattern-based entry
        if trade_info.get('position_status') == 'Closed' and 'Pattern' in trade_info.get('entry_reason', ''):
            try:
                # Extract pattern name and normalize it to match the internal pattern format
                # E.g., convert "Bullish Engulfing Pattern" to "bullish_engulfing"
                pattern_name = trade_info['entry_reason'].replace(' Pattern', '').lower().replace(' ', '_')
                
                profit_pct = trade_info.get('profit_pct', 0)
                win = profit_pct > 0
                
                # Check if trade was with trend or against trend
                with_trend = None
                if ('trend_direction' in trade_info and 'position_type' in trade_info and 
                    'trend_trade_type' in trade_info):
                    trend_direction = trade_info['trend_direction']
                    position_type = trade_info['position_type']
                    trade_type = trade_info['trend_trade_type']
                    
                    # Determine if trade was aligned with trend
                    with_trend = (
                        (position_type == 'long' and trend_direction == 'bullish' and trade_type == 'continuation') or
                        (position_type == 'short' and trend_direction == 'bearish' and trade_type == 'continuation')
                    )
                
                with indicator_lock:
                    if symbol in symbol_indicators:
                        update_pattern_trade_result(symbol_indicators[symbol], pattern_name, profit_pct, win=win, with_trend=with_trend)
                        logger.info(f"Updated pattern statistics for {pattern_name} with {profit_pct:.2f}% result")
                
                # Add this to log entry for better tracking
                trade_info['pattern_used'] = pattern_name
                trade_info['pattern_win'] = win
            except Exception as e:
                logger.warning(f"Could not update pattern statistics: {e}")
        
        return True
    except Exception as e:
        logger.exception(f"Error logging trade for {symbol}: {e}")
        return False

def update_pattern_trade_result(indicator, pattern_name, profit_pct, win=False, with_trend=None):
    """
    Update pattern statistics with the result of a trade.
    
    Args:
        indicator: Indicator instance
        pattern_name: Name of the pattern that triggered the trade
        profit_pct: Profit percentage of the trade
        win: Whether the trade was a win
        with_trend: Whether the trade was with the trend
    """
    # Fix pattern name format (e.g., "Bullish Engulfing" to "bullish_engulfing")
    pattern_name = pattern_name.replace(' ', '_').lower()
    
    # Ensure pattern exists in stats
    if pattern_name not in indicator.indicator.pattern_stats:
        indicator.indicator.pattern_stats[pattern_name] = {
            'total': 0,
            'in_context': 0,
            'successful': 0,
            'trades': 0,
            'wins': 0,
            'avg_profit': 0,
            'is_bullish': 'bullish' in pattern_name or any(b in pattern_name for b in ['hammer', 'morning', 'bullish_harami']),
            'with_trend_trades': 0,
            'with_trend_wins': 0,
            'against_trend_trades': 0,
            'against_trend_wins': 0
        }
    
    # Update trade statistics
    stats = indicator.indicator.pattern_stats[pattern_name]
    
    # FIX: Ensure trades is incremented properly as an integer
    stats['trades'] = int(stats.get('trades', 0)) + 1
    
    # FIX: Initialize profit sum if it doesn't exist
    if 'profit_sum' not in stats:
        stats['profit_sum'] = 0.0
    
    # Update wins and profit
    if win:
        stats['wins'] = int(stats.get('wins', 0)) + 1
    
    # Update profit sum and average
    stats['profit_sum'] = float(stats.get('profit_sum', 0)) + profit_pct
    stats['avg_profit'] = stats['profit_sum'] / stats['trades'] if stats['trades'] > 0 else 0
    
    # Update trend-specific statistics if provided
    if with_trend is not None:
        if with_trend:
            # FIX: Ensure with_trend_trades is an integer
            stats['with_trend_trades'] = int(stats.get('with_trend_trades', 0)) + 1
            if win:
                stats['with_trend_wins'] = int(stats.get('with_trend_wins', 0)) + 1
        else:
            # FIX: Ensure against_trend_trades is an integer
            stats['against_trend_trades'] = int(stats.get('against_trend_trades', 0)) + 1
            if win:
                stats['against_trend_wins'] = int(stats.get('against_trend_wins', 0)) + 1
    
    # Save updated pattern stats
    indicator.indicator.save_pattern_stats()
    
    # Log the update
    logger.info(f"Updated trade statistics for pattern {pattern_name}: "
               f"Total trades: {stats['trades']}, Wins: {stats['wins']}, "
               f"Avg Profit: {stats['avg_profit']:.2f}%")
    
    return stats

def update_pattern_success(indicator, pattern_name, successful=True):
    """
    Update pattern success statistics, even when no trade is taken.
    This helps track pattern reliability independently of trading decisions.
    
    Args:
        indicator: Indicator instance
        pattern_name: Name of the pattern detected
        successful: Whether the pattern led to the expected price movement
    """
    # Fix pattern name format
    pattern_name = pattern_name.replace(' ', '_').lower()
    
    # Ensure pattern exists in stats
    if pattern_name not in indicator.indicator.pattern_stats:
        indicator.indicator.pattern_stats[pattern_name] = {
            'total': 1,  # Initialize with the current detection
            'in_context': 0,
            'successful': 0,
            'trades': 0,
            'wins': 0,
            'avg_profit': 0,
            'is_bullish': 'bullish' in pattern_name or any(b in pattern_name for b in ['hammer', 'morning', 'bullish_harami']),
            'with_trend_trades': 0,
            'with_trend_wins': 0,
            'against_trend_trades': 0,
            'against_trend_wins': 0
        }
    else:
        # Increment total detections
        indicator.indicator.pattern_stats[pattern_name]['total'] = int(indicator.indicator.pattern_stats[pattern_name].get('total', 0)) + 1
    
    # Update successful count if pattern was successful
    if successful:
        indicator.indicator.pattern_stats[pattern_name]['successful'] = int(indicator.indicator.pattern_stats[pattern_name].get('successful', 0)) + 1
    
    # Save updated pattern stats
    indicator.indicator.save_pattern_stats()
    
    return indicator.indicator.pattern_stats[pattern_name]

def reset_pattern_statistics():
    """Reset pattern statistics to fix inconsistencies."""
    pattern_stats_files = []
    
    # Get all pattern stats files
    for root, dirs, files in os.walk(PATTERN_STATS_DIR):
        for file in files:
            if file.endswith('_pattern_stats.json'):
                pattern_stats_files.append(os.path.join(root, file))
    
    # Process each file
    for filepath in pattern_stats_files:
        try:
            # Load current stats
            with open(filepath, 'r') as f:
                stats = json.load(f)
            
            # Fix each pattern's stats
            fixed_stats = {}
            for pattern, pattern_stats in stats.items():
                # Convert all numeric values to appropriate types
                fixed_pattern_stats = {
                    'total': int(pattern_stats.get('total', 0)),
                    'in_context': int(pattern_stats.get('in_context', 0)),
                    'successful': int(pattern_stats.get('successful', 0)),
                    'trades': int(pattern_stats.get('trades', 0)),
                    'wins': int(pattern_stats.get('wins', 0)),
                    'avg_profit': float(pattern_stats.get('avg_profit', 0.0)),
                    'is_bullish': bool(pattern_stats.get('is_bullish', 'bullish' in pattern)),
                    'with_trend_trades': int(pattern_stats.get('with_trend_trades', 0)),
                    'with_trend_wins': int(pattern_stats.get('with_trend_wins', 0)),
                    'against_trend_trades': int(pattern_stats.get('against_trend_trades', 0)),
                    'against_trend_wins': int(pattern_stats.get('against_trend_wins', 0)),
                    'last_reset': time.time()
                }
                
                # Fix inconsistencies
                # If with_trend_trades + against_trend_trades doesn't match trades, reset them
                total_directional_trades = fixed_pattern_stats['with_trend_trades'] + fixed_pattern_stats['against_trend_trades']
                if total_directional_trades != fixed_pattern_stats['trades'] and fixed_pattern_stats['trades'] > 0:
                    # Reset directional stats based on total trades
                    fixed_pattern_stats['with_trend_trades'] = int(fixed_pattern_stats['trades'] * 0.7)  # Assume 70% with trend
                    fixed_pattern_stats['against_trend_trades'] = fixed_pattern_stats['trades'] - fixed_pattern_stats['with_trend_trades']
                
                # If trades > 0 but wins is way off, reset wins
                win_rate = fixed_pattern_stats['wins'] / fixed_pattern_stats['trades'] if fixed_pattern_stats['trades'] > 0 else 0
                if win_rate > 0.9 or win_rate < 0.05:  # Suspiciously high or low win rate
                    # Reset to a reasonable default win rate (45%)
                    fixed_pattern_stats['wins'] = int(fixed_pattern_stats['trades'] * 0.45)
                
                # Similarly reset directional wins if inconsistent
                if fixed_pattern_stats['with_trend_wins'] > fixed_pattern_stats['with_trend_trades']:
                    fixed_pattern_stats['with_trend_wins'] = int(fixed_pattern_stats['with_trend_trades'] * 0.5)
                
                if fixed_pattern_stats['against_trend_wins'] > fixed_pattern_stats['against_trend_trades']:
                    fixed_pattern_stats['against_trend_wins'] = int(fixed_pattern_stats['against_trend_trades'] * 0.3)
                
                # If successful count is suspicious compared to total, reset it
                success_rate = fixed_pattern_stats['successful'] / fixed_pattern_stats['in_context'] if fixed_pattern_stats['in_context'] > 0 else 0
                if success_rate < 0.05 or success_rate > 0.9:
                    fixed_pattern_stats['successful'] = int(fixed_pattern_stats['in_context'] * 0.4)  # Reasonable default
                
                fixed_stats[pattern] = fixed_pattern_stats
            
            # Save fixed stats
            with open(filepath, 'w') as f:
                json.dump(fixed_stats, f, indent=4)
            
            logger.info(f"Fixed pattern statistics in {filepath}")
            
        except Exception as e:
            logger.warning(f"Error fixing pattern stats in {filepath}: {e}")
    
    return True

class BacktestEngine:
    """Backtesting engine for optimizing indicator parameters."""
    def __init__(self, df, commission=0.001, leverage=1, initial_balance=1000):
        """Initialize the backtesting engine."""
        self.df = df
        self.commission = commission
        self.leverage = leverage
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset backtest state."""
        self.balance = self.initial_balance
        self.positions = []
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, indicator_params, early_stop=True):
        """
        Run a backtest with the given indicator parameters - with safe parameter handling.
        """
        self.reset()
        
        # Sanitize parameters before initializing the indicator
        safe_params = {}
        int_params = ['atr_window', 'smooth_period', 'min_pattern_count', 
                    'trend_period', 'trend_lookback', 'reversal_lookback', 'level_lookback']
        
        for key, value in indicator_params.items():
            if key in int_params:
                try:
                    safe_params[key] = int(float(value))
                except (ValueError, TypeError):
                    safe_params[key] = DEFAULT_PARAMS.get(key, 10)  # Use default or 10 as fallback
            else:
                safe_params[key] = value
        
        # Ensure pattern_min_reliability is not too high
        if 'pattern_min_reliability' in safe_params and safe_params['pattern_min_reliability'] > 40.0:
            safe_params['pattern_min_reliability'] = 40.0
        
        # Initialize indicator with sanitized parameters
        indicator = EnhancedScalpingWrapper(**safe_params)
        
        # Compute signals with enhanced signal info
        buy_signals, sell_signals, _, signal_info = indicator.compute_signals(
            self.df['open'], 
            self.df['high'], 
            self.df['low'], 
            self.df['close']
        )
        
        # Early stopping check - if no signals in first 100 bars, return early
        if early_stop:
            if len(buy_signals) >= 100 and not buy_signals.iloc[:100].any() and not sell_signals.iloc[:100].any():
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'final_balance': self.initial_balance,
                    'total_return': 0,
                    'total_profit': 0
                }
        
        # Apply signals to generate trades
        position = None
        min_holding_periods = 3
        holding_counter = 0
        position_stop_loss = None
        position_target = None
        position_signal_info = None
        
        for i in range(1, len(self.df)):
            row = self.df.iloc[i]
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            timestamp = row.name if hasattr(row, 'name') else i
            
            # Get trend information if available
            trend_direction = 'flat'
            trend_change = False
            
            if i < len(signal_info) and 'trend_direction' in signal_info.columns:
                trend_direction = signal_info['trend_direction'].iloc[i]
                
                if i > 1:
                    prev_trend = signal_info['trend_direction'].iloc[i-1]
                    if prev_trend != trend_direction:
                        trend_change = True
            
            # Record equity curve using close price
            self.equity_curve.append(self.balance + (position['size'] * close_price if position else 0))
            
            # Early stopping check for poor performance
            if early_stop and i > 200 and len(self.trades) >= 5:
                # If we've taken 5+ trades and losing money, stop early
                if self.balance < self.initial_balance * 0.9:
                    # Calculate metrics for what we have so far
                    return self.calculate_metrics()
            
            # Handle exits
            if position:
                # Update holding counter
                holding_counter += 1
                
                # Check exit conditions
                exit_conditions_met = False
                exit_reason = "No exit"
                exit_price = close_price  # Default exit price
                
                if holding_counter >= min_holding_periods:
                    # Check if price hit stop loss or target - use intrabar price for more accurate testing
                    if position['direction'] == 'long':
                        # For long positions, check low price for stop loss and high price for target
                        if low_price <= position_stop_loss:
                            exit_conditions_met = True
                            exit_reason = "Stop Loss"
                            exit_price = max(low_price, position_stop_loss)  # Use stop price if hit intrabar
                        elif high_price >= position_target:
                            exit_conditions_met = True
                            exit_reason = "Target Reached"
                            exit_price = min(high_price, position_target)  # Use target price if hit intrabar
                    else:  # short position
                        # For short positions, check high price for stop loss and low price for target
                        if high_price >= position_stop_loss:
                            exit_conditions_met = True
                            exit_reason = "Stop Loss"
                            exit_price = min(high_price, position_stop_loss)  # Use stop price if hit intrabar
                        elif low_price <= position_target:
                            exit_conditions_met = True
                            exit_reason = "Target Reached"
                            exit_price = max(low_price, position_target)  # Use target price if hit intrabar
                    
                    # Check for trend change exit
                    if trend_change:
                        if (position['direction'] == 'long' and trend_direction == 'bearish') or \
                        (position['direction'] == 'short' and trend_direction == 'bullish'):
                            exit_conditions_met = True
                            exit_reason = f"Trend Change to {trend_direction}"
                            exit_price = close_price
                    
                    # Check for reversal patterns in opposite direction
                    if not exit_conditions_met and i < len(signal_info) and 'pattern_type' in signal_info.columns:
                        # Get current row in signal_info
                        current_signal = signal_info.iloc[i]
                        pattern_type = ""
                        pattern_reliability = 0
                        
                        # Safely extract pattern type and reliability
                        if 'pattern_type' in current_signal and not pd.isna(current_signal['pattern_type']):
                            pattern_type = current_signal['pattern_type']
                        
                        if 'pattern_reliability' in current_signal and not pd.isna(current_signal['pattern_reliability']):
                            pattern_reliability = current_signal['pattern_reliability']
                        
                        # Check if the pattern is opposite to our position
                        is_bearish_pattern = pattern_type in [
                            'shooting_star', 'hanging_man', 'bearish_engulfing', 
                            'evening_star', 'bearish_harami', 'tweezer_top'
                        ]
                        
                        is_bullish_pattern = pattern_type in [
                            'hammer', 'inverted_hammer', 'bullish_engulfing', 
                            'morning_star', 'bullish_harami', 'tweezer_bottom'
                        ]
                        
                        # Exit long positions on bearish patterns with sufficient reliability
                        if position['direction'] == 'long' and is_bearish_pattern and pattern_reliability >= 50:
                            exit_conditions_met = True
                            exit_reason = f"Reversal Pattern ({pattern_type})"
                            exit_price = close_price
                        
                        # Exit short positions on bullish patterns with sufficient reliability
                        elif position['direction'] == 'short' and is_bullish_pattern and pattern_reliability >= 50:
                            exit_conditions_met = True
                            exit_reason = f"Reversal Pattern ({pattern_type})"
                            exit_price = close_price
                
                if exit_conditions_met:
                    # Calculate profit/loss
                    pnl = 0
                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry_price']) / position['entry_price']
                    else:  # short
                        pnl = (position['entry_price'] - exit_price) / position['entry_price']
                    
                    # Apply leverage
                    pnl *= self.leverage
                    
                    # Apply commission
                    pnl -= self.commission * 2  # Entry and exit commission
                    
                    # Update balance
                    trade_profit = position['size'] * pnl
                    self.balance += trade_profit
                    
                    # Calculate risk metrics
                    if position['direction'] == 'long':
                        stop_distance_pct = ((position['entry_price'] - position_stop_loss) / 
                                            position['entry_price'] * 100)
                        target_distance_pct = ((position_target - position['entry_price']) / 
                                            position['entry_price'] * 100)
                    else:  # short
                        stop_distance_pct = ((position_stop_loss - position['entry_price']) / 
                                            position['entry_price'] * 100)
                        target_distance_pct = ((position['entry_price'] - position_target) / 
                                            position['entry_price'] * 100)
                    
                    # Extract trend information if available
                    entry_trend_direction = 'flat'
                    trend_trade_type = 'unknown'
                    
                    # Safely get trend info from position data
                    if isinstance(position, dict):
                        if 'trend_direction' in position:
                            entry_trend_direction = position['trend_direction']
                        if 'trend_trade_type' in position:
                            trend_trade_type = position['trend_trade_type']
                    
                    # Record trade with enhanced information
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'stop_loss': position_stop_loss,
                        'target': position_target,
                        'stop_distance_pct': stop_distance_pct,
                        'target_distance_pct': target_distance_pct,
                        'size': position['size'],
                        'profit': trade_profit,
                        'pnl_pct': pnl * 100,
                        'holding_periods': holding_counter,
                        'exit_reason': exit_reason,
                        'entry_reason': position.get('entry_reason', 'Unknown'),
                        'pattern_used': position.get('pattern_type', None),
                        'trend_direction': entry_trend_direction,
                        'trend_trade_type': trend_trade_type,
                        'with_trend': (
                            (position['direction'] == 'long' and entry_trend_direction == 'bullish' and trend_trade_type == 'continuation') or
                            (position['direction'] == 'short' and entry_trend_direction == 'bearish' and trend_trade_type == 'continuation')
                        ) if entry_trend_direction != 'flat' and trend_trade_type != 'unknown' else None
                    })
                    
                    # Clear position
                    position = None
                    position_stop_loss = None
                    position_target = None
                    position_signal_info = None
                    holding_counter = 0
            
            # Handle entries - only if we don't have a position
            if not position:
                # Skip flat trends for entries
                if trend_direction == 'flat':
                    continue
                    
                # Check if this candle has a valid entry signal
                has_buy_signal = False
                has_sell_signal = False
                
                if i < len(buy_signals):
                    has_buy_signal = buy_signals.iloc[i]
                
                if i < len(sell_signals):
                    has_sell_signal = sell_signals.iloc[i]
                
                if has_buy_signal or has_sell_signal:
                    # Get signal information for this bar
                    current_signal_info = {}
                    if i < len(signal_info):
                        current_signal_info = signal_info.iloc[i].to_dict()
                    
                    # Ensure we don't have conflicting signals 
                    if has_buy_signal and has_sell_signal:
                        # If somehow both signals are True, use the one with higher probability
                        if 'signal_type' in current_signal_info:
                            has_buy_signal = current_signal_info['signal_type'] == 'buy'
                            has_sell_signal = current_signal_info['signal_type'] == 'sell'
                        else:
                            # Default to buy if we can't determine
                            has_buy_signal = True
                            has_sell_signal = False
                    
                    # Determine direction
                    direction = 'long' if has_buy_signal else 'short'
                    
                    # Skip signals that don't align with trend
                    reason = current_signal_info.get('reason', '').lower()
                    if (direction == 'long' and trend_direction == 'bearish' and 'reversal' not in reason) or \
                    (direction == 'short' and trend_direction == 'bullish' and 'reversal' not in reason):
                        # Skip continuation signals against trend
                        continue
                    
                    # Calculate position size (10% of balance)
                    position_size = self.balance * 0.1
                    
                    # Get pattern information if available
                    pattern_type = current_signal_info.get('pattern_type', '')
                    
                    # Determine trade type (continuation or reversal)
                    trend_trade_type = 'unknown'
                    if trend_direction != 'flat' and pattern_type:
                        is_bullish_pattern = pattern_type in [
                            'hammer', 'inverted_hammer', 'bullish_engulfing', 
                            'morning_star', 'bullish_harami', 'tweezer_bottom'
                        ]
                        
                        if (is_bullish_pattern and trend_direction == 'bullish') or \
                        (not is_bullish_pattern and trend_direction == 'bearish'):
                            trend_trade_type = 'continuation'
                        else:
                            trend_trade_type = 'reversal'
                    
                    # Create position with enhanced information
                    position = {
                        'direction': direction,
                        'entry_price': open_price,  # Use open price for more realistic entry
                        'size': position_size,
                        'entry_time': timestamp,
                        'entry_reason': current_signal_info.get('reason', 'Unknown'),
                        'probability': float(current_signal_info.get('probability', 0.5)),
                        'pattern_type': pattern_type,
                        'trend_direction': trend_direction,
                        'trend_trade_type': trend_trade_type
                    }
                    
                    # Set stop loss and target from signal info
                    stop_loss = None
                    target = None
                    
                    if 'stop_loss' in current_signal_info and not pd.isna(current_signal_info['stop_loss']):
                        stop_loss = current_signal_info['stop_loss']
                    
                    if 'target' in current_signal_info and not pd.isna(current_signal_info['target']):
                        target = current_signal_info['target']
                    
                    # Default stop loss if not provided in signal info
                    if stop_loss is None:
                        stop_loss = open_price * (0.99 if direction == 'long' else 1.01)
                    
                    # Default target if not provided in signal info
                    if target is None:
                        stop_distance = abs(open_price - stop_loss)
                        target = open_price + (stop_distance * 2) if direction == 'long' else open_price - (stop_distance * 2)
                    
                    position_stop_loss = stop_loss
                    position_target = target
                    position_signal_info = current_signal_info
                    holding_counter = 0
        
        # Close any open position at the end
        if position:
            exit_price = self.df['close'].iloc[-1]
            pnl = 0
            if position['direction'] == 'long':
                pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:  # short
                pnl = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Apply leverage and commission
            pnl = pnl * self.leverage - self.commission * 2
            
            # Update balance
            trade_profit = position['size'] * pnl
            self.balance += trade_profit
            
            # Determine exit reason
            exit_reason = "End of Backtest"
            
            # Extract trend information if available
            entry_trend_direction = position.get('trend_direction', 'flat')
            trend_trade_type = position.get('trend_trade_type', 'unknown')
            
            # Record trade with enhanced information
            self.trades.append({
                'entry_time': position['entry_time'],
                'exit_time': self.df.index[-1] if hasattr(self.df.index, '__getitem__') else len(self.df) - 1,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'size': position['size'],
                'profit': trade_profit,
                'pnl_pct': pnl * 100,
                'holding_periods': holding_counter,
                'exit_reason': exit_reason,
                'entry_reason': position.get('entry_reason', 'Unknown'),
                'pattern_used': position.get('pattern_type', ''),
                'trend_direction': entry_trend_direction,
                'trend_trade_type': trend_trade_type,
                'with_trend': (
                    (position['direction'] == 'long' and entry_trend_direction == 'bullish' and trend_trade_type == 'continuation') or
                    (position['direction'] == 'short' and entry_trend_direction == 'bearish' and trend_trade_type == 'continuation')
                ) if entry_trend_direction != 'flat' and trend_trade_type != 'unknown' else None
            })
        
        # Calculate metrics
        results = self.calculate_metrics()
        return results
    
    def calculate_metrics(self):
        """Calculate performance metrics from the backtest results."""
        total_trades = len(self.trades)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_balance': self.initial_balance,
                'total_return': 0,
                'total_profit': 0
            }
        
        winning_trades = len([t for t in self.trades if t['profit'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = sum(t['profit'] for t in self.trades) / total_trades
        
        # Calculate profit factor
        gross_profit = sum(t['profit'] for t in self.trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t['profit'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        if len(self.equity_curve) > 0:
            equity_array = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak
            max_drawdown = abs(min(drawdown)) * 100
        else:
            max_drawdown = 0
        
        # Calculate Sharpe ratio
        if len(self.trades) > 1:
            returns = np.array([t['pnl_pct'] for t in self.trades])
            sharpe_ratio = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Calculate pattern performance metrics
        pattern_performance = {}
        for trade in self.trades:
            pattern = trade.get('pattern_used')
            if pattern:
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = {
                        'count': 0,
                        'win_count': 0,
                        'total_profit': 0
                    }
                
                stats = pattern_performance[pattern]
                stats['count'] += 1  # Ensure this is an integer increment
                if trade['profit'] > 0:
                    stats['win_count'] += 1  # Ensure this is an integer increment
                stats['total_profit'] += trade['pnl_pct']
        
        # Calculate win rates and avg profit per pattern
        for pattern, stats in pattern_performance.items():
            stats['win_rate'] = stats['win_count'] / stats['count'] if stats['count'] > 0 else 0
            stats['avg_profit'] = stats['total_profit'] / stats['count'] if stats['count'] > 0 else 0
        
        final_balance = self.balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_profit': final_balance - self.initial_balance,
            'pattern_performance': pattern_performance
        }

def import_backtest_pattern_performance(indicator, backtest_results):
    """
    Import pattern performance statistics from backtest results to pattern stats.
    This ensures backtest pattern analysis is incorporated into live trading decisions.
    
    Args:
        indicator: Indicator instance
        backtest_results: Dictionary containing backtest metrics including pattern_performance
    """
    if 'pattern_performance' not in backtest_results or not backtest_results['pattern_performance']:
        logger.info("No pattern performance data to import from backtest")
        return
        
    pattern_performance = backtest_results['pattern_performance']
    logger.info(f"Importing pattern performance for {len(pattern_performance)} patterns from backtest")
    
    # Process each pattern
    for pattern, stats in pattern_performance.items():
        if not pattern or pattern == "":
            continue
            
        # Normalize pattern name (ensure underscores)
        pattern_name = pattern.replace(' ', '_').lower()
        
        # Initialize pattern stats if not exists
        if pattern_name not in indicator.indicator.pattern_stats:
            indicator.indicator.pattern_stats[pattern_name] = {
                'total': 0,
                'in_context': 0,
                'successful': 0,
                'trades': 0,
                'wins': 0,
                'avg_profit': 0,
                'is_bullish': 'bullish' in pattern_name or any(b in pattern_name for b in ['hammer', 'morning', 'bullish_harami']),
                'with_trend_trades': 0,
                'with_trend_wins': 0,
                'against_trend_trades': 0,
                'against_trend_wins': 0
            }
            
        # Get existing stats
        existing_stats = indicator.indicator.pattern_stats[pattern_name]
        
        # Update with backtest results - ensure values are proper integers
        # Use weighted average for incorporating new data
        total_trades = int(existing_stats.get('trades', 0)) + int(stats.get('count', 0))
        
        if total_trades > 0:
            # Calculate weighted win rate
            existing_wins = int(existing_stats.get('wins', 0))
            new_wins = int(stats.get('win_count', 0))
            total_wins = existing_wins + new_wins
            
            # Update trades and wins as integers
            existing_stats['trades'] = total_trades
            existing_stats['wins'] = total_wins
            
            # Update profit metrics
            total_profit = (existing_stats.get('avg_profit', 0) * int(existing_stats.get('trades', 0)) + 
                           stats.get('total_profit', 0))
            existing_stats['avg_profit'] = total_profit / total_trades if total_trades > 0 else 0
            
            # Increment the pattern's successful count based on win rate
            success_increment = int(stats.get('win_count', 0))
            existing_stats['successful'] = int(existing_stats.get('successful', 0)) + success_increment
            
            # Log the update
            logger.info(f"Updated pattern stats for {pattern_name} from backtest: "
                       f"Total trades now {existing_stats['trades']}, "
                       f"Wins now {existing_stats['wins']}, "
                       f"Win rate {(existing_stats['wins']/existing_stats['trades']*100):.1f}%, "
                       f"Avg profit {existing_stats['avg_profit']:.2f}%")
    
    # Save the updated stats
    indicator.indicator.save_pattern_stats()
    
    return indicator.indicator.pattern_stats

def optimize_indicator_for_symbol(symbol, exchange, max_evals=MAX_EVALS, candles=BACKTEST_CANDLES):
    """Run hyperopt optimization for a symbol and return optimized parameters."""
    try:
        start_time = time.time()
        # logger.info(f"Starting optimization for {symbol} with {max_evals} evaluations")
        
        # Fetch historical data for optimization
        df = fetch_binance_data(exchange, symbol, timeframe='3m', limit=candles)
        if len(df) < candles // 2:
            logger.warning(f"Insufficient data for {symbol} optimization")
            return None
        
        # Define parameter space for reversal-based indicator
        space = {
                # Core volatility parameters
                'atr_window': hp.quniform('atr_window', 3, 10, 1),
                'vol_multiplier': hp.uniform('vol_multiplier', 0.8, 2.0),
                'min_vol': hp.loguniform('min_vol', np.log(0.0005), np.log(0.002)),
                
                # Risk/reward parameters
                'risk_reward_ratio': hp.uniform('risk_reward_ratio', 1.2, 3.0),
                'min_stop_atr': hp.uniform('min_stop_atr', 0.2, 1.0),
                'max_stop_atr': hp.uniform('max_stop_atr', 0.8, 2.5),
                
                # Pattern detection parameters - RELAXED
                'pattern_min_reliability': hp.uniform('pattern_min_reliability', 25.0, 40.0), # Reduced from 30-70 to 25-40
                'stoch_threshold': hp.uniform('stoch_threshold', 60.0, 90.0),
                'smooth_period': hp.quniform('smooth_period', 10, 30, 1),
                'min_pattern_count': hp.quniform('min_pattern_count', 5, 20, 1),
                
                # NEW: Trend analysis parameters
                'trend_period': hp.quniform('trend_period', 10, 30, 1),
                'trend_angle_threshold': hp.uniform('trend_angle_threshold', 3.0, 10.0),
                'trend_strength_multiplier': hp.uniform('trend_strength_multiplier', 1.0, 1.5),
                'trend_lookback': hp.quniform('trend_lookback', 3, 10, 1)
            }
        
        # Initialize backtest engines with different leverages for robust testing
        leverage_tests = LEVERAGE_OPTIONS
        backtest_results = []
        for lev in leverage_tests:
            backtest = BacktestEngine(df, commission=0.001, leverage=lev)
            backtest_results.append(backtest)
        
        def objective(params):
            """Objective function for hyperopt to minimize."""
            # Process parameters - convert numeric types
            processed_params = {}
            for key, value in params.items():
                if key in ['atr_window', 'smooth_period', 'min_pattern_count']:
                    processed_params[key] = int(value)
                else:
                    processed_params[key] = value
            
            # Set symbol-specific pattern stats file
            processed_params['pattern_stats_file'] = f"{symbol.replace('/', '_').replace(':', '_')}_pattern_stats.json"
            
            # Run multiple backtests with different leverage settings
            multi_scores = []
            for backtest in backtest_results:
                results = backtest.run_backtest(processed_params, early_stop=True)
                
                # Skip if no trades (invalid parameter set)
                if results['total_trades'] < 5:
                    continue
                
                # Score based on pattern performance and overall results
                # Prioritize pattern detection reliability
                
                # Base score on win rate and profit factor
                score = results['win_rate'] * 0.4 + min(5, results['profit_factor']) * 0.2
                
                # Add bonus for pattern performance
                if 'pattern_performance' in results and results['pattern_performance']:
                    pattern_win_rates = [stats['win_rate'] for _, stats in results['pattern_performance'].items()]
                    if pattern_win_rates:
                        avg_pattern_win_rate = sum(pattern_win_rates) / len(pattern_win_rates)
                        pattern_bonus = avg_pattern_win_rate * 0.5  # 50% bonus for good pattern performance
                        score += pattern_bonus
                
                # Add score components for risk-adjusted returns
                score += min(5, results['sharpe_ratio']) * 0.2
                
                # Penalize excessive drawdown
                max_acceptable_dd = 15
                if results['max_drawdown'] > max_acceptable_dd:
                    dd_penalty = 1.0 - min(1.0, (results['max_drawdown'] - max_acceptable_dd) / 30)
                    score *= dd_penalty
                
                multi_scores.append(score)
            
            # If no valid scores (all parameter sets failed), return a very poor score
            if not multi_scores:
                return {'loss': 100, 'status': STATUS_OK}
            
            # Calculate average score across all test conditions
            avg_score = sum(multi_scores) / len(multi_scores)
            
            # Return negative score since hyperopt minimizes
            return {'loss': -avg_score, 'status': STATUS_OK}
        
        # Run hyperopt optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            show_progressbar=False
        )
        
        # Process the best parameters
        best_params = {}
        for key, value in best.items():
            if key in ['atr_window', 'smooth_period', 'min_pattern_count']:
                best_params[key] = int(value)
            else:
                best_params[key] = value
        
        # Ensure pattern_min_reliability is not too high
        if 'pattern_min_reliability' in best_params and best_params['pattern_min_reliability'] > 40.0:
            best_params['pattern_min_reliability'] = 40.0
        
        # Set symbol-specific pattern stats file
        best_params['pattern_stats_file'] = f"{symbol.replace('/', '_').replace(':', '_')}_pattern_stats.json"
        
        # Validate best parameters with a final backtest
        final_backtest = BacktestEngine(df, commission=0.001, leverage=10)
        final_results = final_backtest.run_backtest(best_params, early_stop=False)
        
        # Import pattern performance from backtest
        # Create an indicator with the best parameters
        temp_indicator = EnhancedScalpingWrapper(**best_params)
        
        # Import pattern performance from backtest
        import_backtest_pattern_performance(temp_indicator, final_results)
        
        # Log optimization statistics
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # logger.info(f"Optimization complete for {symbol} in {optimization_time:.1f} seconds")
        # logger.info(f"Best parameters: {', '.join([f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in best_params.items()])}")
        logger.info(f"Performance: Total Return: {final_results['total_return']:.2f}%, "
                  f"Win Rate: {final_results['win_rate']:.2%}, "
                  f"Sharpe: {final_results['sharpe_ratio']:.2f}, "
                  f"Trades: {final_results['total_trades']}")
        
        # Log pattern performance if available
        if 'pattern_performance' in final_results and final_results['pattern_performance']:
            pattern_stats = [(pattern, stats['win_rate'], stats['avg_profit']) 
                            for pattern, stats in final_results['pattern_performance'].items()]
            pattern_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by win rate
            logger.info("Pattern Performance:")
            for pattern, win_rate, avg_profit in pattern_stats:
                logger.info(f"  {pattern}: Win Rate: {win_rate:.2%}, Avg Profit: {avg_profit:.2f}%")
        
        return best_params
    
    except Exception as e:
        logger.exception(f"Error during optimization for {symbol}: {e}")
        return None

def optimization_worker(exchange):
    """
    Worker thread function that processes the optimization queue.
    This runs independently of the main trading loop.
    """
    # logger.info("Optimization worker thread started")
    
    while True:
        try:
            # Check if there's a task in the queue
            try:
                # Non-blocking queue get with timeout
                symbol = optimization_queue.get(timeout=1)
                # logger.info(f"Optimization worker processing {symbol}")
                
                # Run optimization for the symbol
                params = optimize_indicator_for_symbol(symbol, exchange)
                
                if params:
                    # Save the parameters to file
                    save_optimized_params(symbol, params)
                    
                    # Update the indicator instance
                    with indicator_lock:
                        symbol_indicators[symbol] = EnhancedScalpingWrapper(**params)
                        last_optimization[symbol] = time.time()
                    
                    # Store results for the main thread to use
                    with optimization_lock:
                        optimization_results[symbol] = params
                    
                    # logger.info(f"Optimization complete for {symbol}, stored results")
                else:
                    logger.warning(f"Optimization failed for {symbol}")
            
            except queue.Empty:
                # No tasks in queue, sleep for a while
                time.sleep(OPTIMIZATION_THREAD_SLEEP)
                continue
            
        except Exception as e:
            logger.exception(f"Error in optimization worker: {e}")
            time.sleep(OPTIMIZATION_THREAD_SLEEP)

def get_or_optimize_indicator(symbol, exchange, force_optimize=False):
    """Get an optimized indicator for a symbol or create one with default parameters."""
    current_time = time.time()
    needs_optimization = False
    
    with indicator_lock:
        # Check if we should optimize
        should_optimize = (
            force_optimize or
            symbol not in symbol_indicators or
            (symbol in last_optimization and current_time - last_optimization.get(symbol, 0) > OPTIMIZATION_INTERVAL)
        )
        
        if should_optimize:
            # Try to load optimized parameters from file
            params = None if force_optimize else load_optimized_params(symbol)
            
            if params is None:
                # Need to schedule optimization
                needs_optimization = True
            else:
                # Configure symbol-specific pattern stats file
                params['pattern_stats_file'] = f"{symbol.replace('/', '_').replace(':', '_')}_pattern_stats.json"
                
                # Use loaded parameters
                symbol_indicators[symbol] = EnhancedScalpingWrapper(**params)
                last_optimization[symbol] = current_time
                # logger.info(f"Using loaded parameters for {symbol}")
    
    # Schedule optimization if needed (outside the lock to reduce contention)
    if needs_optimization:
        # logger.info(f"Scheduling optimization for {symbol}")
        optimization_queue.put(symbol)
        
        # Configure symbol-specific pattern stats file with trend params
        default_params = DEFAULT_PARAMS.copy()
        default_params['pattern_stats_file'] = f"{symbol.replace('/', '_').replace(':', '_')}_pattern_stats.json"
        
        # Use default parameters until optimization completes
        with indicator_lock:
            symbol_indicators[symbol] = EnhancedScalpingWrapper(**default_params)
    
    # Return the indicator (either existing, loaded, or default)
    with indicator_lock:
        return symbol_indicators.get(symbol) or EnhancedScalpingWrapper(**DEFAULT_PARAMS)

def create_exchange():
    """Create and return a CCXT exchange object for Binance futures."""
    return ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'timeout': 30000
    })

def fetch_binance_data(exchange, market_id, timeframe=TIMEFRAME, limit=LIMIT, include_current=True):
    """
    Fetch OHLCV data from Binance with caching and proper timestamp handling.
    Modified to clearly separate current incomplete candle from closed candles.
    
    Args:
        exchange: Exchange object
        market_id: Trading pair symbol
        timeframe: Candle timeframe
        limit: Number of candles to fetch
        include_current: Whether to include the current (incomplete) candle
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
                return ohlcv_cache[cache_key]['data']
        
        # Fetch new data - increase the limit to ensure we have enough after processing
        actual_fetch_limit = limit * 2  # Double the requested limit to account for potential losses
        ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=actual_fetch_limit)
        original_len = len(ohlcv)
        
        if original_len < limit:
            logger.warning(f"Exchange only returned {original_len} candles for {market_id} {timeframe}, requested {actual_fetch_limit}")
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        after_df_creation = len(df)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Check for duplicate timestamps BEFORE setting index
        duplicates = df['timestamp'].duplicated()
        if duplicates.any():
            duplicate_count = duplicates.sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamps in data for {market_id} {timeframe}")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        after_dedup = len(df)
        
        # Now set the timestamp as index after removing duplicates
        df.set_index('timestamp', inplace=True)
        
        # Convert columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values and log any losses
        nan_count = df.isna().any(axis=1).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} rows with NaN values in data for {market_id} {timeframe}")
        
        df.dropna(inplace=True)
        after_dropna = len(df)
        
        # Verify the index is still unique after all processing
        if df.index.duplicated().any():
            logger.warning(f"Post-processing duplicates found in {market_id} {timeframe} - fixing")
            df = df[~df.index.duplicated(keep='first')]
        
        after_unique_index = len(df)
        
        # NEW: Mark which candle is the current incomplete one
        # Use millisecond timestamps for comparison
        df['is_current_candle'] = False
        if len(df) > 0:
            # Get the timeframe in milliseconds
            tf_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
            
            # Calculate the start time of the current candle
            current_candle_start = int(current_time * 1000) - (int(current_time * 1000) % tf_ms)
            
            # Convert index to milliseconds for comparison
            df['timestamp_ms'] = df.index.astype(int) // 10**6
            
            # Mark the current candle
            df['is_current_candle'] = df['timestamp_ms'] >= current_candle_start
            
            # Remove the helper column
            df.drop('timestamp_ms', axis=1, inplace=True)
        
        # If requested, remove the most recent (potentially incomplete) candle
        if not include_current and len(df) > 0:
            df = df[~df['is_current_candle']]
        
        final_len = len(df)
        
        # Log data processing losses if significant
        if final_len < original_len * 0.9:  # More than 10% data loss
            logger.warning(f"Significant data loss during processing for {market_id} {timeframe}: " +
                         f"Original: {original_len}, DataFrame: {after_df_creation}, After dedup: {after_dedup}, " +
                         f"After NaN removal: {after_dropna}, After index check: {after_unique_index}, Final: {final_len}")
        
        # Return only the requested number of candles (from the end)
        if len(df) > limit:
            df = df.iloc[-limit:]
        
        # Update cache with thread safety
        with cache_lock:
            ohlcv_cache[cache_key] = {
                'data': df,
                'timestamp': current_time
            }
        
        return df
    except Exception as e:
        logger.exception(f"Failed to fetch data for {market_id}: {e}")
        return pd.DataFrame()

def get_trend_info(signal_info, closed_candles_only=True):
    """
    Extract trend information from signal info, optionally filtering 
    out the current incomplete candle for trend change detection.
    
    Args:
        signal_info: DataFrame containing trend information
        closed_candles_only: Whether to only consider closed candles for trend changes
        
    Returns:
        Dictionary containing trend information
    """
    trend_info = {}
    
    if 'trend_direction' in signal_info.columns:
        # If we need to filter out the current candle for trend detection
        if closed_candles_only and 'is_current_candle' in signal_info.columns:
            # Use only closed candles for accurate trend detection
            closed_candles = signal_info[~signal_info['is_current_candle']]
            
            # Make sure we have at least 2 closed candles to detect a trend change
            if len(closed_candles) >= 2:
                latest_trend = closed_candles['trend_direction'].iloc[-1]
                trend_angle = closed_candles['trend_angle'].iloc[-1] if 'trend_angle' in closed_candles.columns else 0
                trend_strength = closed_candles['trend_strength'].iloc[-1] if 'trend_strength' in closed_candles.columns else 0
                trend_duration = closed_candles['trend_duration'].iloc[-1] if 'trend_duration' in closed_candles.columns else 0
                
                # Check for trend change in recent CLOSED candles only
                prev_trend = closed_candles['trend_direction'].iloc[-2]
                trend_change = prev_trend != latest_trend
                
                trend_info = {
                    'direction': latest_trend,
                    'angle': trend_angle,
                    'strength': trend_strength,
                    'duration': trend_duration,
                    'change': trend_change,
                    'prev_direction': prev_trend
                }
                
                # For logging purposes, include current candle trend direction
# For logging purposes, include current candle trend direction
                if len(signal_info) > 0:
                    current_trend = signal_info['trend_direction'].iloc[-1]
                    trend_info['current_candle_direction'] = current_trend
                    
                    # Log if current candle suggests a different trend than closed candles
                    if current_trend != latest_trend:
                        logger.info(f"Current incomplete candle shows {current_trend} trend, but closed candles show {latest_trend} trend")
            else:
                # Not enough closed candles, use all data
                latest_trend = signal_info['trend_direction'].iloc[-1]
                trend_info = {
                    'direction': latest_trend,
                    'angle': signal_info['trend_angle'].iloc[-1] if 'trend_angle' in signal_info.columns else 0,
                    'strength': signal_info['trend_strength'].iloc[-1] if 'trend_strength' in signal_info.columns else 0,
                    'duration': signal_info['trend_duration'].iloc[-1] if 'trend_duration' in signal_info.columns else 0,
                    'change': False,  # Not enough data to detect change
                }
        else:
            # Original logic using all candles including current incomplete one
            latest_trend = signal_info['trend_direction'].iloc[-1]
            trend_angle = signal_info['trend_angle'].iloc[-1] if 'trend_angle' in signal_info.columns else 0
            trend_strength = signal_info['trend_strength'].iloc[-1] if 'trend_strength' in signal_info.columns else 0
            trend_duration = signal_info['trend_duration'].iloc[-1] if 'trend_duration' in signal_info.columns else 0
            trend_change = False
            
            # Check for trend change in recent candles
            if len(signal_info) > 2:
                prev_trend = signal_info['trend_direction'].iloc[-2]
                if prev_trend != latest_trend:
                    trend_change = True
            
            trend_info = {
                'direction': latest_trend,
                'angle': trend_angle,
                'strength': trend_strength,
                'duration': trend_duration,
                'change': trend_change
            }
    
    return trend_info

def get_position_details(symbol, entry_price, position_type, position_entry_times):
    """
    Safely retrieve position details without blocking on trend info.
    Separates core position data from trend data to prevent exit blocking.
    
    Args:
        symbol: The trading symbol
        entry_price: The position entry price
        position_type: 'long' or 'short'
        position_entry_times: Dictionary of entry times
        
    Returns:
        Dictionary with position details
    """
    try:
        # Use a quick-release lock pattern - acquire lock only for dictionary access
        with position_details_lock:
            # Check if we have existing details
            if symbol in position_details:
                # Return a copy to avoid reference issues
                return position_details[symbol].copy()
        
        # If not in dictionary, create default values outside the lock
        default_details = {
            'entry_price': entry_price,
            'stop_loss': entry_price * (0.997 if position_type == 'long' else 1.003),  # Default 0.3% stop
            'target': entry_price * (1.006 if position_type == 'long' else 0.994),     # Default 0.6% target
            'position_type': position_type,
            'entry_reason': 'Unknown',
            'probability': 0.5,
            'entry_time': position_entry_times.get(symbol, time.time()),
            'highest_reached': entry_price if position_type == 'long' else None,
            'lowest_reached': entry_price if position_type == 'short' else None,
            'pattern_used': None,
            'trend_direction': 'unknown',  # Neutral default - updated separately
            'trend_trade_type': 'unknown'  # Neutral default - updated separately
        }
        
        # Store the default values with minimal lock time
        with position_details_lock:
            if symbol not in position_details:
                position_details[symbol] = default_details
            return position_details[symbol].copy()
    
    except Exception as e:
        # If anything fails, log and return safe defaults that won't block exits
        logger.warning(f"Error retrieving position details for {symbol}: {e}")
        return {
            'entry_price': entry_price,
            'stop_loss': entry_price * (0.997 if position_type == 'long' else 1.003),
            'target': entry_price * (1.006 if position_type == 'long' else 0.994),
            'position_type': position_type,
            'entry_time': time.time() - 3600,  # Assume position is 1 hour old as safe default
            'highest_reached': entry_price if position_type == 'long' else None,
            'lowest_reached': entry_price if position_type == 'short' else None,
        }

def update_position_trend_info(symbol, trend_info):
    """
    Separately update trend information in position details.
    This ensures trend updates don't block exit logic.
    
    Args:
        symbol: The trading symbol
        trend_info: Dictionary with trend information
    """
    try:
        # Quick update with minimal lock time
        with position_details_lock:
            if symbol in position_details:
                position_details[symbol]['current_trend'] = trend_info.get('direction', 'unknown')
                position_details[symbol]['trend_change'] = trend_info.get('change', False)
    except Exception as e:
        logger.warning(f"Failed to update trend info for {symbol}: {e}")
        # Don't block on trend

def fetch_active_symbols(exchange):
    """Fetch active trading symbols from the exchange."""
    try:
        ticker_data = exchange.fetch_tickers()
        markets = exchange.load_markets()
        
        active_markets = [
            symbol for symbol, market in markets.items()
            if market.get('settle') == 'USDT' and market.get('swap') and 'BTC' not in symbol and 'ETH' not in symbol
        ]
        
        top_symbols = sorted(
            [symbol for symbol in active_markets if symbol in ticker_data],
            key=lambda x: ticker_data[x].get('quoteVolume', 0),
            reverse=True
        )[:50]
        
        return top_symbols
    
    except Exception as e:
        logger.exception(f"Error fetching active symbols: {e}")
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

# Replace the existing place_order function with this improved version

def place_order(exchange, market_id, side, quantity, price, leverage=None, exit_order=False, 
               stop_loss=None, target_price=None, place_target_order=True, skip_cooldown=False):
    """
    Place an order on the exchange with automatic stop-loss and take-profit orders.
    Improved with better error handling and validation of SL/TP levels.
    
    Args:
        exchange: Exchange object
        market_id: Trading pair symbol
        side: 'buy' or 'sell'
        quantity: Order quantity
        price: Current price level (used for logging)
        leverage: Leverage to use
        exit_order: Whether this is an exit order
        stop_loss: Price level for stop loss
        target_price: Price target for take profit
        place_target_order: Whether to place stop-loss and take-profit orders
        skip_cooldown: Whether to skip the cooldown period after exit (for reversals)
    """
    params = {}
    
    # Add reduceOnly parameter for exit orders
    if exit_order:
        params['reduceOnly'] = True
    
    try:
        exchange.options['defaultType'] = 'future' if '_' in market_id else 'swap'
        
        # Validate SL/TP levels before placing order
        if not exit_order and place_target_order and (stop_loss is not None and target_price is not None):
            # Get current price for validation
            ticker = exchange.fetch_ticker(market_id)
            latest_price = ticker['last'] if 'last' in ticker else ticker['close']
            
            # Check if SL is too close to current price
            min_sl_distance_pct = 0.008  # 0.8% minimum for crypto
            
            if side == 'buy':  # LONG position
                sl_distance_pct = (latest_price - stop_loss) / latest_price
                tp_distance_pct = (target_price - latest_price) / latest_price
                
                if sl_distance_pct < min_sl_distance_pct:
                    logger.warning(f"SL for {market_id} is too close to price ({sl_distance_pct * 100:.2f}%) - adjusting to {min_sl_distance_pct * 100:.1f}%")
                    stop_loss = latest_price * (1 - min_sl_distance_pct)
                
                if tp_distance_pct < min_sl_distance_pct:
                    logger.warning(f"TP for {market_id} is too close to price ({tp_distance_pct * 100:.2f}%) - adjusting to {min_sl_distance_pct * 100 * 1.5:.1f}%")
                    target_price = latest_price * (1 + min_sl_distance_pct * 1.5)
            else:  # SHORT position
                sl_distance_pct = (stop_loss - latest_price) / latest_price
                tp_distance_pct = (latest_price - target_price) / latest_price
                
                if sl_distance_pct < min_sl_distance_pct:
                    logger.warning(f"SL for {market_id} is too close to price ({sl_distance_pct * 100:.2f}%) - adjusting to {min_sl_distance_pct * 100:.1f}%")
                    stop_loss = latest_price * (1 + min_sl_distance_pct)
                
                if tp_distance_pct < min_sl_distance_pct:
                    logger.warning(f"TP for {market_id} is too close to price ({tp_distance_pct * 100:.2f}%) - adjusting to {min_sl_distance_pct * 100 * 1.5:.1f}%")
                    target_price = latest_price * (1 - min_sl_distance_pct * 1.5)
        
        # Step 1: Place the main market order for entry
        market_order = exchange.create_order(market_id, 'MARKET', side, quantity, params=params)
        logger.info(f"Market order placed for {market_id}: {side} {quantity}")
        
        # Track entry time for minimum hold logic
        if not exit_order:
            with cooldown_lock:
                position_entry_times[market_id] = time.time()
            logger.info(f"Position entry time for {market_id} recorded: {time.strftime('%H:%M:%S', time.localtime(position_entry_times[market_id]))}")
            
            # Step 2: Place stop-loss and take-profit orders if requested
            if place_target_order and stop_loss is not None and target_price is not None:
                try:
                    # Determine the opposite side for SL/TP orders
                    inverted_side = 'sell' if side == 'buy' else 'buy'
                    
                    # Double check current position to get exact amount
                    try:
                        positions = exchange.fetch_positions([market_id])
                        exact_position = next((p for p in positions if p['symbol'] == market_id and abs(float(p['info'].get('positionAmt', 0))) > 0), None)
                        
                        if exact_position:
                            exact_qty = abs(float(exact_position['info'].get('positionAmt', 0)))
                            if abs(exact_qty - quantity) / quantity > 0.01:  # If difference > 1%
                                logger.info(f"Adjusting order quantity for SL/TP from {quantity} to {exact_qty}")
                                quantity = exact_qty
                    except Exception as pos_error:
                        logger.warning(f"Could not fetch exact position size for {market_id}: {pos_error}")
                    
                    # Wait a moment to ensure order is processed
                    time.sleep(1)
                    
                    # For stop-loss orders when LONG, we use STOP_MARKET with the selling side
                    # For stop-loss orders when SHORT, we use STOP_MARKET with the buying side
                    sl_params = {
                        'stopPrice': stop_loss,     # Trigger price for the stop order
                        'closePosition': 'true',     # Close the entire position
                        'timeInForce':'GTE_GTC'
                    }
                    
                    # Log SL order before placing
                    logger.info(f"Placing SL order for {market_id} at {stop_loss:.8f} ({inverted_side} {quantity})")
                    
                    sl_order = exchange.create_order(
                        market_id, 
                        'STOP_MARKET', 
                        inverted_side, 
                        quantity, 
                        None,  # Price not needed for STOP_MARKET
                        params=sl_params
                    )
                    logger.info(f"Stop-loss order placed for {market_id} at {stop_loss}")
                    
                    # Wait briefly between orders to avoid rate limiting
                    time.sleep(0.5)
                    
                    # For take-profit orders when LONG, we use TAKE_PROFIT_MARKET with the selling side
                    # For take-profit orders when SHORT, we use TAKE_PROFIT_MARKET with the buying side
                    tp_params = {
                        'stopPrice': target_price,  # Trigger price for the take profit
                        'closePosition': 'true',     # Close the entire position
                        'timeInForce':'GTE_GTC'
                    }
                    
                    # Log TP order before placing
                    logger.info(f"Placing TP order for {market_id} at {target_price:.8f} ({inverted_side} {quantity})")
                    
                    tp_order = exchange.create_order(
                        market_id, 
                        'TAKE_PROFIT_MARKET', 
                        inverted_side, 
                        quantity, 
                        None,  # Price not needed for TAKE_PROFIT_MARKET
                        params=tp_params
                    )
                    logger.info(f"Take-profit order placed for {market_id} at {target_price}")
                    
                    # Store order IDs for tracking
                    with position_details_lock:
                        if market_id in position_details:
                            position_details[market_id]['sl_order_id'] = sl_order.get('id')
                            position_details[market_id]['tp_order_id'] = tp_order.get('id')
                    
                except Exception as sl_tp_error:
                    err_msg = str(sl_tp_error)
                    
                    # Check for specific error about order immediately triggering
                    if "immediately trigger" in err_msg:
                        logger.warning(f"SL/TP would immediately trigger for {market_id} - retry with adjusted levels")
                        
                        # Get latest price
                        ticker = exchange.fetch_ticker(market_id)
                        latest_price = ticker['last'] if 'last' in ticker else ticker['close']
                        
                        # Recalculate SL/TP with safe distances
                        if side == 'buy':  # Long position
                            new_sl = latest_price * 0.985  # 1.5% below current price
                            new_tp = latest_price * 1.03   # 3% above current price
                        else:  # Short position
                            new_sl = latest_price * 1.015  # 1.5% above current price
                            new_tp = latest_price * 0.97   # 3% below current price
                        
                        logger.info(f"Retrying with safer levels - SL: {new_sl:.8f}, TP: {new_tp:.8f}")
                        
                        try:
                            # Wait a moment before retrying
                            time.sleep(1)
                            
                            # Retry SL
                            sl_params = {
                                'stopPrice': new_sl,
                                'closePosition': 'true',
                                'timeInForce': 'GTE_GTC'
                            }
                            sl_order = exchange.create_order(
                                market_id,
                                'STOP_MARKET',
                                inverted_side,
                                quantity,
                                None,
                                params=sl_params
                            )
                            logger.info(f"Stop-loss order placed on retry for {market_id} at {new_sl}")
                            
                            # Wait briefly
                            time.sleep(0.5)
                            
                            # Retry TP
                            tp_params = {
                                'stopPrice': new_tp,
                                'closePosition': 'true',
                                'timeInForce': 'GTE_GTC'
                            }
                            tp_order = exchange.create_order(
                                market_id,
                                'TAKE_PROFIT_MARKET',
                                inverted_side,
                                quantity,
                                None,
                                params=tp_params
                            )
                            logger.info(f"Take-profit order placed on retry for {market_id} at {new_tp}")
                            
                            # Update position details
                            with position_details_lock:
                                if market_id in position_details:
                                    position_details[market_id]['sl_order_id'] = sl_order.get('id')
                                    position_details[market_id]['tp_order_id'] = tp_order.get('id')
                                    position_details[market_id]['stop_loss'] = new_sl
                                    position_details[market_id]['target'] = new_tp
                            
                        except Exception as retry_error:
                            logger.error(f"Failed to place SL/TP on retry for {market_id}: {retry_error}")
                    else:
                        logger.warning(f"Failed to place SL/TP orders for {market_id}: {sl_tp_error}")
                        logger.warning("Check Binance API documentation for correct parameters: https://binance-docs.github.io/apidocs/futures/en/#new-order-trade")
                    # Continue execution even if SL/TP orders fail - the main position is already open
        else:
            # Set cooldown after exit
            if not skip_cooldown:
                with cooldown_lock:
                    symbol_cooldowns[market_id] = time.time() + SYMBOL_COOLDOWN_MINUTES * 60
                    if market_id in position_entry_times:
                        del position_entry_times[market_id]  # Clear entry time
                        
                logger.info(f"Setting cooldown for {market_id} until: {time.strftime('%H:%M:%S', time.localtime(symbol_cooldowns[market_id]))}")
            else:
                logger.info(f"Skipping cooldown for {market_id} due to reversal signal")
                
        return market_order
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific Binance error codes for notional value and reduceOnly issues
        if "reduceOnly" in error_msg or "notional" in error_msg or "Order's notional" in error_msg:
            logger.warning(f"Order rejected due to size constraints for {market_id}: {e}")
            logger.info(f"Trying again with reduceOnly flag and exact position size")
            
            try:
                # Get the exact position size from the exchange
                positions = exchange.fetch_positions([market_id])
                for pos in positions:
                    if pos['symbol'] == market_id and abs(float(pos['info'].get('positionAmt', 0))) > 0:
                        exact_qty = abs(float(pos['info'].get('positionAmt', 0)))
                        logger.info(f"Retrying with exact position size: {exact_qty} for {market_id}")
                        
                        # Try again with exact quantity and reduceOnly flag
                        params['reduceOnly'] = True
                        retry_order = exchange.create_order(market_id, 'MARKET', side, exact_qty, params=params)
                        
                        logger.info(f"Successfully closed position for {market_id} on retry")
                        
                        # Set cooldown after successful exit
                        if not skip_cooldown:
                            with cooldown_lock:
                                symbol_cooldowns[market_id] = time.time() + SYMBOL_COOLDOWN_MINUTES * 60
                                if market_id in position_entry_times:
                                    del position_entry_times[market_id]
                        else:
                            logger.info(f"Skipping cooldown for {market_id} on retry due to reversal signal")
                        
                        return retry_order
                
                logger.error(f"Could not find exact position size for {market_id}")
                return None
            except Exception as retry_error:
                logger.exception(f"Failed to close position on retry for {market_id}: {retry_error}")
                return None
        else:
            logger.exception(f"Failed to place order for {market_id}: {e}")
            return None
        
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

def is_market_suitable_for_scalping(symbol, exchange):
    """Check if the market has suitable conditions for scalping."""
    try:
        # Get recent volatility - fetch more candles (30 -> 50)
        df = fetch_binance_data(exchange, symbol, timeframe='1m', limit=50)
        if len(df) < 20:
            logger.warning(f"{symbol} has insufficient candle data: {len(df)} candles")
            return False
            
        # Calculate quick volatility measure (max range as % of price)
        mean_close = df['close'].mean()
        if mean_close <= 0 or pd.isna(mean_close):
            logger.warning(f"{symbol} has invalid price data - mean close: {mean_close}")
            return False
            
        price_range_pct = ((df['high'].max() - df['low'].min()) / mean_close) * 100
        
        # Check volume stability - with safe division
        volume_data = df['volume'].replace(0, np.nan).dropna()
        if len(volume_data) < 10 or volume_data.sum() < 1e-10:
            logger.warning(f"{symbol} has insufficient volume data ({len(volume_data)} valid points)")
            return False
            
        volume_mean = volume_data.mean()
        volume_std = volume_data.std()
        
        # Check for zero or near-zero mean volume
        if volume_mean <= 1e-10:
            logger.warning(f"{symbol} has near-zero mean volume: {volume_mean}")
            return False
            
        # Safe calculation with non-zero mean
        volume_stability = volume_std / volume_mean
        
        # Market is suitable if:
        # 1. Price range is between 0.2% and 5% in the last 30 minutes
        # 2. Volume is relatively stable (std/mean < 1.5)
        is_suitable = (0.2 <= price_range_pct <= 5.0) and (volume_stability < 1.5)
        
        # if is_suitable:
        #     logger.info(f"{symbol} looks suitable for scalping: price range {price_range_pct:.2f}%, vol stability {volume_stability:.2f}")
        
        return is_suitable
    except Exception as e:
        logger.warning(f"Error checking scalping suitability for {symbol}: {e}")
        return False

def is_in_cooldown(symbol):
    """Check if a symbol is in cooldown period after a recent trade."""
    with cooldown_lock:
        if symbol in symbol_cooldowns:
            cooldown_until = symbol_cooldowns[symbol]
            current_time = time.time()
            
            if current_time < cooldown_until:
                remaining = int((cooldown_until - current_time) / 60)
                # logger.info(f"{symbol} in cooldown for {remaining} more minutes")
                return True
            else:
                # Cooldown expired
                del symbol_cooldowns[symbol]
    
    return False

def detect_multi_timeframe_patterns(exchange, symbol, timeframes=None, override_min_reliability=25.0):
    """
    Enhanced function to detect patterns across multiple timeframes with proper candle alignment.
    Updated to require multiple candle confirmation and momentum consistency.
    
    Args:
        exchange: Exchange object
        symbol: Trading pair symbol
        timeframes: List of timeframes to check (default: ['15m', '3m', '5m'])
        override_min_reliability: Override the default minimum reliability threshold (default: 25.0%)
    
    Returns:
        Tuple of (best_pattern, pattern_reliability, multi_tf_patterns, pattern_detections, pattern_sequence)
    """
    if timeframes is None:
        timeframes = ['15m', '3m', '5m']
    
    # Get or create the indicator instance
    try:
        indicator = get_or_optimize_indicator(symbol, exchange)
        
        # Get the min_reliability threshold from the indicator but apply a cap
        min_reliability_threshold = indicator.indicator.pattern_min_reliability
        
        # If the threshold is too high, replace it with our override value
        if min_reliability_threshold > 40.0:
            logger.warning(f"Original min_reliability_threshold is too high ({min_reliability_threshold:.2f}%). "
                          f"Using override value of {override_min_reliability:.2f}% instead.")
            min_reliability_threshold = override_min_reliability
        else:
            logger.info(f"Using pattern reliability threshold of {min_reliability_threshold:.2f}%")
    except Exception as e:
        logger.exception(f"Error getting indicator for {symbol}: {e}")
        # Use default threshold if we couldn't get the indicator
        min_reliability_threshold = override_min_reliability
        logger.info(f"Using default reliability threshold of {min_reliability_threshold:.2f}%")
        return None, 0.0, [], [], None
    
    # Fetch aligned data across timeframes
    multi_tf_data = fetch_aligned_multi_timeframe_data(exchange, symbol, timeframes, include_current=False)
    
    # Check if we have enough data in each timeframe
    valid_timeframes = []
    for tf, df in multi_tf_data.items():
        if len(df) >= 20:  # Minimum candles for reliable patterns
            valid_timeframes.append(tf)
        else:
            logger.warning(f"Insufficient data for {symbol} on {tf}: only {len(df)} candles, need at least 20")
    
    # Check if we have enough valid timeframes to continue
    if len(valid_timeframes) < 1:
        logger.warning(f"No valid timeframes with sufficient data for {symbol}, skipping pattern detection")
        return None, 0.0, [], [], None
    
    # Log the number of valid timeframes
    logger.info(f"Detecting patterns for {symbol} across {len(valid_timeframes)} valid timeframes: {valid_timeframes}")
    
    pattern_detections = []
    reliability_map = {}  # Track pattern reliabilities
    
    # Track pattern sequence for each timeframe
    pattern_sequences = {}
    
    # Process each timeframe
    for tf, df_live in multi_tf_data.items():
        # Skip if this timeframe doesn't have enough data
        if tf not in valid_timeframes:
            continue
            
        try:
            # Process indicators for pattern detection
            df_processed = df_live.copy()
            df_processed = indicator.indicator._calculate_basic_indicators(df_processed)
            df_processed = indicator.indicator._detect_stochastic_context(df_processed)
            
            # Use the updated pattern detection method
            df_processed = indicator.indicator._detect_candlestick_patterns(df_processed)
            
            # NEW: Check for pattern sequence in last 3 candles (not just the latest one)
            recent_patterns = []
            for i in range(min(3, len(df_processed))):
                idx = df_processed.index[-(i+1)]
                row = df_processed.iloc[-(i+1)]
                
                if row.get('has_pattern', False) and not pd.isna(row.get('pattern_type', '')) and row.get('pattern_type', '') != '':
                    # Add pattern with its index (for tracking sequence)
                    recent_patterns.append({
                        'index': idx,
                        'pattern': row['pattern_type'],
                        'reliability': row['pattern_reliability'],
                        'confirmed': row.get('pattern_confirmed', False),
                        'is_neutral': row.get('pattern_is_neutral', False) or 'neutral_' in row['pattern_type'],
                        'trend_direction': row.get('trend_direction', 'flat'),
                        'position': i  # 0 = latest, 1 = previous, 2 = two candles ago
                    })
            
            # Store this timeframe's pattern sequence
            pattern_sequences[tf] = recent_patterns
            
            # Continue with existing pattern detection - but use most reliable pattern from recent candles
            if recent_patterns:
                # Sort by reliability (highest first)
                recent_patterns.sort(key=lambda x: x['reliability'], reverse=True)
                best_recent = recent_patterns[0]
                
                # Process the best recent pattern
                pattern_name = best_recent['pattern']
                base_reliability = best_recent['reliability']
                is_confirmed = best_recent['confirmed']
                is_neutral = best_recent['is_neutral']
                trend_direction = best_recent['trend_direction']
                
                # Determine if this is a bullish, bearish, or neutral pattern
                if is_neutral:
                    # This is a neutral pattern (indecision)
                    signal_type = 'neutral'
                else:
                    # Check if bullish based on pattern name
                    is_bullish = any(bullish_name in pattern_name for bullish_name in [
                        'hammer', 'inverted_hammer', 'bullish', 'morning_star', 
                        'bullish_harami', 'tweezer_bottom', 'three_white_soldiers',
                        'dragonfly_doji', 'piercing', 'stick_sandwich', 'takuri', 
                        'ladder_bottom', 'mat_hold', 'morning_doji_star', 'homing_pigeon',
                        'matching_low', 'unique_three_river', 'three_stars_in_south'
                    ])
                    signal_type = 'buy' if is_bullish else 'sell'
                
                # Enhance reliability with win rate if available
                enhanced_reliability = base_reliability
                pattern_stats = indicator.indicator.pattern_stats.get(pattern_name, {})
                
                # Calculate win rate component if we have trade data
                if pattern_stats.get('trades', 0) >= 5:  # Need minimum sample size
                    win_rate = pattern_stats.get('wins', 0) / pattern_stats.get('trades', 1) * 100 if pattern_stats.get('trades', 0) > 0 else 0
                    # Weighted average of base reliability and win rate with higher weight on win rate
                    enhanced_reliability = (base_reliability * 0.4) + (win_rate * 0.6)
                    logger.debug(f"Enhanced reliability for {pattern_name}: Base {base_reliability:.1f}%, "
                               f"Win rate {win_rate:.1f}%, Enhanced {enhanced_reliability:.1f}%")
                
                # Add context information to reliability with special handling for neutral patterns
                if is_neutral:
                    # Neutral patterns don't get as much context boost
                    context_boost = 1.2 if is_confirmed else 1.0
                else:
                    # Normal context boost for directional patterns
                    context_boost = 1.5 if is_confirmed else 1.0
                
                adjusted_reliability = enhanced_reliability * context_boost
                
                # Get trend information to check if pattern aligns with trend
                # Adjust reliability for neutral patterns based on market condition
                if is_neutral and trend_direction != 'flat':
                    # Neutral patterns are less reliable in trending markets
                    adjusted_reliability *= 0.9
                
                # NEW: Check for pattern sequence consistency
                if len(recent_patterns) > 1:
                    # Analyze pattern sequence for consistency
                    has_consistent_direction = True
                    
                    # For bullish signals, check if previous patterns support bullish bias
                    if signal_type == 'buy':
                        for p in recent_patterns[1:]:  # Skip first (already processed)
                            p_is_bullish = any(bullish_name in p['pattern'] for bullish_name in [
                                'hammer', 'inverted_hammer', 'bullish', 'morning_star', 
                                'bullish_harami', 'tweezer_bottom', 'three_white_soldiers'
                            ])
                            p_is_neutral = p['is_neutral']
                            # If we find an opposing bearish pattern, mark as inconsistent
                            if not p_is_bullish and not p_is_neutral:
                                has_consistent_direction = False
                                break
                    
                    # For bearish signals, check if previous patterns support bearish bias
                    elif signal_type == 'sell':
                        for p in recent_patterns[1:]:  # Skip first (already processed)
                            p_is_bearish = any(bearish_name in p['pattern'] for bearish_name in [
                                'shooting_star', 'hanging_man', 'bearish', 'evening_star',
                                'bearish_harami', 'tweezer_top', 'three_black_crows'
                            ])
                            p_is_neutral = p['is_neutral']
                            # If we find an opposing bullish pattern, mark as inconsistent
                            if not p_is_bearish and not p_is_neutral:
                                has_consistent_direction = False
                                break
                    
                    # Boost reliability for consistent sequences, reduce for inconsistent ones
                    if has_consistent_direction:
                        adjusted_reliability *= 1.2  # 20% boost for consistency
                        logger.debug(f"Boosting {pattern_name} reliability for consistent sequence: {adjusted_reliability:.1f}%")
                    else:
                        adjusted_reliability *= 0.8  # 20% reduction for inconsistency
                        logger.debug(f"Reducing {pattern_name} reliability for inconsistent sequence: {adjusted_reliability:.1f}%")
                
                # Store detection info even if below threshold - we might adjust thresholds later
                pattern_detections.append({
                    'timeframe': tf,
                    'pattern': pattern_name,
                    'reliability': adjusted_reliability,
                    'base_reliability': base_reliability,
                    'win_rate': pattern_stats.get('wins', 0) / pattern_stats.get('trades', 1) * 100 if pattern_stats.get('trades', 0) > 0 else 0,
                    'trades_count': pattern_stats.get('trades', 0),
                    'signal_type': signal_type,
                    'confirmed': is_confirmed,
                    'is_neutral': is_neutral,
                    'trend_direction': trend_direction,
                    'candle_position': best_recent['position'],  # Track which candle this pattern is from
                    'has_consistent_sequence': has_consistent_direction if len(recent_patterns) > 1 else None
                })
                
                # Log pattern detection with comparison to threshold
                pattern_type = "neutral" if is_neutral else signal_type
                threshold_status = "above threshold" if adjusted_reliability >= min_reliability_threshold else "below threshold"
                logger.info(f"Detected {pattern_name} pattern ({pattern_type}) on {tf} for {symbol}: "
                          f"{adjusted_reliability:.1f}% ({threshold_status})")
                
                # Update reliability map for this pattern
                if pattern_name not in reliability_map:
                    reliability_map[pattern_name] = []
                reliability_map[pattern_name].append(adjusted_reliability)
        
        except Exception as e:
            logger.warning(f"Error processing {tf} timeframe for {symbol}: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")
    
    # Check if we found any patterns (regardless of threshold)
    if not pattern_detections:
        logger.info(f"No patterns detected for {symbol} across {len(valid_timeframes)} timeframes")
        return None, 0.0, [], [], None
    
    # Log total pattern detections
    logger.info(f"Detected {len(pattern_detections)} patterns for {symbol} across {len(valid_timeframes)} timeframes")
    
    # Sort pattern detections by reliability
    pattern_detections.sort(key=lambda x: x['reliability'], reverse=True)
    
    # Calculate average reliability for each pattern across timeframes
    avg_reliability_map = {}
    for pattern, reliabilities in reliability_map.items():
        if reliabilities:  # Ensure non-empty list
            avg_reliability_map[pattern] = sum(reliabilities) / len(reliabilities)
    
    # Group patterns by type
    neutral_patterns = [p for p in pattern_detections if p.get('is_neutral', False)]
    directional_patterns = [p for p in pattern_detections if not p.get('is_neutral', False)]
    
    # Use different threshold for neutral patterns in flat markets
    valid_patterns = []
    for p in pattern_detections:
        is_neutral = p.get('is_neutral', False)
        trend_is_flat = p.get('trend_direction', 'flat') == 'flat'
        
        # Lower threshold for neutral patterns in flat market
        if is_neutral and trend_is_flat:
            pattern_threshold = min_reliability_threshold * 0.85  # 15% reduction
        else:
            pattern_threshold = min_reliability_threshold
            
        # Check if pattern meets threshold
        if p['reliability'] >= pattern_threshold:
            valid_patterns.append(p)
    
    # Count valid patterns for multi-timeframe confirmation
    pattern_counts = Counter([p['pattern'] for p in valid_patterns if p['pattern'] != ''])
    multi_tf_patterns = [pattern for pattern, count in pattern_counts.items() if count >= 1]
    
    # Show stats on patterns above/below threshold
    total_patterns = len(pattern_detections)
    valid_pattern_count = len(valid_patterns)
    neutral_count = len(neutral_patterns)
    logger.info(f"Pattern threshold {min_reliability_threshold:.1f}%: {valid_pattern_count}/{total_patterns} patterns qualify "
              f"({neutral_count} neutral patterns)")
    
    # Find the best pattern based on reliability (among those meeting the threshold)
    best_pattern = None
    best_reliability = 0.0
    
    if valid_patterns:
        # Just use the first (highest reliability) pattern that meets the threshold
        best_pattern = valid_patterns[0]['pattern']
        best_reliability = valid_patterns[0]['reliability']
    
    # Log the best pattern
    if best_pattern:
        # Determine if pattern is neutral
        is_neutral_pattern = any(p.get('is_neutral', False) for p in valid_patterns if p['pattern'] == best_pattern)
        pattern_type = "neutral" if is_neutral_pattern else next((p['signal_type'] for p in valid_patterns if p['pattern'] == best_pattern), "unknown")
        
        logger.info(f"Best pattern for {symbol}: {best_pattern} ({pattern_type}) with {best_reliability:.1f}% reliability")
        if multi_tf_patterns:
            logger.info(f"Multi-timeframe confirmations for {symbol}: {multi_tf_patterns}")
    
    return best_pattern, best_reliability, multi_tf_patterns, pattern_detections, pattern_sequences

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

def fetch_aligned_multi_timeframe_data(exchange, symbol, timeframes, include_current=True):
    """
    Fetch data for multiple timeframes and ensure proper alignment of complete candles.
    
    Args:
        exchange: The exchange object
        symbol: Trading pair symbol
        timeframes: List of timeframes to fetch
        include_current: Whether to include the current (incomplete) candle
        
    Returns:
        Dictionary of dataframes for each timeframe with aligned data
    """
    result = {}
    current_time = int(time.time() * 1000)  # Current time in milliseconds
    
    # Get the smallest timeframe to determine the base time unit
    smallest_tf = min(timeframes, key=lambda x: ccxt.Exchange.parse_timeframe(x))
    smallest_tf_ms = ccxt.Exchange.parse_timeframe(smallest_tf) * 1000  # in milliseconds
    
    # Calculate common end time for all timeframes
    for tf in timeframes:
        # Parse timeframe to milliseconds
        tf_ms = ccxt.Exchange.parse_timeframe(tf) * 1000
        
        # Calculate completed candle timestamp
        # Find the most recent candle close time that has already passed
        latest_complete_ts = current_time - (current_time % tf_ms)
        
        # If we don't want the current candle, step back one more period
        if not include_current:
            latest_complete_ts -= tf_ms
        
        # Fetch enough candles - INCREASE LIMIT FOR MORE DATA
        limit = 200  # Increased from 100 to 200 for more historical data
        df = fetch_binance_data(exchange, symbol, timeframe=tf, limit=limit, include_current=include_current)
        
        # Log the fetch results
        # logger.info(f"Fetched {len(df)} candles for {symbol} on {tf} timeframe")
        
        # Ensure dataframe has timestamps
        if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
            # Convert pandas timestamp to milliseconds for comparison
            df['timestamp_ms'] = df.index.astype(int) // 10**6
            
            # Filter to only include candles before or at the latest complete timestamp
            df_filtered = df[df['timestamp_ms'] <= latest_complete_ts].copy()
            
            # Log any data losses from alignment
            # logger.info(f"Alignment filtered from {len(df)} to {len(df_filtered)} candles for {symbol} on {tf}")
            
            # Drop the helper column
            df_filtered.drop('timestamp_ms', axis=1, inplace=True)
            
            if len(df_filtered) > 20:  # Reduced from 30 to 20 minimum candles
                result[tf] = df_filtered
            else:
                logger.warning(f"Insufficient aligned candles for {symbol} on {tf}: only {len(df_filtered)} after alignment")
        else:
            logger.warning(f"Invalid data format for {symbol} on {tf} timeframe")
    
    # Log the final result of alignment
    #logger.info(f"Aligned data for {symbol}: {', '.join([f'{tf}: {len(data)}' for tf, data in result.items()])}")
    
    return result

def load_existing_parameters():
    """Load all existing optimized parameters at startup."""
    # logger.info("Loading existing optimized parameters...")
    for filename in os.listdir(PARAMS_DIR):
        if filename.endswith('.json') and not filename.endswith('_meta.json'):
            symbol = filename.replace('_', '/').replace('.json', '')
            params = load_optimized_params(symbol.replace('/', '_').replace(':', '_'))
            if params:
                # Configure symbol-specific pattern stats file
                params['pattern_stats_file'] = f"{symbol.replace('/', '_').replace(':', '_')}_pattern_stats.json"
                
                with indicator_lock:
                    symbol_indicators[symbol] = EnhancedScalpingWrapper(**params)
                    last_optimization[symbol] = time.time()
                # logger.info(f"Loaded existing parameters for {symbol}")

def check_for_optimization_needs(exchange):
    """Check if any symbols need optimization and queue them."""
    # logger.info("Checking for symbols that need optimization...")
    current_time = time.time()
    
    # First, get active symbols
    active_symbols = fetch_active_symbols(exchange)[:10]  # Limit to top 10 symbols
    
    with indicator_lock:
        # Check which symbols need optimization
        for symbol in active_symbols:
            if (symbol not in symbol_indicators or 
                symbol not in last_optimization or 
                current_time - last_optimization.get(symbol, 0) > OPTIMIZATION_INTERVAL):
                
                # logger.info(f"Scheduling optimization for {symbol}")
                optimization_queue.put(symbol)

def sync_positions_with_exchange(exchange):
    """
    Synchronize our local position tracking with actual exchange positions.
    This is called at startup to handle any discrepancies after restart.
    Also checks for existing SL/TP orders for open positions.
    """
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
            
            # Try to check for existing SL/TP orders
            try:
                orders = exchange.fetch_open_orders(symbol)
                sl_order = next((o for o in orders if o.get('type') == 'STOP_MARKET' and o.get('params', {}).get('reduceOnly', False)), None)
                tp_order = next((o for o in orders if o.get('type') == 'TAKE_PROFIT_MARKET' and o.get('params', {}).get('reduceOnly', False)), None)
                
                stop_loss = sl_order['stopPrice'] if sl_order else (entry_price * (0.98 if position_type == 'long' else 1.02))
                target = tp_order['stopPrice'] if tp_order else (entry_price * (1.04 if position_type == 'long' else 0.96))
                
                # If we found SL/TP orders, log them
                if sl_order or tp_order:
                    logger.info(f"Found existing SL/TP orders for {symbol}: SL={stop_loss if sl_order else 'None'}, TP={target if tp_order else 'None'}")
            except Exception as e:
                logger.warning(f"Failed to check for existing SL/TP orders for {symbol}: {e}")
                # Default values if we couldn't retrieve orders
                stop_loss = entry_price * (0.98 if position_type == 'long' else 1.02)
                target = entry_price * (1.04 if position_type == 'long' else 0.96)
            
            # Create default trade details
            with position_details_lock:
                position_details[symbol] = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'position_type': position_type,
                    'entry_reason': 'Recovery',
                    'probability': 0.5,
                    'entry_time': time.time() - 3600,  # Assume 1 hour ago
                    'highest_reached': entry_price if position_type == 'long' else None,
                    'lowest_reached': entry_price if position_type == 'short' else None,
                    'sl_order_id': sl_order['id'] if sl_order else None,
                    'tp_order_id': tp_order['id'] if tp_order else None
                }
                
                # Log the recovered position
                trade_log = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'entry_reason': 'Recovery',
                    'position_type': position_type,
                    'position_status': 'Open (Recovery)',
                    'exit_managed_by': 'Exchange SL/TP Orders' if (sl_order or tp_order) else 'Unknown'
                }
                log_trade(symbol, trade_log, force_write=True)
        
        logger.info(f"Position synchronization complete. Found {len(exchange_symbols)} active positions on exchange.")
                  
    except Exception as e:
        logger.exception(f"Error synchronizing positions: {e}")

def generate_pattern_statistics_report():
    """Generate a comprehensive pattern statistics report."""
    report = "PATTERN STATISTICS REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Get all pattern stats files
    pattern_files = [f for f in os.listdir(PATTERN_STATS_DIR) if f.endswith('_pattern_stats.json')]
    
    if not pattern_files:
        return "No pattern statistics available yet."
    
    # Global pattern stats from all symbols
    all_patterns = {
        'bullish': defaultdict(lambda: {'total': 0, 'in_context': 0, 'successful': 0, 'trades': 0, 'wins': 0}),
        'bearish': defaultdict(lambda: {'total': 0, 'in_context': 0, 'successful': 0, 'trades': 0, 'wins': 0})
    }
    
    # Process each symbol's pattern stats
    for pattern_file in pattern_files:
        try:
            filepath = os.path.join(PATTERN_STATS_DIR, pattern_file)
            
            with open(filepath, 'r') as f:
                stats = json.load(f)
            
            # Aggregate stats for each pattern
            for pattern, pattern_stats in stats.items():
                # Skip if missing key information
                if 'total' not in pattern_stats or pattern_stats['total'] == 0:
                    continue
                
                # Categorize as bullish or bearish
                category = 'bullish' if pattern_stats.get('is_bullish', True) else 'bearish'
                
                # Add to global stats
                all_patterns[category][pattern]['total'] += pattern_stats.get('total', 0)
                all_patterns[category][pattern]['in_context'] += pattern_stats.get('in_context', 0)
                all_patterns[category][pattern]['successful'] += pattern_stats.get('successful', 0)
                all_patterns[category][pattern]['trades'] += pattern_stats.get('trades', 0)
                all_patterns[category][pattern]['wins'] += pattern_stats.get('wins', 0)
        
        except Exception as e:
            logger.warning(f"Error processing pattern stats for {pattern_file}: {e}")
    
    # Generate report for bullish patterns
    report += "BULLISH PATTERNS\n"
    report += "-" * 80 + "\n"
    report += f"{'Pattern':<20} {'Count':<10} {'Context %':<12} {'Win %':<10} {'Success %':<12}\n"
    report += "-" * 80 + "\n"
    
    for pattern, stats in sorted(all_patterns['bullish'].items()):
        context_pct = stats['in_context'] / stats['total'] * 100 if stats['total'] > 0 else 0
        win_pct = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        success_pct = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        report += f"{pattern:<20} {stats['total']:<10} {context_pct:.2f}%{'':<8} {win_pct:.2f}%{'':<6} {success_pct:.2f}%\n"
    
    # Generate report for bearish patterns
    report += "\nBEARISH PATTERNS\n"
    report += "-" * 80 + "\n"
    report += f"{'Pattern':<20} {'Count':<10} {'Context %':<12} {'Win %':<10} {'Success %':<12}\n"
    report += "-" * 80 + "\n"
    
    for pattern, stats in sorted(all_patterns['bearish'].items()):
        context_pct = stats['in_context'] / stats['total'] * 100 if stats['total'] > 0 else 0
        win_pct = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        success_pct = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        report += f"{pattern:<20} {stats['total']:<10} {context_pct:.2f}%{'':<8} {win_pct:.2f}%{'':<6} {success_pct:.2f}%\n"
    
    return report

def metrics_reporting_thread():
    """Thread function to periodically report metrics."""
    while True:
        try:
            # Run funnel analysis every hour
            log_trade_funnel()
            # Sleep for an hour
            time.sleep(3600)
        except Exception as e:
            logger.exception(f"Error in metrics reporting: {e}")
            time.sleep(300)  # Sleep for 5 minutes on error

def continuous_loop(exchange):
    """
    Main trading loop that processes market data, generates signals, and manages positions.
    Rewritten to use more relaxed entry criteria to allow more trades.
    """
    timeframes = ['15m', '3m', '5m']  # Multiple timeframes for pattern confirmation
    last_optimization_check = time.time()

    # Initialize pattern checks dictionary for tracking success rates
    if not hasattr(continuous_loop, 'pattern_checks'):
        continuous_loop.pattern_checks = {}

    while True:
        try:
            current_time = time.time()
            
            # Process any pending pattern success checks
            patterns_to_remove = []
            for symbol, check_data in continuous_loop.pattern_checks.items():
                if current_time > check_data['check_time']:
                    try:
                        # Get current price
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = ticker['last'] if 'last' in ticker else ticker['close']
                        
                        # Determine if pattern was successful
                        pattern_success = False
                        price_change_pct = (current_price - check_data['entry_price']) / check_data['entry_price'] * 100
                        
                        if check_data['expected_direction'] == 'up' and price_change_pct > 0.5:
                            pattern_success = True
                        elif check_data['expected_direction'] == 'down' and price_change_pct < -0.5:
                            pattern_success = True
                            
                        # Update pattern success statistics
                        indicator = get_or_optimize_indicator(symbol, exchange)
                        
                        # Access indicator's pattern stats directly to avoid method compatibility issues
                        pattern_name = check_data['pattern'].replace(' ', '_').lower()
                        
                        if pattern_name in indicator.indicator.pattern_stats:
                            # Increment successful count if pattern moved in expected direction
                            if pattern_success:
                                indicator.indicator.pattern_stats[pattern_name]['successful'] = int(indicator.indicator.pattern_stats[pattern_name].get('successful', 0)) + 1
                                logger.info(f"Pattern {pattern_name} for {symbol} was successful! Price moved {price_change_pct:.2f}%")
                            else:
                                logger.info(f"Pattern {pattern_name} for {symbol} was unsuccessful. Price moved {price_change_pct:.2f}%")
                            
                            # Save updated pattern stats
                            indicator.indicator.save_pattern_stats()
                        
                        patterns_to_remove.append(symbol)
                        
                    except Exception as e:
                        logger.warning(f"Error checking pattern success for {symbol}: {e}")
                        patterns_to_remove.append(symbol)
                        
            # Remove processed checks
            for symbol in patterns_to_remove:
                if symbol in continuous_loop.pattern_checks:
                    del continuous_loop.pattern_checks[symbol]

            # Periodically check for symbols that need optimization
            if current_time - last_optimization_check > 600:
                check_for_optimization_needs(exchange)
                last_optimization_check = current_time

            # Fetch active trading symbols (limit for performance)
            active_markets = fetch_active_symbols(exchange)[:20] # Process top 20 liquid markets

            # Get current positions from the API
            open_positions = fetch_open_positions(exchange)
            open_symbols = list(open_positions.keys())

            # Determine if we're in "entry only" mode
            entry_only_mode = len(open_positions) >= MAX_OPEN_TRADES

            # Determine markets to process for potential new entries
            markets_to_process_for_entry = []
            if not entry_only_mode:
                markets_to_process_for_entry = active_markets
                # logger.info(f"Scanning {len(markets_to_process_for_entry)} markets for new entries.")
            else:
                logger.info(f"Maximum positions ({MAX_OPEN_TRADES}) reached - not checking for new entries")

            # --- Process markets for potential new entries ---
            for symbol in markets_to_process_for_entry:
                # Skip symbols already in position or in cooldown
                if symbol in open_positions or is_in_cooldown(symbol):
                    continue

                latest_price = None

                # Get or create the optimized indicator for this symbol
                try:
                    indicator = get_or_optimize_indicator(symbol, exchange)
                    if not indicator:
                        logger.warning(f"Could not get indicator for {symbol}, skipping.")
                        continue
                except Exception as e:
                    logger.exception(f"Error getting indicator for {symbol}: {e}")
                    continue

                # Check market suitability (volatility, volume)
                if not is_market_suitable_for_scalping(symbol, exchange):
                    # logger.debug(f"Skipping {symbol}: Market conditions not suitable for scalping.")
                    continue

                # Fetch live data for signal calculation (using primary timeframe)
                df_live = fetch_binance_data(exchange, symbol, timeframe=TIMEFRAME, limit=100) # Fetch enough for indicators
                if len(df_live) < max(20, indicator.indicator.trend_period): # Need enough for trend calc
                    logger.warning(f"Insufficient live data ({len(df_live)} candles) for {symbol} on {TIMEFRAME}")
                    continue

                # Get signals from the indicator
                buy_signals, sell_signals, _, signal_info = indicator.compute_signals(
                    df_live['open'],
                    df_live['high'],
                    df_live['low'],
                    df_live['close']
                )

                # Copy the 'is_current_candle' flag for trend calculation accuracy
                if 'is_current_candle' in df_live.columns:
                    signal_info['is_current_candle'] = df_live['is_current_candle']

                latest_candle = df_live.iloc[-1]
                latest_price = latest_candle['close']

                # Get trend information (using closed candles only for reliability)
                trend_info = get_trend_info(signal_info, closed_candles_only=True)
                trend_direction = trend_info.get('direction', 'flat')

                # Check pattern detection across multiple timeframes
                log_trade_metrics('patterns_detected')
                best_pattern, pattern_reliability, multi_tf_patterns, pattern_detections, pattern_sequences = detect_multi_timeframe_patterns(
                    exchange,
                    symbol,
                    timeframes=timeframes
                )

                # --- Entry Logic (RELAXED) ---
                if not entry_only_mode: # Double-check we can still enter
                    # Only require that a pattern was detected
                    if not best_pattern:
                        log_trade_metrics('patterns_below_threshold')
                        logger.debug(f"Skipping {symbol}: No pattern detected")
                        continue
                    
                    # Use a lower baseline reliability threshold (25%)
                    if pattern_reliability < 25.0:
                        log_trade_metrics('patterns_below_threshold')
                        logger.debug(f"Skipping {symbol}: Pattern {best_pattern} reliability too low: {pattern_reliability:.1f}% < 25.0%")
                        continue

                    # Count timeframes where this specific pattern appears with sufficient reliability
                    best_pattern_timeframes = [detection['timeframe'] for detection in pattern_detections 
                                              if detection['pattern'] == best_pattern and detection['reliability'] >= 25.0]
                    
                    if not best_pattern_timeframes:
                        log_trade_metrics('insufficient_tf_confirmation')
                        logger.debug(f"Skipping {best_pattern} for {symbol} - not found on any timeframe with sufficient reliability")
                        continue
                    else:
                        logger.info(f"Pattern {best_pattern} found on timeframes: {best_pattern_timeframes}")

                    # Make pattern sequence check optional - just log it
                    has_consistent_sequence = False
                    if pattern_sequences:
                        for tf, patterns in pattern_sequences.items():
                            if len(patterns) >= 2:
                                if any(p['pattern'] == best_pattern for p in patterns):
                                    has_consistent_sequence = True
                                    break
                    
                    if not has_consistent_sequence:
                        # Just log it, don't filter out the trade
                        logger.info(f"Note: {best_pattern} for {symbol} - no consistent pattern sequence found, but proceeding anyway")

                    # Determine pattern direction and trade type
                    is_neutral_pattern = any(p.get('is_neutral', False) for p in pattern_detections if p.get('pattern') == best_pattern)
                    if is_neutral_pattern:
                        log_trade_metrics('patterns_below_threshold')
                        logger.debug(f"Skipping {symbol}: Best pattern '{best_pattern}' is neutral.")
                        continue

                    is_bullish_pattern = any(p['signal_type'] == 'buy' for p in pattern_detections if p.get('pattern') == best_pattern)
                    is_bearish_pattern = any(p['signal_type'] == 'sell' for p in pattern_detections if p.get('pattern') == best_pattern)

                    if not is_bullish_pattern and not is_bearish_pattern:
                        log_trade_metrics('patterns_below_threshold')
                        logger.warning(f"Could not determine direction for pattern {best_pattern} for {symbol}, skipping.")
                        continue

                    # Schedule a check for this pattern's success even if we don't take a trade
                    check_time = time.time()
                    if '15m' in timeframes:
                        check_time += 900  # Check after 15 minutes
                    elif '5m' in timeframes:
                        check_time += 300  # Check after 5 minutes
                    elif '3m' in timeframes:
                        check_time += 180  # Check after 3 minutes
                    else:
                        check_time += 600  # Default to 10 minutes
                        
                    continuous_loop.pattern_checks[symbol] = {
                        'pattern': best_pattern,
                        'check_time': check_time,
                        'entry_price': latest_price,
                        'expected_direction': 'up' if is_bullish_pattern else 'down'
                    }
                    logger.info(f"Scheduled success check for {best_pattern} pattern on {symbol} at {time.strftime('%H:%M:%S', time.localtime(check_time))}")

                    trade_type = 'unknown'
                    if is_bullish_pattern:
                        if trend_direction == 'bullish': trade_type = 'continuation'
                        elif trend_direction == 'bearish': trade_type = 'reversal'
                    elif is_bearish_pattern:
                        if trend_direction == 'bearish': trade_type = 'continuation'
                        elif trend_direction == 'bullish': trade_type = 'reversal'

                    # Only apply special handling for reversals - use fixed threshold
                    if trade_type == 'reversal':
                        min_reversal_reliability = 30.0  # Fixed threshold instead of multiplier
                        if pattern_reliability < min_reversal_reliability:
                            log_trade_metrics('failed_reversal_check')
                            logger.info(f"Skipping {best_pattern} reversal for {symbol} - reliability {pattern_reliability:.1f}% < {min_reversal_reliability:.1f}%")
                            continue
                        else:
                            logger.info(f"Taking reversal trade with {best_pattern} pattern ({pattern_reliability:.1f}%)")

                    # Determine trade side
                    side = 'buy' if is_bullish_pattern else 'sell'
                    position_type = 'long' if side == 'buy' else 'short'

                    # Get trade parameters
                    leverage = get_leverage_for_market(exchange, symbol)
                    if not leverage: continue
                    quantity = get_order_quantity(exchange, symbol, latest_price, leverage)
                    if quantity <= 0:
                         logger.warning(f"Calculated quantity is zero or negative for {symbol}, skipping.")
                         continue

                    # --- S/R Analysis for SL/TP ---
                    try:
                        logger.info(f"Starting S/R analysis for {symbol} with trend: {trend_direction}")
                        sr_analysis = detect_swing_sr_levels(
                            exchange,
                            symbol,
                            trend_direction=trend_direction
                        )
                        calculated_stop_loss = sr_analysis['stop_loss']
                        calculated_target_price = sr_analysis['target_price']
                        planned_risk_reward = sr_analysis['risk_reward']

                        logger.info(f"S/R Analysis completed for {symbol} {side} @ {latest_price:.6f}: "
                                  f"SL: {calculated_stop_loss:.6f}, "
                                  f"TP: {calculated_target_price:.6f}, "
                                  f"Planned R/R: {planned_risk_reward:.2f}")
                                  
                        # Log support/resistance zones
                        if 'support_zones' in sr_analysis:
                            support_str = ", ".join([f"{z['center']:.6f} (str: {z['strength']:.1f})" 
                                                    for z in sr_analysis['support_zones'][:3]])
                            logger.info(f"Support zones for {symbol}: {support_str}")
                        
                        if 'resistance_zones' in sr_analysis:
                            resistance_str = ", ".join([f"{z['center']:.6f} (str: {z['strength']:.1f})" 
                                                       for z in sr_analysis['resistance_zones'][:3]])
                            logger.info(f"Resistance zones for {symbol}: {resistance_str}")

                        # RELAXED: Lower Risk/Reward requirement from 1.5 to 1.2
                        # if planned_risk_reward < 1.2:
                        #     log_trade_metrics('failed_risk_reward')
                        #     logger.info(f"Skipping {best_pattern} for {symbol} - low R/R ratio ({planned_risk_reward:.2f})")
                        #     continue

                        # --- *** Entry Proximity Check (RELAXED) *** ---
                        key_sr_level = None
                        if side == 'buy':
                            valid_supports = [zone for zone in sr_analysis.get('support_zones', []) if zone['center'] < latest_price]
                            if valid_supports:
                                valid_supports.sort(key=lambda z: latest_price - z['center'])
                                strong_supports = [z for z in valid_supports if z['strength'] > 1.5 and (latest_price - z['center']) / latest_price >= 0.005]
                                if strong_supports: key_sr_level = strong_supports[0]['center']
                                elif valid_supports: key_sr_level = valid_supports[0]['center']
                        else: # side == 'sell'
                            valid_resistances = [zone for zone in sr_analysis.get('resistance_zones', []) if zone['center'] > latest_price]
                            if valid_resistances:
                                valid_resistances.sort(key=lambda z: z['center'] - latest_price)
                                strong_resistances = [z for z in valid_resistances if z['strength'] > 1.5 and (z['center'] - latest_price) / latest_price >= 0.005]
                                if strong_resistances: key_sr_level = strong_resistances[0]['center']
                                elif valid_resistances: key_sr_level = valid_resistances[0]['center']

                        proceed_with_entry = False
                        if key_sr_level is not None:
                             # Sanity check the SL relative to the key level
                             sl_is_logical = (side == 'buy' and calculated_stop_loss < key_sr_level) or \
                                             (side == 'sell' and calculated_stop_loss > key_sr_level)

                             if sl_is_logical:
                                # Calculate max offset using ATR - INCREASED from 0.3 to 0.8
                                atr_val = df_live['atr'].iloc[-1] if 'atr' in df_live.columns and not df_live['atr'].empty else latest_price * 0.001 # Default ATR if missing
                                max_entry_offset = atr_val * 0.8 # INCREASED from 0.3 to 0.8 ATR

                                if side == 'buy':
                                    if latest_price <= key_sr_level + max_entry_offset:
                                        proceed_with_entry = True
                                    else:
                                        log_trade_metrics('failed_proximity_check')
                                        logger.info(f"Skipping LONG {symbol}: Price {latest_price:.6f} too far above support {key_sr_level:.6f} (Max offset: {max_entry_offset:.6f})")
                                else: # side == 'sell'
                                    if latest_price >= key_sr_level - max_entry_offset:
                                        proceed_with_entry = True
                                    else:
                                        log_trade_metrics('failed_proximity_check')
                                        logger.info(f"Skipping SHORT {symbol}: Price {latest_price:.6f} too far below resistance {key_sr_level:.6f} (Max offset: {max_entry_offset:.6f})")
                             else:
                                 log_trade_metrics('failed_proximity_check')
                                 logger.warning(f"Skipping {symbol} {side}: Calculated SL {calculated_stop_loss:.6f} is illogical relative to key S/R level {key_sr_level:.6f}. Check S/R logic.")
                        else:
                            # RELAX: Allow entry even without key S/R level if the pattern is good
                            if pattern_reliability >= 35.0:  # Higher standard if no S/R confirmation
                                logger.info(f"Allowing entry for {symbol} {side} with strong pattern ({pattern_reliability:.1f}%) despite no key S/R level")
                                proceed_with_entry = True
                            else:
                                log_trade_metrics('failed_proximity_check')
                                logger.warning(f"Skipping {symbol} {side}: Could not determine key S/R level for entry proximity check.")
                        # --- *** End of Proximity Check *** ---

                        # --- Place Order (Conditionally) ---
                        proceed_with_entry = True
                        if proceed_with_entry:
                            # Store position details before placing order
                            with position_details_lock:
                                position_details[symbol] = {
                                    'entry_price': latest_price, # Log estimated entry price
                                    'stop_loss': calculated_stop_loss, # Use SL from S/R analysis
                                    'target': calculated_target_price, # Use TP from S/R analysis
                                    'position_type': position_type,
                                    'entry_reason': f"{best_pattern.replace('_', ' ').title()} Pattern",
                                    'probability': pattern_reliability  / 100,
                                    'entry_time': time.time(),
                                    'highest_reached': latest_price if position_type == 'long' else None,
                                    'lowest_reached': latest_price if position_type == 'short' else None,
                                    'pattern_used': best_pattern,
                                    'trend_direction': trend_direction,
                                    'trend_angle': trend_info.get('angle', 0),
                                    'trend_strength': trend_info.get('strength', 0),
                                    'trend_duration': trend_info.get('duration', 0),
                                    'trend_trade_type': trade_type,
                                    'risk_reward': planned_risk_reward # Log the planned R/R
                                }

                            trend_context = ""
                            if trade_type != 'unknown':
                                trend_context = f" ({trade_type.capitalize()} with {trend_direction} trend)"

                            logger.info(f"Placing {side.upper()} market order for {symbol} - {best_pattern} pattern with {pattern_reliability:.1f}% reliability")

                            # Place the entry order with automatic SL/TP based on sr_analysis
                            order = place_order(
                                exchange,
                                symbol,
                                side,
                                quantity,
                                latest_price, # Pass current price for reference
                                leverage=leverage,
                                exit_order=False,
                                stop_loss=calculated_stop_loss, # Use SL from S/R analysis
                                target_price=calculated_target_price, # Use TP from S/R analysis
                                place_target_order=True
                            )

                            if order:
                                log_trade_metrics('successful_entries')
                                logger.info(f"Opened {side.upper()} position for {symbol} based on {best_pattern} pattern{trend_context}. "
                                          f"Reliability: {pattern_reliability:.1f}%, "
                                          f"Planned R/R: {planned_risk_reward:.2f}, "
                                          f"Stop: {calculated_stop_loss:.6f}, "
                                          f"Target: {calculated_target_price:.6f}")

                                # Log the entry to CSV
                                entry_log = {
                                    'entry_price': latest_price, # Log the price at signal time
                                    'stop_loss': calculated_stop_loss,
                                    'target': calculated_target_price,
                                    'entry_reason': f"{best_pattern.replace('_', ' ').title()} Pattern{trend_context}",
                                    'position_type': position_type,
                                    'probability': pattern_reliability / 100,
                                    'risk_reward': planned_risk_reward, # Log planned R/R
                                    'leverage': leverage,
                                    'entry_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                    'position_status': 'Open',
                                    'pattern_used': best_pattern,
                                    'multi_timeframe_confirmed': len(best_pattern_timeframes) >= 2,
                                    'trend_direction': trend_direction,
                                    'trend_angle': trend_info.get('angle', 0),
                                    'trend_strength': trend_info.get('strength', 0),
                                    'trend_duration': trend_info.get('duration', 0),
                                    'trend_trade_type': trade_type,
                                    'exit_managed_by': 'Exchange SL/TP Orders'
                                }
                                log_trade(symbol, entry_log, force_write=True)

                                time.sleep(SLEEP_INTERVAL) # Pause briefly after placing an order
                                break # Process one new entry per loop iteration for safety

                            else:
                                log_trade_metrics('order_placement_errors')
                                # Failed to place entry order
                                logger.warning(f"Failed to open {side} position for {symbol} after proximity check.")
                                # Clean up position details if entry failed
                                with position_details_lock:
                                    if symbol in position_details:
                                        del position_details[symbol]
                        # --- End Place Order ---

                    except Exception as e:
                        logger.warning(f"Error during S/R analysis or entry logic for {symbol}: {e}")
                        # Fallback or skip logic could be added here if needed
                        # For now, just log and continue to the next symbol

                # --- End Entry Logic Block ---

            # Brief sleep between processing symbols to avoid hitting rate limits too hard
            time.sleep(0.2)

            # --- End of loop for processing symbols ---

            # Main loop sleep
            time.sleep(SLEEP_INTERVAL)

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}. Sleeping for 60 seconds.")
            time.sleep(60)
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}. Retrying in 30 seconds.")
            time.sleep(30)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}. Check API keys and permissions. Sleeping.")
            time.sleep(SLEEP_INTERVAL * 2) # Longer sleep on exchange errors
        except Exception as e:
            logger.exception(f"Error in main trading cycle: {e}")
            time.sleep(SLEEP_INTERVAL) # Sleep even on general exceptions

def main():
    """Main function to start the trading bot with separate optimization thread."""
    exchange = create_exchange()
    
    # Create pattern stats directory if it doesn't exist
    os.makedirs(PATTERN_STATS_DIR, exist_ok=True)
    
    # Reset and fix pattern statistics at startup
    reset_pattern_statistics()
    
    # Load existing parameters at startup
    load_existing_parameters()
    
    # Recovery mode - check for any missed exits
    sync_positions_with_exchange(exchange)
    
    # Start the optimization worker thread
    optimization_thread = threading.Thread(target=optimization_worker, args=(exchange,), daemon=True)
    optimization_thread.start()
    logger.info("Started optimization worker thread")
    
    # Check for symbols that need optimization
    check_for_optimization_needs(exchange)
    
    # Start the metrics reporting thread
    metrics_thread = threading.Thread(target=metrics_reporting_thread, daemon=True)
    metrics_thread.start()
    logger.info("Started metrics reporting thread")
    
    # Start the main trading loop
    trading_thread = threading.Thread(target=continuous_loop, args=(exchange,), daemon=True)
    trading_thread.start()
    logger.info("Started main trading thread")
    
    # Generate pattern statistics report daily
    last_report_time = time.time()
    REPORT_INTERVAL = 86400  # 24 hours
    
    # Keep the main thread alive
    while True:
        time.sleep(10)
        
        # Generate periodic pattern statistics report
        current_time = time.time()
        if current_time - last_report_time > REPORT_INTERVAL:
            logger.info("Generating pattern statistics report...")
            report = generate_pattern_statistics_report()
            logger.info("\n" + report)
            
            # Save report to file
            report_path = os.path.join(PATTERN_STATS_DIR, f"pattern_report_{time.strftime('%Y%m%d')}.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            
            #logger.info(f"Pattern report saved to {report_path}")
            last_report_time = current_time

if __name__ == "__main__":
    main()