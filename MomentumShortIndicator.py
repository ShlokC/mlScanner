import numpy as _np
_np.NaN = _np.nan
import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MomentumShortIndicator:
    """
    Indicator for identifying short selling opportunities and exit points in crypto markets.
    
    Entry Criteria:
    1. Coin has gained 20%+ in last 1-3 days (using 5-min candles)
    2. Price has crossed below hma(144) on 5-min candle after being above for multiple candles
    3. Supertrend indicator is red (bearish) on 5-min candle
    
    Exit Criteria:
    1. For shorts: Price sustainably above stop loss (HMA+buffer) or entry price
    2. For longs: Price sustainably below stop loss (HMA-buffer)
    """
    
    def __init__(self, 
                symbol="UNKNOWN",
                lookback_days=3,
                min_price_gain=20.0,  # 20% minimum gain
                hma_period=144,       # hma period for 5-min candles (144 = 12 hours)
                supertrend_factor=1.0, # Supertrend multiplier
                supertrend_length=7,  # Supertrend period
                sl_buffer_pct=2.0,     # SL buffer percent above hma
                entry_sustained_candles=3,  # Candles above HMA before crossunder for entry
                exit_sustained_candles=2,   # Candles above/below threshold for exit
                **kwargs):
        
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.min_price_gain = min_price_gain
        self.hma_period = hma_period
        self.supertrend_factor = supertrend_factor
        self.supertrend_length = supertrend_length
        self.sl_buffer_pct = sl_buffer_pct
        self.entry_sustained_candles = entry_sustained_candles
        self.exit_sustained_candles = exit_sustained_candles
        
        # Internal tracking
        self.price_history = pd.DataFrame()
        self.signal_history = []
        self.crossunder_state = {}
        
        # Track detected crossunders to prevent multiple signals on same crossunder
        self.detected_crossunders = {}
        
        #logger.info(f"Initialized MomentumShortIndicator for {symbol}")
    
    def update_price_data(self, df):
        """Update price history with new data."""
        if df is not None and not df.empty:
            # Make a copy to avoid modifying the original
            new_data = df.copy()
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in new_data.columns:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            # Update the stored price history
            self.price_history = pd.concat([self.price_history, new_data])
            
            # Remove duplicates (keeping the last occurrence)
            self.price_history = self.price_history[~self.price_history.index.duplicated(keep='last')]
            
            # Sort by index (timestamp)
            self.price_history = self.price_history.sort_index()
            
            # Keep only recent data (last 7 days to be safe)
            cutoff_time = self.price_history.index[-1] - pd.Timedelta(days=7)
            self.price_history = self.price_history[self.price_history.index > cutoff_time]
            
            return True
        return False
    
    def check_price_gain(self, df, days=None):
        """
        Check if price has gained more than min_price_gain% within a rolling window.
        This function searches for the maximum gain from any low to any subsequent high
        within the specified period. Optimized for performance.
        
        Args:
            df: DataFrame with OHLC data
            days: Number of days to look back (uses self.lookback_days if None)
                
        Returns:
            Tuple of (has_gained, gain_pct, low_price, high_price, low_idx, high_idx)
        """
        try:
            if days is None:
                days = self.lookback_days

            if df is None or len(df) < 10:  # Need some minimal data
                return False, 0.0, None, None, None, None

            # Calculate candles per day for 5-min timeframe
            candles_per_day = int(24 * 60 / 5)  # 288 candles per day

            # Calculate how many candles to look back
            lookback_candles = candles_per_day * days

            # Limit to available data
            actual_lookback = min(lookback_candles, len(df) - 1)

            # Get recent data to analyze
            recent_data = df.iloc[-actual_lookback:]

            # Extract necessary data as NumPy arrays for faster processing
            lows = recent_data['low'].values
            highs = recent_data['high'].values
            indices = recent_data.index

            n = len(lows)

            # Use NumPy to find the maximum possible gain
            max_gain_pct = 0.0
            best_low_price = None
            best_high_price = None
            best_low_idx = None
            best_high_idx = None

            # For each potential low point
            for i in range(n - 1):
                low_price = lows[i]
                low_idx = indices[i]

                # Skip invalid low prices
                if low_price <= 0:
                    continue

                # Use vectorized operations for all subsequent high prices
                subsequent_highs = highs[i+1:]
                subsequent_indices = indices[i+1:]

                # Calculate gain percentages for all subsequent highs in one operation
                gains = ((subsequent_highs - low_price) / low_price) * 100

                if len(gains) > 0:
                    # Find the maximum gain and its index
                    max_gain_idx = np.argmax(gains)
                    current_max_gain = gains[max_gain_idx]

                    # Update if we found a better gain
                    if current_max_gain > max_gain_pct:
                        max_gain_pct = current_max_gain
                        best_low_price = low_price
                        best_high_price = subsequent_highs[max_gain_idx]
                        best_low_idx = low_idx
                        best_high_idx = subsequent_indices[max_gain_idx]

                        # If we've found a gain that significantly exceeds our criteria, we can stop early
                        if max_gain_pct >= self.min_price_gain * 1.5:
                            break

            # Check if gain exceeds minimum threshold
            has_gained = max_gain_pct >= self.min_price_gain

            return has_gained, max_gain_pct, best_low_price, best_high_price, best_low_idx, best_high_idx

        except Exception as e:
            logger.info(f"Error while checking price gain for {self.symbol}: {e}")
            return False, 0.0, None, None, None, None

    def check_hma_crossunder(self, df, lookback_candles=864):
        """
        Detect when price crosses below HMA with special attention to whether
        this is the first crossunder after a significant high point.
        """
        # Standard initialization
        has_crossunder = False
        crossunder_candle_age = 0
        minutes_ago = 0
        is_first_crossunder = False
        
        try:
            # Ensure adequate data
            if df is None or len(df) < self.hma_period:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            # First, identify if there was a significant gain and find the high point
            has_gain, gain_pct, low_price, high_price, low_idx, high_idx = self.check_price_gain(df)
            
            # If no significant gain or no high point identified, don't proceed
            if not has_gain or high_idx is None:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            # Find or ensure HMA column exists
            hma_columns = [col for col in df.columns if col.startswith('HMA_')]
            if not hma_columns:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            hma_column = hma_columns[0]
            
            # Create price vs HMA relationship (above/below)
            df_valid = df.dropna(subset=['close', hma_column]).copy()
            df_valid['above_hma'] = df_valid['close'] > df_valid[hma_column]
            
            # Find all crossunder points (transition from above to below)
            crossunder_points = []
            for i in range(1, len(df_valid)):
                if df_valid['above_hma'].iloc[i-1] and not df_valid['above_hma'].iloc[i]:
                    crossunder_points.append((df_valid.index[i], i))
            
            if not crossunder_points:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            # We have crossunders, now find the first one AFTER the high point
            post_high_crossunders = [x for x in crossunder_points if x[0] > high_idx]
            
            if not post_high_crossunders:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            # We found crossunders after the high point
            has_crossunder = True
            
            # The first one is the first crossunder after high
            first_timestamp, first_idx = post_high_crossunders[0]
            
            # This is definitionally the first crossunder after high
            is_first_crossunder = True
            
            # Calculate candle age and time
            crossunder_candle_age = len(df_valid) - 1 - first_idx
            
            # Calculate minutes ago
            current_time = df.index[-1]
            if isinstance(current_time, pd.Timestamp) and isinstance(first_timestamp, pd.Timestamp):
                time_diff = current_time - first_timestamp
                minutes_ago = int(time_diff.total_seconds() / 60)
            else:
                minutes_ago = crossunder_candle_age * 5  # Assume 5-min candles
                
            return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
            
        except Exception as e:
            logger.error(f"{self.symbol}: Error in check_hma_crossunder: {e}")
            return False, 0, 0, False
    
    def check_sustained_above_level(self, df, level, num_candles=None):
        """
        Check if price has been sustainably above a specific level.
        Verifies if the most recent `num_candles` candles are all above the level.
        
        Args:
            df: DataFrame with OHLC data
            level: Price level to check against
            num_candles: Number of consecutive candles required above the level
        
        Returns:
            bool: True if the most recent num_candles candles are all above the level
        """
        if num_candles is None:
            num_candles = self.exit_sustained_candles
            
        if df is None or len(df) < num_candles:
            return False
        
        # Get just the most recent num_candles
        recent_candles = df.iloc[-num_candles:]
        
        # Check if ALL of these candles are above the level
        all_above = all(recent_candles['close'] > level)
        
        return all_above

    def check_sustained_below_level(self, df, level, num_candles=None):
        """
        Check if price has been sustainably below a specific level.
        Verifies if the most recent `num_candles` candles are all below the level.
        
        Args:
            df: DataFrame with OHLC data
            level: Price level to check against
            num_candles: Number of consecutive candles required below the level
        
        Returns:
            bool: True if the most recent num_candles candles are all below the level
        """
        if num_candles is None:
            num_candles = self.exit_sustained_candles
            
        if df is None or len(df) < num_candles:
            return False
        
        # Get just the most recent num_candles
        recent_candles = df.iloc[-num_candles:]
        
        # Check if ALL of these candles are below the level
        all_below = all(recent_candles['close'] < level)
        
        return all_below
        
    def check_supertrend_bearish(self, df, supertrend_data=None):
        """
        Check if the Supertrend indicator is bearish (red).
        """
        if df is None or len(df) < self.supertrend_length + 1:
            return False
            
        if supertrend_data is None:
            # Calculate Supertrend using pandas_ta
            supertrend_data = df.ta.supertrend(
                length=self.supertrend_length,
                multiplier=self.supertrend_factor
            )
        
        if supertrend_data is None or supertrend_data.empty:
            return False
        
        # Get direction column name (format: SUPERTd_{length}_{factor})
        direction_col = f"SUPERTd_{self.supertrend_length}_{self.supertrend_factor}"
        
        if direction_col not in supertrend_data.columns:
            # Try to find a similar column if exact match not found
            direction_cols = [col for col in supertrend_data.columns if col.startswith('SUPERTd')]
            if not direction_cols:
                return False
            direction_col = direction_cols[0]
        
        # Get current Supertrend direction value
        current_direction = supertrend_data[direction_col].iloc[-1]
        
        # Direction -1 means bearish (red)
        return current_direction == -1
    
    def determine_stop_loss(self, df):
        """
        Determine stop-loss level for short positions based on HMA.
        For shorts, stop loss is set above current HMA with buffer.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            float: Stop-loss price level or None if cannot be determined
        """
        try:
            if df is None or len(df) < self.hma_period:
                logger.debug(f"{self.symbol}: Insufficient data for stop loss calculation")
                return None
            
            # Check if HMA already exists in the dataframe
            hma_columns = [col for col in df.columns if col.startswith('HMA_')]
            
            # Use existing HMA if available
            if hma_columns:
                hma_column = hma_columns[0]
                current_hma = df[hma_column].iloc[-1]
            else:
                # Calculate HMA if not present
                hma_series = df.ta.hma(length=self.hma_period)
                if hma_series.empty:
                    logger.debug(f"{self.symbol}: Failed to calculate HMA")
                    return None
                current_hma = hma_series.iloc[-1]
            
            # Make sure HMA is valid
            if pd.isna(current_hma) or current_hma <= 0:
                logger.debug(f"{self.symbol}: Invalid HMA value: {current_hma}")
                return None
            
            # Set stop-loss sl_buffer_pct% above HMA for shorts
            stop_loss = current_hma * (1 + self.sl_buffer_pct / 100)
            
            return stop_loss
    
        except Exception as e:
            logger.error(f"{self.symbol}: Error determining stop loss: {e}")
            return None
        
    # Replace generate_exit_signal method in the MomentumShortIndicator class:

    def generate_exit_signal(self, df, current_position, entry_price, stop_loss):
        """
        Generate exit signal based on position type and current market conditions.
        For shorts: Requires BOTH conditions to be met:
        1. Price is sustainably above entry price
        2. Price is sustainably above HMA+buffer
        
        Args:
            df: DataFrame with OHLC data
            current_position: String indicating position type ('long' or 'short')
            entry_price: Original entry price
            stop_loss: Current stop loss level (HMA+buffer)
            
        Returns:
            dict: Exit signal information
        """
        try:
            # Ensure we have enough data
            if df is None or df.empty:
                return {
                    'signal': 'none',
                    'reason': 'insufficient_data',
                    'price': None,
                    'exit_triggered': False
                }
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate current HMA for reference
            hma_series = df.ta.hma(length=self.hma_period)
            current_hma = hma_series.iloc[-1] if not hma_series.empty else None
            
            # Initialize result
            exit_signal = {
                'signal': 'none',
                'reason': 'conditions_not_met',
                'price': current_price,
                'exit_triggered': False,
                'hma_value': current_hma
            }
            
            # Check for exit conditions with sustained threshold checks
            if current_position == 'short':
                # For shorts - check if price is sustainably above BOTH entry price AND HMA+buffer
                sustained_above_sl = self.check_sustained_above_level(df, stop_loss)
                sustained_above_entry = self.check_sustained_above_level(df, entry_price)
                
                # BOTH conditions must be met for exit
                if sustained_above_sl and sustained_above_entry:
                    exit_signal['signal'] = 'exit_short'
                    exit_signal['exit_triggered'] = True
                    exit_signal['reason'] = f'Price {current_price} sustainably above BOTH stop level {stop_loss} (HMA+buffer) AND entry price {entry_price}'
                    exit_signal['execute_reversal'] = True  # Signal to execute reversal (short to long)
                    
                    # Add HMA info for reference
                    if current_hma is not None:
                        hma_buffer = self.sl_buffer_pct
                        exit_signal['hma_value'] = current_hma
                        exit_signal['hma_plus_buffer'] = current_hma * (1 + hma_buffer/100)
            
            elif current_position == 'long':
                # For longs - check if price is sustainably below stop loss
                sustained_below_sl = self.check_sustained_below_level(df, stop_loss)
                
                if sustained_below_sl:
                    exit_signal['signal'] = 'exit_long'
                    exit_signal['exit_triggered'] = True
                    exit_signal['reason'] = f'Price {current_price} sustainably below stop level {stop_loss} for {self.exit_sustained_candles} candles'
                    exit_signal['execute_reversal'] = False  # No reversal for longs (per requirements)
                    
                    # Add HMA info for reference
                    if current_hma is not None:
                        hma_buffer = self.sl_buffer_pct
                        exit_signal['hma_value'] = current_hma
                        exit_signal['hma_minus_buffer'] = current_hma * (1 - hma_buffer/100)
            
            return exit_signal
        
        except Exception as e:
            logger.exception(f"{self.symbol}: Unexpected error in generate_exit_signal: {e}")
            return {
                'signal': 'none',
                'reason': f'error: {str(e)}',
                'price': None,
                'exit_triggered': False
            }
    
    def generate_signal(self, current_position=None, entry_price=None, stop_loss=None):
        """
        Generate trading signals based on market conditions.
        If current_position is provided, checks for exit signals.
        Otherwise, looks for entry signals.
        
        Args:
            current_position: Optional - current position type ('long', 'short', or None)
            entry_price: Optional - entry price if in a position
            stop_loss: Optional - current stop loss level if in a position
            
        Returns:
            dict: Signal information
        """
        try:
            # Ensure we have enough data
            if self.price_history.empty or len(self.price_history) < self.hma_period:
                logger.warning(f"{self.symbol}: Insufficient price history for signal generation")
                return {
                    'signal': 'none',
                    'reason': 'insufficient_data',
                    'price': None,
                    'stop_loss': None,
                    'exit_triggered': False
                }
            
            # Make a copy of the data
            df = self.price_history.copy()
            
            # If we're in a position, check for exit signals
            if current_position is not None and entry_price is not None and stop_loss is not None:
                return self.generate_exit_signal(df, current_position, entry_price, stop_loss)
            
            # Initialize variables for entry signals
            current_price = df['close'].iloc[-1]
            
            # 1. Check for price gain
            has_price_gain, gain_pct, low_price, high_price, low_idx, high_idx = self.check_price_gain(df)
            
            if not has_price_gain:
                return {
                    'signal': 'none',
                    'reason': 'no_significant_price_gain',
                    'price': current_price,
                    'stop_loss': None,
                    'exit_triggered': False,
                    'high_price': high_price,
                    'gain_pct': gain_pct
                }
            
            # Calculate drawdown from high
            drawdown_pct = ((high_price - current_price) / high_price) * 100 if high_price and high_price > 0 else 0
            
            # Calculate minutes since high
            high_minutes_ago = 0
            if high_idx is not None and isinstance(high_idx, pd.Timestamp):
                current_time = df.index[-1]
                if isinstance(current_time, pd.Timestamp):
                    time_diff = current_time - high_idx
                    high_minutes_ago = int(time_diff.total_seconds() / 60)
            
            # 2. Check for HMA crossunder
            has_hma_crossunder, crossunder_age, minutes_ago, is_first_crossunder = self.check_hma_crossunder(df)
            
            # 3. Check if current price is below HMA
            # Find or calculate HMA
            hma_columns = [col for col in df.columns if col.startswith('HMA_')]
            if hma_columns:
                hma_column = hma_columns[0]
                current_hma = df[hma_column].iloc[-1]
            else:
                # Calculate HMA if not present
                hma_series = df.ta.hma(length=self.hma_period)
                if hma_series.empty:
                    current_hma = None
                else:
                    current_hma = hma_series.iloc[-1]
                    # Add to df for later use
                    df[f'HMA_{self.hma_period}'] = hma_series
            
            is_below_hma = current_price < current_hma if current_hma is not None else False
            
            # 4. Check supertrend
            try:
                supertrend_data = df.ta.supertrend(
                    length=self.supertrend_length,
                    multiplier=self.supertrend_factor
                )
                is_supertrend_bearish = self.check_supertrend_bearish(df, supertrend_data)
            except Exception as e:
                logger.error(f"{self.symbol}: Error calculating Supertrend: {e}")
                is_supertrend_bearish = False
            
            # All conditions for a sell signal
            conditions_met = {
                'price_gain_met': has_price_gain,
                'price_gain_pct': f"{gain_pct:.2f}%",
                'drawdown_pct': drawdown_pct,
                'high_minutes_ago': high_minutes_ago,
                'hma_crossunder_met': has_hma_crossunder,
                'is_first_crossunder': is_first_crossunder,
                'crossunder_age': crossunder_age,
                'crossunder_minutes_ago': minutes_ago,
                'is_below_hma': is_below_hma,
                'supertrend_bearish_met': is_supertrend_bearish
            }
            
            # Generate signal when all conditions are met
            if has_price_gain and has_hma_crossunder and is_supertrend_bearish and is_below_hma:
                # Determine stop-loss based on HMA
                stop_loss = self.determine_stop_loss(df)
                
                if stop_loss is None or stop_loss <= 0:
                    return {
                        'signal': 'none',
                        'reason': 'invalid_stop_loss',
                        'price': current_price,
                        'stop_loss': None,
                        'conditions_met': conditions_met,
                        'exit_triggered': False
                    }
                
                # Generate sell signal
                signal = {
                    'signal': 'sell',
                    'reason': f"Gained {gain_pct:.2f}% and now {drawdown_pct:.2f}% down from high, "
                            f"crossed below HMA({self.hma_period}) {minutes_ago} mins ago, "
                            f"supertrend bearish",
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'conditions_met': conditions_met,
                    'has_hma_crossunder': has_hma_crossunder,
                    'is_first_crossunder': is_first_crossunder,
                    'crossunder_age': crossunder_age,
                    'crossunder_minutes_ago': minutes_ago,
                    'exit_triggered': False
                }
                
                return signal
            
            # No signal if conditions are not met
            return {
                'signal': 'none',
                'reason': 'not_all_conditions_met',
                'price': current_price,
                'stop_loss': None,
                'conditions_met': conditions_met,
                'exit_triggered': False
            }
        
        except Exception as e:
            logger.exception(f"{self.symbol}: Unexpected error in generate_signal: {e}")
            return {
                'signal': 'none',
                'reason': f'error: {str(e)}',
                'price': None,
                'stop_loss': None,
                'exit_triggered': False
            }

# Function to integrate with the main trading system
def create_momentum_short_signal_generator(symbol, **kwargs):
    """Factory function to create a momentum short signal generator."""
    return MomentumShortIndicator(symbol=symbol, **kwargs)