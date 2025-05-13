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
    
    Short Entry Criteria:
    1. Coin has gained 20%+ in last 1-3 days (using 5-min candles)
    2. Price has crossed below kama(40) on 5-min candle after being above for multiple candles
    3. Supertrend indicator is red (bearish) on 5-min candle
    
    Exit Criteria:
    1. For shorts: Price sustainably above stop loss (kama+buffer) or entry price
    """
    
    def __init__(self, 
                symbol="UNKNOWN",
                lookback_days=1,
                min_price_change=30.0,  # 20% minimum gain/decrease
                kama_period=40,       # kama period for 5-min candles (40 = 12 hours)
                supertrend_factor=2.0, # Supertrend multiplier
                supertrend_length=7,  # Supertrend period
                sl_buffer_pct=6.0,     # SL buffer percent above/below kama
                entry_sustained_candles=3,  # Candles above/below kama before cross for entry
                exit_sustained_candles=2,   # Candles above/below threshold for exit
                **kwargs):
        
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.min_price_change = min_price_change
        self.kama_period = kama_period
        self.supertrend_factor = supertrend_factor
        self.supertrend_length = supertrend_length
        self.sl_buffer_pct = sl_buffer_pct
        self.entry_sustained_candles = entry_sustained_candles
        self.exit_sustained_candles = exit_sustained_candles
        
        # Internal tracking
        self.price_history = pd.DataFrame()
        self.signal_history = []
        self.crossunder_state = {}
        self.crossover_state = {}
        
        # Track detected crosses to prevent multiple signals on same cross
        self.detected_crossunders = {}
        self.detected_crossovers = {}
        
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
        Check if price has gained more than min_price_change% within a rolling window.
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
            candles_per_day = int(24 * 60 / 5)  # 40 candles per day (5-min timeframe)

            # Calculate how many candles to look back - exactly 24 hours
            lookback_candles = candles_per_day * days

            # Limit to available data
            actual_lookback = min(lookback_candles, len(df) - 1)

            # Get recent data to analyze - exactly last 24 hours worth of data
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
                        if max_gain_pct >= self.min_price_change * 1.5:
                            break

            # Check if gain exceeds minimum threshold (now 20%)
            has_gained = max_gain_pct >= self.min_price_change

            return has_gained, max_gain_pct, best_low_price, best_high_price, best_low_idx, best_high_idx

        except Exception as e:
            logger.info(f"Error while checking price gain for {self.symbol}: {e}")
            return False, 0.0, None, None, None, None
    
    def check_price_decrease(self, df, days=None):
        """
        Check if price has decreased more than min_price_change% within a rolling window.
        This function searches for the maximum decrease from any high to any subsequent low
        within the specified period. Optimized for performance.
        
        Args:
            df: DataFrame with OHLC data
            days: Number of days to look back (uses self.lookback_days if None)
                    
        Returns:
            Tuple of (has_decreased, decrease_pct, high_price, low_price, high_idx, low_idx)
        """
        try:
            if days is None:
                days = self.lookback_days

            if df is None or len(df) < 10:  # Need some minimal data
                return False, 0.0, None, None, None, None

            # Calculate candles per day for 5-min timeframe
            candles_per_day = int(24 * 60 / 5)  # 40 candles per day (5-min timeframe)

            # Calculate how many candles to look back - exactly 24 hours
            lookback_candles = candles_per_day * days

            # Limit to available data
            actual_lookback = min(lookback_candles, len(df) - 1)

            # Get recent data to analyze - exactly last 24 hours worth of data
            recent_data = df.iloc[-actual_lookback:]

            # Extract necessary data as NumPy arrays for faster processing
            lows = recent_data['low'].values
            highs = recent_data['high'].values
            indices = recent_data.index

            n = len(highs)

            # Use NumPy to find the maximum possible decrease
            max_decrease_pct = 0.0
            best_high_price = None
            best_low_price = None
            best_high_idx = None
            best_low_idx = None

            # For each potential high point
            for i in range(n - 1):
                high_price = highs[i]
                high_idx = indices[i]

                # Skip invalid high prices
                if high_price <= 0:
                    continue

                # Use vectorized operations for all subsequent low prices
                subsequent_lows = lows[i+1:]
                subsequent_indices = indices[i+1:]

                # Calculate decrease percentages for all subsequent lows in one operation
                decreases = ((high_price - subsequent_lows) / high_price) * 100

                if len(decreases) > 0:
                    # Find the maximum decrease and its index
                    max_decrease_idx = np.argmax(decreases)
                    current_max_decrease = decreases[max_decrease_idx]

                    # Update if we found a better decrease
                    if current_max_decrease > max_decrease_pct:
                        max_decrease_pct = current_max_decrease
                        best_high_price = high_price
                        best_low_price = subsequent_lows[max_decrease_idx]
                        best_high_idx = high_idx
                        best_low_idx = subsequent_indices[max_decrease_idx]

                        # If we've found a decrease that significantly exceeds our criteria, we can stop early
                        if max_decrease_pct >= self.min_price_change * 1.5:
                            break

            # Check if decrease exceeds minimum threshold (now 20%)
            has_decreased = max_decrease_pct >= self.min_price_change

            return has_decreased, max_decrease_pct, best_high_price, best_low_price, best_high_idx, best_low_idx

        except Exception as e:
            logger.info(f"Error while checking price decrease for {self.symbol}: {e}")
            return False, 0.0, None, None, None, None

    def check_kama_crossunder(self, df, lookback_candles=864):
        """
        Detect when price crosses below kama with special attention to whether
        this is the first crossunder after a significant high point and
        after a sustained period above the kama.
        """
        # Standard initialization
        has_crossunder = False
        crossunder_candle_age = 0
        minutes_ago = 0
        is_first_crossunder = False
        
        try:
            # Ensure adequate data
            if df is None or len(df) < self.kama_period:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            # First, identify if there was a significant gain and find the high point
            has_gain, gain_pct, low_price, high_price, low_idx, high_idx = self.check_price_gain(df)
            
            # If no significant gain or no high point identified, don't proceed
            if not has_gain or high_idx is None:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            # Find or ensure kama column exists
            kama_columns = [col for col in df.columns if col.startswith('kama_')]
            if not kama_columns:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
                
            kama_column = kama_columns[0]
            
            # Create price vs kama relationship (above/below)
            df_valid = df.dropna(subset=['close', kama_column]).copy()
            df_valid['above_kama'] = df_valid['close'] > df_valid[kama_column]
            
            # Find all crossunder points (transition from above to below)
            crossunder_points = []
            for i in range(1, len(df_valid)):
                if df_valid['above_kama'].iloc[i-1] and not df_valid['above_kama'].iloc[i]:
                    crossunder_points.append((df_valid.index[i], i))
            
            if not crossunder_points:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
            
            # Define how many consecutive candles price should be above kama to consider it a valid setup
            required_candles_above_kama = self.entry_sustained_candles
            
            # Define the minimum time between valid crossunders (to prevent multiple signals)
            min_candles_between_crossunders = 12  # At least 12 candles (1 hour for 5-min candles)
            
            # Find valid crossunders - ones where price has been above kama for 'required_candles_above_kama'
            # and that occur after the high point
            valid_crossunders = []
            
            for crossunder_idx, (timestamp, idx) in enumerate(crossunder_points):
                # Skip if this crossunder is before the high point
                if timestamp <= high_idx:
                    continue
                
                # Check if there are enough candles to look back
                if idx < required_candles_above_kama + 1:
                    continue
                
                # Check if price was consistently above kama for the required period before the crossunder
                all_above_kama = True
                for j in range(1, required_candles_above_kama + 1):
                    if not df_valid['above_kama'].iloc[idx - j]:
                        all_above_kama = False
                        break
                
                # Only add if price was consistently above kama
                if all_above_kama:
                    # Check if this is far enough from the previous valid crossunder
                    if valid_crossunders and (idx - valid_crossunders[-1][1]) < min_candles_between_crossunders:
                        # Skip if too close to previous valid crossunder
                        continue
                    
                    valid_crossunders.append((timestamp, idx))
            
            if not valid_crossunders:
                return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
            
            # We found valid crossunders after the high point
            has_crossunder = True
            
            # Get the most recent valid crossunder
            first_timestamp, first_idx = valid_crossunders[-1]
            
            # Check if this is truly the first valid crossunder in the recent period
            is_first_crossunder = len(valid_crossunders) == 1 or (
                len(valid_crossunders) > 1 and 
                first_idx - valid_crossunders[-2][1] >= min_candles_between_crossunders * 2
            )
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
            logger.error(f"{self.symbol}: Error in check_kama_crossunder: {e}")
            return False, 0, 0, False
    
    def check_kama_crossover(self, df, lookback_candles=864):
        """
        Detect when price crosses above kama with special attention to whether
        this is the first crossover after a significant low point and
        after a sustained period below the kama.
        """
        # Standard initialization
        has_crossover = False
        crossover_candle_age = 0
        minutes_ago = 0
        is_first_crossover = False
        
        try:
            # Ensure adequate data
            if df is None or len(df) < self.kama_period:
                return has_crossover, crossover_candle_age, minutes_ago, is_first_crossover
                
            # First, identify if there was a significant decrease and find the low point
            has_decreased, decrease_pct, high_price, low_price, high_idx, low_idx = self.check_price_decrease(df)
            
            # If no significant decrease or no low point identified, don't proceed
            if not has_decreased or low_idx is None:
                return has_crossover, crossover_candle_age, minutes_ago, is_first_crossover
                
            # Find or ensure kama column exists
            kama_columns = [col for col in df.columns if col.startswith('kama_')]
            if not kama_columns:
                return has_crossover, crossover_candle_age, minutes_ago, is_first_crossover
                
            kama_column = kama_columns[0]
            
            # Create price vs kama relationship (above/below)
            df_valid = df.dropna(subset=['close', kama_column]).copy()
            df_valid['below_kama'] = df_valid['close'] < df_valid[kama_column]
            
            # Find all crossover points (transition from below to above)
            crossover_points = []
            for i in range(1, len(df_valid)):
                if df_valid['below_kama'].iloc[i-1] and not df_valid['below_kama'].iloc[i]:
                    crossover_points.append((df_valid.index[i], i))
            
            if not crossover_points:
                return has_crossover, crossover_candle_age, minutes_ago, is_first_crossover
            
            # Define how many consecutive candles price should be below kama to consider it a valid setup
            required_candles_below_kama = self.entry_sustained_candles
            
            # Define the minimum time between valid crossovers (to prevent multiple signals)
            min_candles_between_crossovers = 12  # At least 12 candles (1 hour for 5-min candles)
            
            # Find valid crossovers - ones where price has been below kama for 'required_candles_below_kama'
            # and that occur after the low point
            valid_crossovers = []
            
            for crossover_idx, (timestamp, idx) in enumerate(crossover_points):
                # Skip if this crossover is before the low point
                if timestamp <= low_idx:
                    continue
                
                # Check if there are enough candles to look back
                if idx < required_candles_below_kama + 1:
                    continue
                
                # Check if price was consistently below kama for the required period before the crossover
                all_below_kama = True
                for j in range(1, required_candles_below_kama + 1):
                    if not df_valid['below_kama'].iloc[idx - j]:
                        all_below_kama = False
                        break
                
                # Only add if price was consistently below kama
                if all_below_kama:
                    # Check if this is far enough from the previous valid crossover
                    if valid_crossovers and (idx - valid_crossovers[-1][1]) < min_candles_between_crossovers:
                        # Skip if too close to previous valid crossover
                        continue
                    
                    valid_crossovers.append((timestamp, idx))
            
            if not valid_crossovers:
                return has_crossover, crossover_candle_age, minutes_ago, is_first_crossover
            
            # We found valid crossovers after the low point
            has_crossover = True
            
            # Get the most recent valid crossover
            first_timestamp, first_idx = valid_crossovers[-1]
            
            # Check if this is truly the first valid crossover in the recent period
            is_first_crossover = len(valid_crossovers) == 1 or (
                len(valid_crossovers) > 1 and 
                first_idx - valid_crossovers[-2][1] >= min_candles_between_crossovers * 2
            )
            
            # Calculate candle age and time
            crossover_candle_age = len(df_valid) - 1 - first_idx
            
            # Calculate minutes ago
            current_time = df.index[-1]
            if isinstance(current_time, pd.Timestamp) and isinstance(first_timestamp, pd.Timestamp):
                time_diff = current_time - first_timestamp
                minutes_ago = int(time_diff.total_seconds() / 60)
            else:
                minutes_ago = crossover_candle_age * 5  # Assume 5-min candles
                
            return has_crossover, crossover_candle_age, minutes_ago, is_first_crossover
            
        except Exception as e:
            logger.error(f"{self.symbol}: Error in check_kama_crossover: {e}")
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
        # if num_candles is None:
        #     num_candles = self.exit_sustained_candles
            
        # if df is None or len(df) < num_candles:
        #     return False
        
        # # Get just the most recent num_candles
        # recent_candles = df.iloc[-num_candles:]
        
        # # Check if ALL of these candles are above the level
        # all_above = all(recent_candles['close'] > level)
        
        # return all_above
        return True

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
        # if num_candles is None:
        #     num_candles = self.exit_sustained_candles
            
        # if df is None or len(df) < num_candles:
        #     return False
        
        # # Get just the most recent num_candles
        # recent_candles = df.iloc[-num_candles:]
        
        # # Check if ALL of these candles are below the level
        # all_below = all(recent_candles['close'] < level)
        
        # return all_below
        return True
        
    def check_supertrend_bearish(self, df, supertrend_data=None):
        """
        Check if the Supertrend indicator is bearish (red).
        First checks if SuperTrend direction column already exists in the dataframe,
        only calculates it if necessary.
        """
        if df is None or len(df) < self.supertrend_length + 1:
            return False
            
        # First, check if SuperTrend direction column already exists in the dataframe
        supertrend_dir_columns = [col for col in df.columns if col.startswith('SUPERTd_')]
        
        if supertrend_dir_columns:
            # Use existing SuperTrend column (already calculated in fetch_extended_historical_data)
            direction_col = supertrend_dir_columns[0]
            current_direction = df[direction_col].iloc[-1]
            # Direction -1 means bearish (red)
            return current_direction == -1
            
        # If SuperTrend not found in dataframe, check supertrend_data or calculate it
        if supertrend_data is None:
            # Calculate Supertrend using pandas_ta only if necessary
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
    
    def check_supertrend_bullish(self, df, supertrend_data=None):
        """
        Check if the Supertrend indicator is bullish (green).
        First checks if SuperTrend direction column already exists in the dataframe,
        only calculates it if necessary.
        """
        if df is None or len(df) < self.supertrend_length + 1:
            return False
            
        # First, check if SuperTrend direction column already exists in the dataframe
        supertrend_dir_columns = [col for col in df.columns if col.startswith('SUPERTd_')]
        
        if supertrend_dir_columns:
            # Use existing SuperTrend column (already calculated in fetch_extended_historical_data)
            direction_col = supertrend_dir_columns[0]
            current_direction = df[direction_col].iloc[-1]
            # Direction 1 means bullish (green)
            return current_direction == 1
            
        # If SuperTrend not found in dataframe, check supertrend_data or calculate it
        if supertrend_data is None:
            # Calculate Supertrend using pandas_ta only if necessary
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
        
        # Direction 1 means bullish (green)
        return current_direction == 1

    # NEW METHOD: Detect reversal signals    
    def detect_reversal(self, df, position_type):
        """
        Detect if market is showing signs of reversal against current position
        
        Args:
            df: DataFrame with OHLC data including indicators
            position_type: 'long' or 'short' - current position direction
            
        Returns:
            tuple: (is_reversal, reversal_strength, reversal_reason)
                is_reversal: Boolean indicating if a reversal is detected
                reversal_strength: 0-100 score indicating strength of reversal signal
                reversal_reason: String explanation of the reversal signal
        """
        # Initialize
        is_reversal = False
        reversal_strength = 0  # 0-100 scale
        reversal_reason = ""
        
        try:
            # Ensure we have enough data
            if df is None or len(df) < max(self.kama_period, self.supertrend_length + 5):
                return False, 0, "Insufficient data for reversal detection"
                
            # Check for kama cross events based on position type
            if position_type == 'short':
                # For shorts, check for price crossing above kama (bullish reversal)
                has_crossover, crossover_age, minutes_ago, is_first = self.check_kama_crossover(df)
                is_supertrend_bullish = self.check_supertrend_bullish(df)
                
                # Calculate reversal strength - base score from crossover
                if has_crossover:
                    # Base score for crossover - more recent crossovers are stronger
                    reversal_strength += max(0, 40 - crossover_age)
                    
                    if is_supertrend_bullish:
                        reversal_strength += 30  # Supertrend confirms reversal
                        reversal_reason = f"Price crossed above kama({self.kama_period}) {minutes_ago} minutes ago with bullish supertrend"
                    else:
                        reversal_reason = f"Price crossed above kama({self.kama_period}) {minutes_ago} minutes ago"
                    
                    # Check recent candle momentum (last 3 candles)
                    if len(df) >= 3:
                        recent_candles = df.iloc[-3:]
                        bullish_candles = sum(1 for i in range(len(recent_candles)) 
                                            if recent_candles['close'].iloc[i] > recent_candles['open'].iloc[i])
                        
                        if bullish_candles >= 2:
                            reversal_strength += 15  # Mostly bullish recent candles
                            reversal_reason += " with bullish momentum"
                    
                    # Check volume trend
                    if 'volume' in df.columns and len(df) >= 6:
                        vol_last3 = df['volume'].iloc[-3:].mean()
                        vol_prev3 = df['volume'].iloc[-6:-3].mean()
                        
                        if vol_last3 > vol_prev3 * 1.5:
                            reversal_strength += 15  # Increasing volume supports reversal
                            reversal_reason += " on increasing volume"
                
                # Consider this a reversal if strength is high enough
                is_reversal = reversal_strength >= 60
                
            elif position_type == 'long':
                # For longs, check for price crossing below kama (bearish reversal)
                has_crossunder, crossunder_age, minutes_ago, is_first = self.check_kama_crossunder(df)
                is_supertrend_bearish = self.check_supertrend_bearish(df)
                
                # Calculate reversal strength - base score from crossunder
                if has_crossunder:
                    # Base score for crossunder - more recent crossunders are stronger
                    reversal_strength += max(0, 40 - crossunder_age)
                    
                    if is_supertrend_bearish:
                        reversal_strength += 30  # Supertrend confirms reversal
                        reversal_reason = f"Price crossed below kama({self.kama_period}) {minutes_ago} minutes ago with bearish supertrend"
                    else:
                        reversal_reason = f"Price crossed below kama({self.kama_period}) {minutes_ago} minutes ago"
                    
                    # Check recent candle momentum (last 3 candles)
                    if len(df) >= 3:
                        recent_candles = df.iloc[-3:]
                        bearish_candles = sum(1 for i in range(len(recent_candles)) 
                                            if recent_candles['close'].iloc[i] < recent_candles['open'].iloc[i])
                        
                        if bearish_candles >= 2:
                            reversal_strength += 15  # Mostly bearish recent candles
                            reversal_reason += " with bearish momentum"
                    
                    # Check volume trend
                    if 'volume' in df.columns and len(df) >= 6:
                        vol_last3 = df['volume'].iloc[-3:].mean()
                        vol_prev3 = df['volume'].iloc[-6:-3].mean()
                        
                        if vol_last3 > vol_prev3 * 1.5:
                            reversal_strength += 15  # Increasing volume supports reversal
                            reversal_reason += " on increasing volume"
                
                # Consider this a reversal if strength is high enough
                is_reversal = reversal_strength >= 60
            
            return is_reversal, reversal_strength, reversal_reason
            
        except Exception as e:
            logger.error(f"{self.symbol}: Error in detect_reversal: {e}")
            return False, 0, f"Error detecting reversal: {e}"
    
    def determine_stop_loss(self, df, position_type='short', entry_price=None):
        """
        Determine stop-loss level based on position type.
        For shorts, stop loss is set above entry price with buffer.
        For longs, stop loss is set below entry price with buffer.
        If entry_price is None, falls back to using kama-based stop loss.
        
        Args:
            df: DataFrame with OHLC data
            position_type: 'short' or 'long'
            entry_price: The entry price of the position, if available
                
        Returns:
            float: Stop-loss price level or None if cannot be determined
        """
        try:
            if df is None or len(df) < self.kama_period:
                logger.debug(f"{self.symbol}: Insufficient data for stop loss calculation")
                return None
            
            # If entry_price is provided, use it to calculate stop loss
            if entry_price is not None and entry_price > 0:
                if position_type == 'short':
                    # Set stop-loss sl_buffer_pct% above entry price for shorts
                    stop_loss = entry_price * (1 + self.sl_buffer_pct / 100)
                else:  # long
                    # Set stop-loss sl_buffer_pct% below entry price for longs
                    stop_loss = entry_price * (1 - self.sl_buffer_pct / 100)
                
                return stop_loss
                
            # Fallback to kama-based stop loss if entry_price is not provided
            # Check if kama already exists in the dataframe
            kama_columns = [col for col in df.columns if col.startswith('kama_')]
            
            # Use existing kama if available
            if kama_columns:
                kama_column = kama_columns[0]
                current_kama = df[kama_column].iloc[-1]
            else:
                # Calculate kama if not present
                kama_series = df.ta.kama(length=self.kama_period)
                if kama_series.empty:
                    logger.debug(f"{self.symbol}: Failed to calculate kama")
                    return None
                current_kama = kama_series.iloc[-1]
            
            # Make sure kama is valid
            if pd.isna(current_kama) or current_kama <= 0:
                logger.debug(f"{self.symbol}: Invalid kama value: {current_kama}")
                return None
            
            if position_type == 'short':
                # Set stop-loss sl_buffer_pct% above kama for shorts
                stop_loss = current_kama * (1 + self.sl_buffer_pct / 100)
            else:  # long
                # Set stop-loss sl_buffer_pct% below kama for longs
                stop_loss = current_kama * (1 - self.sl_buffer_pct / 100)
            
            return stop_loss
        
        except Exception as e:
            logger.error(f"{self.symbol}: Error determining stop loss: {e}")
            return None
      
    def generate_exit_signal(self, df, current_position, entry_price, stop_loss):
        """
        Generate exit signal based on position type and current market conditions.
        For shorts: Exit when price is above a specific percentage from entry price
        For longs: Exit when price is below a specific percentage from entry price
        
        Args:
            df: DataFrame with OHLC data
            current_position: String indicating position type ('long' or 'short')
            entry_price: Original entry price
            stop_loss: Current stop loss level (used as fallback)
            
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
            
            # Define exit percentage thresholds
            short_exit_pct = self.sl_buffer_pct  # Exit short if price is 2% above entry price
            long_exit_pct = self.sl_buffer_pct   # Exit long if price is 2% below entry price
            
            # Initialize result
            exit_signal = {
                'signal': 'none',
                'reason': 'conditions_not_met',
                'price': current_price,
                'exit_triggered': False
            }
            
            # NEW: Check for reversal signals
            is_reversal, reversal_strength, reversal_reason = self.detect_reversal(df, current_position)
            
            # If we have a strong reversal signal, exit the position
            if is_reversal and reversal_strength >= 70:
                exit_signal['signal'] = f'exit_{current_position}'
                exit_signal['exit_triggered'] = True
                exit_signal['reason'] = f'Strong reversal detected: {reversal_reason}' 
                exit_signal['reversal_strength'] = reversal_strength
                exit_signal['execute_reversal'] = True  # Signal to execute reversal after exit
                return exit_signal
            
            # Check for exit conditions based on price movement from entry
            if current_position == 'short':
                # Calculate percentage change from entry price
                if entry_price <= 0:
                    return exit_signal  # Invalid entry price
                    
                price_change_pct = ((current_price - entry_price) / entry_price) * 100
                exit_threshold = entry_price * (1 + short_exit_pct/100)
                
                # Exit if price has moved up by the threshold percentage
                if price_change_pct >= short_exit_pct:
                    # Check if this movement is sustained for multiple candles
                    sustained_move = self.check_sustained_above_level(df, exit_threshold)
                    
                    if sustained_move:
                        exit_signal['signal'] = 'exit_short'
                        exit_signal['exit_triggered'] = True
                        exit_signal['reason'] = f'Price {current_price} is {price_change_pct:.2f}% above entry price {entry_price} (threshold: {short_exit_pct}%)'
                        
                        # If also seeing reversal signals, suggest reversal
                        exit_signal['execute_reversal'] = (is_reversal and reversal_strength >= 50)
                
                # Make sure stop loss is valid (above entry price for shorts)
                if stop_loss is not None and stop_loss <= entry_price:
                    # Correct stop loss if it's not above entry price
                    logger.warning(f"Invalid stop loss for SHORT position: {stop_loss} <= entry price {entry_price}. Correcting.")
                    stop_loss = entry_price * (1 + self.sl_buffer_pct/100)
                    
                # Fallback to stop loss check if no percentage-based exit
                if stop_loss is not None and current_price >= stop_loss:
                    sustained_above_sl = self.check_sustained_above_level(df, stop_loss)
                    
                    if sustained_above_sl:
                        exit_signal['signal'] = 'exit_short'
                        exit_signal['exit_triggered'] = True
                        exit_signal['reason'] = f'Price {current_price} sustainably above stop loss {stop_loss}'
                        exit_signal['execute_reversal'] = False  # Don't reverse on stop loss hit
                
            elif current_position == 'long':
                # Calculate percentage change from entry price
                if entry_price <= 0:
                    return exit_signal  # Invalid entry price
                    
                price_change_pct = ((entry_price - current_price) / entry_price) * 100
                exit_threshold = entry_price * (1 - long_exit_pct/100)
                
                # Exit if price has moved down by the threshold percentage
                if price_change_pct >= long_exit_pct:
                    # Check if this movement is sustained for multiple candles
                    sustained_move = self.check_sustained_below_level(df, exit_threshold)
                    
                    if sustained_move:
                        exit_signal['signal'] = 'exit_long'
                        exit_signal['exit_triggered'] = True
                        exit_signal['reason'] = f'Price {current_price} is {price_change_pct:.2f}% below entry price {entry_price} (threshold: {long_exit_pct}%)'
                        
                        # If also seeing reversal signals, suggest reversal
                        exit_signal['execute_reversal'] = (is_reversal and reversal_strength >= 50)
                
                # Make sure stop loss is valid (below entry price for longs)
                if stop_loss is not None and stop_loss >= entry_price:
                    # Correct stop loss if it's not below entry price
                    logger.warning(f"Invalid stop loss for LONG position: {stop_loss} >= entry price {entry_price}. Correcting.")
                    stop_loss = entry_price * (1 - self.sl_buffer_pct/100)
                    
                # Fallback to stop loss check if no percentage-based exit
                if stop_loss is not None and current_price <= stop_loss:
                    sustained_below_sl = self.check_sustained_below_level(df, stop_loss)
                    
                    if sustained_below_sl:
                        exit_signal['signal'] = 'exit_long'
                        exit_signal['exit_triggered'] = True
                        exit_signal['reason'] = f'Price {current_price} sustainably below stop loss {stop_loss}'
                        exit_signal['execute_reversal'] = False  # Don't reverse on stop loss hit
            
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
        Otherwise, looks for entry signals (both short and long).
        
        Args:
            current_position: Optional - current position type ('long', 'short', or None)
            entry_price: Optional - entry price if in a position
            stop_loss: Optional - current stop loss level if in a position
            
        Returns:
            dict: Signal information
        """
        try:
            # Ensure we have enough data
            if self.price_history.empty or len(self.price_history) < self.kama_period:
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
            if current_position is not None and entry_price is not None:
                # If a stop loss is provided, use it; otherwise recalculate based on entry price
                if stop_loss is None:
                    stop_loss = self.determine_stop_loss(df, position_type=current_position, entry_price=entry_price)
                    
                return self.generate_exit_signal(df, current_position, entry_price, stop_loss)
            
            # Initialize variables for entry signals
            current_price = df['close'].iloc[-1]
            
            # Check if SuperTrend direction column already exists in the dataframe
            supertrend_dir_columns = [col for col in df.columns if col.startswith('SUPERTd_')]
            
            # Only calculate SuperTrend if it doesn't already exist in the dataframe
            supertrend_data = None
            if not supertrend_dir_columns:
                try:
                    supertrend_data = df.ta.supertrend(
                        length=self.supertrend_length,
                        multiplier=self.supertrend_factor
                    )
                except Exception as e:
                    logger.error(f"{self.symbol}: Error calculating Supertrend: {e}")
                    supertrend_data = None
                    
            # Find or calculate kama if not present
            kama_columns = [col for col in df.columns if col.startswith('kama_')]
            if not kama_columns:
                # Calculate kama if not present
                kama_series = df.ta.kama(length=self.kama_period)
                if not kama_series.empty:
                    df[f'kama_{self.kama_period}'] = kama_series
                    current_kama = kama_series.iloc[-1]
                else:
                    current_kama = None
            else:
                kama_column = kama_columns[0]
                current_kama = df[kama_column].iloc[-1]
                    
            is_below_kama = current_price < current_kama if current_kama is not None else False
            is_above_kama = current_price > current_kama if current_kama is not None else False
                    
            # First, check for short signal conditions
            # 1. Check for price gain - 20% in 24 hours
            has_price_gain, gain_pct, low_price, high_price, low_idx, high_idx = self.check_price_gain(df)
            
            # 2. Check for kama crossunder
            has_kama_crossunder, crossunder_age, minutes_ago_crossunder, is_first_crossunder = self.check_kama_crossunder(df)
            
            # 3. Check supertrend bearish
            is_supertrend_bearish = self.check_supertrend_bearish(df, supertrend_data)
            
            # *** Check for crossunder on current or previous 6 candles ***
            is_recent_crossunder = crossunder_age <= 3  # 0-6 = current or up to 6 candles ago
            
            # All conditions for a sell (short) signal
            short_conditions_met = {
                'price_gain_met': has_price_gain,
                'price_gain_pct': f"{gain_pct:.2f}%" if has_price_gain else "N/A",
                'high_price': high_price,
                'low_price': low_price,
                'kama_crossunder_met': has_kama_crossunder,
                'is_first_crossunder': is_first_crossunder,  # Still track this but don't require it
                'is_recent_crossunder': is_recent_crossunder,
                'crossunder_age': crossunder_age,
                'crossunder_minutes_ago': minutes_ago_crossunder,
                'is_below_kama': is_below_kama,
                'supertrend_bearish_met': is_supertrend_bearish
            }
            
            # Next, check for long signal conditions
            # 1. Check for price decrease - 20% in 24 hours
            has_price_decrease, decrease_pct, decrease_high_price, decrease_low_price, decrease_high_idx, decrease_low_idx = self.check_price_decrease(df)
            
            # 2. Check for kama crossover
            has_kama_crossover, crossover_age, minutes_ago_crossover, is_first_crossover = self.check_kama_crossover(df)
            
            # *** Check for crossover on current or previous 6 candles ***
            is_recent_crossover = crossover_age <= 6  # 0-6 = current or up to 6 candles ago
            
            # 3. Check supertrend bullish
            is_supertrend_bullish = self.check_supertrend_bullish(df, supertrend_data)
            
            # All conditions for a buy (long) signal
            long_conditions_met = {
                'price_decrease_met': has_price_decrease,
                'price_decrease_pct': f"{decrease_pct:.2f}%" if has_price_decrease else "N/A",
                'high_price': decrease_high_price,
                'low_price': decrease_low_price,
                'kama_crossover_met': has_kama_crossover,
                'is_first_crossover': is_first_crossover,  # Still track this but don't require it
                'is_recent_crossover': is_recent_crossover,
                'crossover_age': crossover_age,
                'crossover_minutes_ago': minutes_ago_crossover,
                'is_above_kama': is_above_kama,
                'supertrend_bullish_met': is_supertrend_bullish
            }
            
            # Calculate drawdown/recovery for both scenarios
            if has_price_gain and high_price:
                drawdown_pct = ((high_price - current_price) / high_price) * 100 if high_price > 0 else 0
                short_conditions_met['drawdown_pct'] = drawdown_pct
                
                # Add this check: Calculate the ratio of drawdown to gain
                drawdown_to_gain_ratio = drawdown_pct / gain_pct if gain_pct > 0 else 0
                short_conditions_met['drawdown_to_gain_ratio'] = drawdown_to_gain_ratio

            if has_price_decrease and decrease_low_price:
                recovery_pct = ((current_price - decrease_low_price) / decrease_low_price) * 100 if decrease_low_price > 0 else 0
                long_conditions_met['recovery_pct'] = recovery_pct
            
            # Generate SHORT signal - REMOVED first crossunder requirement
            if (has_price_gain and is_supertrend_bearish and is_below_kama):
                
                # Add this check: Skip trades that have already retraced too much of their gain
                drawdown_to_gain_ratio = short_conditions_met.get('drawdown_to_gain_ratio', 0)
                if drawdown_to_gain_ratio > 0.7:  # If retraced more than 70% of the gain
                    logger.info(f"Skipping SHORT signal for {self.symbol}: Price already retraced {drawdown_pct:.2f}% which is {drawdown_to_gain_ratio:.2f}x of its {gain_pct:.2f}% gain")
                    return {
                        'signal': 'none',
                        'reason': f'price_already_retraced_too_much (drawdown {drawdown_pct:.2f}% is {drawdown_to_gain_ratio:.2f}x of gain)',
                        'price': current_price,
                        'stop_loss': None,
                        'short_conditions_met': short_conditions_met,
                        'long_conditions_met': long_conditions_met,
                        'exit_triggered': False,
                        'kama_value': current_kama
                    }
                # For new positions, use current price as the intended entry price
                # When generating a signal, the position isn't open yet
                stop_loss = self.determine_stop_loss(df, position_type='short', entry_price=current_price)
                
                if stop_loss is None or stop_loss <= 0:
                    return {
                        'signal': 'none',
                        'reason': 'invalid_stop_loss',
                        'price': current_price,
                        'stop_loss': None,
                        'short_conditions_met': short_conditions_met,
                        'long_conditions_met': long_conditions_met,
                        'exit_triggered': False
                    }
                
                # Generate sell signal
                signal = {
                    'signal': 'sell',
                    'reason': f"Gained {gain_pct:.2f}% in the last 24 hours and now {drawdown_pct:.2f}% down from high, "
                            f"Crossunder below kama({self.kama_period}) within last 6 candles (age: {crossunder_age}), "
                            f"supertrend bearish",
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'short_conditions_met': short_conditions_met,
                    'long_conditions_met': long_conditions_met,
                    'has_kama_crossunder': has_kama_crossunder,
                    'is_first_crossunder': is_first_crossunder,  # Include this info but it's not required
                    'is_recent_crossunder': is_recent_crossunder,
                    'crossunder_age': crossunder_age,
                    'crossunder_minutes_ago': minutes_ago_crossunder,
                    'exit_triggered': False,
                    'kama_value': current_kama
                }
                
                # Log additional debug info for this valid signal
                logger.info(f"VALID SHORT SIGNAL for {self.symbol}: Crossunder within last 6 candles, "
                        f"on candle {crossunder_age}, {minutes_ago_crossunder} minutes ago, drawdown: {drawdown_pct:.2f}%, "
                        f"Entry: {current_price}, SL: {stop_loss} ({self.sl_buffer_pct}% above entry)")
                
                return signal
            
            # ENHANCEMENT: Generate LONG signal as well
            if (has_price_decrease and has_kama_crossover 
                and is_above_kama and is_supertrend_bullish):
                
                recovery_pct = long_conditions_met.get('recovery_pct', 0)
                
                # Similar check for long - skip if already recovered too much
                if recovery_pct > decrease_pct * 0.7:  # If recovered more than 70% of the decrease
                    logger.info(f"Skipping LONG signal for {self.symbol}: Price already recovered {recovery_pct:.2f}% from its {decrease_pct:.2f}% drop")
                    return {
                        'signal': 'none',
                        'reason': f'price_already_recovered_too_much',
                        'price': current_price,
                        'stop_loss': None,
                        'short_conditions_met': short_conditions_met,
                        'long_conditions_met': long_conditions_met,
                        'exit_triggered': False,
                        'kama_value': current_kama
                    }
                
                stop_loss = self.determine_stop_loss(df, position_type='long', entry_price=current_price)
                
                if stop_loss is None or stop_loss <= 0:
                    return {
                        'signal': 'none',
                        'reason': 'invalid_stop_loss',
                        'price': current_price,
                        'stop_loss': None,
                        'short_conditions_met': short_conditions_met,
                        'long_conditions_met': long_conditions_met,
                        'exit_triggered': False
                    }
                
                # Generate buy signal
                signal = {
                    'signal': 'buy',
                    'reason': f"Dropped {decrease_pct:.2f}% recently and now {recovery_pct:.2f}% up from low, "
                            f"Crossover above kama({self.kama_period}) within last 6 candles (age: {crossover_age}), "
                            f"supertrend bullish",
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'short_conditions_met': short_conditions_met,
                    'long_conditions_met': long_conditions_met,
                    'has_kama_crossover': has_kama_crossover,
                    'is_first_crossover': is_first_crossover,
                    'is_recent_crossover': is_recent_crossover,
                    'crossover_age': crossover_age,
                    'crossover_minutes_ago': minutes_ago_crossover,
                    'exit_triggered': False,
                    'kama_value': current_kama
                }
                
                logger.info(f"VALID LONG SIGNAL for {self.symbol}: Crossover within last 6 candles, "
                        f"on candle {crossover_age}, {minutes_ago_crossover} minutes ago, recovery: {recovery_pct:.2f}%, "
                        f"Entry: {current_price}, SL: {stop_loss} ({self.sl_buffer_pct}% below entry)")
                
                return signal
            
            # No signal if conditions are not met
            if has_kama_crossunder and not is_recent_crossunder:
                reason = f"crossunder_not_within_last_6_candles (age: {crossunder_age} candles)"
            elif has_kama_crossover and not is_recent_crossover:
                reason = f"crossover_not_within_last_6_candles (age: {crossover_age} candles)"
            else:
                reason = 'not_all_conditions_met'
                
            return {
                'signal': 'none',
                'reason': reason,
                'price': current_price,
                'stop_loss': None,
                'short_conditions_met': short_conditions_met,
                'long_conditions_met': long_conditions_met,
                'exit_triggered': False,
                'kama_value': current_kama
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