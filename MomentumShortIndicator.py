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
    Indicator for identifying short selling opportunities in crypto markets.
    
    Criteria:
    1. Coin has gained 20%+ in last 1-3 days (using 5-min candles)
    2. Price has crossed below EMA(288) on 5-min candle
    3. Supertrend indicator is red (bearish) on 5-min candle
    """
    
    def __init__(self, 
                symbol="UNKNOWN",
                lookback_days=3,
                min_price_gain=20.0,  # 20% minimum gain
                ema_period=144,       # EMA period for 5-min candles (288 = 24 hours)
                supertrend_factor=3.0, # Supertrend multiplier
                supertrend_length=10,  # Supertrend period
                sl_buffer_pct=1.0,     # SL buffer percent above EMA
                **kwargs):
        
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.min_price_gain = min_price_gain
        self.ema_period = ema_period
        self.supertrend_factor = supertrend_factor
        self.supertrend_length = supertrend_length
        self.sl_buffer_pct = sl_buffer_pct
        
        # Internal tracking
        self.price_history = pd.DataFrame()
        self.signal_history = []
        
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

            # logger.debug(f"{self.symbol}: Maximum gain found: {max_gain_pct:.2f}% "
            #             f"from {best_low_price:.6f} to {best_high_price:.6f}")

            return has_gained, max_gain_pct, best_low_price, best_high_price, best_low_idx, best_high_idx

        except Exception as e:
            logger.info(f"Error while checking price gain for {self.symbol}: {e}")
            return False, 0.0, None, None, None, None

    
    # In MomentumShortIndicator.py:

    def check_ema_crossunder(self, df, ema_values, lookback_candles=30, sustained_period=5, above_threshold=0.7):
        """
        Improved method to detect the first meaningful crossunder after a price rally.
        Uses a rolling analysis of price-to-EMA relationship.
        
        Args:
            df: DataFrame with OHLC data
            ema_values: Series with EMA values aligned with df
            lookback_candles: Max candles to look back
            sustained_period: Minimum number of consecutive candles price should be above EMA
                            to consider a crossunder significant (maintained for compatibility)
            above_threshold: Minimum ratio to consider significant (maintained for compatibility)
                    
        Returns:
            tuple: (has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder)
        """
        has_crossunder = False
        crossunder_candle_age = 0
        minutes_ago = 0
        is_first_crossunder = False
        crossunder_index = None
        
        # Ensure we have enough data
        if len(df) < 10 or len(ema_values) < 10:
            return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
        
        # Limit lookback to available data
        lookback = min(lookback_candles, len(df) - 1)
        
        # Create a Series showing price-to-EMA relationship
        # 1 = price above EMA, -1 = price below EMA
        price_ema_relationship = []
        for i in range(-lookback, 0):
            if abs(i) >= len(df) or abs(i) >= len(ema_values):
                # Skip if indices are out of bounds
                continue
            price_ema_relationship.append(
                (df.index[i], 1 if df['close'].iloc[i] > ema_values.iloc[i] else -1)
            )
        
        # Convert to Series
        if not price_ema_relationship:
            return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
            
        price_ema_series = pd.Series(
            [x[1] for x in price_ema_relationship],
            index=[x[0] for x in price_ema_relationship]
        )
        
        # Find where crossunders occur (relationship changes from 1 to -1)
        crossunder_locations = []
        for i in range(1, len(price_ema_series)):
            if price_ema_series.iloc[i-1] == 1 and price_ema_series.iloc[i] == -1:
                crossunder_locations.append(i)
        
        # If no crossunders found, return early
        if not crossunder_locations:
            return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
        
        # Now find the first meaningful crossunder after a sustained bullish period
        # We'll use sustained_period as the minimum consecutive candles above EMA
        min_candles_above = max(sustained_period, 5)  # Use at least 5 candles
        
        # For each crossunder, check how long price was above EMA before it
        crossunder_significance = []
        
        for location in crossunder_locations:
            if location < 1:  # Need at least one candle before crossunder
                continue
                
            # Look backward from this crossunder to find how long price was consistently above EMA
            consecutive_above = 0
            for k in range(location-1, -1, -1):  # Looking backward
                if k < 0 or k >= len(price_ema_series):
                    break
                    
                if price_ema_series.iloc[k] == 1:
                    consecutive_above += 1
                else:
                    break  # Stop counting when we find price below EMA
            
            # Store this crossunder with its significance
            crossunder_significance.append((location, consecutive_above))
        
        # A crossunder is "significant" if price was above EMA for at least min_candles_above
        significant_crossunders = [x for x in crossunder_significance if x[1] >= min_candles_above]
        
        if significant_crossunders:
            # Take the most recent significant crossunder
            most_recent = max(significant_crossunders, key=lambda x: x[0])
            location, significance = most_recent
            
            # Calculate the candle age relative to the end of the data
            crossunder_candle_age = len(price_ema_series) - location
            
            # This is our "first meaningful crossunder"
            has_crossunder = True
            is_first_crossunder = True
            
            # Calculate time ago if we have datetime index
            if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    crossunder_time = price_ema_series.index[location]
                    current_time = df.index[-1]
                    time_diff = current_time - crossunder_time
                    minutes_ago = int(time_diff.total_seconds() / 60)
                except (IndexError, TypeError, AttributeError) as e:
                    # Fallback: estimate based on timeframe
                    minutes_ago = crossunder_candle_age * 5
            else:
                # Fallback: estimate based on timeframe
                minutes_ago = crossunder_candle_age * 5
        
        return has_crossunder, crossunder_candle_age, minutes_ago, is_first_crossunder
    
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
            
            # Supertrend returns a DataFrame with columns like:
            # SUPERT_7_3.0 (trend value) and SUPERTd_7_3.0 (direction)
            # Column names depend on the parameters used
        
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
    
    def determine_stop_loss(self, df, ema_series=None):
        """
        Determine stop-loss level above EMA.
        """
        if df is None or len(df) < self.ema_period:
            return None
            
        if ema_series is None:
            # Calculate EMA using pandas_ta
            ema_series = df.ta.ema(length=self.ema_period)
        
        # Get current EMA value
        if ema_series.empty:
            return None
            
        current_ema = ema_series.iloc[-1]
        
        # Set stop-loss sl_buffer_pct% above EMA
        stop_loss = current_ema * (1 + self.sl_buffer_pct / 100)
        
        return stop_loss
    
    # In MomentumShortIndicator.py:

    def generate_signal(self):
        """
        Generate trading signal based on criteria:
        1. 20%+ price gain in 1-3 days
        2. Price crossed below EMA(288) for the first time after being above it
        3. Supertrend is bearish (red)
        """
        try:
            # Ensure we have enough data
            if self.price_history.empty or len(self.price_history) < max(self.ema_period, self.supertrend_length) + 1:
                logger.warning(f"{self.symbol}: Insufficient price history for signal generation")
                return {
                    'signal': 'none',
                    'reason': 'insufficient_data',
                    'price': None,
                    'stop_loss': None,
                    'conditions_met': {}
                }
            
            # Make a copy of the data to avoid pandas warning about setting values on a copy
            df = self.price_history.copy()
            
            # Initialize variables with default values in case calculations fail
            current_price = df['close'].iloc[-1] if not df.empty else None
            current_ema = None
            is_below_ema = False
            has_ema_crossunder = False
            crossunder_age = 0
            minutes_ago = 0
            is_first_crossunder = False
            ema_price_diff_pct = 0
            has_price_gain = False
            gain_pct = 0
            low_price = None
            high_price = None
            low_idx = None
            high_idx = None
            drawdown_pct = 0
            is_supertrend_bearish = False
            high_minutes_ago = 0
            
            # Calculate technical indicators using pandas_ta with error handling
            try:
                ema_series = df.ta.ema(length=self.ema_period)
                current_ema = ema_series.iloc[-1] if not ema_series.empty else None
                is_below_ema = current_price < current_ema if current_ema is not None else False
            except Exception as e:
                logger.error(f"{self.symbol}: Error calculating EMA: {e}")
                ema_series = pd.Series()
            
            try:
                supertrend_data = df.ta.supertrend(
                    length=self.supertrend_length,
                    multiplier=self.supertrend_factor
                )
            except Exception as e:
                logger.error(f"{self.symbol}: Error calculating Supertrend: {e}")
                supertrend_data = pd.DataFrame()
            
            # Check for EMA crossunder with error handling
            try:
                has_ema_crossunder, crossunder_age, minutes_ago, is_first_crossunder = self.check_ema_crossunder(
                    df, 
                    ema_series,
                    lookback_candles=30,  # Look back 30 candles (2.5 hours for 5-min candles)
                    sustained_period=6,   # Number of candles to check for predominant behavior 
                    above_threshold=0.8   # 80% of candles must be above EMA to confirm first crossunder
                )
            except Exception as e:
                logger.error(f"{self.symbol}: Error checking EMA crossunder: {e}")
                has_ema_crossunder = False
                crossunder_age = 0
                minutes_ago = 0
                is_first_crossunder = False
            
            # Calculate EMA-to-price percentage
            try:
                ema_price_diff_pct = ((current_ema - current_price) / current_price) * 100 if current_ema and current_price > 0 else 0
            except Exception as e:
                logger.error(f"{self.symbol}: Error calculating EMA-price difference: {e}")
                ema_price_diff_pct = 0
            
            # Use the improved check_price_gain method with error handling
            try:
                has_price_gain, gain_pct, low_price, high_price, low_idx, high_idx = self.check_price_gain(df)
            except Exception as e:
                logger.error(f"{self.symbol}: Error checking price gain: {e}")
                has_price_gain = False
                gain_pct = 0
                low_price = None
                high_price = None
                low_idx = None
                high_idx = None
            
            # Calculate current drawdown from the high
            try:
                drawdown_pct = ((high_price - current_price) / high_price) * 100 if high_price and high_price > 0 else 0
            except Exception as e:
                logger.error(f"{self.symbol}: Error calculating drawdown: {e}")
                drawdown_pct = 0
            
            # Check if supertrend is bearish
            try:
                is_supertrend_bearish = self.check_supertrend_bearish(df, supertrend_data)
            except Exception as e:
                logger.error(f"{self.symbol}: Error checking Supertrend: {e}")
                is_supertrend_bearish = False
            
            # Calculate how long ago the high occurred (in minutes)
            try:
                if high_idx is not None and isinstance(high_idx, pd.Timestamp):
                    current_time = df.index[-1]
                    if isinstance(current_time, pd.Timestamp):
                        time_diff = current_time - high_idx
                        high_minutes_ago = int(time_diff.total_seconds() / 60)
            except Exception as e:
                logger.error(f"{self.symbol}: Error calculating time since high: {e}")
                high_minutes_ago = 0
            
            # Track conditions
            conditions_met = {
                'price_gain_met': has_price_gain,
                'price_gain_pct': f"{gain_pct:.2f}%",
                'low_price': low_price,
                'high_price': high_price,
                'drawdown_pct': drawdown_pct,
                'high_minutes_ago': high_minutes_ago,
                'ema_crossunder_met': has_ema_crossunder,
                'is_first_crossunder': is_first_crossunder,
                'crossunder_age': crossunder_age,
                'crossunder_minutes_ago': minutes_ago,
                'supertrend_bearish_met': is_supertrend_bearish,
                'is_below_ema': is_below_ema,
                'ema_price_diff_pct': ema_price_diff_pct
            }
            
            # All conditions must be met for a sell signal
            # Focus on the first-time crossunder condition
            if has_price_gain and is_supertrend_bearish:
                # Check that we have a valid, recent crossunder
                if has_ema_crossunder and is_first_crossunder:
                    # Only consider recent crossunders (within last 3 candles) for immediate entry
                    if crossunder_age <= 3:
                        # Determine stop-loss with error handling
                        try:
                            stop_loss = self.determine_stop_loss(df, ema_series)
                            if stop_loss is None or stop_loss <= 0:
                                logger.error(f"{self.symbol}: Invalid stop loss calculated: {stop_loss}")
                                return {
                                    'signal': 'none',
                                    'reason': 'invalid_stop_loss',
                                    'price': current_price,
                                    'stop_loss': None,
                                    'conditions_met': conditions_met,
                                    'timestamp': df.index[-1]
                                }
                        except Exception as e:
                            logger.error(f"{self.symbol}: Error determining stop loss: {e}")
                            return {
                                'signal': 'none',
                                'reason': 'stop_loss_error',
                                'price': current_price,
                                'stop_loss': None,
                                'conditions_met': conditions_met,
                                'timestamp': df.index[-1]
                            }
                        
                        # Generate sell signal with detailed reason
                        signal = {
                            'signal': 'sell',
                            'reason': f"Coin gained {gain_pct:.2f}% (from {low_price:.6f} to {high_price:.6f}), "
                                    f"now {drawdown_pct:.2f}% down from high, "
                                    f"just crossed below EMA({self.ema_period}) for the first time, "
                                    f"Supertrend bearish",
                            'price': current_price,
                            'stop_loss': stop_loss,
                            'conditions_met': conditions_met,
                            'ema_value': current_ema,
                            'has_ema_crossunder': has_ema_crossunder,
                            'is_first_crossunder': is_first_crossunder,
                            'crossunder_age': crossunder_age,
                            'crossunder_minutes_ago': minutes_ago,
                            'supertrend_direction': 'bearish',
                            'timestamp': df.index[-1] if not df.empty else None
                        }
                        
                        # Track signal history with error handling
                        try:
                            self.signal_history.append({
                                'timestamp': df.index[-1] if not df.empty else None,
                                'signal': 'sell',
                                'price': current_price,
                                'stop_loss': stop_loss,
                                'reason': signal['reason']
                            })
                        except Exception as e:
                            logger.error(f"{self.symbol}: Error updating signal history: {e}")
                        
                        logger.info(f"{self.symbol}: SELL SIGNAL - {signal['reason']}")
                        return signal
            
            # No signal if conditions are not met
            return {
                'signal': 'none',
                'reason': 'conditions_not_met',
                'price': current_price,
                'stop_loss': None,
                'conditions_met': conditions_met,
                'timestamp': df.index[-1] if not df.empty else None
            }
            
        except Exception as e:
            logger.exception(f"{self.symbol}: Unexpected error in generate_signal: {e}")
            return {
                'signal': 'none',
                'reason': f'error: {str(e)}',
                'price': None,
                'stop_loss': None,
                'conditions_met': {},
                'timestamp': None
            }
# Function to integrate with the main trading system
def create_momentum_short_signal_generator(symbol, **kwargs):
    """Factory function to create a momentum short signal generator."""
    return MomentumShortIndicator(symbol=symbol, **kwargs)

