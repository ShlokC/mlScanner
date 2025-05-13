import numpy as np
import pandas as pd
import logging
import time
from exchange import ExchangeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('grid_logic')

class GridBot:
    def __init__(self, config, exchange_client=None):
        """
        Initialize the Grid Trading Bot with configuration
        
        Args:
            config (dict): Configuration parameters for the grid bot
            exchange_client (ExchangeClient, optional): Exchange client instance
        """
        self.config = config
        self.exchange = exchange_client or ExchangeClient()
        self.symbol = config['symbol']
        self.direction = config['direction']
        
        # Configure price range
        self.lower_price = float(config['price_range']['lower']) if config['price_range']['lower'] else 0
        self.upper_price = float(config['price_range']['upper']) if config['price_range']['upper'] else 0
        
        # Grid configuration
        self.grid_number = int(config['grid']['number']) if config['grid']['number'] else 10
        self.grid_type = config['grid']['type']  # 'Arithmetic' or 'Geometric'
        
        # Investment details
        self.currency = config['investment']['currency']
        self.leverage = config['investment']['leverage'].replace('x', '')  # Remove 'x' to get the numeric value
        self.leverage = float(self.leverage)
        self.investment_percentage = config['investment']['amount']  # This is a percentage value
        
        # Track orders and grid levels
        self.grid_levels = []
        self.active_orders = {}
        
        # Support and resistance levels
        self.support_levels = []
        self.resistance_levels = []
    def find_optimal_grid_settings(self, current_price, grid_count_range=(3, 5), spacing_range=(0.008, 0.018)):
        """
        Find optimal grid settings using ranges for grid count and spacing
        
        Args:
            current_price (float): Current market price
            grid_count_range (tuple): Min and max grid count (e.g., (3, 5))
            spacing_range (tuple): Min and max spacing as percentage (e.g., (0.008, 0.018))
        
        Returns:
            dict: Optimal grid settings or None if using default
        """
        logger.info(f"Finding optimal grid settings for {self.symbol}")
        logger.info(f"Grid count range: {grid_count_range}, Spacing range: {spacing_range[0]*100:.1f}%-{spacing_range[1]*100:.1f}%")
        
        # Get historical data
        df = self.exchange.fetch_binance_data(self.symbol, timeframe='5m', limit=200, include_current=False)
        
        if df.empty:
            logger.warning(f"No data retrieved for {self.symbol}")
            return None
        
        # Find all possible S/R levels
        support_levels, resistance_levels = self._find_support_resistance_levels(df, current_price)
        
        # Find the best combination that fits our grid requirements
        best_combination = self._find_best_sr_combination(
            support_levels, 
            resistance_levels, 
            current_price,
            grid_count_range,
            spacing_range
        )
        
        return best_combination
 
    def _find_support_resistance_levels(self, df, current_price):
        """Find support and resistance levels using simple fractal method"""
        if len(df) < 20:
            return [], []
        
        # Simple bin size based on price range
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / 200  # 200 bins for the whole range
        
        # Track levels with their strength
        support_levels = {}  # price -> strength
        resistance_levels = {}  # price -> strength
        
        # Find fractal points
        window = 2
        for i in range(window, len(df) - window):
            # Support fractal: low point surrounded by higher lows
            if self._is_fractal_low(df, i, window):
                price = df['low'].iloc[i]
                level = round(price / bin_size) * bin_size
                strength = self._calculate_level_strength(df, price, i, is_support=True)
                
                if level not in support_levels:
                    support_levels[level] = 0
                support_levels[level] += strength
            
            # Resistance fractal: high point surrounded by lower highs
            if self._is_fractal_high(df, i, window):
                price = df['high'].iloc[i]
                level = round(price / bin_size) * bin_size
                strength = self._calculate_level_strength(df, price, i, is_support=False)
                
                if level not in resistance_levels:
                    resistance_levels[level] = 0
                resistance_levels[level] += strength
        
        # Add recent highs and lows with extra strength
        recent_data = df.tail(50)
        
        # Recent high
        recent_high = recent_data['high'].max()
        recent_high_level = round(recent_high / bin_size) * bin_size
        if recent_high_level not in resistance_levels:
            resistance_levels[recent_high_level] = 0
        resistance_levels[recent_high_level] += 3.0
        
        # Recent low
        recent_low = recent_data['low'].min()
        recent_low_level = round(recent_low / bin_size) * bin_size
        if recent_low_level not in support_levels:
            support_levels[recent_low_level] = 0
        support_levels[recent_low_level] += 3.0
        
        # Filter close levels and convert to sorted lists
        filtered_supports = self._filter_close_levels(support_levels, current_price * 0.005)
        filtered_resistances = self._filter_close_levels(resistance_levels, current_price * 0.005)
        
        # Sort by strength and filter by current price
        supports = [price for price, strength in filtered_supports.items() 
                    if price < current_price and strength >= 1]
        resistances = [price for price, strength in filtered_resistances.items() 
                    if price > current_price and strength >= 1]
        
        # Sort by strength (not distance)
        supports.sort(key=lambda x: filtered_supports[x], reverse=True)
        resistances.sort(key=lambda x: filtered_resistances[x], reverse=True)
        
        return supports[:10], resistances[:10]
    def _is_fractal_low(self, df, i, window):
        """Check if index i is a fractal low"""
        current_low = df['low'].iloc[i]
        
        # Check left side
        for j in range(1, window + 1):
            if current_low >= df['low'].iloc[i - j]:
                return False
        
        # Check right side
        for j in range(1, window + 1):
            if current_low >= df['low'].iloc[i + j]:
                return False
        
        return True

    def _is_fractal_high(self, df, i, window):
        """Check if index i is a fractal high"""
        current_high = df['high'].iloc[i]
        
        # Check left side
        for j in range(1, window + 1):
            if current_high <= df['high'].iloc[i - j]:
                return False
        
        # Check right side
        for j in range(1, window + 1):
            if current_high <= df['high'].iloc[i + j]:
                return False
        
        return True

    def _calculate_level_strength(self, df, price, index, is_support=True):
        """Calculate the strength of a support/resistance level"""
        strength = 1.0  # Base strength
        
        # Volume bonus
        avg_volume = df['volume'].rolling(20).mean().iloc[index]
        current_volume = df['volume'].iloc[index]
        if current_volume > avg_volume * 1.2:
            strength += 0.5
        
        # Count touches
        touches = self._count_level_touches(df, price, tolerance=price * 0.002)
        strength += touches * 0.3
        
        # Check reaction strength
        reaction_strength = self._check_level_reaction(df, index, is_support)
        strength += reaction_strength
        
        return strength

    def _count_level_touches(self, df, price, tolerance):
        """Count how many times price touches a level"""
        touches = 0
        
        for i in range(len(df)):
            if abs(df['low'].iloc[i] - price) <= tolerance or abs(df['high'].iloc[i] - price) <= tolerance:
                touches += 1
        
        return touches

    def _check_level_reaction(self, df, index, is_support=True):
        """Check how strong the reaction is at a level"""
        if index >= len(df) - 3:
            return 0
        
        reaction_strength = 0
        
        if is_support:
            current_low = df['low'].iloc[index]
            for i in range(1, min(4, len(df) - index)):
                if df['close'].iloc[index + i] > current_low * 1.005:
                    reaction_strength += 0.2
                    break
        else:
            current_high = df['high'].iloc[index]
            for i in range(1, min(4, len(df) - index)):
                if df['close'].iloc[index + i] < current_high * 0.995:
                    reaction_strength += 0.2
                    break
        
        return reaction_strength

    def _filter_close_levels(self, levels_dict, tolerance):
        """Remove levels that are too close to each other"""
        if not levels_dict:
            return {}
        
        sorted_levels = sorted(levels_dict.items(), key=lambda x: x[1], reverse=True)
        filtered = {}
        
        for price, strength in sorted_levels:
            is_too_close = False
            for selected_price in filtered:
                if abs(price - selected_price) <= tolerance:
                    is_too_close = True
                    break
            
            if not is_too_close:
                filtered[price] = strength
        
        return filtered

    def _find_best_sr_combination(self, support_levels, resistance_levels, current_price, grid_count_range, spacing_range):
        """Find the best support/resistance combination within specified ranges"""
        
        # Filter levels near current price
        max_distance = current_price * 0.15  # 15% max distance
        
        valid_supports = [s for s in support_levels if current_price - s <= max_distance and s < current_price]
        valid_resistances = [r for r in resistance_levels if r - current_price <= max_distance and r > current_price]
        
        if not valid_supports or not valid_resistances:
            return None
        
        best_combination = None
        best_score = float('inf')
        
        min_grids, max_grids = grid_count_range
        min_spacing, max_spacing = spacing_range
        
        # Calculate optimal spacing (middle of range)
        optimal_spacing = (min_spacing + max_spacing) / 2
        
        # Try all grid counts in the range
        for grid_count in range(min_grids, max_grids + 1):
            # Try all support/resistance combinations
            for support in valid_supports[:5]:  # Limit to top 5 for performance
                for resistance in valid_resistances[:5]:
                    # Calculate actual range and spacing
                    actual_range_pct = (resistance - support) / support
                    actual_spacing = actual_range_pct / grid_count
                    
                    # Check if spacing is within our desired range
                    if not (min_spacing <= actual_spacing <= max_spacing):
                        continue
                    
                    # Score based on how close we are to optimal spacing
                    spacing_diff = abs(actual_spacing - optimal_spacing)
                    
                    # Prefer combinations closer to current price
                    distance_from_current = (abs(current_price - support) + abs(current_price - resistance)) / (2 * current_price)
                    
                    # Prefer round number levels
                    round_bonus = 0
                    if self._is_round_number(support, current_price) or self._is_round_number(resistance, current_price):
                        round_bonus = -0.1
                    
                    # Prefer spacing closer to the middle of our range
                    spacing_position = (actual_spacing - min_spacing) / (max_spacing - min_spacing)
                    spacing_bonus = -abs(spacing_position - 0.5) * 0.1  # Bonus for being near middle
                    
                    # Combined score (lower is better)
                    total_score = spacing_diff + distance_from_current * 0.5 + round_bonus + spacing_bonus
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_combination = {
                            'support': support,
                            'resistance': resistance,
                            'grid_count': grid_count,
                            'spacing': actual_spacing,
                            'range_pct': actual_range_pct,
                            'score': total_score
                        }
        
        return best_combination

    def _is_round_number(self, price, reference_price):
        """Check if a price is a psychological round number"""
        if reference_price < 0.001:
            round_numbers = [0.0001, 0.0005, 0.001, 0.005]
        elif reference_price < 0.01:
            round_numbers = [0.001, 0.005, 0.01]
        elif reference_price < 0.1:
            round_numbers = [0.01, 0.05]
        else:
            round_numbers = [0.1, 0.5, 1.0]
        
        for rn in round_numbers:
            if abs(price % rn) < rn * 0.01:
                return True
        return False

    def create_grid_based_range(self, current_price, grid_count_range=(3, 5), spacing_range=(0.008, 0.018)):
        """Create price range based on grid requirements within specified ranges"""
        
        # Use middle values from ranges for default calculation
        grid_count = (grid_count_range[0] + grid_count_range[1]) // 2  # e.g., (3+5)//2 = 4
        target_spacing = (spacing_range[0] + spacing_range[1]) / 2      # e.g., (0.008+0.018)/2 = 0.013
        
        required_range_pct = grid_count * target_spacing
        
        # Center the range around current price
        lower_price = current_price * (1 - required_range_pct / 2)
        upper_price = current_price * (1 + required_range_pct / 2)
        
        return {
            'support': lower_price,
            'resistance': upper_price,
            'grid_count': grid_count,
            'spacing': target_spacing,
            'range_pct': required_range_pct
        }
    def analyze_market(self, timeframe='5m', limit=288):
        """
        Analyze market to identify support and resistance levels using direct price methods
        
        Args:
            timeframe (str): Timeframe for analysis (e.g., '5m', '15m', '1h')
            limit (int): Number of candles to analyze
            
        Returns:
            tuple: (support_levels, resistance_levels)
        """
        logger.info(f"Analyzing market data for {self.symbol} on {timeframe} timeframe")
        
        # Fetch historical data
        df = self.exchange.fetch_binance_data(self.symbol, timeframe=timeframe, limit=limit, include_current=False)
        
        if df.empty:
            logger.warning(f"No data retrieved for {self.symbol}")
            return [], []
        
        # Get current price to reference against
        ticker = self.exchange.fetch_ticker(self.symbol)
        if not ticker or 'last' not in ticker:
            logger.warning(f"Could not fetch current price for {self.symbol}")
            return [], []
            
        current_price = ticker['last']
        logger.info(f"Current price for {self.symbol}: {current_price}")
        
        # Find support and resistance using touchpoint analysis
        support_levels, resistance_levels = find_touchpoint_levels(df, current_price, 1)
        
        # Also get fractal levels for additional confirmation
        fractal_supports, fractal_resistances = find_price_fractals(df)
        
        # Combine the results
        all_supports = support_levels + fractal_supports
        all_resistances = resistance_levels + fractal_resistances
        
        # Remove duplicates and sort by proximity to current price
        final_supports = remove_duplicates(all_supports, tolerance=0.005)
        final_resistances = remove_duplicates(all_resistances, tolerance=0.005)
        
        # Sort by distance to current price
        final_supports = sorted(final_supports, key=lambda x: abs(current_price - x))
        final_resistances = sorted(final_resistances, key=lambda x: abs(current_price - x))
        
        # Filter out levels that are too far from current price
        max_distance = current_price * 0.15  # 15% maximum distance
        final_supports = [s for s in final_supports if current_price - s <= max_distance]
        final_resistances = [r for r in final_resistances if r - current_price <= max_distance]
        
        # Filter supports below current price and resistances above current price
        final_supports = [s for s in final_supports if s < current_price]
        final_resistances = [r for r in final_resistances if r > current_price]
        
        # Filter out levels with too low strength
        if len(final_supports) > 0 and len(final_resistances) > 0:
            logger.info(f"Found {len(final_supports)} support levels and {len(final_resistances)} resistance levels")
            logger.info(f"Support levels: {final_supports}")
            logger.info(f"Resistance levels: {final_resistances}")
        else:
            logger.warning("No strong support/resistance levels found, using default percentages")
            final_supports = [current_price * 0.97]
            final_resistances = [current_price * 1.03]
        
        self.support_levels = final_supports
        self.resistance_levels = final_resistances
        
        return final_supports, final_resistances
        
    def calculate_grid_levels(self):
        """
        Calculate grid levels based on the price range and number of grids
        
        Returns:
            list: Grid price levels
        """
        levels = []
        
        # Use identified support and resistance if available
        if self.support_levels and self.resistance_levels:
            # Filter support and resistance within our price range
            filtered_support = [s for s in self.support_levels if self.lower_price <= s <= self.upper_price]
            filtered_resistance = [r for r in self.resistance_levels if self.lower_price <= r <= self.upper_price]
            
            # Combine and sort the levels
            key_levels = sorted(filtered_support + filtered_resistance)
            
            # If we have enough key levels, use them directly
            if len(key_levels) >= self.grid_number - 1:
                # Select evenly distributed key levels
                indices = np.linspace(0, len(key_levels) - 1, self.grid_number - 1, dtype=int)
                selected_levels = [key_levels[i] for i in indices]
                
                # Add upper and lower bounds
                levels = [self.lower_price] + selected_levels + [self.upper_price]
            else:
                # We don't have enough key levels, so include them all and fill the gaps
                levels = [self.lower_price] + key_levels + [self.upper_price]
                
                # Calculate how many more levels we need
                remaining = self.grid_number + 1 - len(levels)
                
                if remaining > 0:
                    # Add remaining levels based on grid type
                    if self.grid_type == 'Arithmetic':
                        # Arithmetic grid (equal price difference)
                        filled_levels = self.generate_arithmetic_grid(self.lower_price, self.upper_price, self.grid_number + 1)
                    else:
                        # Geometric grid (equal percentage difference)
                        filled_levels = self.generate_geometric_grid(self.lower_price, self.upper_price, self.grid_number + 1)
                    
                    # Combine with the key levels
                    all_levels = sorted(set(levels + filled_levels))
                    
                    # Select the required number of levels
                    indices = np.linspace(0, len(all_levels) - 1, self.grid_number + 1, dtype=int)
                    levels = [all_levels[i] for i in indices]
        else:
            # No support/resistance levels, generate grid based on type
            if self.grid_type == 'Arithmetic':
                # Arithmetic grid (equal price difference)
                levels = self.generate_arithmetic_grid(self.lower_price, self.upper_price, self.grid_number + 1)
            else:
                # Geometric grid (equal percentage difference)
                levels = self.generate_geometric_grid(self.lower_price, self.upper_price, self.grid_number + 1)
        
        self.grid_levels = levels
        return levels
        
    def generate_arithmetic_grid(self, lower_price, upper_price, num_levels):
        """
        Generate arithmetic grid levels (equal price difference)
        
        Args:
            lower_price (float): Lower price bound
            upper_price (float): Upper price bound
            num_levels (int): Number of grid levels
            
        Returns:
            list: Grid price levels
        """
        return np.linspace(lower_price, upper_price, num_levels).tolist()
        
    def generate_geometric_grid(self, lower_price, upper_price, num_levels):
        """
        Generate geometric grid levels (equal percentage difference)
        
        Args:
            lower_price (float): Lower price bound
            upper_price (float): Upper price bound
            num_levels (int): Number of grid levels
            
        Returns:
            list: Grid price levels
        """
        return np.geomspace(lower_price, upper_price, num_levels).tolist()
        
    def create_grid_orders(self):
        """
        Create grid orders based on calculated levels
        
        Returns:
            dict: Created orders
        """
        # First calculate grid levels if not already done
        if not self.grid_levels:
            self.calculate_grid_levels()
            
        # Get available balance
        balance = self.exchange.get_balance()
        available_balance = balance.get(self.currency, {}).get('free', 0)
        
        # Calculate investment amount based on percentage
        investment_amount = available_balance * (self.investment_percentage / 100)
        
        # Calculate order size (per grid)
        order_size = investment_amount / self.grid_number
        
        orders = {}
        
        # Create buy and sell orders based on direction
        if self.direction == 'Neutral':
            # For neutral, create both buy and sell orders
            for i in range(len(self.grid_levels) - 1):
                # Buy order at lower grid level
                buy_price = self.grid_levels[i]
                buy_qty = order_size / buy_price
                
                buy_order = self.exchange.create_order(
                    self.symbol,
                    'limit',
                    'buy',
                    buy_qty,
                    buy_price
                )
                
                if buy_order:
                    orders[buy_order['id']] = buy_order
                
                # Sell order at upper grid level
                sell_price = self.grid_levels[i + 1]
                sell_qty = order_size / sell_price
                
                sell_order = self.exchange.create_order(
                    self.symbol,
                    'limit',
                    'sell',
                    sell_qty,
                    sell_price
                )
                
                if sell_order:
                    orders[sell_order['id']] = sell_order
                    
        elif self.direction == 'Long':
            # For long, create only buy orders
            for i in range(len(self.grid_levels)):
                buy_price = self.grid_levels[i]
                buy_qty = order_size / buy_price
                
                buy_order = self.exchange.create_order(
                    self.symbol,
                    'limit',
                    'buy',
                    buy_qty,
                    buy_price
                )
                
                if buy_order:
                    orders[buy_order['id']] = buy_order
                    
        elif self.direction == 'Short':
            # For short, create only sell orders
            for i in range(len(self.grid_levels)):
                sell_price = self.grid_levels[i]
                sell_qty = order_size / sell_price
                
                sell_order = self.exchange.create_order(
                    self.symbol,
                    'limit',
                    'sell',
                    sell_qty,
                    sell_price
                )
                
                if sell_order:
                    orders[sell_order['id']] = sell_order
        
        self.active_orders = orders
        return orders
        
    def start(self):
        """
        Start the grid trading bot
        
        Returns:
            bool: Success or failure
        """
        try:
            # Analyze market to find support and resistance
            self.analyze_market()
            
            # Calculate grid levels
            self.calculate_grid_levels()
            
            # Create grid orders
            self.create_grid_orders()
            
            logger.info(f"Grid bot started for {self.symbol} with {len(self.grid_levels)} levels")
            logger.info(f"Price range: {self.lower_price} - {self.upper_price}")
            logger.info(f"Grid type: {self.grid_type}")
            logger.info(f"Direction: {self.direction}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to start grid bot: {e}")
            return False
    
    def stop(self):
        """
        Stop the grid trading bot and cancel all orders
        
        Returns:
            bool: Success or failure
        """
        try:
            # Cancel all active orders
            for order_id in self.active_orders:
                self.exchange.cancel_order(order_id, self.symbol)
            
            logger.info(f"Grid bot stopped for {self.symbol}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to stop grid bot: {e}")
            return False


# ----- Support & Resistance Detection Functions -----

def find_touchpoint_levels(df, current_price, min_touches=1):
        """
        Simple support and resistance detection based on price touchpoints and strength
        
        Args:
            df (DataFrame): OHLCV dataframe with price data
            current_price (float): Current market price
            min_touches (int): Minimum number of touches required
        
        Returns:
            tuple: (support_levels, resistance_levels) sorted by strength
        """
        if len(df) < 20:
            return [], []
        
        # Simple bin size based on price range
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / 200  # 200 bins for the whole range
        
        # Track levels with their strength
        support_levels = {}  # price -> strength
        resistance_levels = {}  # price -> strength
        
        # Find fractal points (simple method)
        window = 2
        for i in range(window, len(df) - window):
            # Support fractal: low point surrounded by higher lows
            if is_fractal_low(df, i, window):
                price = df['low'].iloc[i]
                level = round(price / bin_size) * bin_size
                
                # Calculate strength based on:
                # 1. How many times it's tested
                # 2. Volume at the level
                # 3. How well it holds
                strength = calculate_level_strength(df, price, i, is_support=True)
                
                if level not in support_levels:
                    support_levels[level] = 0
                support_levels[level] += strength
            
            # Resistance fractal: high point surrounded by lower highs
            if is_fractal_high(df, i, window):
                price = df['high'].iloc[i]
                level = round(price / bin_size) * bin_size
                
                strength = calculate_level_strength(df, price, i, is_support=False)
                
                if level not in resistance_levels:
                    resistance_levels[level] = 0
                resistance_levels[level] += strength
        
        # Add recent highs and lows with extra strength
        recent_data = df.tail(50)
        
        # Recent high
        recent_high = recent_data['high'].max()
        recent_high_level = round(recent_high / bin_size) * bin_size
        if recent_high_level not in resistance_levels:
            resistance_levels[recent_high_level] = 0
        resistance_levels[recent_high_level] += 3.0  # Extra strength for recent levels
        
        # Recent low
        recent_low = recent_data['low'].min()
        recent_low_level = round(recent_low / bin_size) * bin_size
        if recent_low_level not in support_levels:
            support_levels[recent_low_level] = 0
        support_levels[recent_low_level] += 3.0  # Extra strength for recent levels
        
        # Filter levels that are too close to each other
        filtered_supports = filter_close_levels(support_levels, current_price * 0.005)  # 0.5% tolerance
        filtered_resistances = filter_close_levels(resistance_levels, current_price * 0.005)
        
        # Sort by strength and filter by current price
        strong_supports = [(price, strength) for price, strength in filtered_supports.items() 
                        if price < current_price and strength >= min_touches]
        strong_resistances = [(price, strength) for price, strength in filtered_resistances.items() 
                            if price > current_price and strength >= min_touches]
        
        # Sort by strength (highest first)
        strong_supports.sort(key=lambda x: x[1], reverse=True)
        strong_resistances.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the price levels (sorted by strength)
        supports = [price for price, strength in strong_supports[:10]]
        resistances = [price for price, strength in strong_resistances[:10]]
        
        return supports, resistances
def is_fractal_low(df, i, window):
    """Check if index i is a fractal low"""
    current_low = df['low'].iloc[i]
    
    # Check left side
    for j in range(1, window + 1):
        if current_low >= df['low'].iloc[i - j]:
            return False
    
    # Check right side
    for j in range(1, window + 1):
        if current_low >= df['low'].iloc[i + j]:
            return False
    
    return True

def is_fractal_high(df, i, window):
    """Check if index i is a fractal high"""
    current_high = df['high'].iloc[i]
    
    # Check left side
    for j in range(1, window + 1):
        if current_high <= df['high'].iloc[i - j]:
            return False
    
    # Check right side
    for j in range(1, window + 1):
        if current_high <= df['high'].iloc[i + j]:
            return False
    
    return True
def calculate_level_strength(df, price, index, is_support=True):
    """Calculate the strength of a support/resistance level"""
    strength = 1.0  # Base strength
    
    # Volume bonus
    avg_volume = df['volume'].rolling(20).mean().iloc[index]
    current_volume = df['volume'].iloc[index]
    if current_volume > avg_volume * 1.2:  # 20% above average
        strength += 0.5
    
    # Test how many times this level is touched
    touches = count_level_touches(df, price, tolerance=price * 0.002)  # 0.2% tolerance
    strength += touches * 0.3
    
    # Check if it's a strong level (good bounces/rejections)
    reaction_strength = check_level_reaction(df, index, is_support)
    strength += reaction_strength
    
    return strength
def count_level_touches(df, price, tolerance):
    """Count how many times price touches a level"""
    touches = 0
    
    for i in range(len(df)):
        if abs(df['low'].iloc[i] - price) <= tolerance or abs(df['high'].iloc[i] - price) <= tolerance:
            touches += 1
    
    return touches
def check_level_reaction(df, index, is_support=True):
    """Check how strong the reaction is at a level"""
    if index >= len(df) - 3:  # Not enough data to check reaction
        return 0
    
    reaction_strength = 0
    
    if is_support:
        # For support, check if price bounced up
        current_low = df['low'].iloc[index]
        for i in range(1, min(4, len(df) - index)):
            if df['close'].iloc[index + i] > current_low * 1.005:  # 0.5% bounce
                reaction_strength += 0.2
                break
    else:
        # For resistance, check if price rejected down
        current_high = df['high'].iloc[index]
        for i in range(1, min(4, len(df) - index)):
            if df['close'].iloc[index + i] < current_high * 0.995:  # 0.5% rejection
                reaction_strength += 0.2
                break
    
    return reaction_strength
def filter_close_levels(levels_dict, tolerance):
    """Remove levels that are too close to each other, keeping the stronger one"""
    if not levels_dict:
        return {}
    
    sorted_levels = sorted(levels_dict.items(), key=lambda x: x[1], reverse=True)
    filtered = {}
    
    for price, strength in sorted_levels:
        # Check if this level is too close to an already selected level
        is_too_close = False
        for selected_price in filtered:
            if abs(price - selected_price) <= tolerance:
                is_too_close = True
                break
        
        if not is_too_close:
            filtered[price] = strength
    
    return filtered

def validate_sr_levels(df, supports, resistances, validation_threshold=0.3):
    """Simple validation - just check if levels are reasonable"""
    validated_supports = []
    validated_resistances = []
    
    # For supports: remove any that are too far below recent lows
    recent_low = df.tail(100)['low'].min()
    for level in supports:
        if level >= recent_low * 0.9:  # Within 10% of recent low
            validated_supports.append(level)
    
    # For resistances: remove any that are too far above recent highs
    recent_high = df.tail(100)['high'].max()
    for level in resistances:
        if level <= recent_high * 1.1:  # Within 10% of recent high
            validated_resistances.append(level)
    
    # Ensure we have at least something
    if not validated_supports and supports:
        validated_supports = supports[:2]
    if not validated_resistances and resistances:
        validated_resistances = resistances[:2]
    
    return validated_supports, validated_resistances
def find_price_fractals(df, window_size=2):
    """
    Find price fractals (significant swing points with confirmation)
    
    Args:
        df (DataFrame): OHLCV dataframe
        window_size (int): Window size for fractal detection
    
    Returns:
        tuple: (support_fractals, resistance_fractals)
    """
    support_fractals = []
    resistance_fractals = []
    
    # Each price fractal has a center point with window_size candles on each side
    for i in range(window_size, len(df) - window_size):
        # Check for support fractal (low point surrounded by higher lows)
        is_support = True
        for j in range(1, window_size + 1):
            if df['low'].iloc[i] >= df['low'].iloc[i-j] or df['low'].iloc[i] >= df['low'].iloc[i+j]:
                is_support = False
                break
                
        if is_support:
            support_fractals.append(df['low'].iloc[i])
            
        # Check for resistance fractal (high point surrounded by lower highs)
        is_resistance = True
        for j in range(1, window_size + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i-j] or df['high'].iloc[i] <= df['high'].iloc[i+j]:
                is_resistance = False
                break
                
        if is_resistance:
            resistance_fractals.append(df['high'].iloc[i])
    
    # Recent fractal points are more important - add extra weight by duplicating them
    if len(df) > window_size * 4:
        recent_start = -window_size * 10  # Focus on recent fractals
        
        recent_support_fractals = []
        for i in range(max(window_size, len(df) + recent_start), len(df) - window_size):
            # Check for support fractal (low point surrounded by higher lows)
            is_support = True
            for j in range(1, window_size + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i-j] or df['low'].iloc[i] >= df['low'].iloc[i+j]:
                    is_support = False
                    break
                    
            if is_support:
                # Add with double weight for recent fractals
                recent_support_fractals.append(df['low'].iloc[i])
                
        recent_resistance_fractals = []
        for i in range(max(window_size, len(df) + recent_start), len(df) - window_size):
            # Check for resistance fractal (high point surrounded by lower highs)
            is_resistance = True
            for j in range(1, window_size + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i-j] or df['high'].iloc[i] <= df['high'].iloc[i+j]:
                    is_resistance = False
                    break
                    
            if is_resistance:
                # Add with double weight for recent fractals
                recent_resistance_fractals.append(df['high'].iloc[i])
        
        # Combine with extra weight for recent
        support_fractals.extend(recent_support_fractals)
        resistance_fractals.extend(recent_resistance_fractals)
    
    # Remove duplicates
    support_fractals = remove_duplicates(support_fractals, tolerance=0.005)
    resistance_fractals = remove_duplicates(resistance_fractals, tolerance=0.005)
    
    return support_fractals, resistance_fractals

def validate_sr_levels(df, supports, resistances, validation_threshold=0.6):
    """
    Validate support and resistance levels by checking if price actually respects them
    
    Args:
        df (DataFrame): OHLCV dataframe
        supports (list): Potential support levels
        resistances (list): Potential resistance levels
        validation_threshold (float): Minimum percentage of times price must respect a level
    
    Returns:
        tuple: (validated_supports, validated_resistances)
    """
    validated_supports = []
    validated_resistances = []
    
    # For each support level, check how often it actually acts as support
    for level in supports:
        respect_count = 0
        test_count = 0
        
        # Check if price approaches the level from above and bounces
        for i in range(1, len(df)):
            # Price is near the support level (within 0.5%)
            if abs(df['low'].iloc[i] - level) / level < 0.005:
                test_count += 1
                # Price bounces higher after touching
                if i < len(df) - 1 and df['close'].iloc[i+1] > df['open'].iloc[i]:
                    respect_count += 1
        
        # Include level if it has been tested and respected often enough
        if test_count > 0 and respect_count / test_count >= validation_threshold:
            validated_supports.append(level)
        # Or if it hasn't been tested much but is a significant low
        elif test_count < 3 and level < df['low'].quantile(0.1):
            validated_supports.append(level)
    
    # For each resistance level, check how often it actually acts as resistance
    for level in resistances:
        respect_count = 0
        test_count = 0
        
        # Check if price approaches the level from below and reverses
        for i in range(1, len(df)):
            # Price is near the resistance level (within 0.5%)
            if abs(df['high'].iloc[i] - level) / level < 0.005:
                test_count += 1
                # Price drops lower after touching
                if i < len(df) - 1 and df['close'].iloc[i+1] < df['open'].iloc[i]:
                    respect_count += 1
        
        # Include level if it has been tested and respected often enough
        if test_count > 0 and respect_count / test_count >= validation_threshold:
            validated_resistances.append(level)
        # Or if it hasn't been tested much but is a significant high
        elif test_count < 3 and level > df['high'].quantile(0.9):
            validated_resistances.append(level)
    
    # Make sure we don't end up with empty lists
    if not validated_supports:
        validated_supports = supports[:min(3, len(supports))]
    if not validated_resistances:
        validated_resistances = resistances[:min(3, len(resistances))]
    
    return validated_supports, validated_resistances

def remove_duplicates(levels, tolerance=0.01):
    """
    Remove duplicate price levels that are within tolerance of each other
    
    Args:
        levels (list): List of price levels
        tolerance (float): Percentage tolerance for considering levels as duplicates
        
    Returns:
        list: Deduplicated list of price levels
    """
    if not levels:
        return []
        
    # Sort levels
    sorted_levels = sorted(levels)
    
    # Initialize result with the first level
    result = [sorted_levels[0]]
    
    # Check each level against the last added one
    for level in sorted_levels[1:]:
        last = result[-1]
        
        # If levels are far enough apart, add the new one
        if (level - last) / last > tolerance:
            result.append(level)
    
    return result