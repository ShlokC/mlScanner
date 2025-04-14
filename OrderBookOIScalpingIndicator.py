import pandas as pd
import numpy as np
import os
import json
import time
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
PATTERN_STATS_DIR_NAME = 'pattern_stats'
# Define scalping-focused lookback periods (in minutes)
SHORT_TERM_MINUTES = 5
MEDIUM_TERM_MINUTES = 15
LONG_TERM_MINUTES = 60
# Global variable for file queue
file_update_queue = None

class EnhancedOrderBookVolumeIndicator:
    """
    Enhanced scalping indicator using order book dynamics and volume metrics
    with advanced order book delta analysis for more reliable signals.
    """

    def __init__(self,
                file_queue,  # Queue for file operations
                symbol='GLOBAL',  # Symbol for the instrument
                
                # Basic parameters (minimizing static thresholds)
                ob_depth=20,  # Depth of order book to analyze
                
                # Volatility parameters for baseline
                atr_window=5,  # ATR window for volatility estimation
                
                # Trailing stop parameters
                trailing_start_pct=0.3,  # Wait for higher profit before trailing
                trailing_distance_pct=0.15,  # Trailing distance
                
                # Ignore any other parameters
                **kwargs):

        global file_update_queue
        file_update_queue = file_queue
        self.symbol = symbol
        self.bid_cluster_history = deque(maxlen=10)  # History of bid clusters (last 10 snapshots)
        self.ask_cluster_history = deque(maxlen=10)  # History of ask clusters (last 10 snapshots)
        # Store minimal parameters
        self.ob_depth = ob_depth
        self.atr_window = atr_window
        self.trailing_start_pct = trailing_start_pct
        self.trailing_distance_pct = trailing_distance_pct

        # Initialize tracking data structures
        self.order_book_cache = {}
        self.volume_history = deque(maxlen=100)  # Store rolling volume
        self.price_history_for_atr = pd.DataFrame()
        
        # Order book delta tracking (new)
        self.order_book_history = deque(maxlen=20)  # Last 20 order book snapshots
        self.delta_metrics = {}  # For storing recent changes
        
        # Track market trades to detect absorption (new)
        self.recent_market_trades = deque(maxlen=50)  # Recent trades
        self.absorption_events = deque(maxlen=10)  # Recent absorption events
        
        # Price level tracking (new)
        self.price_level_activity = defaultdict(lambda: {
            'bids': {'size': 0, 'additions': 0, 'cancellations': 0, 'executions': 0, 'history': deque(maxlen=10)},
            'asks': {'size': 0, 'additions': 0, 'cancellations': 0, 'executions': 0, 'history': deque(maxlen=10)}
        })
        
        # Iceberg order detection (new)
        self.potential_icebergs = {}  # Track price levels with potential icebergs
        
        # Imbalance tracking with dynamic window (new)
        self.trade_flow = deque(maxlen=100)  # Store trade flow data
        self.trade_flow_metrics = {}  # Metrics over various windows

        # Price level significance (new)
        self.significant_levels = {}
        
        # Order flow dynamics (new)
        self.order_flow_history = deque(maxlen=20)
        self.order_flow_metrics = {
            'aggressive_buy_volume': 0,
            'aggressive_sell_volume': 0,
            'passive_buy_volume': 0,
            'passive_sell_volume': 0,
            'aggression_ratio': 1.0
        }
        
        # Track recent price points
        self.recent_prices = deque(maxlen=100)
        self.volatility_estimate = 0.001  # Initial volatility estimate (0.1%)

        # Simplified pattern tracking
        self.patterns = {
            'order_book_absorption': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': True},
            'iceberg_detection': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': None},
            'large_bid_delta': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': True},
            'large_ask_delta': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': False},
            'order_pulling_detected': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': False},
            'order_defense_detected': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': True},
            'aggressive_buying': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': True},
            'aggressive_selling': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': False},
            'hidden_support': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': True},
            'hidden_resistance': {'total': 0, 'trades': 0, 'wins': 0, 'is_bullish': False}
        }

        # Setup path for pattern stats
        self.pattern_stats_dir = os.path.abspath(PATTERN_STATS_DIR_NAME)
        os.makedirs(self.pattern_stats_dir, exist_ok=True)
        symbol_file = symbol.replace('/', '_').replace(':', '_')
        self.full_pattern_stats_path = os.path.join(self.pattern_stats_dir, f"{symbol_file}_enhanced_pattern_stats.json")
        
        # Load existing pattern stats
        self._load_pattern_stats()
        
        logger.info(f"Initialized EnhancedOrderBookVolumeIndicator for {symbol}")

    def price_decimals(self, price):
        """Determine the number of decimal places for price formatting."""
        if price is None or not isinstance(price, (int, float)) or price == 0:
            return 2
        price = abs(price)
        if price >= 1000: return 1
        if price >= 100: return 2
        if price >= 10: return 3
        if price >= 1: return 4
        if price >= 0.01: return 5
        if price >= 0.001: return 6
        if price >= 0.0001: return 7
        return 8

    def update_order_book(self, bids, asks, timestamp=None):
        """
        Update the order book cache with delta tracking, absorption detection, 
        and iceberg order monitoring.
        """
        if timestamp is None: 
            timestamp = time.time()
        
        if not bids or not asks:
            return

        # Store previous order book snapshot for delta analysis
        prev_book = None
        if self.order_book_cache:
            prev_book = {
                'bids': self.order_book_cache.get('bids', []),
                'asks': self.order_book_cache.get('asks', []),
                'timestamp': self.order_book_cache.get('timestamp', 0),
                'mid_price': self.order_book_cache.get('mid_price', 0)
            }

        # Sort bids (highest to lowest) and asks (lowest to highest)
        bids = sorted(bids, key=lambda x: x[0], reverse=True)[:self.ob_depth]
        asks = sorted(asks, key=lambda x: x[0])[:self.ob_depth]
        
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        current_price = mid_price  # Use mid price as current price reference
        spread = asks[0][0] - bids[0][0] if bids and asks else 0
        spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0
        
        # Track price for volatility estimation
        if mid_price > 0:
            self.recent_prices.append(mid_price)
            if len(self.recent_prices) >= 2:
                # Estimate volatility as average of recent price changes
                changes = [abs(self.recent_prices[i] - self.recent_prices[i-1]) / self.recent_prices[i-1] 
                           for i in range(1, len(self.recent_prices))]
                self.volatility_estimate = np.mean(changes) if changes else 0.001

        # =====================
        # ORDER BOOK DELTA ANALYSIS (New)
        # =====================
        
        # Convert book to dictionaries for easier comparison
        bid_dict = {price: qty for price, qty in bids}
        ask_dict = {price: qty for price, qty in asks}
        
        delta_metrics = {
            'timestamp': timestamp,
            'bid_additions': [],         # New or increased bids
            'bid_cancellations': [],     # Removed or decreased bids
            'ask_additions': [],         # New or increased asks
            'ask_cancellations': [],     # Removed or decreased asks
            'bid_executions': [],        # Likely executed bids
            'ask_executions': [],        # Likely executed asks
            'significant_changes': False, # Flag for significant changes
            'absorption_detected': False, # Flag for order absorption
            'pulling_detected': False,    # Flag for order pulling
            'iceberg_signals': []         # Potential iceberg orders
        }
        
        # Process deltas if we have a previous book
        if prev_book and prev_book['bids'] and prev_book['asks']:
            prev_bid_dict = {price: qty for price, qty in prev_book['bids']}
            prev_ask_dict = {price: qty for price, qty in prev_book['asks']}
            
            # Time between snapshots
            time_delta = timestamp - prev_book['timestamp']
            if time_delta <= 0:
                time_delta = 0.1  # Prevent division by zero
            
            # Check price movement
            price_moved = abs(mid_price - prev_book['mid_price']) / prev_book['mid_price'] > self.volatility_estimate
            
            # Track bid deltas (additions, cancellations, executions)
            for price, qty in bids:
                if price in prev_bid_dict:
                    prev_qty = prev_bid_dict[price]
                    # Volume increased (addition)
                    if qty > prev_qty:
                        delta_metrics['bid_additions'].append((price, qty - prev_qty))
                    # Volume decreased (cancellation or execution)
                    elif qty < prev_qty:
                        # If price moved down, likely executed
                        if price_moved and mid_price < prev_book['mid_price']:
                            delta_metrics['bid_executions'].append((price, prev_qty - qty))
                        # Otherwise might be cancellation
                        else:
                            delta_metrics['bid_cancellations'].append((price, prev_qty - qty))
                else:
                    # New bid
                    delta_metrics['bid_additions'].append((price, qty))
            
            # Check for disappeared bids
            for price, qty in prev_book['bids']:
                if price not in bid_dict:
                    # If price moved down, likely executed
                    if price_moved and mid_price < prev_book['mid_price'] and price >= mid_price:
                        delta_metrics['bid_executions'].append((price, qty))
                    else:
                        delta_metrics['bid_cancellations'].append((price, qty))
            
            # Track ask deltas
            for price, qty in asks:
                if price in prev_ask_dict:
                    prev_qty = prev_ask_dict[price]
                    # Volume increased (addition)
                    if qty > prev_qty:
                        delta_metrics['ask_additions'].append((price, qty - prev_qty))
                    # Volume decreased (cancellation or execution)
                    elif qty < prev_qty:
                        # If price moved up, likely executed
                        if price_moved and mid_price > prev_book['mid_price']:
                            delta_metrics['ask_executions'].append((price, prev_qty - qty))
                        # Otherwise might be cancellation
                        else:
                            delta_metrics['ask_cancellations'].append((price, prev_qty - qty))
                else:
                    # New ask
                    delta_metrics['ask_additions'].append((price, qty))
            
            # Check for disappeared asks
            for price, qty in prev_book['asks']:
                if price not in ask_dict:
                    # If price moved up, likely executed
                    if price_moved and mid_price > prev_book['mid_price'] and price <= mid_price:
                        delta_metrics['ask_executions'].append((price, qty))
                    else:
                        delta_metrics['ask_cancellations'].append((price, qty))
        
            # =====================
            # ABSORPTION DETECTION (New)
            # =====================
            
            # Detecting large executions without significant price movement
            bid_execution_vol = sum(qty for _, qty in delta_metrics['bid_executions'])
            ask_execution_vol = sum(qty for _, qty in delta_metrics['ask_executions'])
            
            # Typical volume in top levels
            typical_bid_vol = np.median([qty for _, qty in bids[:5]]) if bids else 0
            typical_ask_vol = np.median([qty for _, qty in asks[:5]]) if asks else 0
            
            # Calculate relative execution size
            rel_bid_execution = bid_execution_vol / typical_bid_vol if typical_bid_vol > 0 else 0
            rel_ask_execution = ask_execution_vol / typical_ask_vol if typical_ask_vol > 0 else 0
            
            # Detect absorption: large executions with minimal price movement
            if not price_moved:
                # Price didn't move significantly
                if rel_bid_execution > 3:  # Large bid execution
                    delta_metrics['absorption_detected'] = True
                    self.absorption_events.append({
                        'timestamp': timestamp,
                        'price': mid_price,
                        'type': 'bid_absorption',
                        'volume': bid_execution_vol,
                        'rel_size': rel_bid_execution
                    })
                    self._track_pattern('order_book_absorption')
                    logger.debug(f"{self.symbol} Bid absorption detected: {bid_execution_vol} @ {mid_price:.6f}")
                
                elif rel_ask_execution > 3:  # Large ask execution
                    delta_metrics['absorption_detected'] = True
                    self.absorption_events.append({
                        'timestamp': timestamp,
                        'price': mid_price,
                        'type': 'ask_absorption',
                        'volume': ask_execution_vol,
                        'rel_size': rel_ask_execution
                    })
                    self._track_pattern('order_book_absorption')
                    logger.debug(f"{self.symbol} Ask absorption detected: {ask_execution_vol} @ {mid_price:.6f}")
            
            # =====================
            # ICEBERG DETECTION (New)
            # =====================
            
            # Identify constant replenishment at levels
            for side, executions, additions in [
                ('bids', delta_metrics['bid_executions'], delta_metrics['bid_additions']),
                ('asks', delta_metrics['ask_executions'], delta_metrics['ask_additions'])
            ]:
                # Group by price
                exec_by_price = defaultdict(float)
                add_by_price = defaultdict(float)
                
                for price, qty in executions:
                    exec_by_price[price] += qty
                for price, qty in additions:
                    add_by_price[price] += qty
                
                # Check for matching executions and additions
                for price, exec_qty in exec_by_price.items():
                    if price in add_by_price:
                        add_qty = add_by_price[price]
                        # If significant volume was executed and then replenished
                        if exec_qty > 0 and add_qty > 0:
                            # Calculate replenishment ratio
                            replenishment_ratio = add_qty / exec_qty
                            
                            if 0.7 <= replenishment_ratio <= 1.3:  # Roughly matching
                                # Track potential iceberg
                                if price not in self.potential_icebergs:
                                    self.potential_icebergs[price] = {
                                        'side': side,
                                        'count': 1,
                                        'last_seen': timestamp,
                                        'total_volume': exec_qty,
                                        'confidence': 0.2  # Initial confidence
                                    }
                                else:
                                    iceberg = self.potential_icebergs[price]
                                    if iceberg['side'] == side:
                                        # Update existing tracking
                                        iceberg['count'] += 1
                                        iceberg['last_seen'] = timestamp
                                        iceberg['total_volume'] += exec_qty
                                        # Increase confidence (capped)
                                        iceberg['confidence'] = min(0.95, iceberg['confidence'] + 0.15)
                                        
                                        # Add to signal if confidence is high enough
                                        if iceberg['confidence'] >= 0.5:
                                            delta_metrics['iceberg_signals'].append({
                                                'price': price,
                                                'side': side,
                                                'confidence': iceberg['confidence'],
                                                'volume': exec_qty
                                            })
                                            self._track_pattern('iceberg_detection')
                                            logger.debug(f"{self.symbol} Iceberg order detected on {side} side at {price:.6f}, confidence: {iceberg['confidence']:.2f}")
            
            # =====================
            # ORDER PULLING DETECTION (New)
            # =====================
            
            # Detect significant cancellations near the mid price
            # This can signal weakening support/resistance or spoofing
            bid_cancel_near_mid = sum(qty for price, qty in delta_metrics['bid_cancellations'] 
                                    if (mid_price - price) / mid_price < self.volatility_estimate * 5)
            
            ask_cancel_near_mid = sum(qty for price, qty in delta_metrics['ask_cancellations'] 
                                     if (price - mid_price) / mid_price < self.volatility_estimate * 5)
            
            # Calculate relative cancellation size
            rel_bid_cancel = bid_cancel_near_mid / typical_bid_vol if typical_bid_vol > 0 else 0
            rel_ask_cancel = ask_cancel_near_mid / typical_ask_vol if typical_ask_vol > 0 else 0
            
            # Significant cancellations indicate order pulling
            if rel_bid_cancel > 2:
                delta_metrics['pulling_detected'] = True
                self._track_pattern('order_pulling_detected')
                logger.debug(f"{self.symbol} Significant bid cancellations near mid: {bid_cancel_near_mid} @ {mid_price:.6f}")
            
            if rel_ask_cancel > 2:
                delta_metrics['pulling_detected'] = True
                self._track_pattern('order_pulling_detected')
                logger.debug(f"{self.symbol} Significant ask cancellations near mid: {ask_cancel_near_mid} @ {mid_price:.6f}")
        
        # Add to history
        self.order_book_history.append({
            'timestamp': timestamp,
            'bids': bids,
            'asks': asks,
            'mid_price': mid_price,
            'delta_metrics': delta_metrics
        })
        
        # Store delta metrics
        self.delta_metrics = delta_metrics
        
        # =====================
        # ORDER BOOK IMBALANCE (Updated)
        # =====================
        
        # Calculate volume at different price depths
        bid_volume_total = sum(qty for _, qty in bids)
        ask_volume_total = sum(qty for _, qty in asks)
        
        # Calculate volume in the nearest 25% of the order book
        bid_volume_nearest = sum(qty for price, qty in bids 
                               if (mid_price - price) / mid_price < self.volatility_estimate * 10)
        
        ask_volume_nearest = sum(qty for price, qty in asks 
                               if (price - mid_price) / mid_price < self.volatility_estimate * 10)
        
        # Dynamic imbalance calculation based on volatility
        imbalance_total = bid_volume_total / ask_volume_total if ask_volume_total > 0 else 1.0
        imbalance_nearest = bid_volume_nearest / ask_volume_nearest if ask_volume_nearest > 0 else 1.0
        
        # Update order flow metrics
        self.order_flow_metrics = {
            'aggressive_buy_volume': sum(qty for _, qty in delta_metrics['ask_executions']),
            'aggressive_sell_volume': sum(qty for _, qty in delta_metrics['bid_executions']),
            'passive_buy_volume': sum(qty for _, qty in delta_metrics['bid_additions']),
            'passive_sell_volume': sum(qty for _, qty in delta_metrics['ask_additions']),
        }
        
        # Calculate aggression ratio (buy vs sell aggressiveness)
        agg_buy = self.order_flow_metrics['aggressive_buy_volume']
        agg_sell = self.order_flow_metrics['aggressive_sell_volume']
        
        if agg_buy > 0 or agg_sell > 0:
            self.order_flow_metrics['aggression_ratio'] = agg_buy / (agg_buy + agg_sell)
        else:
            self.order_flow_metrics['aggression_ratio'] = 0.5  # Neutral
        
        # Store in history
        self.order_flow_history.append({
            'timestamp': timestamp,
            'aggressive_buy_volume': self.order_flow_metrics['aggressive_buy_volume'],
            'aggressive_sell_volume': self.order_flow_metrics['aggressive_sell_volume'],
            'aggression_ratio': self.order_flow_metrics['aggression_ratio'],
            'mid_price': mid_price
        })
        
        # Store enhanced metrics in cache
        self.order_book_cache = {
            'bids': bids,
            'asks': asks,
            'timestamp': timestamp,
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,
            
            # Volume metrics
            'bid_volume_total': bid_volume_total,
            'ask_volume_total': ask_volume_total,
            'bid_volume_nearest': bid_volume_nearest,
            'ask_volume_nearest': ask_volume_nearest,
            
            # Dynamic imbalance metrics
            'imbalance_total': imbalance_total,
            'imbalance_nearest': imbalance_nearest,
            
            # Order flow metrics
            'aggressive_buy_volume': self.order_flow_metrics['aggressive_buy_volume'],
            'aggressive_sell_volume': self.order_flow_metrics['aggressive_sell_volume'],
            'aggression_ratio': self.order_flow_metrics['aggression_ratio'],
            
            # Delta metrics
            'pulling_detected': delta_metrics['pulling_detected'],
            'absorption_detected': delta_metrics['absorption_detected'],
            'iceberg_signals': delta_metrics['iceberg_signals'],
            
            # Support/resistance levels with confidence
            'support_levels': self._calculate_support_levels(bids, mid_price),
            'resistance_levels': self._calculate_resistance_levels(asks, mid_price),
            
            # Market state analysis
            'volatility_estimate': self.volatility_estimate,
            'market_buying_pressure': self._calculate_buying_pressure(),
            'market_selling_pressure': self._calculate_selling_pressure()
        }

    def _calculate_support_levels(self, bids, mid_price):
        """
        Dynamically calculate support levels with confidence scores,
        incorporating recent price action.
        Returns list of (price, confidence, strength) tuples.
        """
        if not bids:
            return []
            
        support_levels = []
        
        # IMPROVEMENT: Initialize historical tracking if needed
        if not hasattr(self, 'historical_levels'):
            self.historical_levels = {}
            
        if not hasattr(self, 'price_rejection_points'):
            self.price_rejection_points = {
                'supports': [],  # (price, timestamp, strength)
                'resistances': []  # (price, timestamp, strength)
            }
        
        # Find clusters of large bids with enhanced wall detection
        clusters = self._find_clusters(bids, mid_price, 'bids')
        
        # Current time for decay calculations
        current_time = time.time()
        
        # IMPROVEMENT: Track recent price levels where price rebounded
        # This would ideally come from analyzing recent price swings
        # For now, use a simplified approach based on recent price history
        recent_support_rejections = self._find_recent_support_rejections()
        
        # Add absorption evidence
        absorption_prices = [event['price'] for event in self.absorption_events 
                            if event['type'] == 'bid_absorption' 
                            and current_time - event['timestamp'] < 300]  # Last 5 minutes
        
        # Score support levels
        for cluster_price, cluster_vol, wall_strength in clusters:
            # Base confidence on wall strength
            volume_confidence = min(0.7, wall_strength * 0.15)
            
            # IMPROVEMENT: Add confidence if price recently rebounded from this level
            rejection_bonus = 0.0
            for rejection_price, rejection_time, rejection_strength in recent_support_rejections:
                # If within volatility range
                if abs(cluster_price - rejection_price) / mid_price < self.volatility_estimate * 5:
                    # Calculate time decay factor (full strength for 30 minutes, then decays)
                    time_decay = max(0, 1 - (current_time - rejection_time) / (30 * 60))
                    rejection_bonus += rejection_strength * time_decay * 0.4  # Higher weight for actual price rejection
            
            # Add confidence if we've seen absorption here
            absorption_bonus = 0.0
            for abs_price in absorption_prices:
                # If within volatility range
                if abs(cluster_price - abs_price) / mid_price < self.volatility_estimate * 3:
                    absorption_bonus += 0.2
            
            # Check for order replenishment (iceberg evidence)
            iceberg_bonus = 0.0
            for price, iceberg in self.potential_icebergs.items():
                if (iceberg['side'] == 'bids' and 
                    abs(price - cluster_price) / mid_price < self.volatility_estimate * 2):
                    iceberg_bonus += iceberg['confidence'] * 0.3
            
            # Add historical persistence bonus
            persistence_bonus = 0.0
            level_key = f"support_{int(cluster_price * 100)}"  # Round to 2 decimals for key
            
            if level_key in self.historical_levels:
                # Get historical record
                hist_record = self.historical_levels[level_key]
                # Calculate time decay factor (full strength for 5 minutes, then decays)
                time_since_update = current_time - hist_record['last_update']
                decay_factor = max(0, 1 - (time_since_update - 300) / 3600)  # Decay over an hour
                
                if decay_factor > 0:
                    # Add persistence bonus based on historical confidence and decay
                    persistence_bonus = hist_record['confidence'] * decay_factor * 0.3
                    
                # Update historical record
                hist_record['last_update'] = current_time
                hist_record['confidence'] = max(hist_record['confidence'], 
                                            volume_confidence + absorption_bonus + iceberg_bonus)
                hist_record['observation_count'] = hist_record.get('observation_count', 0) + 1
            else:
                # Create new historical record
                self.historical_levels[level_key] = {
                    'price': cluster_price,
                    'confidence': volume_confidence + absorption_bonus + iceberg_bonus,
                    'last_update': current_time,
                    'observation_count': 1
                }
            
            # IMPROVEMENT: Give extra weight to price action confirmation
            # Price action is a stronger signal than order book
            rejection_factor = 1.0 + (rejection_bonus * 1.5)
            
            # Calculate final confidence (capped at 0.95)
            confidence = min(0.95, (volume_confidence + absorption_bonus + iceberg_bonus + persistence_bonus) * rejection_factor)
            
            support_levels.append((cluster_price, confidence, wall_strength))
            # IMPROVEMENT: Add pure price action support levels even if not in order book
            # This ensures we don't miss important technical levels just because
            # there are currently no orders there
            for rejection_price, rejection_time, rejection_strength in recent_support_rejections:
                # Skip if too close to an existing level
                if any(abs(rejection_price - level[0]) / mid_price < self.volatility_estimate * 3 for level in support_levels):
                    continue
                    
                # Calculate time decay
                time_decay = max(0, 1 - (current_time - rejection_time) / (30 * 60))
                
                # Pure technical level - confidence based solely on price action
                if time_decay > 0.3:  # Only include relatively recent rejections
                    # Calculate confidence based on strength and recency
                    tech_confidence = min(0.8, rejection_strength * time_decay)
                    
                    # Use a default wall strength
                    tech_wall_strength = rejection_strength * 2
                    
                    support_levels.append((rejection_price, tech_confidence, tech_wall_strength))
            
            # Sort by confidence (highest first)
            return sorted(support_levels, key=lambda x: x[1], reverse=True)
    def _find_recent_support_rejections(self):
        """
        Analyze recent price data to find support rejection points.
        Returns list of (price, timestamp, strength) tuples.
        """
        if not hasattr(self, 'price_history_for_atr') or self.price_history_for_atr.empty:
            return []
        
        rejections = []
        
        # Get recent price data
        df = self.price_history_for_atr.copy()
        if len(df) < 5:
            return []
        
        try:
            # Simplistic approach: Look for candles with long lower wicks
            # (indicates buyers stepping in at that level)
            for i in range(1, min(30, len(df) - 1)):
                candle = df.iloc[-i]
                prev_candle = df.iloc[-i-1] if i < len(df) - 1 else None
                next_candle = df.iloc[-i+1] if i > 1 else None
                
                # Skip if missing data
                if candle.empty or pd.isna(candle['low']) or pd.isna(candle['close']):
                    continue
                    
                # Calculate lower wick as percentage of candle
                candle_range = candle['high'] - candle['low']
                if candle_range <= 0:
                    continue
                    
                lower_wick = (candle['close'] - candle['low']) if candle['close'] > candle['low'] else (candle['open'] - candle['low'])
                lower_wick_ratio = lower_wick / candle_range
                
                # Strong rejection if:
                # 1. Long lower wick (>40% of candle)
                # 2. Price closed higher than it opened
                # 3. Previous candle was bearish
                is_strong_rejection = (
                    lower_wick_ratio > 0.4 and
                    lower_wick > 0 and
                    (prev_candle is None or prev_candle['close'] < prev_candle['open']) and
                    (next_candle is None or next_candle['close'] > next_candle['open'])
                )
                
                if is_strong_rejection:
                    # Get timestamp (use index if it's a timestamp, otherwise use position)
                    timestamp = candle.name.timestamp() if hasattr(candle.name, 'timestamp') else time.time() - (i * 60)
                    
                    # Calculate rejection strength (0-1 scale)
                    rejection_strength = min(0.9, lower_wick_ratio)
                    
                    # Add to rejections
                    rejections.append((candle['low'], timestamp, rejection_strength))
        
        except Exception as e:
            logger.error(f"Error in _find_recent_support_rejections: {e}")
            return []
        
        return rejections
    def _calculate_resistance_levels(self, asks, mid_price):
        """
        Dynamically calculate resistance levels with confidence scores,
        incorporating recent price action.
        Returns list of (price, confidence, strength) tuples.
        """
        if not asks:
            return []
            
        resistance_levels = []
        
        # IMPROVEMENT: Initialize historical tracking if needed
        if not hasattr(self, 'historical_levels'):
            self.historical_levels = {}
            
        if not hasattr(self, 'price_rejection_points'):
            self.price_rejection_points = {
                'supports': [],  # (price, timestamp, strength)
                'resistances': []  # (price, timestamp, strength)
            }
        
        # Find clusters of large asks with enhanced wall detection
        clusters = self._find_clusters(asks, mid_price, 'asks')
        
        # Current time for decay calculations
        current_time = time.time()
        
        # IMPROVEMENT: Track recent price levels where price retraced
        recent_resistance_rejections = self._find_recent_resistance_rejections()
        
        # Add absorption evidence
        absorption_prices = [event['price'] for event in self.absorption_events 
                            if event['type'] == 'ask_absorption' 
                            and current_time - event['timestamp'] < 300]  # Last 5 minutes
        
        # Score resistance levels
        for cluster_price, cluster_vol, wall_strength in clusters:
            # Base confidence on wall strength
            volume_confidence = min(0.7, wall_strength * 0.15)
            
            # IMPROVEMENT: Add confidence if price recently retraced from this level
            rejection_bonus = 0.0
            for rejection_price, rejection_time, rejection_strength in recent_resistance_rejections:
                # If within volatility range
                if abs(cluster_price - rejection_price) / mid_price < self.volatility_estimate * 5:
                    # Calculate time decay factor (full strength for 30 minutes, then decays)
                    time_decay = max(0, 1 - (current_time - rejection_time) / (30 * 60))
                    rejection_bonus += rejection_strength * time_decay * 0.4  # Higher weight for actual price rejection
            
            # Add confidence if we've seen absorption here
            absorption_bonus = 0.0
            for abs_price in absorption_prices:
                # If within volatility range
                if abs(cluster_price - abs_price) / mid_price < self.volatility_estimate * 3:
                    absorption_bonus += 0.2
            
            # Check for order replenishment (iceberg evidence)
            iceberg_bonus = 0.0
            for price, iceberg in self.potential_icebergs.items():
                if (iceberg['side'] == 'asks' and 
                    abs(price - cluster_price) / mid_price < self.volatility_estimate * 2):
                    iceberg_bonus += iceberg['confidence'] * 0.3
            
            # Add historical persistence bonus
            persistence_bonus = 0.0
            level_key = f"resistance_{int(cluster_price * 100)}"  # Round to 2 decimals for key
            
            if level_key in self.historical_levels:
                # Get historical record
                hist_record = self.historical_levels[level_key]
                # Calculate time decay factor (full strength for 5 minutes, then decays)
                time_since_update = current_time - hist_record['last_update']
                decay_factor = max(0, 1 - (time_since_update - 300) / 3600)  # Decay over an hour
                
                if decay_factor > 0:
                    # Add persistence bonus based on historical confidence and decay
                    persistence_bonus = hist_record['confidence'] * decay_factor * 0.3
                    
                # Update historical record
                hist_record['last_update'] = current_time
                hist_record['confidence'] = max(hist_record['confidence'], 
                                            volume_confidence + absorption_bonus + iceberg_bonus)
                hist_record['observation_count'] = hist_record.get('observation_count', 0) + 1
            else:
                # Create new historical record
                self.historical_levels[level_key] = {
                    'price': cluster_price,
                    'confidence': volume_confidence + absorption_bonus + iceberg_bonus,
                    'last_update': current_time,
                    'observation_count': 1
                }
            
            # IMPROVEMENT: Give extra weight to price action confirmation
            # Price action is a stronger signal than order book
            rejection_factor = 1.0 + (rejection_bonus * 1.5)
            
            # Calculate final confidence (capped at 0.95)
            confidence = min(0.95, (volume_confidence + absorption_bonus + iceberg_bonus + persistence_bonus) * rejection_factor)
            
            resistance_levels.append((cluster_price, confidence, wall_strength))
        
        # IMPROVEMENT: Add pure price action resistance levels even if not in order book
        for rejection_price, rejection_time, rejection_strength in recent_resistance_rejections:
            # Skip if too close to an existing level
            if any(abs(rejection_price - level[0]) / mid_price < self.volatility_estimate * 3 for level in resistance_levels):
                continue
                
            # Calculate time decay
            time_decay = max(0, 1 - (current_time - rejection_time) / (30 * 60))
            
            # Pure technical level - confidence based solely on price action
            if time_decay > 0.3:  # Only include relatively recent rejections
                # Calculate confidence based on strength and recency
                tech_confidence = min(0.8, rejection_strength * time_decay)
                
                # Use a default wall strength
                tech_wall_strength = rejection_strength * 2
                
                resistance_levels.append((rejection_price, tech_confidence, tech_wall_strength))
        
        # Sort by confidence (highest first)
        return sorted(resistance_levels, key=lambda x: x[1], reverse=True)
        # Helper method for determining appropriate tick size
    def _find_recent_resistance_rejections(self):
        """
        Analyze recent price data to find resistance rejection points.
        Returns list of (price, timestamp, strength) tuples.
        """
        if not hasattr(self, 'price_history_for_atr') or self.price_history_for_atr.empty:
            return []
        
        rejections = []
        
        # Get recent price data
        df = self.price_history_for_atr.copy()
        if len(df) < 5:
            return []
        
        try:
            # Simplistic approach: Look for candles with long upper wicks
            # (indicates sellers stepping in at that level)
            for i in range(1, min(30, len(df) - 1)):
                candle = df.iloc[-i]
                prev_candle = df.iloc[-i-1] if i < len(df) - 1 else None
                next_candle = df.iloc[-i+1] if i > 1 else None
                
                # Skip if missing data
                if candle.empty or pd.isna(candle['high']) or pd.isna(candle['close']):
                    continue
                    
                # Calculate upper wick as percentage of candle
                candle_range = candle['high'] - candle['low']
                if candle_range <= 0:
                    continue
                    
                upper_wick = (candle['high'] - candle['close']) if candle['close'] < candle['high'] else (candle['high'] - candle['open'])
                upper_wick_ratio = upper_wick / candle_range
                
                # Strong rejection if:
                # 1. Long upper wick (>40% of candle)
                # 2. Price closed lower than it opened
                # 3. Previous candle was bullish
                is_strong_rejection = (
                    upper_wick_ratio > 0.4 and
                    upper_wick > 0 and
                    (prev_candle is None or prev_candle['close'] > prev_candle['open']) and
                    (next_candle is None or next_candle['close'] < next_candle['open'])
                )
                
                if is_strong_rejection:
                    # Get timestamp (use index if it's a timestamp, otherwise use position)
                    timestamp = candle.name.timestamp() if hasattr(candle.name, 'timestamp') else time.time() - (i * 60)
                    
                    # Calculate rejection strength (0-1 scale)
                    rejection_strength = min(0.9, upper_wick_ratio)
                    
                    # Add to rejections
                    rejections.append((candle['high'], timestamp, rejection_strength))
        
        except Exception as e:
            logger.error(f"Error in _find_recent_resistance_rejections: {e}")
            return []
        
        return rejections
    def _is_psychological_level(self, price):
        """
        Improved check if a price is a psychological level with finer granularity.
        """
        # Convert to string for easier pattern detection
        price_str = str(float(price))
        
        # Round number check (whole numbers)
        if price_str.endswith('.0') or price_str.endswith('.00'):
            return True
        
        # Check for clear decimal patterns
        # Major round levels (.00, .000)
        if price_str.endswith('00') or price_str.endswith('000'):
            return True
        
        # Common Fibonacci-based levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        decimal_part = price - int(price)
        
        # Check if decimal part is close to a Fibonacci level
        for level in fib_levels:
            if abs(decimal_part - level) < 0.005:
                return True
        
        # Check for common fraction patterns (.25, .5, .75)
        for pattern in ['.25', '.5', '.50', '.75']:
            if price_str.endswith(pattern):
                return True
        
        # For higher prices, check if it's a multiple of 5 or 10
        if price >= 10:
            if price % 5 == 0:
                return True
            if price % 10 == 0:
                return True
        
        # For very high prices, check for hundreds/thousands values
        if price >= 100:
            if price % 100 == 0:
                return True
        if price >= 1000:
            if price % 500 == 0:
                return True
            if price % 1000 == 0:
                return True
        
        return False
    def _estimate_tick_size(self, price):
        """Estimate appropriate tick size based on price magnitude."""
        if price >= 10000:
            return 1.0  # $1 for very high-priced assets
        elif price >= 1000:
            return 0.1  # $0.10 for high-priced assets
        elif price >= 100:
            return 0.01  # $0.01 for medium-priced assets
        elif price >= 10:
            return 0.001  # $0.001 for lower-priced assets
        elif price >= 1:
            return 0.0001  # $0.0001 for low-priced assets
        else:
            return 0.00001  # Very fine-grained for very low-priced assets
    def _calculate_sl_tp_from_walls(self, signal_type, current_price, support_levels, resistance_levels, volatility_factor):
        """
        Calculate optimal stop loss and take profit levels based on order book walls.
        Prioritizes using actual walls with high order density.
        
        Args:
            signal_type (str): Strategy type ('fade_long', 'fade_short', 'breakout_long', 'breakout_short')
            current_price (float): Current market price
            support_levels (list): List of support levels as (price, confidence, strength, volume) tuples
            resistance_levels (list): List of resistance levels as (price, confidence, strength, volume) tuples
            volatility_factor (float): Current market volatility as a decimal of price
            
        Returns:
            dict: SL/TP information with price levels and their basis
        """
        result = {
            'stop_loss': 0,
            'target': 0,
            'sl_basis': 'none',
            'tp_basis': 'none',
            'risk_reward': 0
        }
        
        # Early validation
        if not current_price or current_price <= 0 or not (support_levels or resistance_levels):
            return result
        
        # Separate levels by their relationship to current price
        supports_below = [(price, conf, strength, vol) for price, conf, strength, vol in support_levels if price < current_price]
        supports_above = [(price, conf, strength, vol) for price, conf, strength, vol in support_levels if price > current_price]
        resistance_below = [(price, conf, strength, vol) for price, conf, strength, vol in resistance_levels if price < current_price]
        resistance_above = [(price, conf, strength, vol) for price, conf, strength, vol in resistance_levels if price > current_price]
        
        # Sort each category by distance to price (closest first)
        supports_below.sort(key=lambda x: current_price - x[0])
        supports_above.sort(key=lambda x: x[0] - current_price)
        resistance_below.sort(key=lambda x: current_price - x[0])
        resistance_above.sort(key=lambda x: x[0] - current_price)
        
        # Dynamic buffer calculation based on volatility and wall strength
        def calculate_buffer(wall_strength, is_breakout=False):
            # Normalize wall strength to 0-1 range for consistency
            norm_strength = min(1.0, wall_strength / 10) if wall_strength > 0 else 0.2
            
            if is_breakout:
                # Tighter buffers for breakout trades (just beyond the broken level)
                base_buffer = volatility_factor * 1.5
                strength_adj = 0.7  # Strength matters less for breakout SL placement
            else:
                # Wider buffers for fade trades (beyond the wall)
                base_buffer = volatility_factor * 2.5
                strength_adj = 1.3  # Strength matters more for fade SL placement
            
            # Stronger walls = tighter buffers (more confident in the level)
            buffer = base_buffer * (1.0 - (norm_strength * strength_adj * 0.5))
            
            # Ensure minimum and maximum reasonable buffers based on volatility
            min_buffer = volatility_factor * 0.7
            max_buffer = volatility_factor * 3.0
            buffer = max(min(buffer, max_buffer), min_buffer)
            
            return buffer
        
        # CASE 1: FADE LONG (Buy at support)
        if signal_type == 'fade_long' and supports_below:
            # Use the closest strong support below price for SL
            support_wall = supports_below[0]
            support_price, support_conf, support_strength, support_vol = support_wall
            
            # Calculate dynamic buffer below support
            buffer_pct = calculate_buffer(support_strength, is_breakout=False)
            
            # Place SL below the support wall
            sl = support_price * (1 - buffer_pct)
            result['stop_loss'] = sl
            result['sl_basis'] = 'below_support_wall'
            
            # Find target at nearest significant resistance above
            if resistance_above:
                # Filter for significant walls
                strong_resistance = [r for r in resistance_above if r[2] > 3.0 or r[1] > 0.6]
                
                # If no strong walls, use any available resistance
                target_walls = strong_resistance if strong_resistance else resistance_above
                
                # Get the first significant wall at reasonable distance
                min_target_distance = volatility_factor * 3
                target_wall = None
                
                for wall in target_walls:
                    res_price = wall[0]
                    distance_pct = (res_price - current_price) / current_price
                    if distance_pct >= min_target_distance:
                        target_wall = wall
                        break
                
                # If found a suitable resistance, use it for TP
                if target_wall:
                    res_price, res_conf, res_strength, res_vol = target_wall
                    
                    # Calculate buffer based on resistance strength
                    tp_buffer = calculate_buffer(res_strength, is_breakout=False) * 0.5
                    
                    # Place TP slightly below resistance
                    tp = res_price * (1 - tp_buffer)
                    result['target'] = tp
                    result['tp_basis'] = 'resistance_wall'
                else:
                    # No suitable wall - use dynamic R:R
                    sl_distance = current_price - sl
                    
                    # Calculate R:R based on volatility - higher R:R in volatile markets
                    target_rr = 1.5 + (volatility_factor * 40)
                    target_rr = min(max(target_rr, 1.5), 4.0)
                    
                    tp = current_price + (sl_distance * target_rr)
                    result['target'] = tp
                    result['tp_basis'] = 'dynamic_rr'
        
        # CASE 2: BREAKOUT LONG (Buy after breaking resistance)
        elif signal_type == 'breakout_long' and resistance_below:
            # Use the broken resistance for SL
            broken_res = resistance_below[0]
            res_price, res_conf, res_strength, res_vol = broken_res
            
            # Calculate tight buffer (just below broken level)
            buffer_pct = calculate_buffer(res_strength, is_breakout=True)
            
            # Place SL just below broken resistance
            sl = res_price * (1 - buffer_pct)
            result['stop_loss'] = sl
            result['sl_basis'] = 'below_broken_resistance'
            
            # Find next resistance level for TP
            if resistance_above:
                # Filter for significant walls at greater distance
                min_target_distance = volatility_factor * 5  # Further for breakouts
                suitable_targets = []
                
                for wall in resistance_above:
                    res_price = wall[0]
                    distance_pct = (res_price - current_price) / current_price
                    if distance_pct >= min_target_distance:
                        # Score each potential target by strength and distance
                        distance_score = 1.0 / (1.0 + distance_pct)  # Closer is better, but with diminishing returns
                        strength_score = wall[2] / 10 if wall[2] > 0 else 0.1
                        wall_score = strength_score * 0.7 + distance_score * 0.3  # Prioritize strength
                        suitable_targets.append((wall, wall_score))
                
                # Sort by score and take the best
                if suitable_targets:
                    suitable_targets.sort(key=lambda x: x[1], reverse=True)
                    best_target = suitable_targets[0][0]
                    
                    res_price, res_conf, res_strength, res_vol = best_target
                    
                    # Calculate buffer based on resistance strength
                    tp_buffer = calculate_buffer(res_strength, is_breakout=True) * 0.3
                    
                    # Place TP slightly below resistance
                    tp = res_price * (1 - tp_buffer)
                    result['target'] = tp
                    result['tp_basis'] = 'next_resistance_wall'
                else:
                    # No suitable wall - use higher R:R for breakout
                    sl_distance = current_price - sl
                    
                    # Higher R:R for breakout trades
                    target_rr = 2.0 + (volatility_factor * 50)
                    target_rr = min(max(target_rr, 2.0), 5.0)
                    
                    tp = current_price + (sl_distance * target_rr)
                    result['target'] = tp
                    result['tp_basis'] = 'dynamic_breakout_rr'
        
        # CASE 3: FADE SHORT (Sell at resistance)
        elif signal_type == 'fade_short' and resistance_above:
            # Use the closest strong resistance above price for SL
            resistance_wall = resistance_above[0]
            res_price, res_conf, res_strength, res_vol = resistance_wall
            
            # Calculate dynamic buffer above resistance
            buffer_pct = calculate_buffer(res_strength, is_breakout=False)
            
            # Place SL above the resistance wall
            sl = res_price * (1 + buffer_pct)
            result['stop_loss'] = sl
            result['sl_basis'] = 'above_resistance_wall'
            
            # Find target at nearest significant support below
            if supports_below:
                # Filter for significant walls
                strong_support = [s for s in supports_below if s[2] > 3.0 or s[1] > 0.6]
                
                # If no strong walls, use any available support
                target_walls = strong_support if strong_support else supports_below
                
                # Get the first significant wall at reasonable distance
                min_target_distance = volatility_factor * 3
                target_wall = None
                
                for wall in target_walls:
                    sup_price = wall[0]
                    distance_pct = (current_price - sup_price) / current_price
                    if distance_pct >= min_target_distance:
                        target_wall = wall
                        break
                
                # If found a suitable support, use it for TP
                if target_wall:
                    sup_price, sup_conf, sup_strength, sup_vol = target_wall
                    
                    # Calculate buffer based on support strength
                    tp_buffer = calculate_buffer(sup_strength, is_breakout=False) * 0.5
                    
                    # Place TP slightly above support
                    tp = sup_price * (1 + tp_buffer)
                    result['target'] = tp
                    result['tp_basis'] = 'support_wall'
                else:
                    # No suitable wall - use dynamic R:R
                    sl_distance = sl - current_price
                    
                    # Calculate R:R based on volatility
                    target_rr = 1.5 + (volatility_factor * 40)
                    target_rr = min(max(target_rr, 1.5), 4.0)
                    
                    tp = current_price - (sl_distance * target_rr)
                    result['target'] = tp
                    result['tp_basis'] = 'dynamic_rr'
        
        # CASE 4: BREAKOUT SHORT (Sell after breaking support)
        elif signal_type == 'breakout_short' and supports_above:
            # Use the broken support for SL
            broken_sup = supports_above[0]
            sup_price, sup_conf, sup_strength, sup_vol = broken_sup
            
            # Calculate tight buffer (just above broken level)
            buffer_pct = calculate_buffer(sup_strength, is_breakout=True)
            
            # Place SL just above broken support
            sl = sup_price * (1 + buffer_pct)
            result['stop_loss'] = sl
            result['sl_basis'] = 'above_broken_support'
            
            # Find next support level for TP
            if supports_below:
                # Filter for significant walls at greater distance
                min_target_distance = volatility_factor * 5  # Further for breakouts
                suitable_targets = []
                
                for wall in supports_below:
                    sup_price = wall[0]
                    distance_pct = (current_price - sup_price) / current_price
                    if distance_pct >= min_target_distance:
                        # Score each potential target by strength and distance
                        distance_score = 1.0 / (1.0 + distance_pct)  # Closer is better, but with diminishing returns
                        strength_score = wall[2] / 10 if wall[2] > 0 else 0.1
                        wall_score = strength_score * 0.7 + distance_score * 0.3  # Prioritize strength
                        suitable_targets.append((wall, wall_score))
                
                # Sort by score and take the best
                if suitable_targets:
                    suitable_targets.sort(key=lambda x: x[1], reverse=True)
                    best_target = suitable_targets[0][0]
                    
                    sup_price, sup_conf, sup_strength, sup_vol = best_target
                    
                    # Calculate buffer based on support strength
                    tp_buffer = calculate_buffer(sup_strength, is_breakout=True) * 0.3
                    
                    # Place TP slightly above support
                    tp = sup_price * (1 + tp_buffer)
                    result['target'] = tp
                    result['tp_basis'] = 'next_support_wall'
                else:
                    # No suitable wall - use higher R:R for breakout
                    sl_distance = sl - current_price
                    
                    # Higher R:R for breakout trades
                    target_rr = 2.0 + (volatility_factor * 50)
                    target_rr = min(max(target_rr, 2.0), 5.0)
                    
                    tp = current_price - (sl_distance * target_rr)
                    result['target'] = tp
                    result['tp_basis'] = 'dynamic_breakout_rr'
        
        # If no wall-based calculation succeeded, use volatility-based fallback
        if result['stop_loss'] == 0:
            # Fallback SL based purely on volatility
            sl_distance = current_price * volatility_factor * 1.5
            
            if signal_type in ['fade_long', 'breakout_long']:
                result['stop_loss'] = current_price - sl_distance
                result['sl_basis'] = 'volatility_fallback'
            else:
                result['stop_loss'] = current_price + sl_distance
                result['sl_basis'] = 'volatility_fallback'
        
        if result['target'] == 0:
            # Fallback TP based on conservative R:R
            sl_distance = abs(current_price - result['stop_loss'])
            target_rr = 1.5  # Conservative 1.5:1 R:R
            
            if signal_type in ['fade_long', 'breakout_long']:
                result['target'] = current_price + (sl_distance * target_rr)
                result['tp_basis'] = 'rr_fallback'
            else:
                result['target'] = current_price - (sl_distance * target_rr)
                result['tp_basis'] = 'rr_fallback'
        
        # Ensure proper positioning of SL/TP
        if signal_type in ['fade_long', 'breakout_long']:
            # For longs, SL must be below current price
            if result['stop_loss'] >= current_price:
                result['stop_loss'] = current_price * (1 - volatility_factor)
                result['sl_basis'] = 'adjusted_fallback'
                
            # TP must be above current price
            if result['target'] <= current_price:
                result['target'] = current_price * (1 + volatility_factor * 2)
                result['tp_basis'] = 'adjusted_fallback'
        else:
            # For shorts, SL must be above current price
            if result['stop_loss'] <= current_price:
                result['stop_loss'] = current_price * (1 + volatility_factor)
                result['sl_basis'] = 'adjusted_fallback'
                
            # TP must be below current price
            if result['target'] >= current_price:
                result['target'] = current_price * (1 - volatility_factor * 2)
                result['tp_basis'] = 'adjusted_fallback'
        
        # Calculate actual R:R
        sl_distance = abs(result['stop_loss'] - current_price)
        tp_distance = abs(result['target'] - current_price)
        if sl_distance > 0:
            result['risk_reward'] = tp_distance / sl_distance
        else:
            result['risk_reward'] = 1.0  # Default fallback
        
        # Ensure minimum R:R
        min_rr = 1.3 + (volatility_factor * 20)
        min_rr = min(max(min_rr, 1.3), 3.0)
        
        if result['risk_reward'] < min_rr:
            if signal_type in ['fade_long', 'breakout_long']:
                result['target'] = current_price + (sl_distance * min_rr)
            else:
                result['target'] = current_price - (sl_distance * min_rr)
            
            result['tp_basis'] = f"{result['tp_basis']}_adjusted"
            result['risk_reward'] = min_rr
        
        return result
    def _find_clusters(self, levels, mid_price, side):
        """
        Enhanced wall detection to better identify price levels with high-density orders.
        Focuses on finding strong support/resistance levels with dense order concentration.
        
        Args:
            levels (list): List of (price, qty) tuples for bids or asks.
            mid_price (float): Current mid price for volatility scaling.
            side (str): 'bids' for support, 'asks' for resistance.
        
        Returns:
            list: Significant zones as (weighted_price, confidence, strength, total_qty) tuples.
        """
        if not levels or not mid_price or mid_price <= 0:
            return []

        # Initialize tracking for market-specific order size statistics
        if not hasattr(self, 'market_order_stats'):
            self.market_order_stats = {
                'avg_order_size': None,
                'max_order_size': None,
                'order_size_hist': [],  # Keep track of historical order sizes
                'price_level_liquidity': {}  # Track liquidity at price levels over time
            }
        
        # Track market liquidity (total volume in order book)
        total_market_liquidity = sum(qty for _, qty in levels)
        
        # Adjust scale dynamically based on price magnitude and volatility
        price_magnitude = np.log10(mid_price) if mid_price > 0 else 0
        base_tick_scale = 10 ** (np.floor(price_magnitude) - 4)
        volatility_scale = self.volatility_estimate * mid_price if self.volatility_estimate and mid_price else 0.0001
        
        # Finer price binning for more precise wall detection
        volatility_adjusted_scale = max(base_tick_scale, volatility_scale * 0.1)

        # Pre-process to identify large orders and compute order size statistics
        all_order_sizes = [qty for _, qty in levels]
        if all_order_sizes:
            current_avg_size = np.mean(all_order_sizes)
            current_max_size = max(all_order_sizes)
            
            # Update global tracking of market order size
            if self.market_order_stats['avg_order_size'] is None:
                self.market_order_stats['avg_order_size'] = current_avg_size
                self.market_order_stats['max_order_size'] = current_max_size
            else:
                # Exponential moving average for stability
                self.market_order_stats['avg_order_size'] = 0.9 * self.market_order_stats['avg_order_size'] + 0.1 * current_avg_size
                self.market_order_stats['max_order_size'] = max(0.9 * self.market_order_stats['max_order_size'], current_max_size)
            
            # Keep a history of recent order sizes
            self.market_order_stats['order_size_hist'] = self.market_order_stats['order_size_hist'][-100:] + [current_avg_size]
        
        # Calculate dynamic thresholds for "large" orders based on market conditions
        avg_order_size = self.market_order_stats['avg_order_size'] or np.mean(all_order_sizes)
        large_order_threshold = avg_order_size * 5  # An order 5x the average size is considered large
        
        # Round prices to group into zones with appropriate precision
        rounded_levels = []
        for price, qty in levels:
            if side == 'bids':
                rounded_price = np.floor(price / volatility_adjusted_scale) * volatility_adjusted_scale
            else:  # asks
                rounded_price = np.ceil(price / volatility_adjusted_scale) * volatility_adjusted_scale
            rounded_levels.append((rounded_price, qty, price))

        # Aggregate orders by rounded price, with enhanced metrics
        price_zones = {}
        for rounded_price, qty, orig_price in rounded_levels:
            if rounded_price not in price_zones:
                price_zones[rounded_price] = {
                    'total_qty': 0,               # Total volume at this level
                    'orders': [],                  # Individual orders
                    'min_price': float('inf'),     # Min price in zone
                    'max_price': 0,                # Max price in zone
                    'order_count': 0,              # Number of orders
                    'price_points': set(),         # Unique price points
                    'max_order_size': 0,           # Size of largest single order
                    'large_order_count': 0,        # Count of "large" orders
                    'large_order_volume': 0,       # Volume from large orders
                    'size_distribution': [],       # Order sizes for distribution analysis
                    'price_concentration': 0       # Measure of order concentration
                }
            zone = price_zones[rounded_price]
            zone['total_qty'] += qty
            zone['orders'].append((orig_price, qty))
            zone['min_price'] = min(zone['min_price'], orig_price)
            zone['max_price'] = max(zone['max_price'], orig_price)
            zone['order_count'] += 1
            zone['price_points'].add(orig_price)
            zone['max_order_size'] = max(zone['max_order_size'], qty)
            zone['size_distribution'].append(qty)
            
            # Track large orders
            if qty > large_order_threshold:
                zone['large_order_count'] += 1
                zone['large_order_volume'] += qty
        
        # Calculate price concentration - important for true wall detection
        for price, zone in price_zones.items():
            # Create a histogram of orders by exact price point
            price_counts = {}
            for p, q in zone['orders']:
                if p not in price_counts:
                    price_counts[p] = 0
                price_counts[p] += q
            
            # Find the most concentrated price point
            if price_counts:
                max_price_vol = max(price_counts.values())
                # Concentration ratio: volume at most concentrated price / total volume
                zone['price_concentration'] = max_price_vol / zone['total_qty'] if zone['total_qty'] > 0 else 0
        
        # Select cluster history based on side
        cluster_history = self.bid_cluster_history if side == 'bids' else self.ask_cluster_history
        
        # Calculate net flow from order book deltas for Wall stability assessment
        additions = self.delta_metrics.get(f'{side[:-1]}_additions', [])  # e.g., 'bid_additions'
        cancellations = self.delta_metrics.get(f'{side[:-1]}_cancellations', [])
        net_flow = defaultdict(float)
        
        for price, qty in additions:
            rounded = self._round_price(price, side, volatility_adjusted_scale)
            net_flow[rounded] += qty
        for price, qty in cancellations:
            rounded = self._round_price(price, side, volatility_adjusted_scale)
            net_flow[rounded] -= qty
        
        # Enhanced volume distribution analysis for significance
        volumes = [zone['total_qty'] for zone in price_zones.values()]
        if not volumes:
            return []
        
        total_volume = sum(volumes)
        if total_volume <= 0:
            return []

        median_vol = np.median(volumes)
        mean_vol = np.mean(volumes)
        vol_std = np.std(volumes) if len(volumes) > 1 else mean_vol * 0.5 or 1.0
        
        # Calculate market-adaptive percentiles for volume significance
        concentration_ratio = vol_std / mean_vol if mean_vol > 0 else 1.0
        significance_percentile = min(75 + concentration_ratio * 5, 85)
        
        # Dynamically determine z-score threshold based on market conditions
        z_score_base = 1.5
        z_score_threshold = z_score_base + (self.volatility_estimate * 50)
        z_score_threshold = min(max(z_score_threshold, 1.2), 2.5)  # Keep reasonable
        
        # Importance of large orders increases in less liquid markets
        large_order_importance = 0.5 * (1.0 - min(1.0, total_market_liquidity / (median_vol * 50)))
        
        significant_zones = []
        for rounded_price, zone in price_zones.items():
            # Extract key metrics
            volume = zone['total_qty']
            order_count = zone['order_count']
            max_order_size = zone['max_order_size']
            large_order_count = zone['large_order_count']
            large_order_volume = zone['large_order_volume']
            price_concentration = zone['price_concentration']
            
            # Calculate importance relative to market
            volume_ratio = volume / median_vol if median_vol > 0 else 1.0
            vol_pct_of_total = (volume / total_volume) * 100 if total_volume > 0 else 0
            z_score = (volume - mean_vol) / vol_std if vol_std > 0 else 0
            
            # Calculate price spread of the zone - tighter = better wall
            price_spread = (zone['max_price'] - zone['min_price']) / mid_price if mid_price > 0 else 0
            price_tightness = 1.0 - min(1.0, price_spread / (volatility_adjusted_scale * 2))
            
            # Order density: more orders at same price = stronger wall
            order_density = order_count / len(zone['price_points']) if zone['price_points'] else 1.0
            
            # Check if this is a psychological level
            is_psych_level = self._is_psychological_level(rounded_price)
            
            # Calculate large order ratio
            large_order_ratio = large_order_volume / volume if volume > 0 else 0
            
            # Enhanced strength calculation with greater emphasis on:
            # 1. Concentration of volume at exact price levels (true walls)
            # 2. Large individual orders (whales)
            # 3. Order density
            base_strength = (
                (z_score * 0.4) +                            # Statistical significance
                (volume_ratio * 0.3) +                       # Relative to average liquidity
                (vol_pct_of_total * 0.05) +                  # Percentage of total book
                (order_density * 0.4) +                      # Density of orders
                (price_concentration * 0.8) +                # Concentration at exact price (critical)
                (price_tightness * 0.4) +                    # Tightness of price range
                (large_order_ratio * large_order_importance * 0.7)  # Large order importance
            )
            
            # Apply multipliers for special conditions
            psych_multiplier = 1.5 if is_psych_level else 1.0
            
            # Large orders are more important for wall strength
            large_order_multiplier = 1.0
            if large_order_count > 0:
                large_order_multiplier = 1.0 + (large_order_ratio * 0.8)  # Up to 80% bonus
            
            # Calculate final wall strength
            wall_strength = base_strength * psych_multiplier * large_order_multiplier
            
            # Persistence boost: stronger walls persist over time
            persistence_count = sum(1 for hist_set in cluster_history if rounded_price in hist_set)
            persistence_score = persistence_count / cluster_history.maxlen if cluster_history.maxlen else 0
            wall_strength *= (1 + persistence_score * 0.5)  # Up to 50% boost for persistence
            
            # Flow boost: adjust strength based on net order flow
            net_flow_value = net_flow.get(rounded_price, 0)
            flow_ratio = net_flow_value / volume if volume > 0 else 0
            flow_weight = 0.2
            wall_strength *= (1 + flow_ratio * flow_weight)
            
            # Enhanced significance criteria with focus on concentrated liquidity
            is_volume_significant = volume > np.percentile(volumes, significance_percentile) if len(volumes) >= 4 else volume > mean_vol
            is_pct_significant = vol_pct_of_total >= max(8.0, 5.0 + concentration_ratio * 3)  # Dynamic threshold
            is_zscore_significant = z_score >= z_score_threshold
            is_concentration_significant = price_concentration > 0.6  # 60% of volume at same price = significant
            is_large_orders_significant = large_order_volume > volume * 0.5  # 50% from large orders
            is_psych_with_volume = is_psych_level and volume > median_vol * 0.5
            
            # Combined significance criteria - prioritize concentration and large orders
            is_significant = (
                is_volume_significant or 
                is_pct_significant or 
                is_zscore_significant or 
                is_concentration_significant or  # New criteria
                is_large_orders_significant or   # New criteria
                is_psych_with_volume
            )
            
            if is_significant:
                # Calculate weighted average price for precise level identification
                wgt_price = sum(p * q for p, q in zone['orders']) / volume if volume else rounded_price
                
                # Calculate confidence score (0-1) - how certain we are this is a real wall
                # Higher with more concentration, large orders, and persistence
                confidence_base = 0.4
                
                # Normalize wall strength to 0-1 scale for confidence calculation
                norm_strength = min(1.0, wall_strength / 10) if wall_strength > 0 else 0.2
                
                confidence_boost = (
                    (price_concentration * 0.3) +             # Concentration at price
                    (large_order_ratio * 0.25) +              # Large orders
                    (persistence_score * 0.15) +              # Persistence over time
                    (norm_strength * 0.2)                     # Wall strength
                )
                confidence = min(0.95, confidence_base + confidence_boost)
                
                # Include total quantity as fourth value for better SL/TP calculation
                significant_zones.append((wgt_price, confidence, wall_strength, volume))
        
        # Update cluster history
        cluster_history.append(set(price_zones.keys()))
        
        # Sort by strength for easier access to strongest walls
        return sorted(significant_zones, key=lambda x: x[2], reverse=True)
    def _round_price(self, price, side, scale):
        """Helper to round prices consistently with clustering."""
        return np.floor(price / scale) * scale if side == 'bids' else np.ceil(price / scale) * scale    
    def _is_psychological_level(self, price):
        """Check if a price is a psychological level (round number)."""
        # Convert to string to check digits
        price_str = str(price)
        
        # Check for whole numbers
        if price_str.endswith('.0') or price_str.endswith('.00'):
            return True
            
        # Check if ends with multiple zeros
        if price_str.endswith('00') or price_str.endswith('000'):
            return True
            
        # Check for .5 levels (half numbers)
        if price_str.endswith('.5') or price_str.endswith('.50'):
            return True
            
        # Check for .25 or .75 levels (quarter numbers)
        if price_str.endswith('.25') or price_str.endswith('.75'):
            return True
            
        return False
    
    def _calculate_buying_pressure(self):
        """Calculate current buying pressure from recent order flow."""
        if not self.order_flow_history:
            return 0.5  # Neutral
            
        # Get recent history (last 10 snapshots)
        recent = list(self.order_flow_history)[-min(10, len(self.order_flow_history)):]
        
        # Calculate aggressive buy ratio
        total_aggressive_buy = sum(entry['aggressive_buy_volume'] for entry in recent)
        total_aggressive_sell = sum(entry['aggressive_sell_volume'] for entry in recent)
        
        # Aggressive ratio (0-1 scale)
        if total_aggressive_buy + total_aggressive_sell > 0:
            aggressive_ratio = total_aggressive_buy / (total_aggressive_buy + total_aggressive_sell)
        else:
            aggressive_ratio = 0.5
        
        # Get bid additions vs cancellations if we have delta metrics
        bid_additions = sum(qty for _, qty in self.delta_metrics.get('bid_additions', []))
        bid_cancellations = sum(qty for _, qty in self.delta_metrics.get('bid_cancellations', []))
        
        # Order placement ratio
        if bid_additions + bid_cancellations > 0:
            placement_ratio = bid_additions / (bid_additions + bid_cancellations)
        else:
            placement_ratio = 0.5
        
        # Combine metrics (weighted average)
        buying_pressure = (aggressive_ratio * 0.7) + (placement_ratio * 0.3)
        
        return buying_pressure
    
    def _calculate_selling_pressure(self):
        """Calculate current selling pressure from recent order flow."""
        if not self.order_flow_history:
            return 0.5  # Neutral
            
        # Get recent history (last 10 snapshots)
        recent = list(self.order_flow_history)[-min(10, len(self.order_flow_history)):]
        
        # Calculate aggressive sell ratio
        total_aggressive_buy = sum(entry['aggressive_buy_volume'] for entry in recent)
        total_aggressive_sell = sum(entry['aggressive_sell_volume'] for entry in recent)
        
        # Aggressive ratio (0-1 scale)
        if total_aggressive_buy + total_aggressive_sell > 0:
            aggressive_ratio = total_aggressive_sell / (total_aggressive_buy + total_aggressive_sell)
        else:
            aggressive_ratio = 0.5
        
        # Get ask additions vs cancellations if we have delta metrics
        ask_additions = sum(qty for _, qty in self.delta_metrics.get('ask_additions', []))
        ask_cancellations = sum(qty for _, qty in self.delta_metrics.get('ask_cancellations', []))
        
        # Order placement ratio
        if ask_additions + ask_cancellations > 0:
            placement_ratio = ask_additions / (ask_additions + ask_cancellations)
        else:
            placement_ratio = 0.5
        
        # Combine metrics (weighted average)
        selling_pressure = (aggressive_ratio * 0.7) + (placement_ratio * 0.3)
        
        return selling_pressure

    def update_volume_data(self, timestamp, volume, price=None):
        """Update volume history and calculate volume metrics."""
        if volume is None or volume <= 0: 
            return
            
        if timestamp is None: 
            timestamp = time.time()

        # Store in history
        self.volume_history.append({
            'timestamp': timestamp,
            'volume': float(volume),
            'price': float(price) if price is not None else None
        })
        
        # Calculate volume metrics immediately
        self._calculate_volume_metrics()
        
    def _calculate_volume_metrics(self):
        """Calculate volume-based metrics for better signal generation."""
        # Default metrics structure
        self.volume_metrics = {
            'latest_volume': 0,
            'avg_volume': 0,
            'volume_trend': 'neutral',
            'volume_strength': 'normal',
            'volume_spike': False,
            'volume_drop': False,
            'relative_volume': 1.0,
            'volume_momentum': 0
        }
        
        # Need at least a few data points
        if len(self.volume_history) < 3:
            if self.volume_history:
                self.volume_metrics['latest_volume'] = self.volume_history[-1]['volume']
            return
        
        # Get latest and historical volume data
        volumes = [entry['volume'] for entry in self.volume_history]
        latest_volume = volumes[-1]
        self.volume_metrics['latest_volume'] = latest_volume
        
        # Calculate average volume (excluding most recent)
        historical_volumes = volumes[:-1]
        avg_volume = sum(historical_volumes) / len(historical_volumes)
        self.volume_metrics['avg_volume'] = avg_volume
        
        # Calculate relative volume
        relative_volume = latest_volume / avg_volume if avg_volume > 0 else 1.0
        self.volume_metrics['relative_volume'] = relative_volume
        
        # Determine volume strength and detect spikes/drops
        # Use dynamic thresholds based on historical variation
        if len(historical_volumes) >= 5:
            vol_std = np.std(historical_volumes)
            vol_mean = np.mean(historical_volumes)
            if vol_mean > 0:
                z_score = (latest_volume - vol_mean) / vol_std if vol_std > 0 else 0
                
                if z_score > 2.0:
                    self.volume_metrics['volume_strength'] = 'high'
                    self.volume_metrics['volume_spike'] = True
                elif z_score < -1.5:
                    self.volume_metrics['volume_strength'] = 'low'
                    self.volume_metrics['volume_drop'] = True
        else:
            # Fallback if insufficient data
            if relative_volume > 2.0:
                self.volume_metrics['volume_strength'] = 'high'
                self.volume_metrics['volume_spike'] = True
            elif relative_volume < 0.5:
                self.volume_metrics['volume_strength'] = 'low'
                self.volume_metrics['volume_drop'] = True
        
        # Calculate volume trend (increasing or decreasing)
        if len(volumes) >= 6:
            first_half = sum(volumes[:len(volumes)//2])
            second_half = sum(volumes[len(volumes)//2:])
            
            if second_half > first_half * 1.2:
                self.volume_metrics['volume_trend'] = 'increasing'
            elif second_half < first_half * 0.8:
                self.volume_metrics['volume_trend'] = 'decreasing'
        
        # Calculate volume momentum (rate of change)
        if len(volumes) >= 3:
            recent_vols = volumes[-3:]
            if recent_vols[0] > 0:
                self.volume_metrics['volume_momentum'] = (recent_vols[2] - recent_vols[0]) / recent_vols[0]

    def _update_price_history(self, df):
        """Update price history for ATR calculation and multi-timeframe analysis."""
        if df is not None and not df.empty:
            new_data = df[['open', 'high', 'low', 'close', 'volume']].copy() if 'volume' in df.columns else df[['open', 'high', 'low', 'close']].copy()
            
            # If the DataFrame has a volume column, use it to update volume metrics
            if 'volume' in new_data.columns:
                latest_row = new_data.iloc[-1]
                self.update_volume_data(
                    timestamp=latest_row.name.timestamp() if hasattr(latest_row.name, 'timestamp') else time.time(),
                    volume=latest_row['volume'],
                    price=latest_row['close']
                )
            
            # Update price history
            self.price_history_for_atr = pd.concat([self.price_history_for_atr, new_data])
            self.price_history_for_atr = self.price_history_for_atr[~self.price_history_for_atr.index.duplicated(keep='last')]
            self.price_history_for_atr = self.price_history_for_atr.iloc[-100:]  # Keep only recent data

    def _calculate_atr(self, window=None):
        """Calculate ATR using price history with validation."""
        if window is None: 
            window = self.atr_window
            
        df = self.price_history_for_atr
        
        if df is None or len(df) < window + 1:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        # Ensure we have enough valid data
        valid_rows = (~high.isna()) & (~low.isna()) & (~close.isna())
        if valid_rows.sum() < window:
            return None
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Use pandas rolling to calculate the ATR
        atr = tr.rolling(window=window, min_periods=window).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else None

    
    def generate_signals(self, df=None, price=None, order_book=None, open_interest=None):
        """
        Generate trading signals with enhanced wall detection and SL/TP calculation.
        Handles both breakout and fade trading with fully dynamic parameters.
        """
        # Update data sources
        if order_book:
            self.update_order_book(order_book.get('bids', []), order_book.get('asks', []))
        if df is not None:
            self._update_price_history(df)
        
        # Get current price and volatility
        current_price = price if price is not None and price > 0 else self.order_book_cache.get('mid_price')
        current_atr = self._calculate_atr() or (current_price * self.volatility_estimate)
        volatility_factor = current_atr / current_price if current_price else 0.002
        
        # Initialize signal dictionary
        signal = {
            'signal': 'none', 'reason': '', 'strength': 0, 'strategy_type': 'none',
            'entry_price': current_price, 'stop_loss': 0, 'target': 0,
            'sl_basis': 'none', 'tp_basis': 'none', 'risk_reward': 0,
            'nearest_support': None, 'nearest_resistance': None,
            'probability': 0.5, 'all_supports': [], 'all_resistances': []
        }
        
        # Basic validation
        if not current_price or current_price <= 0 or not self.order_book_cache:
            return signal
        
        # Get enhanced support/resistance levels with better wall detection
        support_levels = self.order_book_cache.get('support_levels', [])
        resistance_levels = self.order_book_cache.get('resistance_levels', [])
        
        # Store all levels for reference
        signal['all_supports'] = support_levels
        signal['all_resistances'] = resistance_levels
        
        # Calculate dynamic proximity thresholds based on volatility
        close_proximity = min(volatility_factor * 0.5, 0.003)
        medium_proximity = min(volatility_factor * 1.0, 0.006)
        breakout_proximity = min(volatility_factor * 2.0, 0.01)
        
        # Trade signals dictionary to track potential setups
        trade_signals = {
            'fade_long': {'level': None, 'strength': 0, 'score': 0, 'reasons': []},
            'fade_short': {'level': None, 'strength': 0, 'score': 0, 'reasons': []},
            'breakout_long': {'level': None, 'strength': 0, 'score': 0, 'reasons': []},
            'breakout_short': {'level': None, 'strength': 0, 'score': 0, 'reasons': []}
        }
        
        # STEP 1: Identify potential trade setups using enhanced wall detection
        
        # Find walls below and above current price
        supports_below = [(price, conf, strength, vol) for price, conf, strength, vol in support_levels 
                        if price < current_price]
        supports_above = [(price, conf, strength, vol) for price, conf, strength, vol in support_levels 
                        if price > current_price]
        resistance_below = [(price, conf, strength, vol) for price, conf, strength, vol in resistance_levels 
                        if price < current_price]
        resistance_above = [(price, conf, strength, vol) for price, conf, strength, vol in resistance_levels 
                        if price > current_price]
        
        # Sort each by distance to price (closest first)
        for level_list in [supports_below, supports_above, resistance_below, resistance_above]:
            level_list.sort(key=lambda x: abs(x[0] - current_price) / current_price)
        
        # Analyze potential fade trades
        
        # FADE LONG: Buy at support
        if supports_below:
            support_wall = supports_below[0]  # Closest support
            support_price, support_conf, support_strength, support_vol = support_wall
            
            # Calculate proximity as percentage of price
            proximity_pct = (current_price - support_price) / current_price
            
            # Consider fade long if close to strong support
            if proximity_pct <= medium_proximity:
                # Higher score for stronger walls and closer proximity
                proximity_score = ((medium_proximity - proximity_pct) / medium_proximity) * 30 if medium_proximity > 0 else 15
                strength_score = min(support_strength, 10) * 5  # Scale strength to 0-50
                total_score = proximity_score + strength_score
                
                # Only consider if wall is significant
                is_strong_enough = support_strength >= 3.0 or support_conf >= 0.6
                
                if is_strong_enough and proximity_pct <= close_proximity:
                    trade_signals['fade_long']['level'] = support_wall
                    trade_signals['fade_long']['strength'] = support_strength
                    trade_signals['fade_long']['score'] = total_score
                    trade_signals['fade_long']['reasons'] = [
                        f"Support Wall @{support_price:.6f} (strength:{support_strength:.2f}, volume:{support_vol:.1f}, proximity:{proximity_pct*100:.3f}%)"
                    ]
        
        # FADE SHORT: Sell at resistance
        if resistance_above:
            resistance_wall = resistance_above[0]  # Closest resistance
            resistance_price, resistance_conf, resistance_strength, resistance_vol = resistance_wall
            
            # Calculate proximity as percentage of price
            proximity_pct = (resistance_price - current_price) / current_price
            
            # Consider fade short if close to strong resistance
            if proximity_pct <= medium_proximity:
                # Higher score for stronger walls and closer proximity
                proximity_score = ((medium_proximity - proximity_pct) / medium_proximity) * 30 if medium_proximity > 0 else 15
                strength_score = min(resistance_strength, 10) * 5  # Scale strength to 0-50
                total_score = proximity_score + strength_score
                
                # Only consider if wall is significant
                is_strong_enough = resistance_strength >= 3.0 or resistance_conf >= 0.6
                
                if is_strong_enough and proximity_pct <= close_proximity:
                    trade_signals['fade_short']['level'] = resistance_wall
                    trade_signals['fade_short']['strength'] = resistance_strength
                    trade_signals['fade_short']['score'] = total_score
                    trade_signals['fade_short']['reasons'] = [
                        f"Resistance Wall @{resistance_price:.6f} (strength:{resistance_strength:.2f}, volume:{resistance_vol:.1f}, proximity:{proximity_pct*100:.3f}%)"
                    ]
        
        # Analyze potential breakout trades
        
        # BREAKOUT LONG: Buy above broken resistance
        if resistance_below:
            broken_resistance = resistance_below[0]  # Closest broken resistance
            resistance_price, resistance_conf, resistance_strength, resistance_vol = broken_resistance
            
            # Calculate proximity as percentage of price
            proximity_pct = (current_price - resistance_price) / current_price
            
            # Consider breakout long if recently broken strong resistance
            if proximity_pct <= breakout_proximity:
                # Higher score for stronger walls and fresher breakouts
                proximity_score = ((breakout_proximity - proximity_pct) / breakout_proximity) * 40 if breakout_proximity > 0 else 20
                strength_score = min(resistance_strength, 10) * 5  # Scale strength to 0-50
                total_score = proximity_score + strength_score
                
                # Only consider if wall is significant
                is_strong_enough = resistance_strength >= 2.5 or resistance_conf >= 0.5
                
                if is_strong_enough:
                    trade_signals['breakout_long']['level'] = broken_resistance
                    trade_signals['breakout_long']['strength'] = resistance_strength
                    trade_signals['breakout_long']['score'] = total_score
                    trade_signals['breakout_long']['reasons'] = [
                        f"Breakout Above Resistance @{resistance_price:.6f} (strength:{resistance_strength:.2f}, volume:{resistance_vol:.1f}, proximity:{proximity_pct*100:.3f}%)"
                    ]
        
        # BREAKOUT SHORT: Sell below broken support
        if supports_above:
            broken_support = supports_above[0]  # Closest broken support
            support_price, support_conf, support_strength, support_vol = broken_support
            
            # Calculate proximity as percentage of price
            proximity_pct = (support_price - current_price) / current_price
            
            # Consider breakout short if recently broken strong support
            if proximity_pct <= breakout_proximity:
                # Higher score for stronger walls and fresher breakouts
                proximity_score = ((breakout_proximity - proximity_pct) / breakout_proximity) * 40 if breakout_proximity > 0 else 20
                strength_score = min(support_strength, 10) * 5  # Scale strength to 0-50
                total_score = proximity_score + strength_score
                
                # Only consider if wall is significant
                is_strong_enough = support_strength >= 2.5 or support_conf >= 0.5
                
                if is_strong_enough:
                    trade_signals['breakout_short']['level'] = broken_support
                    trade_signals['breakout_short']['strength'] = support_strength
                    trade_signals['breakout_short']['score'] = total_score
                    trade_signals['breakout_short']['reasons'] = [
                        f"Breakout Below Support @{support_price:.6f} (strength:{support_strength:.2f}, volume:{support_vol:.1f}, proximity:{proximity_pct*100:.3f}%)"
                    ]
        
        # STEP 2: Add confirmation signals based on order book, price action, and volume
        
        # Add different confirmations based on strategy type
        for strategy, data in trade_signals.items():
            if data['level'] is None:
                continue
            
            # CONFIRMATION TYPE 1: Price action confirmation
            if df is not None and not df.empty:
                latest_candle = df.iloc[-1]
                if not latest_candle.empty:
                    try:
                        # Different price action confirmation based on strategy
                        if strategy == 'fade_long' and 'low' in latest_candle:
                            # For fade long: Check for lower wick (rejection)
                            candle_range = latest_candle['high'] - latest_candle['low'] if 'high' in latest_candle else 0
                            if candle_range > 0:
                                close_price = latest_candle['close'] if 'close' in latest_candle else current_price
                                open_price = latest_candle['open'] if 'open' in latest_candle else current_price
                                
                                lower_wick = close_price - latest_candle['low'] if close_price > latest_candle['low'] else open_price - latest_candle['low']
                                lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
                                
                                if lower_wick_ratio > 0:
                                    normalized_wick = lower_wick / current_atr if current_atr > 0 else 0
                                    wick_score = lower_wick_ratio * 30 + (normalized_wick * 10 if normalized_wick > 0.5 else 0)
                                    data['score'] += wick_score
                                    data['reasons'].append(f"Price Rejection (wick:{lower_wick_ratio:.2f}, {normalized_wick:.1f}x ATR)")
                        
                        elif strategy == 'fade_short' and 'high' in latest_candle:
                            # For fade short: Check for upper wick (rejection)
                            candle_range = latest_candle['high'] - latest_candle['low'] if 'low' in latest_candle else 0
                            if candle_range > 0:
                                close_price = latest_candle['close'] if 'close' in latest_candle else current_price
                                open_price = latest_candle['open'] if 'open' in latest_candle else current_price
                                
                                upper_wick = latest_candle['high'] - close_price if close_price < latest_candle['high'] else latest_candle['high'] - open_price
                                upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
                                
                                if upper_wick_ratio > 0:
                                    normalized_wick = upper_wick / current_atr if current_atr > 0 else 0
                                    wick_score = upper_wick_ratio * 30 + (normalized_wick * 10 if normalized_wick > 0.5 else 0)
                                    data['score'] += wick_score
                                    data['reasons'].append(f"Price Rejection (wick:{upper_wick_ratio:.2f}, {normalized_wick:.1f}x ATR)")
                        
                        elif strategy == 'breakout_long' and 'close' in latest_candle and 'open' in latest_candle:
                            # For breakout long: Check for bullish momentum
                            is_bullish = latest_candle['close'] > latest_candle['open']
                            if is_bullish:
                                momentum_score = 15
                                data['score'] += momentum_score
                                data['reasons'].append("Bullish Momentum")
                        
                        elif strategy == 'breakout_short' and 'close' in latest_candle and 'open' in latest_candle:
                            # For breakout short: Check for bearish momentum
                            is_bearish = latest_candle['close'] < latest_candle['open']
                            if is_bearish:
                                momentum_score = 15
                                data['score'] += momentum_score
                                data['reasons'].append("Bearish Momentum")
                    except Exception as e:
                        logger.debug(f"Error in price action analysis: {e}")
            
            # CONFIRMATION TYPE 2: Volume analysis
            if hasattr(self, 'volume_metrics'):
                try:
                    vol_ratio = self.volume_metrics.get('relative_volume', 1.0)
                    
                    # Volume requirements differ by strategy
                    vol_threshold = 1.5 if strategy.startswith('breakout') else 1.0
                    
                    if vol_ratio > vol_threshold:
                        vol_score = min(30, (vol_ratio - vol_threshold) * 10)
                        data['score'] += vol_score
                        data['reasons'].append(f"Volume Confirmation ({vol_ratio:.1f}x avg)")
                except Exception as e:
                    logger.debug(f"Error in volume analysis: {e}")
            
            # CONFIRMATION TYPE 3: Order book flow
            if hasattr(self, 'order_flow_metrics') and hasattr(self, 'delta_metrics'):
                try:
                    # Different order flow confirmations by strategy
                    if strategy == 'fade_long':
                        # For fade long: Look for bid absorption
                        if self.delta_metrics.get('absorption_detected', False):
                            for event in self.absorption_events:
                                if event['type'] == 'bid_absorption' and time.time() - event['timestamp'] < 60:
                                    abs_size = event.get('rel_size', 1.0)
                                    flow_score = min(30, abs_size * 10)
                                    data['score'] += flow_score
                                    data['reasons'].append(f"Buy Absorption ({abs_size:.1f}x)")
                                    break
                    
                    elif strategy == 'fade_short':
                        # For fade short: Look for ask absorption
                        if self.delta_metrics.get('absorption_detected', False):
                            for event in self.absorption_events:
                                if event['type'] == 'ask_absorption' and time.time() - event['timestamp'] < 60:
                                    abs_size = event.get('rel_size', 1.0)
                                    flow_score = min(30, abs_size * 10)
                                    data['score'] += flow_score
                                    data['reasons'].append(f"Sell Absorption ({abs_size:.1f}x)")
                                    break
                    
                    elif strategy == 'breakout_long':
                        # For breakout long: Look for aggressive buying
                        aggression_ratio = self.order_flow_metrics.get('aggression_ratio', 0.5)
                        if aggression_ratio > 0.5 + (volatility_factor * 5):
                            flow_score = (aggression_ratio - 0.5) * 50
                            data['score'] += flow_score
                            data['reasons'].append(f"Aggressive Buying ({aggression_ratio:.2f})")
                    
                    elif strategy == 'breakout_short':
                        # For breakout short: Look for aggressive selling
                        aggression_ratio = self.order_flow_metrics.get('aggression_ratio', 0.5)
                        if aggression_ratio < 0.5 - (volatility_factor * 5):
                            flow_score = (0.5 - aggression_ratio) * 50
                            data['score'] += flow_score
                            data['reasons'].append(f"Aggressive Selling ({aggression_ratio:.2f})")
                except Exception as e:
                    logger.debug(f"Error in order flow analysis: {e}")
        
        # STEP 3: Calculate dynamic signal thresholds based on market conditions
        
        # Base threshold adjusted for volatility
        base_threshold = 50 + (volatility_factor * 500)
        base_threshold = min(max(base_threshold, 40), 80)
        
        # Different thresholds for different strategies
        fade_threshold = base_threshold
        breakout_threshold = base_threshold * 0.8  # Lower threshold for breakouts
        
        # STEP 4: Select the best signal
        
        best_signal = None
        best_score = 0
        
        for strategy, data in trade_signals.items():
            if data['level'] is None:
                continue
            
            # Apply appropriate threshold
            threshold = breakout_threshold if strategy.startswith('breakout') else fade_threshold
            
            # Select if above threshold and better than current best
            if data['score'] >= threshold and data['score'] > best_score:
                best_signal = strategy
                best_score = data['score']
        
        # Early exit if no signal passes threshold
        if best_signal is None:
            return signal
        
        # STEP 5: Set signal properties
        
        best_data = trade_signals[best_signal]
        signal['signal'] = 'buy' if best_signal in ['fade_long', 'breakout_long'] else 'sell'
        signal['strategy_type'] = best_signal
        signal['reason'] = ", ".join(best_data['reasons'])
        signal['strength'] = best_data['score']
        signal['probability'] = min(0.95, best_data['score'] / 100)
        
        # STEP 6: Calculate SL/TP using the enhanced function
        
        sltp_result = self._calculate_sl_tp_from_walls(
            best_signal, 
            current_price, 
            support_levels, 
            resistance_levels, 
            volatility_factor
        )
        
        # Apply the results
        signal['stop_loss'] = sltp_result['stop_loss']
        signal['target'] = sltp_result['target']
        signal['sl_basis'] = sltp_result['sl_basis']
        signal['tp_basis'] = sltp_result['tp_basis']
        signal['risk_reward'] = sltp_result['risk_reward']
        
        # Record nearest support/resistance for reference
        for level in support_levels:
            level_price = level[0]
            if level_price < current_price and (signal['nearest_support'] is None or 
                abs(level_price - current_price) < abs(signal['nearest_support'] - current_price)):
                signal['nearest_support'] = level_price
        
        for level in resistance_levels:
            level_price = level[0]
            if level_price > current_price and (signal['nearest_resistance'] is None or 
                abs(level_price - current_price) < abs(signal['nearest_resistance'] - current_price)):
                signal['nearest_resistance'] = level_price
        
        return signal

    def check_exit_conditions(self, current_price, entry_price, stop_loss, target, position_type,
                             trailing_activated=False, highest_reached=None, lowest_reached=None,
                             order_book=None):
        """
        Check exit conditions with enhanced order book delta analysis.
        """
        result = {
            'exit_triggered': False,
            'exit_reason': 'No exit',
            'profit_pct': 0.0,
            'new_stop_loss': stop_loss,
            'trailing_activated': trailing_activated,
            'highest_reached': highest_reached,
            'lowest_reached': lowest_reached
        }
        
        # Update data if provided
        if order_book:
            self.update_order_book(order_book.get('bids', []), order_book.get('asks', []))
        
        # Validate inputs
        if not entry_price or entry_price <= 0:
            return result
        
        # Calculate profit percentage
        if position_type == 'long':
            result['profit_pct'] = (current_price - entry_price) / entry_price * 100
        else:  # short
            result['profit_pct'] = (entry_price - current_price) / entry_price * 100
        
        # Trailing stop logic
        if position_type == 'long':
            # Update highest price
            if highest_reached is None or current_price > highest_reached:
                highest_reached = current_price
            result['highest_reached'] = highest_reached
            
            # Check if trailing should activate
            if not trailing_activated and highest_reached is not None:
                profit_achieved_pct = (highest_reached - entry_price) / entry_price * 100
                if profit_achieved_pct >= self.trailing_start_pct:
                    trailing_activated = True
                    result['trailing_activated'] = True
                    logger.info(f"{self.symbol} LONG Trailing Stop Activated at {profit_achieved_pct:.2f}% profit")
            
            # Update trailing stop if activated
            if trailing_activated and highest_reached is not None:
                # Dynamic trailing distance based on volatility
                trailing_dist_pct = max(self.trailing_distance_pct, self.volatility_estimate * 20)
                new_stop = highest_reached * (1 - trailing_dist_pct/100)
                if new_stop > stop_loss:
                    result['new_stop_loss'] = new_stop
                    stop_loss = new_stop  # Update for immediate check
        
        else:  # short position
            # Update lowest price
            if lowest_reached is None or current_price < lowest_reached:
                lowest_reached = current_price
            result['lowest_reached'] = lowest_reached
            
            # Check if trailing should activate
            if not trailing_activated and lowest_reached is not None:
                profit_achieved_pct = (entry_price - lowest_reached) / entry_price * 100
                if profit_achieved_pct >= self.trailing_start_pct:
                    trailing_activated = True
                    result['trailing_activated'] = True
                    logger.info(f"{self.symbol} SHORT Trailing Stop Activated at {profit_achieved_pct:.2f}% profit")
            
            # Update trailing stop if activated
            if trailing_activated and lowest_reached is not None:
                # Dynamic trailing distance based on volatility
                trailing_dist_pct = max(self.trailing_distance_pct, self.volatility_estimate * 20)
                new_stop = lowest_reached * (1 + trailing_dist_pct/100)
                if new_stop < stop_loss:
                    result['new_stop_loss'] = new_stop
                    stop_loss = new_stop  # Update for immediate check
        
        # Check standard exit conditions
        if position_type == 'long':
            if current_price <= stop_loss:
                result['exit_triggered'] = True
                result['exit_reason'] = 'Stop Loss'
            elif current_price >= target:
                result['exit_triggered'] = True
                result['exit_reason'] = 'Target Reached'
        else:  # short
            if current_price >= stop_loss:
                result['exit_triggered'] = True
                result['exit_reason'] = 'Stop Loss'
            elif current_price <= target:
                result['exit_triggered'] = True
                result['exit_reason'] = 'Target Reached'
        
        # Enhanced exit conditions based on order book deltas
        if not result['exit_triggered'] and self.delta_metrics and self.order_book_cache:
            # Get order flow metrics
            aggression_ratio = self.order_book_cache.get('aggression_ratio', 0.5)
            
            # Check for significant order cancellations (order pulling)
            if position_type == 'long' and result['profit_pct'] > 0.5:
                # For long positions, check for:
                
                # 1. Rapid bid cancellations (support being pulled)
                if self.delta_metrics.get('pulling_detected', False):
                    bid_cancel_near_mid = sum(qty for price, qty in self.delta_metrics.get('bid_cancellations', []) 
                                          if (current_price - price) / current_price < self.volatility_estimate * 10)
                    
                    # Get recent bid volume for comparison
                    recent_bid_vol = self.order_book_cache.get('bid_volume_nearest', 0)
                    
                    # If significant cancellations relative to book size
                    if recent_bid_vol > 0 and bid_cancel_near_mid / recent_bid_vol > 0.3:
                        result['exit_triggered'] = True
                        result['exit_reason'] = 'Support Pulling'
                
                # 2. Sharp shift in aggression to selling
                elif aggression_ratio < 0.3 and result['profit_pct'] > 1.0:
                    result['exit_triggered'] = True
                    result['exit_reason'] = 'Aggressive Selling'
            
            elif position_type == 'short' and result['profit_pct'] > 0.5:
                # For short positions, check for:
                
                # 1. Rapid ask cancellations (resistance being pulled)
                if self.delta_metrics.get('pulling_detected', False):
                    ask_cancel_near_mid = sum(qty for price, qty in self.delta_metrics.get('ask_cancellations', []) 
                                          if (price - current_price) / current_price < self.volatility_estimate * 10)
                    
                    # Get recent ask volume for comparison
                    recent_ask_vol = self.order_book_cache.get('ask_volume_nearest', 0)
                    
                    # If significant cancellations relative to book size
                    if recent_ask_vol > 0 and ask_cancel_near_mid / recent_ask_vol > 0.3:
                        result['exit_triggered'] = True
                        result['exit_reason'] = 'Resistance Pulling'
                
                # 2. Sharp shift in aggression to buying
                elif aggression_ratio > 0.7 and result['profit_pct'] > 1.0:
                    result['exit_triggered'] = True
                    result['exit_reason'] = 'Aggressive Buying'
            
            # 3. Check for strong absorption against position
            if self.delta_metrics.get('absorption_detected', False):
                for event in list(self.absorption_events)[-3:]:  # Check last 3 events
                    # Only consider very recent events (last 10 seconds)
                    if time.time() - event['timestamp'] > 10:
                        continue
                    
                    # For long positions, ask absorption is bearish
                    if position_type == 'long' and event['type'] == 'ask_absorption' and result['profit_pct'] > 0.5:
                        result['exit_triggered'] = True
                        result['exit_reason'] = 'Sell Absorption'
                    
                    # For short positions, bid absorption is bullish
                    elif position_type == 'short' and event['type'] == 'bid_absorption' and result['profit_pct'] > 0.5:
                        result['exit_triggered'] = True
                        result['exit_reason'] = 'Buy Absorption'
        
        # Log exit reason if triggered
        if result['exit_triggered']:
            logger.info(f"{self.symbol} EXIT {position_type.upper()}: {result['exit_reason']} | "
                       f"Profit: {result['profit_pct']:.2f}%")
        
        return result

    def _track_pattern(self, pattern_name):
        """Track pattern detection for statistics."""
        if pattern_name in self.patterns:
            self.patterns[pattern_name]['total'] += 1
    
    def update_pattern_trade_result(self, pattern_name, profit_pct, win=False, with_trend=None):
        """Update pattern trade result statistics and queue save."""
        pattern_name = pattern_name.lower()
        
        # Map signal reasons to pattern names
        mapped_pattern = self._map_reason_to_pattern(pattern_name)
        
        if mapped_pattern and mapped_pattern in self.patterns:
            self.patterns[mapped_pattern]['trades'] += 1
            if win:
                self.patterns[mapped_pattern]['wins'] += 1
            # Queue save
            self._save_pattern_stats()
        else:
            logger.debug(f"Unknown pattern for stats update: {pattern_name} -> {mapped_pattern}")
    
    def _map_reason_to_pattern(self, reason):
        """Map signal reasons to pattern names."""
        reason_lower = reason.lower()
        
        # Simple mapping of common terms
        if 'absorption' in reason_lower and ('buy' in reason_lower or 'bid' in reason_lower):
            return 'order_book_absorption'
        if 'absorption' in reason_lower and ('sell' in reason_lower or 'ask' in reason_lower):
            return 'order_book_absorption'
        if 'iceberg' in reason_lower:
            return 'iceberg_detection'
        if 'bid delta' in reason_lower or 'buy delta' in reason_lower:
            return 'large_bid_delta'
        if 'ask delta' in reason_lower or 'sell delta' in reason_lower:
            return 'large_ask_delta'
        if 'pulling' in reason_lower:
            return 'order_pulling_detected'
        if 'defense' in reason_lower:
            return 'order_defense_detected'
        if 'aggressive buy' in reason_lower:
            return 'aggressive_buying'
        if 'aggressive sell' in reason_lower:
            return 'aggressive_selling'
        if 'support' in reason_lower:
            return 'hidden_support'
        if 'resistance' in reason_lower:
            return 'hidden_resistance'
        
        # Direct pattern name match
        if reason_lower in self.patterns:
            return reason_lower
        
        # No match
        return None
    
    def _load_pattern_stats(self):
        """Load pattern statistics from file."""
        try:
            if os.path.exists(self.full_pattern_stats_path):
                with open(self.full_pattern_stats_path, 'r') as f:
                    loaded_stats = json.load(f)
                
                # Update patterns with loaded data
                for pattern, stats in loaded_stats.items():
                    if pattern in self.patterns:
                        self.patterns[pattern].update(stats)
        except Exception as e:
            logger.warning(f"Error loading pattern stats for {self.symbol}: {e}")
    
    def _save_pattern_stats(self):
        """Queue pattern statistics for saving."""
        global file_update_queue
        if file_update_queue is None:
            return
        
        try:
            task = {
                'type': 'pattern_stats',
                'symbol': self.symbol,
                'data': self.patterns
            }
            file_update_queue.put(task)
        except Exception as e:
            logger.warning(f"Error queueing pattern stats for {self.symbol}: {e}")

    def compute_signals(self, open_, high, low, close, order_book=None, open_interest=None):
        """
        Compatibility method for the original interface - fixed to use latest price directly.
        """
        try:
            # Create DataFrame for internal use
            df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close})
            if isinstance(open_, pd.Series) and hasattr(open_, 'index'):
                df.index = open_.index
            
            # Get latest price - DIRECTLY from close parameter, no caching
            latest_price = None
            if isinstance(close, pd.Series) and not close.empty:
                latest_price = float(close.iloc[-1])  # Ensure it's a float for calculations
                if pd.isna(latest_price):
                    logger.warning(f"{self.symbol} Latest price is NaN, using alternate method")
                    latest_price = None
                    
            # Fallback if we couldn't get a price from the series
            if latest_price is None or latest_price <= 0:
                # Try to get price from order book
                if order_book and 'bids' in order_book and 'asks' in order_book and order_book['bids'] and order_book['asks']:
                    mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
                    latest_price = mid_price
                    logger.warning(f"{self.symbol} Using order book mid price: {latest_price}")
                else:
                    logger.error(f"{self.symbol} No valid price found for signal generation")
                    # Return empty signals
                    empty_buy = pd.Series(False, index=df.index)
                    empty_sell = pd.Series(False, index=df.index)
                    empty_exit = pd.Series(False, index=df.index)
                    empty_df = pd.DataFrame(index=df.index)
                    empty_df['signal_type'] = 'none'
                    empty_df['reason'] = 'No valid price'
                    return empty_buy, empty_sell, empty_exit, empty_df
            
            logger.debug(f"{self.symbol} Using price {latest_price} for signal generation")
            
            # Update order book before generating signals
            if order_book:
                self.update_order_book(order_book.get('bids', []), order_book.get('asks', []))
                
            # Generate signal with explicit current price
            signal_dict = self.generate_signals(
                df=df,
                price=latest_price,  # Pass explicitly, don't rely on cache
                order_book=None  # Already updated above
            )
            
            # Prepare output
            out_df = pd.DataFrame(index=df.index)
            out_df['buy_signal'] = False
            out_df['sell_signal'] = False
            out_df['exit_signal'] = False
            
            # Set signal flags
            if not df.empty:
                latest_idx = df.index[-1]
                if signal_dict['signal'] == 'buy':
                    out_df.loc[latest_idx, 'buy_signal'] = True
                elif signal_dict['signal'] == 'sell':
                    out_df.loc[latest_idx, 'sell_signal'] = True
            
            # Create signal info
            signal_info = self._create_signal_info(df, signal_dict)
            
            return out_df['buy_signal'], out_df['sell_signal'], out_df['exit_signal'], signal_info
        except Exception as e:
            logger.error(f"Error in compute_signals: {e}")
            # Return empty signals in case of error
            empty_df = pd.DataFrame(index=df.index if 'df' in locals() and df is not None else [])
            empty_df['signal_type'] = 'none'
            empty_df['reason'] = f'Error: {str(e)}'
            
            # Create empty signal series
            empty_buy = pd.Series(False, index=empty_df.index)
            empty_sell = pd.Series(False, index=empty_df.index)
            empty_exit = pd.Series(False, index=empty_df.index)
            
            return empty_buy, empty_sell, empty_exit, empty_df
    
    def _create_signal_info(self, df, signal_dict):
        """Create signal info DataFrame from signal dictionary."""
        signal_info = pd.DataFrame(index=df.index)
        
        # Initialize columns
        signal_info['signal_type'] = 'none'
        signal_info['reason'] = 'none'
        signal_info['probability'] = 0.0
        signal_info['entry_price'] = df['close']
        signal_info['stop_loss'] = float('nan')
        signal_info['target'] = float('nan')
        signal_info['atr'] = float('nan')
        signal_info['strength'] = 0
        signal_info['sl_basis'] = 'none'
        signal_info['tp_basis'] = 'none'
        signal_info['risk_reward'] = 0.0
        signal_info['wall_strength'] = 0.0
        
        # Add ATR if calculated
        current_atr = self._calculate_atr()
        if current_atr is not None:
            signal_info['atr'] = current_atr
        
        # Update last row with signal data if available
        if signal_dict['signal'] != 'none' and not df.empty:
            latest_idx = df.index[-1]
            signal_info.loc[latest_idx, 'signal_type'] = signal_dict['signal']
            signal_info.loc[latest_idx, 'reason'] = signal_dict['reason']
            signal_info.loc[latest_idx, 'probability'] = signal_dict['probability']
            signal_info.loc[latest_idx, 'entry_price'] = signal_dict['entry_price']
            signal_info.loc[latest_idx, 'stop_loss'] = signal_dict['stop_loss']
            signal_info.loc[latest_idx, 'target'] = signal_dict['target']
            signal_info.loc[latest_idx, 'strength'] = signal_dict['strength']
            signal_info.loc[latest_idx, 'sl_basis'] = signal_dict['sl_basis']
            signal_info.loc[latest_idx, 'tp_basis'] = signal_dict['tp_basis']
            signal_info.loc[latest_idx, 'risk_reward'] = signal_dict['risk_reward']
            
            # Add wall strength if available
            if signal_dict['signal'] == 'buy' and 'nearest_support' in signal_dict:
                for level in self.order_book_cache.get('support_levels', []):
                    if len(level) >= 3 and level[0] == signal_dict['nearest_support']:
                        signal_info.loc[latest_idx, 'wall_strength'] = level[2]
                        break
            elif signal_dict['signal'] == 'sell' and 'nearest_resistance' in signal_dict:
                for level in self.order_book_cache.get('resistance_levels', []):
                    if len(level) >= 3 and level[0] == signal_dict['nearest_resistance']:
                        signal_info.loc[latest_idx, 'wall_strength'] = level[2]
                        break
        
        return signal_info


class EnhancedOrderBookVolumeWrapper:
    """Wrapper class for compatibility with the original interface."""
    
    def __init__(self, **kwargs):
        # Pass all kwargs to the indicator
        self.indicator = EnhancedOrderBookVolumeIndicator(**kwargs)
        logger.info(f"Enhanced OrderBookVolumeWrapper initialized for {kwargs.get('symbol', 'GLOBAL')}")
    
    # Pass-through methods
    def compute_signals(self, open_, high, low, close, order_book=None, open_interest=None):
        return self.indicator.compute_signals(open_, high, low, close, order_book, open_interest)
    
    def update_order_book(self, bids, asks, timestamp=None):
        return self.indicator.update_order_book(bids, asks, timestamp)
    
    def update_open_interest(self, open_interest, timestamp=None, price=None):
        # Ignore open interest updates as we're not using it anymore
        return None
    
    def check_exit_conditions(self, current_price, entry_price, stop_loss, target, position_type,
                             trailing_activated=False, highest_reached=None, lowest_reached=None,
                             order_book=None, open_interest=None):
        # Just pass all parameters except open_interest
        return self.indicator.check_exit_conditions(
            current_price, entry_price, stop_loss, target, position_type,
            trailing_activated, highest_reached, lowest_reached,
            order_book
        )
    
    def update_pattern_trade_result(self, pattern_name, profit_pct, win=False, with_trend=None):
        return self.indicator.update_pattern_trade_result(pattern_name, profit_pct, win, with_trend)