import ccxt
import os
import time
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('exchange')

class ExchangeClient:
    def __init__(self, exchange='binance', api_key=None, api_secret=None):
        self.exchange_id = exchange
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY')
        self.exchange = self._create_exchange()
    
    def _create_exchange(self):
        return ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'timeout': 30000
        })
    
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
    
    def fetch_active_symbols(self):
        """Fetch symbols with the highest gains or losses in the last 24 hours using 5-minute candles."""
        try:
            # Get market data
            markets = self.exchange.load_markets()
            
            # Select all USDT-settled futures markets, excluding BTC and ETH
            active_markets = [
                symbol for symbol, market in markets.items()
                if market.get('settle') == 'USDT' and market.get('swap') and 'BTC' not in symbol and 'ETH' not in symbol
            ]
            
            # Time frame for exactly 24 hours (using 5m candles)
            now = int(time.time() * 1000)
            day_ago = now - (24 * 60 * 60 * 1000)  # Exactly 24 hours ago
            
            logger.info(f"Analyzing {len(active_markets)} markets for 24h movement using 5m candles")
            
            # Lists to store gainers and losers
            gain_symbols = []
            loss_symbols = []
            
            for symbol in active_markets:
                try:
                    time.sleep(0.1)  # Rate limiting
                    
                    # Fetch exactly 24 hours of 5-minute candles (288 candles = 24 hours)
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='5m', since=day_ago, limit=288)
                    
                    if not ohlcv or len(ohlcv) < 200:  # Need sufficient data
                        continue
                    
                    # Get first and last candle for simple price comparison
                    first_candle = ohlcv[0]
                    last_candle = ohlcv[-1]
                    
                    # Use closing prices for accurate 24h comparison
                    first_close = first_candle[4]
                    last_close = last_candle[4]
                    
                    if first_close <= 0:  # Avoid division by zero
                        continue
                    
                    # Calculate 24h percentage change
                    pct_change = ((last_close - first_close) / first_close) * 100
                    
                    # Determine if it's a gain or loss
                    if pct_change > 0:
                        # It's a gain
                        gain_symbols.append({
                            'symbol': symbol,
                            'movement_type': "gain",
                            'movement_pct': pct_change
                        })
                    elif pct_change < 0:
                        # It's a loss (store absolute value)
                        loss_symbols.append({
                            'symbol': symbol,
                            'movement_type': "loss",
                            'movement_pct': abs(pct_change)
                        })
                    
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort gain symbols by highest percentage (descending)
            sorted_gain_symbols = sorted(gain_symbols, key=lambda x: x['movement_pct'], reverse=True)
            
            # Sort loss symbols by highest percentage (descending)
            sorted_loss_symbols = sorted(loss_symbols, key=lambda x: x['movement_pct'], reverse=True)
            
            # Take top 15 from each category (or fewer if not available)
            max_per_category = 15
            top_gains = sorted_gain_symbols[:max_per_category]
            top_losses = sorted_loss_symbols[:max_per_category]
            
            # Combine gainers and losers with gains first
            sorted_symbols = top_gains + top_losses
            
            # Log the results
            logger.info(f"Found {len(top_gains)} gainers and {len(top_losses)} losers")
            for i, item in enumerate(sorted_symbols[:10]):
                logger.info(
                    f"{i+1}. {item['symbol']}: {item['movement_type'].capitalize()}={item['movement_pct']:.1f}%"
                )
            
            # Return the symbols without the metadata
            result_symbols = [item['symbol'] for item in sorted_symbols[:30]]
            return result_symbols
            
        except Exception as e:
            logger.exception(f"Error finding suitable symbols: {e}")
            return []
            
    def get_balance(self):
        """Fetch account balance."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.exception(f"Error fetching balance: {e}")
            return {}
    
    def create_order(self, symbol, order_type, side, amount, price=None):
        """Create a new order."""
        try:
            return self.exchange.create_order(symbol, order_type, side, amount, price)
        except Exception as e:
            logger.exception(f"Error creating order: {e}")
            return None
    
    def cancel_order(self, order_id, symbol):
        """Cancel an existing order."""
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.exception(f"Error canceling order: {e}")
            return None
    
    def fetch_ticker(self, symbol):
        """Fetch current ticker for a symbol."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.exception(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def fetch_ohlcv(self, symbol, timeframe='5m', since=None, limit=None):
        """Fetch OHLCV data for a symbol."""
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            logger.exception(f"Error fetching OHLCV for {symbol}: {e}")
            return []
    
    def fetch_order_status(self, order_id, symbol):
        """Fetch the status of an order."""
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.exception(f"Error fetching order status for {order_id}: {e}")
            return None
            
    def fetch_binance_data(self, market_id, timeframe='5m', limit=100, include_current=True):
        """
        Fetch OHLCV data from Binance with proper timestamp handling.
        
        Args:
            market_id (str): The trading pair symbol
            timeframe (str): Timeframe for candles (e.g., '1m', '5m', '1h')
            limit (int): Maximum number of candles to return
            include_current (bool): Whether to include the current (potentially incomplete) candle
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Fetch data - increase the limit to ensure we have enough after processing
            actual_fetch_limit = limit * 2  # Double the requested limit to account for potential losses
            ohlcv = self.exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=actual_fetch_limit)
            
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
            
            # Identify current candle
            current_candle_timestamp = None
            if len(df) > 0:
                tf_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
                current_time_ms = int(time.time() * 1000)
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
            
            logger.debug(f"Fetched {len(df)} candles for {market_id}, current candle included: {include_current}")
            return df
            
        except Exception as e:
            logger.exception(f"Failed to fetch data for {market_id}: {e}")
            return pd.DataFrame()