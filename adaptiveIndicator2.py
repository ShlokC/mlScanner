import numpy as np
from collections import deque
import logging
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AdaptiveIndicator:
    def __init__(self, market_id, timeframes, lookback=100, min_history_length=20, historical_candles=None):
        self.market_id = market_id
        self.timeframes = timeframes
        self.initial_lookback = lookback
        self.min_history_length = min_history_length
        self.history = {tf: deque(maxlen=1000) for tf in timeframes}
        self.optimal_params = {tf: {'lookback': lookback, 'alpha': 1.0} for tf in timeframes}

        if historical_candles:
            for tf in timeframes:
                for candle in historical_candles.get(tf, []):
                    self.store_candle(tf, candle)

    def _optimize_params(self, closes, tf):
        """Optimize XGBRegressor hyperparameters using Hyperopt."""
        if len(closes) < self.min_history_length:
            logger.warning(f"{tf}: Insufficient data (len={len(closes)} < {self.min_history_length})")
            return self._get_default_xgb_params()

        # Define the hyperparameter search space
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1),
            'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0))
        }

        def objective(params):
            # Convert to integer for some parameters
            params = {
                'n_estimators': int(params['n_estimators']),
                'max_depth': int(params['max_depth']),
                'min_child_weight': int(params['min_child_weight']),
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'gamma': params['gamma']
            }

            # Prepare features
            X, y = self._prepare_features(closes)

            # Ensure enough data for time series cross-validation
            if len(X) < 10:
                return {'loss': float('inf'), 'status': STATUS_OK}

            # Time Series Cross-Validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            cv_rmse = []

            for train_idx, test_idx in tscv.split(X):
                if len(test_idx) < 2:
                    continue

                # Create and train XGBRegressor with current hyperparameters
                model = XGBRegressor(
                    **params,
                    random_state=42
                )
                model.fit(X[train_idx], y[train_idx])
                
                # Predict and evaluate
                y_pred = model.predict(X[test_idx])
                r2 = r2_score(y[test_idx], y_pred)
                rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
                
                cv_scores.append(r2)
                cv_rmse.append(rmse)

            if not cv_scores:
                return {'loss': float('inf'), 'status': STATUS_OK}

            # Combined loss metric
            mean_r2 = np.mean(cv_scores)
            mean_rmse = np.mean(cv_rmse) / np.mean(y)  # Normalized RMSE
            loss = -mean_r2 + mean_rmse  # Lower is better
            
            return {'loss': loss, 'status': STATUS_OK}

        # Run Hyperopt optimization
        trials = Trials()
        best = fmin(
            fn=objective, 
            space=space, 
            algo=tpe.suggest, 
            max_evals=30,  # Increased number of evaluations
            trials=trials
        )

        # Extract best parameters
        best_params = space_eval(space, best)
        
        # Convert to integer parameters
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])

        logger.info(f"{tf}: Optimized XGBRegressor Params: {best_params}")
        
        return best_params
    def _get_default_xgb_params(self):
        """Return default XGBRegressor parameters when optimization is not possible."""
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42
        }

    def _prepare_features(self, closes):
        df = pd.DataFrame({'close': closes})
        for lag in range(1, 4):
            df[f'lag_{lag}'] = df['close'].shift(lag)
        df['ma_5'] = df['close'].rolling(5).mean()  # Add moving average feature
        df.dropna(inplace=True)
        X = df.drop('close', axis=1).values
        y = df['close'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    def _calculate_momentum(self, closes):
        """Calculate momentum using Rate of Change and RSI"""
        roc = (closes[-1] / closes[-3] - 1) * 100  # 2-period ROC (since closes[-1] vs closes[-3] is 2 bars)
        
        # RSI calculation
        delta = pd.Series(closes).diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = -delta.where(delta < 0, 0).rolling(7).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        return roc, rsi
    def _calculate_trend(self, closes, tf):
        """Modified trend calculation to use optimized XGBRegressor parameters."""
        if len(closes) < self.min_history_length:
            return 0, 0, np.inf, 0

        # Get optimized or default parameters
        xgb_params = self._optimize_params(closes, tf)
        
        # Truncate closes based on previous optimization method
        closes = closes[-xgb_params.get('lookback', self.initial_lookback):]

        # Prepare features
        X, y = self._prepare_features(closes)
        
        # Create XGBRegressor with optimized parameters
        model = XGBRegressor(**xgb_params)
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate evaluation metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate normalized slope and trend score
        time = np.arange(len(y_pred))
        slope, _ = np.polyfit(time, y_pred, 1)
        slope_normalized = slope / np.mean(closes)
        
        # Calculate trend_score based on predicted price changes
        predicted_changes = np.diff(y_pred)
        trend_score = np.mean(predicted_changes > 0) - np.mean(predicted_changes < 0)

        return r2, rmse, slope_normalized, trend_score

    def _calculate_volatility(self, highs, lows):
        true_ranges = highs - lows
        return np.mean(true_ranges)
    def get_market_exit_state(self, tf):
        """Determine market state with enhanced ranging detection."""
        history = list(self.history[tf])
        if len(history) < self.min_history_length:
            return 'unknown'

        closes = np.array([c[3] for c in history[-self.optimal_params[tf]['lookback']:]])
        highs = np.array([c[1] for c in history[-self.optimal_params[tf]['lookback']:]])
        lows = np.array([c[2] for c in history[-self.optimal_params[tf]['lookback']:]])
        r2, rmse, slope, trend_score = self._calculate_trend(closes, tf)
        support, resistance, valid_swings = self.find_swing_points(tf, history)

        # Calculate swing trend score
        swing_highs = sorted([s for s in valid_swings if s[0] == 'resistance'], key=lambda x: x[2])
        swing_lows = sorted([s for s in valid_swings if s[0] == 'support'], key=lambda x: x[2])

        high_diffs = [swing_highs[i][1] - swing_highs[i-1][1] for i in range(1, len(swing_highs))]
        low_diffs = [swing_lows[i][1] - swing_lows[i-1][1] for i in range(1, len(swing_lows))]

        if len(high_diffs) > 0:
            high_trend = (sum(1 for d in high_diffs if d > 0) - sum(1 for d in high_diffs if d < 0)) / len(high_diffs)
        else:
            high_trend = 0

        if len(low_diffs) > 0:
            low_trend = (sum(1 for d in low_diffs if d > 0) - sum(1 for d in low_diffs if d < 0)) / len(low_diffs)
        else:
            low_trend = 0

        swing_trend_score = (high_trend + low_trend) / 2

        # Enhanced decision logic
        slope_threshold = 0.005  # Keep as baseline, adjustable via testing
        atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])  # 14-period ATR
        current_close = closes[-1]
        in_range = (current_close > support - 0.5 * atr) and (current_close < resistance + 0.5 * atr)

        # Conditions for ranging market
        # if len(swing_highs) < 3 or len(swing_lows) < 3:  # Not enough swing points
        #     return 'ranging'
        if r2 < 0.8 or rmse > np.mean(closes) * 0.05:  # Poor model fit
            return 'ranging'
        elif trend_score > 0.2:  # Threshold for uptrend
            return 'uptrend'
        elif trend_score < -0.2:  # Threshold for downtrend
            return 'downtrend'
        # if abs(swing_trend_score) < 0.5 or abs(slope) < slope_threshold or in_range:
        #     return 'ranging'
        # elif swing_trend_score > 0.5 and slope > 0:
        #     return 'uptrend'
        # elif swing_trend_score < -0.5 and slope < 0:
        #     return 'downtrend'
        else:
            return 'ranging'
    def get_market_state(self, tf):
        """Determine market state with enhanced ranging detection."""
        history = list(self.history[tf])
        if len(history) < self.min_history_length:
            return 'unknown'

        closes = np.array([c[3] for c in history[-self.optimal_params[tf]['lookback']:]])
        highs = np.array([c[1] for c in history[-self.optimal_params[tf]['lookback']:]])
        lows = np.array([c[2] for c in history[-self.optimal_params[tf]['lookback']:]])
        r2, rmse, slope, trend_score = self._calculate_trend(closes, tf)
        support, resistance, valid_swings = self.find_swing_points(tf, history)

        # Calculate swing trend score
        swing_highs = sorted([s for s in valid_swings if s[0] == 'resistance'], key=lambda x: x[2])
        swing_lows = sorted([s for s in valid_swings if s[0] == 'support'], key=lambda x: x[2])

        high_diffs = [swing_highs[i][1] - swing_highs[i-1][1] for i in range(1, len(swing_highs))]
        low_diffs = [swing_lows[i][1] - swing_lows[i-1][1] for i in range(1, len(swing_lows))]

        if len(high_diffs) > 0:
            high_trend = (sum(1 for d in high_diffs if d > 0) - sum(1 for d in high_diffs if d < 0)) / len(high_diffs)
        else:
            high_trend = 0

        if len(low_diffs) > 0:
            low_trend = (sum(1 for d in low_diffs if d > 0) - sum(1 for d in low_diffs if d < 0)) / len(low_diffs)
        else:
            low_trend = 0

        swing_trend_score = (high_trend + low_trend) / 2

        # Enhanced decision logic
        slope_threshold = 0.005  # Keep as baseline, adjustable via testing
        atr = np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])  # 14-period ATR
        current_close = closes[-1]
        in_range = (current_close > support - 0.5 * atr) and (current_close < resistance + 0.5 * atr)

        # Conditions for ranging market
        
        if r2 < 0.8 or rmse > np.mean(closes) * 0.05:  # Poor model fit
            return 'ranging'
        elif trend_score > 0.2:  # Threshold for uptrend
            return 'uptrend'
        elif trend_score < -0.2:  # Threshold for downtrend
            return 'downtrend'
        if len(swing_highs) < 3 or len(swing_lows) < 3:  # Not enough swing points
            return 'ranging'
        if abs(swing_trend_score) < 0.5 or abs(slope) < slope_threshold or in_range:
            return 'ranging'
        elif swing_trend_score > 0.5 and slope > 0:
            return 'uptrend'
        elif swing_trend_score < -0.5 and slope < 0:
            return 'downtrend'
        else:
            return 'ranging'

    def find_swing_points(self, tf, history, window=5):
        """Identify swing highs and lows and return them along with support/resistance."""
        volatility = np.std([c[1] - c[2] for c in history[-30:]])  # 30-period volatility
        window = int(3 + volatility * 1.5)  # Smaller window for higher volatility
        window = max(3, min(window, 10))  # Cap between 3-10
        highs = np.array([c[1] for c in history])
        lows = np.array([c[2] for c in history])
        
        # Find local maxima/minima using rolling window
        local_max = (highs == pd.Series(highs).rolling(window, center=True).max()).values
        local_min = (lows == pd.Series(lows).rolling(window, center=True).min()).values
        
        valid_swings = []
        for i in range(len(history)):
            if local_max[i]:
                valid = True
                for j in range(max(0, i-3), i):
                    if highs[j] > highs[i]:
                        valid = False
                        break
                if valid:
                    valid_swings.append(('resistance', highs[i], i))
            elif local_min[i]:
                valid = True
                for j in range(max(0, i-3), i):
                    if lows[j] < lows[i]:
                        valid = False
                        break
                if valid:
                    valid_swings.append(('support', lows[i], i))
        
        # Get most recent meaningful levels
        recent_swings = [s for s in valid_swings if s[2] > len(history) - 10]
        if not recent_swings:
            current_support = np.min(lows[-10:])
            current_resistance = np.max(highs[-10:])
        else:
            supports = [s[1] for s in recent_swings if s[0] == 'support']
            resistances = [s[1] for s in recent_swings if s[0] == 'resistance']
            current_support = np.mean(supports[-2:]) if supports else np.min(lows[-10:])
            current_resistance = np.mean(resistances[-2:]) if resistances else np.max(highs[-10:])
        
        return current_support, current_resistance, valid_swings
    def get_overall_exit_signal(self):
        """
        Generates an overall exit signal based purely on market state across timeframes.
        Returns:
        - signal: 1 (exit long), -1 (exit short), 0 (no exit)
        - reasoning: List of market state reasonings per timeframe
        """
        market_states = {}
        reasonings = []
        
        # Get market state for each timeframe
        for tf in self.timeframes:
            if len(self.history[tf]) >= self.min_history_length:
                state = self.get_market_exit_state(tf)
                market_states[tf] = state
                reasonings.append(f"{tf}: Market state - {state}")
            else:
                reasonings.append(f"{tf}: Insufficient data")
                market_states[tf] = 'unknown'

        if not market_states:
            return 0, ["No sufficient data across timeframes"]

        # Count market states across timeframes
        uptrend_count = sum(1 for state in market_states.values() if state == 'uptrend')
        downtrend_count = sum(1 for state in market_states.values() if state == 'downtrend')
        ranging_count = sum(1 for state in market_states.values() if state == 'ranging')
        total_valid = uptrend_count + downtrend_count + ranging_count

        if total_valid == 0:
            return 0, reasonings

        # Exit signal logic based purely on market state consensus
        signal = 0
        overall_reasoning = "Overall exit signal: "

        # Require majority consensus (at least half of valid timeframes)
        threshold = total_valid / 2

        if uptrend_count > downtrend_count and uptrend_count >= threshold:
            # Strong uptrend might signal exit for shorts
            signal = 1
            overall_reasoning += "Strong uptrend consensus"
        elif downtrend_count > uptrend_count and downtrend_count >= threshold:
            # Strong downtrend might signal exit for longs
            signal = -1
            overall_reasoning += "Strong downtrend consensus"
        elif ranging_count >= threshold:
            # Ranging market suggests holding position
            signal = 0
            overall_reasoning += "Hold - Ranging market consensus"
        else:
            # No clear consensus
            signal = 0
            overall_reasoning += "Hold - No clear market state consensus"

        reasonings.append(overall_reasoning)
        logger.info(overall_reasoning)
        return signal, reasonings

    def generate_signal(self, tf):
        history = list(self.history[tf])
        if len(history) < self.min_history_length:
            return 0, "Insufficient data"

        market_state = self.get_market_state(tf)
        support, resistance = self.find_swing_points(tf, history)[:2]  # Only need support/resistance
        last_close = history[-1][3]
        prev_close = history[-2][3]
        
        closes = [c[3] for c in history]
        roc, rsi = self._calculate_momentum(closes)
        
        signal = 0
        reasoning = f"{tf}: "
        confirmation_threshold = 0.005 * (resistance - support)

        if market_state == 'uptrend':
            if last_close > resistance + confirmation_threshold and roc > 2:
                reasoning += f"Strong breakout (Close {last_close} > Resistance {resistance:.4f} with ROC {roc:.1f}%"
                signal = 1
            elif last_close < support - confirmation_threshold and rsi < 40:
                reasoning += f"Reversal signal (Close {last_close} < Support {support:.4f} with RSI {rsi:.1f})"
                signal = -1
            elif prev_close < resistance and last_close > resistance:
                reasoning += f"Initial breakout (Close {last_close} > Resistance {resistance:.4f})"
                signal = 0.5
            else:
                reasoning += "Consolidating in uptrend"
        elif market_state == 'downtrend':
            if last_close < support - confirmation_threshold and roc < -2:
                reasoning += f"Strong breakdown (Close {last_close} < Support {support:.4f} with ROC {roc:.1f}%"
                signal = -1
            elif last_close > resistance + confirmation_threshold and rsi > 60:
                reasoning += f"Reversal signal (Close {last_close} > Resistance {resistance:.4f} with RSI {rsi:.1f})"
                signal = 1
            elif prev_close > support and last_close < support:
                reasoning += f"Initial breakdown (Close {last_close} < Support {support:.4f})"
                signal = -0.5
            else:
                reasoning += "Consolidating in downtrend"
        else:  # Ranging market
            atr = np.mean([c[1] - c[2] for c in history[-14:]])
            if last_close > resistance + atr * 0.5:
                reasoning += f"Range breakout with momentum (RSI {rsi:.1f})"
                signal = 1
            elif last_close < support - atr * 0.5:
                reasoning += f"Range breakdown with momentum (RSI {rsi:.1f})"
                signal = -1
            elif rsi > 70 and last_close > resistance:
                reasoning += f"Overbought at resistance (RSI {rsi:.1f})"
                signal = -1
            elif rsi < 30 and last_close < support:
                reasoning += f"Oversold at support (RSI {rsi:.1f})"
                signal = 1
            else:
                reasoning += f"Range: {support:.4f}-{resistance:.4f} (RSI {rsi:.1f})"

        reasoning += f" | ROC: {roc:.1f}%, RSI: {rsi:.1f}"
        return signal, reasoning

    def store_candle(self, tf, candle):
        self.history[tf].append(candle)

    def get_overall_signal(self):
        signals = []
        reasonings = []

        for tf in self.timeframes:
            if len(self.history[tf]) >= self.min_history_length:
                signal, reasoning = self.generate_signal(tf)
                signals.append(signal)
                reasonings.append(reasoning)
                #logger.info(reasoning)

        if not signals:
            return 0, "No signals available"

        buy_signals = signals.count(1)
        sell_signals = signals.count(-1)

        if buy_signals > sell_signals and buy_signals >= len(signals) // 2:
            overall_signal = 1
            overall_reasoning = "Overall signal: Buy"
        elif sell_signals > buy_signals and sell_signals >= len(signals) // 2:
            overall_signal = -1
            overall_reasoning = "Overall signal: Sell"
        else:
            overall_signal = 0
            overall_reasoning = "Overall signal: Hold"

        #logger.info(overall_reasoning)
        return overall_signal, reasonings

    def get_market_trend_summary(self):
        overall_signal, reasonings = self.get_overall_signal()

        if overall_signal == 1:
            trend = "Upward"
        elif overall_signal == -1:
            trend = "Downward"
        else:
            trend = "Sideways"

        summary = f"Overall Market Trend: {trend}\nReasonings:\n"
        for reasoning in reasonings:
            summary += f"- {reasoning}\n"
        return summary