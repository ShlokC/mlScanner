import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

class DynamicTrendClassifier:
    def __init__(self, timeframes):
        self.timeframes = timeframes
        self.trend_metrics = {tf: {
            'r2_history': [],
            'rmse_history': [],
            'trend_score_history': [],
            'dynamic_thresholds': {
                'r2_low': 0.5,
                'r2_high': 0.8,
                'rmse_multiplier': 1.0,
                'trend_score_threshold': 0.0
            }
        } for tf in timeframes}

    def update_dynamic_thresholds(self, tf, r2, rmse, trend_score):
        """
        Dynamically update trend classification thresholds based on historical performance
        """
        metrics = self.trend_metrics[tf]
        
        # Maintain sliding window of metrics (last 10 observations)
        metrics['r2_history'].append(r2)
        metrics['rmse_history'].append(rmse)
        metrics['trend_score_history'].append(trend_score)
        
        if len(metrics['r2_history']) > 10:
            metrics['r2_history'] = metrics['r2_history'][-10:]
            metrics['rmse_history'] = metrics['rmse_history'][-10:]
            metrics['trend_score_history'] = metrics['trend_score_history'][-10:]
        
        # Dynamically adjust thresholds
        dynamic_thresholds = metrics['dynamic_thresholds']
        
        # R2 thresholds
        dynamic_thresholds['r2_low'] = max(0.3, np.percentile(metrics['r2_history'], 25))
        dynamic_thresholds['r2_high'] = min(0.9, np.percentile(metrics['r2_history'], 75))
        
        # RMSE threshold (relative to mean of closes)
        dynamic_thresholds['rmse_multiplier'] = np.percentile(metrics['rmse_history'], 50)
        
        # Trend score threshold
        dynamic_thresholds['trend_score_threshold'] = np.percentile(metrics['trend_score_history'], 50)
        
        return dynamic_thresholds

    def classify_trend(self, closes, tf):
        """
        Classify market trend using sophisticated, adaptive criteria
        """
        # Ensure sufficient data
        if len(closes) < 20:
            return 'unknown'
        
        # Prepare features and model
        X, y = self._prepare_features(closes)
        
        # Optimize XGBRegressor (assuming the previous optimization method)
        xgb_params = self._optimize_params(closes, tf)
        model = XGBRegressor(**xgb_params)
        
        # Fit and predict
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Calculate trend score
        predicted_changes = np.diff(y_pred)
        trend_score = np.mean(predicted_changes > 0) - np.mean(predicted_changes < 0)
        
        # Update and get dynamic thresholds
        dynamic_thresholds = self.update_dynamic_thresholds(tf, r2, rmse, trend_score)
        
        # Comprehensive trend classification
        trend_classification = self._classify_trend_with_dynamics(
            r2, 
            rmse, 
            trend_score, 
            np.mean(closes), 
            dynamic_thresholds
        )
        
        return trend_classification

    def _classify_trend_with_dynamics(self, r2, rmse, trend_score, mean_close, dynamic_thresholds):
        """
        Advanced trend classification with multiple dynamic criteria
        """
        # Unpack dynamic thresholds
        r2_low = dynamic_thresholds['r2_low']
        r2_high = dynamic_thresholds['r2_high']
        rmse_multiplier = dynamic_thresholds['rmse_multiplier']
        trend_score_threshold = dynamic_thresholds['trend_score_threshold']
        
        # Comprehensive trend assessment
        trend_assessment = {
            'r2_quality': self._assess_r2_quality(r2, r2_low, r2_high),
            'rmse_stability': self._assess_rmse_stability(rmse, mean_close, rmse_multiplier),
            'trend_momentum': self._assess_trend_momentum(trend_score, trend_score_threshold)
        }
        
        # Voting mechanism for trend classification
        trend_votes = {
            'uptrend': sum(1 for v in trend_assessment.values() if v > 0),
            'downtrend': sum(1 for v in trend_assessment.values() if v < 0),
            'ranging': sum(1 for v in trend_assessment.values() if v == 0)
        }
        
        # Determine final trend based on votes
        if trend_votes['uptrend'] >= 2:
            return 'uptrend'
        elif trend_votes['downtrend'] >= 2:
            return 'downtrend'
        else:
            return 'ranging'

    def _assess_r2_quality(self, r2, r2_low, r2_high):
        """
        Assess R2 quality:
        - High R2: Strong predictive power (positive)
        - Low R2: Weak predictive power (negative)
        - Mid-range R2: Neutral
        """
        if r2 >= r2_high:
            return 1  # Strong trend indication
        elif r2 <= r2_low:
            return -1  # Weak trend indication
        else:
            return 0  # Neutral

    def _assess_rmse_stability(self, rmse, mean_close, rmse_multiplier):
        """
        Assess RMSE stability:
        - Low RMSE relative to mean close: Stable trend (positive)
        - High RMSE: Unstable trend (negative)
        """
        normalized_rmse = rmse / mean_close
        if normalized_rmse < 0.02 * rmse_multiplier:
            return 1  # Very stable, strong trend
        elif normalized_rmse > 0.1 * rmse_multiplier:
            return -1  # Unstable, weak trend
        else:
            return 0  # Moderate stability

    def _assess_trend_momentum(self, trend_score, trend_score_threshold):
        """
        Assess trend momentum:
        - Positive trend score: Upward momentum
        - Negative trend score: Downward momentum
        - Near-zero score: Ranging
        """
        if trend_score > trend_score_threshold:
            return 1  # Upward momentum
        elif trend_score < -trend_score_threshold:
            return -1  # Downward momentum
        else:
            return 0  # Neutral momentum

    def get_market_state(self, tf):
        """
        Wrapper method to get market state using the dynamic trend classifier
        """
        history = list(self.history[tf])
        if len(history) < self.min_history_length:
            return 'unknown'

        closes = np.array([c[3] for c in history])
        return self.classify_trend(closes, tf)

    def get_overall_market_state(self):
        """
        Aggregate market state across timeframes
        """
        timeframe_states = {}
        for tf in self.timeframes:
            timeframe_states[tf] = self.get_market_state(tf)
        
        # Voting mechanism for overall market state
        state_votes = {
            'uptrend': sum(1 for state in timeframe_states.values() if state == 'uptrend'),
            'downtrend': sum(1 for state in timeframe_states.values() if state == 'downtrend'),
            'ranging': sum(1 for state in timeframe_states.values() if state == 'ranging')
        }
        
        # Determine overall market state
        if state_votes['uptrend'] > max(state_votes['downtrend'], state_votes['ranging']):
            return 'uptrend'
        elif state_votes['downtrend'] > max(state_votes['uptrend'], state_votes['ranging']):
            return 'downtrend'
        else:
            return 'ranging'