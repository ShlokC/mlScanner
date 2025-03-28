import numpy as np
import pandas as pd
from collections import deque
import logging
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import TimeSeriesSplit

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DynamicTrendClassifier:
    def __init__(self, timeframes, min_history_length=10):
        self.timeframes = timeframes
        self.min_history_length = min_history_length
        self.scalers = {}  # Store scalers per timeframe
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

    def _prepare_features(self, closes):
        """Create features and target variable with proper temporal alignment"""
        df = pd.DataFrame({'close': closes})
        
        # Lagged features
        for lag in range(1, 4):
            df[f'lag_{lag}'] = df['close'].shift(lag)
        
        # Moving average
        df['ma_5'] = df['close'].rolling(5).mean()
        
        # Volatility
        df['volatility_5'] = df['close'].rolling(5).std()
        
        # RSI (simplified)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target: next period's return
        df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        return df.drop(['close', 'target'], axis=1).values, df['target'].values

    def _train_evaluate_model(self, X_train, y_train, X_test, y_test, params):
        """Train and evaluate an ensemble of models with given parameters"""
        n_models = 3  # Number of models in the ensemble
        models = []
        y_preds = []
        
        for _ in range(n_models):
            model = XGBRegressor(
            **params,
            early_stopping_rounds=10,  # Now in constructor
            random_state=np.random.randint(0, 1000)
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        models.append(model)
        y_preds.append(model.predict(X_test))
        
        y_pred_avg = np.mean(y_preds, axis=0)
        
        # Calculate metrics on averaged predictions
        mse = mean_squared_error(y_test, y_pred_avg)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_avg)
        r2 = r2_score(y_test, y_pred_avg)
        directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred_avg))
        
        return {
            'models': models,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }

    def _optimize_params(self, X_train, y_train, tf):
        """Hyperparameter optimization using Hyperopt with time-series cross-validation"""
        space = {
            'max_depth': hp.quniform('max_depth', 3, 12, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.4, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1.0),
            'gamma': hp.uniform('gamma', 0, 1.0),
            'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(100)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-6), np.log(100)),
        }

        def objective(space_params):
            params = {
                'n_estimators': 1000,
                'max_depth': int(space_params['max_depth']),
                'learning_rate': space_params['learning_rate'],
                'subsample': space_params['subsample'],
                'colsample_bytree': space_params['colsample_bytree'],
                'gamma': space_params['gamma'],
                'min_child_weight': int(space_params['min_child_weight']),
                'reg_alpha': space_params['reg_alpha'],
                'reg_lambda': space_params['reg_lambda'],
            }

            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                if len(val_idx) < 2:
                    continue
                model = XGBRegressor(**params, random_state=42)
                model.fit(
                    X_train[train_idx], y_train[train_idx],
                    eval_set=[(X_train[val_idx], y_train[val_idx])],                    
                    verbose=False
                )
                y_pred = model.predict(X_train[val_idx])
                mse = mean_squared_error(y_train[val_idx], y_pred)
                cv_scores.append(mse)

            return {'loss': np.mean(cv_scores) if cv_scores else float('inf'),
                    'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
        best_params = space_eval(space, best)
        best_params.update({
            'n_estimators': 1000,
            'max_depth': int(best_params['max_depth']),
            'min_child_weight': int(best_params['min_child_weight'])
        })
        return best_params

    def update_dynamic_thresholds(self, tf, r2, rmse, trend_score):
        """Update dynamic thresholds with exponential moving average"""
        metrics = self.trend_metrics[tf]
        metrics['r2_history'].append(r2)
        metrics['rmse_history'].append(rmse)
        metrics['trend_score_history'].append(trend_score)
        
        ema_alpha = 0.2
        metrics['dynamic_thresholds'].update({
            'r2_low': self._ema(metrics['r2_history'], ema_alpha, 0.5),
            'r2_high': self._ema(metrics['r2_history'], ema_alpha, 0.8),
            'rmse_multiplier': self._ema(metrics['rmse_history'], ema_alpha, 1.0),
            'trend_score_threshold': self._ema(metrics['trend_score_history'], ema_alpha, 0.0)
        })
        return metrics['dynamic_thresholds']

    def _ema(self, values, alpha, initial):
        """Exponential Moving Average calculation"""
        if not values:
            return initial
        ema = initial
        for value in values:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def classify_trend(self, closes, tf):
        """Main trend classification method"""
        if len(closes) < self.min_history_length:
            return 'unknown'
        # Calculate adaptive threshold based on recent returns' volatility
        if len(closes) > 1:
            returns = np.diff(closes) / closes[:-1]
            window = min(10, len(returns))  # Use last 50 returns or available data
            std_returns = np.std(returns[-window:])
            threshold = 0.01 * std_returns  # Multiplier can be tuned (e.g., 0.1 to 1.0)
        else:
            threshold = 0.001  # Fallback default
        X, y = self._prepare_features(closes)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[tf] = scaler

        best_params = self._optimize_params(X_train_scaled, y_train, tf)
        results = self._train_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, best_params)
        
        # Train final ensemble on entire dataset
        X_full_scaled = self.scalers[tf].transform(X)
        final_models = []
        for _ in range(3):
            model = XGBRegressor(**best_params, random_state=np.random.randint(0, 1000))
            model.fit(X_full_scaled, y)
            final_models.append(model)
        
        y_full_preds = [model.predict(X_full_scaled) for model in final_models]
        y_full_pred_avg = np.mean(y_full_preds, axis=0)
        
        # Update dynamic thresholds with a trend score based on recent predictions
        trend_score = np.mean(y_full_pred_avg[-3:])  # Average return as trend score
        self.update_dynamic_thresholds(tf, results['r2'], results['rmse'], trend_score)
        
        return self._determine_trend(y_full_pred_avg[-3:], threshold=threshold)

    def _determine_trend(self, recent_predictions, threshold=0.001):
        """Determine trend based on recent predicted returns"""
        if len(recent_predictions) < 2:
            return 'unknown'
        avg_return = np.mean(recent_predictions)
        if avg_return > threshold:
            return 'uptrend'
        elif avg_return < -threshold:
            return 'downtrend'
        else:
            return 'ranging'

    def get_market_state(self, tf):
        """Wrapper method to get market state"""
        history = list(self.history[tf])  # Assumes history is defined elsewhere
        if len(history) < self.min_history_length:
            return 'unknown'
        closes = np.array([c[3] for c in history])  # Assumes OHLC format
        return self.classify_trend(closes, tf)

    def get_overall_market_state(self):
        """Aggregate market state across timeframes"""
        timeframe_states = {tf: self.get_market_state(tf) for tf in self.timeframes}
        state_votes = {
            'uptrend': sum(1 for state in timeframe_states.values() if state == 'uptrend'),
            'downtrend': sum(1 for state in timeframe_states.values() if state == 'downtrend'),
            'ranging': sum(1 for state in timeframe_states.values() if state == 'ranging')
        }
        if state_votes['uptrend'] > max(state_votes['downtrend'], state_votes['ranging']):
            return 'uptrend'
        elif state_votes['downtrend'] > max(state_votes['uptrend'], state_votes['ranging']):
            return 'downtrend'
        else:
            return 'ranging'