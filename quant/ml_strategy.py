import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class MLStrategy:
    def __init__(self, start_date, end_date, n_random_features=20):
        self.start_date = start_date
        self.end_date = end_date
        self.n_random_features = n_random_features
        self.data = None
        self.model = None
        self.train_end = None
    
    def generate_mock_data(self):
        """生成模拟行情数据"""
        dates = pd.date_range(self.start_date, self.end_date)
        n = len(dates)
        # 生成基础价格数据
        close = np.random.randn(n).cumsum() + 100
        data = pd.DataFrame({
            'date': dates, 
            'close': close, 
            'open': close * (1 + np.random.randn(n) * 0.02), 
            'high': close * (1 + abs(np.random.randn(n)) * 0.02), 
            'low': close * (1 - abs(np.random.randn(n)) * 0.02), 
            'volume': np.random.randint(1000, 10000, n)
        })
        return data
    
    def generate_random_features(self, df):
        """生成随机特征"""
        for i in range(self.n_random_features):
            feature_type = random.choice(['ma', 'momentum', 'volatility'])
            if feature_type == 'ma':
                window = random.randint(5, 30)
                df[f"random_ma_{i}"] = df['close'].rolling(window=window).mean()
            elif feature_type == 'momentum':
                period = random.randint(1, 10)
                df[f"random_mom_{i}"] = df['close'].pct_change(period)
            else:
                window = random.randint(5, 20)
                df[f"random_vol_{i}"] = df['close'].rolling(window=window).std()
        return df
    
    def generate_label(sekf, df):
        """生成训练标签"""
        df['returns'] = df['close'].pct_change()
        df['label'] = np.where(df['returns'].shift(-1) > 0, 1, 0)
        return df
    
    def process_data(self):
        """数据处理主流程"""
        # 生成模拟数据
        df = self.generate_mock_data()
        # 生成技术指标特征
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        # 生成随机特征
        df = self.generate_random_features(df)
        # 生成标签
        df = self.generate_label(df)
        # 处理缺失值
        df.dropna(inplace=True)
        self.data = df
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_model(self):
        """训练模型"""
        if self.data is None:
            raise ValueError("Data nust be processed before training")
        feature_cols = [col for col in self.data.columns if col not in ['date', 'label', 'returns']]
        X = self.data[feature_cols]
        y = self.data['label']
        # 按时间顺序分割训练集和验证集
        train_end = self.data.index[int(len(self.data) * 0.8)]
        X_train, X_test = X[:train_end], X[train_end:]
        y_train, y_test = y[:train_end], y[train_end:]
        # 保存分割点供回测使用
        self.train_end = train_end
        # 初始化并训练模型
        model_params = {
            'learning_rate': 0.05, 
            'n_estimators': 100, 
            'max_depth': 5, 
            'random_state': 42, 
            'objective': 'binary', 
            'metric': ['auc', 'binary_logloss'], 
            'early_stopping_rounds': 10
        }
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        self.model = model
        # 评估模型
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        print("\nValidation Results:")
        print(f"Best Iteration: {model.best_iteration_}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        # 特征重要性分析
        importance = pd.DataFrame({
            'feature': feature_cols, 
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.show()
        return model


    def backtest(self):
        """策略回测"""
        if self.model is None:
            raise ValueError("Model must be trained before backtesting")
        feature_cols = [col for col in self.data.columns if col not in ['date', 'label', 'returns']]
        test_data = self.data[self.train_end:].copy()
        test_data['pred_proba'] = self.model.predict_proba(test_data[feature_cols])[:, 1]
        test_data['signal'] = np.where(test_data['pred_proba'] > 0.5, 1, 0)
        # Calculate strategy returns
        test_data['ml_returns'] = test_data['signal'].shift(1) * test_data['returns']
        # Calculate original buy-and-hold returns
        test_data['orig_returns'] = test_data['returns']

        def calculate_metrics(strategy_returns, strategy_name):
            """Calculate and display strategy performance metrics"""
            metrics = {}
            metrics['total_returns'] = (1 + strategy_returns).cumprod().iloc[-1] - 1
            metrics['annual_returns'] = metrics['total_returns'] / (len(strategy_returns) / 252)
            metrics['sharpe'] = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            metrics['max_drawdown'] = (((1 + strategy_returns).cumprod() / 
                                    (1 + strategy_returns).cumprod().cummax()) - 1).min()
            print(f"\n{strategy_name} Performance Metrics:")
            print(f"Total Returns: {metrics['total_returns']:.2%}")
            print(f"Annual Returns: {metrics['annual_returns']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            return metrics
         # 计算ML策略指标
        ml_metrics = calculate_metrics(test_data['ml_returns'].dropna(), 'ML Strategy')
        # 计算原始策略指标
        orig_metrics = calculate_metrics(test_data['orig_returns'].dropna(), 'Original Strategy')
        # 绘制收益曲线
        plt.figure(figsize=(15, 7))
        ((1 + test_data['ml_returns']).cumprod()).plot(label='ML Strategy', color='red', alpha=0.8)
        ((1 + test_data['orig_returns']).cumprod()).plot(label='Buy & Hold', color='green', alpha=0.8)
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return ml_metrics, orig_metrics

def main():
    # 初始化策略
    ml_strategy = MLStrategy(
        start_date='2020-01-01', 
        end_date='2023-12-31', 
        n_random_features=20
    )
    # 数据处理
    print("Precossing data...")
    df = ml_strategy.process_data()
    print(f"Data Shape: {df.shape}")
    # 训练模型
    print("\nTraining model...")
    model = ml_strategy.train_model()
    # 回测
    print("\nRunning backtest...")
    ml_metrics, orig_metrics = ml_strategy.backtest()
    # 输出策略对比结果
    print("\nStrategy Comparison:")
    print(f"ML Strategy Sharpe: {ml_metrics['sharpe']:.2f}")
    print(f"Buy & Hold Sharpe: {orig_metrics['sharpe']:.2f}")
    # 计算策略收益差异
    relative_improvement = (ml_metrics['total_returns'] - orig_metrics['total_returns']) / abs(orig_metrics['total_returns'])
    print(f"\nML Strategy outperformance: {relative_improvement:.2%}")

if __name__ == "__main__":
    main()