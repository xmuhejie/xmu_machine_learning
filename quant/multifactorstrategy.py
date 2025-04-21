# @author: Lhejie
# @version: 1.0
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import socket, socks, time
from functools import wraps

# 装饰器函数
def retry_on_failure(retries=3, delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise e
                    print(f"Attempt {i+1} failed, retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# 多因子策略的实现
class MultiFactorStrategy:
    def __init__(self, symbols, lookback_momentum=25, lookback_vol=20, lookback_size=20, position_size=0.1):
        """
        初始化多因子策略
        
        Args:
            symbols: 股票代码列表
            lookback_momentum: 动量因子回看期
            lookback_vol: 波动率因子回看期
            lookback_size: 市值因子回看期
            position_size: 单个股票持仓比例
        """
        self.symbols = symbols
        self.lookback_momentum = lookback_momentum
        self.lookback_vol = lookback_vol
        self.lookback_size = lookback_size
        self.position_size = position_size
        self.data = {} # 初始化多只股票数据
        self.factors = {} # 初始化因子
        self.capital = 100000.0 # 初始资金
        self.equity = self.capital
        self.positions = {}
        self.trades = []

    @retry_on_failure(retries=3, delay=5)
    def fetch_data(self, start_date, end_date):
        """获取多只股票数据"""
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
        socket.socket = socks.socksocket
        for symbol in self.symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)
                if data.empty:
                    raise Exception(f"No data downloading for {symbol}")
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                data.dropna(inplace=True)
                self.data[symbol] = data
                print(f"Successfully download {len(data)} records for {symbol}")
            except Exception as e:
                print(f"Error downloading data for {symbol}: {e}")
                return None
        return self.data
    
    def calculate_momentum_factor(self):
        """计算动量因子"""
        momentum = {}
        for symbol, data in self.data.items():
            returns = data['Close'].pct_change()
            momentum[symbol] = returns.rolling(window=self.lookback_momentum).mean()
        return pd.DataFrame(momentum)
    
    def calculate_volatility_factor(self):
        """计算波动率因子"""
        volatility = {}
        for symbol, data in self.data.items():
            returns = data['Close'].pct_change()
            volatility[symbol] = returns.rolling(window=self.lookback_vol).std()
        return pd.DataFrame(volatility)
    
    def calculate_size_factor(self):
        """计算市值因子"""
        size = {}
        for symbol, data in self.data.items():
            size[symbol] = (data['Close'] * data['Volume']).fillna(0)
        return pd.DataFrame(size)
    
    def normalize_factor(self, factor_df):
        """因子标准化"""
        return (factor_df - factor_df.mean()) / factor_df.std()
    
    def calculate_combined_score(self, weights=[0.4, 0.3, 0.3]):
        """计算因子综合得分"""
        momentum = self.normalize_factor(self.calculate_momentum_factor())
        volatility = -self.normalize_factor(self.calculate_volatility_factor()) # 波动率越小越好
        size = self.normalize_factor(self.calculate_size_factor())
        combined_score = (momentum * weights[0] + volatility * weights[1] + size * weights[2])
        return combined_score
    
    def generate_signals(self, top_n=10):
        """生成交易信号"""
        combined_score = self.calculate_combined_score()
        signals = pd.DataFrame(0, index=combined_score.index, columns=combined_score.columns)
        for date in combined_score.index:
            scores = combined_score.loc[date].dropna()
            if len(scores) > 0:
                top_stocks = scores.nlargest(top_n).index
                signals.loc[date, top_stocks] = 1
        return signals
    
    def backtest(self, top_n=10):
        """策略回测"""
        signals = self.generate_signals(top_n)
        portfolio_value = pd.Series(index=signals.index, dtype=float)
        portfolio_value.iloc[0] = self.capital
        # 记录持仓
        holdings = {}
        for i in range(1, len(signals)):
            current_date = signals.index[i]
            prev_date = signals.index[i-1]
            # 计算当日收益
            returns = {}
            position_value = 0
            valid_positions = 0
            # 遍历所有信号为1的股票
            for symbol in signals.columns[signals.loc[prev_date] == 1]:
                try:
                    # 确保两个日期数据都存在
                    if (current_date in self.data[symbol].index and prev_date in self.data[symbol].index):
                        # 计算收益率
                        prev_price = self.data[symbol]['Close'][prev_date]
                        curr_price = self.data[symbol]['Close'][current_date]
                        if prev_price > 0:
                            ret = (curr_price / prev_price) - 1
                            returns[symbol] = ret
                            valid_positions +=1
                            # 记录持仓
                            if symbol not in holdings:
                                holdings[symbol] = {
                                    'entry_date': prev_date, 
                                    'entry_price': prev_price
                                }
                except Exception as e:
                    print(f"Error calculating returns for {symbol} on {current_date}: {e}")
                    continue
            # 更新组合价值
            if valid_positions > 0:
                portfolio_return = np.mean(list(returns.values()))
                portfolio_value[current_date] = portfolio_value[prev_date] * (1 + portfolio_return)
            else:
                portfolio_value[current_date] = portfolio_value[prev_date]
            current_holdings = signals.columns[signals.loc[current_date] ==1]
            for symbol in list(holdings.keys()):
                if symbol not in current_holdings:
                    exit_price = self.data[symbol]['Close'][current_date]
                    entry_price = holdings[symbol]['entry_price']
                    hold_return = (exit_price / entry_price) - 1
                    self.trades.append({
                        'symbol': symbol, 
                        'entry_date': holdings[symbol]['entry_date'], 
                        'exit_date': current_date, 
                        'entry_price': entry_price, 
                        'exit_price': exit_price, 
                        'return': hold_return
                    })
                    del holdings[symbol]
        self.equity = portfolio_value.iloc[-1]                
        portfolio_value = portfolio_value.fillna(method='ffill')
        return portfolio_value

    def plot_strategy(self, portfolio_value):
        """可视化策略结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        # 绘制组合价值
        ax1.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', alpha=0.8, color='black')
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # 绘制收益率
        returns = portfolio_value.pct_change()
        ax2.plot(returns.index, returns.cumsum(), label='Cumulative Returns', alpha=0.8, color='black')
        ax2.set_title("Cumulative Returns")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Returns")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        plt.show()
    
    def print_performance(self, portfolio_value):
        """输出策略表现"""
        returns = portfolio_value.pct_change()
        total_return = (self.equity - self.capital) / self.capital
        annual_return = total_return / (len(portfolio_value) / 252)
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility
        max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()

        print("\n==== Strategy Performance ====")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annual Return: {annual_return:.2%}")
        print(f"Volatility: {volatility:.2}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Fianl Portfolio Value: {self.equity:.2f}")

def main():
    # 策略实例化
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    strategy = MultiFactorStrategy(symbols)
    # 设置回测时间
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    # 运行策略
    if strategy.fetch_data(start_date, end_date) is not None:
        portfolio_value = strategy.backtest(top_n=3)
        strategy.plot_strategy(portfolio_value)
        strategy.print_performance(portfolio_value)
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()