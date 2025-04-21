# 导包
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import socket, socks, time
from functools import wraps

# 装饰器函数,用于重复执行失败的函数
def retry_on_failure(retries=3, delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:  # 最后一次重试
                        raise e
                    print(f"Attempt {i+1} failed, retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# 定义海龟策略的类
class TurtleStrategy:
    def __init__(self, symbol, entry_n=20, exit_n=10, atr_n=15, risk_ratio=0.1, units=4, position_size=0.1, 
                 max_loss_percent=0.05):
        """
        初始化海龟交易系统
        :param symbol: 交易品种代码
        :param entry_n: 入场周期(一般为20)
        :param exit_n: 退出周期(一般为10)
        :param atr_n: ATR计算周期
        :param risk_ratio: 风险系数
        :param units: 加仓次数
        :param position_size: 单次持仓比例
        :param max_loss_percent: 最大损失比例
        """
        self.symbol = symbol
        self.entry_n = entry_n
        self.exit_n = exit_n
        self.atr_n = atr_n
        self.risk_ratio = risk_ratio
        self.units = units
        self.position_size = position_size
        self.max_loss_percent = max_loss_percent
        self.data = None
        self.positions = [] # 当前持仓列表
        self.capital = 100000.0 # 初始资金
        self.equity = self.capital # 当前权益
        self.trades = [] # 交易记录
    
    @retry_on_failure(retries=3, delay=5)
    def fetch_data(self, start_date, end_date):
        """获取股票数据"""
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
        socket.socket = socks.socksocket
        try:
            self.data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)
            if self.data.empty:
                raise Exception("No data download")
            # 列扁平化
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = [col[0] if isinstance(col, tuple) else col for col in self.data.columns]
                required_columns = ["Open", "High", "Low", "Close", "Volume"]
                missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise Exception(f"Missing required columns: {missing_columns}")
            # 去除NaN的无效数据
            self.data.dropna(inplace=True)
            print(f"Successfully download {len(self.data)} records")
            return self.data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
        
    def calculate_indicators(self):
        # 计算市场波动幅度ATR
        high_low = self.data['High'] - self.data['Low']
        high_close = abs(self.data['High'] - self.data['Close'].shift(1))
        low_close = abs(self.data['Low'] - self.data['Close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1) # 按照列进行合并
        true_range = ranges.max(axis=1) # 取出每列中的最大值
        self.data['ATR'] = true_range.rolling(window=self.atr_n).mean()
        # 计算入场通道
        self.data['Highest_High'] = self.data['High'].rolling(window=self.entry_n).max()
        self.data['Lowest_Low'] = self.data['Low'].rolling(window=self.entry_n).min()
        # 计算出场通道1
        self.data['Exit_High'] = self.data['High'].rolling(window=self.exit_n).max()
        self.data['Exit_Low'] = self.data['Low'].rolling(window=self.exit_n).min()
    
    def calculate_position_size(self, price, atr):
        dollar_volatility = atr * price
        position_risk = self.capital * self.risk_ratio
        unit_size = position_risk / (2 * atr)
        max_units = self.capital * self.position_size / price
        # print(f"Unit Size: {unit_size}, Max Units: {max_units}")
        return min(int(unit_size), int(max_units))
    
    def add_position(self, date, price, direction, atr):
        """加仓操作"""
        if len(self.positions) >= self.units:
            return False
        size = self.calculate_position_size(price, atr)
        stop_price = price - 2 * atr if direction == 'long' else price + 2 * atr
        position = {
            'date': date, 
            'direction': direction, 
            'size': size, 
            'entry_price': price, 
            'stop_price': stop_price, 
            'atr': atr
        }
        total_risk = self.calculate_total_risk(position)
        if total_risk > self.max_loss_percent * self.capital:
            return False
        self.positions.append(position)
        return True
    
    def calculate_total_risk(self, new_position=None):
        """计算仓位总风险"""
        total_risk = 0
        for pos in self.positions:
            risk = pos['size'] * abs(pos['entry_price'] - pos['stop_price'])
            total_risk += risk
        if new_position:
            new_risk = new_position['size'] * abs(new_position['entry_price'] - new_position['stop_price'])
            total_risk += new_risk
        return total_risk
    
    def update_stops(self, current_price, current_atr):
        for position in self.positions:
            if position['direction'] == 'long':
                new_stop = current_price - 2 * current_atr
                position['stop_price'] = max(position['stop_price'], new_stop)
            else:
                new_stop = current_price + 2 * current_atr
                position['stop_price'] = min(position['stop_price'], new_stop)
    
    def close_position(self, date, price, position, reason):
        """清仓操作"""
        # print(f"Buy Price: {position['entry_price']}, Sell Price: {price}")
        pnl = (price - position['entry_price']) * position['size']
        # print(f"P&L: {pnl}, Equity after trade: {self.equity}")
        if position['direction'] == 'short':
            pnl = -pnl
        trade = {
            'entry_date': position['date'], 
            'exit_date': date, 
            'direction': position['direction'], 
            'entry_price': position['entry_price'], 
            'exit_price': price, 
            'size': position['size'], 
            'pnl': pnl, 
            'reason': reason
        }
        self.trades.append(trade)
        self.equity += pnl
        return pnl
    
    def genarate_signals(self):
        """生成第一次加仓信号"""
        self.calculate_indicators()
        self.data['Signal'] = 0
        self.data['Position'] = 0
        self.data['Equity'] = self.capital

        for i in range(len(self.data)):
            current_row = self.data.iloc[i]
            # 检查清仓
            for position in self.positions[:]:
                if position['direction'] == 'long':
                    if current_row['Low'] <= position['stop_price']:
                        pnl = self.close_position(current_row.name, position['stop_price'], position, 'stop_loss')
                        # print(f"Closed long position at {position['stop_price']} on {current_row.name}")
                        self.positions.remove(position)
                else:
                    if current_row['High'] >= position['stop_price']:
                        pnl = self.close_position(current_row.name, position['stop_price'], position, 'stop_loss')
                        # print(f"Closed long position at {position['stop_price']} on {current_row.name}")
                        self.positions.remove(position)
            # 出场信号
            if self.positions:
                if any(p['direction'] == 'long' for p in self.positions):
                    if current_row['Close'] < self.data['Exit_Low'].shift(1).iloc[i]:
                        for position in [p for p in self.positions if p['direction'] == 'long']:
                            pnl = self.close_position(current_row.name, current_row['Close'], position, 'exit_signal')
                            self.positions.remove(position)
                if any(p['direction'] == 'short' for p in self.positions):
                    if current_row['Close'] > self.data['Exit_High'].shift(1).iloc[i]:
                        for position in [p for p in self.positions if p['direction'] == 'short']:
                            pnl = self.close_position(current_row.name, current_row['Close'], position, 'exit_signal')
                            self.positions.remove(position)
            # 出场信号
            if len(self.positions) == 0:
                if current_row['Close'] > self.data['Highest_High'].shift(1).iloc[i]:
                    # print(f"Buy signal at {current_row['Close']} on {current_row.name}")
                    self.add_position(current_row.name, current_row['Close'], 'long', current_row['ATR'])
                    # print(f"Added position at {current_row['Close']} on {current_row.name}")
                    self.data.at[current_row.name, 'Signal'] = 1
                elif current_row['Close'] < self.data['Lowest_Low'].shift(1).iloc[i]:
                    # print(f"Sell signal at {current_row['Close']} on {current_row.name}")
                    self.add_position(current_row.name, current_row['Close'], 'short', current_row['ATR'])
                    # print(f"Added position at {current_row['Close']} on {current_row.name}")
                    self.data.at[current_row.name, 'Signal'] = -1
            if self.positions:
                last_position = self.positions[-1]
                if last_position['direction'] == 'long':
                    if (current_row['Close'] >= last_position['entry_price'] + 0.5 * current_row['ATR']):
                        self.add_position(current_row.name, current_row['Close'], 'long', current_row['ATR'])
                else:
                    if (current_row['Close'] <= last_position['entry_price'] - 0.5 * current_row['ATR']):
                        self.add_position(current_row.name, current_row['Close'], 'short', current_row['ATR'])
            # 更新清仓和记录持仓
            self.update_stops(current_row['Close'], current_row['ATR'])
            self.data.at[current_row.name, 'Position'] = sum([p['size'] if p['direction'] == 'long' else -p['size'] for p in self.positions])
            self.data.at[current_row.name, 'Equity'] = self.equity

    def plot_strategy(self):
        """策略执行可视化"""
        fig ,(ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])
        # 价格和加仓信号图
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', color='black', alpha=0.7)
        ax1.plot(self.data.index, self.data['Highest_High'], label='Entry Channel Upper', color='green', linestyle='--', alpha=0.5)
        ax1.plot(self.data.index, self.data['Lowest_Low'], label='Entry Channcel Lower', color='red', linestyle='--', alpha=0.5)
        ax1.plot(self.data[self.data['Signal'] == 1].index, self.data['Close'][self.data['Signal'] == 1], '^', markersize=10, color='g', label='Long Entry')
        ax1.plot(self.data[self.data['Signal'] == -1].index, self.data['Close'][self.data['Signal'] == -1], 'v', markersize=10, color='r', label='Short Entry')
        ax1.set_title(f"Turtle Trading Strategy - {self.symbol}")
        ax1.set_xlabel("date")
        ax1.set_ylabel("Price")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 仓位图
        ax2.fill_between(self.data.index, self.data['Position'], 0, where=self.data['Position']>=0, color='g', alpha=0.5, label='Long')
        ax2.fill_between(self.data.index, self.data['Position'], 0, where=self.data['Position']<=0, color='r', alpha=0.5, label='Short')
        ax2.set_title("Position Size")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 资金曲线图
        ax3.plot(self.data.index, self.data['Equity'], label='Equity', color='blue', alpha=0.8)
        ax3.set_title("Equity Curve")
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_performance(self):
        """输出策略表现"""
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) == 0:
            print("No trades executed")
            return
        
        print("\n==== Strategy Performance Statistics ====")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(trades_df[trades_df['pnl'] > 0])}")
        print(f"Losing Trades: {len(trades_df[trades_df['pnl'] < 0])}")
        print(f"Total P&L: ${trades_df['pnl'].sum():.2f}")
        print(f"Largest Win: ${trades_df['pnl'].max():.2f}")
        print(f"Largest Loss: ${trades_df['pnl'].min():.2f}")
        print(f"Win Rate: {len(trades_df[trades_df['pnl'] > 0]) / len(trades_df):.2%}")
        print(f"Final Equity: ${self.equity:.2f}")
        print(f"Return: {(self.equity - self.capital) / self.capital:.2%}")

def main():
    # 策略实例化
    tutle_strategy = TurtleStrategy('AAPL')
    # 设置时间
    end_date = datetime.now()
    start_date = end_date - timedelta(365)
    # 运行策略
    if tutle_strategy.fetch_data(start_date, end_date) is not None:
        tutle_strategy.genarate_signals()
        tutle_strategy.plot_strategy()
        tutle_strategy.print_performance()
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()