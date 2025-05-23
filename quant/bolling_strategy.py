# 导入工具包
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from functools import wraps
import time
# import pandas_datareader as pdr
# 设置文字和画布大小以及文字样式
plt.style.use('seaborn-v0_8-ticks')
# 设置字体为微软字体（如微软雅黑）
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 设置字体大小
plt.rcParams['font.size'] = 12  # 可根据需要调整字体大小

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

class BollingerBandsStrategy:
    def __init__(self, symbol, lookback=20, std_dev=2):
        self.symbol = symbol # 股票代码
        self.lookback = lookback # 回看时间
        self.std_dev = std_dev # 标准差个数
        self.data = None
    
    @retry_on_failure(retries=3, delay=5)
    def fetch_data(self, start_date, end_date):
        # 获取股票数据
        try:
            # 设置备选的数据获取方法
            self.data = yf.download(self.symbol, start=start_date, end=end_date, progress=False, auto_adjust=True, interval='1d')
            if self.data.empty:
                raise Exception('No data download')
        except Exception as e:
            print(f"Error downloading data: {e}")
        # 多种列扁平化
        if len(self.data.columns[0]) > 1:
            self.data.columns = [col[0] for col in self.data.columns]
            print(f'已经进行列扁平化操作')
        else:
            print(f'无需进行列扁平化操作')
        return self.data
    
    def calculate_bollinger_bands(self):
        # 计算移动平均线
        self.data['SMA'] = self.data['Close'].rolling(window=self.lookback).mean()
        # 计算标准差
        self.data['STD'] = self.data['Close'].rolling(window=self.lookback).std()
        # 计算上下轨
        self.data['Upper'] = self.data['SMA'] + (self.std_dev * self.data['STD'])
        self.data['Lower'] = self.data['SMA'] - (self.std_dev * self.data['STD'])
    
    def generate_signals(self):
        # 生成交易信号
        self.data['Signal'] = 0
        self.data.loc[self.data['Close'] > self.data['Upper'], 'Signal'] = -1 # 卖出信号
        self.data.loc[self.data['Close'] < self.data['Lower'], 'Signal'] = 1 # 买入信号

    def plot_strategy(self):
        # 创建图表
        plt.figure(figsize=(15, 7))
        # 绘制收盘价和布林带
        plt.plot(self.data.index, self.data['Close'], label='Close price', color='black', alpha=0.8)
        plt.plot(self.data.index, self.data['SMA'], label='SMA', color='blue', linestyle='--', alpha=0.8)
        plt.plot(self.data.index, self.data['Upper'], label='Upper band', color='red', linestyle='-.', alpha=0.8)
        plt.plot(self.data.index, self.data['Lower'], label='Lower band', color='red', linestyle='-.', alpha=0.8)
        # 标记买入点和卖出点
        plt.plot(self.data[self.data['Signal'] == 1].index, self.data['Lower'][self.data['Signal'] == 1], '^', markersize=10, color='g', label='Buy signal')
        plt.plot(self.data[self.data['Signal'] == -1].index, self.data['Upper'][self.data['Signal'] == -1], 'v', markersize=10, color='r', label='Sell signal')
        plt.title(f'Bollinger Bands Strategy - {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

# 使用案例
if __name__ == '__main__':
    # 创建策略实例
    strategy = BollingerBandsStrategy('AAPL', lookback=20, std_dev=2)
    # 获取最近一年的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    # 获取数据并计算指标
    strategy.fetch_data(start_date, end_date)
    strategy.calculate_bollinger_bands()
    strategy.generate_signals()
    # 可视化结果
    strategy.plot_strategy()
    # 打印策略结果统计
    print('\n策略统计信息:')
    print(f"买入信号次数: {len(strategy.data[strategy.data['Signal'] == 1])}")
    print(f"卖出信号次数: {len(strategy.data[strategy.data['Signal'] == -1])}")
