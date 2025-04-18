# 导入工具包
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from functools import wraps
import time
import socks
import socket
# 设置文字和画布大小以及文字样式
plt.style.use('seaborn-v0_8-ticks')
# 设置字体为微软字体（如微软雅黑）
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 设置字体大小
plt.rcParams['font.size'] = 12  # 可根据需要调整字体大小

class RSIStrategy:
    def __init__(self, symbol, period=14, overbought=70, oversold=30):
        """
        初始化RSI策略
        :param symbol: 股票代码
        :param period: RSI计算周期
        :param overbought: 超买阈值
        :param oversold: 超卖阈值
        """
        self.symbol = symbol
        self.period = period
        self.overbought = overbought # RSI超买值
        self.oversold = oversold # RSI超卖值
        self.data = None # 股票数据
    
    def fetch_data(self, start_date, end_date):
        """获取股票数据并进行预处理"""
        # 设置 SOCKS5 代理，替换默认 socket
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)  # 这里是代理地址和端口
        socket.socket = socks.socksocket  # 用 SOCKS5 替换原生的 socket
        try:
            self.data = yf.download(self.symbol, start=start_date, end=end_date, interval='1d', auto_adjust=True)
            if self.data.empty:
                raise Exception('No data download')
            # 检查并进行列名扁平化
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = [col[0] if isinstance(col, tuple) else col for col in self.data.columns]
                print(f'已经进行列扁平化操作')
            else:
                print(f'无需进行列扁平化操作')
            # 确保数据包含必要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise Exception(F'Missing required columns: {missing_columns}')
            # 删除任何包含NaN的行
            self.data = self.data.dropna()
            print(f'成功获取 {len(self.data)} 条数据记录')
            return self.data
        except Exception as e:
            print(f'Error downloading data: {e}')
            return None
        
    def calculate_rsi(self):
        """计算RSI指标"""
        if self.data is None or self.data.empty:
            return False
        # 计算价格变化
        delta = self.data['Close'].diff()
        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        # 计算相对强度
        rs = gain / loss
        # 计算RSI
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return True
    
    
