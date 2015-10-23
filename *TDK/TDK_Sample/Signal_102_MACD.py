# coding=utf-8
# Signal_102_KDJ.py
#

import numpy as np
import pandas as pd
import sklearn as sk
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *


# MACD 的signal在低位是建仓时段 高位是平仓阶段
# 本策略基于放量检测进行 
# 基本量 hist差值 
# 买入信号 hist差值拉平（小于预定阈值） 或者 hist差值转负
# 卖出信号 
# 通过KDJ指标来确定市场超买超卖状态
# KDJ指标可以通过STOCH&STOCKF来接受
# 当出现macd

class Signal_102_KDJ(Signal):

    required_data = ["ClosePrice","HighestPrice","LowestPrice","Volume"]

    def prebatch(self):

        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        Volume = self.eod.Volume
        # Raw inputs

        # 计算指数收益率
        indexrtn = np.empty(self.index.ClosePrice.shape[0])
        idx = np.where(self.index.ticker_names=='000300')[0][0]
        indexClose = self.index.ClosePrice[:,idx]
        indexrtn[1:] = np.log(indexClose[1:]/indexClose[:-1])
        # 计算个股收益率
        rtn = np.empty(Close.shape)
        rtn[1:,:] = np.log(Close[1:,:]/Close[:-1,:])
        #学习了！如何计算收益率和生成差分序列
        indexKAMA = ta.KAMA(indexClose,)

        MACD,MACDSIG,MACDHIST = self.function_wrapper("MACD", Close, fastperiod=12, slowperiod=26,signalperiod=9)
        vol_ma = self.function_wrapper('MA',Volume,timeperiod= 5)
        slowk, slowd = self.function_wrapper('STOCH',High, Low, Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        fastk, fastd = self.function_wrapper('STOCHF',High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        self.slowj = 3 * slowk - 2 * slowd
        self.fastj = 3 * fastk - 2 * fastd
        self.MACDHIST = MACDHIST
        self.MACD = MACD
        ###
        # slow - k d j & fast - k d j
        # 获取KDJ指标
        # RSV = 100 * (Close - ta.MIN(Close,timeperiod=9))/(ta.MAX(Close,timeperiod=9)-ta.MIN(Close,timeperiod=9))
        # 不需要直接计算RSV 通过STOCH & STOCHF & STOCHRSI 来进行
        ###

        short_period = 10
        long_period = 30
        long_ma = self.function_wrapper('MA',Close,timeperiod=long_period)
        short_ma = self.function_wrapper('MA',Close,timeperiod=short_period)
        long_kama = self.function_wrapper("KAMA", Close , timeperiod=long_period)
        self.kama = long_kama


        # 获取所有标的的时序
        # ndarray格式 (日期,股票) 0轴为日期


    def generate(self, di):
        
        r = []
        last_wgt = self.weights[-1]
        # 初始化权重列表和定义上一期的权重

        MACDHIST = self.MACDHIST
        MACD = self.MACD
        KAMA = self.kama
        KAMADIFF = KAMA[di-1:di,].T - KAMA[di-2:di-1,].T
        slowj = self.slowj
        observed = self.fastj[di-1:di,].T
        # 接受数据 格式为（天数，股数） 0轴为天数
        

        for ix, ticker in enumerate(observed):
            
            w = last_wgt[ix]
            
            if ticker <= 10 :
                w = abs(ticker)*200
            if ticker >= 90 :
                w = 0.
            if KAMADIFF[ix]>0:
                w = 0.
            
            r.append(w)
        
        res = np.array(r)
        return res


signal = Signal_102_KDJ