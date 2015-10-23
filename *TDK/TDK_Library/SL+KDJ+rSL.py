# coding=utf-8
# Signal_STOPL.py

import numpy as np
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_STOPL(Signal):

    def prebatch(self):

        # 引入指数
        index = self.index.ClosePrice
        idx = np.where(self.index.ticker_names=='000905')[0][0]
        index_close = index[:,idx]
        index_kama = ta.MA(index_close,timeperiod=60)
        
        # 引入因子
        F1 = self.factors.BP
        F1_msk = np.ma.array(F1,mask = np.isnan(F1))
        F1_z = st.zscore(F1_msk)
        
        
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        K,D = self.function_wrapper('STOCHF', High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        J = 3*K-2*D
        
        # Kdiff = np.empty(self.eod.ClosePrice.shape)
        # Kdiff[1:,:] = K[1:,:] - K[:-1,:]
        
        diff2 = np.empty(Close.shape)
        diff2[1:,:] = Close[1:,:] - Close[:-1,:]
        self.diff2 = diff2
        
        JD = np.empty(J.shape)
        JD[1:,:] = J[1:,:] - K[:-1,:]
        self.diff = index_close - index_kama
    
        self.close = Close
        self.K = K
        self.D = D
        self.J = J
        self.JD =JD
        self.names = self.eod.ticker_names
        self.F1 = F1
        self.s = F1_z
        

    def generate(self,di):

        r = []
        last_wgt = self.weights[-1]
        
        names = self.names
        diff = self.diff[di-2:di]
        K = self.K[di-1,:].T
        D = self.D[di-1,:].T
        J = self.J[di-1,:].T
        JD = self.JD[di-1,:].T
        diff2 = self.diff2[di-1,:].T
        close = self.close
        rtn2 = diff2/close[di-2,:].T
        
        # 接受factor
        F1 = self.F1[di-1,:].T
        s = self.s[di-1]
        
        # 计算score
        score = st.scoreatpercentile(F1[~np.isnan(F1)],10)
        # 比例要慎重 否则等于压制了之前策略的优势
        # 选股策略可以压换手率

        for ix, ticker in enumerate(F1):
            # 未引入factors之前使用names做枚举循环 
            w = last_wgt[ix]
            
            if diff[1] < 0 or rtn2[ix] < -0.097:
                # 这里要多试试 此处设置的是前一日跌超过9.5%止损
                # 可以参考的思路是前两天累计跌多少才止损 那样可以抓点反弹
                # 9.5% - 1.22 & 2.63
                # 9.8% - 1.23 & 2.62
                # 9.6% - 1.23 & 2.64
                # 9.7% - 1.24 & 2.64
                w = 0
            elif J[ix] < 10:
                w = 100.-J[ix]
                # w = 100.-J[ix] -1.17-2.57
                # w = 50.-J[ix] -1.15-2.56
                # w = 10.-J[ix] -0.93-2.34
                # w = s[ix]*(10.-J[ix]) - 0.64-2.53
                # w = s[ix]*(100.-J[ix])/80 - 0.8-2.24
                # 1.5 乘数用于降低权重差 BP不加入权且去尾10% 1.2-2.6
                # 不用BP 乘数1 换手率高 后期可以加入其它技术指标
                # 1.3 1.2－2.62
                # 最优乘数和选股范围有关
            elif J[ix] > 93:
                w = 0
            r.append(w)

        res = np.array(r)
        return res    

signal = Signal_STOPL