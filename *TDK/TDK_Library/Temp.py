# coding=utf-8
# Signal_Temp.py

import numpy as np
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_Temp(Signal):

    def DIFF(self,data):    # 可用状态
        diff_data = np.empty(data.shape)
        diff_data[1:,:] = data[1:,:] - data[:-1,:]
        return diff_data

    def prebatch(self):

        # 引入指数
        index = self.index.ClosePrice
        idx = np.where(self.index.ticker_names=='000905')[0][0]
        index_close = index[:,idx]
        index_ma = ta.MA(index_close,timeperiod=60)
        
        # 引入因子
        F1 = self.factors.BP
        F1_msk = np.ma.array(F1,mask = np.isnan(F1))
        F1_z = st.zscore(F1_msk)
        
        # RAW DATA
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        
        # KDJ
        K,D = self.function_wrapper('STOCHF', High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        J = 3*K-2*D
        KD = self.DIFF(K)
        DD = self.DIFF(D)
        JD = self.DIFF(J)
            
        # MACD
        DIF,DEA,HIST = self.function_wrapper("MACD", Close, fastperiod=12, slowperiod=26,signalperiod=9)
        DIFD = self.DIFF(DIF)
        DEAD = self.DIFF(DEA)
        HISTD = self.DIFF(HIST)
        
        # BB
        BBU, BBM, BBL = self.function_wrapper('BBANDS', Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        BBUD = self.DIFF(BBU)
        BBMD = self.DIFF(BBM)
        BBLD = self.DIFF(BBL)
        
        # Diff
        self.diff = index_close - index_ma
        self.diff2 = self.DIFF(Close)
        diff2 = np.empty(Close.shape)
        diff2[1:,:] = Close[1:,:] - Close[:-1,:]
        self.diff2 = diff2
    
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
        # diff = self.diff[di-2:di]
        diff = self.diff[di-2:di] + self.diff[di-3:di-1]
        # 过去两天的变动
        K = self.K[di-1,:].T
        D = self.D[di-1,:].T
        J = self.J[di-1,:].T
        JD = self.JD[di-1,:].T
        diff2 = self.diff2[di-1,:].T
        close = self.close
        rtn2 = diff2/close[di-3,:].T
        
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
            
            if diff[1] < -3 or rtn2[ix] < -0.095:
                w = 0
                
            #elif index_J > 100 and index_JD <= 0:
                #引入了指数kdj
            #    w = 0
                
            elif J[ix] < 10 or (J[ix] < 25 and J[ix] > D[ix] ):
                w = 100.-J[ix]

            elif J[ix] > 93:
                w = 0
            r.append(w)

        res = np.array(r)
        return res    

signal = Signal_Temp