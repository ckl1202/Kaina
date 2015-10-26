# coding=utf-8
# Signal_Template.py

import numpy as np
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *

status = 0

class Signal_Template(Signal):

    def DIFF(self,data):    # 可用状态
        diff_data = np.empty(data.shape)
        diff_data[1:] = data[1:] - data[:-1]
        return diff_data

    def prebatch(self):
        # INDEX
        idx = np.where(self.index.ticker_names=='000300')[0][0]
        index_Close = self.index.ClosePrice[:,idx]
        index_High = self.index.HighestPrice[:,idx]
        index_Low = self.index.LowestPrice[:,idx]
        # self.index_CD = np.empty(index_Close.shape)
        # self.index_CD[1:,:] = index_Close[1:] - index_Close[:-1]
        # self.DIFF ?? 
        index_Volume = self.index.Volume[:,idx]
        index_ma = ta.MA(index_Close,timeperiod=60)
        self.index_volma = ta.MA(index_Volume,timeperiod=5)
        self.index_ma_diff = self.DIFF(index_ma)
        self.index_Close = index_Close
        self.index_CD = self.DIFF(index_Close)
        self.index_Volume = index_Volume
        self.index_ma = index_ma
        # INDEX_KDJ
        index_K,index_D = ta.STOCHF(index_High, index_Low, index_Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        index_J = 3*index_K-2*index_D
        self.index_K = index_K
        self.index_D = index_D
        self.index_J = index_J
        self.index_KD = self.DIFF(index_K)
        self.index_DD = self.DIFF(index_D)
        self.index_JD = self.DIFF(index_J)
        # INDEX_MACD
        index_DIF,index_DEA,index_HIST = ta.MACD(index_Close, fastperiod=12, slowperiod=26,signalperiod=9)
        self.index_DIFD = self.DIFF(index_DIF)
        self.index_DEAD = self.DIFF(index_DEA)
        self.index_HISTD = self.DIFF(index_HIST)
        self.index_DIF = index_DIF
        self.index_DEA = index_DEA
        self.index_HIST = index_HIST
        # FACTORS
        F1 = self.factors.BP
        F1_msk = np.ma.array(F1,mask = np.isnan(F1))
        F1_z = st.zscore(F1_msk)
        F2 = self.factors.TOT_D_MKTASSETS
        F2_msk = np.ma.array(F2,mask = np.isnan(F2))
        F2_z = st.zscore(F2_msk)
        F3 = self.factors.TOT_D_MKTEQUITY
        F3_msk = np.ma.array(F3,mask = np.isnan(F3))
        F3_z = st.zscore(F3_msk)
        self.fscore = 1.5*F1_z + F2_z + 0.5*F3_z
        # fscore! to HELP DISTINGUISH between fscore and percentagescore
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        # For sorting the tickers
        # Weighted average of factors - Multi-factor Model
        # RAW DATA
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        self.CD = self.DIFF(Close)
        self.Close = Close
        # KDJ
        K,D = self.function_wrapper('STOCHF',High,Low,Close,fastk_period=5,fastd_period=3,fastd_matype=0)
        J = 3*K-2*D
        self.KD = self.DIFF(K)
        self.DD = self.DIFF(D)
        self.JD = self.DIFF(J)
        self.K = K
        self.D = D
        self.J = J
        # MACD
        DIF,DEA,HIST = self.function_wrapper("MACD", Close, fastperiod=12, slowperiod=26,signalperiod=9)
        self.DIFD = self.DIFF(DIF)
        self.DEAD = self.DIFF(DEA)
        self.HISTD = self.DIFF(HIST)
        self.DIF = DIF
        self.DEA = DEA
        self.HIST = HIST
        # BB
        BBU,BBM,BBL = self.function_wrapper('BBANDS', Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        self.BBUD = self.DIFF(BBU)
        self.BBMD = self.DIFF(BBM)
        self.BBLD = self.DIFF(BBL)
        self.BBU = BBU
        self.BBM = BBM
        self.BBL = BBL
        # Ticker_names
        self.names = self.eod.ticker_names


    def generate(self,di):
        r = []
        last_wgt = self.weights[-1]
        names = self.names
        # Mandatory setting
        diff2 = self.CD[di-2:di] + self.CD[di-3:di-1]
        CD = self.CD[di-1,:].T
        
        index_ST = self.index_Close - self.index_ma
        
        
        K = self.K[di-1,:].T
        D = self.D[di-1,:].T
        J = self.J[di-1,:].T
        JD = self.JD[di-1,:].T
        close = self.Close[di-4:di-2,:]
        rtn2 = (diff2/close).T
        
        # Factors' score
        F1 = self.F1[di-1,:].T
        fscore = self.fscore[di-1].T
        
        # Percent's score based on F1 (Altenatively F2,F3,fscore)
        score = st.scoreatpercentile(F1[~np.isnan(F1)],10)

        for ix, ticker in enumerate(names):
            w = last_wgt[ix]
            
            if J[ix] < 2 or (J[ix] < 25 and J[ix] > D[ix] ):
                w = 100.-J[ix]
            elif J[ix] > 93:
                w = 0
            r.append(w)

        res = np.array(r)
        return res    

signal = Signal_Template