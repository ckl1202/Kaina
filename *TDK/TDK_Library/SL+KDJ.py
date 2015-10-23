# coding=utf-8
# Signal_STOPL.py

import numpy as np
import talib as ta
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_STOPL(Signal):

    def prebatch(self):

        # 引入指数作用
        index = self.index.ClosePrice
        idx = np.where(self.index.ticker_names=='000905')[0][0]
        index_close = index[:,idx]
        index_kama = ta.MA(index_close,timeperiod=60)
        
        
        
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        K,D = self.function_wrapper('STOCHF', High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        J = 3*K-2*D
        
        # Kdiff = np.empty(self.eod.ClosePrice.shape)
        # Kdiff[1:,:] = K[1:,:] - K[:-1,:]
        
        JD = np.empty(J.shape)
        JD[1:,:] = J[1:,:] - K[:-1,:]
        self.diff = index_close - index_kama
    
        self.K = K
        self.D = D
        self.J = J
        self.JD =JD
        self.names = self.eod.ticker_names
        

    def generate(self,di):

        r = []
        last_wgt = self.weights[-1]
        names = self.names
        diff = self.diff[di-2:di]
        K = self.K[di-1,:].T
        D = self.D[di-1,:].T
        J = self.J[di-1,:].T
        JD = self.JD[di-1,:].T

        for ix, ticker in enumerate(names):
            w = last_wgt[ix]
            
            if diff[1] < 0:
                w = 0
            elif J[ix] < 5:
                w = 100.-J[ix]
            elif J[ix] > 80: #and JD[ix] < 0:
                w = 0
            r.append(w)

        res = np.array(r)
        return res    

signal = Signal_STOPL