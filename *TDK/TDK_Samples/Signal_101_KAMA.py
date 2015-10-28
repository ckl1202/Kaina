# coding=utf-8
# Signal_101_kama.py

import numpy as np

from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_101_kama(Signal):    


    def prebatch(self):

        data = self.eod.ClosePrice    
        short_period = 10
        long_period = 30
        short_ma = self.function_wrapper("MA", data, timeperiod=short_period)
        long_ma = self.function_wrapper("MA", data, timeperiod=long_period)
        short_kama = self.function_wrapper("KAMA", data, timeperiod=short_period)
        long_kama = self.function_wrapper("KAMA", data, timeperiod=long_period)
        MACD,MACDSIG,MACDHIST = self.function_wrapper("MACD", data, fastperiod=12, slowperiod=26,signalperiod=9)
        
        self.kama = long_kama
        
    def generate(self,di):

        r = []

        data = self.kama
        observed = data[di-1:di].T - data[di-2:di-1].T
        
        last_wgt = self.weights[-1]

        for ix, ticker in enumerate(observed):
            w = last_wgt[ix]

            if ticker[0] < 0:
                w = 1.

            if ticker[0] > 0:
                w = 0.

            r.append(w)

        res = np.array(r)
        return res

signal = Signal_101_kama
