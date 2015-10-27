# coding=utf-8
# Signal_101_lite.py

import numpy as np

from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_101_lite(Signal):

    def prebatch(self):

        data = self.eod.ClosePrice    
        five_ma = self.function_wrapper("MA", data, timeperiod=5)    
        ten_ma = self.function_wrapper("MA", data, timeperiod=10)
        self.dif = five_ma - ten_ma   

    def generate(self,di):

        r = []

        data = self.dif    
        observed = data[di-2:di].T    
        last_wgt = self.weights[-1]

        for ix, ticker in enumerate(observed):
            w = last_wgt[ix]

            if ticker[1] < 0 and ticker[0] > 0:
                w = ticker[1]/ticker[0]

            if ticker[1] > 0 and ticker[0] < 0:
                w = abs(ticker[1]/ticker[0])

            r.append(w)

        res = np.array(r)
        return res    

signal = Signal_101_lite