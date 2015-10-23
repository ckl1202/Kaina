# coding=utf-8
# Signal_FT2.py
#

import numpy as np
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_FT2(Signal):

    def prebatch(self):
        F1 = self.factors.BP
        F1_msk = np.ma.array(F1,mask=np.isnan(F1))
        F1_z = st.zscore(F1_msk)
        
        self.names = self.eod.ticker_names
        self.F1 = F1
        self.s = F1_z

    def generate(self, di):
        r = []
        last_wgt = self.weights[-1]

        F1 = self.F1[di-1,:].T
        s = self.s[di-1]

        score = st.scoreatpercentile(F1[~np.isnan(F1)], 90)

        for ix, ticker in enumerate(self.names):
            w = last_wgt[ix]
            if ticker >= score:
                w = s[ix]
                # w = np.exp(s[ix])
            else:
                w = 0.

            r.append(w)
        res = np.array(r)
        return res 

signal = Signal_FT2