# coding=utf-8
# Signal_FT.py

import numpy as np
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_FT(Signal):

    def prebatch(self):
        
        score = self.factors.BP
        s_msk = np.ma.array(score,mask=np.isnan(score))
        s_z = st.zscore(g_msk)
        
        self.s = s_z

    def generate(self,di):

        w = self.s[di-1]
        
        return w   

signal = Signal_FT