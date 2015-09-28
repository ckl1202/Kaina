# coding=utf-8
# Signal_102_Fund.py
#
'''

 本范例 旨在介绍基本面数据的获取和应用，
 
 关于signal编写的基本框架介绍请参考Signal_101
 
 基本面数据包括KeyItems 和 Factors，
 分别以 fund 和 factors 的DataManager管理

 KeyItems 是财务报表（三大表）数据，名称中带有Q#，
 代表当季（# = 0）或之前#季（# = 1，2，3，4）的财务数据

 Factors 是引伸元素，由KeyItems 计算得到

 本例旨在介绍引用方法，
 全面的KeItems 和 Factors 请参考基本面数据文档 

'''

import numpy as np
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_102(Signal):

    def prebatch(self):
        self.shr = self.fund.CAPQ0_FLOAT_SHR    # 读取fund 数据，
                                                # 获取各时点下最近期财报提供的可流通股数
        self.roe = self.factors.ROE             # 读取factors 数据，获取各时点下ROE

    def generate(self, di):
        r = []
        last_wgt = self.weights[-1]
        shr = self.shr[di-1,:].T                # 截取“昨日”数据（相对di时点下）
        roe = self.roe[di-1,:].T
        score_ = st.scoreatpercentile(shr[~np.isnan(shr)], 80) # 计算可流通股数的前20%位置

        for ix, ticker in enumerate(shr):       # 针对各支股票的循环处理
            w = last_wgt[ix]
            if ticker >= score_:        # 可流通数位列前20%的，赋予权重为该股ROE值
                w = roe[ix]

            r.append(w)
        res = np.array(r)
        return res 

signal = Signal_102
