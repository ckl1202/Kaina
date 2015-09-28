# coding=utf-8
# Signal_102_GroupNIndex.py
#
'''

 本范例旨在介绍行业分组group和指数index数据的获取和使用

 关于signal编写的基本框架介绍请参考Signal_101


 group 由1-90 数字指代不同行业分组，
 每只股票的 group 数字对应其所在行业分组


 index 数据包含绝大多数市场指数、行业指数与概念指数，
 其用法与个股eod 数据使用十分相似

'''

import numpy as np
from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_102(Signal):

    def prebatch(self):

        datenum = self.index.ClosePrice.shape[0]
        ZZ500 = '000905'                                # 中证500指数代码
        ZZrtn = np.empty(datenum)

        ZZidx = np.where(self.index.ticker_names==ZZ500)[0][0]  # 确定中证500指数所在位置
        ZZclose = self.index.ClosePrice[:,ZZidx]        # 利用位置获取中证500指数每日收盘价     
        ZZrtn[1:] = np.log(ZZclose[1:]/ZZclose[:-1])    # 计算中证500指数每日收益水平

        close = self.eod.ClosePrice
        rtn = np.empty(self.eod.ClosePrice.shape)
        rtn[1:,:] = np.log(close[1:,:]/close[:-1,:])    # 计算各股每日收益水平

        self.res = (rtn.T - ZZrtn).T                    # 计算各股与中证500指数的每日收益水平差
        self.names = self.eod.ticker_names              # 获取各股代码

        self.grp = self.all.group                       # 获取各股所属行业分组 group 信息


    def generate(self, di):

        r = []
        last_wgt = self.weights[-1]

        absrtn = self.res[di-1,:].T
        names = self.names

        grp = self.grp[di-1,:].T
        grpset = set()
        [grpset.add(g) for g in grp if not np.isnan(g)]
        uni = []

        for g in grpset:
            data = absrtn[grp == g]
            dname = names[grp == g]
            mdata = np.ma.masked_array(data, mask = np.isnan(data))
            adata = np.mean(mdata)

            if adata > 0:                               # 如果该分组对中证500指数的前日收益水平为正
                for i in xrange(len(data)):             # 则将该分组下的股票列入选股列表
                    uni.append(dname[i])

        for ix, ticker in enumerate(names):

            w = last_wgt[ix]

            if ticker in uni:                           # 对选股列表中的股票
                w = abs(absrtn[ix])                     # 为其赋予正权重
            else:
                w = 0.0

            r.append(w)

        res = np.array(r)
        return res

signal = Signal_102
