# coding=utf-8
# Signal_101.py
'''
 如果你希望实验并查看基于本范例的修改，
 请一定在配置文件yaml中指定为其他 Name
'''
import numpy as np

from simlib.signal.base import Signal
from simlib.signal.lib import *


class Signal_101(Signal):    #继承Signal是必须的

    required_data = ["ClosePrice"]    # required_data用来准备数据，不是必须的。
                                      # 忽略required_data 会准备所有DataManager的数据
    '''

     目前已有的DataManager：
     eod (量价数据)： ticker_names, dates, UpDownLimitStatus, AdjFactor, Turonver, VWAP,
                      ClosePrice, HighestPrice, LowestPrice, OpenPrice, PreClosePrice, Volume
     index (指数数据)
     spread: ticker_names, dates, Spread
     group (行业分组): group
     fund (基本面数据)
     factors (基本面数据)
     
     “dates” 和 “ticker_names” 是默认准备的，无须required_data中声明

    '''

# prebatch 函数实现对数据的准备工作，并非必需
# 你当然也可以编写其他特有的合适的函数
    def prebatch(self):

        data = self.eod.ClosePrice    # 读取EOD数据 ClosePrice。
                                      # 同样的方法可以用来读取准备好的其他DataManager数据

# 函数self.function_wrapper() 完成了对ta-lib 的包装，可以用来调用相关计算方法。
# 变量名 = self.function_wrapper(“ta-lib函数名”，输入数据，函数参数1，函数参数2，… … )
# ta-lib是开源库，函数相关用法可以通过Python “help”命令或阅读其源代码了解
        five_ma = self.function_wrapper("MA", data, timeperiod=5)    #计算data值的五日移动平均值
        ten_ma = self.function_wrapper("MA", data, timeperiod=10)    #计算data值的十日移动平均值
        self.dif = five_ma - ten_ma    #将不同期移动平局数据取差值赋值全局变量dif待用。

# generate 函数是信号计算和确认的核心部分，其输入变量是时间节点di
# 函数任务是返回 di 时点的所有头寸相对权重比例weights
    def generate(self,di):

        r = []

        data = self.dif    #获得dif值。这里的数据包含universe所有标的完整的时间序列，不能直接用于计算。
        observed = data[di-2:di].T    # 截取di时点下“昨天”和“前天”的数据。
                                      # 注意：任何包含di时点或之后未来数据的计算行为系统都将拒绝并报错
        last_wgt = self.weights[-1]   # 定义lat_wgt表示前一时点weights

        for ix, ticker in enumerate(observed):    #针对每一个标的资产
            w = last_wgt[ix]    #第ix资产配置权重w初始化为前一时点权重

# 信号判断主体：如果五日移动平均线上穿十日移动平均线，则持有该资产多头头寸；
# 反之，不持该资产或持有其空头头寸（允许做空情况下）
            if ticker[1] < 0 and ticker[0] > 0:
                w = ticker[1]/ticker[0]

            if ticker[1] > 0 and ticker[0] < 0:
                w = abs(ticker[1]/ticker[0])
# 只做多情况下，负数权重将被认定为“不持有”

            r.append(w)

        res = np.array(r)
        return res    #返回universe涵盖标的权重数组，完成di时点计算和信号输出。

signal = Signal_101

''' 

 剩下的就可以交给TDK强大的回测和交易平台处理了。
 需要提示的是，权重w在计算、给定和返回过程中，不需要信号编写者刻意对其归一处理
 平台会根据返回值的相对关系自动完成归一处置
 信号内不必要的归一处理很可能导致事倍功半

'''
