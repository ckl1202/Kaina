# coding=utf-8
# library.py
# This script has no accordant yaml file.

import numpy as np
import pandas as pd
import sklearn as sk
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *
# Initial IMPORTs

class library(Signal):  
	# required_data = ["ClosePrice","HighestPrice","LowestPrice","Volume"]
	def logdiff(self,data,n=1):
		LDdata = np.zeros(data.shape)
        LDdata[1:,:] = np.log(data[1:,:]/data[:-n,:])
		return LDdata
	def diff(self,data,n=1):
		Ddata = np.zeros(data.shape)
		Ddata[1:,:] = data[1:,:]-data[:-n,:]
		return Ddata
	def prebatch(self):

        idx = np.where(self.index.ticker_names=='000300')[0][0]
        indexClose = self.index.ClosePrice[:,idx]

        indexKAMAF = ta.KAMA(indexClose,timeperiod=long_period)
        indexKAMA = ta.KAMA(indexClose,timeperiod=short_period)

        Volume = self.eod.Volume
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice

        indexrtn = self.logdiff(indexClose)
        # indexrtn = np.zeros(indexClose.shape)
        # indexrtn[1:] = np.log(indexClose[1:]/indexClose[:-1])
        # ATTENTION!!! The proper way to generate the differential data.
        rtn = self.diff(Close)
        # rtn = np.zeros(Close.shape)
        # rtn[1:,:] = np.log(Close[1:,:]/Close[:-1,:])
        # Return series to support the stoploss algo

        growth = self.factor.YOY_EPS_G
		value = self.factor.BP
		# Fetch the funda data

		# FactorScore = "..."
		# Calculate the score according to Multi-Factor Model
		# The formula could be different according to the market status

        DIF,DEA,MACDHIST = self.function_wrapper("MACD", Close, fastperiod=12, slowperiod=26,signalperiod=9)
        slowk, slowd = self.function_wrapper('STOCH',High, Low, Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        fastk, fastd = self.function_wrapper('STOCHF',High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        slowj = 3 * slowk - 2 * slowd
        fastj = 3 * fastk - 2 * fastd

        BBU, BBM, BBL = self.function_wrapper('BBANDS', Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)

        RSI6 = self.function_wrapper('RSI',Close, timeperiod=6)
        RSI12 = self.function_wrapper('RSI',Close, timeperiod=12)
        RSI24 = self.function_wrapper('RSI',Close, timeperiod=24)
        
        short_period = 10
        long_period = 30
        short_ma = self.function_wrapper("MA", data, timeperiod=short_period)
        long_ma = self.function_wrapper("MA", data, timeperiod=long_period)
        short_kama = self.function_wrapper("KAMA", data, timeperiod=short_period)
        long_kama = self.function_wrapper("KAMA", data, timeperiod=long_period)


        MHD = self.diff(MACDHIST)
        KAMAD = self.diff(long_kama)

        ###### RAW DATA ######
        ###### TRANS ######

        self.res = (rtn.T - ZZrtn).T  
        # Trans ATTENTION
        self.kama = long_kama
        # ... #
        # Transfer variables

        # MUST HAVE
        self.names = self.eod.ticker_names
        self.grp = self.all.group
        # Group data & ticker_names

        # STATUS
        self.status = 0

	def generate(self,di):

		r = []
		last_wgt = self.weights[-1] 
		names = self.names
		# names作为主循环的numerate目标 提取ix再对数据进行处理
		group = self.grp
		# 想要添加一个具有记忆性的模式指标 last_status = self.status[-1] 
		self.status = self.StatusId(self.status,di)

		######
		# 进行之前函数处理结果的传达
		# 重新处理数据 注意命名规则
		######
		######
		# 熊市采用防御型方法 选择行业？？？ 选股？？？ 全范围？？？
		# 只买超跌股 降低交易量 且设置严格的止损止盈（后续完善） 例如连续几天内收益率4%和－5%等
		# 盘整即做波段行情 主要关注
		######
		# 待解决的问题是风格转换时原本的持股权重变换问题
		# 当牛市判定转横盘时 所有股权重归零
		# 保护机制和止损策略有差别 止损策略基于个股收益率决定 指数收益率用于判断趋势转换 尤其是横盘转熊
		######


		if self.status = 1:
		# 牛市策略－准备个股数据
		# 激进型

			w = self.v[di-1] 

			for ix, ticker in enumerate(names):
				w = last_wgt[ix]

				r.append(w)
			Bull = np.array(r)
			return Bull

		if self.status = 0:
		# 波动 寻找超跌股票
			for ix, ticker in enumerate(names):
				w = last_wgt[ix]
				# MACD RT & Cross

				r.append(w)
			Flat = np.array(r)
			return Flat

		if self.status = -1:
		# 同样寻找超跌 如何缩小范围？ 分配权重？ 根据基本面
			for ix, ticker in enumerate(names):
				w = last_wgt[ix]
				r.append(w)
			Bear = np.array(r)
			return Bear


            if ticker in uni:
                w = abs(absrtn[ix])
            else:
                w = 0.0
            r.append(w)

signal = library

