# coding=utf-8
# Signal_DULL.py
import numpy as np
import pandas as pd
import sklearn as sk
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *

class Signal_DULL(Signal):
# 本部分提供盘整状态的波段交易策略

	def logdiff(self,data,n=1):
		LDdata = np.zeros(data.shape)
        LDdata[1:,:] = np.log(data[1:,:]/data[:-n,:])
		return LDdata
	def diff(self,data,n=1):
		Ddata = np.zeros(data.shape)
		Ddata[1:,:] = data[1:,:]-data[:-n,:]
		return Ddata
	def prebatch(self):

		### RAW DATA ###
		Volume = self.eod.Volume
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        rtn = self.diff(Close)

        ### MACD & diffs ###
		DIF,DEA,MACDHIST = self.function_wrapper("MACD", data, fastperiod=12, slowperiod=26,signalperiod=9)        
		DIF_diff = self.diff(DIF)
		DEA_diff = self.diff(DEA)
		MACDHIST_diff = self.diff(MACDHIST)
		DIF_diff_rate = DIF_diff / DIF
		DEA_diff_rate = DEA_diff / DEA
		MACDHIST_diff_rate = MACDHIST_diff / MACDHIST

		### KDJ & diffs ###
		KDJ_K, KDJ_D = self.function_wrapper('STOCHF', High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
		KDJ_J = 3 * KDJ_K - 2 * KDJ_D
		KDJ_K_diff = self.diff(KDJ_K)
		KDJ_D_diff = self.diff(KDJ_D)
		KDJ_J_diff = self.diff(KDJ_J)

		# RSI(6,12,24)
		RSI6 = self.function_wrapper('RSI',Close, timeperiod=6)
        RSI12 = self.function_wrapper('RSI',Close, timeperiod=12)
        RSI24 = self.function_wrapper('RSI',Close, timeperiod=24)

		### Trans DATA
		self.J_diff = KDJ_J_diff
		### ... ###

		### MUST HAVE ###
		self.grp = self.all.group

	def generate(self,di):

		r = []
		last_wgt = self.weights[-1]
		names  = self.eod.ticker_names


		for ix, ticker in enumerate(names):
			if DIF_diff
				w = 
		### UNFINISHED ###



signal = Signal_DULL