# coding=utf-8
# Signal_KDJ_RT.py
# This signal utilize the rapid return pheno of KDJ, mainly J and D.

import numpy as np
import pandas as pd
import sklearn as sk
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *

class Signal_KDJ_RT(Signal):

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

        ### KDJ & diffs ###
		KDJ_K, KDJ_D = self.function_wrapper('STOCHF', High, Low, Close, fastk_period=5, fastd_period=3, fastd_matype=0)
		KDJ_J = 3 * KDJ_K - 2 * KDJ_D
		KDJ_K_diff = self.diff(KDJ_K)
		KDJ_D_diff = self.diff(KDJ_D)
		KDJ_J_diff = self.diff(KDJ_J)

		self.J = KDJ_J
		self.JD = KDJ_J_diff
		self.D = KDJ_D
		self.DD = KDJ_D_diff

	def generate(self,di):

		r = []
		last_wgt = self.weights[-1]
		names = self.eod.ticker_names

		J = self.J[di-1].T
		JD = self.JD[di:di-2].T
		# 注意转置 
		D = self.D[di-1].T
		DD = self.DD[di:di-2].T

		for ix, ticker in enumerate(names):
			w = last_wgt[ix]
			if J[ix]>80:
				if JD[ix][0] > 0 and JD[ix][1] < 0 and abs(DD[ix][0])<2:
					w = 1./J[ix]
			if J[ix]<20:
				if JD[ix][0] > 0 and JD[ix][1] < 0 and abs(DD[ix][0])<2:
					w = last_wgt[ix] - 1./J[ix]
			r.append(w)
		res = np.array(r)
		return res		

Signal = Signal_KDJ_RT
