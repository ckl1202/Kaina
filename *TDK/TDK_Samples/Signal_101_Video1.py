# coding=utf-8
# Signal_101.py

import numpy as np
from simlib.signal.base import Signal
from simlib.signal.lib import *

class Signal_101_Video(Signal):
	# required_data = ["ClosePrice"]

	def prebatch(self):

		vol = self.eod.Volume
		close = self.eod.ClosePrice

		vol_ma_20 = self.function_wrapper('MA',vol,timeperiod=20)
		close_ma_5 = self.function_wrapper('MA',close,timeperiod=5)

		self.dif_c = close - close_ma_5
		self.dif_v = vol - vol_ma_20
		# 定义类中的全局变量以便下面函数调用
		# 不用的部分可以不用加self

	def generate(self,di):

		r = []

		dif_c = self.dif_c[di-1,:].T
		dif_v = self.dif_v[di-1,:].T

		last_wgt = self.weights[-1]

		for ix,ticker in enumerate(dif_c):
			w = last_wgt[ix]
		# 第ix只股票 
			if dif_v[ix] > 0:
				if ticker > 0:
				# ticker就是dif_c?
					w = 1.
					# 进行买入动作
				else:
					w = 0.
					# 注意化为float
			r.append(w)

		res = np.array(r)
		return res

Signal = Signal_101_Video

