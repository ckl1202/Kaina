# coding=utf-8
# Signal_101_Video2.py

import numpy as np
import talib
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *

class Signal_101_Video2(Signal):
	# required_data = ["ClosePrice"]

	def prebatch(self):

		szzz = '000001'
		idx = np.where(self.index.ticker_names==szzz)
		mkt_vol = self.index.Volume[:,idx]
		# 找到上证综指在一系列等大小的ndarray中的位置
		vol_ma = talib.MA(mkt_vol,timeperiod=60)
		reqing = mkt_vol - vol_ma

		growth = self.factor.YOY_EPS_G
		value = self.factor.BP
		# 需要做空值处理 null值需要掩盖

		g_msk = np.ma.array(growth,mask=np.isnan(growth))
		v_msk = np.ma.array(value,mask=np.isnan(value))

		# zscore打分的方法来做
		g_z = st.zscore(g_msk).data
		v_z = st.zscore(v_msk).data
		# 注意zscore的用法

		self.g = g_z
		self.v = v_z
		self.reqing = reqing
		# 声明全局变量

	def generate(self,di):

		if self.reqing[di-1,:] > 0:
			w = self.v[di-1]
			# 直接把zscore生成权重
		else:
			w = self.g[di-1]
		return w

Signal = Signal_101_Video2

