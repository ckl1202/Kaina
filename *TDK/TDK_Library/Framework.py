# coding=utf-8
# Signal_TI_Framework.py

import numpy as np
import pandas as pd
import sklearn as sk
import talib as ta
import scipy.stats as st
from simlib.signal.base import Signal
from simlib.signal.lib import *

status = 0

class Signal_TI_Framework(Signal):

    def DIFF(self,data):    # 可用状态
        diff_data = np.empty(data.shape)
        diff_data[1:] = data[1:] - data[:-1]
        return diff_data

    def prebatch(self):
        # INDEX
        idx = np.where(self.index.ticker_names=='000300')[0][0]
        index_Close = self.index.ClosePrice[:,idx]
        index_High = self.index.HighestPrice[:,idx]
        index_Low = self.index.LowestPrice[:,idx]
        # self.index_CD = np.empty(index_Close.shape)
        # self.index_CD[1:,:] = index_Close[1:] - index_Close[:-1]
        # self.DIFF ?? 
        index_Volume = self.index.Volume[:,idx]
        index_ma = ta.MA(index_Close,timeperiod=60)
        self.index_volma = ta.MA(index_Volume,timeperiod=5)
        self.index_ma_diff = self.DIFF(index_ma)
        self.index_Close = index_Close
        self.index_CD = self.DIFF(index_Close)
        self.index_Volume = index_Volume
        self.index_ma = index_ma
        # INDEX CLOSE PRICE / CLOSE MA / VOLUME MA / CLOSE DIFF / MA DIFF
        # INDEX_KDJ
        index_K,index_D = ta.STOCHF(index_High, index_Low, index_Close, fastk_period=5, fastd_period=3, fastd_matype=0)
        index_J = 3*index_K-2*index_D
        self.index_K = index_K
        self.index_D = index_D
        self.index_J = index_J
        self.index_KD = self.DIFF(index_K)
        self.index_DD = self.DIFF(index_D)
        self.index_JD = self.DIFF(index_J)
        # INDEX_MACD
        index_DIF,index_DEA,index_HIST = ta.MACD(index_Close, fastperiod=12, slowperiod=26,signalperiod=9)
        self.index_DIFD = self.DIFF(index_DIF)
        self.index_DEAD = self.DIFF(index_DEA)
        self.index_HISTD = self.DIFF(index_HIST)
        self.index_DIF = index_DIF
        self.index_DEA = index_DEA
        self.index_HIST = index_HIST
        # FACTORS
        F1 = self.factors.BP
        F1_msk = np.ma.array(F1,mask = np.isnan(F1))
        F1_z = st.zscore(F1_msk)
        F2 = self.factors.TOT_D_MKTASSETS
        F2_msk = np.ma.array(F2,mask = np.isnan(F2))
        F2_z = st.zscore(F2_msk)
        F3 = self.factors.TOT_D_MKTEQUITY
        F3_msk = np.ma.array(F3,mask = np.isnan(F3))
        F3_z = st.zscore(F3_msk)
        self.fscore = 1.5*F1_z + F2_z + 0.5*F3_z
        # fscore! to HELP DISTINGUISH between fscore and percentagescore
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        # For sorting the tickers
        # Weighted average of factors - Multi-factor Model
        # RAW DATA
        Close = self.eod.ClosePrice
        High = self.eod.HighestPrice
        Low = self.eod.LowestPrice
        self.CD = self.DIFF(Close)
        self.Close = Close
        # KDJ
        K,D = self.function_wrapper('STOCHF',High,Low,Close,fastk_period=5,fastd_period=3,fastd_matype=0)
        J = 3*K-2*D
        self.KD = self.DIFF(K)
        self.DD = self.DIFF(D)
        self.JD = self.DIFF(J)
        self.K = K
        self.D = D
        self.J = J
        # MACD
        DIF,DEA,HIST = self.function_wrapper("MACD", Close, fastperiod=12, slowperiod=26,signalperiod=9)
        self.DIFD = self.DIFF(DIF)
        self.DEAD = self.DIFF(DEA)
        self.HISTD = self.DIFF(HIST)
        self.DIF = DIF
        self.DEA = DEA
        self.HIST = HIST
        # BB
        BBU,BBM,BBL = self.function_wrapper('BBANDS', Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        self.BBUD = self.DIFF(BBU)
        self.BBMD = self.DIFF(BBM)
        self.BBLD = self.DIFF(BBL)
        self.BBU = BBU
        self.BBM = BBM
        self.BBL = BBL
        # Ticker_names
        self.names = self.eod.ticker_names


    ##### TI FUNCTIONS #####
    ### Complete Version ###
    def BULLDULL(self,di):
        # 牛入盘整
        if (self.index_J[di-1] > 85 and self.index_JD[di-1] < -30) 
        or (self.index_DIFD[di-1] < -20) : # J急跌或者DIF连续两天急跌
            return True
        pass
    def BEARDULL(self,di):
        # 熊入盘整
        # KDJ 地位金叉
        if self.index_J[di-1] > self.index_D[di-1] and 
            return True
    def DULLBULL(self,di):
        # 激活牛市
        # 60日均线向上且出现放量
        uptrend = 5
        if self.index_ma_diff[di-1] > uptrend and self.index_vol[di-1] > 1.2 * self.index_volma[di-1] :
            return True
    def DULLBEAR(self,di):
        # 激活熊市
        # 60日均线向下且明显缩量
        downtrend = -10
        if self.index_ma_diff[di-1] < downtrend and self.index_vol[di-1] < 0.8 * self.index_volma[di-1]:
            return True

    ### Simplified Version ###
    def protected(self,di):
        if self.index_DIF[di-1] < 0: # and self.index_DIFD[di-1] < 0:
            return True
    def resume(self,di):
        if self.index_DIF[di-1] > 0: # and self.index_DIFD[di-1] > 5:
            return True
    ########################

    def generate(self, di):
        ### MANDATORY SETTING ###
        r = []
        last_wgt = self.weights[-1]
        names = self.names

        ### TI commands ###
        global status
        if status == 0:
            if self.DULLBULL(di):
                status = 1
            elif self.DULLBEAR(di):
                status = -1
        if status == 1:
            if self.BULLDULL(di):
                status = 0
        if status == -1:
            if self.BEARDULL(di):
                status = 0
        # Simplified
        if status = 0:
            if self.resume(di):
                status = 1
        if status = 1:
            if self.protected(di):
                status = 0

        ### self. ID ###
        names = self.names
        rtn = self.rtn[di-1,:].T
        rtn2 = (self.CD / self.Close)[di-1,:].T

        if status == 0:
            for ix,ticker in enumerate(names):
                w = 0.
                # Internally defined
                r.append(w)
        if status == 1:
            for ix,ticker in enumerate(names):
                w = last_wgt[ix]
                if J[ix] < 2 or (J[ix] < 25 and J[ix] > D[ix]):
                    w = 100.-J[ix]
                elif J[ix] > 93 or (J[ix] > 75 and J[ix] < D[ix]):
                    w = 0.
                r.append(w)
                # if self.index_DIF[di-1] < 5 :
                # 对熊市的反应比均线快 抓反弹效果也好些 但是对市场急跌没有防御力
                # 07年05月～09月以及10月～11月 需要补充其它保护机制

        ####################
        # DULLLLLLLLLLLLLL #
        ####################
        if status = 0:
            for ix,ticker in enumerate(names):
                w = last_wgt[ix]
                if rtn2[ix] < -0.095:
                    w = 0
                elif J[ix] < 10 or (J[ix] < 25 and J[ix] > D[ix]):
                    w = 100.-J[ix]
                elif J[ix] > 93:
                    w = 0
                r.append(w)
        ############################
        # BULLLLLLLLLLLLLLLLLLLLLL #
        ############################
        if status = 1:
            for ix,ticker in enumerate(names):
                w = last_wgt[ix]
                if J[ix] < -5:
                    w = 100.-J[ix]
        
        ############################
        # BEARRRRRRRRRRRRRRRRRRRRR #
        ############################
        if status = -1:
        F1 = self.F1[di-1,:].T
        s = self.s[di-1]
        score = st.scoreatpercentile(F1[~np.isnan(F1)], 70)
            for ix, ticker in enumerate(self.names):
                w = last_wgt[ix]
                if ticker >= score:
                    w = np.exp(s[ix])
                else:
                    w = 0.
                r.append(w)

        ### RETURN A NDARRAY ###
        ### MANDATORY ###
        res = np.array(r)
        return res 

signal = Signal_TI_Framework