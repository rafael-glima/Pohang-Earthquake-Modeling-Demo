# -*- coding: utf-8 -*-
#' % Pohang Earthquake Sequence Modeling
#' % Rafael Lima & Jaesik Choi (SAIL-UNIST)
#' % November, 2018

#' # Introduction

#' This demo produces a comparison graph of Goodness-of-Fit plots among a Homogeneous Poisson Process and a Univariate Hawkes Process with Power-Law Kernel.

#+ echo=False
# This an example of a script that can be published using
# [Pweave](http://mpastell.com/pweave). The script can be executed
# normally using Python or published to HTML with Pweave
# Text is written in markdown in lines starting with "`#'` " and code
# is executed and results are included in the published document.
# The concept is similar to
# publishing documents with [MATLAB](http://mathworks.com) or using
# stitch with [Knitr](http://http://yihui.name/knitr/demo/stitch/).

# Notice that you don't need to define chunk options (see
# [Pweave docs](http://mpastell.com/pweave/usage.html#code-chunk-options)
# ),
# but you do need one line of whitespace between text and code.
# If you want to define options you can do it on using a line starting with
# `#+`. just before code e.g. `#+ term=True, caption='Fancy plots.'`. 
# If you're viewing the HTML version have a look at the
# [source](FIR_design.py) to see the markup.

# The code and text below comes mostly
# from my blog post [FIR design with SciPy](http://mpastell.com/2010/01/18/fir-with-scipy/),
# but I've updated it to reflect new features in SciPy. 

#' ## Goodness-of-Fit Plot

#' This is a comparison among the Homogeneous Poisson Process and the Hawkes Process with Power-Law kernel:

#+ echo=False

# import urllib.request as urllib2
from bs4 import BeautifulSoup
import sys
import os
sys.path.append(os.getcwd())
sys.path.append("..")

# hoover = 'http://www.kma.go.kr/weather/earthquake_volcano/domesticlist.jsp startTm=2017-01-01&endTm=2017-11-21&startSize=0&endSize=7&startLat=&endLat=&startLon=&endLon=&lat=&lon=&dist=&keyword=&x=49&y=7'
# req = urllib2.Request(hoover)
# page = urllib2.urlopen(req)
# soup = BeautifulSoup(page, 'lxml')#.text)

# print(soup.prettify())
import time
import numpy as np
import datetime
import urllib.request as urllib2
from urllib.request import Request, urlopen
#from urllib import urlencode, quote_plus
from urllib.parse import urlencode, quote_plus

#import SimHawkesProcesses
from SimHawkesProcesses import simHP
#from SimSeqNonParamKernel import SimSeqNonParamKernel

from joblib import Parallel, delayed
import multiprocessing
import parmap
from multiprocessing import Pool, freeze_support
import itertools

import scipy.io
import math
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random as rd

import pandas as pd
#import numpy as np
import datetime as DT
import time
#import matplotlib.pyplot as plt

#from K1_Est import K1_Est
#from K1_Class import K1_Class
#from EXP_Class import EXP_Class
#from PWL_Class import PWL_Class
#from SQR_Class import SQR_Class
#from SNS_Class import SNS_Class
from trainGD_PWL import trainGD_PWL
from trainGD_EXP import trainGD_EXP

from IntensityGrid import IntensityGrid
import pickle

url = 'http://www.kma.go.kr/weather/earthquake_volcano/domesticlist.jsp?startTm=2017-01-01&endTm=2017-11-21&startSize=3&endSize=7&startLat=&endLat=&startLon=&endLon=&lat=&lon=&dist=&keyword=&x=49&y=7' #'http://newsky2.kma.go.kr/service/ErthqkInfoService/EarthquakeReport'
#queryParams = '?' + urlencode({ quote_plus('ServiceKey') : '서비스키', quote_plus('numOfRows') : '10', quote_plus('pageNo') : '1', quote_plus('fromTmFc') : '20170101', quote_plus('toTmFc') : '20170720', quote_plus('serviceKey') : 'TEST_SERVICE_KEY' })

request = Request(url)# + queryParams)
request.get_method = lambda: 'GET'
response_body = urlopen(request).read()
#print(response_body)

data = []

soup = BeautifulSoup(response_body, 'lxml')
#print(soup)
table = soup.find("table")
table_body = table.find('tbody')
rows = table_body.find_all('tr')

for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])

dates = np.array([])
for i in range(len(data)-1):
	#print(type(data[i][1]))
	dates = np.append(dates, data[i][1])

#print(dates)
#print(len(data))

for i in range(len(dates)):

	tmp = datetime.datetime.strptime(dates[i], '%Y/%m/%d %H:%M:%S')
	tmp = time.mktime(tmp.timetuple())
	dates[i] = float(tmp)
	#print(dates[i])

#print(dates)
dates = dates[::-1]
dates = np.float64(dates)

diff_dates = np.array([0.])

for i in range(1, len(dates)):

	tmp = dates[i]-dates[i-1]
	diff_dates = np.append(diff_dates,tmp)
#print(diff_dates)

seq = np.cumsum(diff_dates)#[::-1]

#print(seq)

taumax = seq[-1]

poisson_intens = len(seq)/seq[-1]

########## Poisson Process #################
t = 0.
poisson_seq = np.array([])
#print('taumax: ' + repr(taumax))

while t <= taumax:
	#print('t: '+ repr(t))
	t += rd.expovariate(poisson_intens)
	poisson_seq = np.append(poisson_seq,t)

Param_GD_PWL = trainGD_PWL(seq)
mu = Param_GD_PWL["PWL_coeffs"][3]
#print(Param_GD_PWL)
#print('PWL parameters: ' + repr(Param_GD_PWL['PWL_statcriter']))
Delta_GD = Param_GD_PWL["PWL_coeffs"]
sim_GD_PWL = simHP(1, Param_GD_PWL, len(seq), taumax, Delta_GD[3])
#print('seq.shape: ' + repr(seq.shape))
#print('sim_GD_PWL.shape: ' + repr(sim_GD_PWL.shape))
min_len = min(len(seq), len(sim_GD_PWL),len(poisson_seq))
plt.plot(seq[:min_len], sim_GD_PWL[:min_len], 'y-.',linewidth=5.0, label='Gradient Descent Hawkes')
plt.plot(seq[:min_len], poisson_seq[:min_len], 'b-.',linewidth=5.0, label='Poisson Process')
plt.plot(seq,seq, 'g-', linewidth=5.0, label='Original Sequence')
plt.legend(loc='lower right')
plt.savefig('Earthquake_Pohang_QQ_Plots_with_GD.png')
plt.close()

Param_PWL = Param_GD_PWL

#Param_GD_EXP = trainGD_EXP(seq)
#print(Param_GD_EXP)
#print('EXP parameters: ' + repr(Param_GD_EXP['EXP_statcriter']))
#Delta_GD = Param_GD_EXP["EXP_coeffs"]
#sim_GD_EXP = simHP(1, Param_GD_EXP, len(seq), taumax, Delta_GD[2])
#print('seq.shape: ' + repr(seq.shape))
#print('sim_GD_EXP.shape: ' + repr(sim_GD_EXP.shape))
#min_len = min(len(seq), len(sim_GD_EXP),len(poisson_seq))
#plt.plot(seq[:min_len], sim_GD_EXP[:min_len], 'y-.',linewidth=5.0, label='Gradient Descent Hawkes')
#plt.plot(seq[:min_len], poisson_seq[:min_len], 'b-.',linewidth=5.0, label='Poisson Process')
#plt.plot(seq,seq, 'g-', linewidth=5.0, label='Original Sequence')
#plt.legend(loc='lower right')
#plt.savefig('Earthquake_Pohang_QQ_Plots_with_GD.png')
#plt.close()

sim_seq = seq

Param_EXP = {'K1_Type': 'EXP','EXP_coeffs':[1.,4.]}

#Param_PWL = {'K1_Type': 'PWL','PWL_coeffs':[1.,1.,1.]}

Param_SQR = {'K1_Type': 'SQR', 'SQR_coeffs':[0.5,.9]}

Param_SNS = {'K1_Type': 'SNS', 'SNS_coeffs':[1.,2.3]}

EXP_Grid = IntensityGrid(Param_EXP,sim_seq,mu)

PWL_Grid = IntensityGrid(Param_PWL,sim_seq,mu)

# SQR_Grid = IntensityGrid(Param_SQR,sim_seq,mu)

# SNS_Grid = IntensityGrid(Param_SNS,sim_seq,mu)

grid = EXP_Grid['grid']

# plt.subplot(411)
# plt.plot(grid,EXP_Grid['intensitygrid'],'b-',linewidth=3)
# plt.axis([0,8,0,2])
# ax = plt.gca()
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor','k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

#plt.subplot(412)
plt.plot(grid,PWL_Grid['intensitygrid'],'g-',linewidth=3, label="Hawkes Power-Law Kernel")
plt.plot(grid, poisson_intens*range(len(grid)), 'b-', linewidth=3, label="Poisson Process")
#plt.axis([0,8,0,3.5])
ax = plt.gca()
ax.set_yticks([])
ax.set_xticks([])
markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
plt.setp(markerline, 'markerfacecolor', 'k')
plt.setp(baseline, 'color','k', 'linewidth', 3)
plt.legend()

# plt.subplot(413)
# plt.plot(grid,SQR_Grid['intensitygrid'],'r-',linewidth=3)
# plt.axis([0,8,0,1.5])
# ax = plt.gca()
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor', 'k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

# plt.subplot(414)
# plt.plot(grid,SNS_Grid['intensitygrid'],'y-',linewidth=3)
# ax = plt.gca()
# plt.axis([0,8,0,2.5])
# ax.set_yticks([])
# ax.set_xticks([])
# markerline, stemlines, baseline = plt.stem(sim_seq, 0.5*mu*np.ones(len(sim_seq)), 'k-')
# plt.setp(markerline, 'markerfacecolor', 'k')
# plt.setp(baseline, 'color','k', 'linewidth', 3)

plt.savefig('intens_comparison_demo.png')
plt.close()
