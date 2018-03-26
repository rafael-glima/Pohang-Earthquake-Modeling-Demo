# -*- coding: utf-8 -*-
#' % Pohang Earthquake Sequence Modeling
#' % Rafael Lima & Jaesik Choi (SAIL-UNIST)
#' % March, 2018

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
#from SimHawkesProcesses import simHP
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
#from trainGD_PWL import trainGD_PWL
#from trainGD_EXP import trainGD_EXP

#from IntensityGrid import IntensityGrid
import pickle

seq = [0.]

mags = [0.]

for i in range(1,11):

	#url = 'http://www.kma.go.kr/weather/earthquake_volcano/domesticlist.jsp?startTm=2017-01-01&endTm=2017-11-21&startSize=3&endSize=7&startLat=&endLat=&startLon=&endLon=&lat=&lon=&dist=&keyword=&x=49&y=7' #'http://newsky2.kma.go.kr/service/ErthqkInfoService/EarthquakeReport'
	url = 'http://www.weather.go.kr/weather/earthquake_volcano/domesticlist.jsp?startSize=999&endSize=999&pNo='+repr(i)+'&startLat=999.0&endLat=999.0&startLon=999.0&endLon=999.0&lat=999.0&lon=999.0&dist=999.0&keyword=&startTm=2017-01-01&endTm=2017-11-21' #'http://newsky2.kma.go.kr/service/ErthqkInfoService/EarthquakeReport'
	print(url)
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
		#print(cols)
		cols = [ele.text.strip() for ele in cols]
		data.append([ele for ele in cols if ele])

	dates = np.array([])
	for i in range(len(data)-1):
		#print(type(data[i][1]))
		dates = np.append(dates, data[i][1])

	marks = np.array([])
	for i in range(len(data)-1):
		#print(type(data[i][2]))
		marks = np.append(marks, float(data[i][2]))

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

	diff_dates[0] += seq[-1]

	seq = np.append(seq, np.cumsum(diff_dates))#[::-1]

	mags = np.append(mags,marks)

# print(seq)

#print(marks.astype(np.float64))

# print(len(mags))

# print(len(seq))

seq = seq[1:]

mags = mags[1:]

np.savetxt('earthquake_seq.txt', seq, fmt='%d')
np.savetxt('earthquake_mags.txt', mags, fmt='%f')
