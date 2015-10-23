import pandas as pd
from datetime import datetime as dt
import numpy as np
from string import join
import os


def tryfloat(x):
    '''Safely converts string to float.'''
    try:
        res = float(x)
    except ValueError:
        return np.nan
    return res


def getpars(x):
    '''Get necessary paramters.'''
    try:
        res = [str(x.median(skipna=True)), str(x.max(skipna=True)), str(x.min(skipna=True))]
    except:
        res = ['', '', '']
    return res


def load_file(filename):
    #Read the data to memory
    data = pd.read_csv(filename)
    #IMPORTANT!!! One need to specify format for datetime conversion
    #Current format provided by rp5 is day.month.year hour:minute
    data['time'] = pd.to_datetime(data['time'], format='%d.%m.%Y %H:%M')
    #Convert parameters of interest to floats
    data['T'] = data['T'].map(lambda x: tryfloat(str(x).strip().replace(',','.'))) #temperature     
    data['U'] = data['U'].map(lambda x: tryfloat(str(x).strip().replace(',','.'))) #humidity
    data['RRR'] = data['RRR'].map(lambda x: tryfloat(str(x).strip().replace(',','.'))) #precipitation
    return data


allowed_months = {'May':5, 'June':6, 'July': 7, 'August' :8}

for fitem in [f for f in os.listdir('.') if os.path.isfile(f) and '.csv' in f]:
    with open('out_'+fitem,'w') as ff:
        data = load_file(fitem)
        for month in allowed_months:
            ff.write(month+','*28+'\n')
            for year in range(2004, 2016):
                crit = data['time'].map(lambda x: x.year==year and x.month==allowed_months[month])
                curdata = data.copy()[crit]
                row = str(year) + ',' + join(getpars(curdata['T']), sep=',') + ',' + join(getpars(curdata['U']), sep=',') + ',' + join(getpars(curdata['RRR']), sep=',')
                ff.write(row+'\n')
                   



