from __future__ import print_function
import pandas as pd

import numpy as np  
 
 
data = pd.read_csv('input.csv')
w=data


lind =   list(map(lambda x: float(x.replace(',','.')), w['leaf_ind']))
ctop =   list(map(lambda x: float(x.replace(',','.')), w['curvtop']))
cbas =   list(map(lambda x: float(x.replace(',','.')), w['curvbasis']))
pwid =   list(map(lambda x: float(x.replace(',','.')), w['pwid']))


bins_leafind =np.min(lind)+np.array([0, (np.max(lind)-np.min(lind))/3.0,  2.0*(np.max(lind)-np.min(lind))/3.0, np.max(lind)-np.min(lind)])  
bins_curvtop =np.min(ctop)+np.array([0, (np.max(ctop)-np.min(ctop))/3.0,  2.0*(np.max(ctop)-np.min(ctop))/3.0, np.max(ctop)-np.min(ctop)])
bins_curvbasis =np.min(cbas)+np.array([0, (np.max(cbas)-np.min(cbas))/3.0,  2.0*(np.max(cbas)-np.min(cbas))/3.0, np.max(cbas)-np.min(cbas)])
bins_pwid =np.min(pwid)+np.array([0, (np.max(pwid)-np.min(pwid))/3.0,  2.0*(np.max(pwid)-np.min(pwid))/3.0, np.max(pwid)-np.min(pwid)])



for name in np.unique(data['short']):
    lind =  list(map(lambda x: float(x.replace(',','.')), w[w['short']==name]['leaf_ind']))
    ctop =  list(map(lambda x: float(x.replace(',','.')), w[w['short']==name]['curvtop']))
    cbas =  list(map(lambda x: float(x.replace(',','.')), w[w['short']==name]['curvbasis']))
    pwid =  list(map(lambda x: float(x.replace(',','.')), w[w['short']==name]['pwid']))
    bins = (1.0,1.5,3.0,10.0)
    lind = 1.0/np.array(lind)
    lind = np.histogram(lind, bins=bins)[0]/float(np.sum(np.histogram(lind, bins=bins)[0]))
    #bins = (0.0,5, 10.0, 1000.0)
    ctop = np.histogram(ctop, bins=bins_curvtop)[0]/float(np.sum(np.histogram(ctop, bins=bins_curvtop)[0]))
    #bins = (0.0,5, 10.0, 1000.0)
    cbas = np.histogram(cbas, bins=bins_curvbasis)[0]/float(np.sum(np.histogram(cbas, bins=bins_curvbasis)[0]))
    #bins = (-10.,-0.05, 0.05, 10.0)
    pwid = np.histogram(pwid, bins=bins_pwid)[0]/float(np.sum(np.histogram(pwid, bins=bins_pwid)[0]))
    lind = map(lambda x: "{0:.2f}".format(x), lind)
    ctop = map(lambda x: "{0:.2f}".format(x), ctop)
    cbas = map(lambda x: "{0:.2f}".format(x), cbas)
    pwid = map(lambda x: "{0:.2f}".format(x), pwid)
    print(str(name)+','+','.join(list(lind)))
    #print(str(name)+','+','.join(list(lind))+','+','.join(list(ctop))+','+','.join(list(cbas))+','+','.join(list(pwid)))
    