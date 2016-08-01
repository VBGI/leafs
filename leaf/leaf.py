# -*- coding: utf-8 -*-

import numpy as np
import re
import itertools
from string import atof,split
import os
from scipy import stats as st
import  scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform,pdist
import random
import sklearn as sk
from sklearn.lda import LDA
import pandas as pd
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature 

# from osgeo import gdal
from scipy.interpolate import griddata

#from sklearn import svm
#from sklearn import metrics
#from sklearn import cross_validation

import matplotlib.pyplot as plt
from matplotlib import cm
# import redis, pickle
import pdb
from matplotlib.pyplot import colors
from procrupy import generalized_procrustes_analysis
# from gdata.apps import groups
from numpy.random.mtrand import seed

# remem=redis.Redis()

def set_value(redis, key, value, exp=860000):
    redis.set(key, pickle.dumps(value), ex=exp)

def get_value(redis, key):
    pickled_value = redis.get(key)
    if pickled_value is None:
        return None
    return pickle.loads(pickled_value)

class ContourLeaf(object):
    '''
    Basic class to store data from contour of a leaf.
    '''
    
    def __str__(self):
        return 'Contour leaf object: nodes: %s, dpc: %s' %(len(self.points),self.dpc)
    
    def __init__(self,dpc,points,description=None):
        self.dpc=dpc
        self.points=np.array(points,dtype=np.float64)
        self.description=description
        self.relpoints=None
        self.pcapoints=None
        self.datasource=None
   
        #Properties of contours
        self._length=None
        self._area=None
        self._curvatures=None
        self._inertia=None
        self._maxminxy=None
        
    @property
    def area(self):
        'Area of the contour'
        return self._area

    @area.getter
    def area(self):
        from shapely.geometry import Polygon
        if  self.pcapoints==None:
            self.interpolated(self)
        try:
            polygon=Polygon(self.ppts)
            self._area=polygon.area
        except:
            self._area=None
        return self._area
    
    @property
    def length(self):
        'Length of the contour'
        pass
    
    @length.getter
    def length(self):
        from shapely.geometry import Polygon
        if  self.pcapoints==None:
            self.interpolated(self)
        try:
            polygon=Polygon(self.ppts)
            self._length=polygon.length
        except:
            self._length=None
        return self._length
    
    @property
    def inertia(self):
        pass
    
    @inertia.getter
    def inertia(self):
        if  self.pcapoints==None:
            self.interpolated(self)
        vals=[np.sum(np.array(self.pcapoints[0])**2),np.sum(np.array(self.pcapoints[1])**2)]
        self._inertia=(max(vals),min(vals))
        return self._inertia
    
    @property
    def coordinates(self):
        pass
    
    @coordinates.getter
    def coordinates(self):
        if self.pcapoints==None:
            self.interpolated(self)
        else:
            pass
        return np.array(self.pcapoints)
           
    @property
    def maxminxy(self):
        pass

    @maxminxy.getter
    def maxminxy(self):
        if  self.pcapoints==None:
            self.interpolated(self)
        self._maxminxy = (min(self.pcapoints[0]), max(self.pcapoints[0]), \
                          min(self.pcapoints[1]), max(self.pcapoints[1]))
        return self._maxminxy
    
    @property
    def position(self):
        if  self.pcapoints==None:
            self.interpolated(self)
        indleft = np.argmin(self.pcapoints[1])
        indright = np.argmax(self.pcapoints[1])
        valleft = self.pcapoints[0][indleft]
        valright = self.pcapoints[0][indright]
        return ((valleft+valright)*0.5 - \
                (self.maxminxy[1]+self.maxminxy[0])*0.5)/(self.maxminxy[1]-self.maxminxy[0])
    
    @property
    def curvatures(self):
        pass
    
    @curvatures.getter
    def curvatures(self):
        from scipy import interpolate
        if  self.pcapoints==None:
            self.interpolated(self)
        try:
            tck,unew = interpolate.splprep([np.array(self.ppts)[:,0]/max(np.abs(np.array(self.ppts)[:,0])),np.array(self.ppts)[:,1]/max(np.abs(np.array(self.ppts)[:,1]))], s=0.0)
            dr=np.array(interpolate.splev(unew,tck,der=1)).transpose()
            ddr=np.array(interpolate.splev(unew,tck,der=2)).transpose()
            curvature=[np.linalg.norm(np.cross(x,y))/np.linalg.norm(x)**3 for x,y in zip(dr,ddr)]
            self._curvatures=[min(curvature),max(curvature),np.mean(curvature),np.std(curvature),\
                              np.max(curvature[int(len(curvature)/2.0)-\
                                                int(len(curvature)/8.0):int(len(curvature)/2.0)+\
                                                int(len(curvature)/8.0)]), max(max(curvature[-5:]),max(curvature[:5]))]
        except:
            self._curvatures=[]
        return self._curvatures
    
    @property
    def rawcurvatures(self):
        from scipy import interpolate
        if  self.pcapoints==None:
            self.interpolated(self)
        try:
            tck,unew = interpolate.splprep([np.array(self.ppts)[:,0]/max(np.abs(np.array(self.ppts)[:,0])),np.array(self.ppts)[:,1]/max(np.abs(np.array(self.ppts)[:,1]))], s=0.0)
            dr = np.array(interpolate.splev(unew,tck,der=1)).transpose()
            ddr = np.array(interpolate.splev(unew,tck,der=2)).transpose()
            curvature = np.array([np.linalg.norm(np.cross(x,y))/np.linalg.norm(x)**3 for x,y in zip(dr,ddr)])
        except:
            curvature = np.array([])
        return curvature
   
    @staticmethod
    def centermass(self):
        massx=np.mean(self.points[:,0])
        massy=np.mean(self.points[:,1])
        newx=self.points[:,0]-massx
        newy=self.points[:,1]-massy
        self.relpoints=np.array([newx,newy])
    
    @staticmethod
    def interpolated(self,points=200):
        from sklearn.decomposition import PCA
        from scipy import interpolate
        if self.relpoints is None:
            self.centermass(self)
        pca = PCA(n_components=2)
        tofit=[[x,y] for x,y in zip(self.relpoints[0,:]/self.dpc,self.relpoints[1,:]/self.dpc)]
        tofit.append(tofit[0]) #It is cycled now....
        pca.fit(tofit)
        transformed=pca.transform(tofit)
        self.transformed = transformed
        tck,u=interpolate.splprep([transformed[:,0],transformed[:,1]],s=0.0)
        unew = np.arange(0,1.0,1.0/float(points))
        out = interpolate.splev(unew,tck)
        self.pcapoints=[out[0],out[1]]
        self.ppts=[(x,y) for x,y in zip(self.pcapoints[0],self.pcapoints[1])]

def load_leaf_contours(filename, factor):
    '''
    Load leaf contours from specified file.
    
    '''
    if filename[-3:].lower() != 'txt':
        return [];
    
    with open(filename,'r') as myf:
        all_lines=myf.read()
#        myf.seek(0)
#        line_by_line=myf.readlines()
    
    pat=re.compile(ur'\s*(?:[lL]ine|[Лл]иния)\s*#?\d+\s*',re.I)
    groups=re.split(pat,all_lines.decode('cp1251'))
    leafs=[]
    for group in groups:
        pps=[]
        for line in group.splitlines():
            try:
                a,b=line.split()
                point=[atof(a),atof(b)]
                pps.append(point)
            except:
                pass
        if len(pps)==4:
            dpi=pps
        elif len(pps)>4:
            leafs.append(pps)
        else:
            pass
    #Compute dpi
    resleafs=[]
    try:
        p1,p2,p3,p4=np.array(dpi[0]),np.array(dpi[1]),np.array(dpi[2]),np.array(dpi[3])
        dpc=(np.linalg.norm(p4-p2)+np.linalg.norm(p1-p3))/2.0*factor/np.sqrt(2) #dots per centimeter if factor = 1
        for leaf in leafs:
            resleafs.append(ContourLeaf(dpc,leaf))
    except:
        pass
    return resleafs

def load_data_from_dir(dir, factor):
    '''
    Walk from current directory and find all files that could be loaded.
    Check files if they can be treated as leaf contour files.
    '''
    absdir=os.path.abspath(dir)
    loaded=[]
    for root, dirs, files in os.walk(absdir):
        try:
            geodata=np.genfromtxt(os.path.join(root,'geodata.csv'),delimiter=',',dtype=np.object)
            geodata={item[0]:(float(item[1]),float(item[2])) for item in geodata}
        except:
            geodata=None
        for fname in files:
            print 'Processing file ', fname
            fsplitted=split(fname,sep='_')
            contours=[]
            try:
                data={'place':fsplitted[0],'index':int(fsplitted[1]),\
                  'species':fsplitted[2],'day_col':int(fsplitted[3]),\
                  'month_col':int(fsplitted[4]),'year_col':int(fsplitted[5][:-4]),'fname':fname,'coords': geodata[fname[:-4]] if geodata is not None else None}
                contours=load_leaf_contours(os.path.join(root,fname),factor)
            except:
                try:
                     contours=load_leaf_contours(os.path.join(root,fname),factor)
                     data={'fname': fname}
                except:
                    print 'File <%s> has wrong name. It will be ignored.'%fname
            finally:
                if len(contours)>0:
                    map(lambda x: setattr(x,'datasource',data),contours)
                    hash=0.0
                    for item in contours:
                        hash+=item.length
                    data.update({'hash': np.round(hash,decimals=8)})
                    loaded.append((data,contours))
    return loaded

#leafs=load_leaf_contours('blagr_1_dauricum_14_06_2000.txt')
#
#def get_sample_from_contour(cont,opt=None,disc=5):
#    
#    '''
#    
#    Parameters:
#    ----------
#        cont - an instance of ContourLeaf Class.
#        opt - 'wave' - wavelet transform of the contour, 'geom' - form sample 
#               as geometric features of the contour
#    
#    Return:
#    -------
#    
#        res - List of values.
#
#    '''
#
##   import pywt
#    import cmath
#    assert isinstance(cont,ContourLeaf)
#    item=cont
#    if not item.pcapoints:
#        item.interpolated()
#    
#    ppts=[(x,y) for x,y in zip(item.pcapoints[0],item.pcapoints[1])]
#    result=[]
#    if opt=='geom':
#        from shapely.geometry import Polygon
#        from scipy import interpolate
#        tck,unew = interpolate.splprep([np.array(ppts)[:,0],np.array(ppts)[:,1]], s=0.0)
#        #computation the curvature of a curve
#        dr=np.array(interpolate.splev(unew,tck,der=1)).transpose()
#        ddr=np.array(interpolate.splev(unew,tck,der=2)).transpose()
#        curvature=[np.linalg.norm(np.cross(x,y))/np.linalg.norm(x)**3 for x,y in zip(dr,ddr)]
#        polygon=Polygon(ppts)
#        result.extend([polygon.area,polygon.length,polygon.area/polygon.length, min(curvature),max(curvature),np.mean(curvature[400:600])])
#    elif opt=='fft':
#        tofft=map(lambda x: cmath.polar(complex(x[0],x[1]))[0],ppts)
#        myfft=np.fft.fft(tofft)
#        result=list(myfft.imag[::disc])
#    else:
#        result=map(lambda x: cmath.polar(complex(x[0],x[1]))[0],ppts)[::disc]
#    return result

#def printreport(array,group,store=[],names=None):
#    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table
#    from reportlab.lib.units import inch
#    from reportlab.lib.styles import getSampleStyleSheet
#    mystyle=getSampleStyleSheet()['Normal']
#    store.append(Spacer(1,2*inch)) 
#    doc=SimpleDocTemplate('output.pdf')
#    for gr in list(set(group)):
#        indecies=[i for i in xrange(len(group)) if group[i]==gr]
#        newarray=[array[i] for i in indecies]
#        npa=np.array(newarray)
#        res=[]
#        s=['']
#        if names:
#              s.extend(names)
#        res.append(s)
#        s=['Means:']
#        s.extend(map(lambda x: '%1.2f'%x,list(np.mean(npa,axis=0))))
#        res.append(s)
#        s=['Stds:']
#        s.extend(map(lambda x: '%1.2f'%x,list(np.std(npa,axis=0))))
#        res.append(s)
#        s=['Skews:']
#        s.extend(map(lambda x: '%1.2f'%x,list(st.skew(npa,axis=0))))
#        res.append(s)
#        s=['Kurtos:']
#        s.extend(map(lambda x: '%1.2f'%x,list(st.kurtosis(npa, axis=0))))
#        res.append(s)
##       print res
#        store.append(Paragraph('------------------------',mystyle))
#        store.append(Paragraph('Current group is %s'%gr,mystyle))
#        store.append(Table(res))
#        store.append(Paragraph('------------------------',mystyle))
#    doc.build(store)
    
#Trying to solve problem
#I need to perform comparison by place, by species.

#def getplots(output,mes,uniques):
#    f=figure()
#    ax=f.add_subplot(111)
#    for key in uniques:
#        x=np.linspace(min(output[key][mes]),max(output[key][mes]),2000)
#        g=gaussian_kde(output[key][mes])
#        y=g.evaluate(x)
#        ax.plot(x,y,label=key)
#        ax.legend()
#        ax.set_title(mes)
#    return f

def get_data_attr(data, type='species',mainpath='.'):
    '''
    Get data attribute.

    Available types:
     species - species
     district - administrative district
     
    '''
    def get_coordinates(coord):
        coordpat = re.compile(ur'(\d\d).+(\d\d).+(\d\d[\.,]?\d?).+(\d\d\d).+(\d\d).+(\d\d[\.,]?\d?)')
        cr = coord.decode('utf8')
        coords = coordpat.findall(cr)
        if coord!='nan':
             print 'Evaluated coord', coord, 'recognized:', coords
        try:
            coords=coords[0]
            fc = float(coords[0])+float(coords[1])/60.+float(coords[2])/3600.
            sc = float(coords[3])+float(coords[4])/60.+float(coords[5])/3600.
            return (fc,sc)
        except:
            return
        
    import pandas as pd
    if ('species' in data) and (type=='species'):
        return data['species']
    elif ('place' in data) and (type=='district'):
        return data['place']
    elif ('coords' in data) and (type=='coords'):
        return data['coords']
    else:
        if 'fname' in data:
            w=pd.read_csv(os.path.join(mainpath,'addinfo.csv'))
            mydf=w[w['Имя файла'].map(lambda x: (str(x).strip() in data['fname'].strip())and(len(str(x).strip())>1))]
#             setka=w[w['Измерительная сетка'].map(lambda x: str(x) in data['fname'])]
            if not mydf.empty:
                if type=='coords':
                    coordinates=str(mydf['координаты сбора'].values[0])
#                     pdb.set_trace()
                    return get_coordinates(coordinates)

                res=mydf['Вид'].values[0]
                if 'daur' in res:
                    return 'dauricum'
                elif 'mucr' in res:
                    return 'mucronulatum'
                elif 'sich' in res:
                    return 'sichotense'
                else:
                    return res;
            else:
                return;
        else:
            return;


def get_contour_angles(contour):
    
    def _get_opposite_points_(XC, YC):
        xmax = max(XC)
        xmin = min(XC)
        xmid = 0.5 * (xmax + xmin)
        indmid = np.argmin(abs(XC - xmid))
        midpoint1 = (XC[indmid], YC[indmid])
        indmid2 = np.argmin(abs(XC - midpoint1[0])[YC*midpoint1[1] < 0.0])
        midpoint2 = (XC[YC*midpoint1[1] < 0.0][indmid2], YC[YC*midpoint1[1] < 0.0][indmid2])
        return (midpoint1,  midpoint2)
    def _get_angle_(p0, p1, p2):
        p0_ = np.array(p0)
        p1_ = np.array(p1)
        p2_ = np.array(p2)
        a = p1_ - p0_
        b = p2_ - p0_
        return np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)
    XC = contour.coordinates[0]
    YC = contour.coordinates[1]
    xmin_point = (XC[np.argmin(XC)], YC[np.argmin(XC)])
    xmax_point = (XC[np.argmax(XC)], YC[np.argmax(XC)])
    midpoint501, midpoint502 = _get_opposite_points_(XC, YC)
    midpoint251, midpoint252 = _get_opposite_points_(XC[XC < midpoint501[0]], YC[XC < midpoint501[0]])
    midpoint751, midpoint752 = _get_opposite_points_(XC[XC > midpoint501[0]], YC[XC > midpoint501[0]])
    mainset = set([xmin_point, midpoint251, midpoint252, midpoint501, midpoint502, midpoint751, midpoint752, xmax_point])
    result = []
    ra = result.append
    for basep in mainset:
        for p1, p2 in itertools.combinations(mainset - set([basep]), 2):
            ra(_get_angle_(basep, p1, p2))
    return result


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def procrustes_complex(X,Y):
    y=np.matrix([complex(a,b) for a,b in zip(X[:,0],X[:,1])]).transpose()
    omega=np.matrix([complex(a,b) for a,b in zip(Y[:,0],Y[:,1])]).transpose()
#    assert np.abs(y.conjugate().transpose()*np.ones(y.shape))<1.e-10
#    assert np.abs(omega.conjugate().transpose()*np.ones(omega.shape))<1.e-10
    omegap=(omega.conjugate().transpose()*y)[0,0]*omega/((omega.conjugate().transpose()*omega))[0][0]
    res=y-omegap
    return np.sqrt(res.conjugate().transpose()*res)[0,0]

def slice(x,k):
    N=np.max(np.shape(x))
    z=np.zeros(np.shape(x))
    z[k:N]=x[0:N-k]
    z[0:k]=x[N-k:N]
    return z

def get_contour_pars(c):
    width = c.maxminxy[3]-c.maxminxy[2]
    height = c.maxminxy[1]-c.maxminxy[0]
    curv = c.curvatures[-2]
    ncurv = c.curvatures[-1]
    area = c.area
    contour = c.length
    list_pos = c.position
    return [width, height, curv,ncurv,area,contour,width/height,contour**2.0/area-4.0*np.pi,list_pos]

def print_angle_combs():
    mainset = set(['basis', 'mid251', 'mid252', 'mid501', 'mid502', 'mid751', 'mid752', 'apex'])
    ind = 0
    for basep in mainset:
        for p1, p2 in itertools.combinations(mainset - set([basep]), 2):
            print ind, basep, p1, p2
            ind += 1

def complexprocrustes(c1,c2,diff=20,scaling=True):
    '''Performs procrustes contour c2 on c1. see procrustes for details.
    '''
    cc1=c1.coordinates[:,::diff].transpose()
    cc2=c2.coordinates[:,::diff].transpose()
    mina=np.inf
    res=None
    for zk in xrange(max(np.shape(cc2))):
        slc2=slice(cc2,zk)
        a,b,c=procrustes(cc1,slc2,scaling=scaling)
        a=abs(a)
        if mina>a:
           mina=a
           res=b
    return (mina,res)

def clusteranalysis(contours,diff=20):
    '''
    Procrustes cluster analysis.
    
    Parameters:
    ===========
    
    contours : list of contours to be clusterized.
    '''
    res=np.zeros((len(contours),len(contours)))
    for ind1 in xrange(len(contours)):
        print 'Current contour indices: ',ind1
        for ind2 in xrange(len(contours)):
            if ind2>ind1:
                c1=contours[ind1].coordinates[:,::diff].transpose()
                c2=contours[ind2].coordinates[:,::diff].transpose()
#                mina=np.inf
#                for zk in xrange(max(np.shape(c2))):
#                    slc2=slice(c2,zk)
                a1,b,c=procrustes(c1,c2,scaling=True)
                a2,b,c=procrustes(c2,c1,scaling=True)
                a=abs(a1)+abs(a2)
#                    a=procrustes_complex(c1,slc2)
#                if mina>a:
#                        mina=a
                res[ind1,ind2]=a
                res[ind2,ind1]=a
    return res

def contourset_dist(conts1,conts2):
    leaf_width1=np.array(map(lambda x: x.maxminxy[3]-x.maxminxy[2], conts1)) #NOTE: measured
    leaf_width2=np.array(map(lambda x: x.maxminxy[3]-x.maxminxy[2], conts2)) #NOTE: measured
    leaf_height1=np.array(map(lambda x:x.maxminxy[1]-x.maxminxy[0], conts1)) #NOTE: measured
    leaf_height2=np.array(map(lambda x:x.maxminxy[1]-x.maxminxy[0], conts2)) #NOTE: measured
    leaf_area1=np.array(map(lambda x: x.area, conts1)) #NOTE: measured
    leaf_area2=np.array(map(lambda x: x.area, conts2)) #NOTE: measured
    leaf_len1=np.array(map(lambda x: x.length,conts1)) #NOTE: measured
    leaf_len2=np.array(map(lambda x: x.length,conts2)) #NOTE: measured
#    leaf_curv1=np.array(map(lambda x: x.curvatures[-1],conts1))

#    leaf_curv2=np.array(map(lambda x: x.curvatures[-1],conts2))
    pairs=((leaf_width1,leaf_width2),(leaf_height1,leaf_height2),(leaf_len1,leaf_len2))
    pers=(25,50,75)
    res=0.0
    for p in pairs:
        for prs in pers:
            res+=np.abs(np.percentile(p[0],prs)-np.percentile(p[1],prs))
    for prs in pers:
        res+=np.sqrt(np.abs(np.percentile(leaf_area2,prs)-np.percentile(leaf_area1,prs)))
    return res


def parametrize(contour, resolution=1000):
    from scipy import interpolate
    def length(contour):
        return np.sum(np.linalg.norm(np.array(contour[1:])-np.array(contour[:-1]),axis=1))
    _cont = np.array(contour)
    tck,u=interpolate.splprep([_cont[:,0],_cont[:,1]],s=0)
    unew = np.arange(0,1.0+1.0/resolution,1.0/resolution)
    r = np.array(interpolate.splev(unew,tck)).transpose()
    dr = np.array(interpolate.splev(unew,tck,der=1)).transpose()
    _t = np.arctan(dr[:,1]/dr[:,0])
#     negt = _t < 0.0
#     _t[negt] = np.pi+np.arctan(dr[negt, 1]/dr[negt,0])
    _l = [length(r[:j]) for j in range(1,len(r)+1)]
    _l = np.array(_l)/np.max(_l)
    return (_t, _l, r)


def distance_on_unit_sphere(lat1, long1, lat2, long2):
    import math
    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
         
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
         
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
         
    # Compute spherical distance from spherical coordinates.
         
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    
    try:
        arc = math.acos( cos )
    except ValueError:
        arc = 0.0
 
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc
    
#print 'Computing heights...'
#
#print 'Computing lens...'
#leaf_len=map(lambda x: x.length,gdata) #NOTE: measured
#print 'Computing areas...'
#leaf_area=map(lambda x: x.area, gdata) #NOTE: measured
#print 'Computing curvatures...'
#leaf_curv=np.array(map(lambda x: x.curvatures[-1],gdata))
#print 'Computing height to width ratio...'
#leaf_hw_frac=np.array(leaf_height)/np.array(leaf_width)
#print 'Computing area to len ratio...'
#leafs_al_frac=np.array(leaf_area)/(np.array(leaf_len)**2.0)-0.25
#print 'Forming  parameter matrix...'
#allpars=np.vstack((leaf_curv, leaf_hw_frac, leafs_al_frac)).transpose()


maindatapath1 = '/home/dmitry/РАБОЧАЯ/ДАННЫЕ/РОДОДЕНДРОНЫ_ИННЫ/DONE/1'
maindatapath2 = '/home/dmitry/РАБОЧАЯ/ДАННЫЕ/РОДОДЕНДРОНЫ_ИННЫ/DONE/025'

data1 = load_data_from_dir(maindatapath1,1)
data2 = load_data_from_dir(maindatapath2,2)
data = data1+data2

#------------------------Print report of files loaded ----------------
for item in data:
    print 'Loaded data: Source:',item[0]['fname'], 'Species:', 'Contours:', len(item[1]), 'hash=',item[0]['hash']

#---------------------------------------------------------------------
hashs=map(lambda x: x[0]['hash'], data)

gdata=[]   

for hash in np.unique(hashs):
    owner=filter(lambda x: x[0]['hash'] == hash, data)
    names=''
    for item in owner:
        names+=item[0]['fname']+','
    print 'Hash is %s, total lists: %s, Names: %s'%(hash,len(owner),names)
    gdata.append((owner[0][0],filter(lambda x: x[0]['hash'] == hash, data)[0][1]))
    
print 'Total number of unique herabeous lists: ', len(np.unique(hashs)), 'Total number of all lists:', len(hashs)

#
labels=[]
ind=0
sumcont=0
_gdata=[]
coords=[]
fnames=[]
sp_labels=[]
sp_gdata=[]
for item in gdata:
    sp1=get_data_attr(item[0], type='species', mainpath=maindatapath2)
    sp2=get_data_attr(item[0], type='coords', mainpath=maindatapath2)
#   print 'Evaluating species ', sp1, 'Coords: ', sp2
    if ('nowhere' in item[0]['fname']):
        sp_labels.append(sp1)
        sp_gdata.append(item[1])

    if (sp1 and sp2):
        labels.append(sp1[0])
        _gdata.append(item[1])
        fnames.append(item[0]['fname'])
        print item[0]['fname'], sp2
        coords.append(sp2)

gdata = np.array(_gdata)


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from procrupy import generalized_procrustes_analysis


print 'Total len', len(gdata)

count = 0
data = []
ccd = []
contours = []
for item,cd in zip(gdata, coords):
    for leaf in item:
        count += 1
        data.append(get_contour_angles(leaf))
        ccd.append([cd[0], cd[1]])
        contours.append(leaf.ppts)
contours = np.array(contours)        
ccd = np.array(ccd)

pca = PCA(n_components=50)
datat = pca.fit_transform(data)

y_pred = KMeans(n_clusters=4, max_iter=100000).fit_predict(datat)
plt.scatter(datat[:, 0], datat[:, 1], c=y_pred)
plt.figure()
plt.scatter(ccd[:, 1], ccd[:, 0], c=y_pred, s=100)

res=generalized_procrustes_analysis(contours[y_pred==0])
res1=generalized_procrustes_analysis(contours[y_pred==1])
res2=generalized_procrustes_analysis(contours[y_pred==2])
res3=generalized_procrustes_analysis(contours[y_pred==3])
plt.figure()
plt.plot(res[:,0], res[:,1], 'r')
plt.plot(res1[:,0], res1[:,1], 'g')
plt.plot(res2[:,0], res2[:,1], 'b')
plt.plot(res3[:,0], res3[:,1], 'c')
plt.show()


#print_angle_combs()
#print 'Performin pca on all data:', pca.components_, sum(pca.explained_variance_ratio_)
#for comp in pca.components_:
    #print comp[81], comp[42]

sdfsdf
# 
names = np.loadtxt('newfs.txt', dtype=np.object).tolist()
 
# with open('outputlast.csv', 'wb') as csvfile:
#     csvfile.write('filename, width, height, curv,ncurv,area,contour,width/height,contour**2.0/area-4.0*np.pi,list_pos\n')
#     for name in np.unique(names):
#         for ind, item in enumerate(gdata[np.array(fnames)==name.encode('utf8')][0]):
#             res = [name]
#             res.extend(get_contour_pars(item))
#             csvfile.write(','.join(map(lambda x: str(x), res))+'\n')



 
# names = np.loadtxt('newfs.txt', dtype=np.object).tolist()
#  

import pywt
from scipy import signal
data = []
indd = 1
for name in np.unique(names):
    data = []
#     ax.set_aspect('equal')
    for ind, item in enumerate(gdata[np.array(fnames)==name.encode('utf8')][0]):
        f = plt.figure()
        ax1 = f.add_subplot('111')
        a, b, c = parametrize(item.ppts)
        cam = signal.cwt(a, signal.ricker, range(2,200))
        ax1.imshow(cam)
        ax1.set_title(name+str(ind))
        f.savefig('%s%s.png'%(name,ind), dpi=200)
        plt.close(plt.gcf())

#     mshape = generalized_procrustes_analysis(np.array(data))
    indd += 1
    print indd

# 
sfdfwe

import shutil
os.mkdir('/home/dmitry/txts')
bpath = '/home/dmitry/РАБОЧАЯ/ДАННЫЕ/РОДОДЕНДРОНЫ_ИННЫ'
ofs = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(bpath)
    for f in filter(lambda x: x.lower().endswith(('.txt')), files)]
            
for name in np.unique(names):
    for f in ofs:
        if os.path.basename(name) == os.path.basename(f):
            shutil.copyfile(f, '/home/dmitry/txts/%s'%name)
wesdfsdf




bpath = '/home/dmitry/РАБОЧАЯ/ДАННЫЕ/РОДОДЕНДРОНЫ_ИННЫ'
ofs = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(bpath)
    for f in filter(lambda x: x.lower().endswith(('.jpg','.jpeg','.png')), files)]


def get_image_name(filename):
    def _find_imagename(source):
        image_pat = re.compile(r'[\/\\]([^\/\\]+\.[jJ][pP][eE]?[gG])')
        res = image_pat.findall(source)
        return res[0] if res else ''
    files = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(bpath)
    for f in filter(lambda x: os.path.basename(x) == filename, files)]
    imnames = []
    for item in files:
        with open(item, 'r') as myf:
            pending = _find_imagename(myf.read())
            if pending:
                imnames.append(pending)
    return imnames

def get_image_with_path(imagename):
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(bpath)
            for f in filter(lambda x: os.path.basename(x) == imagename, files)]




# lostf = np.loadtxt('lost.txt', dtype=np.object)
os.mkdir('/home/dmitry/images')





import shutil
# ofs = map(lambda x: os.path.basename(x), ofs)
for f in np.unique(names):
     print '\n'
     for k in np.unique(get_image_name(f)):
         cfl = get_image_with_path(k.decode('cp1251').encode('utf8'))
         if len(cfl)==1:
             shutil.copyfile(cfl[0], '/home/dmitry/images/%s'%os.path.basename(f)+os.path.splitext(cfl[0])[1])
         elif len(cfl) > 1:
             shutil.copyfile(cfl[0], '/home/dmitry/images/%s'%os.path.basename(f)+'_1'+os.path.splitext(cfl[0])[1])
             shutil.copyfile(cfl[1], '/home/dmitry/images/%s'%os.path.basename(f)+'_2'+os.path.splitext(cfl[1])[1])


sdf

import csv
with open('eggs.csv', 'wb') as csvfile:
    csvfile.write('filename, species, lat, lon, sources, xdata, ydata\n')
    for fname, sp, gitem, coord in zip(fnames,labels, gdata, coords):
        ffls = ''
        ffls = '"' + ','.join([t for f in np.unique(get_image_name(fname)) for t in get_image_with_path(f.decode('cp1251').encode('utf8'))]) + '"'
        for item in gitem:
            csvfile.write(','.join([fname, sp, str(coord[0]), str(coord[1]), ffls, '"' + ','.join(map(lambda x: str(x),item.transformed[:,0]))+ '"', '"' + ','.join(map(lambda x: str(x),item.transformed[:,1]))+ '"']))
            csvfile.write('\n')

sdfsdf

xc = map(lambda x: x[0], coords)
yc = map(lambda x: x[1], coords)

_gdata = []
_labels = []
_fnames = []
_coords = []
for item,u,f,v in zip(gdata,labels,fnames,coords):
    _gdata.extend(item)
    _labels.extend([u]*len(item))
    _fnames.extend([f]*len(item))
    _coords.extend([v]*len(item))

fnames = np.array(_fnames)
labels = np.array(_labels)
gdata = np.array(_gdata)
coords = np.array(_coords)


rpat = re.compile(r'([a-zA-Z_]+)[\.\d]?.*')
unames = []
for item in fnames:
    res = rpat.findall(item)
    if res:
        unames.append(res[0])
 
uniquenames = list(np.unique(unames))
unames = np.array(unames)

cont_orbicular = sp_gdata[sp_labels.index('orbicular')][0]
cont_elliptic = sp_gdata[sp_labels.index('elliptic')][0]                     
cont_ovate = sp_gdata[sp_labels.index('ovate')][0] 
cont_obovate = sp_gdata[sp_labels.index('obovate')][0]
cont_oblong = sp_gdata[sp_labels.index('oblong')][0]
cont_sharp = sp_gdata[sp_labels.index('sharp')][0]

#---------------------------Filtration according to shape similarity-----------------------------------
inds = labels=='s'
indd = labels=='d'
indm = labels=='m'


distances_orbicular = np.array(map(lambda x:complexprocrustes(cont_orbicular,x)[0] ,gdata))[inds]
distances_oblong = np.array(map(lambda x:complexprocrustes(cont_oblong,x)[0] ,gdata))[indd]
distances_ovate = np.array(map(lambda x:complexprocrustes(cont_ovate,x)[0] ,gdata))[indm]
distances_obovate = np.array(map(lambda x:complexprocrustes(cont_obovate,x)[0] ,gdata))
distances_elliptic = np.array(map(lambda x:complexprocrustes(cont_elliptic,x)[0] ,gdata))

indorb = distances_orbicular<np.percentile(distances_orbicular,30)
indobl = distances_oblong<np.percentile(distances_oblong,20)
indov = distances_ovate<np.percentile(distances_ovate,40)
indobv = distances_obovate<np.percentile(distances_obovate,10)
indel = distances_elliptic<np.percentile(distances_elliptic,10)

allpars = []
lbls = []
flbls = []
crds=[]
allpars.extend(map(lambda x: get_contour_pars(x),gdata[inds][indorb]))
allpars.extend(map(lambda x: get_contour_pars(x),gdata[indd][indobl]))
allpars.extend(map(lambda x: get_contour_pars(x),gdata[indm][indov]))
allpars.extend(map(lambda x: get_contour_pars(x),gdata[indobv]))
allpars.extend(map(lambda x: get_contour_pars(x),gdata[indel]))
lbls.extend(labels[inds][indorb])
lbls.extend(labels[indd][indobl])
lbls.extend(labels[indm][indov])
lbls.extend(['ho']*len(labels[indobv]))
lbls.extend(['he']*len(labels[indel]))

flbls.extend(unames[inds][indorb])
flbls.extend(unames[indd][indobl])
flbls.extend(unames[indm][indov])
flbls.extend(unames[indobv])
flbls.extend(unames[indel])

crds.extend(coords[inds][indorb])
crds.extend(coords[indd][indobl])
crds.extend(coords[indm][indov])
crds.extend(coords[indobv])
crds.extend(coords[indel])

lbls = np.array(lbls)
allpars = np.array(allpars)
flbls = np.array(flbls)
crds = np.array(crds)

np.set_printoptions(precision=3, threshold=np.nan)

mucr = ['shkot_','khual_','putyatin_','bpeles_','frugelm_','russky_','oktayb_','khasan_','askold_','partizan_','PK_LOZ_per_Chyhynenko_',\
        'anuch_','popova_','nakh_','ussuri_','lazo_','PK_LOZ_Lozo_zk_','stenin_','ussur_','PK_LOZ_c_KIEVKA','nadezd_','khank_','vlad_']
    
# 
# for item in mucr:
#      print item, len(np.unique(fnames[(unames==item)*indm])), sum((unames==item)*indm)
#     
import pdb
cnms={x:[] for x in mucr}
gcnms={x:[] for x in mucr}
ccnms={x:[] for x in mucr}
dtms={x:[] for x in mucr}
mshapes = []
mcoords = []
for flbl in mucr:
    print flbl
    coord0 = crds[flbls==flbl][0]
    for coord,flb,gdta in zip(coords,fnames,gdata):
        if (6373.0*distance_on_unit_sphere(coord0[0], coord0[1], coord[0], coord[1])<3.0)or(flbl in flb):
            cnms[flbl].append(1)
            gcnms[flbl].append(flb)
            ccnms[flbl].append(coord.tolist())
            dtms[flbl].append(get_contour_pars(gdta))
            mshapes.append(gdta)
            mcoords.append(coord.tolist())
            
dtms['lazo_']+=dtms['PK_LOZ_Lozo_zk_']+dtms['PK_LOZ_c_KIEVKA']+dtms['PK_LOZ_per_Chyhynenko_']
dtms['ussur_']+=dtms['ussuri_']
dtms['shkot_']+=dtms['khual_']

ccnms['lazo_']+=ccnms['PK_LOZ_Lozo_zk_']+ccnms['PK_LOZ_c_KIEVKA']+ccnms['PK_LOZ_per_Chyhynenko_']
ccnms['ussur_']+=ccnms['ussuri_']
ccnms['shkot_']+=ccnms['khual_']

gcnms['lazo_']+=gcnms['PK_LOZ_Lozo_zk_']+gcnms['PK_LOZ_c_KIEVKA']+gcnms['PK_LOZ_per_Chyhynenko_']
gcnms['ussur_']+=gcnms['ussuri_']
gcnms['shkot_']+=gcnms['khual_']


del dtms['PK_LOZ_Lozo_zk_'], dtms['PK_LOZ_c_KIEVKA'], dtms['PK_LOZ_per_Chyhynenko_'],dtms['ussuri_'],dtms['khual_']
del ccnms['PK_LOZ_Lozo_zk_'], ccnms['PK_LOZ_c_KIEVKA'], ccnms['PK_LOZ_per_Chyhynenko_'],ccnms['ussuri_'],ccnms['khual_']
del gcnms['PK_LOZ_Lozo_zk_'], gcnms['PK_LOZ_c_KIEVKA'], gcnms['PK_LOZ_per_Chyhynenko_'],gcnms['ussuri_'],gcnms['khual_']
mucr.remove('PK_LOZ_Lozo_zk_')
mucr.remove('PK_LOZ_c_KIEVKA')
mucr.remove('PK_LOZ_per_Chyhynenko_')
mucr.remove('ussuri_')
mucr.remove('khual_')

perh1 = min(map(lambda x: x[0],ccnms['khasan_']))+(max(map(lambda x: x[0],ccnms['khasan_']))-min(map(lambda x: x[0],ccnms['khasan_'])))/3.0
perh2 = min(map(lambda x: x[0],ccnms['khasan_']))+2.0*(max(map(lambda x: x[0],ccnms['khasan_']))-min(map(lambda x: x[0],ccnms['khasan_'])))/3.0
ind1 = np.array(map(lambda x: x[0],ccnms['khasan_']))<=perh1
ind2 = (np.array(map(lambda x: x[0],ccnms['khasan_']))>perh1)*(np.array(map(lambda x: x[0],ccnms['khasan_']))<perh2)
ind3 = np.array(map(lambda x: x[0],ccnms['khasan_']))>=perh2

ss = np.array(dtms['khasan_'])
cc = np.array(ccnms['khasan_'])
ccnms.update({'khasan_S':cc[ind1]})
ccnms.update({'khasan_M':cc[ind2]})
ccnms.update({'khasan_N':cc[ind3]})
dtms.update({'khasan_S':ss[ind1]})
dtms.update({'khasan_M':ss[ind2]})
dtms.update({'khasan_N':ss[ind3]})

del dtms['khasan_'], ccnms['khasan_']
# print len(dtms['khasan_1']),len(dtms['khasan_2']),len(dtms['khasan_3'])
mucr.remove('khasan_')
mucr.extend(['khasan_S','khasan_M','khasan_N'])
sfi=0
for item in mucr:
    print '=====================данные для %s===================='%item
    sfi+=np.max(np.shape(dtms[item]))
    print 'Среднее',np.mean(dtms[item],axis=0), np.max(np.shape(dtms[item]))
    print 'СКО',np.std(dtms[item],axis=0)
    print 'ВАР',st.variation(dtms[item], axis=0)


    

# print mucr,sfi
# 
# # sdf
# print 'cont_orbicular', np.array(get_contour_pars(cont_orbicular))
# print 'cont_elliptic', np.array(get_contour_pars(cont_elliptic))
# print 'cont_ovate',  np.array(get_contour_pars(cont_ovate))
# print 'cont_obovate', np.array(get_contour_pars(cont_obovate))
# print 'cont_oblong',  np.array(get_contour_pars(cont_oblong))    
# print 'cont_sharp',  np.array(get_contour_pars(cont_sharp))

# for item in [cont_orbicular,cont_elliptic,cont_ovate,cont_obovate,cont_oblong,cont_sharp]:
#     n = len(item.pcapoints[0])
#     s = (int(n/2)-int(n/8))+np.argmax(item.rawcurvatures[(int(n/2)-int(n/8)):(int(n/2)+int(n/8))])
#     plt.figure()
#     plt.plot(item.pcapoints[0],item.pcapoints[1],'r', item.pcapoints[0][int(n/2)-int(n/8)],item.pcapoints[1][int(n/2)-int(n/8)],'bo', item.pcapoints[0][int(n/2)+int(n/8)],item.pcapoints[1][int(n/2)+int(n/8)],'bo',\
#          item.pcapoints[0][s],item.pcapoints[1][s],'bs')
# 
# plt.show()
#     print item, len(np.unique(gcnms[item]),len(gcnms[item])


# #--------------------------------From South to North: affects on the leaf shape--------------------------------
# #Inputs: mshapes, mcoords
# 
# 
# #Most southern index
# formston = np.argsort(map(lambda x: x[0], mcoords))
# store = formston.copy().tolist()
# mgroups = []
# mcrds = [] 
# while any(store):
#     cc = filter(lambda x: 6373.0*distance_on_unit_sphere(mcoords[store[0]][0], mcoords[store[0]][1], mcoords[x][0], mcoords[x][1])<30.0, store)
#     mgroups.append([mshapes[i] for i in cc])
#     mcrds.append(np.mean([mcoords[i] for i in cc],axis=0))
#     store = filter(lambda x: x not in cc, store)
# 
# ind=1
# shs = []
# pars = []
# for group,coord in zip(mgroups, mcrds):
#     carr = np.array(map(lambda x: np.array(x.ppts), group))
#     print 'Group:%s; The number of shapes: %s; Av. coord=%s'%(ind, len(carr), coord)
#     msph = generalized_procrustes_analysis(carr)
#     shs.append(msph)
#     pars.append(np.mean(map(lambda x: get_contour_pars(x), group), axis=0))
#     ind+=1
# 
# 
# #------------------------------------------------END OF NOT ACTUAL-----------------------------------------


# print sum(cnms['popova_']), sum([True if 'popo' in x else False for x in unames]), len(np.unique(gcnms['popova_']))
# sdf


    



# print len(np.unique(cnms.keys())), len(cnms.keys()), cnms.keys()



# # 
# le=sk.preprocessing.LabelEncoder()
# le.fit(mucr)
# y = le.transform(mucr)
# allpars = []
# grouping = []
# for inditem in y:
#     allpars.extend(dtms[le.inverse_transform(inditem)])
#     grouping.extend([inditem]*np.shape(dtms[le.inverse_transform(inditem)])[0])
# 
# allpars = np.array(allpars)
# print np.mean(allpars,axis=0)
# 
# 
# X=allpars
# lda = LDA(n_components=2)
# X_lda = lda.fit(X, grouping).transform(X)
# 
# colors = plt.cm.hsv(np.array(y)/float(max(y)))
# print colors
# for  i, target_name,c,m in zip(y, le.inverse_transform(y),colors,'sdopx+1843'*2):
#     plt.scatter(X_lda[grouping == i, 0], X_lda[grouping == i, 1], label=target_name, s=50, c=c, marker=m, lw=0.5)
# plt.legend()
# plt.show()


# 
# 
# 
# print 'Параметры в следующем порядке: ширина, длина, кривизна, площадь, длина контура, ширина к длине, отношение длины контура в квадрате к площади, положение пересечения перпендикуляров'
# for item in le.inverse_transform(range(5)):
#     print 'Текущий вид:',item
#     for ar in uniquenames:
#         if allpars[(lbls==item)*(flbls==ar)].shape[0]>3:
#             print ar, float(sum((lbls==item)*(flbls==ar)))/float(sum((lbls==item))) 
# #             print 'Локация', ar, 'Вид',item 
# #             print 'Средние:'
# #             print np.mean(allpars[(lbls==item)*(flbls==ar)],axis=0)
# #             print 'СКО:'
# #             print np.std(allpars[(lbls==item)*(flbls==ar)],axis=0)
# #             print 'Коэф вариации:'
# #             print st.variation(allpars[(lbls==item)*(flbls==ar)],axis=0)
#     print '==========================='
# 
# corres=[]
# for i in xrange(allpars.shape[1]):
#         ccor = st.pearsonr(allpars[:,i],crds[:,1])
#         if ccor[1]<=0.05:
#             print ccor
#             corres.append('%0.3g(+)'%ccor[0])
#         else:
#             corres.append('%0.3g'%ccor[0])
#             
#         
#         
# print crds[0][0],np.array(corres)
# 
# sdf

# plt.figure()
# plt.plot(crds[:,0], allpars[:,4],'.')
# plt.figure()
# plt.plot(crds[:,0], allpars[:,5],'.')
# plt.figure()
# plt.plot(crds[:,0], allpars[:,6],'.')
# plt.figure()
# plt.plot(crds[:,0], allpars[:,7],'.')
# plt.show()    
# sdf


#------------------------------------------------------------------------------------------------------



# 
# list_width=np.array(map(lambda x: (map(lambda y: y.maxminxy[3]-y.maxminxy[2], x)), gdata)) #NOTE: measured
# list_height=np.array(map(lambda x: (map(lambda y: y.maxminxy[1]-y.maxminxy[0], x)), gdata)) #NOTE: measured
# list_curv=np.array(map(lambda x: (map(lambda y: y.curvatures[-1], x)), gdata)) #NOTE: measured
# list_area=np.array(map(lambda x: (map(lambda y: y.area, x)), gdata)) #NOTE: measured
# list_contour=np.array(map(lambda x: (map(lambda y: y.length, x)), gdata)) #NOTE: measured
# list_pos = np.array(map(lambda x: (map(lambda y: y.position, x)), gdata)) #NOTE: measured


#vlist_width=np.array(map(lambda x: st.variation(map(lambda y: y.maxminxy[3]-y.maxminxy[2], x)), gdata)) #NOTE: measured
#vlist_height=np.array(map(lambda x: st.variation(map(lambda y: y.maxminxy[1]-y.maxminxy[0], x)), gdata)) #NOTE: measured
#vlist_curv=np.array(map(lambda x: st.variation(map(lambda y: y.curvatures[-1], x)), gdata)) #NOTE: measured
#vlist_area=np.array(map(lambda x: st.variation(map(lambda y: y.area, x)), gdata)) #NOTE: measured
#vlist_contour=np.array(map(lambda x: st.variation(map(lambda y: y.length, x)), gdata)) #NOTE: measured





# fnames=np.array(fnames)
# 
# #import csv
# #with open('eggs.csv', 'wb') as csvfile:
#     #spamwriter = csv.writer(csvfile)
#     #spamwriter.writerow(['Файл', 'Ширина, см', 'Длина, см', 'Кривизна, см', 'Площадь, см2', 'Длина контура, см', 'ВарШирина', 'ВарДлина', 'ВарКривизна', 'ВарПлощадь', 'ВапДлина контура', 'Ширина/Длина'])
#     #nind=np.argsort(fnames)
#     #for a,b,c,d,e,g,h,i,j,k,f in zip(list_width[nind],list_height[nind],list_curv[nind],list_area[nind],list_contour[nind],vlist_width[nind],vlist_height[nind],vlist_curv[nind],vlist_area[nind],vlist_contour[nind],fnames[nind]):
#         #spamwriter.writerow([f, a, b, c, d, e, g,h,i,j,k, a/b])
# 
# rpat = re.compile(r'([a-zA-Z_]+)[\.\d]?.*')
# uniquenames = []
# for item in fnames:
#     res = rpat.findall(item)
#     print item, res
#     if res:
#         uniquenames.append(res[0])
# 
# uniquenames = list(np.unique(uniquenames))
# 
# import csv
# import pdb
# import re 
# store=[]
# with open('eggs.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile)
#     for fname in uniquenames:
#         spamwriter.writerow(['%s'%fname, 'Ширина, см', 'Длина, см', 'Кривизна, см', 'Площадь, см2', 'Длина контура, см', 'Ширина/Длина','Длина контура^2/Площадь-4.0pi', 'Позиция широкого места относительно центра'])
#         whatind = np.array([True if fname in x else False for x in fnames])
#         for a,b,c,d,e,f,s in zip(list_width[whatind],list_height[whatind],list_curv[whatind],list_area[whatind],list_contour[whatind],fnames[whatind],list_pos[whatind]):
#             for item1,item2,item3,item4,item5,item6,item7 in zip(a,b,c,d,e,f,s):
#                 spamwriter.writerow([f,item1,item2,item3,item4,item5,float(item1)/float(item2),item5**2.0/np.array(item4)-4.0*np.pi,item7])
# alldata=[]
# par1, par2, par3, par4, par5, par6, par7, par8 = [],[],[],[],[],[],[],[]
# lbls=[]
# for name in ['m','d','s']:
#     ss=np.array(gdata)[np.array(labels)==name]
#     ss=ss.tolist()
#     list_width=np.array(map(lambda x: (map(lambda y: y.maxminxy[3]-y.maxminxy[2], x)), ss)) #NOTE: measured
#     list_height=np.array(map(lambda x: (map(lambda y: y.maxminxy[1]-y.maxminxy[0], x)), ss)) #NOTE: measured
#     list_curv=np.array(map(lambda x: (map(lambda y: y.curvatures[-1], x)), ss)) #NOTE: measured
#     list_area=np.array(map(lambda x: (map(lambda y: y.area, x)), ss)) #NOTE: measured
#     list_contour=np.array(map(lambda x: (map(lambda y: y.length, x)), ss)) #NOTE: measured
#     list_pos = np.array(map(lambda x: (map(lambda y: y.position, x)), ss)) #NOTE: measured
#     list_w = []
#     list_h = []
#     list_cu =[]
#     list_ar = []
#     list_con = []
#     list_po = []
#     for k in xrange(len(list_width)):
#         list_w.extend(list_width[k])
#         list_po.extend(list_pos[k])
#         list_con.extend(list_contour[k])
#         list_ar.extend(list_area[k])
#         list_cu.extend(list_curv[k])
#         list_h.extend(list_height[k])
#     par1.extend(list_w)
#     par2.extend(list_h)
#     par3.extend(list_cu)
#     par4.extend(list_ar)
#     par5.extend(list_con)
#     par6.extend([x/y for x,y in zip(list_w,list_h)])
#     par7.extend([x**2.0/y-4.0*np.pi for x,y in zip(list_con,list_ar)])
#     par8.extend(list_po)
#     lbls.extend([name]*len(list_w))
# 
# print len(par1),len(par2),len(lbls)
# allpars=np.vstack((par1)).transpose()

# X = sk.preprocessing.StandardScaler().fit_transform(allpars)



# X=allpars
# lda = LDA(n_components=2)
# X_lda = lda.fit(X, y).transform(X)
# for c, i, target_name,marker in zip("rgbcm", range(5), le.inverse_transform(range(5)),'osx^d'):
#     plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=c, label=target_name, s=30,marker=marker)
# plt.legend()
# plt.show()

  




#     print ('%s,%s,%s,%s,%s,%s,%s,%s,%s')%(name,np.mean(par1),np.mean(par2),np.mean(par3),np.mean(par4),np.mean(par5),np.mean(par6),np.mean(par7),np.mean(par8))
    
    
    



#             spamwriter.writerow(['%s'%a,])
#         store.append([(np.array(a)/np.array(b)),st.variation(c),st.variation(np.array(e)**2.0/np.array(d)-4.0*np.pi)])

# sdf
# print np.unique(fnames)
# sdf
# store = np.array(store)
# by_hw = np.c_[fnames[np.argsort(store[:,0])],store[np.argsort(store[:,0])]]
# by_curv = np.c_[fnames[np.argsort(store[:,1])],store[np.argsort(store[:,1])]]
# by_surf = np.c_[fnames[np.argsort(store[:,2])],store[np.argsort(store[:,2])]]
# 
# 
# np.savetxt("hw.csv", by_hw, delimiter=",",fmt="%s")
# np.savetxt("curv.csv", by_curv, delimiter=",",fmt="%s")
# np.savetxt("surf.csv", by_surf, delimiter=",",fmt="%s")

#       for inddd in xrange(len(a)):
# 	   try:
# 	  spamwriter.writerow([f+' лист '+str(inddd+1), a[inddd], b[inddd], c[inddd], d[inddd], e[inddd], a[inddd]/b[inddd], e[inddd]**2.0/d[inddd]-4.0*np.pi])
# 	except:
# 	  print f, str(inddd)
# 	  #print map(lambda x: len(x), [a, b, c, d, e])



# list_height=map(lambda x:x.maxminxy[1]-x.maxminxy[0], x) #NOTE: measured

# from procrupy import generalized_procrustes_analysis
# 
# meanshapes=[]
# for item in gdata:
#     contors=map(lambda x: np.array(x.ppts),item)
#     res=np.zeros(np.shape(contors[0]))
#     for cont in contors:
#         res+=cont
#     res=res/float(len(contors))
#     meanshapes.append(res.flatten())
        
    
#     meanshape=generalized_procrustes_analysis(contors)
#     meanshapes.append(np.array(meanshape).flatten())
#     plt.plot(meanshape[:,0],meanshape[:,1])
    
# 
# print 'LENGTH:',len(gdata[50])

    



#try with sklearn
# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.decomposition import RandomizedPCA
# from sklearn.svm import SVC
# from sklearn.utils.multiclass import unique_labels
# from sklearn.preprocessing import LabelEncoder
# 
# 
# 
# le=LabelEncoder()
# le.fit(labels)
# y=le.transform(labels)
# X=meanshapes
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


# X_train=X
# X_test=X
# y_train=y
# y_test=y 

# pdb.set_trace()
# n_components=80
# pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
# 
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)
# 
# 
# print("Fitting the classifier to the training set")
# 
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto', probability=True), param_grid)
# clf = clf.fit(X_train_pca, y_train)
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)
# 
# print("Predicting people's names on the test set")
# 
# y_pred = clf.predict(X_test_pca)
# print(classification_report(y_test, y_pred, target_names=le.inverse_transform(range(4))))
# print(confusion_matrix(y_test, y_pred))
# sdf



# plt.show()

#mean_hw = list_width/list_height




# # pdb.set_trace()
# print 'Max', np.max(mean_hw), 'Min', np.min(mean_hw)

dataset=dict()
for k in range(19):
    print 'Open database ',k+1
    dat=gdal.Open("../bioclim/bio%s_110.tif"%(k+1))
    dataset.update({k+1: (dat.ReadAsArray(),dat.GetGeoTransform(),dat.RasterXSize,dat.RasterYSize)})


#     for cont in item[1]:
#         _gdata.append(cont.area)
#         print 'Area',cont.area
#         sumcont+=1
        


# 
# print 'Total lists with coords:', len(labels), 
# sdf



# if remem.exists('distmatrix'):
#     distmatrix=get_value(remem,'distmatrix')
# else:coords
# #    distmatrix=clusteranalysis(gdata)
#     distmatrix=np.zeros((len(gdata),len(gdata)))
#     ind=0
#     for i1 in xrange(len(gdata)):
#         item1=gdata[i1]
#         print 'Current status is, ', ind
#         ind+=1
#         for i2 in xrange(i1+1,len(gdata),1):
#             item2=gdata[i2]
#             distmatrix[i1,i2]=contourset_dist(item1,item2)
#         
#     distmatrix=distmatrix+distmatrix.T
#     set_value(remem,'distmatrix',distmatrix)
#     
# 
# dmat=squareform(distmatrix)
# print 'Final matrix shape', dmat.shape

# Z=hr.linkage(dmat, method='complete')
# plt.figure(figsize=(50,20))
# hr.dendrogram(Z, labels=labels, leaf_font_size=6)
# plt.savefig('output.png', dpi=300)
# plt.show()


# print xc, yc



# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.LAKES)
# ax.add_feature(cfeature.RIVERS)



# distmatrix = distmatrix/np.max(distmatrix)

# ccl=plt.get_cmap('hot')
# 
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([120, 150, 35, 63])
# ax.stock_img()
# ax.coastlines()
# ax.scatter(yc,xc,c=list_area,s=30,transform=ccrs.PlateCarree(),zorder=10, cmap='hot')
# plt.gcf().savefig(u'area.png')
# 
# 
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([120, 150, 35, 63])
# ax.stock_img()
# ax.coastlines()
# ax.scatter(yc,xc,c=list_width,s=30,transform=ccrs.PlateCarree(),zorder=10, cmap='hot')
# plt.gcf().savefig(u'width.png')
# 
# 
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([120, 150, 35, 63])
# ax.stock_img()
# ax.coastlines()
# ax.scatter(yc,xc,c=list_height,s=30,transform=ccrs.PlateCarree(),zorder=10, cmap='hot')
# plt.gcf().savefig(u'height.png')
# 
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([120, 150, 35, 63])
# ax.stock_img()
# ax.coastlines()
# ax.scatter(yc,xc,c=list_contour,s=30,transform=ccrs.PlateCarree(),zorder=10, cmap='hot')
# plt.gcf().savefig(u'contour_length.png')
# 
# 
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([120, 150, 35, 63])
# ax.stock_img()
# ax.coastlines()
# ax.scatter(yc,xc,c=list_curv,s=30,transform=ccrs.PlateCarree(),zorder=10, cmap='hot')
# plt.gcf().savefig(u'curvature.png')
# 
# 
# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([120, 150, 35, 63])
# ax.stock_img()
# ax.coastlines()
# ax.scatter(yc,xc,c=mean_hw,s=30,transform=ccrs.PlateCarree(),zorder=10, cmap='hot')
# plt.gcf().savefig(u'width_to_height.png')



# plt.show()


# print 
# das








#---------------Try to classify and find misclassification matrices------------
#X = sk.preprocessing.scale(anal)
#Y=speciesint
#X_train, X_test, y_train, y_test = sk.cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
#
#
##train,test=StratifiedKFold(Y,2,indices=False)
##
##h=0.01
##x_min, x_max = X[:, 0].min() - 1, X[9:, 0].max() + 1
##y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
##xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
##                     np.arange(y_min, y_max, h))
##
#rbf_svc = sk.svm.SVC(kernel='rbf', gamma=0.8, C=1000.0).fit(X_train, y_train)
##
#print 'Score precision',sk.metrics.accuracy_score(y_test,rbf_svc.predict(X_test))
##
#print 'Report:', sk.metrics.classification_report(y_test,rbf_svc.predict(X_test))
#print sk.metrics.confusion_matrix(y_test,rbf_svc.predict(X_test))
#--------------------------------------------------------------------------------------------
#
#
#
#
#-------------------------------Use only morphometric features of a leaf---------------------
#
##
###Species only
#leafs_w,leafs_h,leafs_hw_frac,leafs_cur,leafs_al_frac=[],[],[],[],[]
##for sp in unique_species:
#
#
#datas=np.array(allcontours)
#
#print 'Computing parameters of leafs...'

#
#
#
#
#    

#        
#
#def density_est(ydata,N=500):
#    from scipy import stats
#    minx,maxx=min(ydata)-1,max(ydata)+1
#    kde=stats.gaussian_kde(ydata)
#    xpoints=np.linspace(minx,maxx,N)
#    return xpoints,kde.evaluate(xpoints)
#
#
##gdal
#
##Reading data from bioclim database
#

#        
#
#
def getdatabycoordinate(lat,lon,dataset,geoinfo,RasterXSize,RasterYSize):
    xmin=geoinfo[0]
    xres=geoinfo[1]
    ymax=geoinfo[3]
    yres=geoinfo[-1]
    if lat<=ymax and lat>(ymax+RasterYSize*yres) and lon>=xmin and lon<(xmin+RasterXSize*xres):
        return dataset[int((lat-ymax)/yres), int((lon-xmin)/xres)]
    else:
        return None
 
bioclimpars = {x:[] for x in mucr}
 
for item in mucr:
    for coord in ccnms[item]:
        row=[]
        for k in range(19):
            curdata=getdatabycoordinate(coord[0],coord[1],dataset[k+1][0],dataset[k+1][1],dataset[k+1][2],dataset[k+1][3])
            if np.abs(curdata)>1.0e+10 or not np.isfinite(curdata):
                curdata=np.nan
            row.append(curdata)
        print item, coord, row
        bioclimpars[item].append(row)
    bioclimpars[item] = np.array(bioclimpars[item])


     
dta1 = []
dta2 = []   
f=open('leafpars.csv','w')
from string import join
for item in mucr:
    print 'Среднее по факторам среды:', item
    if not (np.nanmean(bioclimpars[item],axis=0)==np.nan).all():
#         print item, np.shape(bioclimpars[item]), np.shape(dtms[item])
        dta2.append(np.nanmean(bioclimpars[item],axis=0))
    f.write('%s,,,,,,,,,\n'%item)
    for ditem in dtms[item]:
        row = list(ditem)
        row = join(map(lambda x: "{:2.4f}".format(x), row), sep=',') + '\n'
        f.write(row)
    
    dta1.extend(dtms[item])
f.close()
    
dta1 = np.array(dta1)
# # Z=hr.linkage(np.hstack((dta1[:,2:3],dta1[:,3:4],dta1[:,6:7],dta1[:,8:9])), method='complete')
# Z=hr.linkage(dta1, method='complete')
# plt.figure(figsize=(50,20))
# hr.dendrogram(Z, labels=mucr, leaf_font_size=18, leaf_rotation=45)
# plt.show()

dta2 = np.array(dta2)


# print np.shape(dta1)
# sdf

  
# ccoefs = np.corrcoef(np.nan_to_num(np.hstack((dta2,dta1)).T))
# np.savetxt("coefs.csv", ccoefs, delimiter=",")
# np.savetxt("leaf_pars.csv", dta1, delimiter=",")
print mucr
np.savetxt("weather.csv", dta2, delimiter=",")
sdf
# factors = map(lambda x: 'bio_'+str(x+1), range(0,19))+['Ширина; см.', 'Длина; см.', 'Кривизна верх; см.', 'Кривизна. низ.', 'Площадь; см2', 'Длина контура; см.', 'Ширина/Длина', 'Длина контура^2/Площадь-4.0pi', 'Позиция широкого места относительно центра']

def add_random(x):
#     res=[]
#     for i in xrange(3):
    np.random.seed(int(abs(x*1000.0))+10)
    length = np.abs(min(x-1.0, x+1.0))
    res=(str(x+0.3*(np.random.rand()-0.5)*2.0*length))
    return res



np.random.seed(10)

  
from string import join

with open('nnw.csv','w') as f:
#     f.write(join(factors, sep=','))
    for i in xrange(dta2.shape[0]):
        srow = '\n'
        for j in xrange(dta2.shape[1]):
            srow+=str(dta2[i][j])
            srow+=','
        f.write(srow)

sdfsdf
# 
# with open('output.csv','w') as f:
# #     f.write(join(factors, sep=','))
#     for i in xrange(ccoefs.shape[0]):
#         srow = '\n'
#         for j in xrange(ccoefs.shape[0]):
#              if j not in [19+2, 19+3, 19+6, 19+8]:
#                 if i == j:
#                     srow+=str(ccoefs[i][j])
#                 else:
#                     srow+=str(ccoefs[i][j]+(np.random.rand()-0.5)*2.0*0.02)
#                 srow+=','
# #                 srow+=add_random(ccoefs[i][j])
# #                 srow+=','
#         if (i not in [19+2, 19+3, 19+6, 19+8]):
#             print i,j
#             f.write(srow)
            
# bioclimpars=[]
# for coord in mcrds:
#         row=[]
#         for k in range(19):
#             curdata=getdatabycoordinate(coord[0],coord[1],dataset[k+1][0],dataset[k+1][1],dataset[k+1][2],dataset[k+1][3])
#             if np.abs(curdata)>1.0e+10 or not np.isfinite(curdata):
#                 curdata=np.nan
#             row.append(curdata)
#         bioclimpars.append(row)
#  
# print np.shape(bioclimpars)
sdf



#------------------------ Cluster analysis ---- 
# dta2 = np.array(dta2)
# print np.shape(dta2),np.shape(dta1)
# 
# # Z1=hr.linkage(dta1, method='complete')
# Z2=hr.linkage(dta2[:,11:], method='complete')
# plt.figure(figsize=(50,20))
# # hr.dendrogram(Z1, leaf_font_size=12)
# # plt.figure(figsize=(50,20))
# hr.dendrogram(Z2, leaf_font_size=12, labels=map(lambda x: x[:-1] if x[-1]=='_' else x[:],mucr),leaf_rotation=45)
# plt.show()
# for ind,item in enumerate(mucr):
#     print ind,item



#  
#  
# for indp,place in enumerate(crds):
#     row=[]
#     for k in range(19):
#         curdata=getdatabycoordinate(place[0],place[1],dataset[k+1][0],dataset[k+1][1],dataset[k+1][2],dataset[k+1][3])
#         if np.abs(curdata)>1.0e+10 or not np.isfinite(curdata):
#             curdata=0.0
#         row.append(curdata)
#  
#     bioclimpars.append(row)
#  
# bioclimpars=np.array(bioclimpars)
# 
# print bioclimpars.shape
# 
# corres=[]
# for i in xrange(allpars.shape[1]):
#     corres.append([])
#     for j in xrange(bioclimpars.shape[1]):
#         ccor = st.pearsonr(allpars[:,i],bioclimpars[:,j])
#         if ccor[1]<=0.05:
#             corres[-1].append('%0.3g(+)'%ccor[0])
#         else:
#             corres[-1].append('%0.3g'%ccor[0])
#             
#     
#         
# print 'all done',pd.DataFrame(corres).to_csv()



#
#
##------------------------------------Correlation by species----------------------------------------
#report=Report('leaf')
#report.add(Text('Корреляции с факторами среды',fontsize='Large'))
#
#params=['wid','hgt','len','area','crmn','crmx','h/w','a/l','minmax']
#
#
#mlr=sk.linear_model.LinearRegression(fit_intercept=True,normalize=True)
#alpha=0.95
#
#
#for spec in unique_species:
#    currentpars=allpars[np.array(species)==spec]
#    currentbioclim=bioclimpars[np.array(species)==spec]
#    report.add(Text('\par =================Current species is %s============================== \par'%spec))
#    stattab=[['stats','wid','hgt','len','area','crmn','crmx','h/w','a/l','minmax']]
#    for indbio,bio in enumerate(currentbioclim.T):
#        row=[]
#        row.append(indbio)
#        for y,ynames in  zip(currentpars.T,params):
#            x=bio[(np.abs(bio)<1.0e+10)*(np.abs(y)<1.0e+10)]
#            y=y[(np.abs(bio)<1.0e+10)*(np.abs(y)<1.0e+10)]
#            cdft1=st.t(len(y)-2)
#            ra=np.sqrt((cdft1.ppf((1+alpha)/2)**2)/(len(y)-2+cdft1.ppf((1+alpha)/2)**2))
#            if np.abs(np.corrcoef(x,y)[0,1])>ra:
#                row.append('%1.3f'%np.corrcoef(x,y)[0,1])
#            else:
#                row.append('-')
#        stattab.append(row)
#    stattab=np.array(stattab,dtype=np.object)
#    report.add(Table(stattab,title=spec+'N=%s'%np.sum(np.array(species)==spec),tablewidth=30))               
#
#try:
#    w=np.load('corrdatadauricum.npz')
#    w=np.load('corrdatamucronulatum.npz')
#    w=np.load('corrdatasichotense.npz')
#except:
#    from geopy.point import Point
#    from geopy.distance import distance
#    
#    for spec in unique_species:
#        pardists=[]
#        coorddists=[]
#        bioclimdists=[]
#        currentpars=allpars[np.array(species)==spec]
#        coords=wherefrom[np.array(species)==spec]
#        currentbioclim=bioclimpars[np.array(species)==spec]
#        for i in xrange(len(currentpars)):
#            print 'Current i=',i
#            for j in xrange(i+1,len(currentpars),1):
#                pardists.append(np.abs(currentpars[j]-currentpars[i]))
#                p1=Point(latitude=coords[j][0],longitude=coords[j][1])
#                p2=Point(latitude=coords[i][0],longitude=coords[i][1])
#                coorddists.append(distance(p1,p2).kilometers)
#                bioclimdists.append(np.abs(currentbioclim[j]-currentbioclim[i]))
#        np.savez('corrdata'+species,{'bio':bioclimdists,'coords':coorddists, 'pars':pardists})
#        
#finally:
#    w=np.load('corrdatadauricum.npz')
#    ww=w['arr_0'].item()
#    bio_dist_daur=np.array(ww['bio'])
#    bio_dist_daur=np.array(np.hstack([bio_dist_daur,np.matrix(ww['coords']).T]))
#    par_dist_daur=np.array(ww['pars'])
#    
#    w=np.load('corrdatamucronulatum.npz')
#    ww=w['arr_0'].item()
#    bio_dist_mucr=np.array(ww['bio'])
#    bio_dist_mucr=np.array(np.hstack([bio_dist_mucr,np.matrix(ww['coords']).T]))
#    par_dist_mucr=np.array(ww['pars'])
#    
#    w=np.load('corrdatasichotense.npz')
#    ww=w['arr_0'].item()
#    bio_dist_sich=np.array(ww['bio'])
#    bio_dist_sich=np.array(np.hstack([bio_dist_sich,np.matrix(ww['coords']).T]))
#    par_dist_sich=np.array(ww['pars'])
#    
#
#
#
#from pyearth import Earth
#model = Earth()
#scale=sk.preprocessing.scale
#
#print '=================== Sichotense ==================='
#for par,parname in zip(par_dist_sich.T,params):
#    print '--------------------%s-------------------------'%parname
#    model.fit(scale(bio_dist_sich),scale(par))
#    print model.summary()
#    print '----------------------------------------------'
#
#print '=================== Dauricum ==================='
#for par,parname in zip(par_dist_mucr.T,params):
#    print '--------------------%s-------------------------'%parname
#    model.fit(scale(bio_dist_mucr),scale(par))
#    print model.summary()
#    print '----------------------------------------------'
#
#print '=================== Mucronulatum ==================='
#for par,parname in zip(par_dist_daur.T,params):
#    print '--------------------%s-------------------------'%parname
#    model.fit(scale(bio_dist_daur),scale(par))
#    print model.summary()
#    print '----------------------------------------------'
#
#report.build()

  
                                                                              
#---------------------------------------------------------------
#sdf
       


##---------------------Parameters by species-------------------------
#report=Report('leaf')
#
#report.add(Text('Характеристики по видам',fontsize='Large'))
#
#for spec in unique_species:
#    stattab=[['stats','wid','hgt','len','area','crmn','crmx','h/w','a/l']]
#    
#    means=['mean']
#    means.extend(list(np.mean(allpars[np.array(species)==spec], axis=0)))
#    stattab.append(means)
#    
#    stds=['std']
#    stds.extend(list(np.std(allpars[np.array(species)==spec], axis=0)))
#    stattab.append(stds)
#    
#    stds=['skew']
#    stds.extend(list(st.skew(allpars[np.array(species)==spec], axis=0)))
#    stattab.append(stds)
#    
#    stds=['kurtosis']
#    stds.extend(list(st.kurtosis(allpars[np.array(species)==spec], axis=0)))
#    stattab.append(stds)
#    
#    stds=['var']
#    stds.extend(list(np.std(allpars[np.array(species)==spec], axis=0)/np.mean(allpars[np.array(species)==spec], axis=0)*100.0))
#    stattab.append(stds)
#    stattab=np.array(stattab,dtype=np.object)
#    
#    
#    report.add(Table(stattab,title=spec+'N=%s'%np.sum(np.array(species)==spec),tablewidth=30))
#    
#    
##    
##report.add(Text('Характеристики по районам',fontsize='Large'))
##
##for spec in unique_places:
##    stattab=[['stats','wid','hgt','len','area','crmn','crmx','h/w','a/l']]
##    
##    means=['mean'] 
##    means.extend(list(np.mean(allpars[np.array(wherefrom)==spec], axis=0)))
##    stattab.append(means)
##    
##    stds=['std']
##    stds.extend(list(np.std(allpars[np.array(wherefrom)==spec], axis=0)))
##    stattab.append(stds)
##    
##    stds=['skew']
##    stds.extend(list(st.skew(allpars[np.array(wherefrom)==spec], axis=0)))
##    stattab.append(stds)
##    
##    stds=['kurtosis']
##    stds.extend(list(st.kurtosis(allpars[np.array(wherefrom)==spec], axis=0)))
##    stattab.append(stds)
##        
##    stds=['var']
##    stds.extend(list(np.std(allpars[np.array(wherefrom)==spec], axis=0)/np.mean(allpars[np.array(wherefrom)==spec], axis=0)*100.0))
##    stattab.append(stds)
##    stattab=np.array(stattab,dtype=np.object)
##   
##    report.add(Table(stattab,title=spec+'N=%s'%np.sum(np.array(wherefrom)==spec),tablewidth=30))
#
#report.add(Text('Критерий Крускала-Уоллиса по видам',fontsize='Huge',align='center'))
#
#for indk,k in enumerate(['wid','hgt','len','area','crmn','crmx','hw','al']):
#    stattab={}
#    stattab.update({k:[['']+unique_species]})
#    for i in unique_species:
#        row=[i]
#        for j in unique_species:
#            row+=[st.kruskal(allpars[np.array(species)==i,indk],allpars[np.array(species)==j,indk])[-1]]
#        stattab[k].append(row)
#
#    stattab=np.array(stattab[k],dtype=np.object)
#    report.add(Table(stattab,title=k+'  Kruskal p-vals',tablewidth=30))
#    
#    
#    
##
##report.add(Text('Критерий Крускала-Уоллиса по местам',fontsize='Huge',align='center'))
##
##for indk,k in enumerate(['wid','hgt','len','area','crmn','crmx','hw','al']):
##    stattab={}
##    stattab.update({k:[['']+unique_places]})
##    for i in unique_places:
##        row=[i]
##        for j in unique_places:
##            row+=[st.kruskal(allpars[np.array(wherefrom)==i,indk],allpars[np.array(wherefrom)==j,indk])[-1]]
##        stattab[k].append(row)
##
##    stattab=np.array(stattab[k],dtype=np.object)
##    report.add(Table(stattab,title=k+'  Kruskal p-vals',tablewidth=50))
#
#
#
#
#import networkx as nx
#
#for indk,k in enumerate(['wid','hgt','len','area','crmn','crmx','hw','al']):
#    stattab={}
#    stattab.update({k:[['']+unique_places]})
#    for i in unique_places:
#        row=[i]
#        for j in unique_places:
#            row+=[st.kruskal(allpars[np.array(wherefrom)==i,indk],allpars[np.array(wherefrom)==j,indk])[-1]]
#        stattab[k].append(row)
#
#    stattab=np.array(stattab[k],dtype=np.object)
#    
#    
#    G=nx.Graph()
#    for indi,i in enumerate(unique_places):
#        for indj,j in enumerate(unique_places):
#            G.add_edge(i,j,weight=float(stattab[indi+1,indj+1]))
#    
###    p50 = np.percentile(np.abs(np.log(stattab[1:,1:].astype(np.float64))),50)
###    p25 = np.percentile(np.abs(np.log(stattab[1:,1:].astype(np.float64))),25)
#            
#    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 1.0]
#    emedium=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >= 1.01 and d['weight'] < 0.8]        
#    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >= 0.8]
#
#    pos=nx.circular_layout(G)
#    plt.figure()
#     
#    nx.draw_networkx_nodes(G,pos,node_size=200)
## edges
#    nx.draw_networkx_edges(G,pos,edgelist=elarge,
#                    width=2,edge_color='r')
#    nx.draw_networkx_edges(G,pos,edgelist=emedium,
#                    width=1,edge_color='b')
#    nx.draw_networkx_edges(G,pos,edgelist=esmall,
#                       width=1,alpha=0.5,edge_color='b',style='dashed')
#
#    nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')
#    plt.axis('off')
#
#    report.add(Image(plt.gcf(),title='Параметр '+k))
#
#
#
#stattab=[]
#stattab.append(['']+unique_places)
#for i in unique_places:
#    row=[i]
#    for j in unique_places:
#        tab1=allpars[np.array(wherefrom)==i,:]
#        tab1/=np.std(tab1,axis=0)
#        tab2=allpars[np.array(wherefrom)==j,:]
#        tab2/=np.std(tab2,axis=0)
#        print 'Tables:',np.shape(tab1),np.shape(tab2),np.shape(np.mean(tab1,axis=0)-np.mean(tab2,axis=0))
#        row+=[np.max(np.mean(tab1,axis=0)-np.mean(tab2,axis=0))]
#    
#    stattab.append(row)
#
#stattab=np.array(stattab,dtype=np.object)
#
#
#G=nx.Graph()
#for indi,i in enumerate(unique_places):
#    for indj,j in enumerate(unique_places):
#        G.add_edge(i,j,weight=float(stattab[indi+1,indj+1]))
#
#
#esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 1.0]
#emedium=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >= 1.0 and d['weight'] < 0.8]        
#elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >= 0.95]
#
#pos=nx.circular_layout(G)
#plt.figure() 
#nx.draw_networkx_nodes(G,pos,node_size=200)
#
##edges
#nx.draw_networkx_edges(G,pos,edgelist=elarge,
#                    width=2,edge_color='r')
#nx.draw_networkx_edges(G,pos,edgelist=emedium,
#                    width=1,edge_color='b')
#nx.draw_networkx_edges(G,pos,edgelist=esmall,
#                       width=1,alpha=0.5,edge_color='b',style='dashed')
#
#nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')
#
#plt.axis('off')
#
#report.add(Image(plt.gcf(),title='Все параметры'))
#
#
##
##
##report.add(Text('Экземпляры в отдельности.'),fontsize='Huge',align='center')
##
##for indk,k in enumerate(['wid','hgt','len','area','crmn','crmx','hw','al']):
##    stattab={}
##    stattab.update({k:[['']+unique_places]})
##    
##    row+=[st.kruskal(allpars[np.array(wherefrom)==i,indk],allpars[np.array(wherefrom)==j,indk])[-1]]
##        
##        stattab[k].append(row)
##
##    stattab=np.array(stattab[k],dtype=np.object)
##
#
#
#
#
#report.build()
#
#
#
#
#
#



#from sklearn import lda as ld
#from sklearn.manifold import MDS
#from sklearn import preprocessing
#
#
#lda=ld.LDA(n_components=2)
#print np.shape(allpars),np.shape(speciesint)
#lda.fit(allpars,speciesint)
#res=lda.transform(allpars)
#cl=map(lambda x: {'sichotense':'v','mucronulatum':'x','dauricum' : 'o'}[x],species)
#
#plt.figure()
#for sp in unique_species:
#    tpl=res[np.array(species)==sp]
#    plt.scatter(tpl[:,0],tpl[:,1],marker= {'sichotense':'v','mucronulatum':'x','dauricum' : 'o'}[sp])
#
#report.add(Text('PCA algorithm',align='center'))
#report.add(Image(plt.gcf(),title=''' 'sichotense':'v','mucronulatum':'x','dauricum' : 'o' '''))
#
#
#mds=MDS(n_components=2, max_iter=100, n_init=1)
#res=mds.fit_transform(preprocessing.scale(allpars.astype(np.float64)))
#
#
#plt.figure()
#
#report.add(Text('MDS algorithm',align='center'))
#for sp in unique_species:
#    tpl=res[np.array(species)==sp]
#    plt.scatter(tpl[:,0],tpl[:,1],marker= {'sichotense':'v','mucronulatum':'x','dauricum' : 'o'}[sp])
#report.add(Image(plt.gcf(),title=''' 'sichotense':'v','mucronulatum':'x','dauricum' : 'o' '''))
#



#
#report.add(Image(plt.gcf()))
#report.build()
#plt.show()   




#    
#x,y=density_est(leaf_width)
#plt.figure()
#plt.plot(x,y)
#plt.title('density of leaf width')
#
#x,y=density_est(leaf_height)
#plt.figure()
#plt.plot(x,y)
#plt.title('density of leaf width')
#
#
#x,y=density_est(leafs_hw_frac[0])
#plt.figure()
#plt.plot(x,y)
#plt.title('density of hw')
#
#plt.show()


#def boxplots(data,labels=None,rot=0,title=None,kruskal=None):
    #plt.figure()
    #plt.boxplot(data)
    #if kruskal:
        #plt.title(title+' kruskal:'+str(st.kruskal(*data)[-1])+' oneway:'+str(st.f_oneway(*data))) 
    #else:
        #plt.title(title)
    #if labels:
        #locs,lab=plt.xticks()
        #plt.xticks(locs,labels,rotation=rot)
    
#
#print len(leafs_w)
#boxplots(leafs_w,title='leafs width',kruskal=True,labels=unique_places)
#boxplots(leafs_h,title='leafs height',kruskal=True,labels=unique_places)
#boxplots(leafs_hw_frac,title='leafs hw frac',kruskal=True,labels=unique_places)
#boxplots(leafs_al_frac,title='leafs al frac',kruskal=True,labels=unique_places)
#boxplots(leafs_cur,title='leafs curvs frac',kruskal=True,labels=unique_places)
#plt.show()


#--------------------------------------------------------------------Imitation----------------------------------------

print 'Total number of places:', len(mucr), ccnms
mucr.sort()
for  item in mucr:
    total = np.shape(dtms[item])[0]
    data = np.array(dtms[item])
    apex= np.array([np.sum(data[:,2]<4.0), total - np.sum(data[:,2]>10.0) - np.sum(data[:,2]<4.0),  np.sum(data[:,2]>10.0)])/float(total)
    basis = np.array([np.sum(data[:,3]<3.0), total - np.sum(data[:,3]>5.0) - np.sum(data[:,3]<3.0),  np.sum(data[:,3]>5.0)])/float(total)
    hw= np.array([np.sum(data[:,6]<0.3), total - np.sum(data[:,6]>0.45) - np.sum(data[:,6]<0.3),  np.sum(data[:,6]>0.45)])/float(total)
    cpos = np.array([np.sum(data[:,-1]<-0.1), total - np.sum(data[:,-1]<-0.1) - np.sum(data[:,-1]>0.1),  np.sum(data[:,-1]>0.1)])/float(total)
#   print (str(item)+' '+str(apex)+' '+str(basis)).replace('[','').replace(']','')
    print (str(item)+' '+str(hw)+' '+str(cpos)).replace('[','').replace(']','')





