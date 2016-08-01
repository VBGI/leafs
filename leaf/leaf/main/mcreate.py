from .models import LeafData


import csv
from  datetime import datetime
import re
import os 
from geoposition import Geoposition
from decimal import Decimal
from django.core.files import File
import matplotlib.pyplot as plt

def _get_data_from_filename(filename):
    datepat = re.compile(r'.+\_(?P<day>\d\d)\_(?P<month>\d\d)\_(?P<year>\d\d\d\d).+')
    match = datepat.match(filename)
    return (match.group('day'), match.group('month'), match.group('year')) if match else None

def to_decimal(s):
    try:
        res = Decimal(s)
    except:
        res = Decimal()
    return res
    
with open('eggs.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        ld = LeafData.objects.create()
        cdate = _get_data_from_filename(row[0])
        if cdate:
            try:
               ld.collected = datetime(int(cdate[2]), int(cdate[1]), int(cdate[0]))
            except:
               pass
        ld.filename = row[0]
        ld.species = row[1]
        ld.where = Geoposition(to_decimal(row[2]), to_decimal(row[3]))
        srcs = row[4].split(',')
        if len(srcs) > 1:
            with open(srcs[0], 'r') as f:
                fd = File(f)
                ld.source1.save(os.path.basename(srcs[0]),fd, save=True)
            with open(srcs[1], 'r') as f:
                fd = File(f)
                ld.source2.save(os.path.basename(srcs[1]),fd, save=True)
        elif len(srcs) == 1 and len(srcs[0]) > 0:
            with open(srcs[0], 'r') as f:
                fd = File(f)
                ld.source1.save(os.path.basename(srcs[0]),fd, save=True)
        ld.xdata = row[5]
        ld.ydata = row[6]
        with open('leafcont%s.png'%ld.pk,'w+') as inpf:
            f = plt.figure()
            ax = f.add_subplot('111')
            x = map(lambda x: float(x), row[5].split(','))
            y = map(lambda x: float(x), row[6].split(','))
            ax.plot(x,y)
            ax.set_aspect('equal')
            ax.grid('on')
            f.savefig(inpf, dpi=200)
            plt.close(plt.gcf())
            inpf.seek(0)
            ld.leafcont.save('leafcont%s.png'%ld.pk, File(inpf), save=True)
        ld.save()
       