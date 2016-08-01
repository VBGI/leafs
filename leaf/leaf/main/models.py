from django.db import models

from geoposition.fields import GeopositionField
# Create your models here.
from django.utils.encoding import python_2_unicode_compatible

@python_2_unicode_compatible
class LeafData(models.Model):
    CHOICES = (('m', 'Mucronulatum'),
               ('s', 'Sichotense'),
               ('d', 'Dauricum'))
    xdata = models.TextField(blank=True, editable=False)
    ydata = models.TextField(blank=True, editable=False)
    collected = models.DateField(blank=False, null=True)
    species = models.CharField(blank=False, default='', choices=CHOICES, max_length=1)
    approved = models.BooleanField(default=False)
    source1 = models.ImageField(blank=True, null=True)
    source2 = models.ImageField(blank=True, null=True)
    leafcont = models.ImageField(blank=True, null=True)
    filename = models.CharField(max_length=50, blank=True, default='')
    where = GeopositionField()
    
    def __str__(self):
        return self.species+':'+'%s'%self.collected+':'+'%s'%self.approved
   
    def image_tag1(self):
        return u'<img src="%s" width="800" />' % self.source1.url
    image_tag1.short_description = 'Image1'
    image_tag1.allow_tags = True
   
    def image_tag2(self):
        return u'<img src="%s" width="800" />' % self.source2.url
    image_tag2.short_description = 'Image2'
    image_tag2.allow_tags = True
    
    def lfcont(self):
        return u'<img src="%s" width="800" />' % self.leafcont.url
    lfcont.short_description = 'Contour'
    lfcont.allow_tags = True
    
    def smcont(self):
        return u'<img src="%s" width="200" />' % self.leafcont.url
    smcont.short_description = 'Contour'
    smcont.allow_tags = True