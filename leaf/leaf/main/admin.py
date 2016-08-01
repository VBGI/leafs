from django.contrib import admin

# Register your models here.

from .models import LeafData


class LeafDataAdmin(admin.ModelAdmin):
    list_filter = ('approved',)
    list_display = ('species', 'approved', 'collected', 'where', 'smcont')
    fields = ('species','where','collected','lfcont','source1','image_tag1','source2','image_tag2', 'approved')
    readonly_fields = ('image_tag1','image_tag2', 'lfcont')
    


admin.site.register(LeafData, LeafDataAdmin)