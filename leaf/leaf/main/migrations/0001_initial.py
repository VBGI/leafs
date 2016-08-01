# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import geoposition.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LeafData',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('xdata', models.TextField(editable=False, blank=True)),
                ('ydata', models.TextField(editable=False, blank=True)),
                ('collected', models.DateField(null=True)),
                ('species', models.CharField(default=b'', max_length=1, choices=[(b'm', b'Mucronulatum'), (b's', b'Sichotense'), (b'd', b'Dauricum')])),
                ('approved', models.BooleanField(default=False)),
                ('source1', models.ImageField(null=True, upload_to=b'', blank=True)),
                ('source2', models.ImageField(null=True, upload_to=b'', blank=True)),
                ('leafcont', models.ImageField(null=True, upload_to=b'', blank=True)),
                ('filename', models.CharField(default=b'', max_length=50, blank=True)),
                ('where', geoposition.fields.GeopositionField(max_length=42)),
            ],
        ),
    ]
