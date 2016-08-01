#!/bin/bash
 
NAME="bgigun_app" # Name of the application
DJANGODIR=/home/scidam/webapps/leafcont/leaf # Django project directory
ACTIVATE_DIR=/home/scidam/webapps/leafcont/env/bin
PORT=24961
#SOCKFILE=/webapps/hello_django/run/gunicorn.sock # we will communicte using this unix socket

#USER=hello # the user to run as
#GROUP=webapps # the group to run as
NUM_WORKERS=1 # how many worker processes should Gunicorn spawn
NUM_THREADS=1
DJANGO_SETTINGS_MODULE=leaf.settings # which settings file should Django use
DJANGO_WSGI_MODULE=leaf.wsgi # WSGI module name
TIMEOUT=120

echo "Starting $NAME as `leaf_app`"
# Activate the virtual environment
cd $ACTIVATE_DIR

source activate
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
export PYTHONPATH=$DJANGODIR:$PYTHONPATH

# Start your Django Unicorn
# Programs meant to be run under supervisor should not daemonize themselves (do not use --daemon)
exec ../bin/gunicorn ${DJANGO_WSGI_MODULE}:application \
     --name $NAME \
     --workers $NUM_WORKERS \
     --bind=localhost:$PORT \
     --log-level=info \
     --worker-class=gevent \
     --max-requests 50 \
     --max-requests-jitter 40 \
     --threads $NUM_THREADS \
     --timeout $TIMEOUT
     
