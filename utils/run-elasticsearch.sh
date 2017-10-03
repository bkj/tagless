#!/bin/bash

# run-elasticsearch.sh
#
# Rough example of how to use the Elasticsearch version

python -m tagless \
    --sampler elasticsearch \
    --filenames fnames \
    --img-dir /srv/e2/instagram/trickle/images/ \
    --labels 'lax,nyc,no_label' \
    --require-authentication 