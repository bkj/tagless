#!/bin/bash

source cmds.sh
docker build -t tagless .

# --
# Run workflow

IMG_DIR="/home/ubuntu/_data0"
FEAT_DIR="$(pwd)/feats"
RUN_ID="test"

tagless_featurize        $IMG_DIR $FEAT_DIR $RUN_ID
tagless_label_server     $IMG_DIR $FEAT_DIR $RUN_ID
tagless_inference_server $IMG_DIR $FEAT_DIR $RUN_ID

# --
# test inference server

curl -X POST localhost:6000/inference -H "Content-Type: application/octet-stream" --data-binary @'./gun.jpg'
curl -X POST localhost:6000/inference -H "Content-Type: application/octet-stream" --data-binary @'./forest.jpg'
curl -X POST localhost:6000/inference -H "Content-Type: application/octet-stream" --data-binary @'./deer.jpg'