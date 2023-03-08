
function tagless_featurize {
    IMG_DIR=$1
    FEAT_DIR=$2
    RUN_ID=$3
    docker run \
        --gpus all --ipc=host \
        --mount type=bind,source=$IMG_DIR,target=/imgs \
        --mount type=bind,source=$FEAT_DIR,target=/feats \
        -it tagless \
            python -m tagless.featurize.clip_featurize --run_id $RUN_ID
}

function tagless_label_server {
    IMG_DIR=$1
    FEAT_DIR=$2
    RUN_ID=$3
    docker run \
        --gpus all --ipc=host -p 5000:5000 \
        --mount type=bind,source=$IMG_DIR,target=/imgs \
        --mount type=bind,source=$FEAT_DIR,target=/feats \
        -it tagless \
            python -m tagless.label_server --indir /feats/$RUN_ID
}

function tagless_inference_server {
    IMG_DIR=$1
    FEAT_DIR=$2
    RUN_ID=$3
    docker run \
        --gpus all --ipc=host -p 6000:6000 \
        --mount type=bind,source=$IMG_DIR,target=/imgs \
        --mount type=bind,source=$FEAT_DIR,target=/feats \
        -it tagless \
            python -m tagless.inference_server --clf_dir /feats/$RUN_ID
}