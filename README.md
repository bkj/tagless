## taglas

Tagging interface w/ linearized active search (github.com/bkj/simple_las) and uncertainty samplign

### Usage

```

    cd $TARGET_DIR
    mkdir -p ./{data,results}
    # expect set of images to be in `imgs` directory
    
    # Featurize images
    find ./imgs/ -type f | python -m tdesc --model vgg16 --crow > .crow
    python $TAGLESS_ROOT/utils/prep.py --inpath ../data/crow-feats ./data/crow
    
    # Run server
    python -m tagless --outpath ./results/run-v0 --crow ./data/crow
    
    # Connect to localhost:5000
```
