## tagless

Tagging interface w/ transfer learning, linearized active search and uncertainty sampling:

|                   |                              | 
| ----------------- | ---------------------------- |
| Transfer learning | https://github.com/bkj/tdesc |
| Linearized active search (LAS) |  https://github.com/bkj/simple_las | 
| Uncertainty sampling | https://github.com/bkj/libact | 

Under active development -- some things are broken or don't have sensible APIs exposed.

### Usage

```

    cd $TARGET_DIR
    mkdir -p ./{data,results}
    # expect set of images to be in `imgs` directory
    
    # Featurize images
    find ./imgs/ -type f | python -m tdesc --model vgg16 --crow > .crow
    python $TAGLESS_ROOT/utils/prep.py --inpath ../data/crow-feats ./data/crow
    
    # Run server
    python -m tagless --outpath ./results/my-labels --crow ./data/crow
    
    # Connect to localhost:5000 + start tagging
```
