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
    python $TAGLESS_ROOT/tagless/prep.py --inpath .crow ./data/crow
    
    # Run server
    python -m tagless --outpath ./results/my-labels --crow ./data/crow
    
    # Connect to localhost:5000 + start tagging
```

### Notes

Uncertainty sampling computes the score for each unlabeled image at each iteration.  ATM we're using a linear SVM, so the runtime of this step increases linearly w/ the size of the corpus.  On my machine, predicting on ~350K images takes ~2.5s, which is unacceptably slow.  Thus, for big corpora, we may want to fall back to some kind of approximate matrix-vector product. That'll take a little bit of thought thought.  For now I'll recommend running on a subset of the data.

__Idea__: Feature vectors are normalized relus -- so norm=1 and all positive entries.  Could maybe do uncertainty sampling via `faiss` by using vector orthogonal to SVM feature vector and take the largest/smallest entries.  Have to check my work on that one though.

### Dependencies

This has been tested on Ubuntu 16.04 w/ Python 2.7
