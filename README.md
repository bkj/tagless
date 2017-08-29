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

### Notes

Uncertainty sampling computes the score for each unlabeled image at each iteration.  ATM we're using a linear SVM, so the runtime of this step increases linearly w/ the size of the corpus.  On my machine, predicting on ~350K images takes ~2.5s, which is unacceptably slow.  Thus, for big corpora, we may want to fall back to some kind of approximate matrix-vector product. That'll take a little bit of thought thought.  For now I'll recommend running on a subset of the data.

__Idea__: Feature vectors are normalized relus -- so norm=1 and all positive entries.  Could maybe do uncertainty sampling via `faiss` by using vector orthogonal to SVM feature vector and take the largest/smallest entries.  Have to check my work on that one though.