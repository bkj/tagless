#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

import os
import sys
import json
import atexit
import argparse
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from flask import Flask, Response, request, abort, \
    render_template, send_from_directory, jsonify

from simple_las import SimpleLAS

app = Flask(__name__)

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crow', type=str, default='./.crow')
    parser.add_argument('--seeds', type=str, default='./.crow-seeds')
    parser.add_argument('--no-seeds', action='store_true')
    parser.add_argument('--img-dir', type=str, default='./')
    
    parser.add_argument('--alpha', type=float, default=1e-6)
    
    return parser.parse_args()

args = parse_args()

# --
# Helpers

def init_las(crow, seeds=None):
    
    global labs
    
    if seeds:
        print >> sys.stderr, 'init_las: seeds'
        feats = np.vstack([
            np.load('%s.feats.npy' % seeds),
            np.load('%s.feats.npy' % crow),
        ])
        
        seed_labs = np.load('%s.labs.npy' % seeds)
        labs = np.hstack([
            seed_labs, 
            np.load('%s.labs.npy' % crow)
        ])
        
        init_labels = {i:1 for i in range(seed_labs.shape[0])}
    else:
        print >> sys.stderr, 'init_las: no seeds'
        feats = np.load('%s.feats.npy' % crow)
        labs = np.load('%s.labs.npy' % crow)
        rand_idx = np.random.choice(len(labs))
        init_labels = {rand_idx:0}
    
    print >> sys.stderr, 'init_las: initializing SimpleLAS'
    simp = SimpleLAS(feats, init_labels=init_labels, pi=0.05, eta=0.5, alpha=args.alpha, n=10, verbose=True)
    
    def save():
        simp.save('./simple_las')
    
    atexit.register(save)
    return simp


def load_image(filename, default_width=300, default_height=300):
    w, h = Image.open(filename).size
    aspect = float(w) / h
    
    if aspect > float(default_width) / default_height:
        width = min(w, default_width)
        height = int(width / aspect)
    else:
        height = min(h, default_width)
        width = int(height * aspect)
    
    return {
        'src': filename,
        'width': int(width),
        'height': int(height),
    }

# --
# Endpoints

@app.route('/label', methods=['POST'])
def test():
    global simp
    global sent
    req = request.get_json()
    
    print json.dumps(req)
    sys.stdout.flush()
    
    # Record annotation (if not already annotated)
    filename = '/'.join(req['image_path'].split('/')[3:]) # Everything after the domain
    idx = np.where(labs == filename)[0][0]
    if idx in simp.unlabeled_idxs:
        simp.setLabel(idx, req['label'])
        
        # Next image for annotation
        idxs = simp.next_message
        out = []
        for idx in idxs:
            if idx not in sent:
                # out.append(load_image(os.path.join(args.img_dir, labs[idx])))
                out.append(load_image(labs[idx]))
                sent.add(idx)
                print len(sent)
        
        return jsonify(out)
    else:
        return jsonify([])


@app.route('/imgs/<path:filename>')
def get_image(filename):
    return send_from_directory('./imgs', filename)


# @app.route('/<path:filename>')
# def get_file(filename):
#     print 'get_file'
#     return send_from_directory('.', filename)


@app.route('/')
def index():
    global simp
    global sent
    
    idxs = simp.next_message
    images = []
    for idx in idxs:
        if idx not in sent:
            images.append(load_image(os.path.join(args.img_dir, labs[idx])))
            sent.add(idx)
    
    return render_template('index.html', **{'images': images})


if __name__ == "__main__":
    sent = set([])
    labs = None
    simp = init_las(args.crow, None if args.no_seeds else args.seeds)
    print >> sys.stderr, 'server.py: ready'
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
