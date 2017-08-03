#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

import os
import sys
import json
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
    parser.add_argument('--img-dir', type=str, default='./imgs')
    return parser.parse_args()

args = parse_args()

# --
# Helpers

def init_las(crow, seeds=None):
    
    global labs
    
    if seeds:
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
        feats = np.load('%s.feats.npy' % crow)
        labs = np.load('%s.labs.npy' % crow)
        rand_idx = np.random.choice(len(labs))
        init_labels = {rand_idx:0}
    
    simp = SimpleLAS(feats, init_labels=init_labels, pi=0.05, eta=0.5, alpha=1e-6, n=3)
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
    filename = os.path.basename(req['image_path'])
    idx = np.where(labs == filename)[0][0]
    if idx in simp.unlabeled_idxs:
        simp.setLabel(idx, req['label'])
        
        # Next image for annotation
        idxs = simp.next_message
        out = []
        for idx in idxs:
            if idx not in sent:
                out.append(load_image(os.path.join(args.img_dir, labs[idx])))
                sent.add(idx)
                print len(sent)
        
        return jsonify(out)
    else:
        return jsonify([])


@app.route('/imgs/<path:filename>')
def get_image(filename):
    return send_from_directory(args.img_dir, filename)


@app.route('/<path:filename>')
def get_file(filename):
    return send_from_directory('.', filename)


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
    simp = init_las(args.feats, args.seeds)
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
