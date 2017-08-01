#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from flask import Flask, Response, request, abort, \
    render_template, send_from_directory, jsonify

from simple_las import SimpleLAS

app = Flask(__name__, template_folder='templates', static_folder='static')

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats', type=str, default='./crow-feats')
    parser.add_argument('--img-dir', type=str, default='./imgs')
    return parser.parse_args()

args = parse_args()

# --
# Helpers

def init_las(feats, labs):
    print >> sys.stderr, 'server.py: init_las'
    # if not os.path.exists('.')
    # print >> sys.stderr, 'load feats: start'
    # df = pd.read_csv('./crow-feats', sep='\t', header=None)
    # print >> sys.stderr, 'load feats: done'
    
    # feats = np.array(df[range(1, df.shape[1])])
    # feats /= np.sqrt((feats ** 2).sum(axis=1, keepdims=True))
    # feats = feats.T
    
    # labs = np.array(df[0])
    # labs = np.array(map(os.path.basename, labs))
    
    # np.save('./feats', feats)
    # np.save('./labs', labs)
    init_labels = {0:0} # !! Say that first image is negative -- avoid initialization
    simp = SimpleLAS(feats, init_labels=init_labels, pi=0.05, eta=0.5, alpha=1e-6, n=10)
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
            next_filename = labs[idx]
            out.append(load_image(os.path.join(args.img_dir, next_filename)))
        
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
    return render_template('index.html', **{
        'images': [load_image(f) for f in glob(os.path.join(args.img_dir, '*jpg'))[1:13]]
    })


if __name__ == "__main__":
    feats = np.load('./.feats.npy')
    labs = np.load('./.labs.npy')
    simp = init_las(feats.T, labs)
    app.run(debug=True, host='0.0.0.0')


