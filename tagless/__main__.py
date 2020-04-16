#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

from __future__ import print_function

import os
import re
import sys
import h5py
import json
import atexit
import argparse
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tempfile import NamedTemporaryFile
from flask import Flask, Response, request, abort, \
    render_template, send_from_directory, jsonify, send_file

from tagless.simple_las_sampler import SimpleLASSampler
# from tagless.uncertainty_sampler import UncertaintySampler
# from tagless.validation_sampler import ValidationSampler
# from tagless.random_sampler import RandomSampler

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    # parser.add_argument('--inpath', type=str, default='./data/crow')
    parser.add_argument('--seeds', type=str, default='')
    parser.add_argument('--img-dir', type=str, default='/home/bjohnson/projects/moco/')
    
    parser.add_argument('--mode', type=str, default='las')
    parser.add_argument('--n-las', type=int, default=float('inf'))
    parser.add_argument('--no-permute', action="store_true")
    
    return parser.parse_args()

# --
# Helpers

def get_patch(inpath, patch_idx):
    assert patch_idx < 8
    
    img = np.load(os.path.join(args.img_dir, inpath))
    
    row = patch_idx // 4
    col = patch_idx % 4
    
    img = img[:3].transpose(1, 2, 0)
    img = img[256 * row:256 * (row + 1), 256 * col:256 * (col + 1)]
    return img

def load_image(row, default_width=300, default_height=300):
    print(row.img_name)
    
    img = get_patch(row.img_name, row.patch_idx)
    
    fname = f'/tmp/{row.geohash}_{row.patch_idx}.png'
    Image.fromarray(img, 'RGB').save(fname)
    return {
        "src"    : fname,
        "height" : img.shape[0],
        "width"  : img.shape[1],
    }
    
    # w, h = Image.open(filename).size
    # aspect = float(w) / h
    
    # if aspect > float(default_width) / default_height:
    #     width = min(w, default_width)
    #     height = int(width / aspect)
    # else:
    #     height = min(h, default_width)
    #     width = int(height * aspect)
    
    # return {
    #     'src': filename,
    #     'width': int(width),
    #     'height': int(height),
    # }


# --
# Server

class TaglessServer:
    
    def __init__(self, args, max_outstanding=64):
        self.app = Flask(__name__)
        
        self.mode = args.mode
        if self.mode == 'las':
            sampler = SimpleLASSampler(feat_path=None, meta_path=None, seeds=args.seeds if args.seeds else None, prefix=args.img_dir)
        else:
            raise Exception('TaglessServer: unknown mode %s' % args.mode, file=sys.stderr)
        
        self.sampler = sampler
        
        self.app.add_url_rule('/', 'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_file('/' + x))
        self.app.add_url_rule('/label', 'view_3', self.label, methods=['POST'])
        
        self.n_las = args.n_las
        self.outpath = args.outpath
        self.sent = set([])
        
        self.outstanding = 0
        self.max_outstanding = max_outstanding
        
        def save():
            self.sampler.save(self.outpath)
        
        # atexit.register(save)
        
    def index(self):
        idxs = self.sampler.get_next()
        images = []
        for idx in idxs:
            if idx not in self.sent:
                images.append(load_image(self.sampler.meta.iloc[idx]))
                self.sent.add(idx)
                self.outstanding += 1
        
        print('self.outstanding', self.outstanding)
        return render_template('index.html', **{'images': images})
    
    def label(self):
        self.outstanding -= 1
        print('self.outstanding', self.outstanding)
        req = request.get_json()
        
        print(req)
        
        # Everything after the domain (as absolute path)
        idx = os.path.basename(req['image_path']).split('.')[0]
        
        out = []
        if not self.sampler.is_labeled(idx):
            self.sampler.set_label(idx, req['label'])
            
            # Next image for annotation
            if self.outstanding < self.max_outstanding:
                next_idxs = self.sampler.get_next()
                for next_idx in next_idxs:
                    if next_idx not in self.sent:
                        out.append(load_image(self.sampler.meta.iloc[next_idx]))
                        self.sent.add(next_idx)
                        self.outstanding += 1
            
        req.update({
            'n_hits'    : self.sampler.n_hits(),
            'n_labeled' : self.sampler.n_labeled(),
            'mode'      : self.mode,
        })
        print(json.dumps(req))
        sys.stdout.flush()
        
        if (req['n_hits'] > self.n_las) and (self.mode == 'las'):
            if req['n_labeled'] > req['n_hits']:
                self._switch_sampler()
        
        print('self.outstanding', self.outstanding)
        return(jsonify(out))
    
    def _switch_sampler(self):
        raise Exception()
        """ switch from LAS to uncertainty sampling """
        
        # Save LAS sampler
        self.sampler.save(self.outpath)
        
        print('TaglessServer: las -> uncertainty | start', file=sys.stderr)
        X, y = self.sampler.get_data()
        self.sampler = UncertaintySampler(X, y, self.sampler.labs)
        self.mode = 'uncertainty'
        print('TaglessServer: las -> uncertainty | done', file=sys.stderr)

if __name__ == "__main__":
    args = parse_args()
    server = TaglessServer(args)
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False)
