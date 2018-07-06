#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

from __future__ import print_function

import os
import re
import requests
import sys
import h5py
import json
import atexit
import argparse
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from flask import Flask, Response, request, abort, \
    render_template, send_from_directory, jsonify, send_file

from simple_las_sampler import SimpleLASSampler
from uncertainty_sampler import UncertaintySampler
from validation_sampler import ValidationSampler
from random_sampler import RandomSampler

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--inpath', type=str, default='./data/crow')
    parser.add_argument('--seeds', type=str, default='')
    parser.add_argument('--img-dir', type=str, default=os.getcwd())
    
    parser.add_argument('--mode', type=str, default='las')
    parser.add_argument('--n-las', type=int, default=float('inf'))
    parser.add_argument('--no-permute', action="store_true")
    parser.add_argument('--n-return', type=int, default=10)
    
    return parser.parse_args()

# --
# Helpers

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
# Server

class TaglessServer:
    
    def __init__(self, args, max_outstanding=64):
        self.app = Flask(__name__)
        
        self.mode = args.mode
        if self.mode == 'validation':
            f = h5py.File(args.inpath)
            preds, labs, y = f['preds'].value, f['labs'].value, f['y'].value
            sampler = ValidationSampler(preds=preds, labs=labs, y=y, no_permute=args.no_permute)
        elif self.mode == 'las':
            sampler = SimpleLASSampler(crow=args.inpath, seeds=args.seeds if args.seeds else None, prefix=args.img_dir,n=args.n_return)
        elif self.mode == 'random':
            sampler = RandomSampler(crow=args.inpath, prefix=args.img_dir)
        elif self.mode == 'uncertainty':
            f = h5py.File(args.inpath)
            X, y, labs = f['X'].value, f['y'].value, f['labs'].value
            sampler = UncertaintySampler(X=X, y=y, labs=labs)
        else:
            raise Exception('TaglessServer: unknown mode %s' % args.mode, file=sys.stderr)
        
        self.sampler = sampler
        
        self.app.add_url_rule('/', 'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_file('/' + x))
        self.app.add_url_rule('/label', 'view_3', self.label, methods=['POST'])
        self.app.add_url_rule('/twitter_tag', 'view_4', self.twitter_tag, methods=['POST'])
        
        self.n_las = args.n_las
        self.outpath = args.outpath
        self.sent = set([])
        
        self.outstanding = 0
        self.max_outstanding = max_outstanding
        
        def save():
            self.sampler.save(self.outpath)
        
        atexit.register(save)
        
    def index(self):
        idxs = self.sampler.get_next()
        images = []
        for idx in idxs:
            if idx not in self.sent:
                images.append(load_image(self.sampler.labs[idx]))
                self.sent.add(idx)
                self.outstanding += 1
        
        print('self.outstanding', self.outstanding)
        return render_template('index.html', **{'images': images})

    def get_handle(self,nid):
        resp = requests.get("https://twitter.com/intent/user?user_id=" + nid)
        handle = resp.text.split("<title>")[1].split(")")[0].split("@")[1]
        return handle    

    def twitter_tag(self):
        self.outstanding -= 1
        print('self.outstanding', self.outstanding)
        req = request.get_json()
        filename = req['image_path']
        # Record annotation (if not already annotated)
        idx = np.where(self.sampler.labs == filename)[0][0]
        out = []
        if not self.sampler.is_labeled(idx):
            self.sampler.set_label(idx, req['label'])

            # Next image for annotation
            if self.outstanding < self.max_outstanding:
                idxs = self.sampler.get_next()
                for idx in idxs:
                    if idx not in self.sent:
                        out.append(self.sampler.labs[idx])
                        self.sent.add(idx)
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
        out_ids = map(self.get_handle,out)
        return(jsonify(out_ids))

    def label(self):
        self.outstanding -= 1
        print('self.outstanding', self.outstanding)
        req = request.get_json()
        
        # Everything after the domain (as absolute path)
        filename = '/' + '/'.join(req['image_path'].split('/')[3:]) 
        
        # Record annotation (if not already annotated)
        idx = np.where(self.sampler.labs == filename)[0][0]
        out = []
        if not self.sampler.is_labeled(idx):
            self.sampler.set_label(idx, req['label'])
            
            # Next image for annotation
            if self.outstanding < self.max_outstanding:
                idxs = self.sampler.get_next()
                for idx in idxs:
                    if idx not in self.sent:
                        out.append(load_image(self.sampler.labs[idx]))
                        self.sent.add(idx)
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
