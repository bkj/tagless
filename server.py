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

from simple_las_sampler import SimpleLASSampler
from uncertainty_sampler import UncertaintySampler

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', type=str, default='simple_las')
    parser.add_argument('--crow', type=str, default='./.crow')
    parser.add_argument('--seeds', type=str, default='')
    parser.add_argument('--img-dir', type=str, default='./')
    
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
    
    def __init__(self, args, outpath='./results/tagless', n_las=1):
        self.app = Flask(__name__)
        
        self.app.add_url_rule('/', 'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_from_directory('.', x))
        self.app.add_url_rule('/label', 'view_3', self.label, methods=['POST'])
        
        sampler = SimpleLASSampler(args.crow, args.seeds if args.seeds else None)
        self.mode = 'las'
        self.sampler = sampler
        self.sent = set([])
        self.n_las = n_las
        self.outpath = outpath
        
        def save():
            self.sampler.save(outpath)
        
        atexit.register(save)
        
    def index(self):
        idxs = self.sampler.get_next()
        images = []
        for idx in idxs:
            if idx not in self.sent:
                images.append(load_image(os.path.join(args.img_dir, self.sampler.labs[idx])))
                self.sent.add(idx)
        
        return render_template('index.html', **{'images': images})
    
    def label(self):
        req = request.get_json()
        
        # Record annotation (if not already annotated)
        filename = '/'.join(req['image_path'].split('/')[3:]) # Everything after the domain
        idx = np.where(self.sampler.labs == filename)[0][0]
        out = []
        if not self.sampler.is_labeled(idx):
            self.sampler.set_label(idx, req['label'])
            
            # Next image for annotation
            idxs = self.sampler.get_next()
            for idx in idxs:
                if idx not in self.sent:
                    out.append(load_image(self.sampler.labs[idx]))
                    self.sent.add(idx)
        
        req.update({
            'n_hits' : self.sampler.n_hits(),
            'n_labeled' : self.sampler.n_labeled(),
            'mode' : self.mode,
        })
        print json.dumps(req)
        sys.stdout.flush()
        
        if (req['n_hits'] > self.n_las) and (self.mode == 'las'):
            if req['n_labeled'] > req['n_hits']:
                self._switch_sampler()
        
        return jsonify(out)
    
    def _switch_sampler(self):
        """ switch from LAS to uncertainty sampling """
        
        # Save LAS sampler
        self.sampler.save(self.outpath)
        
        print >> sys.stderr, 'TaglessServer: switch_sampler'
        X, y = self.sampler.get_data()
        self.sampler = UncertaintySampler(X, y, self.sampler.labs)
        self.mode = 'uncertainty'

if __name__ == "__main__":
    args = parse_args()
    server = TaglessServer(args)
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False)
