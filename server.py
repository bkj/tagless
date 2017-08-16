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

app = Flask(__name__)

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', type=str, default='simple_las')
    
    parser.add_argument('--crow', type=str, default='./.crow')
    parser.add_argument('--seeds', type=str, default='./.crow-seeds')
    parser.add_argument('--no-seeds', action='store_true')
    parser.add_argument('--img-dir', type=str, default='./')
    
    return parser.parse_args()

args = parse_args()
if args.sampler == 'simple_las':
    MySampler = SimpleLASSampler

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
    
    def __init__(self, sampler, outpath='./results/tagless'):
        self.app = Flask(__name__)
        
        self.app.add_url_rule('/', 'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_from_directory('.', x))
        self.app.add_url_rule('/label', 'view_3', self.label, methods=['POST'])
        
        self.sampler = sampler
        self.sent = set([])
        
        def save():
            self.sampler.save(outpath)
        
        atexit.register(save)
        
    def index(self):
        idxs = self.sampler.next_message
        images = []
        for idx in idxs:
            if idx not in self.sent:
                images.append(load_image(os.path.join(args.img_dir, self.sampler.labs[idx])))
                self.sent.add(idx)
        
        return render_template('index.html', **{'images': images})
    
    def label(self):
        req = request.get_json()
        
        print json.dumps(req)
        sys.stdout.flush()
        
        # Record annotation (if not already annotated)
        filename = '/'.join(req['image_path'].split('/')[3:]) # Everything after the domain
        idx = np.where(self.sampler.labs == filename)[0][0]
        if idx in self.sampler.unlabeled_idxs:
            self.sampler.setLabel(idx, req['label'])
            
            # Next image for annotation
            idxs = self.sampler.next_message
            out = []
            for idx in idxs:
                if idx not in self.sent:
                    out.append(load_image(self.sampler.labs[idx]))
                    self.sent.add(idx)
                    print len(self.sent)
            
            return jsonify(out)
        else:
            return jsonify([])


if __name__ == "__main__":
    sampler = MySampler(args.crow, None if args.no_seeds else args.seeds)
    server = TaglessServer(sampler)
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False)
