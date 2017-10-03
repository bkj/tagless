#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

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
from datetime import datetime

from flask import Flask, Response, request, abort, \
    render_template, send_from_directory, jsonify, send_file
from flask_basicauth import BasicAuth


from simple_las_sampler import SimpleLASSampler
from uncertainty_sampler import UncertaintySampler
from elasticsearch_sampler import ElasticsearchSampler

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sampler', type=str, default='simple_las')
    
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--crow', type=str, default='./data/crow')
    parser.add_argument('--seeds', type=str, default='')
    parser.add_argument('--img-dir', type=str, default=os.getcwd())
    parser.add_argument('--n-las', type=int, default=5)
    
    parser.add_argument('--es-host', type=str, default='localhost')
    parser.add_argument('--es-port', type=int, default=9200)
    parser.add_argument('--es-index', type=str, default='tagless-%s' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--filenames', type=str, default='')
    parser.add_argument('--labels', type=str, default='')
    
    parser.add_argument('--hot-start', type=str)
    
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--img-dim', type=int, default=300)
    parser.add_argument('--allow-relabel', action='store_true')
    parser.add_argument('--require-authentication', action='store_true')
    parser.add_argument('--username', type=str, default="tagless")
    parser.add_argument('--password', type=str, default='90-poi')
    
    args = parser.parse_args()
    if args.sampler != 'elasticsearch':
        assert args.outpath is not None
    
    print >> sys.stderr, vars(args)
    
    return args

# --
# Helpers

def load_image(filename, width=300, height=300):
    w, h = Image.open(filename).size
    aspect = float(w) / h
    
    if aspect > float(width) / height:
        width = min(w, width)
        height = int(width / aspect)
    else:
        height = min(h, width)
        width = int(height * aspect)
    
    return {
        'src': filename,
        'width': int(width),
        'height': int(height),
    }

# --
# Server

class TaglessServer:
    
    def __init__(self, args):
        self.app = Flask(__name__)
        
        # Configure authentication
        if args.require_authentication:
            self.app.config['BASIC_AUTH_USERNAME'] = args.username
            self.app.config['BASIC_AUTH_PASSWORD'] = args.password
            self.app.config['BASIC_AUTH_FORCE'] = True
            basic_auth = BasicAuth(self.app)
        
        if args.sampler == 'elasticsearch':
            self.mode = 'elasticsearch'
            sampler = ElasticsearchSampler(args.filenames, args.es_host, args.es_port, args.es_index)
            sampler.labs = np.array([os.path.join(args.img_dir, l) for l in sampler.labs])
            sampler._init_index()
            
        elif not args.hot_start:
            self.mode = 'las'
            sampler = SimpleLASSampler(args.crow, args.seeds if args.seeds else None)
            sampler.labs = np.array([os.path.join(args.img_dir, l) for l in sampler.labs])
        else:
            self.mode = 'uncertainty'
            f = h5py.File(args.hot_start)
            X, y, labs = f['X'].value, f['y'].value, f['labs'].value
            sampler = UncertaintySampler(X, y, labs)
        
        self.sampler = sampler
        
        self.app.add_url_rule('/', 'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_file('/' + x))
        self.app.add_url_rule('/label', 'view_3', self.label, methods=['POST'])
        self.app.add_url_rule('/meta', 'view_4', self.meta)
        
        self.n_las = args.n_las
        self.outpath = args.outpath
        self.sent = set([])
        
        # Set annotation colors
        self.classes = args.labels.split(',')
        if len(self.classes) > 4:
            raise Exception('!! too many classes')
        
        self.keycodes = [81, 87, 69, 82]
        self.keycodes = self.keycodes[:len(self.classes)]
        
        self.colors = ['blue', 'red', 'green', 'orange']
        self.colors = self.colors[:len(self.classes)]
        
        # Set image dimensions
        self.img_dim = args.img_dim
        self.allow_relabel = args.allow_relabel
        
        # Set action at exit
        def save():
            self.sampler.save(self.outpath)
        
        atexit.register(save)
    
    def meta(self):
        return jsonify({
            "keycodes" : self.keycodes,
            "colors"   : self.colors,
            "classes"  : self.classes,
            "allow_relabel" : self.allow_relabel,
        })
    
    def index(self):
        idxs = self.sampler.get_next()
        images = []
        for idx in idxs:
            if idx not in self.sent:
                images.append(load_image(self.sampler.labs[idx], self.img_dim, self.img_dim))
                self.sent.add(idx)
        
        return render_template('index.html', **{'images': images})
    
    def label(self):
        session_id = request.headers['Tagless-Session-Id']
        req = request.get_json()
        
        # Everything after the domain (as absolute path)
        filename = '/' + '/'.join(req['image_path'].split('/')[3:]) 
        
        # Record annotation (if not already annotated)
        idx = np.where(self.sampler.labs == filename)[0][0]
        out = []
        if not self.sampler.is_labeled(idx):
            self.sampler.set_label(idx, req['label'], session_id=session_id)
            
            # Next image for annotation
            idxs = self.sampler.get_next(session_id)
            for idx in idxs:
                if idx not in self.sent:
                    out.append(load_image(self.sampler.labs[idx], self.img_dim, self.img_dim))
                    self.sent.add(idx)
        
        elif self.allow_relabel:
            print >> sys.stderr, 'relabeling!'
            self.sampler.set_label(idx, req['label'], session_id=session_id)
        
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
        
        print >> sys.stderr, 'TaglessServer: las -> uncertainty | start'
        X, y = self.sampler.get_data()
        self.sampler = UncertaintySampler(X, y, self.sampler.labs)
        self.mode = 'uncertainty'
        print >> sys.stderr, 'TaglessServer: las -> uncertainty | done'


if __name__ == "__main__":
    args = parse_args()
    TaglessServer(args).app.run(
        host='0.0.0.0',
        port=args.port,
        debug=True,
        use_reloader=False
    )
