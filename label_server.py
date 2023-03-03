#!/usr/bin/env python

"""
    __main__.py
"""

import os
import json
import torch
import arrow
import argparse
import numpy as np
from rich import print
from sklearn.svm import LinearSVC
from joblib import dump

from PIL import Image
from flask import Flask, request, render_template, jsonify, send_file

import clip

# --
# Helpers

def load_image(filename, default_width=300, default_height=300):
    w, h   = Image.open(filename).size
    aspect = float(w) / h
    
    if aspect > float(default_width) / default_height:
        width  = min(w, default_width)
        height = int(width / aspect)
    else:
        height = min(h, default_width)
        width  = int(height * aspect)
    
    return {
        'src'    : filename,
        'width'  : int(width),
        'height' : int(height),
    }

# --
# Server

class CLIPServer:
    
    def __init__(self, imgs, feats, fnames, outpath):
        self.app = Flask(__name__)
        
        self.app.add_url_rule('/',         'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_file('../' + x))
        self.app.add_url_rule('/label',    'view_3', self.label, methods=['POST'])
        self.app.add_url_rule('/search',   'view_4', self.search, methods=['POST'])
        
        self.fnames = np.load(fnames)
        self.model, self.preprocess = clip.load('ViT-L/14@336px', device='cpu')
        
        self.feats = np.load(feats)
        self.feats = self.feats / np.sqrt((self.feats ** 2).sum(axis=-1, keepdims=True))
        
        self.rank   = None
        
        self.labels  = []
        self.fout    = open(outpath, 'w')
    
    def _get_imgs(self, idxs, sims):
        fnames = self.fnames[idxs]
        images = [load_image(fname) for fname in fnames]
        for image, sim, idx in zip(images, sims, idxs):
            image['sim'] = float(sim)
            image['idx'] = int(idx)
        
        return jsonify(images)
    
    def search(self, k=16):
        req   = request.get_json()
        query = req['query']
        
        with torch.no_grad():
            text   = clip.tokenize([query])
            qt_enc = self.model.encode_text(text).squeeze()
            qt_enc /= qt_enc.norm()
            qt_enc = qt_enc.numpy()
        
        sims = self.feats @ qt_enc
        self.idxs = np.argsort(-sims)
        self.sims = sims[self.idxs]
        
        curr_idxs, self.idxs = self.idxs[:k], self.idxs[k:]
        curr_sims, self.sims = self.sims[:k], self.sims[k:]
        
        return self._get_imgs(curr_idxs, curr_sims)
    
    def label(self):
        req               = request.get_json()
        req['image_path'] = req['image_path']
        req['timestamp']  = arrow.now().strftime('%Y-%m-%d %H:%M:%S')
        req['idx']        = int(req['idx'])
        
        print(json.dumps(req))
        self.labels.append(req)
        self.fout.write(json.dumps(req) + '\n')
        self.fout.flush()
        
        curr_idxs, self.idxs = self.idxs[:1], self.idxs[1:]
        curr_sims, self.sims = self.sims[:1], self.sims[1:]
        
        # >>
        # re-rank via SVC
        if (len(self.labels) > 0) and (len(self.labels) % 10 == 0):
            pos_idxs  = [x['idx'] for x in self.labels if x['label'] == True]
            neg_idxs  = [x['idx'] for x in self.labels if x['label'] == False]
            
            if len(pos_idxs) > 0:    
                neg_feats = self.feats[neg_idxs]
                pos_feats = self.feats[pos_idxs]
                
                X = np.row_stack([pos_feats, neg_feats])
                y = np.hstack([1 * np.ones(len(pos_feats)), 0 * np.ones(len(neg_feats))])
                
                clf  = LinearSVC().fit(X, y)
                sims = clf.decision_function(self.feats)
                
                sims[neg_idxs] = -np.inf
                sims[pos_idxs] = -np.inf # don't show these again
                
                self.idxs = np.argsort(-sims)
                self.sims = sims[self.idxs]
                
                curr_idxs, self.idxs = self.idxs[:1], self.idxs[1:]
                curr_sims, self.sims = self.sims[:1], self.sims[1:]
                
                dump(clf, '/feats/test/model.joblib')

        # <<
        
        return self._get_imgs(curr_idxs, curr_sims)
    
    def index(self):
        return render_template('index.html')

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs',    type=str, default='/imgs')
    parser.add_argument('--feats',   type=str, default='/feats/test/feats.npy')
    parser.add_argument('--fnames',  type=str, default='/feats/test/fnames.npy')
    parser.add_argument('--outpath', type=str, default='/feats/test/out.jl')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args   = parse_args()
    
    # <<
    # Fix absolute path issue
    assert args.imgs == '/imgs'
    _ = os.symlink('/imgs', 'imgs')
    # >>
    
    print('CLIPServer: init')
    server = CLIPServer(args.imgs, args.feats, args.fnames, args.outpath)
    print('... starting server ...')
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False)
