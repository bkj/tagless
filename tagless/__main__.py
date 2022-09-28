#!/usr/bin/env python

"""
    server.py
    
    Server for simple_las activate labeling of images
"""

import os
import arrow
import json
import torch
import bcolz
import numpy as np

from PIL import Image
from flask import Flask, Response, request, abort, \
    render_template, send_from_directory, jsonify, send_file

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
    
    def __init__(self):
        self.app = Flask(__name__)
        
        self.app.add_url_rule('/',         'view_1', self.index)
        self.app.add_url_rule('/<path:x>', 'view_2', lambda x: send_file('../' + x))
        self.app.add_url_rule('/label',    'view_3', self.label, methods=['POST'])
        self.app.add_url_rule('/search',   'view_4', self.search, methods=['POST'])
        
        self.fnames = np.load(os.path.join('out', 'fnames.npy'))
        self.model, self.preprocess = clip.load('ViT-L/14@336px', device='cpu')
        
        self.feats = bcolz.open('out/feats.bcolz')[:]
        self.feats = self.feats / np.sqrt((self.feats ** 2).sum(axis=-1, keepdims=True))
        
        self.rank   = None
        
        # self.labels  = []
        self.fout    = open('out2.jl', 'w')
    
    def _get_imgs(self, idxs, sims):
        fnames = self.fnames[idxs]
        fnames = [os.path.join('data', os.path.basename(fname)) for fname in fnames]
        images = [load_image(fname) for fname in fnames]
        for image, sim in zip(images, sims):
            image['sim'] = sim
        
        return jsonify(images)
    
    def search(self, k=64):
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
        req = request.get_json()
        req['image_path'] = os.path.basename(req['image_path'])
        req['timestamp']  = arrow.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(json.dumps(req))
        print(json.dumps(req), file=self.fout)
        self.fout.flush()
        
        curr_idxs, self.idxs = self.idxs[:1], self.idxs[1:]
        curr_sims, self.sims = self.sims[:1], self.sims[1:]
        
        return self._get_imgs(curr_idxs, curr_sims)
    
    def index(self):
        return render_template('index.html')

# --

if __name__ == "__main__":
    server = CLIPServer()
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False)
