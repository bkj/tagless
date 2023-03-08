#!/usr/bin/env python

"""
    __main__.py
"""

import os
import io
import torch
import argparse
import numpy as np
from rich import print
from joblib import load
from glob import glob

from PIL import Image
from flask import Flask, request, jsonify

import clip

class InferenceServer:
    
    def __init__(self, clf_dir, model_str='ViT-L/14@336px'):
        self.app  = Flask(__name__)
        
        clf_paths = sorted(glob(os.path.join(clf_dir, '*.joblib')))
        self.clfs = {os.path.basename(clf_path).replace('.joblib', ''):load(clf_path) for clf_path in clf_paths}
        
        self.model, self.preprocess = clip.load(model_str, device='cpu')
        self.app.add_url_rule('/inference', 'inference', self.inference, methods=['POST'])
    
    def inference(self):
        img   = Image.open(io.BytesIO(request.data))
        with torch.no_grad():
            feat = self.model.visual(self.preprocess(img)[None])
            feat = feat.detach().cpu().numpy()
            feat = feat / np.sqrt((feat ** 2).sum(axis=-1, keepdims=True))
        
        scores = {k:float(v.decision_function(feat)) for k,v in self.clfs.items()}
        return jsonify(scores)

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_dir',   type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args   = parse_args()
    print('InferenceServer: init')
    server = InferenceServer(args.clf_dir)
    print('... starting server ...')
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False, port=6000)
