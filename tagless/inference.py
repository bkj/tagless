#!/usr/bin/env python

"""
    __main__.py
"""

import io
import torch
import argparse
from rich import print
from joblib import load

from PIL import Image
from flask import Flask, request, jsonify

import clip

class InferenceServer:
    
    def __init__(self, clf_path):
        self.app = Flask(__name__)
        self.clf = load(clf_path)
        self.model, self.preprocess = clip.load('ViT-L/14@336px', device='cpu')
        self.app.add_url_rule('/inference', 'inference', self.inference, methods=['POST'])
    
    def inference(self):
        img   = Image.open(io.BytesIO(request.data))
        with torch.no_grad():
            feat  = self.model.visual(self.preprocess(img)[None])
            feat  = feat.detach().cpu().numpy()
        
        score = float(self.clf.decision_function(feat))
        return jsonify({"score" : score})

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_path',   type=str, default='/feats/test/model.joblib')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args   = parse_args()
    print('InferenceServer: init')
    server = InferenceServer(args.clf_path)
    print('... starting server ...')
    server.app.run(debug=True, host='0.0.0.0', use_reloader=False, port=6000)
