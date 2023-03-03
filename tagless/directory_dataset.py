#!/usr/bin/env python

"""
    tagless/directory_dataset.py
"""

import os
import numpy as np
from pathlib import Path

from PIL import Image

# --
# Helpers

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# --
# Dataset

class DirectoryDataset:
    def __init__(self, root, transform=None):
        """
            - Recursively finds all image files under the root directory
        """
        
        self.fnames     = np.array([str(fname) for fname in Path('/imgs').rglob('*') if is_image_file(str(fname))])
        self.transform  = transform
        self.bad_fnames = set([])
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        
        try:
            img = pil_loader(fname)
        except:
            self.bad_fnames.add(fname)
            img = Image.new(mode='RGB', size=(64, 64))
        
        if self.transform:
            img = self.transform(img)
        
        return img

