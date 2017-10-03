#!/usr/bin/env python

"""
    elasticsearch_sampler.py
"""

import sys
import h5py
import time
import numpy as np
from hashlib import md5
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

class ElasticsearchSampler():
    def __init__(self, filenames, es_host, es_port, es_index):
        self.labs = open(filenames).read().splitlines()
        
        self.labeled_idxs = set([])
        self.hits = set([])
        self.open_sessions = set([])
        
        self.es_index = es_index
        self.doc_type = "image"
        self.client = Elasticsearch([{
            "host" : es_host,
            "port" : es_port
        }])
    
    def _init_index(self):
        if not self.client.indices.exists(self.es_index):
            self.client.indices.create(**{
                "index" : self.es_index
            })
        
        def gen():
            for idx, lab in enumerate(self.labs):
                yield {
                    "_id" : md5(lab).hexdigest(),
                    "_type" : self.doc_type,
                    "_index" : self.es_index,
                    "_op_type" : "index", # set to create to avoid overwrites
                    "_source" : {
                        "path" : lab,
                        "idx" : idx,
                    }
                }
        
        for _ in streaming_bulk(self.client, gen(), raise_on_exception=False):
            pass
    
    def _get_seed(self):
        return int(round(time.time() * 10000))
    
    def get_next(self, session_id=None):
        res = self.client.search(**{
            "index" : self.es_index,
            "body" : {
                "size" : 10 if not session_id else 1,
                "query" : {
                    "function_score" : {
                        "query" : {
                            "bool" : {
                                "must_not" : {
                                    "term" : {
                                        "annotated" : False
                                    }
                                }
                            }
                        },
                        "functions" : [
                            {
                                "random_score" : {
                                    "seed" : self._get_seed()
                                }
                            }
                        ]
                    }
                }
            }
        })
        
        return np.array([r['_source']['idx'] for r in res['hits']['hits']])
    
    def set_label(self, idx, lab, session_id=None):
        self.labeled_idxs.add(idx)
        if lab:
            self.hits.add(idx)
        
        res = self.client.update(**{
            "index" : self.es_index,
            "doc_type" : self.doc_type,
            "id" : md5(self.labs[idx]).hexdigest(),
            "body" : {
                "doc" : {
                    "annotated" : True,
                    "lab" : lab,
                    "session_id" : session_id,
                    "timestamp" : datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                }
            }
        })
        
        print >> sys.stderr, res
    
    def get_data(self):
        return None, None
    
    def n_hits(self):
        return sum(self.hits)
    
    def n_labeled(self):
        return len(self.labeled_idxs)
    
    def is_labeled(self, idx):
        return idx in self.labeled_idxs
    
    def save(self, path):
        pass