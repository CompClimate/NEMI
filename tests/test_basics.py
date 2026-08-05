'''
Test basic functionality of the method
'''

import pytest
import numpy as np
from nemi import NEMI

def test_micro_nemi_pack(embeddings_path):
    '''
    Run the method on noise for an ensemble
    '''

    X = np.random.random((100,5))

    nemi  = NEMI()
    nemi.run(X, n=3, embeddings=embeddings_path)
    nemi.plot('clusters')

    

def test_micro_nemi(embeddings_path):
    '''
    Run the method on noise for single member
    '''

    X = np.random.random((100,5))

    nemi  = NEMI()
    nemi.run(X, n=1, embeddings=embeddings_path)
    nemi.plot('clusters')
