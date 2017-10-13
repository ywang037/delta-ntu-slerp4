from keras import backend as K
import os
import importlib

def set_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
        print('{} backend is set'.format(K.backend()))
    elif K.backend() == backend:
        print('{} backend has already been set'.format(K.backend()))
# exmaple to invoke
# set_keras_backend("theano") or
# set_keras_backend("tensorflow")
