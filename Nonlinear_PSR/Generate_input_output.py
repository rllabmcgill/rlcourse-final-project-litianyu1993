from sp2learn import Sample, Learning, Hankel
import numpy as np
import scipy.sparse as sparse
from scipy import sparse, io
import gc

def gen_input_output(hankel, sample_number):
    #hankel has to be a DOK sparse matrix
    input = np.random.rand(sample_number, hankel.shape[0])
    input = (input - np.mean(input, axis=0)) / np.std(input, axis=0)
    input = sparse.dok_matrix(input)
    output = np.dot(input, hankel)
    return input, output

def normalization(x):
    return (x-np.mean(x))/np.std(x)
def scale(x):
    return x/(max(x)-min(x))
def gen(trainfile, lhankel):

    new_hankel = lhankel[0]#take the vanilla hankel matrix to construct the p-net and s-net
    sample_number = 10000
    return gen_input_output(new_hankel, sample_number)
