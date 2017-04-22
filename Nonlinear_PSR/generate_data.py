import numpy as np
import create_files as cf
from hmmlearn import hmm
import random
import os
import csv
from random import shuffle
import matplotlib.pyplot as plt
import pebl
def sum_to_one_vec(x):
    return x/np.sum(x)
def sum_to_one_matrix(x):
    return (x.transpose()/np.sum(x, axis=1)).transpose()
def symmetrize(a):
    a_l = []
    for i in range(0, a.shape[0]):
        a_l.append((a[i].dot(a[i].T)))
        #print np.all(np.linalg.eigvals(a_l[-1]) > 0)
    return np.asarray(a_l)

def para_sigmoid(x, t, m):
    #print x.shape
    #print m[t-1].shape
    return 1.0/(1.0+np.exp(-t*np.dot(m[t-1], x)))
def gen_initial(m):
    x = np.random.rand(10)
    x = x*2.0-1.0
    return 1.0/(1.0+np.exp(-0.5*np.dot(m, x)))
def gen_term_vec():
    return np.random.rand(10)
def term_function(x, term_vec, m):
    #term_vec = np.random.rand(10)
    #term_vec = term_vec*4.0-2.0
    temp = 1.0/(1.0+np.exp(-0.5*np.dot(m, x)))
    return np.sum(np.dot(temp, term_vec))

num_sample = 100
max_size = 25
term_prob = 0.06
num_sym = 4.0
#sym = []
sym_prob = (1-term_prob)/num_sym
sym = np.zeros(np.int(10000*term_prob)).reshape(-1, 1)
m = []
for i in range(0, np.int(num_sym)):
    temp = (i+1)*np.ones(np.int(10000*sym_prob)).reshape(-1, 1)
    m.append(np.random.rand(10, 10))
    #print sym.shape
    #print temp.shape
    sym = np.concatenate((sym, temp), axis = 0)
term_m = np.random.rand(10, 10)
sym = np.asarray(sym).ravel().astype(int)
#print sym
#print shuffle(sym)
sample = []
b_vec_ori = gen_initial(np.random.rand(10, 10))
term_vec = gen_term_vec()
term_prob = []
for j in range(0, num_sample):
    b_vec = b_vec_ori
    sym = random.sample(sym, len(sym))
    #print sym.shape

    index = 0
    for i in range(0, len(sym)):
        if sym[i] == 0:
            index = i
            break
    #print sym[0:index+1]
    sample.append(sym[0:index+1])
    for i in range(0, len(sample[-1])-1):
        #print b_vec
        b_vec = para_sigmoid(b_vec, sample[-1][i], m)
        b_vec = b_vec/term_function(b_vec, term_vec, term_m)
        #b_vec = b_vec+0.1*np.random.rand(10)
        #b_vec = sum_to_one_vec(b_vec)
        #print b_vec
        #b_vec = b_vec/term_function(b_vec, term_vec)
    term_prob.append(term_function(b_vec, term_vec, term_m))
print sum_to_one_vec(term_prob)
