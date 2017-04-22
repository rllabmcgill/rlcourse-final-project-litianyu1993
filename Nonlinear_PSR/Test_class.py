from sp2learn import Sample, Learning, Hankel
import sp2learn as sp
from keras.layers import Dense, Activation
import numpy as np
import create_files as cf
from scipy import sparse
import theano
import gc
from numpy import linalg as LA
from keras.optimizers import SGD
from keras.models import Sequential
from numpy.linalg import matrix_rank
from numpy import genfromtxt
from sklearn import datasets, linear_model
import Freeze_version_new as FV
def sum_to_one(x):
    return x/sum(x)
def normalization(x):
    return (x-np.mean(x))/np.std(x)
def sum_to_one(x):
    x = abs(x)
    return x/sum(x)
def sigmoid(x):
    return 1.0-np.exp(-10.0*x)
class test:
    def __init__(self, train_file, test_file, test_result_file):
        self.train_file = train_file
        self.test_file = test_file
        self.test_result_file = test_result_file
        pT = Sample(adr=train_file)
        cols = pT.select_columns(1000)
        rows = pT.select_rows(1000)
        #self.hankelinstance is based on a DOK sparse matrix of hankel matrix
        self.lhankel = Hankel(sample_instance=pT,lcolumns=cols, lrows=rows,
                          version="classic",
                         partial=True, sparse=True).lhankel
        print self.lhankel[0].shape
        #for i in range(0, len(self.lhankel)):
        #    self.lhankel[i] = sigmoid(np.asarray(self.lhankel[i].todense()))
        #    self.lhankel[i] = sparse.dok_matrix(self.lhankel[i])
        #following is to convert hankel matrix to a conditional probability one
        #print np.asarray(self.lhankel[0].todense()).shape
        #max_ele = np.max(self.lhankel[0].todense())
        #for i in range(0, len(self.lhankel)):
        #    temp = self.lhankel[i].todense()
        #    print np.asarray(temp).shape
        #    temp = temp/max_ele
        #    self.lhankel[i] = sparse.dok_matrix(temp)


    def spectral(self, assumed_rank, gen_input = 0):
        fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file, geninput=gen_input)
        sp_result = fv.spectral_method_likelihood(self.train_file, assumed_rank)
        self.test_true_result = fv.test_result
        self.sp_result = sp_result
        gc.collect()
        self.WER = fv.cal_WER(linear_tran=True)
        return sp_result


    def tran_nonlinear(self, assumed_rank, lr, num_epoch, not_first):
        fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file, geninput=not_first)
        fv.build_tran_nonlinear(assumed_rank=assumed_rank, alpha = lr, train_epoch1=num_epoch)
        tran_nonlinear_result = fv.predict_likelihood().reshape(fv.test_result.shape)
        self.tran_nonlinear_result = tran_nonlinear_result
        self.test_true_result = fv.test_result
        gc.collect()
        #self.WER = fv.cal_WER(linear_tran=False)
        return self.tran_nonlinear_result

    def fac_nonlinear(self, assumed_rank, lr, num_epoch, not_first):
        fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file, geninput=not_first)
        fv.build_fac_nonlinear_new(assumed_rank=assumed_rank, alpha = lr, train_epoch1=num_epoch)
        #fv.build_fac_nonlinear_interactive_train(assumed_rank=assumed_rank)
        fac_nonlinear_result = fv.predict_likelihood().reshape(fv.test_result.shape)
        self.tran_nonlinear_result = fac_nonlinear_result
        self.test_true_result = fv.test_result
        gc.collect()
        #self.WER = fv.cal_WER(linear_tran=False)
        return self.tran_nonlinear_result

    def both_nonlinear(self, assumed_rank, lr, num_epoch, not_first):
        fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file, geninput=not_first)
        fv.build_both_nonlinear(assumed_rank=assumed_rank, alpha = lr, train_epoch1 = num_epoch)
        fac_nonlinear_result = fv.predict_likelihood().reshape(fv.test_result.shape)
        self.tran_nonlinear_result = fac_nonlinear_result
        self.test_true_result = fv.test_result
        gc.collect()
        #self.WER = fv.cal_WER(linear_tran=False)
        return self.tran_nonlinear_result

    def all_linear(self, assumed_rank, lr, num_epoch, not_first):
        fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file,
                       geninput=not_first)
        fv.build_all_linear(assumed_rank=assumed_rank, alpha=lr, train_epoch=num_epoch)
        fac_nonlinear_result = fv.predict_likelihood().reshape(fv.test_result.shape)
        self.tran_nonlinear_result = fac_nonlinear_result
        self.test_true_result = fv.test_result
        gc.collect()
        return fac_nonlinear_result

    def cal_perprelexity(self, model_likelihood, true_likelihood):
        temp = true_likelihood * np.log2(model_likelihood)
        return 2 ** (-sum(temp))
    def cal_l1(self, model_likelihood, true_likelihood):
        return np.mean(abs(model_likelihood - true_likelihood))

    def cal_l2(self, model_likelihood, true_likelihood):
        return np.mean((model_likelihood - true_likelihood)**2.0)

    def cross_validation(self, assumed_rank, lr, num_epoch):
        sp_perprelexity = []
        fac_perprelexity = []
        tran_perprelexity = []
        both_perprelexity = []
        flag = 0
        notchange = 1
        for i in range(0, len(assumed_rank)):
            print '!!!'
            if flag == 0:
                pre = i
                flag = 1
            if i != pre:
                notchange = 0
                pre = i
            else:
                notchange = 1
                pre = i
            sp_temp = self.spectral(assumed_rank=assumed_rank[i])
            sp_temp = sum_to_one(sp_temp)
            sp_perprelexity.append(self.cal_perprelexity(sp_temp, self.test_true_result))
            print sp_perprelexity[-1]
            print '???'
            for j in range(0,len(lr)):
                for k in range(0, len(num_epoch)):
                    fac_temp = self.fac_nonlinear(assumed_rank=assumed_rank[i], lr = lr[j], num_epoch = num_epoch[j], not_first=notchange)
                    fac_temp = sum_to_one(fac_temp)
                    tran_temp = self.tran_nonlinear(assumed_rank=assumed_rank[i], lr = lr[j], num_epoch = num_epoch[j], not_first=notchange)
                    tran_temp = sum_to_one(tran_temp)
                    both_temp = self.both_nonlinear(assumed_rank=assumed_rank[i], lr = lr[j], num_epoch = num_epoch[j], not_first=notchange)
                    both_temp = sum_to_one(both_temp)
                    fac_perprelexity.append(self.cal_perprelexity(fac_temp, self.test_true_result))
                    tran_perprelexity.append(self.cal_perprelexity(tran_temp, self.test_true_result))
                    both_perprelexity.append(self.cal_perprelexity(both_temp, self.test_true_result))
                    print "For assumed rank = "+str(assumed_rank[i]) +' and lr = '+str(lr[j])+' and num epoch = '+str(num_epoch[k])+' the result is:'
                    print "Spectral: " + str(sp_perprelexity[-1])
                    print "Fac_nonlinear: " + str(fac_perprelexity[-1])
                    print "Tran_nonlinear: " +str(tran_perprelexity[-1])
                    print "Both_nonlinear: " + str(both_perprelexity[-1])
        sp_perprelexity = np.asarray(sp_perprelexity)
        fac_perprelexity = np.asarray(fac_perprelexity).reshape(len(assumed_rank), len(lr), len(num_epoch))
        tran_perprelexity = np.asarray(tran_perprelexity).reshape(len(assumed_rank), len(lr), len(num_epoch))
        both_perprelexity = np.asarray(both_perprelexity).reshape(len(assumed_rank), len(lr), len(num_epoch))
        np.savetxt("sp_perprelexity.csv", sp_perprelexity, delimiter=',')
        np.savetxt("fac_perprelexity.csv", fac_perprelexity, delimiter=',')
        np.savetxt("tran_perprelexity.csv", tran_perprelexity, delimiter=',')
        np.savetxt("both_perprelexity.csv", both_perprelexity, delimiter=',')
        return sp_perprelexity, fac_perprelexity, tran_perprelexity, both_perprelexity





























