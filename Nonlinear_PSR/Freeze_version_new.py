from sp2learn import Sample, Learning, Hankel
from keras.layers import Dense, Activation
import numpy as np
from scipy import sparse, io
import time
import theano
from numpy import linalg as LA
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.models import Sequential
from numpy.linalg import matrix_rank
from numpy import genfromtxt
from keras.optimizers import Adamax
from keras import backend as K
from sklearn import datasets, linear_model
import Generate_input_output as GIO
def custom(x):
    return -0.5*K.log(K.abs(2./(x+1.) - 1.))
def tanh_inv(x):
    temp = -(np.log((2.0/(x+1.0)) - 1.0 + np.finfo(float).eps))/2.0
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            if np.isnan(temp[i, j]) == True:
                temp[i,j] = np.finfo(float).eps
    return temp
def normalization(x):
    return (x-np.mean(x))/np.std(x)
def normalize_matrix(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis = 0)
def scale(x):
    return x/(max(x)-min(x))
class freeze:
    def __init__(self, lhankel, trainfile, testfile, testresultfile, geninput):
        self.lhankel = lhankel
        self.nbl = len(lhankel) - 1
        self.input, self.output = GIO.gen(trainfile, lhankel)
        self.input_mean = 0.0
        self.input_std = 0.0
        self.output_mean = []
        self.output_std = []

        test = []
        with open(testfile, 'r') as lines:
            for line in lines:
                temp = line.split(' ')
                #print temp
                temp2 = []
                for i in range(1, len(temp)):
                    if temp[i] != '\n' and temp[i]!=' \n' and temp[i]!='\n ' and temp[i]!='\r\n':
                        #print temp[i]
                        try:
                            temp2.append(np.int(float(temp[i])))
                        except ValueError, e:
                            print repr(temp[i])
                            print "error", e, "on line", i
                test.append(temp2)

        test = np.asarray(test)
        test_reult = genfromtxt(testresultfile, delimiter=' ')
        self.test = test
        self.test_result = test_reult
    def transfer_next_state(self, current_state, ao, linear = True):
        if linear == True:
            current = np.dot(current_state, self.pred_model[ao]).ravel()
            #print current
            term = np.dot(current, self.term)
            #print term
            #time.sleep(1)
            return current/term
        else:
            current = self.pred_model[ao].predict(current_state).ravel().reshape(1, -1)
            term = self.term.predict(current)[0][0]
            return (current[0]/term)
    def build_all_linear(self, assumed_rank, alpha = 0.1, train_epoch = 50):
        model_PS = Sequential()
        model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=assumed_rank))
        model_PS.add(Activation("linear"))
        model_PS.add(Dense(output_dim=self.output.shape[1]))
        model_PS.add(Activation("linear"))
        model_PS.compile(loss='mean_squared_error', optimizer=SGD(lr=alpha, momentum=0.9, nesterov=True),
                         metrics=['mean_squared_error'])
        model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch, batch_size=64,
                     validation_split=0.1)

        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)

        P = np.asarray(weights_arr[0][0])
        S = np.asarray(weights_arr[2][0])
        #print
        # temp = tanh_inv(new_hankel[:, 0].reshape(1, -1))
        temp = self.lhankel[0][:, 0].todense().reshape(1, -1)
        temp = np.dot(temp, np.linalg.pinv(P).transpose())
        term = temp.reshape(-1, 1)
        # term = np.dot(np.linalg.pinv(P), new_hankel[:, 0])  # linear factorization
        init = np.dot(self.lhankel[0][0, :].todense(), np.linalg.pinv(S))
        init = init.reshape(1, -1)
        self.init = init
        self.term = term
        A_sigma_one = []
        A_sigma_two = []
        for i in range(0, self.nbl):

            self.output = np.dot(self.input, self.lhankel[i + 1])

            model = Sequential()
            model.add(Dense(input_dim=self.input.shape[1], output_dim=assumed_rank, weights=weights_arr[0], trainable=True))
            model.add(Activation("linear"))
            ######################################
            model.add(Dense(output_dim=assumed_rank))
            #model.add(Activation("linear"))
            #model.add(Dense(output_dim=assumed_rank))
            ######################################
            model.add(Activation("linear"))
            model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[2], trainable=True))
            model.add(Activation("linear"))
            model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True, clipnorm=1.),
                          metrics=['mean_squared_error'])
            model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch, batch_size=64,
                      validation_split=0.1)

            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            A_sigma_one.append(weights_list[2])
            A_sigma_two.append(weights_list[4])

        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(
                Dense(input_dim=assumed_rank, output_dim=assumed_rank * 2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("linear"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)
        self.pred_model = pred_model
    def build_fac_nonlinear_new(self, assumed_rank, alpha, train_epoch1 = 100, train_epoch2 = 20):
        #scipy.sparse.linalg.svds(self.lhankel[0]):
        model_PS = Sequential()
        model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(2*self.input.shape[1])))
        model_PS.add(Activation('tanh'))
        model_PS.add(Dense(output_dim=assumed_rank))
        model_PS.add(Dense(output_dim=np.int(2*self.output.shape[1])))
        model_PS.add(Activation('tanh'))
        model_PS.add(Dense(output_dim=self.output.shape[1]))
        model_PS.add(Activation("linear"))
        model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                         metrics=['mean_squared_error'])
        model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64,
                     validation_split=0.1, verbose=0)
        #hankel_lam = model_PS.predict(np.identity(1000))

        print self.input.shape
        print self.output.shape
        #print np.linalg.matrix_rank(hankel_lam)
        #print hankel_lam

        #U, s, V = np.linalg.svd(hankel_lam, full_matrices=True)
        #print U
        #print s
        #print V
        def tanh(x):
            return 2.0/(1+np.exp(-2.0*x)) - 1.0

        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)
        '''
        X = np.identity(1000)
        model = Sequential()
        model.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5*self.input.shape[1]), weights=weights_arr[0], trainable=False))
        model.add(Activation("tanh"))
        model.add(Dense(output_dim= assumed_rank, weights=weights_arr[2], trainable=False))
        X = model.predict(X)
        # = tanh(np.dot(X, weights_arr[0]))
        # = np.dot(X, weights_arr[2])
        Y = np.sum(self.lhankel[0], axis = 1)
        from sklearn import datasets, linear_model
        X_test = X[800:]
        X = X[0:800]
        Y_test = Y[800:]
        Y = Y[0:800]

        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        pred = np.asarray(regr.predict(X_test)).ravel()
        Y_test = np.asarray(Y_test).ravel()
        print("Mean squared error: %.10f"
                      % np.mean((regr.predict(X_test) - Y_test) ** 2))
        '''
        U1 = np.asmatrix(weights_arr[0][0])
        V1 = np.asmatrix(weights_arr[2][0])
        U2 = np.asmatrix(weights_arr[3][0])
        V2 = np.asmatrix(weights_arr[5][0])
        U1_T = U1.transpose()
        U2_T = U2.transpose()
        V1_T = V1.transpose()
        V2_T = V2.transpose()
        e1 = np.zeros(self.lhankel[0].shape[0])
        e1[0] = 1.0
        e1 = np.asmatrix(e1)
        e3 = np.zeros(self.lhankel[0].shape[1])
        e3[0] = 1.0
        e3 = np.asmatrix(e3)
        e3_T = e3.transpose()
        e2 = np.matrix(np.zeros(assumed_rank))
        e2[0] = 1.0
        IR = np.identity(assumed_rank)
        e2_T =e2.transpose()
        e1_T = e1.transpose()
        init = tanh(e1.dot(U1)).dot(V1)
        term = V2_T.dot(tanh(U2_T.dot(e2_T)))
        term = tanh(IR.dot(U2)).dot(V2).dot(e3_T)
        term = Sequential()
        term.add(Dense(input_dim = assumed_rank, output_dim = 2*self.output.shape[1], weights = weights_arr[3], trainable = False))
        term.add(Activation('tanh'))
        term.add(Dense(output_dim = self.lhankel[0].shape[1], weights = weights_arr[5], trainable = False))

        #temp = self.lhankel[0].todense()[:, 0].reshape(-1, 1)
        #temp = np.dot(np.linalg.pinv(U1), temp)
        #temp = tanh_inv(temp)
        #temp = np.dot(np.linalg.pinv(V1), temp)
        #term = temp.reshape(-1, 1)

        #temp = self.lhankel[0][0,:].todense().reshape(1, -1)
        #temp = tanh_inv(temp.dot(np.linalg.pinv(V2)))
        #temp = np.dot(temp, np.linalg.pinv(U2))
        #init = temp.reshape(1, -1)

        self.init = np.asarray(init)
        self.term = term
        #print self.init.shape
        #print self.term.shape

        A_sigma_one = []
        A_sigma_two = []
        for i in range(0, self.nbl):
            self.output = np.dot(self.input, self.lhankel[i + 1])
            model = Sequential()
            model.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(2*self.input.shape[1]), weights=weights_arr[0], trainable=False))
            model.add(Activation('tanh'))
            model.add(Dense(output_dim= assumed_rank, weights=weights_arr[2], trainable=False))
            #############################################
            model.add(Dense(output_dim=assumed_rank * 2))
            model.add(Activation("linear"))
            model.add(Dense(output_dim=assumed_rank))
            ############################################
            model.add(Dense(output_dim=np.int(2*self.output.shape[1]), weights = weights_arr[3],trainable = False))
            model.add(Activation('tanh'))
            model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[5], trainable=False))
            model.add(Activation("linear"))
            model.compile(loss='mean_squared_error', optimizer=Adamax(),
                          metrics=['mean_squared_error'])
            model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch2, batch_size=64,
                      validation_split=0.1, verbose=0)
            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            A_sigma_one.append(weights_list[3])
            A_sigma_two.append(weights_list[5])

        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(Dense(input_dim=assumed_rank, output_dim=assumed_rank * 2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("linear"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)
        self.pred_model = pred_model

    def build_fac_nonlinear_interactive_train(self, assumed_rank, total_iter = 5, train_epoch1 = 100, train_epoch2 = 100):

        model_tran = []
        flag = 0
        flag2 = 0
        for iter_time in range(0, total_iter):
            if flag == 0:
                model_PS = Sequential()
                model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5 * self.input.shape[1])))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=assumed_rank))
                model_PS.add(Dense(output_dim=np.int(1.5 * self.output.shape[1])))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=self.output.shape[1]))
                model_PS.add(Activation("linear"))
                model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                                 metrics=['mean_squared_error'])
                model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64,
                             validation_split=0.1, verbose=0)
            else:
                model_PS = Sequential()
                model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5 * self.input.shape[1]), weights=weights_arr[0]))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=assumed_rank, weights=weights_arr[2]))
                model_PS.add(Dense(output_dim=np.int(1.5 * self.output.shape[1]), weights=weights_arr[3]))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[5]))
                model_PS.add(Activation("linear"))
                model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                                 metrics=['mean_squared_error'])
                model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64,
                             validation_split=0.1, verbose=0)
                flag = 1

            weights_list = []
            for layer in model_PS.layers:
                weights_list.append(layer.get_weights())
            weights_arr = np.asarray(weights_list)
            flag = 1

            for i in range(0, self.nbl):
                self.output = np.dot(self.input, self.lhankel[i + 1])
                model = Sequential()
                model.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5 * self.input.shape[1]),
                                weights=weights_arr[0]))
                model.add(Activation("tanh"))
                model.add(Dense(output_dim=assumed_rank, weights=weights_arr[2]))
                #############################################
                if flag2==0:
                    model.add(Dense(output_dim=assumed_rank * 2))
                    model.add(Activation("linear"))
                    model.add(Dense(output_dim=assumed_rank))
                else:
                    model.add(Dense(output_dim = assumed_rank * 2, weights = temp_weights[0]))
                    model.add(Activation('linear'))
                    model.add(Dense(output_dim = assumed_rank))
                ############################################
                model.add(Dense(output_dim=np.int(1.5 * self.output.shape[1]), weights=weights_arr[3]))
                model.add(Activation('tanh'))
                model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[5]))
                model.add(Activation("linear"))
                model.compile(loss='mean_squared_error', optimizer=Adamax(),
                              metrics=['mean_squared_error'])
                model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch2, batch_size=64,
                          validation_split=0.1, verbose=0)
                temp_weights = []
                weights_list = []
                for layer in model.layers:
                    weights_list.append(layer.get_weights())
                weights_arr = np.asarray(weights_list)
                temp_weights.append(weights_arr[3])
                weights_arr = np.delete(weights_arr, 3, axis=0)
                #print weights_arr.shape
                temp_weights.append(weights_arr[3])
                weights_arr = np.delete(weights_arr, 3, axis=0)
                temp_weights.append(weights_arr[3])
                #print weights_arr.shape
                weights_arr = np.delete(weights_arr, 3, axis=0)
                #print weights_arr.shape
                flag2 = 1

                if iter_time >= total_iter-1:
                    model_tran.append(model)

        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)
        U1 = np.asarray(weights_arr[0][0])
        V1 = np.asarray(weights_arr[2][0])
        U2 = np.asarray(weights_arr[3][0])
        V2 = np.asarray(weights_arr[5][0])
        temp = self.lhankel[0].todense()[:, 0].reshape(-1, 1)
        temp = np.dot(np.linalg.pinv(U1), temp)
        temp = tanh_inv(temp)
        temp = np.dot(np.linalg.pinv(V1), temp)
        term = temp.reshape(-1, 1)

        temp = self.lhankel[0][0, :].todense().reshape(1, -1)
        temp = tanh_inv(temp.dot(np.linalg.pinv(V2)))
        temp = np.dot(temp, np.linalg.pinv(U2))
        init = temp.reshape(1, -1)

        self.init = init
        self.term = term

        A_sigma_one = []
        A_sigma_two = []
        for model in model_tran:
            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            A_sigma_one.append(weights_list[3])
            A_sigma_two.append(weights_list[5])
        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(
                Dense(input_dim=assumed_rank, output_dim=assumed_rank * 2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("linear"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)
        self.pred_model = pred_model


    def build_fac_nonlinear(self, assumed_rank, alpha, train_epoch1 = 100, train_epoch2 = 50):

        model_PS = Sequential()
        model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=assumed_rank))
        model_PS.add(Activation("tanh"))
        model_PS.add(Dense(output_dim=self.output.shape[1]))
        model_PS.add(Activation("linear"))
        model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                         metrics=['mean_squared_error'])
        model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64, validation_split=0.1, verbose=0)

        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)

        P = np.asarray(weights_arr[0][0])
        S = np.asarray(weights_arr[2][0])
        temp = self.lhankel[0].todense()[:, 0].reshape(1, -1)
        #print temp
        temp = tanh_inv(temp)
        #print np.isnan(temp)
        temp = np.dot(temp, np.linalg.pinv(P).transpose())
        #print np.isnan(temp)
        term = temp.reshape(-1, 1)
        #term = np.dot(np.linalg.pinv(P), new_hankel[:, 0])  # linear factorization
        init = np.dot(self.lhankel[0][0, :].todense(), np.linalg.pinv(S))
        init = init.reshape(1, -1)
        self.init = init
        self.term = term
        #self.init and self.term are all numpy array

        A_sigma_one = []
        A_sigma_two = []
        for i in range(0, self.nbl):
            #input = (input - np.mean(input, axis=0)) / np.std(input, axis=0)
            self.output = np.dot(self.input, self.lhankel[i + 1])
            #output = (output - np.mean(output, axis=0)) / np.std(output, axis=0)

            model = Sequential()
            model.add(Dense(input_dim=self.input.shape[1], output_dim=assumed_rank, weights=weights_arr[0], trainable=False))
            model.add(Activation("tanh"))
            #############################################
            model.add(Dense(output_dim=assumed_rank * 2))
            model.add(Activation("linear"))
            model.add(Dense(output_dim=assumed_rank))
            ############################################
            model.add(Activation("linear"))
            model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[2], trainable=False))
            model.add(Activation("linear"))
            model.compile(loss='mean_squared_error', optimizer=Adamax(),
                          metrics=['mean_squared_error'])
            model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch2, batch_size=64, validation_split=0.1, verbose=0)
            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            # print np.asarray(weights_list[2]).shape
            A_sigma_one.append(weights_list[2])
            A_sigma_two.append(weights_list[4])

        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(Dense(input_dim=assumed_rank, output_dim=assumed_rank*2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("linear"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)
        self.pred_model = pred_model

    def build_both_nonlinear_interactive_train(self, assumed_rank, total_iter = 5, train_epoch1 = 20, train_epoch2 = 10):
        model_tran = []
        flag = 0
        flag2 = 0
        for iter_time in range(0, total_iter):
            if flag == 0:
                model_PS = Sequential()
                model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5 * self.input.shape[1])))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=assumed_rank))
                model_PS.add(Dense(output_dim=np.int(1.5 * self.output.shape[1])))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=self.output.shape[1]))
                model_PS.add(Activation("linear"))
                model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                                 metrics=['mean_squared_error'])
                model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64,
                             validation_split=0.1, verbose=0)
            else:
                model_PS = Sequential()
                model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5 * self.input.shape[1]),
                                   weights=weights_arr[0]))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=assumed_rank, weights=weights_arr[2]))
                model_PS.add(Dense(output_dim=np.int(1.5 * self.output.shape[1]), weights=weights_arr[3]))
                model_PS.add(Activation("tanh"))
                model_PS.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[5]))
                model_PS.add(Activation("linear"))
                model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                                 metrics=['mean_squared_error'])
                model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64,
                             validation_split=0.1, verbose=0)
                flag = 1

            weights_list = []
            for layer in model_PS.layers:
                weights_list.append(layer.get_weights())
            weights_arr = np.asarray(weights_list)
            flag = 1

            for i in range(0, self.nbl):
                self.output = np.dot(self.input, self.lhankel[i + 1])
                model = Sequential()
                model.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(1.5 * self.input.shape[1]),
                                weights=weights_arr[0]))
                model.add(Activation("tanh"))
                model.add(Dense(output_dim=assumed_rank, weights=weights_arr[2]))
                #############################################
                if flag2 == 0:
                    model.add(Dense(output_dim=assumed_rank * 2))
                    model.add(Activation("tanh"))
                    model.add(Dense(output_dim=assumed_rank))
                else:
                    model.add(Dense(output_dim=assumed_rank * 2, weights=temp_weights[0]))
                    model.add(Activation('tanh'))
                    model.add(Dense(output_dim=assumed_rank))
                ############################################
                model.add(Dense(output_dim=np.int(1.5 * self.output.shape[1]), weights=weights_arr[3]))
                model.add(Activation('tanh'))
                model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[5]))
                model.add(Activation("linear"))
                model.compile(loss='mean_squared_error', optimizer=Adamax(),
                              metrics=['mean_squared_error'])
                model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch2, batch_size=64,
                          validation_split=0.1, verbose=0)
                temp_weights = []
                weights_list = []
                for layer in model.layers:
                    weights_list.append(layer.get_weights())
                weights_arr = np.asarray(weights_list)
                temp_weights.append(weights_arr[3])
                weights_arr = np.delete(weights_arr, 3, axis=0)
                # print weights_arr.shape
                temp_weights.append(weights_arr[3])
                weights_arr = np.delete(weights_arr, 3, axis=0)
                temp_weights.append(weights_arr[3])
                # print weights_arr.shape
                weights_arr = np.delete(weights_arr, 3, axis=0)
                # print weights_arr.shape
                flag2 = 1

                if iter_time >= total_iter - 1:
                    model_tran.append(model)

        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)
        U1 = np.asarray(weights_arr[0][0])
        V1 = np.asarray(weights_arr[2][0])
        U2 = np.asarray(weights_arr[3][0])
        V2 = np.asarray(weights_arr[5][0])
        temp = self.lhankel[0].todense()[:, 0].reshape(-1, 1)
        temp = np.dot(np.linalg.pinv(U1), temp)
        temp = tanh_inv(temp)
        temp = np.dot(np.linalg.pinv(V1), temp)
        term = temp.reshape(-1, 1)

        temp = self.lhankel[0][0, :].todense().reshape(1, -1)
        temp = tanh_inv(temp.dot(np.linalg.pinv(V2)))
        temp = np.dot(temp, np.linalg.pinv(U2))
        init = temp.reshape(1, -1)

        self.init = init
        self.term = term

        A_sigma_one = []
        A_sigma_two = []
        for model in model_tran:
            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            A_sigma_one.append(weights_list[3])
            A_sigma_two.append(weights_list[5])
        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(
                Dense(input_dim=assumed_rank, output_dim=assumed_rank * 2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("tanh"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)
        self.pred_model = pred_model

    def build_both_nonlinear(self, assumed_rank, alpha, train_epoch1 = 100, train_epoch2 = 50):

        model_PS = Sequential()
        model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(2 * self.input.shape[1])))
        model_PS.add(Activation("tanh"))
        model_PS.add(Dense(output_dim=assumed_rank))
        model_PS.add(Dense(output_dim=np.int(2 * self.output.shape[1])))
        model_PS.add(Activation("tanh"))
        model_PS.add(Dense(output_dim=self.output.shape[1]))
        model_PS.add(Activation("linear"))
        model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                         metrics=['mean_squared_error'])
        model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64,
                     validation_split=0.1, verbose=0)
        def tanh(x):
            return 2.0/(1.0+np.exp(-2.0*x)) - 1.0
        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)

        U1 = np.asarray(weights_arr[0][0])
        V1 = np.asarray(weights_arr[2][0])
        U2 = np.asarray(weights_arr[3][0])
        V2 = np.asarray(weights_arr[5][0])
        U1_T = U1.transpose()
        U2_T = U2.transpose()
        V1_T = V1.transpose()
        V2_T = V2.transpose()
        e1 = np.zeros(self.lhankel[0].shape[0])
        e1[0] = 1.0
        e3 = np.zeros(self.lhankel[0].shape[1])
        e3[0] = 1.0
        e3 = np.asmatrix(e3)
        e3_T = e3.transpose()
        e1 = np.asmatrix(e1)
        e2 = np.matrix(np.zeros(assumed_rank))
        e2[0] = 1.0
        IR = np.identity(assumed_rank)
        e2_T =e2.transpose()
        e1_T = e1.transpose()
        init = tanh(e1.dot(U1)).dot(V1)
        term = V2_T.dot(tanh(U2_T.dot(e2_T)))
        term = tanh(IR.dot(U2)).dot(V2).dot(e3_T)

        term = Sequential()
        term.add(Dense(input_dim = assumed_rank, output_dim = self.lhankel[0].shape[1] *2, weights = weights_arr[3], trainable = False))
        term.add(Activation('tanh'))
        term.add(Dense(output_dim = self.lhankel[0].shape[1], weights = weights_arr[5], trainable = False))


        self.init = init
        self.term = term

        A_sigma_one = []
        A_sigma_two = []
        for i in range(0, self.nbl):
            self.output = np.dot(self.input, self.lhankel[i + 1])
            model = Sequential()
            model.add(Dense(input_dim=self.input.shape[1], output_dim=np.int(2 * self.input.shape[1]),
                            weights=weights_arr[0], trainable=False))
            model.add(Activation("tanh"))
            model.add(Dense(output_dim=assumed_rank, weights=weights_arr[2], trainable=False))
            #############################################
            model.add(Dense(output_dim=assumed_rank * 2))
            model.add(Activation("tanh"))
            model.add(Dense(output_dim=assumed_rank))
            ############################################
            model.add(Dense(output_dim=np.int(2 * self.output.shape[1]), weights=weights_arr[3], trainable=False))
            model.add(Activation('tanh'))
            model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[5], trainable=False))
            model.add(Activation("linear"))
            model.compile(loss='mean_squared_error', optimizer=Adamax(),
                          metrics=['mean_squared_error'])
            model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch2, batch_size=64,
                      validation_split=0.1, verbose=0)
            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            A_sigma_one.append(weights_list[3])
            A_sigma_two.append(weights_list[5])

        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(Dense(input_dim=assumed_rank, output_dim=assumed_rank * 2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("tanh"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)
        self.pred_model = pred_model

    def build_tran_nonlinear(self, assumed_rank, alpha, train_epoch1 = 100, train_epoch2 = 50):

        model_PS = Sequential()
        model_PS.add(Dense(input_dim=self.input.shape[1], output_dim=assumed_rank))
        model_PS.add(Activation("linear"))
        model_PS.add(Dense(output_dim=self.output.shape[1]))
        model_PS.add(Activation("linear"))
        model_PS.compile(loss='mean_squared_error', optimizer=Adamax(),
                         metrics=['mean_squared_error'])
        model_PS.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch1, batch_size=64, validation_split=0.1, verbose=0)

        weights_list = []
        for layer in model_PS.layers:
            weights_list.append(layer.get_weights())
        weights_arr = np.asarray(weights_list)

        P = np.asarray(weights_arr[0][0])
        S = np.asarray(weights_arr[2][0])
        #temp = tanh_inv(new_hankel[:, 0].reshape(1, -1))

        #temp = self.lhankel[0][:, 0].todense().reshape(1, -1)
        #temp = np.dot(temp, np.linalg.pinv(P).transpose())
        #term = temp.reshape(-1, 1)

        term = Sequential()
        term.add(Dense(input_dim = assumed_rank, output_dim = self.output.shape[1], weights = weights_arr[2], trainable = False))


        #term = np.dot(np.linalg.pinv(P), new_hankel[:, 0])  # linear factorization
        init = np.dot(self.lhankel[0][0, :].todense(), np.linalg.pinv(S))
        #init = (init - self.input_mean[0])/self.input_std[0]
        init = init.reshape(1, -1)
        self.init = init
        self.term = term
        A_sigma_one = []
        A_sigma_two = []
        for i in range(0, self.nbl):
            self.output = np.dot(self.input, self.lhankel[i + 1])
            model = Sequential()
            model.add(Dense(input_dim=self.input.shape[1], output_dim=assumed_rank, weights=weights_arr[0], trainable=False))
            model.add(Activation("linear"))
            ######################################
            model.add(Dense(output_dim=assumed_rank*2))
            model.add(Activation("tanh"))
            model.add(Dense(output_dim=assumed_rank))
            ######################################
            model.add(Activation("linear"))
            model.add(Dense(output_dim=self.output.shape[1], weights=weights_arr[2], trainable=False))
            model.add(Activation("linear"))
            model.compile(loss='mean_squared_error', optimizer=Adamax(),
                          metrics=['mean_squared_error'])
            model.fit(self.input.todense(), self.output.todense(), nb_epoch=train_epoch2, batch_size=64, validation_split=0.1, verbose=0)


            weights_list = []
            for layer in model.layers:
                weights_list.append(layer.get_weights())
            weights_list = np.asarray(weights_list)
            A_sigma_one.append(weights_list[2])
            A_sigma_two.append(weights_list[4])

        pred_model = []
        for i in range(0, self.nbl):
            model = Sequential()
            model.add(Dense(input_dim=assumed_rank, output_dim=assumed_rank*2, weights=A_sigma_one[i], trainable=False))
            model.add(Activation("tanh"))
            model.add(Dense(output_dim=assumed_rank, weights=A_sigma_two[i], trainable=False))
            model.add(Activation("linear"))
            pred_model.append(model)

        self.pred_model = pred_model
    def spectral_method(self, train_file, assumed_rank):
        pT = Sample(adr=train_file)
        S_app = Learning(sample_instance=pT)
        linear_transition = S_app.BuildAutomatonFromHankel(self.lhankel, self.nbl, sparse=True)
        test = self.test
        total_count = 0.0
        corr_count = 0.0
        likelihood_van = []

        for i in range(0, test.shape[0]):
            print i
            for j in range(1, len(test[i])-1):
                likelihood = []
                for k in range(0, self.nbl):
                    #print np.append(test[i][0:j], k)
                    likelihood.append(linear_transition.val(np.append(test[i][0:j], k)))
                maxv = likelihood[0]
                ind = 0
                for l in range(0, len(likelihood)):
                    if maxv < likelihood[l]:
                        maxv = likelihood[l]
                        ind = l
                total_count += 1.0
                #print j
                #print i
                if ind == test[i][j]:
                    corr_count += 1.0
        return corr_count/total_count
    def spectral_method_likelihood(self, train_file, assumed_rank):
        pT = Sample(adr=train_file)
        S_app = Learning(sample_instance=pT)
        linear_transition = S_app.BuildAutomatonFromHankel(self.lhankel, self.nbl, assumed_rank, sparse=True)
        #linear_transition.initial /= S_app.sample_object.nbEx
        test = self.test
        likelihood_van = []
        for i in range(0, test.shape[0]):
            likelihood_van.append(linear_transition.val(test[i]))
        self.init = np.asarray(linear_transition.initial).reshape(1, -1)
        self.term = np.asarray(linear_transition.final).reshape(-1, 1)
        self.pred_model = np.asarray(linear_transition.transitions)
        return np.asarray(likelihood_van)
    def update_belif_states(self, symbol, linear_tran):
        if linear_tran == True:
            numerator = np.dot(self.init, self.pred_model[symbol])
        else:
            numerator = self.pred_model[symbol].predict(self.init)
        denom = np.dot(numerator, self.term)
        self.belif_vec = (numerator/denom).reshape(1, -1)

    def reset_belif_states(self):
        self.belif_vec = self.init

    def get_symbol_prediction(self, linear_tran):
        predicted_symbol = -1
        maxscore = np.finfo(float).eps
        for symbol in range(0, len(self.lhankel)-1):
            symbol_score = self.get_obs_prob(symbol, linear_tran)
            if symbol_score > maxscore:
                predicted_symbol = symbol
                maxscore = symbol_score
        return predicted_symbol

    def get_obs_prob(self, symbol, linear_tran):
        if linear_tran == True:
            prob = self.belif_vec.dot(self.pred_model[symbol])
        else:
            prob = self.pred_model[symbol].predict(self.belif_vec)
        prob = prob.dot(self.term)
        prob = max(prob, np.finfo(float).eps)
        return prob
    def cal_WER(self, linear_tran, test = None):
        if test == None: test = self.test
        errors = 0.0
        numpredictions = 0.0
        self.belif_vec = self.init
        for seq in test:
            for obs in seq:
                numpredictions += 1.0
                predsymbol = self.get_symbol_prediction(linear_tran)
                self.update_belif_states(obs, linear_tran)
                if predsymbol != obs:
                    errors += 1.0
            self.reset_belif_states()
        return float(errors)/float(numpredictions)

    def find_max_index(self, x):
        pred_model = self.pred_model
        likelihood = []
        for i in range(0, self.nbl):
            temp = pred_model[i].predict(x)
            temp = temp.dot(self.term)
            likelihood.append(temp)
        likelihood = np.asarray(likelihood).ravel()
        #print likelihood.shape
        maxv = likelihood[0]
        #print maxv.shape
        ind = 0
        for i in range(0, len(likelihood)):
            if maxv < likelihood[i]:
                maxv = likelihood[i]
                ind =i
        return ind
    def predict_next_symbol(self, test = None):
        if test == None: test = self.test
        init = self.init
        term = self.term
        pred_model = self.pred_model
        total_count = 0.0
        corr_count = 0.0
        for i in range(0, test.shape[0]):
            for j in range(1, len(test[i])-1):
                temp = init.reshape(1, -1)
                for k in range(0, j):
                    temp = pred_model[test[i][k]].predict(temp)
                total_count += 1.0
                if self.find_max_index(temp) == test[i][j]:
                    corr_count += 1.0
        return corr_count/total_count
    def predict_likelihood(self, test = None):
        if test == None: test = self.test
        likelihood_net = []
        init = self.init
        term = self.term
        pred_model = self.pred_model
        for i in range(0, test.shape[0]):
            temp = init.reshape(1, -1)
            for j in range(0, len(test[i])):
                temp = pred_model[test[i][j]].predict(temp)
                #print np.isnan(temp)
            #print np.isnan(term)
            #temp = temp.dot(term)
            temp = term.predict(temp)[0][0]
            #print temp.shape
            #print temp[0]
            #print np.isnan(temp)
            #if np.isnan(temp) == True:
            #    temp = 0
            #    print 'here'
            likelihood_net.append(temp.ravel())
        return np.asarray(likelihood_net)
    def classification(self, model_likelihood, true_result):
        count = 0.
        temp = np.mean(model_likelihood)
        for i in range(0,len(model_likelihood)):
            if model_likelihood[i] > temp:
                model_likelihood[i] = 1.0
            else:
                model_likelihood[i] = 0.0
            if model_likelihood[i] == true_result[i]:
                count += 1.
        return count/len(model_likelihood)