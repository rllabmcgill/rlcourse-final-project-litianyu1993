import os
import pandas as pd
import numpy as np
import sp2learn as sp
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def creat_filesss(folder_path, train_file, model_file, num_cross_val):
    with cd(folder_path):

        # read training files
        #train_file = 'train_pauto3.txt'
        #model_file = 'model_pauto3.txt'
        trainning_set = []
        with open(train_file, 'r') as lines:
            for line in lines:
                temp = line.split(' ')
                # print temp
                temp2 = []
                for i in range(0, len(temp)):
                    # print temp[i]
                    if temp[i] != '\n':
                        temp2.append(np.int(float(temp[i])))
                trainning_set.append(temp2)
        # trainning_set = np.asarray(trainning_set)
        extra_num = trainning_set[0]
        trainning_set = trainning_set[1:]

        trainning_set_temp = []
        for i in range(0, len(trainning_set)):
            temp3 = []
            for j in range(1, len(trainning_set[i])):
                temp3.append(trainning_set[i][j])
            trainning_set_temp.append(temp3)

        # calculate training likelihood
        true_model = sp.Automaton.load_Spice_Automaton(model_file)
        train_likelihood = []
        num_samples = len(trainning_set)
        for i in range(0, len(trainning_set)):
            train_likelihood.append(true_model.val(trainning_set_temp[i]))
        train_likelihood = np.asarray(train_likelihood)
        num_train_samples = np.int(0.7 * num_samples)
        for i in range(0, num_cross_val):
            rand_index_train = np.random.choice(range(num_samples), num_train_samples, replace=False)

            rand_index_test = []
            for j in range(0, num_samples):
                if j not in rand_index_train:
                    rand_index_test.append(j)
            rand_index_test = np.asarray(rand_index_test)
            rand_index_train = np.sort(rand_index_train)
            #print rand_index_test.shape
            trainning_set = np.asarray(trainning_set)
            train_likelihood = np.asarray(train_likelihood)
            train_samples = trainning_set[rand_index_train]
            # train_samples = train_samples[0]
            test_samples = trainning_set[rand_index_test]
            test_likelihood = train_likelihood[rand_index_test]
            print test_likelihood.shape
            f = open("train_cross_" + str(i) + ".txt", 'w')
            f.write(str(len(train_samples)) + ' ')
            f.write(str(extra_num[1]))
            f.write('\n')
            for k in range(0, len(train_samples)):
                for l in range(0, len(train_samples[k])):
                    f.write(str(train_samples[k][l]) + ' ')
                f.write('\n')
            f.close()

            f = open("test_cross_" + str(i) + ".txt", 'w')
            for k in range(0, len(test_samples)):
                for l in range(0, len(test_samples[k])):
                    f.write(str(test_samples[k][l]) + ' ')
                f.write('\n')
            f.close()

            np.savetxt("likelihood_cross_" + str(i) + ".txt", test_likelihood)
            #print train_samples
            #print test_samples
            #print test_likelihood

#creat_filesss("./Data/pauto3", 'train_pauto3.txt', 'model_pauto3.txt', 10)
