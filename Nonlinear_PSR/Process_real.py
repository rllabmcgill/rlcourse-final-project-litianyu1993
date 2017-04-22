import create_files as cf
import numpy as np
import pebl
import csv
import random
from sklearn.utils import shuffle
num_files = 4
from sklearn.model_selection import KFold
with cf.cd('./temp'):
    df = []
    for i in range(1, 15):
        data = np.genfromtxt('temp'+str(i)+'.csv', delimiter=',')
        data = data.astype(int)
        df.append(data)
    df = np.asarray(df).reshape(14, -1)
    #df = df.transpose()
    #print df
    df_seq = []
    for i in range(0, 14):
        temp = df[i][0]
        temp_sep = []
        for j in range(0, len(temp) - 20):
            temp_sep.append(temp[j: j+20])
        df_seq.append(np.asarray(temp_sep))
    df_seq = np.asarray(df_seq)
    for i in range(0,14):
        test = []
        flag = 0
        for j in range(0, 14):
            if j != i:
                if flag == 0:
                    train = df_seq[j]
                    flag = 1
                else:
                    #print train.shape
                    train = np.vstack([train, df_seq[j]])
                    #print train.shape
        test = df[i]
        #train = random.shuffle(train)
        train = shuffle(train)
        print train.shape
        print test[0]
        np.savetxt("train" + str(i) + '.txt', train-1, fmt='%i', delimiter=' ')
        np.savetxt("test" + str(i) + '.txt', test[0].reshape(1, -1)-1, fmt='%i', delimiter=' ')



'''
for i in range(0, num_files):
    with cf.cd("./Data/real"+str(i+1)):
        data = np.genfromtxt("real.csv", delimiter=',')
        temp = []
        i = 0
        count = 0
        max_num = np.max(data)
        min_num = np.min(data)
        data = pebl.data.fromfile("real.csv")
        data = pebl.discretizer.maximum_entropy_discretize(indata=data, numbins=5)
        data.tofile('disc_data.txt')





for i in range(0, num_files):
    with cf.cd("./Data/real"+str(i+1)):
        indexs = []
        datas = []
        df =[]
        with open('disc_data.txt', 'r') as f:
            next(f)  # skip headings
            reader = csv.reader(f, delimiter='\t')
            for index, data in reader:
                indexs.append(index)
                datas.append(data)
            temp = []
            for i in range(0, len(datas)):
                temp.append(int(datas[i]))
            temp = np.asarray(temp)
            print temp
            for i in range(0, 10000):
                randm = np.random.randint(0, len(temp)-20, size = 1)
                df.append(temp[randm:randm+20])
            df = np.asarray(df)
            np.savetxt('ready_data.csv', df, fmt='%i',delimiter=',')

for i in range(0, num_files):
    with cf.cd("./Data/real"+str(i+1)):
        data = np.genfromtxt('ready_data.csv', delimiter=',')
        kf = KFold(n_splits = 10)
        index = 0
        for train_index, test_index in kf.split(data):

            data_train, data_test = data[train_index], data[test_index]
            data_train = np.insert(data_train, 0, 20, axis = 1)
            data_test = np.insert(data_test, 0, 20, axis = 1)
            np.savetxt("train"+str(index)+'.txt', data_train, fmt='%i',delimiter = ' ')
            np.savetxt("test" + str(index) + '.txt', data_test, fmt='%i', delimiter=' ')
            with file("train"+str(index)+'.txt', 'r') as original: data2 = original.read()
            with file("train"+str(index)+'.txt', 'w') as modified: modified.write("10000 15\n" + data2)
            index += 1
'''