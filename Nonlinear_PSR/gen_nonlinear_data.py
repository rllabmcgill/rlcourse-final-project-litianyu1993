import numpy as np
import create_files as cf

def logistic(x):
    return 1.0/(1.0+np.exp(-x))
def tran(m1,m2,x):
    return logistic(np.dot(x, m1)).dot(m2)
def move_to_next_state(m1_vec, m2_vec, x, sym):
    temp = tran(m1_vec[sym], m2_vec[sym], x)
    return sum_to_one(temp)
def term(m1, m2, x):
    return logistic(np.dot(x, m1)).dot(m2)
def sum_to_one(x):
    return x/np.sum(x)
def sum_to_one_col(x):
    return x/np.sum(x, axis =1)
def emit(m_e, x):
    return x.dot(m_e)
def sample_emis(x):
    x = sum_to_one(x)
    sample = np.random.multinomial(1, x, size=1).ravel()
    for i in range(0, len(sample)):
        if sample[i] != 0:
            return i
def decide_term(x, m1_vec, m2_vec, symbol, m1_term, m2_term):
    temp = term(m1_term, m2_term, x)
    next = []
    for i in range(0, len(symbol)):
        next.append(term(m1_term, m2_term, move_to_next_state(m1_vec, m2_vec, x, symbol[i])))
    #print next
    for i in range(0, len(symbol)):
        if max(next[i]) >= temp:
            return False
    return True

rank = [100]
symbol = [0, 1, 2, 3]
total_num = 20000
num_sample = 20
num_test = 1000
max_len_train = 30
max_len_test = 20
with cf.cd('./sythetic'):
    for k in range(0, len(rank)):
        structure_m = []
        with cf.cd('./sy'+str(k)):
            for i in range(0, rank[k]):
                temp = []
                for l in range(0, rank[k]):
                    if l == i+1:
                        temp.append(0.2)
                    elif i == rank[k] and l == 0:
                        temp.append(0.2)
                    else: temp.append(0)
                temp = np.asarray(temp)
                structure_m.append(temp)
            structure_m = np.asarray(structure_m)
            print '***************************'
            init = np.random.rand(rank[k])
            init = sum_to_one(init)
            m1_term = np.random.rand(rank[k], rank[k]) + structure_m
            m2_term = np.random.rand(rank[k], 1)
            m1 = []
            m2 = []
            for j in range(0, len(symbol)):
                m1.append(np.random.rand(rank[k], rank[k])+ structure_m)
                m2.append(np.random.rand(rank[k], rank[k])+ structure_m)
            m_e = np.random.rand(rank[k], len(symbol))
            total_sample = []
            temp_init = init
            for j in range(0, total_num):
                single_sample = []
                for i in range(0, max_len_train):
                    temp = emit(m_e, temp_init)
                    letter = sample_emis(temp)
                    single_sample.append(letter)
                    temp_init = move_to_next_state(m1, m2, temp_init, letter)
                    if decide_term(temp_init, m1, m2, symbol, m1_term, m2_term) == True:
                        break
                single_sample = np.insert(np.asarray(single_sample), 0, len(single_sample))
                total_sample.append(np.asarray(single_sample))
            total_sample = np.asarray(total_sample)
            f = open('train_pauto3.txt', 'w')
            f.write(str(total_num) +' '+str(len(symbol))+'\n')
            for j in range(0, len(total_sample)):
                for i in range(0, len(total_sample[j])):
                    f.write(str(total_sample[j][i])+' ')
                f.write('\n')
            f.close()

            temp_init = init
            total_sample = []
            test_likelihood = []
            for j in range(0, num_test):
                single_sample = []
                sample_size = np.random.randint(1, max_len_test, 1)
                sample = np.random.randint(0, len(symbol), size=sample_size)
                for i in range(0, len(sample)):
                    #temp = emit(m_e, init)
                    letter = sample[i]
                    single_sample.append(letter)
                    temp_init = move_to_next_state(m1, m2, temp_init, letter)
                test_likelihood.append(term(m1_term, m2_term, temp_init))
                single_sample = np.insert(np.asarray(single_sample), 0, len(single_sample))
                total_sample.append(np.asarray(single_sample))
            total_sample = np.asarray(total_sample)
            test_likelihood = np.asarray(test_likelihood).reshape(-1, 1)
            #test_likelihood = sum_to_one(test_likelihood)
            test_likelihood -= min(test_likelihood)
            test_likelihood = sum_to_one(test_likelihood)
            np.savetxt('likelihood_pauto3.txt', test_likelihood, delimiter=' ')
            f = open('test_pauto3.txt', 'w')
            f.write(str(num_test) +' ' + str(len(symbol)) + '\n')
            for j in range(0, len(total_sample)):
                for i in range(0, len(total_sample[j])):
                    f.write(str(total_sample[j][i]) + ' ')
                f.write('\n')
            f.close()



