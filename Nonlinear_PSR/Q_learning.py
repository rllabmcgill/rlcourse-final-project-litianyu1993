import numpy as np
import Freeze_version_new as FV
import Test_class as tc
import matplotlib.pyplot as plt
import time
from sp2learn import Sample, Learning, Hankel

class Q_learning:
    def __init__(self, PSR_num_states, num_actions, train_file, test_file, test_result_file, type):
        self.num_actions = num_actions
        self.actions =  np.arange(num_actions)
        self.test_file = test_file
        self.test_result_file = test_result_file
        self.train_file = train_file
        self.type = type
        self.PSR_num_states = PSR_num_states
    def deciper_action(self, x):
        for i in range(0, len(x)):
            if x[i] == 0:
                x[i] = 0
            else:
                x[i] = 1
        return x
    def deciper_reward(self, x):
        for i in range(0, len(x)):
            if x[i] == 2:
                x[i] = 10000
            else:
                x[i] = 0
        return x
    def taking_argmax_action(self, w, current_state, actions):
        actions = np.zeros(self.num_actions)
        ini_actions = actions
        ini_actions[0] = 1
        x = np.insert(current_state, len(current_state), ini_actions)
        x = np.append(x, 1)
        x = np.squeeze(x)
        w = np.squeeze(w)
        max_v = np.dot(x, w)
        max_action = actions[0]
        for i in range(0, len(actions)):
            actions = np.zeros(self.num_actions)
            actions[i] = 1
            x = np.insert(current_state, len(current_state), actions)
            x = np.append(x, 1)
            temp = np.dot(x, w)
            print temp
            #print x
            if temp > max_v:
                max_v = temp
                max_action = actions[i]
        return max_action, max_v
    def Q_learner(self, gamma = 0.9, alpha = 0.01):
        pT = Sample(adr=self.train_file)
        cols = pT.select_columns(1000)
        rows = pT.select_rows(1000)
        self.lhankel = Hankel(sample_instance=pT, lcolumns=cols, lrows=rows,
                              version="classic",
                              partial=True, sparse=True).lhankel
        linear = False
        if self.type == 'linear':
            linear = True
            fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file,
                           geninput=1)
            fv.spectral_method_likelihood(self.train_file, self.PSR_num_states)
            self.model_tran = fv.pred_model
            self.model_init = fv.init
            self.model_term = fv.term
        elif self.type == 'tran':
            fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file,
                           geninput=1)
            fv.build_tran_nonlinear(assumed_rank=self.PSR_num_states, alpha=0.1, train_epoch1=100)
            self.model_tran = fv.pred_model
            self.model_init = fv.init
            self.model_term = fv.term
        elif self.type == 'fac':
            fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file,
                           geninput=1)
            fv.build_fac_nonlinear_new(assumed_rank=self.PSR_num_states, alpha=0.1, train_epoch1=100)
            self.model_tran = fv.pred_model
            self.model_init = fv.init
            self.model_term = fv.term
        else:
            fv = FV.freeze(self.lhankel, self.train_file, testfile=self.test_file, testresultfile=self.test_result_file,
                           geninput=1)
            fv.build_both_nonlinear(assumed_rank=self.PSR_num_states, alpha=0.1, train_epoch1=100)
            self.model_tran = fv.pred_model
            self.model_init = fv.init
            self.model_term = fv.term
        self.fv = fv

        diff = []
        w = np.random.rand(self.PSR_num_states+self.num_actions+1)
        w = np.squeeze(w)

        #print np.asarray(self.model_init).reshape(-1,)
        for i in range(0, len(self.aos)):
            ao = self.aos[i]
            current_state = np.asarray(self.model_init).reshape(-1,)
            #print current_state.shape
            actions = self.deciper_action(ao)
            reward = self.deciper_reward(ao)
            for j in range(0, len(ao)-1):
                current_state = np.asarray(current_state).reshape(1,-1)
                #print current_state
                next_state = fv.transfer_next_state(current_state, ao[j], linear)
                #print next_state
                next_state = np.asarray(next_state).reshape(-1, )
                current_state = np.asarray(current_state).reshape(-1, )
                #print next_state
                max_action, max_v = self.taking_argmax_action(w, next_state, self.actions)
                #print max_v
                #print current_state.shape
                #print actions[j]
                acs = np.zeros(self.num_actions)
                acs[actions[j]] = 1
                x = np.insert(current_state, len(current_state),acs)
                x = np.insert(x, len(x), 1)
                x = np.squeeze(x)
                current_value = np.dot(w,x)
                #print current_value
                #time.sleep(1)
                #print x
                delta = reward[j+1] + gamma*max_v - current_value
                if reward[j+1] == 0:
                    if delta <= -1:
                        delta = -1
                    elif delta >= 1:
                        delta = 1
                #print delta
                w_new = w + alpha*delta*x
                diff.append(np.sum(abs(w_new-w)))
                #print w
                print w
                current_state = next_state
            #time.sleep(1)
        print diff
        plt.plot(diff)
        plt.show()
        self.diff = diff

        self.w = w

    def test_Q(self, num_states = 20, steps = 1000):
        current_state = self.model_init
        index = np.random.randint(0, num_states, 1)
        reward = 0
        for i in range(0, steps):
            current_state = np.asarray(current_state).reshape(-1, )
            max_action, max_v = self.taking_argmax_action(self.w, current_state, self.actions)
            print max_action
            prob = np.random.rand(1)
            if max_action == 0:
                if prob <= 0.5:
                    if index != 0:
                        index -= 1
                else:
                    if index != self.PSR_num_states - 1:
                        index += 1
                obs = 0
            elif max_action == 1:
                if index == self.PSR_num_states - 1:
                    print 'here'
                    obs = 1
                    reward += 1
                else:
                    obs = 0
                index = self.PSR_num_states - 1
            if max_action == 0 and obs == 0:
                ao = 0
            elif max_action == 1 and obs == 0:
                ao = 1
            elif max_action == 1 and obs == 1:
                ao = 2
            if self.type == 'linear':
                linear = True
            else: linear = False
            current_state = np.asarray(current_state).reshape(1, -1)
            current_state = self.fv.transfer_next_state(current_state, ao, linear)
        return reward
    def gen_grid(self, num_samples = 100):
        return 0
    def gen_float_reset(self, num_samples=100, num_states=20):

        aos = []
        for i in range(0, num_samples):
            ao = []
            starting = np.random.randint(0, num_states, 1)
            index = starting
            action = np.random.randint(0, 2, 1)  # 1 as reset, 0 as float
            length = np.random.randint(2, 50, 1)
            count = 0
            while count <= length:
                count += 1
                prob = np.random.rand(1)
                if action == 0:
                    if prob <= 0.5:
                        if index != 0:
                            index -= 1
                    else:
                        if index != num_states - 1:
                            index += 1
                    obs = 0
                elif action == 1:
                    if index == num_states - 1:
                        obs = 1
                    else:
                        obs = 0
                    index = num_states - 1
                if action == 0 and obs == 0:
                    ao.append(0)
                elif action == 1 and obs == 0:
                    ao.append(1)
                elif action == 1 and obs == 1:
                    ao.append(2)
                else:
                    ao.append(3)
                action = np.random.randint(0, 2, 1)
            aos.append(np.asarray(ao))
            self.aos = aos
        return aos
    def gen_data(self, k, num_samples = 20000):
        aos = self.gen_float_reset(num_samples, k)
        f = open('train_pauto3.txt', 'w')
        f.write(str(len(aos)) + ' ' + str(3) + '\n')
        for j in range(0, len(aos)):
            f.write(str(len(aos[j])) + ' ')
            # len(aos[j])
            for i in range(0, len(aos[j])):
                f.write(str(aos[j][i]) + ' ')
            f.write('\n')
        f.close()
        aos = self.gen_float_reset(500, k)
        temp = len(aos)
        for i in range(0, 500):
            length = np.random.randint(2, 50, 1)
            aos.append(np.random.randint(0, 3, length))
        f = open('test_pauto3.txt', 'w')
        for j in range(0, len(aos)):
            f.write(str(len(aos[j])) + ' ')
            for i in range(0, len(aos[j])):
                f.write(str(aos[j][i]) + ' ')
            f.write('\n')
        f.close()

        result = []
        for j in range(0, temp):
            result.append(1)
        for j in range(0, 500):
            result.append(0)
        np.savetxt('likelihood_pauto3.txt', (result), delimiter=' ')

ql = Q_learning(10, 2, "train_pauto3.txt", 'test_pauto3.txt', 'likelihood_pauto3.txt', 'linear')
ql.gen_data(20, 1000)
ql.Q_learner(alpha = 0.01)
print ql.test_Q(100, 1000)

