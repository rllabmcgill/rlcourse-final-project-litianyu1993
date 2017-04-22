import numpy as np
import os.path
from keras.regularizers import l2

import Test_class as TC
import create_files as cf
num_files = 7
def sum_to_one(x):
    x = abs(x)
    return x/sum(x)
assumed_rank = [100, 200, 300, 400, 500]
sp_wer = []
fac_wer = []
tran_wer = []
both_wer = []


for i in range(5, num_files):
    with cf.cd("./Data/real"+str(i+1)):

        for j in range(0, len(assumed_rank)):
            total_sp_wer = []
            total_fac_wer = []
            total_tran_wer = []
            total_both_wer = []
            for i in range(0, 14):
                train_file = 'train' + str(i) + '.txt'
                test_file = 'test' + str(i ) + '.txt'
                test_result_file = 'test' + str(i) + '.txt'
                tc = TC.test(train_file, test_file, test_result_file)
                sp_temp = tc.spectral(assumed_rank=assumed_rank[j])
                sp_temp = sum_to_one(sp_temp)
                total_sp_wer.append(tc.WER)
                print "Spectral For state number "+str(assumed_rank[j])+'of file ' + str(i) + ' is:'
                sp_wer.append(tc.WER)
                print sp_wer[-1]

                if os.path.exists("input.csv") == True:
                    notchange = 1
                else:
                    notchange = 0

                fac_temp = tc.fac_nonlinear(assumed_rank=assumed_rank[j], lr=0.1, num_epoch=100,
                                                not_first=notchange)
                #fac_temp = sum_to_one(fac_temp)
                print "Fac_non state number " + str(assumed_rank[j]) + 'of file ' + str(i + 1) + ' is:'
                print tc.WER
                total_fac_wer.append(tc.WER)

                tran_temp = tc.tran_nonlinear(assumed_rank=assumed_rank[j], lr=0.1, num_epoch=200,
                                            not_first=notchange)
                #tran_temp = sum_to_one(tran_temp)
                print "tran_non state number " + str(assumed_rank[j]) + 'of file ' + str(i + 1) + ' is:'
                print tc.WER
                total_tran_wer.append(tc.WER)

                both_temp = tc.both_nonlinear(assumed_rank=assumed_rank[j], lr=0.1, num_epoch=200,
                                                not_first=notchange)
                #both_temp = sum_to_one(both_temp)
                print "Both_non state number " + str(assumed_rank[j]) + 'of file ' + str(i + 1) + ' is:'
                print tc.WER
                total_both_wer.append(tc.WER)

            total_sp_wer = np.asarray(total_sp_wer)
            total_fac_wer = np.asarray(total_fac_wer)
            total_tran_wer = np.asarray(total_tran_wer)
            total_both_wer = np.asarray(total_both_wer)

            sp_wer.append(np.mean(total_sp_wer))
            fac_wer.append(np.mean(total_fac_wer))
            tran_wer.append(np.mean(total_tran_wer))
            both_wer.append(np.mean(total_both_wer))

        sp_result = np.asarray(sp_wer).reshape(len(assumed_rank), -1)
        tran_result = np.asarray(tran_wer).reshape(len(assumed_rank), -1)
        fac_result = np.asarray(fac_wer).reshape(len(assumed_rank), -1)
        both_result = np.asarray(both_wer).reshape(len(assumed_rank), -1)

        np.savetxt("sp_result.csv", sp_result, delimiter=',')
        np.savetxt("tran_result.csv", tran_result, delimiter=',')
        np.savetxt("fac_result.csv", fac_result, delimiter=',')
        np.savetxt("both_result.csv", both_result, delimiter=',')

        print "sp result: "+ str(sp_result)
        print "tran_result: "+str(tran_result)
        print "fac_result: "+str(fac_result)
        print "both_result: "+str(both_result)
