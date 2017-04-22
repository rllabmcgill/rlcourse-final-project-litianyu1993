import numpy as np
import os.path
import matplotlib.pyplot as plt
import Test_class as TC
import create_files as cf
num_files = 3
def sum_to_one(x):
    x = abs(x)
    return x/sum(x)
assumed_rank = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]

sp_per = []
sp_l1 = []
sp_l2 = []
sp_wer = []

fac_per = []
fac_l1 = []
fac_l2 = []
fac_wer = []
fac_var = []

tran_per = []
tran_l1 = []
tran_l2 = []
tran_wer = []
tran_var = []

both_per = []
both_l1 = []
both_l2 = []
both_wer = []
both_var = []


num_repeat = 1
for i in range(2, num_files):
    with cf.cd("./Data/pauto"+str(i+1)):
        train_file = 'train_pauto' + str(i+1) + '.txt'
        test_file = 'test_pauto' + str(i+1) + '.txt'
        test_result_file = 'likelihood_pauto' + str(i+1) + '.txt'
        tc = TC.test(train_file, test_file, test_result_file)
        for j in range(0, len(assumed_rank)):

            sp_temp = tc.spectral(assumed_rank=assumed_rank[j])
            sp_temp -= min(sp_temp)
            sp_temp = sum_to_one(sp_temp)
            sp_wer.append(tc.WER)

            sp_per.append(tc.cal_perprelexity(sp_temp, tc.test_true_result))
            sp_l1.append(tc.cal_l1(sp_temp, tc.test_true_result))
            sp_l2.append(tc.cal_l2(sp_temp, tc.test_true_result))
            plt.plot(np.log(sp_temp), color = 'red')
            plt.plot(np.log(tc.test_true_result), color = 'green')
            plt.savefig("sp_"+str(i)+str(j)+'.png')
            plt.close()
            print "Spectral For state number "+str(assumed_rank[j])+'of file ' + str(i+1) + ' is:'
            print sp_per[-1]
            print sp_l1[-1]
            print sp_l2[-1]
            print sp_wer[-1]

            total_fac_per = []
            total_fac_l1 = []
            total_fac_l2 = []
            total_fac_wer = []

            total_tran_per =[]
            total_tran_l1 = []
            total_tran_l2 = []
            total_tran_wer = []

            total_both_per = []
            total_both_l1 = []
            total_both_l2 = []
            total_both_wer = []

            for k in range(0, num_repeat):
                if os.path.exists("input.csv") == True:
                    notchange = 1
                else:
                    notchange = 0


                tran_temp = tc.tran_nonlinear(assumed_rank=assumed_rank[j], lr=0.1, num_epoch=200,
                                            not_first=notchange)
                tran_temp -= min(tran_temp)
                tran_temp = sum_to_one(tran_temp)
                print "tran_non state number " + str(assumed_rank[j]) + 'of file ' + str(i + 1) + ' is:'
                print tc.cal_perprelexity(tran_temp, tc.test_true_result)
                print tc.cal_l1(tran_temp, tc.test_true_result)
                print tc.cal_l2(tran_temp, tc.test_true_result)
                print tc.WER
                total_tran_per.append(tc.cal_perprelexity(tran_temp, tc.test_true_result))
                total_tran_l1.append(tc.cal_l1(tran_temp, tc.test_true_result))
                total_tran_l2.append(tc.cal_l2(tran_temp, tc.test_true_result))
                total_tran_per.append(tc.WER)
                plt.plot(np.log(tran_temp), color='red')
                plt.plot(np.log(tc.test_true_result), color='green')
                plt.savefig("tran_" + str(i) + str(j)+'.png')
                plt.close()


                fac_temp = tc.fac_nonlinear(assumed_rank=assumed_rank[j], lr=0.1, num_epoch=100,
                                             not_first=notchange)
                fac_temp -= min(fac_temp)
                fac_temp = sum_to_one(fac_temp)
                print "Fac_non state number " + str(assumed_rank[j]) + 'of file ' + str(i + 1) + ' is:'
                print tc.cal_perprelexity(fac_temp, tc.test_true_result)
                print tc.cal_l1(fac_temp, tc.test_true_result)
                print tc.cal_l2(fac_temp, tc.test_true_result)
                print tc.WER
                total_fac_per.append(tc.cal_perprelexity(fac_temp, tc.test_true_result))
                total_fac_l1.append(tc.cal_l1(fac_temp, tc.test_true_result))
                total_fac_l2.append(tc.cal_l2(fac_temp, tc.test_true_result))
                total_fac_per.append(tc.WER)
                plt.plot(np.log(fac_temp), color='red')
                plt.plot(np.log(tc.test_true_result), color='green')
                plt.savefig("fac_" + str(i) + str(j)+'.png')
                plt.close()


                both_temp = tc.both_nonlinear(assumed_rank=assumed_rank[j], lr=0.1, num_epoch=50,
                                              not_first=notchange)
                both_temp -= min(both_temp)
                both_temp = sum_to_one(both_temp)
                print "Both_non state number " + str(assumed_rank[j]) + 'of file ' + str(i + 1) + ' is:'
                print tc.cal_perprelexity(both_temp, tc.test_true_result)
                print tc.cal_l1(both_temp, tc.test_true_result)
                print tc.cal_l2(both_temp, tc.test_true_result)
                print tc.WER
                total_both_per.append(tc.cal_perprelexity(both_temp, tc.test_true_result))
                total_both_l1.append(tc.cal_l1(both_temp, tc.test_true_result))
                total_both_l2.append(tc.cal_l2(both_temp, tc.test_true_result))
                total_both_wer.append(tc.WER)
                plt.plot(np.log(both_temp), color='red')
                plt.plot(np.log(tc.test_true_result), color='green')
                plt.savefig("both_" + str(i) + str(j)+'.png')
                plt.close()

            total_fac_l2 = np.asarray(total_fac_l2)
            total_fac_l1 = np.asarray(total_fac_l1)
            total_fac_per = np.asarray(total_fac_per)
            total_fac_wer = np.asarray(total_fac_wer)

            total_tran_l2 = np.asarray(total_tran_l2)
            total_tran_l1 = np.asarray(total_tran_l1)
            total_tran_per = np.asarray(total_tran_per)
            total_tran_wer = np.asarray(total_tran_wer)

            total_both_l2 = np.asarray(total_both_l2)
            total_both_l1 = np.asarray(total_both_l1)
            total_both_per = np.asarray(total_both_per)
            total_both_wer = np.asarray(total_both_wer)

            fac_per.append(np.mean(total_fac_per))
            fac_l1.append(np.mean(total_fac_l1))
            fac_l2.append(np.mean(total_fac_l2))
            fac_wer.append(np.mean(total_fac_wer))
            fac_var.append(np.std(total_fac_per))

            tran_per.append(np.mean(total_tran_per))
            tran_l1.append(np.mean(total_tran_l1))
            tran_l2.append(np.mean(total_tran_l2))
            tran_wer.append(np.mean(total_tran_wer))
            tran_var.append(np.std(total_tran_per))

            both_per.append(np.mean(total_both_per))
            both_l1.append(np.mean(total_both_l1))
            both_l2.append(np.mean(total_both_l2))
            both_wer.append(np.mean(total_both_wer))
            both_var.append(np.std(total_both_per))


        sp_result = np.asarray([sp_per, sp_l1, sp_l2, sp_wer]).reshape(len(assumed_rank), -1)
        tran_result = np.asarray([tran_per, tran_l1, tran_l2, tran_wer, tran_var]).reshape(-1, len(assumed_rank))
        fac_result = np.asarray([fac_per, fac_l1, fac_l2, fac_wer, fac_var]).reshape(-1, len(assumed_rank))
        both_result = np.asarray([both_per, both_l1, both_l2, both_wer, both_var]).reshape(-1, len(assumed_rank))

        np.savetxt("sp_result.csv", sp_result, delimiter=',')
        np.savetxt("tran_result.csv", tran_result, delimiter=',')
        np.savetxt("fac_result.csv", fac_result, delimiter=',')
        np.savetxt("both_result.csv", both_result, delimiter=',')

        print "sp result: "+ str(sp_result)
        print "tran_result: "+str(tran_result)
        print "fac_result: "+str(fac_result)
        print "both_result: "+str(both_result)
