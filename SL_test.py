from core.sup_te import Test_Supervised_2D

# window size = k^2-1
k_arr = [3,5,7,9,11,13,15,17]
delta_arr = [0.05, 0.10, 0.20, 0.25]
ep_ = 15

# Available test dataset : 1) Set13_256, 2) Set13_512, 3) BSD20
test_data = 'BSD20'

case_ = None

for delta_ in delta_arr:
    for k_ in k_arr:

        te_Supervised_2D = Test_Supervised_2D(case = None, delta=delta_, k = k_, test_data = test_data, ep = ep_)
        te_Supervised_2D.test_model()

