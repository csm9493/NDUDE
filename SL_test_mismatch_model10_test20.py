from core.sup_te import Test_Supervised_2D

delta_arr = [0.20]
model_delta_ = 0.1

k_arr = [3,5,7,9,11,13,15,17]
ep_ = 15

test_data = 'BSD20'
case_ = None

for delta_ in delta_arr:
    for k_ in k_arr:

        te_Supervised_2D = Test_Supervised_2D(case = None, delta=delta_, model_delta = model_delta_, k = k_, test_data = test_data, ep = ep_)
        te_Supervised_2D.test_model()


