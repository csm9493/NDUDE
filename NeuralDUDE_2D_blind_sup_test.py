from core.NDUDE_2D_sup_te import Test_NDUDE_2D_sup

# window size = K^2-1
k_arr = [3,5,7,9,11,13,15,17]
delta_arr = [0.05, 0.1, 0.2, 0.25]

ep_ = 15

# Available test dataset : 1) Set13_256, 2) Set13_512, 3) BSD20
test_data = 'BSD20'

# if not a blind case
is_blind_ = True

case_ = None

for delta_ in delta_arr:
    for k_ in k_arr:

        te_NDUDE_2D_sup = Test_NDUDE_2D_sup(case = case_, delta=delta_, k = k_, test_data = test_data, ep = ep_, is_blind = is_blind_)
        te_NDUDE_2D_sup.test_model()


