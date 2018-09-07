from core.NDUDE_2D_sup_te import Test_NDUDE_2D_sup

delta_arr = [0.20] # Test delta
model_delta_ = 0.1 # Model delta

k_arr = [3,5,7,9,11,13,15,17]
ep_ = 15
test_data = 'BSD20'

is_blind_ = True
case_ = None

for delta_ in delta_arr:
    for k_ in k_arr:

        te_NDUDE_2D_sup = Test_NDUDE_2D_sup(case = case_, delta=delta_, model_delta=model_delta_, k = k_, test_data = test_data, ep = ep_, is_blind = is_blind_)
        te_NDUDE_2D_sup.test_model()



