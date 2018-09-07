from core.NDUDE_ft import FT_NDUDE

delta_arr = [0.5]
best_k_arr = [17]
sup_ep_arr = [14]

model_delta_ = 0.1

ep_ = 15
test_data = 'BSD20'
mini_bt_size = 256

is_blind_ = False
is_randinit_ = False
is_2DDUDE_ = True

case_ = None

for idx in range (len(delta_arr)):
    
    delta_ = delta_arr[idx]
    k_ = best_k_arr[idx]
    sup_ep_ = sup_ep_arr[idx]

    ft_NDUDE = FT_NDUDE(case = case_, delta=delta_, k = k_, test_data = test_data, model_delta=model_delta_, ep = ep_, sup_ep = sup_ep_, mini_batch_size = mini_bt_size, is_randinit = is_randinit_, is_blind = is_blind_, is_2DDUDE = is_2DDUDE_)
    ft_NDUDE.test_model()




