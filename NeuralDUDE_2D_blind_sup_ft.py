from core.NDUDE_ft import FT_NDUDE

# select a delta and a model information for the fine tuning
delta_arr = [0.05, 0.1, 0.2, 0.25]
best_k_arr = [13, 15, 13, 15]
sup_ep_arr = [11, 15, 10, 12]

ep_ = 15
mini_bt_size = 256

# Available test dataset : 1) Set13_256, 2) Set13_512, 3) BSD20
test_data = 'BSD20'

is_blind_ = False     # if 1D-NeuralDUDE
is_2DDUDE_ = True     # if do not use the blind model
is_randinit_ = True  # if do not use the random initialized model (Sup+FT)


case_ = None

for idx in range (len(delta_arr)):
    
    delta_ = delta_arr[idx]
    k_ = best_k_arr[idx]
    sup_ep_ = sup_ep_arr[idx]

    ft_NDUDE = FT_NDUDE(case = case_, delta=delta_, k = k_, test_data = test_data, ep = ep_, sup_ep = sup_ep_, mini_batch_size = mini_bt_size, is_randinit = is_randinit_, is_blind = is_blind_, is_2DDUDE = is_2DDUDE_)
    ft_NDUDE.test_model()




