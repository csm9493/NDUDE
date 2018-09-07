from core.NDUDE_ft import FT_NDUDE

# window size = k^2-1
k_arr = [3,5,7,9,11,13,15,17]
delta_arr = [0.05, 0.1, 0.2, 0.25]

ep_ = 15
mini_bt_size = 256

# Available test dataset : 1) Set13_256, 2) Set13_512, 3) BSD20
test_data = 'BSD20'

is_2DDUDE_ = True   # if 2D-NeuralDUDE
is_randinit_ = True  # if use the random initialized model
is_blind_ = False    # if do not use the blind model

case_ = None

for delta_ in delta_arr:
    for k_ in k_arr:

        ft_NDUDE = FT_NDUDE(case = None, delta=delta_, k = k_, test_data = test_data, ep = ep_, mini_batch_size = mini_bt_size, is_randinit = is_randinit_, is_blind = is_blind_, is_2DDUDE = is_2DDUDE_)
        ft_NDUDE.test_model()

