from core.DUDE import DUDE

# window size = 2*k
k_arr = [2,3,4,5,6,7,8,9,10,11,12,24,40,60,84,112,144]

# Available test dataset : 1) Set13_256, 2) Set13_512, 3) BSD20
test_data = 'Set13_256' 

# if 1D DUDE
is_2DDUDE_ = False

case_ = None

for k_ in k_arr:

    dude = DUDE(case = case_, delta=delta_, k = k_, test_data = test_data, is_2DDUDE = is_2DDUDE_)
    dude.run_DUDE()

