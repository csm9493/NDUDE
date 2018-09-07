from core.DUDE import DUDE

# window size = k
k_arr = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,48,80,120,168,224,288]

# Available test dataset : 1) Set13_256, 2) Set13_512, 3) BSD20
test_data = 'Set13_256'

# if 2D-DUDE
is_2DDUDE_ = True

case_ = None

for k_ in k_arr:

    dude = DUDE(case = None, delta=delta_, k = k_, test_data = test_data, is_2DDUDE = is_2DDUDE_)
    dude.run_DUDE()

