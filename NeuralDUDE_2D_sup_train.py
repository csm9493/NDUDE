from core.NDUDE_2D_sup_tr import Train_NDUDE_2D

# window size = k^2-1
k_arr = [3,5,7,9,11,13,15,17]
delta_arr = [0.05, 0.10, 0.20, 0.25]

mini_batch_size_ = 512
ep_ = 15

case_ = None

for delta_ in delta_arr:
    for k_ in k_arr:

        tr_NDUDE_2D = Train_NDUDE_2D(case = case_, delta=delta_, mini_batch_size=mini_batch_size_, k = k_, ep=ep_)
        tr_NDUDE_2D.train_model()