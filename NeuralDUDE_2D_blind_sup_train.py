from core.NDUDE_2D_sup_tr_blind import Train_NDUDE_2D

# window size = K^2-1
k_arr = [3,5,7,9,11,13,15,17]

mini_batch_size_ = 512
ep_ = 15

case_ = None

for k_ in k_arr:

    tr_NDUDE_2D = Train_NDUDE_2D(case = case_, mini_batch_size=mini_batch_size_, k = k_, ep=ep_)
    tr_NDUDE_2D.train_model()