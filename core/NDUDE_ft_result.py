import keras
import numpy as np
import h5py

import keras.backend as K
from keras.utils import np_utils

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Save_result(keras.callbacks.Callback):
    
    def __init__(self, save_file_name, true_img, noisy_img):
       
        self.save_file_name = save_file_name
        self.true_img = true_img
        self.noisy_img = noisy_img
        
        self.x_axis = true_img.shape[1]
        self.y_axis = true_img.shape[2]
        
        self.img_idx = 0
        
        self.erate_result_for_save = []
        self.estloss_result_for_save = []
        self.image_for_save = []
        
        self.num_mappings = 3
       
    
    def set_data(self, noisy_context, L, categorical_noisy_img):
        self.noisy_context = noisy_context
        self.L = L
        self.categorical_noisy_img = categorical_noisy_img
        
        return
    
    def get_prediction_label(self, flatten_noisy, prediction):

        denoising_result = np.zeros((prediction.shape[0],))
        argmax_result = np.argmax(prediction, axis=1)

        for idx in range(prediction.shape[0]):

            if argmax_result[idx] == 0:
                denoising_result[idx] = flatten_noisy[idx]
            elif argmax_result[idx] == 1:
                denoising_result[idx] = 0
            else:
                denoising_result[idx] = 1

        return denoising_result, argmax_result
    
    def get_error_rate(self, true, true_hat):
        
        error = np.zeros(len(true))
        for i in range(len(true)):
            error[i]=int(true[i]!=true_hat[i])
            
        return sum(error)/len(true)  
    
    def save_result(self):
        
        f = h5py.File('./result_data/' + self.save_file_name + ".hdf5", "w")
        f.create_dataset('error_rate', data=self.erate_result_for_save)
        f.create_dataset('est_loss', data=self.estloss_result_for_save)
        f.create_dataset('denoised_images', data=self.image_for_save)
        f.close()
        
        return
        
    def on_train_begin(self, logs={}):
        self.best_erate = 1
        self.erate_arr = []
        self.estloss_arr = []
        self.best_img = []
        
        return
        
    def on_train_end(self, logs={}):
        self.img_idx += 1
        
        self.erate_result_for_save.append(self.erate_arr)
        self.estloss_result_for_save.append(self.estloss_arr)
        self.image_for_save.append(self.best_img)
        
        return
        
    def on_epoch_end(self, epoch, logs={}):
        
        ## get error rate
        prediction_result = self.model.predict(self.noisy_context, batch_size = 2048, verbose=0)
        flatten_true = self.true_img[self.img_idx].flatten()
        flatten_noisy = self.noisy_img[self.img_idx].flatten()

        denoising_result, argmax_result = self.get_prediction_label(flatten_noisy, prediction_result)
        error_rate = self.get_error_rate(flatten_true, denoising_result)

        self.erate_arr.append(error_rate)
        
        ## get estimated loss
        emp_dist = np.dot(self.categorical_noisy_img,self.L)
        categorical_argmax_result = np_utils.to_categorical(argmax_result,self.num_mappings)
        est_loss = np.mean(np.sum(emp_dist*categorical_argmax_result,axis=1))
        self.estloss_arr.append(est_loss)

        ## save image
        if self.best_erate > error_rate:
            self.best_img = denoising_result.reshape(self.x_axis, self.y_axis)
            self.best_erate = error_rate

        print ('img_idx : ' + str(self.img_idx+1)+ ' ep : ' + str(epoch+1) + ' est_loss : ' + str(est_loss)+ ' error_rate : ' + str(error_rate))
