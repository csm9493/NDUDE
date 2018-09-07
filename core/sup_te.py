import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils

from sklearn.feature_extraction import image
import h5py
import numpy as np
import random

class Test_Supervised_2D:
    
    def __init__(self, case = None, delta=0.05, model_delta=None, k = 3, test_data = 'BSD20', ep = 15):
        self.model_output = 3
        self.delta = delta
        
        if test_data == 'BSD20':
            self.save_file_name = 'Supervised_2D_test_result_BSD20_k'+str(k)+'_delta'+str(int(self.delta*100))
        elif test_data == 'Set13_256':
            self.save_file_name = 'Supervised_2D_test_result_Set13_256_k'+str(k)+'_delta'+str(int(self.delta*100))
        else:
            self.save_file_name = 'Supervised_2D_test_result_Set13_512_k'+str(k)+'_delta'+str(int(self.delta*100))
            
        self.k = k
        self.test_data = test_data
        self.ep = ep
        self.nb_classes = 2
        
        if test_data == 'BSD20':
            self.num_te_data = 20
        elif test_data =='Set13_512':
            self.num_te_data = 8
        else:
            self.num_te_data = 5
         

        if case != None :
            self.save_file_name += '_' + str(case)

        self.model_delta = model_delta
        if self.model_delta != None:
            self.save_file_name += '_model_delta'  + str(int(self.model_delta*100))
        
        print (self.save_file_name)
        
        return
    
    def get_data(self):
        
        if self.test_data == 'BSD20':
            data_file_name = 'NDUDE_test_data_BSD20.hdf5'
        elif self.test_data == 'Set13_512':
            data_file_name = 'NDUDE_test_data_Set13_512.hdf5'
        else:
            data_file_name = 'NDUDE_test_data_Set13_256.hdf5'
            
            
        f = h5py.File('./data/'+data_file_name, 'r')
        true_img = np.array(f["true_img"])
        
        noisy_img_name = 'delta' + str(int(self.delta*100))
        noisy_img = np.array(f[noisy_img_name])
        
        self.x_axis = true_img.shape[1]
        self.y_axis = true_img.shape[2]
        
        self.mini_batch_size = 1024
        
        return true_img, noisy_img
    
    def make_model(self, ep):
        
            
        if self.model_delta == None:
            model_delta = self.delta
        else:
            # mismatched case
            model_delta = self.model_delta

        model_file_name = 'Supervised_2D_training_data_k'+str(self.k)+'_delta'+str(int(model_delta*100))+'_ep'+str(ep).zfill(2)+'.hdf5'
        print (model_file_name)
            
        model = load_model('./models/'+model_file_name)
            
        return model

    def generate_context(self, origin_img):
        
        x_size = origin_img.shape[0]
        y_size = origin_img.shape[1]
        
#         context_data = np.zeros((x_size*y_size,self.k*self.k))

        img = origin_img.copy()
        padding_binary_bsd_data = np.pad(img,(self.k//2,self.k//2),'constant',constant_values=(0, 0))

        patches = image.extract_patches_2d(padding_binary_bsd_data, (self.k,self.k))
        flatten_patches = patches.reshape((patches.shape[0],patches.shape[1]*patches.shape[2]))

#         context_data[:,0:(self.k*self.k-1)//2] =  flatten_patches[:,0:(self.k*self.k-1)//2]
#         context_data[:,(self.k*self.k-1)//2:] =  flatten_patches[:,(self.k*self.k-1)//2+1:]
        
        return flatten_patches
    
    def get_prediction_label(self, prediction):
        
        denoising_result = np.zeros((prediction.shape[0]))
        for idx in range(prediction.shape[0]):
            if prediction[idx] > 0.5:
                denoising_result[idx] = 1
            else:
                denoising_result[idx] = 0
        
        return denoising_result
    
    def get_error_rate(self, true, true_hat):
        
        error = np.zeros(len(true))
        for i in range(len(true)):
            error[i]=int(true[i]!=true_hat[i])
            
        return sum(error)/len(true)  
    
    def get_onehot_context(self, context, nb_classes):
        
        onehot_context = np_utils.to_categorical(context,nb_classes)
        flatten_onehot_context = onehot_context.reshape(onehot_context.shape[0],onehot_context.shape[1]*onehot_context.shape[2])
        
        return flatten_onehot_context
    
    def save_result(self, results, images):
        
        f = h5py.File('./result_data/' + self.save_file_name + ".hdf5", "w")
        f.create_dataset('error_rate', data=results)
        f.create_dataset('denoised_images', data=images)
        f.close()
        
        return
    
    def test_model(self):
        
        true_img, noisy_img = self.get_data()
        result_for_save = np.zeros((self.num_te_data,self.ep))
        image_for_save = np.zeros((self.num_te_data, self.x_axis, self.y_axis))
        temp_best_error_rate = np.zeros((self.num_te_data,))
        
        for ep in range(self.ep):
            test_model = self.make_model(ep+1)

            for img_idx in range(self.num_te_data):
                
                noisy_context = self.generate_context(noisy_img[img_idx])
                flatten_onehot_context = self.get_onehot_context(noisy_context, self.nb_classes)
                
                prediction_result = test_model.predict(flatten_onehot_context, batch_size = self.mini_batch_size, verbose=2)
                flatten_true = true_img[img_idx].flatten()
                flatten_noisy = noisy_img[img_idx].flatten()
                
                denoising_result = self.get_prediction_label(prediction_result)
                error_rate = self.get_error_rate(flatten_true, denoising_result)
                
                result_for_save[img_idx,ep] = error_rate
                
                if ep == 0:
                    image_for_save[img_idx,:,:] = denoising_result.reshape(self.x_axis, self.y_axis)
                    temp_best_error_rate[img_idx] = error_rate
                    
                if temp_best_error_rate[img_idx] > error_rate:
                    temp_best_error_rate[img_idx] = error_rate
                    image_for_save[img_idx,:,:] = denoising_result.reshape(self.x_axis, self.y_axis)
                    
            print ('ep : ' + str(ep+1) + ' error_rate : ' + str(np.mean(result_for_save[:,ep])))
            
            del test_model
                
        self.save_result(result_for_save, image_for_save)
