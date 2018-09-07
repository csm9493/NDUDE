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

from .NDUDE_ft_result import Save_result

class FT_NDUDE:
    
    def __init__(self, case = None, delta=0.05, model_delta = None, k = 3, test_data = 'BSD20', ep = 10, lr=0.001, sup_ep = None, mini_batch_size = 128, is_randinit = True, is_blind = False, is_2DDUDE = True):
        self.model_output = 3
        self.delta = delta
        self.is_randinit = is_randinit
        self.mini_batch_size = mini_batch_size
        
        if is_2DDUDE == True:
            self.save_file_name = 'NDUDE_2D'
        else:
            self.save_file_name = 'NDUDE_1D'
        
        if is_blind == True:
            self.save_file_name += '_blind'
        if is_randinit == True:
            self.save_file_name += '_randinit_ft_test_result_'
        else:
            self.save_file_name += '_sup_ft_test_result_' 
        if test_data == 'BSD20':
            self.save_file_name += 'BSD20_k'+str(k)+'_delta'+str(int(self.delta*100))
        elif test_data == 'Set13_256':
            self.save_file_name += 'Set13_256_k'+str(k)+'_delta'+str(int(self.delta*100))
        else:
            self.save_file_name += 'Set13_512_k'+str(k)+'_delta'+str(int(self.delta*100))    
     
        self.k = k
        self.test_data = test_data
        self.ep = ep
        self.is_blind = is_blind
        self.is_2DDUDE = is_2DDUDE
        self.is_randinit = is_randinit
        self.sup_ep = sup_ep
        self.learning_rate = lr
        
        if test_data == 'BSD20':
            self.num_te_data = 20
        elif test_data =='Set13_512':
            self.num_te_data = 8
        else:
            self.num_te_data = 5
            
        self.model_delta = model_delta
        if self.model_delta != None:
            self.save_file_name += '_model_delta'  + str(int(self.model_delta*100))
         
        
        if case != None :
            self.save_file_name += '_' + str(case)
            
        print (self.save_file_name)
            
        self.nb_classes = 2
            
        return
    
    def get_data(self):
        
        if self.test_data == 'BSD20':
            data_file_name = 'NDUDE_test_data_BSD20.hdf5'
        elif self.test_data == 'Set13_512':
            data_file_name = 'NDUDE_test_data_Set13_512.hdf5'
        else:
            data_file_name = 'NDUDE_test_data_Set13_256.hdf5'
            
            
        print (data_file_name)
        f = h5py.File('./data/'+data_file_name, 'r')
        true_img = np.array(f["true_img"])
        
        noisy_img_name = 'delta' + str(int(self.delta*100))
        noisy_img = np.array(f[noisy_img_name])
        
        return true_img, noisy_img
    
    def make_model(self):
        
        if self.is_2DDUDE == True:
            
            ## 2D-NDUDE
            
            if self.is_randinit == False:
                
                if self.is_blind == False:
                    
                    if self.model_delta == None:
                        model_delta = self.delta
                    else:
                        # mismatched case
                        model_delta = self.model_delta
                    
                    model_file_name = 'NDUDE_2D_sup_training_data_k'+str(self.k)+'_delta'+str(int(model_delta*100))+'_ep'+str(self.sup_ep).zfill(2)+'.hdf5'
                    print (model_file_name)
                else:
                    model_file_name = 'NDUDE_2D_blind_sup_training_data_k'+str(self.k)+'_ep'+str(self.sup_ep).zfill(2)+'.hdf5'
                    print (model_file_name)

                model = load_model('./models/'+model_file_name)

                K.set_value(model.optimizer.lr, self.learning_rate)
                print('learning rate : ' + str(K.get_value(model.optimizer.lr)))

            else:

                units = 128       
                num_of_layers = 12

                input_shape = ((self.k*self.k-1)*self.nb_classes,)
                input_layer = Input(shape=input_shape)
                layer_ = input_layer

                for layer_idx in range(num_of_layers):
                    layer_ = Dense(units, kernel_initializer='he_uniform')(layer_)
                    layer_ = Activation('relu')(layer_)

                layer_ = Dense(self.model_output, kernel_initializer='he_uniform')(layer_)
                layer_ = Activation('softmax')(layer_)

                output_layer = layer_

                model = Model(inputs=[input_layer], outputs=[output_layer])

                adam=Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

                model.compile(loss='poisson', optimizer=adam)
                
        else:
            
            ## 1D-NDUDE
            
            units = 40       
            num_of_layers = 4

            input_shape = (self.k*2*self.nb_classes,)
            input_layer = Input(shape=input_shape)
            layer_ = input_layer

            for layer_idx in range(num_of_layers):
                layer_ = Dense(units, kernel_initializer='he_uniform')(layer_)
                layer_ = Activation('relu')(layer_)

            layer_ = Dense(self.model_output, kernel_initializer='he_uniform')(layer_)
            layer_ = Activation('softmax')(layer_)

            output_layer = layer_

            model = Model(inputs=[input_layer], outputs=[output_layer])

            adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

            model.compile(loss='poisson', optimizer=adam)
  
        return model
               
    def get_L_new(self, delta):
        
        pi_matrix = np.array(([1-delta, delta],[delta, 1-delta]))
        PI_INV = np.linalg.inv(pi_matrix)
        RHO = np.zeros((2, 3))
        LAMBDA = np.array([[0, 1], [1, 0]])
        MAP = np.array([[0, 0, 1], [1, 0, 1]])
        for xx in range(2):
            for ss in range(3):
                for zz in range(2):
                    RHO[xx][ss] += pi_matrix[xx][zz] * LAMBDA[xx][MAP[zz][ss]]

        L = np.matmul(PI_INV, RHO)
        L_new = -L + np.amax(L)
        return L, L_new

    def generate_context_pseudolabel(self, noisy_img):
        
        if self.is_2DDUDE == True:
            x_size = noisy_img.shape[0]
            y_size = noisy_img.shape[1]

            ## generate context data
            context_data = np.zeros((x_size*y_size,self.k*self.k-1))

            img = noisy_img.copy()
            padding_binary_data = np.pad(img,(self.k//2,self.k//2),'constant',constant_values=(0, 0))

            patches = image.extract_patches_2d(padding_binary_data, (self.k,self.k))
            flatten_patches = patches.reshape((patches.shape[0],patches.shape[1]*patches.shape[2]))

            context_data[:,0:(self.k*self.k-1)//2] =  flatten_patches[:,0:(self.k*self.k-1)//2]
            context_data[:,(self.k*self.k-1)//2:] =  flatten_patches[:,(self.k*self.k-1)//2+1:]
            
        else:
            
            flatten_noisy_img = noisy_img.flatten()
            padded_1d_noisy_arr = np.pad(flatten_noisy_img,(self.k,self.k),'constant',constant_values=(0, 0))
            print (padded_1d_noisy_arr.shape)
            
            len_1d_noisy_arr = flatten_noisy_img.shape[0]
            context_data = np.zeros((len_1d_noisy_arr, self.k*2))
            
            ## generate context data
            for idx in range(len_1d_noisy_arr):
                temp_context_data = np.zeros((self.k*2,))
                temp_context_data[:self.k] = padded_1d_noisy_arr[idx:idx+self.k]
                temp_context_data[self.k:] = padded_1d_noisy_arr[idx+self.k+1:idx+self.k*2+1]
                
                context_data[idx,:] = temp_context_data[:]

        #generate pesudo label
        
#         L=np.array([[self.delta, -self.delta/(1-2*self.delta), (1-self.delta)/(1-2*self.delta)],[self.delta, (1-self.delta)/(1-2*self.delta), -self.delta/(1-2*self.delta)]])
#         L_new=-L+(1-self.delta)/(1-2*self.delta)
               
        L, L_new = self.get_L_new(self.delta)
        
        flatten_noisy_img = noisy_img.flatten().copy()
        categorical_noisy_img = np_utils.to_categorical(flatten_noisy_img, self.nb_classes)
        
        pseudo_label = np.dot(categorical_noisy_img, L_new)
        
        return context_data, pseudo_label, L, categorical_noisy_img
    
    def get_onehot_context(self, context, nb_classes):
        
        onehot_context = np_utils.to_categorical(context,nb_classes)
        flatten_onehot_context = onehot_context.reshape(onehot_context.shape[0],onehot_context.shape[1]*onehot_context.shape[2])
        
        return flatten_onehot_context
 
    def test_model(self):
        
        true_img, noisy_img = self.get_data()
        
        save_results = Save_result(self.save_file_name, true_img, noisy_img)

        for img_idx in range(self.num_te_data):
            
            model = self.make_model()

            context_data, pseudo_label, L, categorical_noisy_img = self.generate_context_pseudolabel(noisy_img[img_idx])
            flatten_onehot_context = self.get_onehot_context(context_data, self.nb_classes)
            save_results.set_data(flatten_onehot_context, L, categorical_noisy_img)

            model.fit(flatten_onehot_context, pseudo_label, verbose=2, batch_size = self.mini_batch_size, epochs = self.ep, callbacks=[save_results])
            
            del model
            
        save_results.save_result()


#         self.save_result(result_for_save, image_for_save)
