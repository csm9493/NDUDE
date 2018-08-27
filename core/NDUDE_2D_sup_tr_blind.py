import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import h5py
import numpy as np

import tensorflow as tf

import random
from sklearn.feature_extraction import image

# from .fcaide_sup_get_results import Get_results

class Train_NDUDE_2D:
    
    def __init__(self, case = None, mini_batch_size=256, k = 3, ep=50):
        self.num_mappings = 3
        self.mini_batch_size = mini_batch_size
        self.training_data_path = '../data/'
        self.epochs = ep
        self.training_data_file_name = 'NDUDE_sup_blind_source_data.hdf5'
        self.save_file_name = 'NDUDE_2D_blind_sup_training_data_k'+str(k)
        self.k = k
        self.nb_classes = 2
         
        print (self.save_file_name)
        if case != None :
            self.save_file_name += '_' + str(case)
            
        return
    
    def make_model(self):
        
        model = None
        train_model = None
        
        units = 128       
        num_of_layers = 12
        
        input_shape = ((self.k*self.k-1)*self.nb_classes,)
        input_layer = Input(shape=input_shape)
        layer_ = input_layer

        for layer_idx in range(num_of_layers):
            layer_ = Dense(units, kernel_initializer='he_uniform')(layer_)
            layer_ = Activation('relu')(layer_)

        layer_ = Dense(self.num_mappings, kernel_initializer='he_uniform')(layer_)
        layer_ = Activation('softmax')(layer_)

        output_layer = layer_

        model = Model(inputs=[input_layer], outputs=[output_layer])
        print (model.summary())

        adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

        model.compile(loss='poisson', optimizer=adam)

        return model
    
    def add_noise(self, img, delta):
    
        flatten_img = img.reshape((img.shape[0]*img.shape[1],)).copy()
        img_len = flatten_img.shape[0]

        for idx in range(img_len):
            rand_value = random.random()
            if flatten_img[idx] == 0:
                if rand_value < delta:
                    flatten_img[idx] = 1
            else:
                if rand_value < delta:
                    flatten_img[idx] = 0

        noisy_img = flatten_img.reshape(img.shape[0],img.shape[1])

        return noisy_img
    
    def generate_context(self, origin_img):
        
        x_size = origin_img.shape[0]
        y_size = origin_img.shape[1]
        
        context_data = np.zeros((x_size*y_size,self.k*self.k-1))

        img = origin_img.copy()
        padding_binary_bsd_data = np.pad(img,(self.k//2,self.k//2),'constant',constant_values=(0, 0))

        patches = image.extract_patches_2d(padding_binary_bsd_data, (self.k,self.k))
        flatten_patches = patches.reshape((patches.shape[0],patches.shape[1]*patches.shape[2]))

        context_data[:,0:(self.k*self.k-1)//2] =  flatten_patches[:,0:(self.k*self.k-1)//2]
        context_data[:,(self.k*self.k-1)//2:] =  flatten_patches[:,(self.k*self.k-1)//2+1:]
        
        return context_data
    
    
    def get_onehot_context(self, context, nb_classes):
        
        onehot_context = np_utils.to_categorical(context,nb_classes)
        flatten_onehot_context = onehot_context.reshape(onehot_context.shape[0],onehot_context.shape[1]*onehot_context.shape[2])
        
        return flatten_onehot_context
    
    def generate_blind_data(self, images):
        
        num_images = images.shape[0]
        x_size = images.shape[1]
        y_size = images.shape[2]
        
        k = self.k
        
        X = np.zeros((num_images*x_size*y_size,(k*k-1)*self.nb_classes))
        Y = np.zeros((num_images*x_size*y_size,3))

        for img_idx in range(num_images):
            
            img = images[img_idx]
            
            ## randomly select delta
            delta = random.uniform(0.05, 0.25)
            noisy_img = self.add_noise(img, delta)

            ## generate X_data
            context_data = self.generate_context(noisy_img)
            flatten_onehot_context = self.get_onehot_context(context_data, self.nb_classes)

            X[img_idx*(x_size*y_size):(img_idx+1)*(x_size*y_size),:] = flatten_onehot_context[:]

            ## generate Y_data
            label_data = np.zeros((x_size*y_size,3))

            flatten_img = img.flatten()
            flatten_noisy_img = noisy_img.flatten()

            for idx in range(x_size*y_size):
                true_pixel = flatten_img[idx]
                noisy_pixel = flatten_noisy_img[idx]

                if (true_pixel == 0) and (noisy_pixel == 0):
                    label_data[idx,0] = 1
                    label_data[idx,1] = 1
                elif (true_pixel == 0) and (noisy_pixel == 1):
                    label_data[idx,1] = 1
                elif (true_pixel == 1) and (noisy_pixel == 0):
                    label_data[idx,2] = 1
                else:
                    label_data[idx,0] = 1
                    label_data[idx,2] = 1

            Y[img_idx*(x_size*y_size):(img_idx+1)*(x_size*y_size),:] = label_data[:]
            
        return X,Y
    
    def generate_blind_sequences(self,batch_size, tr_images, num_context_):
        while True:
            # generate sequences for training

            X,Y = self.generate_blind_data(tr_images)

            num_context = num_context_
            context_idx = range(0, num_context)
            
            context_idx = np.random.permutation(context_idx)

            batches = int(num_context/batch_size)
            remainder_samples = num_context%batch_size
            
            if remainder_samples:
                batches = batches + 1
            # generate batches of samples
            for idx in range(0, batches):
                if idx == batches - 1:
                    batch_idxs = context_idx[idx*batch_size:]
                else:
                    batch_idxs = context_idx[idx*batch_size:idx*batch_size+batch_size]
                    
                batch_idxs = sorted(batch_idxs)
                
                batch_X = X[batch_idxs]
                batch_Y = Y[batch_idxs]
                
                yield batch_X, batch_Y
        

    def train_model(self):

        tr_data_location = './data/'+ self.training_data_file_name
        modelcheckpoint = ModelCheckpoint(filepath = './models/'+ self.save_file_name + '_ep{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        callbacks_list = [modelcheckpoint]
        
        model = self.make_model()

        with h5py.File(tr_data_location, "r") as tr_data:
            
            tr_images = tr_data["source_img"]
            num_context = tr_data["source_img"].shape[0] * tr_data["source_img"].shape[1] * tr_data["source_img"].shape[2]
            
            blind_sequence_generator = self.generate_blind_sequences(self.mini_batch_size, tr_images, num_context)
            model.fit_generator(generator=blind_sequence_generator,
                                              steps_per_epoch=(num_context/(self.mini_batch_size)+1),
                                              epochs=self.epochs,callbacks=callbacks_list,verbose=1)

