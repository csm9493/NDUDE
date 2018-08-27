import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import h5py
import numpy as np

import tensorflow as tf

import random

# from .fcaide_sup_get_results import Get_results

class Train_Supervised_2D:
    
    def __init__(self, case = None, delta=0.05, mini_batch_size=256, k = 3, ep=10):
        self.num_output = 1
        self.delta = delta
        self.mini_batch_size = mini_batch_size
        self.training_data_path = '../data/'
        self.epochs = ep
        self.training_data_file_name = 'Supervised_2D_training_data_k'+str(k)+'_delta'+str(int(self.delta*100))+'.hdf5'
        self.save_file_name = 'Supervised_2D_training_data_k'+str(k)+'_delta'+str(int(self.delta*100))
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
        
        input_shape = ((self.k*self.k)*self.nb_classes,)
        input_layer = Input(shape=input_shape)
        layer_ = input_layer

        for layer_idx in range(num_of_layers):
            layer_ = Dense(units, kernel_initializer='he_uniform')(layer_)
            layer_ = Activation('relu')(layer_)

        layer_ = Dense(self.num_output, kernel_initializer='he_uniform')(layer_)
        layer_ = Activation('sigmoid')(layer_)

        output_layer = layer_

        model = Model(inputs=[input_layer], outputs=[output_layer])
        print (model.summary())

        adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

        model.compile(loss='binary_crossentropy', optimizer=adam)

        return model

    def train_model(self):

        tr_data_location = './data/'+ self.training_data_file_name
        modelcheckpoint = ModelCheckpoint(filepath = './models/'+ self.save_file_name + '_ep{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        callbacks_list = [modelcheckpoint]
        
        model = self.make_model()

        with h5py.File(tr_data_location, "r") as tr_data:
            
            num_data = tr_data["X_data"].shape[0]
            
            X = np.array(tr_data["X_data"][:num_data],dtype = np.float)
            Y = np.array(tr_data["Y_data"][:num_data],dtype = np.float)

            model.fit(x=X, y=Y, batch_size=self.mini_batch_size, epochs=self.epochs, verbose=1, callbacks=callbacks_list)
            
            del X
            del Y

