from keras.utils import np_utils

from sklearn.feature_extraction import image
import h5py
import numpy as np
import random
import math

mask_arr_3by3 = [np.array(([0,2,0],
                     [0,0,0],
                     [0,0,0])),
           np.array(([0,2,0],
                     [0,0,0],
                     [0,2,0])),
           np.array(([0,2,0],
                     [2,0,0],
                     [0,2,0])),
           np.array(([0,2,0],
                     [2,0,2],
                     [0,2,0])),
           np.array(([2,2,0],
                     [2,0,2],
                     [0,2,0])),
           np.array(([2,2,0],
                     [2,0,2],
                     [0,2,2])),
           np.array(([2,2,0],
                     [2,0,2],
                     [2,2,2])),
           ]

mask_arr_5by5 = [np.array(([0,0,2,0,0],
                           [0,2,2,2,0],
                           [0,2,0,2,0],
                           [0,2,2,2,0],
                           [0,0,0,0,0])),
                 np.array(([0,0,2,0,0],
                           [0,2,2,2,0],
                           [0,2,0,2,0],
                           [0,2,2,2,0],
                           [0,0,2,0,0])),
                 np.array(([0,0,2,0,0],
                           [0,2,2,2,0],
                           [2,2,0,2,0],
                           [0,2,2,2,0],
                           [0,0,2,0,0])),
                 np.array(([0,0,2,0,0],
                           [0,2,2,2,0],
                           [2,2,0,2,2],
                           [0,2,2,2,0],
                           [0,0,2,0,0])),
                 np.array(([0,2,2,0,0],
                           [0,2,2,2,0],
                           [2,2,0,2,2],
                           [0,2,2,2,0],
                           [0,0,2,0,0])),
                 np.array(([0,2,2,0,0],
                           [0,2,2,2,0],
                           [2,2,0,2,2],
                           [0,2,2,2,0],
                           [0,0,2,2,0])),
                 np.array(([0,2,2,0,0],
                           [0,2,2,2,0],
                           [2,2,0,2,2],
                           [0,2,2,2,0],
                           [0,2,2,0,0])),
                 np.array(([0,2,2,2,0],
                           [0,2,2,2,0],
                           [2,2,0,2,2],
                           [0,2,2,2,0],
                           [0,2,2,2,0])),
                 np.array(([0,2,2,2,0],
                           [2,2,2,2,0],
                           [2,2,0,2,0],
                           [0,2,2,2,2],
                           [0,2,2,2,0])),
                 np.array(([0,2,2,2,0],
                           [2,2,2,2,0],
                           [2,2,0,2,2],
                           [0,2,2,2,2],
                           [0,2,2,2,0])),
                 np.array(([0,2,2,2,0],
                           [2,2,2,2,0],
                           [2,2,0,2,2],
                           [2,2,2,2,2],
                           [0,2,2,2,0])),
                 np.array(([0,2,2,2,0],
                           [2,2,2,2,2],
                           [2,2,0,2,2],
                           [2,2,2,2,2],
                           [0,2,2,2,0])),
                 np.array(([2,2,2,2,0],
                           [2,2,2,2,2],
                           [2,2,0,2,2],
                           [2,2,2,2,2],
                           [0,2,2,2,0])),
                 np.array(([2,2,2,2,0],
                           [2,2,2,2,2],
                           [2,2,0,2,2],
                           [2,2,2,2,2],
                           [0,2,2,2,2])),
                 np.array(([2,2,2,2,0],
                           [2,2,2,2,2],
                           [2,2,0,2,2],
                           [2,2,2,2,2],
                           [2,2,2,2,2])),
                ]
           

class DUDE:
    
    def __init__(self, case = None, delta=0.05, k = 3, test_data = 'BSD20', is_2DDUDE = True):
        self.model_output = 3
        self.delta = delta
        
        if is_2DDUDE == True:
            self.save_file_name = 'DUDE_2D_'
        else:
            self.save_file_name = 'DUDE_1D_'
        
        if test_data == 'BSD20':
            self.save_file_name += 'BSD20_k'+str(k)+'_delta'+str(int(self.delta*100))
        elif test_data == 'Set13_256':
            self.save_file_name += 'Set13_256_k'+str(k)+'_delta'+str(int(self.delta*100))
        else:
            self.save_file_name += 'Set13_512_k'+str(k)+'_delta'+str(int(self.delta*100))    
     
        self.k = k
        self.test_data = test_data
        self.is_2DDUDE = is_2DDUDE
        
        if test_data == 'BSD20':
            self.num_te_data = 20
        elif test_data =='Set13_512':
            self.num_te_data = 8
        else:
            self.num_te_data = 5
         
        print (self.save_file_name)
        if case != None :
            self.save_file_name += '_' + str(case)
            
        self.erate_result_for_save = []
        self.estloss_result_for_save = []
        self.image_for_save = []
        
        self.binary_outputs = 2
        self.num_mappings = 3
            
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
        
        self.x_axis = true_img.shape[1]
        self.y_axis = true_img.shape[2]
        
        return true_img, noisy_img
     
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
    
    
    def get_k_for_2d_context(self, k):
        return int(math.sqrt(k+1))

    
    def dude(self, flatten_noisy_img, k, delta):
       # print "Running DUDE algorithm"
        len_flatten_noisy_img = len(flatten_noisy_img)
        s_hat = np.zeros(len_flatten_noisy_img, dtype=np.int)

        th_0=2*delta*(1-delta)
        th_1=delta**2+(1-delta)**2

        frequency_table={}

        if self.is_2DDUDE == False:
            
            ## 1D-DUDE
            
            k_for_1d_context = k

            for i in range(k_for_1d_context,len_flatten_noisy_img-k_for_1d_context):

                context=flatten_noisy_img[i-k_for_1d_context:i].tolist()+flatten_noisy_img[i+1:i+k_for_1d_context+1].tolist()

                context = np.array((context),dtype = np.int)
                context_str = str(context)

                if context_str not in frequency_table:
                    frequency_table[context_str]=np.zeros(2,dtype=np.int)
                    frequency_table[context_str][int(flatten_noisy_img[i])]=1
                else:
                    frequency_table[context_str][int(flatten_noisy_img[i])]+=1

            for i in range(k_for_1d_context,len_flatten_noisy_img-k_for_1d_context):

                context=flatten_noisy_img[i-k_for_1d_context:i].tolist()+flatten_noisy_img[i+1:i+k_for_1d_context+1].tolist()

                context = np.array((context),dtype = np.int)
                context_str = str(context)
                
                ratio = float(frequency_table[context_str][1]) / float(np.sum(frequency_table[context_str]))

                if ratio < th_0:
                    s_hat[i]=1
                elif ratio >= th_1:
                    s_hat[i]=2
                else:
                    s_hat[i]=0
        
        else:
            
            # 2D-DUDE
            
            # get 2D-DUDE patch
            
            if k >= 2 and k < 8:
                context_data = np.zeros((len_flatten_noisy_img,k))
                k_for_2d_patch = 3
            elif k >= 9 and k < 24:
                context_data = np.zeros((len_flatten_noisy_img,k))
                k_for_2d_patch = 5
            else:
                k_for_2d_patch = self.get_k_for_2d_context(k)
                context_data = np.zeros((len_flatten_noisy_img,k))
                

            img = flatten_noisy_img.reshape(self.x_axis, self.y_axis)
            padding_binary_data = np.pad(img,(k_for_2d_patch//2,k_for_2d_patch//2),'constant',constant_values=(0, 0))

            patches = image.extract_patches_2d(padding_binary_data, (k_for_2d_patch,k_for_2d_patch))
            flatten_patches = patches.reshape((patches.shape[0],patches.shape[1]*patches.shape[2]))

            if k >= 2 and k < 9:
                for patch_idx in range(len_flatten_noisy_img):
                    
                    masked_patch = patches[patch_idx] + mask_arr_3by3[k-2]
                    flatten_masked_patch = masked_patch.flatten()
                    
                    context_idx = 0
                    
                    for idx in range(flatten_masked_patch.shape[0]):
                        if flatten_masked_patch[idx] >=2 :
                            context_data[patch_idx, context_idx] = flatten_masked_patch[idx]
                            context_idx += 1
                            
            elif k >= 9 and k < 24:
                for patch_idx in range(len_flatten_noisy_img):
                    
                    masked_patch = patches[patch_idx] + mask_arr_5by5[k-9]
                    flatten_masked_patch = masked_patch.flatten()
                    
                    context_idx = 0
                    
                    for idx in range(flatten_masked_patch.shape[0]):
                        if flatten_masked_patch[idx] >=2 :
                            context_data[patch_idx, context_idx] = flatten_masked_patch[idx]
                            context_idx += 1
                            
            else:
                context_data[:,0:(k_for_2d_patch*k_for_2d_patch-1)//2] =  flatten_patches[:,0:(k_for_2d_patch*k_for_2d_patch-1)//2]
                context_data[:,(k_for_2d_patch*k_for_2d_patch-1)//2:] =  flatten_patches[:,(k_for_2d_patch*k_for_2d_patch-1)//2+1:]

            # get s_hat
                
            for i in range(len_flatten_noisy_img):

                context = context_data[i]

                context = np.array((context),dtype = np.int)
                context_str = str(context)

                if context_str not in frequency_table:
                    frequency_table[context_str]=np.zeros(2,dtype=np.int)
                    frequency_table[context_str][int(flatten_noisy_img[i])]=1
                else:
                    frequency_table[context_str][int(flatten_noisy_img[i])]+=1

            for i in range(len_flatten_noisy_img):

                context = context_data[i]

                context = np.array((context),dtype = np.int)
                context_str = str(context)

                ratio = float(frequency_table[context_str][1]) / float(np.sum(frequency_table[context_str]))

                if ratio < th_0:
                    s_hat[i]=1
                elif ratio >= th_1:
                    s_hat[i]=2
                else:
                    s_hat[i]=0

        return s_hat, frequency_table
    
    def get_denoising_result(self, s_hat, flatten_noisy_img):

        denoising_result = np.zeros((s_hat.shape[0],))

        for idx in range(s_hat.shape[0]):

            if s_hat[idx] == 0:
                denoising_result[idx] = flatten_noisy_img[idx]
            elif s_hat[idx] == 1:
                denoising_result[idx] = 0
            else:
                denoising_result[idx] = 1

        return denoising_result
    
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
 
    def run_DUDE(self):
        
        true_img, noisy_img = self.get_data()

        for img_idx in range(self.num_te_data):
            
            flatten_true_img = true_img[img_idx].flatten()
            flatten_noisy_img = noisy_img[img_idx].flatten()

            L, L_new = self.get_L_new(self.delta)
            
            #get error rate
            prediction_result, m = self.dude(flatten_noisy_img, self.k, self.delta)
            denoising_result = self.get_denoising_result(prediction_result,flatten_noisy_img)
            error_rate = self.get_error_rate(flatten_true_img, denoising_result)
            
            self.erate_result_for_save.append(error_rate)
            self.image_for_save.append(denoising_result.reshape(self.x_axis,self.y_axis))
            
            #get estimated loss
            categorical_noisy_img = np_utils.to_categorical(flatten_noisy_img,self.binary_outputs)
            categorical_prediction_result = np_utils.to_categorical(prediction_result,self.num_mappings)
            
            emp_dist=np.dot(categorical_noisy_img,L)
            est_loss=np.mean(np.sum(emp_dist*categorical_prediction_result,axis=1))
            
            self.estloss_result_for_save.append(error_rate)
            
            print ('img_idx : ' + str(img_idx+1) + ' est_loss : ' + str(est_loss)+ ' error_rate : ' + str(error_rate))
                                  
        self.save_result()

            