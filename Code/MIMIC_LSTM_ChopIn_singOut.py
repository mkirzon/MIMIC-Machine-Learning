
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import pandas as pd
import sys, os, time, csv, random
from pandas import DataFrame as df
from numpy import linalg as LA
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adamax


# ### Load data from pickle file and prepare training set and testing set

# In[3]:

DATA_PATH = '/Users/hi08060204/Desktop/Peter_ML/MIMIC_Project/Data/Processed/mimic3_11_resampled-imputed_5000.npz'

nb_samples = None
test_split = 0.2

# Loading data from .npz file into dictionary form
data = np.load(DATA_PATH)
X_raw = data['X'][:nb_samples]
y_raw = data['ylos'][:nb_samples]

# Spliting data for the use of training and testing
X_raw_test = X_raw[:len(X_raw)*test_split]
y_raw_test = y_raw[:len(y_raw)*test_split]
X_raw_train = X_raw[len(X_raw)*test_split:]
y_raw_train = y_raw[len(y_raw)*test_split:]


# ### Prepare sequential output, specifying remaining hours

# In[4]:

X_train = X_raw_train
y_train_list = [ [ [left_hours] for left_hours in reversed(range(ep.shape[0]))] for ep in X_train ] 
y_train = np.array(y_train_list)

X_test = X_raw_test
y_test_list = [ [ [left_hours] for left_hours in reversed(range(ep.shape[0]))] for ep in X_test ]
y_test = np.array(y_test_list)


# ## Create a dataframe for batch manipulation

# In[5]:

# Create training dataframe
pred_hours = 8
# hours_left = []
# dis_list = []
new_epiosdes = []
count_down = []
y_out = []

for x, y in zip(X_raw_train, y_train_list):
    dis = np.random.binomial(1, 0.5)
    
    if dis==1:
        rand_cut = random.randrange(0, -pred_hours, -1)
#         hours_left.append(-rand_cut)
#         dis_list.append(dis)
        y_out.append([dis, -rand_cut])
        
        if rand_cut==0:
            new_epiosdes.append(x[:None]) 
            count_down.append(y[:None])
        else:
            new_epiosdes.append(x[:rand_cut])
            count_down.append(y[:rand_cut])
        
    else:

        rand_cut = random.randrange(-pred_hours, -x.shape[0], -1)
#         hours_left.append(-rand_cut)
#         dis_list.append(dis)
        y_out.append([dis, -rand_cut])

        new_epiosdes.append(x[:rand_cut])
        count_down.append(y[:rand_cut])


train_df = df({'episodes': new_epiosdes, 
               'ori_hours' : [h.shape[0] for h in X_raw_train],
               'hours_pass': [h.shape[0] for h in new_epiosdes],
               'ylos': y_raw_train,
               'count_down': count_down,
#                'hours_left': hours_left,
#                'dis': dis_list,
               'dis_hrLeft': y_out}) #.sort_values(by ='hours')


# train_df


# In[6]:

# pred_hours = 8
# hours_left = []
# dis_list = []
new_epiosdes = []
count_down = []
y_out = []

for x, y in zip(X_raw_test, y_test_list):
    dis = np.random.binomial(1, 0.5)
    
    if dis==1:
        rand_cut = random.randrange(0, -pred_hours, -1)
#         hours_left.append(-rand_cut)
#         dis_list.append(dis)
        y_out.append([dis, -rand_cut])

        
        if rand_cut==0:
            new_epiosdes.append(x[:None]) 
            count_down.append(y[:None])
        else:
            new_epiosdes.append(x[:rand_cut])
            count_down.append(y[:rand_cut])
        
    else:
        rand_cut = random.randrange(-pred_hours, -x.shape[0], -1)
#         hours_left.append(-rand_cut)
#         dis_list.append(dis)
        y_out.append([dis, -rand_cut])

        new_epiosdes.append(x[:rand_cut])
        count_down.append(y[:rand_cut])
        
test_df = df({'episodes': new_epiosdes, 
               'ori_hours' : [h.shape[0] for h in X_raw_test],
               'hours_pass': [h.shape[0] for h in new_epiosdes],
               'ylos': y_raw_test,
               'count_down': count_down,
#                'hours_left': hours_left,
#                'dis': dis_list,
               'dis_hrLeft': y_out}) #.sort_values(by ='hours')


# test_df


# #### Group batches that have same time length

# In[ ]:

# Create a model with input = (nb_samples, tsteps, nb_features); output = (nb_samples, 1)
print('Creating Model')
nb_features = 14
last_layer_nodes = 1024

buildStart_time = time.time()
model = Sequential()
model.add(LSTM(output_dim = 64, 
               return_sequences = True, 
               input_dim = nb_features,
               # input_shape = (tsteps, nb_features), 
               # dropout_U=0.5, 
               activation='sigmoid', 
               inner_activation='hard_sigmoid'))

# model.add(TimeDistributedDense(output_dim = 64, activation="sigmoid"))

model.add(LSTM(output_dim = last_layer_nodes, 
               # return_sequences = True,
               # input_dim = nb_features,
               # input_shape = (tsteps, nb_features), 
               # dropout_U=0.5,
               activation='sigmoid',
               inner_activation='hard_sigmoid'))

# model.add(TimeDistributedDense(output_dim = 1, activation="linear"))

# model.add(Dense(output_dim = 32,
#                 activation = 'linear'))

model.add(Dense(output_dim = 1,
                activation = 'sigmoid'))

buildEnd_time = time.time()


# Compile the model
print("Build model time:", buildEnd_time - buildStart_time, "secs")
highLearn_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer= highLearn_adam) #, sample_weight_mode = "temporal")

compileEnd_time = time.time()
print("Compile time:", compileEnd_time-buildEnd_time, "secs")



# In[ ]:

train_df_gb = train_df.groupby('hours_pass')
nb_epochs = 800

cutBatch = 90000
cutTimeLen = 100
loss_trend = []
acc_trend = []
uni_train_hours = np.unique(np.asarray(train_df['hours_pass']))
np.random.shuffle(uni_train_hours)

trainStart_time = time.time()
temp = trainStart_time

# Number of times of iteration on the "WHOLE" data set
for epoch_it in range(nb_epochs):
    inEpo_loss_trend = []
    inEpo_acc_trend = []
    batchCount = 0
    np.random.shuffle(uni_train_hours)
    
    # Grouping and training on batches of samples according to the length of time
    for time_len in uni_train_hours:
        
        if batchCount > cutBatch or time_len > cutTimeLen:
            continue
            
        else:
            maxTrainLen = time_len
            lastTrainBat = batchCount
            
            g = train_df_gb.get_group(time_len)

            batch_size = len( g['episodes'])
            X_train = np.dstack(np.asarray(g['episodes']))
            X_train = np.transpose(X_train, (2,0,1))
            
            y_train = [ d_h[0] for d_h in g['dis_hrLeft']]



            trainMid_time = time.time()
            
#             # Create output mask
#             out_mask_step = 10
#             output_mask = np.zeros(shape = (batch_size, time_len))    
#             ## Specify timesteps to be revealed
#             x = np.ones(shape = (batch_size, output_mask[:,::-10].shape[1] ) )
#             output_mask[:,::-out_mask_step] = x
            
            (loss, acc) = model.train_on_batch(X_train, 
                                               y_train,
                                               accuracy = True) #, sample_weight = output_mask)
        
        
            p = ("Epoch:" 
                 + str(epoch_it+1)
                 + "/"
                 + str(nb_epochs)
                 + " Time length: " 
                 + str(time_len) 
                 + ", # of samples: " 
                 + str(batch_size)
                 + " || loss = "
                 + str(round(loss,4))
                 + ", acc = "
                 + str(round(acc,4))
                 + ". It's been "
                 + str(round(trainMid_time-trainStart_time, 1))
                 + ' secs')
            sys.stdout.write('\r' + p)

            inEpo_loss_trend.append(float(loss))
            inEpo_acc_trend.append(float(acc))
            batchCount = batchCount + 1
    
    
    aEpoch_time = time.time()
    epochLapse = aEpoch_time - temp
    avg_loss = sum(inEpo_loss_trend)/len(inEpo_loss_trend)
    avg_acc = sum(inEpo_acc_trend)/len(inEpo_acc_trend)
    
    loss_trend.append(avg_loss)
    acc_trend.append(avg_acc)
    
    print('\nThis epoch takes:', round(epochLapse,4), 
          "secs. Avg. Loss:", round(avg_loss,4), 
          "Avg. Acc:", round(avg_acc,4),
          "\n")
    
    temp = aEpoch_time

trainEnd_time = time.time()
print ("\nDone!\nThis takes:", trainEnd_time-trainStart_time, "seconds")


# In[ ]:

# Test all the output

y_test = [ d_h[0] for d_h in test_df['dis_hrLeft']]
y_predict = []


for X_test in test_df['episodes']: 
    
    X_test = X_test[None,:,:] 
    y_result = model.predict(X_test)
    y_predict.append(float(y_result))

# compare = df({'y_predict': y_predict, 'y_test': y_test})  

# y_predict = np.asarray(compare.y_predict)
# y_test = np.asarray(compare.y_test)

y_predict = np.array(y_predict)
y_test = np.array(y_test)

testLoss = abs(y_predict-y_test)
# mseTestLoss = LA.norm(testLoss)
# print(mseTestLoss)
 
# np.unique(compare.y_predict)


# In[ ]:

# model.reset_states()
X_test = train_df.episodes.iloc[225][None,:,:]
X_test.shape


# In[ ]:

y_result = model.predict(X_test)
y_result.shape
y_test = train_df.dis_hrLeft.iloc[225]
print(y_test)
y_result


# In[ ]:

SAVE_PATH = "/Users/hi08060204/Desktop/Peter_ML/MIMIC_Project/Result"
plt.figure()
plt.plot(loss_trend)
# plt.show()
plt.savefig(os.path.join(SAVE_PATH,"loss_trend.png"))

plt.figure()
plt.plot(acc_trend)
plt.savefig(os.path.join(SAVE_PATH,"acc_trend.png"))


# In[ ]:



