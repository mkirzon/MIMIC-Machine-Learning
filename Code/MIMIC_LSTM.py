from __future__ import print_function
import numpy as np
import sys, os, time, csv, random, logging
import cPickle as pickle
from pandas import DataFrame as df
from numpy import linalg as LA
import matplotlib.pyplot as plt
# import pandas as pd

np.random.seed(1337)  # for reproducibility

# from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense #, Dropout, Activation, TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop, SGD

##################################################################################
# ### Load data from pickle file and prepare training set and testing set
##################################################################################


def load_data(DATA_PATH='/Users/hi08060204/Desktop/OneDrive/Courses/4/Peter_ML/MIMIC_Project/Data/Processed',
              filename='mimic3_11_resampled-imputed_5000.npz',
              nb_samples_using=None,
              test_split=0.2,
              ):

    # Loading data from .npz file into dictionary form
    data = np.load(os.path.join(DATA_PATH, filename))
    X_raw = data['X'][:nb_samples_using]
    y_raw = data['ylos'][:nb_samples_using]

    # Spliting data for the use of training and testing
    X_raw_test = X_raw[:len(X_raw) * test_split]
    y_raw_test = y_raw[:len(y_raw) * test_split]
    X_raw_train = X_raw[len(X_raw) * test_split:]
    y_raw_train = y_raw[len(y_raw) * test_split:]

    return {'X_raw_train': X_raw_train,
            'y_raw_train': y_raw_train,
            'X_raw_test': X_raw_test,
            'y_raw_test': y_raw_test}

##################################################################################
# ### Prepare the date according to output mode
##################################################################################


def cut_data(X, output_mode="binary", pred_hours=[8], min_ob_window = 4):
    # X = dataframe['episodes']
    X_out = []
    y_out = []



    if output_mode == "binary":

        pred_hours = pred_hours[0]

        for x in X:

            if x.shape[0] < pred_hours + min_ob_window:
                continue

            else:

                dis = np.random.binomial(1, 0.5)

                if dis == 1:
                    rand_cut = random.randrange(0, -pred_hours, -1)
                    y_out.append(dis)

                    if rand_cut == 0:

                        X_out.append(x[:None])

                    else:

                        X_out.append(x[:rand_cut])



                else:

                    rand_cut = random.randrange(-pred_hours, -x.shape[0], -1)
                    y_out.append(dis)
                    X_out.append(x[:rand_cut])




    elif output_mode == "continuous":

        for x in X:

            if x.shape[0] < pred_hours + min_ob_window:
                continue

            else:
                rand_cut = np.random.binomial(min_ob_window, x.shape[0] - 1)

                X_out.append(x[:rand_cut])
                y_out.append(x.shape[0] - rand_cut)


    elif output_mode == "multi-class":

        pred_hours = [0] + pred_hours
        nb_class = len(pred_hours)

        for x in X:

            ix = random.randrange(0, nb_class)
            y = [0 for _ in range(nb_class)]

            if ix == 0:

                if x.shape[0] < pred_hours[-1] + min_ob_window + 1:
                    continue

                else:

                    rand_cut = random.randrange(-pred_hours[-1], -(x.shape[0]-min_ob_window), -1)


            else:

                if x.shape[0] < pred_hours[-1] + min_ob_window:
                    continue

                else:

                    rand_cut = random.randrange(-pred_hours[ix], -pred_hours[ix - 1])

            y[ix] = 1
            y_out.append(y)

            X_out.append(x[:rand_cut])

    return X_out, y_out


##################################################################################
# ### Construct anc Prepare dataframe for better manipulation
##################################################################################


def prepare_df(X, Y, output_mode="binary", pred_hours=[8], min_ob_window = 4):

    (X_out, y_out) = cut_data(X, output_mode = output_mode, pred_hours = pred_hours, min_ob_window= min_ob_window)

    out_df = df({'X': X_out,
                 'hours_pass': [ h.shape[0] for h in X_out ],
                 # 'ylos': Y,
                 'y': y_out})  # .sort_values(by ='hours')

    return out_df


##################################################################################
# ### Configure the model
##################################################################################


def create_model(nb_features=14, output_mode="binary", output_dim_list=[64, 1024, 1]):
    print('Creating Model...')

    buildStart_time = time.time()
    model = Sequential()

    if len(output_dim_list) == 1:
        model.add(LSTM(output_dim=output_dim_list[0],
                       activation='sigmoid',
                       inner_activation='hard_sigmoid'))
    else:

        for i, out_d in enumerate(output_dim_list[:-1]):

            if len(output_dim_list[:-1]) == 1 and i == 0:

                model.add(LSTM(output_dim=out_d,
                               input_dim=nb_features,
                               activation='sigmoid',
                               inner_activation='hard_sigmoid'))

            elif len(output_dim_list[:-1]) > 1 and i == 0:

                model.add(LSTM(output_dim=out_d,
                               return_sequences=True,
                               input_dim=nb_features,
                               activation='sigmoid',
                               inner_activation='hard_sigmoid'))

            elif i == len(output_dim_list[:-1])-1:

                model.add(LSTM(output_dim=out_d,
                               activation='sigmoid',
                               inner_activation='hard_sigmoid'))

            else:
                model.add(LSTM(output_dim=out_d,
                               activation='sigmoid',
                               return_sequences=True,
                               inner_activation='hard_sigmoid'))

        # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # For different output mode, use different output activation function and loss function

        if output_mode == "binary":

            model.add(Dense(output_dim=output_dim_list[-1],
                            activation='sigmoid'))
            buildEnd_time = time.time()
            print("Build model time:", buildEnd_time - buildStart_time, "secs")

            model.compile(loss="binary_crossentropy", optimizer="adam")

        elif output_mode == "multi-class":

            model.add(Dense(output_dim=output_dim_list[-1],
                            activation='softmax'))
            buildEnd_time = time.time()
            print("Build model time:", buildEnd_time - buildStart_time, "secs")

            model.compile(loss="categorical_crossentropy", optimizer="adam")

        elif output_mode == "continuous":

            model.add(Dense(output_dim=output_dim_list[-1],
                            activation='linear'))
            buildEnd_time = time.time()
            print("Build model time:", buildEnd_time - buildStart_time, "secs")

            model.compile(loss="mae", optimizer="adam")

        else:

            model.add(Dense(output_dim=output_dim_list[-1],
                            activation='sigmoid'))
            buildEnd_time = time.time()
            print("Build model time:", buildEnd_time - buildStart_time, "secs")

            model.compile(loss="mse", optimizer="adam")

    compileEnd_time = time.time()
    print("Compile time:", compileEnd_time - buildEnd_time, "secs")

    return model


##################################################################################
# ### Segment dataframe for validation
##################################################################################

def set_val(input_df, val_split = 0.1):

    input_ix = range(len(input_df))
    np.random.shuffle(input_ix)

    cut_point = int(val_split * len(input_df))
    val_ix = input_ix[:cut_point]
    train_ix = input_ix[cut_point:]

    train_df = input_df.iloc[train_ix]
    val_df = input_df.iloc[val_ix]

    return train_df, val_df


##################################################################################
# ### Train the model
##################################################################################


def train_model(model, train_df, nb_epochs=10, val_split=0.1, samp_limit=False, cutBatch=9000, cutTimeLen=5000):


    train_loss_trend = []
    train_acc_trend = []
    val_loss_trend = []
    val_acc_trend = []
    input_train_df = train_df

    trainStart_time = time.time()
    temp_time = trainStart_time

    # Number of times of iteration on the "WHOLE" data set

    print("Training Model...")
    for epoch_it in range(nb_epochs):

        train_df, val_df = set_val(input_df= input_train_df, val_split=val_split)
        train_df_gb = train_df.groupby('hours_pass')
        uni_train_hours = np.unique(np.asarray(train_df['hours_pass']))
        np.random.shuffle(uni_train_hours)

        inEpo_train_loss_trend = []
        inEpo_train_acc_trend = []
        inEpo_val_loss_trend = []
        inEpo_val_acc_trend = []
        batchCount = 0
        np.random.shuffle(uni_train_hours)

        # Grouping and training on batches of samples according to the length of time
        for time_len in uni_train_hours:

            if samp_limit == True and (batchCount > cutBatch or time_len > cutTimeLen):
                continue
            else:
                pass


            maxTrainLen = time_len
            lastTrainBat = batchCount

            g = train_df_gb.get_group(time_len)

            batch_size = len(g['X'])
            X_train = np.dstack(np.asarray(g['X']))
            X_train = np.transpose(X_train, (2, 0, 1))
            y_train = [y for y in g['y']]

            #             # Create output mask
            #             out_mask_step = 10
            #             output_mask = np.zeros(shape = (batch_size, time_len))
            #             ## Specify timesteps to be revealed
            #             x = np.ones(shape = (batch_size, output_mask[:,::-10].shape[1] ) )
            #             output_mask[:,::-out_mask_step] = x

            (loss, acc) = model.train_on_batch(X_train,
                                               y_train,
                                               accuracy=True)  # , sample_weight = output_mask)


            trainMid_time = time.time()

            p = ("Epoch:"
                 + str(epoch_it + 1)
                 + "/"
                 + str(nb_epochs)
                 + " Time length: "
                 + str(time_len)
                 + ", # of samples: "
                 + str(batch_size)
                 + " || loss = "
                 + str(round(loss, 4))
                 + ", acc = "
                 + str(round(acc, 4))
                 + ". It's been "
                 + str(round(trainMid_time - trainStart_time, 1))
                 + ' secs')
            sys.stdout.write('\r' + p)

            inEpo_train_loss_trend.append(float(loss))
            inEpo_train_acc_trend.append(float(acc))
            batchCount = batchCount + 1

        inEpo_avg_loss = sum(inEpo_train_loss_trend) / len(inEpo_train_loss_trend)
        inEpo_avg_acc = sum(inEpo_train_acc_trend) / len(inEpo_train_acc_trend)

        train_loss_trend.append(inEpo_avg_loss)
        train_acc_trend.append(inEpo_avg_acc)
        avg_loss = sum(train_loss_trend) / len(train_loss_trend)
        avg_acc = sum(train_acc_trend) / len(train_acc_trend)

        aEpoch_time = time.time()
        epochLapse = aEpoch_time - temp_time
        temp_time = aEpoch_time

        p =("Epoch:" + str(epoch_it + 1) + "/" + str(nb_epochs)
            + " it takes: " + str(round(epochLapse, 4))
            + " secs, # batches: " + str(len(uni_train_hours))
            + ", train_loss: " + str(round(inEpo_avg_loss,4))
            + " train_acc: " + str(round(inEpo_avg_acc,4))
            + " || (Overall) Avg. Loss: " + str(round(avg_loss, 4))
            + " Avg. Acc: " + str(round(avg_acc, 4))
            + "\n"
            )

        sys.stdout.write('\n' + '\r' + p)



        if val_split > 0:


            for samp in val_df.iterrows():

                y_val = [ samp[1]['y'] ]
                X_val = samp[1]['X'][None,:,:]



                val_loss, val_acc = model.test_on_batch(X_val, y_val, accuracy=True)

                inEpo_val_loss_trend.append(float(val_loss))
                inEpo_val_acc_trend.append(float(val_acc))


            inEpo_val_avg_loss = sum(inEpo_val_loss_trend) / len(inEpo_val_loss_trend)
            inEpo_val_avg_acc = sum(inEpo_val_acc_trend) / len(inEpo_val_acc_trend)

            val_loss_trend.append(inEpo_val_avg_loss)
            val_acc_trend.append(inEpo_val_avg_acc)
            val_avg_loss = sum(val_loss_trend) / len(val_loss_trend)
            val_avg_acc = sum(val_acc_trend) / len(val_acc_trend)

            p = ("Validation:"
                  + " val_loss: " + str(round(inEpo_val_avg_loss,4))
                  + " val_acc: " + str(round(inEpo_val_avg_acc,4))
                  + " || (Overall) Avg. Loss: " + str(round(val_avg_loss, 4))
                  + " Avg. Acc: " + str(round(val_avg_acc, 4))
                  + "\n"
                  + "\n")

            sys.stdout.write('\r' + p)


    trainEnd_time = time.time()
    print("\nDone!\nTraining takes:", trainEnd_time - trainStart_time, "seconds\n")

    train_result = df({'train_loss': train_loss_trend,
                       'train_acc': train_acc_trend,
                       'val_loss': val_loss_trend,
                       'val_acc': val_acc_trend
                       })

    return model, train_result


##################################################################################
# ### Test the model
##################################################################################


def test_model(model, test_df):

    y_predict = []
    test_loss_trend = []
    test_acc_trend = []
    test_mv_avg_loss = []
    test_mv_avg_acc = []

    print("Now Predicting...")

    for i, test_samp in enumerate(test_df.iterrows()):

        X_test = test_samp[1]['X'][None,:,:]
        y_test = [ test_samp[1]['y'] ]


        y_result = model.predict(X_test)

        if y_result.shape[1] == 1:

            if float(y_result) > 0.5:
                y_predict.append(1)
            else:
                y_predict.append(0)
        else:
            max_ix = np.argmax(y_result)
            y_class = np.zeros(y_result.shape[1])
            y_class[max_ix] = 1
            y_predict.append(y_class)


        test_loss, test_acc = model.test_on_batch(X_test, y_test, accuracy = True)

        test_loss_trend.append(float(test_loss))
        test_acc_trend.append(float(test_acc))

        test_avg_loss = sum(test_loss_trend)/len(test_loss_trend)
        test_avg_acc = sum(test_acc_trend)/len(test_acc_trend)

        test_mv_avg_loss.append(test_avg_loss)
        test_mv_avg_acc.append(test_avg_acc)


        per = round( ((i+1)*100)/test_df.shape[0], 4)
        p = (str(i+1)
             + "/"
             + str(len(test_df['X']))
             + " predicted. "
             + str(per)
             + "% || Avg. loss: "
             + str(round(test_avg_loss, 4))
             + ", Avg. acc: "
             + str(round(test_avg_acc, 4))
             )

        sys.stdout.write('\r' + p)


    y_test = [y for y in test_df['y']]
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)

    testLoss = abs(y_predict-y_test)
    mseTestLoss = LA.norm(testLoss)

    test_result = df({'y_predict': y_predict,
                      'y_test': y_test,
                      'loss': test_loss_trend,
                      'acc': test_acc_trend,
                      'mv_loss': test_mv_avg_loss,
                      'mv_acc': test_mv_avg_acc
                      })


    return test_result


##################################################################################
# ### Save Result
##################################################################################


def save_figure(SAVE_PATH, filename, data):

    plt.clf()
    plt.plot(data)
    plt.savefig(os.path.join(SAVE_PATH, filename))

    return True



def save_result(model,
                train_result_df,
                test_result_df,
                SAVE_ROOT_PATH = "/Users/hi08060204/Desktop/OneDrive/Courses/4/Peter_ML/MIMIC_Project/Result",
                pred_hours = [8],
                output_mode="binary",
                output_dim_list = [64, 1024, 1],
                nb_epochs = 100,
                output_figure = False):

    print("\n\nNow Saving Result...")

    out_str = "_win"

    for pd in pred_hours:
        out_str = out_str + "_" + str(pd)

    out_str = out_str + "_od"

    for od in output_dim_list:
        out_str = out_str + "_" + str(od)

    out_str = output_mode + out_str + "_" + "Epo" + str(nb_epochs)

    SAVE_PATH = SAVE_ROOT_PATH + "/" + out_str

    try:
        os.makedirs(SAVE_PATH)
        print("\ncreate new folder")
    except:
        pass

    if output_figure:
        train_loss_fn = "train_loss_" + out_str + ".png"
        train_acc_fn = "train_acc_" + out_str + ".png"
        val_loss_fn = "val_loss_" + out_str + ".png"
        val_acc_fn = "val_acc_" + out_str + ".png"
        test_loss_fn = "test_loss_" + out_str + ".png"
        test_acc_fn = "test_acc_" + out_str + ".png"
        test_mv_avg_loss_fn = "test_mv_avg_loss_" + out_str + ".png"
        test_mv_avg_acc_fn = "test_mv_avg_acc_" + out_str + ".png"

        save_figure(SAVE_PATH=SAVE_PATH, filename=train_loss_fn, data=train_result_df['train_loss'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=train_acc_fn, data=train_result_df['train_acc'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=val_loss_fn, data=train_result_df['val_loss'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=val_acc_fn, data=train_result_df['val_acc'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=test_loss_fn, data=test_result_df['loss'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=test_acc_fn, data=test_result_df['acc'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=test_mv_avg_loss_fn, data=test_result_df['mv_loss'])
        save_figure(SAVE_PATH=SAVE_PATH, filename=test_mv_avg_acc_fn, data=test_result_df['mv_acc'])
    else:
        pass


    result_fn = SAVE_PATH + "/" + "result_" + out_str + ".npz"
    np.savez(result_fn,train_loss=train_result_df['train_loss'],
                       train_acc=train_result_df['train_acc'],
                       val_loss=train_result_df['val_loss'],
                       val_acc=train_result_df['val_acc'],
                       test_loss=test_result_df['loss'],
                       test_acc=test_result_df['acc'],
                       y_predict=test_result_df['y_predict'],
                       y_test=test_result_df['y_test']
             )

    config_fn = "model_config_" + out_str + ".yaml"
    model_config = model.to_yaml()
    pickle.dump(model_config, open(os.path.join(SAVE_PATH, config_fn), "wb"))

    weights_fn = "model_weights_" + out_str + ".h5"
    model.save_weights(os.path.join(SAVE_PATH, weights_fn), overwrite=True)

    print(out_str)

    return True

##################################################################################
# ### Setting up parameters
##################################################################################

# DATA_PATH = "/Users/hi08060204/Desktop/OneDrive/Courses/4/Peter_ML/MIMIC_Project/Data/Processed"
# input_fn = 'mimic3_11_resampled-imputed_5000.npz'
# RESULT_PATH = "/Users/hi08060204/Desktop/OneDrive/Courses/4/Peter_ML/MIMIC_Project/Result"
#
# pred_hours = [8]
# min_ob_window = 4
# nb_epochs = 200
# output_mode = "binary"
# output_dim = [16, 64, 1]


DATA_PATH = sys.argv[1]
input_fn = sys.argv[2]
RESULT_PATH = sys.argv[3]

pred_hours = [int(sys.argv[4])]
min_ob_window = int(sys.argv[5])
nb_epochs = int(sys.argv[6])
output_mode = sys.argv[7]
output_dim = [int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]), int(sys.argv[11])]

##################################################################################
# ### Reconstruct .txt file from output
##################################################################################




##################################################################################
# ### Execution
##################################################################################

raw_data = load_data(DATA_PATH=DATA_PATH,
                     filename= input_fn,
                     nb_samples_using=None,
                     test_split=0.2)

(X_raw_train, y_raw_train, X_raw_test, y_raw_test) = (raw_data['X_raw_train'], raw_data['y_raw_train'], raw_data['X_raw_test'], raw_data['y_raw_test'])

train_df = prepare_df(X = X_raw_train,
                      Y = y_raw_train,
                      output_mode=output_mode,
                      pred_hours=pred_hours,
                      min_ob_window = min_ob_window)

test_df = prepare_df(X=X_raw_test,
                     Y=y_raw_test,
                     output_mode=output_mode,
                     pred_hours=pred_hours,
                     min_ob_window=min_ob_window)

model = create_model(nb_features=14, output_mode=output_mode, output_dim_list=output_dim)

model, train_result_df = train_model(model = model,
                                     train_df= train_df,
                                     nb_epochs= nb_epochs,
                                     val_split=0.1,
                                     samp_limit=False,
                                     cutBatch=90000,
                                     cutTimeLen=9000
                                     )

test_result_df = test_model(model = model, test_df=test_df)

save_result(model=model,
            train_result_df=train_result_df,
            test_result_df=test_result_df,
            SAVE_ROOT_PATH=RESULT_PATH,
            pred_hours=pred_hours,
            output_mode=output_mode,
            output_dim_list=output_dim,
            nb_epochs=nb_epochs,
            output_figure=True)

# print(model.get_config())





