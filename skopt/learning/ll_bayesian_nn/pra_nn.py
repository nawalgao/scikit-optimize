#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:56:15 2018

@author: nimishawalgaonkar
"""

import keras.backend as K

class KerasDropoutPrediction(object):
   def __init__(self,model):
       self.f = K.function(
               [model.layers[0].input,
                K.learning_phase()],
               [model.layers[-1].output])
   def predict(self,x, n_iter=10):
       result = []
       for _ in range(n_iter):
           result.append(self.f([x , 1]))
       result = np.array(result).reshape(n_iter,len(x)).T
       return result
def fit_model(prob_id):
   trainFilename = ‘../folds/train-%s-20_01_01.csv’%prob_id
   testFilename = ‘../folds/test-%s-20_01_01.csv’%prob_id
   trainDF = pd.read_csv(trainFilename)
   testDF = pd.read_csv(testFilename)
   headers = trainDF.columns.values.tolist()
   inputs = list(set(headers) - set(outputs))
   inputs = list(set(inputs) - set(fold))
   trainX = trainDF.loc[:, inputs].as_matrix()
   testX = testDF.loc[:, inputs].as_matrix()
   trainY = trainDF[output].as_matrix()
   testY = testDF[output].as_matrix()
   preProcModelInput = preprocessing.MinMaxScaler()
   preProcModelInput.fit_transform(trainX)
   trainX = preProcModelInput.transform(trainX)
   testX = preProcModelInput.transform(testX)
   preProcModelOutput = preprocessing.MinMaxScaler()
   preProcModelOutput.fit_transform(trainY.reshape(-1, 1))
   trainY = preProcModelOutput.transform(trainY.reshape(-1, 1))
   trainY = np.squeeze(np.asarray(trainY))
   testY = preProcModelOutput.transform(testY.reshape(-1, 1))
   testY = np.squeeze(np.asarray(testY))
   # design network
   nunits = 512
   dropout = 0.5
   hidden_size = 1
   uq = True
   input_shape = (trainX.shape[1],)
   inputs = Input(shape=input_shape)
   x = Dense(nunits, activation=‘relu’)(inputs)
   x = Dropout(dropout)(x, training=uq)
   for j in range(hidden_size):
       x = Dense(nunits, activation=‘relu’)(x)
       x = Dropout(dropout)(x, training=uq)
   level_all = Dense(1, name=‘output’)(x)
   model = Model(inputs=inputs, outputs=level_all)
   model.compile(loss=‘mse’, optimizer=‘adam’, metrics=[‘mae’])
   model.summary()
   #model = Sequential()
   #model.add(Dense(512, activation=‘relu’, input_shape=(trainX.shape[1],)))
   #model.add(Dropout(0.5))
   #model.add(Dense(512, activation=‘relu’))
   #model.add(Dropout(0.5))
   #model.add(Dense(1))
   #model.compile(loss=‘mse’, optimizer=‘adam’,metrics=[‘mae’])
   # fit network
   history = model.fit(trainX, trainY, epochs=100, batch_size=512, validation_data=(testX, testY), verbose=0, shuffle=True)
   # plot history
   pyplot.plot(history.history[‘loss’], label=‘train’)
   pyplot.plot(history.history[‘val_loss’], label=‘test’)
   pyplot.legend()
   pyplot.show()
   testYhat = model.predict(testX)
   kdp = KerasDropoutPrediction(model)
   y_pred_do = kdp.predict(testX, n_iter=100)
   y_pred_do_mean = y_pred_do.mean(axis=1)
   print(y_pred_do.shape)
   pyplot.scatter(testYhat, y_pred_do_mean, s=80, marker=“+”)
   pyplot.show()
   diffs = testYhat - testY
   yhat = preProcModelOutput.inverse_transform(testYhat.reshape(-1, 1))
   y = preProcModelOutput.inverse_transform(testY.reshape(-1, 1))
   res_df = np.concatenate((y, yhat), axis=1)
   return res_df