import tensorflow as tf
from keras.layers import Conv1D,Dense,MaxPooling1D,Bidirectional,LSTM,Input,GlobalAveragePooling1D
from keras.layers import Dropout,LeakyReLU,BatchNormalization,Concatenate,add
from keras.models import Model
from keras import optimizers

def DBL(filters,kernel_size,n,connect_layer):
  x = Conv1D(filters,kernel_size,padding='same',activation=LeakyReLU(alpha=0.1))(connect_layer)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  for i in range(1,n):
    x = Conv1D(filters,kernel_size,padding='same',activation=LeakyReLU(alpha=0.1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
  return x

def res_block1D(filters,alpha,beta,n,connect_layer):
  x = MaxPooling1D()(connect_layer)
  x = DBL(filters,kernel_size=2,n=1,connect_layer=x)
  for i in range(0,n):
    previous_block_activation = x
    x = DBL(alpha,kernel_size=1,n=1,connect_layer=x)
    x = DBL(beta,kernel_size=2,n=1,connect_layer=x)
    x = add([x,previous_block_activation])
  return x

def model_main(input_shape):
  input_layer = Input(shape=input_shape)
  x = DBL(filters=32,kernel_size=1,n=1,connect_layer=input_layer)
  x = res_block1D(connect_layer = x,filters=64,alpha=32,beta=64,n=1)
  x = res_block1D(connect_layer = x,filters=128,alpha=64,beta=128,n=2)
  x = res_block1D(connect_layer = x,filters=256,alpha=128,beta=256,n=8)
  x = GlobalAveragePooling1D()(x)
  x1 = Bidirectional(LSTM(256,activation = 'tanh',dropout=0.3))(input_layer)
  concat_layer = Concatenate(axis=1)([x1,x])
  x2 = Dense(64,activation=LeakyReLU(alpha=0.1))(concat_layer)
  x2 = Dense(1,activation = 'linear')(x2)
  model = Model(inputs=input_layer,outputs=x2)
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001,epsilon=1e-3), loss='mse')
  return model