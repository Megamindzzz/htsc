
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import keras
import random
from factor_processing_utils import filter_extreme_3sigma_df,neutralization,filter_extreme_3sigma,norm,standardize_series
from neural_utils import LossHistory
from keras.layers import Conv2D,LSTM,Concatenate,Dense,Dropout,BatchNormalization,Input
from keras import layers
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime,timedelta
import time
from keras.utils import np_utils
from keras.models import Model
from keras.models import load_model


# In[ ]:


TensorBoard


# In[2]:


start_date = '2015-01-05'
end_date = '2019-05-10'


# In[3]:


feature_list = []
with pd.HDFStore('../AShare/valuation','r') as data:
    cap = data['market_cap']
    cap.index = pd.to_datetime(cap.index)
    cap = cap.loc[start_date:end_date]
    #feature_list.append(cap)
    pb = data['pb_ratio']
    pb.index = pd.to_datetime(pb.index)
    pb = pb.loc[start_date:end_date]
    feature_list.append(pb)
    pe = data['pe_ratio']
    pe.index = pd.to_datetime(pe.index)
    pe = pe.loc[start_date:end_date]
    feature_list.append(pe)
    ps = data['ps_ratio']
    ps.index = pd.to_datetime(ps.index)
    ps = ps.loc[start_date:end_date]
    feature_list.append(ps)
with pd.HDFStore('../AShare/industry','r') as data:
    industry_code = data['industry_code']
    industry_code.index = pd.to_datetime(industry_code.index)
    industry_code = industry_code.loc[start_date:end_date]
with pd.HDFStore('../AShare/primary','r') as data:
    close = data['close'].loc[start_date:end_date]
    adjfactor = data['adj_factor'].loc[start_date:end_date]
    close_adj = close*adjfactor
with pd.HDFStore('../AShare/indicator','r') as data:
    for key in data.keys():
        try:
            temp = data[key].astype(np.float64)
            if temp.index[0]==datetime.strptime("2016-01-04", "%Y-%m-%d"):  
                temp.index = pd.to_datetime(temp.index)
                feature_list.append(temp)
            else:
                print(key)
        except:
            print(key)
len(feature_list)


# In[4]:


with pd.HDFStore('../factors/worldquant101') as data:
    for key in data.keys():
        try:
            temp = data[key].astype(np.float64)
            if temp.index[0]==datetime.strptime("2016-01-04", "%Y-%m-%d"):  
                temp.index = pd.to_datetime(temp.index)
                feature_list.append(temp)
            else:
                print(key)
        except:
            print(key)
len(feature_list)


# In[5]:


returns = close_adj.pct_change().shift(-1)


# In[6]:


def preprocessing(factor,day,cap):
    factor= factor.fillna(0.001)
    #print(factor.loc[day])
    #print(feature_list.index(factor))
    factor_no_extreme = filter_extreme_3sigma(factor.loc[day])
    factor_no_extreme = standardize_series(factor_no_extreme)
    factor_series = neutralization(factor_no_extreme,industry_code.loc[day],cap.loc[day])
    factor_series.columns = [str(random.randint(0,200))]
    return factor_series
    
def data_test(date_list,feature_list):
    input_data_all = None
    for i in range(len(date_list)): 
        print('processing:',date_list[i])
        day_interval = date_list[i:i+1]
        #returns = close_adj.pct_change().shift(-1)
        input_data = pd.DataFrame()
        for day in day_interval:
            cap_day = cap.loc[day]
            cap_day = np.log(cap_day.astype(np.float64))
            cap_day = (cap_day-cap_day.mean())/cap_day.std()
            cap_day.columns = ['cap']
            input_data =  pd.concat([input_data,cap_day],axis=1)
            #p = Pool(4)
            res = []
            for factor in feature_list:
#                 res.append(p.apply_async(preprocessing, args=(factor,day,cap, ))
#                 p.close()
#                 p.join()
#                 for ii in res:
#                     input_data = pd.concat([input_data,ii.get()],axis=1)
                factor= factor.fillna(0.001)
                factor_no_extreme = filter_extreme_3sigma(factor.loc[day])
                factor_no_extreme = standardize_series(factor_no_extreme)
                factor_series = neutralization(factor_no_extreme,industry_code.loc[day],cap.loc[day])
                factor_series.columns = [str(random.randint(0,200))]
                input_data = pd.concat([input_data,factor_series],axis=1)

        input_data.fillna(0.001,inplace=True)
        if input_data_all is None:
            input_data_all = input_data.values
        else:
            input_data_all = np.row_stack([input_data_all,input_data.values])
    return input_data_all
date_l = close.index[:1]
temp_data = data_test(date_l,feature_list)


# In[7]:


print(len(feature_list))
temp_data.shape


# In[8]:


def data_prepare(returns,feature_list):
    input_data_all = None
    for i in range(len(returns.index)): 
        print('processing:',returns.index[i])
        day_interval = returns.index[i:i+1]
        #returns = close_adj.pct_change().shift(-1)
        input_data = pd.DataFrame()
        for day in day_interval:
            cap_day = cap.loc[day]
            cap_day = np.log(cap_day.astype(np.float64))
            cap_day = (cap_day-cap_day.mean())/cap_day.std()
            cap_day.columns = ['cap']
            input_data =  pd.concat([input_data,cap_day],axis=1)
            for factor in feature_list:
                #factor = norm(factor)
                factor= factor.fillna(0.001)
                factor_no_extreme = filter_extreme_3sigma(factor.loc[day].dropna(axis=0))
                factor_no_extreme = standardize_series(factor_no_extreme)
                factor_series = neutralization(factor_no_extreme,industry_code.loc[day],cap.loc[day])
                factor_series.columns = [str(random.randint(0,200))]
                input_data = pd.concat([input_data,factor_series],axis=1)

                
        return_day = returns.loc[day,:]
        return_day.columns = ['return']
        y_data = return_day.dropna(axis=0)
        lenth = len(y_data)
        my_list = y_data.values.tolist()
        largest_index = np.argsort(my_list)[-1*int(0.25*lenth):]
        smllest_index = np.argsort(my_list)[:int(0.25*lenth)]
        middle_index = np.argsort(my_list)[int(0.4*lenth):int(0.6*lenth)]
        y_data.iloc[:] = -1
        y_data.iloc[largest_index] = 2
        y_data.iloc[smllest_index] = 1
        y_data.iloc[middle_index] = 0
        #print(y_data)
        input_data = pd.concat([input_data,y_data],axis=1)
        #print(input_data)
        #input_data.columns = [str(i) for i in range(59)]
        
        input_data.iloc[:,:-1] = input_data.iloc[:,:-1].fillna(0)
        input_data = input_data.dropna(axis=0)
        input_data = input_data[input_data.iloc[:,-1] != -1]
        #print(input_data.head())
        if input_data_all is None:
            input_data_all = input_data.values
        else:
            input_data_all = np.row_stack([input_data_all,input_data.values])
    return input_data_all
# return_t = returns.iloc[:1,:]
# temp_data = data_prepare(return_t,feature_list)


# In[9]:


len(feature_list)


# In[10]:


temp_data.shape


# In[11]:


def build_model(n_features=len(feature_list)+1):
    input_layer = Input(shape=(n_features,), name='main_input')
    x = Dense(128,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    x = Dense(256,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    x = Dense(256,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    x = Dense(128,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    output = Dense(3,activation='softmax',name='output')(x)
    return Model(inputs=input_layer,outputs=output)


# In[12]:


def training_roll_and_test(returns):
    ####################真正使用，这里日期需要精细调整,这里有问题的，每20天重新训练模型
    data_matrix = None
   # n = 20
    index = returns.index
    predict_df = pd.DataFrame(columns = returns.columns)
    for i in range(len(index)//20):
        date_list = index[i*20:i*20+250]
        date_predict = index[i*20+250:i*20+270]
        if len(date_predict)==0:
            print('Done')
            return predict_df
        return_t = None
        #return_t = None
        if len(date_list)!=250: 
            if len(date_list)<20:
                print('Done')
                #return 1
            else:
                date_list = index[i*20:]
        #elif len(date_predict)
            
        #close_adj = close.loc[date_list]*adjfactor[date_list]
        features = []
        for factor in feature_list:
            if data_matrix is None:
                features.append(factor.loc[date_list])
            else:
                date_append = index[(i-1)*20+250:(i-1)*20+270]
                features.append(factor.loc[date_append])
        if return_t is None:
            return_t = returns.loc[date_list]
        else:
            date_append = index[(i-1)*20+250:(i-1)*20+270]
            return_t = returns.loc[date_append]
        if data_matrix is None:
            data_matrix = data_prepare(return_t,features)
        else:
            temp = data_prepare(return_t,features)
            data_matrix = np.row_stack([data_matrix[i*20:],temp])
        n = data_matrix.shape[0]
        trainX = data_matrix[:int(0.9*n),:-1]
        trainY = data_matrix[:int(0.9*n),-1]
        trainY = np_utils.to_categorical(trainY,num_classes=3)
        validationX = data_matrix[int(0.9*n):,:-1]
        validationY = data_matrix[int(0.9*n):,-1]
        validationY =  np_utils.to_categorical(validationY,num_classes=3)
        
        history = LossHistory()
        tensorboard = TensorBoard(log_dir='log')
        checkpoint = ModelCheckpoint(filepath='weight'+str(i)+'.h5',monitor='val_acc',mode='auto' ,save_best_only='True')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        if i>0:
            weights = './weights/'+'weight'+str(i-1)+'.h5'
            model_finetune = model.load_model(weights)
            model.fit(trainX, trainY,
                        batch_size=16, nb_epoch=50,
                        verbose=1,
                        validation_data=(validationX, validationY),
                        callbacks=[history,tensorboard,checkpoint,early_stopping])
        else:
            weights = None
        model = build_model()
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.fit(trainX, trainY,
                        batch_size=16, nb_epoch=50,
                        verbose=1,
                        validation_data=(validationX, validationY),
                        callbacks=[history,tensorboard,checkpoint,early_stopping])
        model_use = load_model('weight'+str(i)+'.h5')
        
        test_features = data_test(date_predict,feature_list)
        pred = model_use.predict_proba(test_features)[:,0]
        values = pred.reshape(len(close.columns),-1)
        temp_df = pd.DataFrame(values,index=date_predict,columns=close.columns)
        predict_df = pd.concat([predict_df,temp_df])
    return predict_df   

pdf = training_roll_and_test(returns)            
 
            


# In[13]:


model_use = load_model('./weights/'+'weight'+'0'+'.h5')


values = pred.reshape(len(close.columns),-1)
temp_df = pd.DataFrame(values,index=date_predict,columns=close.columns)


# In[ ]:


index = returns.index    
date_predict = index[0*20+250:0*20+270]
test_features = data_test(date_predict,feature_list)


# In[ ]:


pred = model_use.predict(test_features)
pred


# In[32]:


input_data_all = None
for i in range(len(close_adj.index[:-4])): 
    print('processing:',i)
    day_interval = close_adj.index[i:i+5]
    input_data = pd.DataFrame()
    for day in day_interval:
        cap_day = cap.loc[day]
        cap_day = np.log(cap_day.astype(np.float64))
        cap_day = (cap_day-cap_day.mean())/cap_day.std()
        cap_day.columns = ['cap']
        input_data =  pd.concat([input_data,cap_day],axis=1)
        for factor_name in factor_dict.keys():
            factor = factor_dict[factor_name]
            factor_no_extreme = filter_extreme_3sigma(factor.loc[day].dropna(axis=0))
            factor_series = neutralization(factor_no_extreme,industry_code.loc[day],cap.loc[day])
            factor_series.columns = [factor_name]
            input_data = pd.concat([input_data,factor_series],axis=1)
    return_day = returns.loc[day]
    return_day.columns = ['return']
    y_data = return_day.dropna(axis=0)
    lenth = len(y_data)
    my_list = y_data.values.tolist()
    largest_index = np.argsort(my_list)[-1*int(0.3*lenth):]
    smllest_index = np.argsort(my_list)[:int(0.3*lenth)]
    #middle_index = np.argsort(my_list)[int(0.4*lenth):int(0.6*lenth)]
    y_data.iloc[:] = -1
    y_data.iloc[largest_index] = 1
    y_data.iloc[smllest_index] = 0
    #y_data.iloc[middle_index] = 1
    #print(y_data)
    input_data = pd.concat([input_data,y_data],axis=1)
    input_data = input_data.dropna(axis=0)
    input_data = input_data[input_data.iloc[:,-1] != -1]
    if input_data_all is None:
        input_data_all = input_data.values
    else:
        input_data_all = np.row_stack([input_data_all,input_data.values])
input_data_all.shape


# In[14]:


trainX = data_matrix[:int(0.9*n),:-1]
trainY = data_matrix[:int(0.9*n),-1]
trainY = np_utils.to_categorical(trainY,num_classes=3)
validationX = data_matrix[int(0.9*n):,:-1]
validationY = data_matrix[int(0.9*n):,-1]
validationY =  np_utils.to_categorical(validationY,num_classes=3)

history = LossHistory()
tensorboad = Tensorboard(log_dir='log')
checkpoint = ModelCheckpoint(filepath='weight'+str(i)+'.h5',monitor='val_acc',mode='auto' ,save_best_only='True')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model = build_model()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(trainX, trainY,
                batch_size=16, nb_epoch=50,
                verbose=1,
                validation_data=(validationX, validationY),
                callbacks=[history,tensorbord,checkpoint,early_stopping])


# In[7]:


for day in close_adj.index[:1]:
    cap_day = cap.loc[day]
    cap_day = np.log(cap_day.astype(np.float64))
    cap_day = (cap_day-cap_day.mean())/cap_day.std()
    cap_day.columns = ['cap']
    input_data = cap_day
    for factor_name in factor_dict.keys():
        factor = factor_dict[factor_name]
        factor_no_extreme = filter_extreme_3sigma(factor.loc[day].dropna(axis=0))
        factor_series = neutralization(factor_no_extreme,industry_code.loc[day],cap.loc[day])
        factor_series.columns = [factor_name]
        input_data = pd.concat([input_data,factor_series],axis=1)
   # print(input_data.head())
    return_day = returns.loc[day]
    return_day.columns = ['return']
    y_data = return_day.dropna(axis=0)
    lenth = len(y_data)
    my_list = y_data.values.tolist()
    largest_index = np.argsort(my_list)[-1*int(0.3*lenth):]
    smllest_index = np.argsort(my_list)[:int(0.3*lenth)]
    #middle_index = np.argsort(my_list)[int(0.4*lenth):int(0.6*lenth)]
    y_data.iloc[:] = -1
    y_data.iloc[largest_index] = 1
    y_data.iloc[smllest_index] = 0
    #y_data.iloc[middle_index] = 1
    #print(y_data)
    input_data = pd.concat([input_data,y_data],axis=1)
   # print(input_data.head())
    input_data = input_data.dropna(axis=0)
    input_data_cut = input_data[input_data.iloc[:,-1] != -1]
data_matrix = input_data_cut.values


# In[7]:


for day in close_adj.index[1:]:
    cap_day = cap.loc[day]
    cap_day = np.log(cap_day.astype(np.float64))
    cap_day = (cap_day-cap_day.mean())/cap_day.std()
    cap_day.columns = ['cap']
    input_data = cap_day
    for factor_name in factor_dict.keys():
        factor = factor_dict[factor_name]
        factor_no_extreme = filter_extreme_3sigma(factor.loc[day].dropna(axis=0))
        factor_series = neutralization(factor_no_extreme,industry_code.loc[day],cap.loc[day])
        factor_series.columns = [factor_name]
        input_data = pd.concat([input_data,factor_series],axis=1)
   # print(input_data.head())
    return_day = returns.loc[day]
    return_day.columns = ['return']
    y_data = return_day.dropna(axis=0)
    lenth = len(y_data)
    my_list = y_data.values.tolist()
    largest_index = np.argsort(my_list)[-1*int(0.3*lenth):]
    smllest_index = np.argsort(my_list)[:int(0.3*lenth)]
    #middle_index = np.argsort(my_list)[int(0.4*lenth):int(0.6*lenth)]
    y_data.iloc[:] = -1
    y_data.iloc[largest_index] = 1
    y_data.iloc[smllest_index] = 0
   # y_data.iloc[middle_index] = 1
    #print(y_data)
    input_data = pd.concat([input_data,y_data],axis=1)
   # print(input_data.head())
    input_data = input_data.dropna(axis=0)
    input_data_cut = input_data[input_data.iloc[:,-1] != -1]
    data_matrix = np.row_stack((data_matrix,input_data_cut.values))


# In[35]:


from keras.utils import np_utils
# y = data_matrix[:,-1]
# yy = np_utils.to_categorical(y,num_classes=2)
# yy.shape[0]


# In[11]:


y.min()


# In[9]:


n_features = data_matrix.shape[1]
n_features


# In[10]:


n = data_matrix.shape[0]
n


# In[11]:


trainX = data_matrix[:int(0.8*n),:-1]
trainY = data_matrix[:int(0.8*n),-1]
trainY = np_utils.to_categorical(trainY,num_classes=2)
validationX = data_matrix[int(0.8*n):int(0.9*n),:-1]
validationY = data_matrix[int(0.8*n):int(0.9*n),-1]
validationY =  np_utils.to_categorical(validationY,num_classes=2)
test_dataX = data_matrix[int(0.9*n):,:-1]
test_dataY = data_matrix[int(0.9*n):,-1]
test_dataY =  np_utils.to_categorical(test_dataY,num_classes=2)


# In[45]:


from keras_utils import *


# In[13]:


from keras.models import Model


# In[17]:


def build_model(n_features=11):
    input_layer = Input(shape=(n_features,), name='main_input')
    x = Dense(128,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    x = Dense(256,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    x = Dense(256,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    x = Dense(128,activation='relu')(input_layer)
    x = BatchNormalization(name='bn1')(x)
    x = Dropout(0.3)(x)
    output = Dense(3,activation='softmax',name='output')(x)
    return Model(inputs=input_layer,outputs=output)


# In[35]:


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
history = LossHistory()


# In[19]:


model = build_model()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(trainX, trainY,
                batch_size=16, nb_epoch=50,
                verbose=1,
                validation_data=(validationX, validationY),
                callbacks=[history])


# In[20]:


score = model.evaluate(test_dataX, test_dataY, verbose=1)


# In[21]:


score


# In[27]:


import matplotlib.pyplot as plt


# In[44]:


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor,GradientBoostingClassifier
# from xgboost import XGBClassifier
# from xgboost import XGBRegressor


# In[45]:


model = GradientBoostingClassifier()
trainXX = data_matrix[:int(0.8*n),:-1]
trainYY = data_matrix[:int(0.8*n),-1]
model.fit(trainXX,trainYY)


# In[46]:


from sklearn.metrics import accuracy_score


# In[47]:


y_pred = model.predict(data_matrix[int(0.9*n):,:-1])


# In[49]:


accuracy = accuracy_score(data_matrix[int(0.9*n):,-1],y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))


# In[1]:


import numpy as np


# In[2]:


input_data = np.load('input_data.npy')


# In[3]:


label_data = np.load('label.npy')


# In[4]:


import classification_network


# In[5]:


model = classification_network.ResNet50(input_shape=(11,5,1))


# In[6]:


model.summary()


# In[9]:


import random


# In[10]:


index = [i for i in range(len(input_data))]  
random.shuffle(index) 
data = input_data[index]
label = label_data[index]


# In[32]:


splitpoint1 = int(round(num * 0.8))
splitpoint2 = int(round(num * 0.9))
(X_train, X_val, X_test) = (data[0:splitpoint1], data[splitpoint1:splitpoint2], data[splitpoint2:])
(Y_train, Y_val, X_test) = (label[0:splitpoint1], label[splitpoint1:splitpoint2], data[splitpoint2:])


# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train,
                batch_size=16, nb_epoch=10,
                verbose=1,
                validation_data=(X_val, Y_val),
                callbacks=[history])

# In[ ]:
c = 3
print(c)
#%