#!/usr/bin/env python
# coding: utf-8

# In[10]:


import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tqdm import tqdm
import datetime
import math
import random
import os
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import xgboost as xgb
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

use_balance=False
use_feature_extractor=True
feature_num=128
regularize_alpha=1.0
data_days=100
random_seed=666


# In[2]:


def reset_seed():
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

def debug(x,log=""):
    print(log,type(x),x)
    
def safe_divide(u,v):
    if v==0:
        return 0
    return u/v
    
def log_data(data):
    result=data.copy()
    for i in list(result.columns):
        if "次数" in i:
            result[i]=result[i].map(lambda x:math.log(x+1))
    return result

class MinMaxNormalizer:
    def fit(self,data):
        self.columns=list(data.columns)
        self.min_value={column:data[column].min() for column in self.columns}
        self.max_value={column:data[column].max() for column in self.columns}
    def normalize(self,data):
        result=data.copy()
        for column in list(result.columns):
            mi,ma=self.min_value[column],self.max_value[column]
            d=ma-mi
            if mi==0 and ma==1:
                continue
            elif mi==ma:
                result[column]=result[column].map(lambda i:0)
            else:
                result[column]=result[column].map(lambda i:(i-mi)/d)
        return result
    
def read_data(data_path,date_everyday):
    user_embedding_list,label_sheet_list=[],[]
    for date in tqdm(date_everyday):
        user_embedding=pd.read_csv("%suser_embedding_%s.csv" % (data_path,date))
        label_sheet=pd.read_csv("%slabel_sheet_%s.csv" % (data_path,date))
        user_embedding_list.append(user_embedding)
        label_sheet_list.append(label_sheet)
    return user_embedding_list,label_sheet_list

def get_call_dict(call_data):
    d=dict()
    for user_name,date,clock,time in zip(call_data["EM_USERNAME"],call_data["呼叫日期"],call_data["呼叫时间"],call_data["时长"]):
        clock=int(clock.split(":")[0])
        if clock<9 or clock>21:
            continue
        if (user_name,date,clock) not in d:
            d[(user_name,date,clock)]=0
        if time>=45:
            d[(user_name,date,clock)]=1
    return d

def precut(user_embedding_list,label_sheet_list,date_everyday,call_data):
    call_dict=get_call_dict(call_data)
    sheet_select=[]
    for i in tqdm(range(len(user_embedding_list))):
        select=[]
        for user_name,date in zip(label_sheet_list[i]["EM_USERNAME"],label_sheet_list[i]["日期"]):
            flag=False
            for clock in range(9,22):
                if (user_name,date,clock) in call_dict:
                    flag=True
            select.append(flag)
        user_embedding_list[i]=user_embedding_list[i].loc[select]
        user_embedding_list[i].index=list(range(user_embedding_list[i].shape[0]))
        label_sheet_list[i]=label_sheet_list[i].loc[select]
        label_sheet_list[i].index=list(range(label_sheet_list[i].shape[0]))
        sheet_select.append(user_embedding_list[i].shape[0]!=0)
    user_embedding_list=[i for i,j in zip(user_embedding_list,sheet_select) if j]
    label_sheet_list=[i for i,j in zip(label_sheet_list,sheet_select) if j]
    date_everyday=[i for i,j in zip(date_everyday,sheet_select) if j]
    return user_embedding_list,label_sheet_list,date_everyday


# In[3]:


# 重置随机种子
reset_seed()
# 原始通话数据
call_data=pd.read_csv("/home/zhongjie/rec_20201205/KD_MODEL_EM_CALL.txt",sep='\t',names=['ID','EM_USERNAME','呼叫类型','呼叫结果','呼叫日期','呼叫时间','时长','业务员ID','业务员','团队','主管','部门','INSERT_TIME','UPDATE_TIME'])
# 特征数据路径
data_path="/home/zhongjie/call_rec_test/feature/"
# 数据日期
date_everyday=[(datetime.datetime(2020,5,10)+datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(data_days)]
# 读取数据
user_embedding_list,label_sheet_list=read_data(data_path,date_everyday)
# 裁剪数据 仅保留在 call_data 中当天出现过的数据
user_embedding_list,label_sheet_list,date_everyday=precut(
    user_embedding_list,label_sheet_list,date_everyday,call_data
)
# 长尾分布数据log化
for i in tqdm(range(len(date_everyday))):
    user_embedding_list[i]=log_data(user_embedding_list[i])
# 归一化
normalizer=MinMaxNormalizer()
normalizer.fit(pd.concat(user_embedding_list))
for i in tqdm(range(len(date_everyday))):
    user_embedding_list[i]=normalizer.normalize(user_embedding_list[i])
# 数据集划分
reset_seed()
ls=list(range(data_days))
random.shuffle(ls)
user_embedding_list_=[i for i,j in zip(user_embedding_list,range(len(user_embedding_list))) if ls.index(j)>int(data_days*0.7)]
label_sheet_list_=[i for i,j in zip(label_sheet_list,range(len(label_sheet_list))) if ls.index(j)>int(data_days*0.7)]
date_everyday_=[i for i,j in zip(date_everyday,range(len(date_everyday))) if ls.index(j)>int(data_days*0.7)]
user_embedding_list=[i for i,j in zip(user_embedding_list,range(len(user_embedding_list))) if ls.index(j)<=int(data_days*0.7)]
label_sheet_list=[i for i,j in zip(label_sheet_list,range(len(label_sheet_list))) if ls.index(j)<=int(data_days*0.7)]
date_everyday=[i for i,j in zip(date_everyday,range(len(date_everyday))) if ls.index(j)<=int(data_days*0.7)]
print("train:",date_everyday)
print("test:",date_everyday_)


# In[4]:


def get_xgb_data_day(user_embedding,label_sheet,call_dict):
    x_data=[]
    y_data=[]
    for clock in range(9,22):
        select=[(user_name,date,clock) in call_dict for user_name,date in zip(label_sheet["EM_USERNAME"],label_sheet["日期"])]
        x_data.append(user_embedding.loc[select].values)
        y_data.append([float(call_dict[(user_name,date,clock)]) for user_name,date in zip(label_sheet["EM_USERNAME"],label_sheet["日期"])                       if (user_name,date,clock) in call_dict])
    return x_data,y_data

def get_xgb_data(user_embedding_list,label_sheet_list,call_data):
    call_dict=get_call_dict(call_data)
    x_data=[[] for i in range(13)]
    y_data=[[] for i in range(13)]
    for user_embedding,label_sheet in tqdm(zip(user_embedding_list,label_sheet_list)):
        x_data_day,y_data_day=get_xgb_data_day(user_embedding,label_sheet,call_dict)
        for i in range(13):
            x_data[i].append(x_data_day[i])
            y_data[i].append(y_data_day[i])
    for i in range(13):
        x_data[i]=np.concatenate(x_data[i],axis=0)
        y_data[i]=np.concatenate(y_data[i],axis=0)
    return x_data,y_data

class XgbModel:
    def __init__(self,x_data,y_data):
#         params = {
#             'booster': 'gbtree',
#             'objective': 'multi:softprob',
#             'num_class': 2,
#             'gamma': 0.2,
#             'max_depth': 2,
#             'subsample': 0.8,
#             'colsample_bytree': 0.8,
#             'min_child_weight': 1,
#             'eta': 0.05,
#             'seed': 666,
#             'nthread': 20,
#             'eval_metric': 'merror'
#         }
        params = {
            'objective': 'multi:softprob',
            'num_class': 2,
            'max_depth': 2,
            'gamma': 30.0,
            'seed': 666,
            'nthread': 30,
            'eval_metric': 'merror'
        }
        plst = list(params.items())
        self.model=[]
        for i in tqdm(range(13)):
            dtrain = xgb.DMatrix(x_data[i],y_data[i])
            self.model.append(xgb.train(plst, dtrain, 200))
    def predict(self,x_data):
        y_pred=[]
        for i in range(13):
            y_pred.append(self.model[i].predict(xgb.DMatrix(x_data))[:,1])
        y_pred=np.column_stack(y_pred)
        return y_pred

def get_xgb_prob():
    x_data,y_data=get_xgb_data(user_embedding_list,label_sheet_list,call_data)
    xgb_model=XgbModel(x_data,y_data)
    y_pred=xgb_model.predict(user_embedding_list[0].values)
    classifier_prob_list=[xgb_model.predict(user_embedding.values) for user_embedding in user_embedding_list]
    classifier_prob_list_=[xgb_model.predict(user_embedding.values) for user_embedding in user_embedding_list_]
    return classifier_prob_list,classifier_prob_list_

XGB_prob,XGB_prob_=get_xgb_prob()


# In[5]:


def get_classifier_data_day(user_embedding,label_sheet,call_dict):
    select=[]
    y_data=[]
    for user_name,date in zip(label_sheet["EM_USERNAME"],label_sheet["日期"]):
        y=[-1.0]*13
        flag=False
        for clock in range(9,22):
            if (user_name,date,clock) in call_dict:
                y[clock-9]=float(call_dict[(user_name,date,clock)])
                flag=True
        if flag:
            y_data.append(y)
        select.append(flag)
    return user_embedding.loc[select].values,np.array(y_data)

def get_classifier_data(user_embedding_list,label_sheet_list,call_data):
    call_dict=get_call_dict(call_data)
    x_data_list,y_data_list=[],[]
    for user_embedding,label_sheet in tqdm(zip(user_embedding_list,label_sheet_list)):
        x,y=get_classifier_data_day(user_embedding,label_sheet,call_dict)
        x_data_list.append(x)
        y_data_list.append(y)
    x=np.concatenate(x_data_list,axis=0)
    y=np.concatenate(y_data_list,axis=0)
    return x,y

def create_DNN_model():
    classifier = keras.Sequential(
        [
            keras.layers.Dense(979),
            keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.5)),
            keras.layers.Dense(13, activation=tf.nn.sigmoid),
        ]
    )
    return classifier

def mask_binary_crossentropy(y_true, y_pred):
    mask=K.minimum(y_true,0.)+1.
    y_pred_fix=K.clip(y_pred,K.epsilon(),1.-K.epsilon())
    loss=-y_true*K.log(y_pred_fix)-(1.-y_true)*K.log(1.-y_pred_fix)
    return K.mean(mask*loss, axis=-1)

def create_LR_model():
    classifier = keras.Sequential(
        [
            keras.layers.Dense(979),
            keras.layers.Dense(13, activation=tf.nn.sigmoid),
        ]
    )
    return classifier


def get_LR_prob():
    train_x,train_y=get_classifier_data(user_embedding_list,label_sheet_list,call_data)
    reset_seed()
    classifier=create_LR_model()
    classifier.compile(optimizer='Adam',loss=mask_binary_crossentropy,metrics=[])
    classifier.fit(train_x,train_y,batch_size=32,epochs=30,verbose=2)
    classifier_prob_list=[classifier(user_embedding.values) for user_embedding in user_embedding_list]
    classifier_prob_list_=[classifier(user_embedding.values) for user_embedding in user_embedding_list_]
    return classifier_prob_list,classifier_prob_list_

def get_DNN_prob():
    train_x,train_y=get_classifier_data(user_embedding_list,label_sheet_list,call_data)
    reset_seed()
    classifier=create_DNN_model()
    classifier.compile(optimizer='Adam',loss=mask_binary_crossentropy,metrics=[])
    classifier.fit(train_x,train_y,batch_size=32,epochs=30,verbose=2)
    classifier_prob_list=[classifier(user_embedding.values) for user_embedding in user_embedding_list]
    classifier_prob_list_=[classifier(user_embedding.values) for user_embedding in user_embedding_list_]
    return classifier_prob_list,classifier_prob_list_

LR_prob,LR_prob_=get_LR_prob()
DNN_prob,DNN_prob_=get_DNN_prob()


# In[6]:


def get_classifier_info(classifier_prob_list,label_sheet_list,date_everyday,call_data,need_distribute_info=False):
    info=dict()
    call_dict=get_call_dict(call_data)
    for classifier_prob,label_sheet,date in zip(classifier_prob_list,label_sheet_list,date_everyday):
        clocks=tf.argmax(classifier_prob,1).numpy()+9
        ac,miss=0,0
        for user_name,clock in zip(label_sheet["EM_USERNAME"],clocks):
            if (user_name,date,clock) in call_dict:
                if call_dict[(user_name,date,clock)]==1:
                    ac+=1
                else:
                    miss+=1
        info[date]={"ac":ac,"miss":miss,"rate":safe_divide(ac,ac+miss)}
    if need_distribute_info:
        distribute_info=dict()
        for classifier_prob,date in zip(classifier_prob_list,date_everyday):
            d=[0]*13
            clocks=tf.argmax(classifier_prob,1).numpy()+9
            for i in clocks:
                d[i-9]+=1
            distribute_info[date]=d
        return info,distribute_info
    else:
        return info
    
def get_real_info(date_everyday,call_data,need_distribute_info=False):
    call_dict=get_call_dict(call_data)
    info={date:{"ac":0,"miss":0} for date in date_everyday}
    for user_name,date,clock in call_dict:
        if date in info:
            if call_dict[(user_name,date,clock)]==1:
                info[date]["ac"]+=1
            else:
                info[date]["miss"]+=1
    for date in date_everyday:
        info[date]["rate"]=safe_divide(info[date]["ac"],info[date]["ac"]+info[date]["miss"])
    if need_distribute_info:
        distribute_info={date:[0]*13 for date in date_everyday}
        for user_name,date,clock in call_dict:
            if date in distribute_info:
                distribute_info[date][clock-9]+=1
        return info,distribute_info
    else:
        return info
        

def test_classifier_model(classifier_prob_list,label_sheet_list,date_everyday,call_data):
    info,distribute_info=get_classifier_info(classifier_prob_list,label_sheet_list,date_everyday,call_data,need_distribute_info=True)
    ac=sum([info[i]["ac"] for i in date_everyday])
    miss=sum([info[i]["miss"] for i in date_everyday])
    num=[0]*13
    for i in date_everyday:
        for clock in range(9,22):
            num[clock-9]+=distribute_info[i][clock-9]
    print("",ac,miss,safe_divide(ac,ac+miss))
    print("",num)
    
def test_real_model(date_everyday,call_data):
    info,distribute_info=get_real_info(date_everyday,call_data,need_distribute_info=True)
    ac=sum([info[i]["ac"] for i in date_everyday])
    miss=sum([info[i]["miss"] for i in date_everyday])
    num=[0]*13
    for i in date_everyday:
        for clock in range(9,22):
            num[clock-9]+=distribute_info[i][clock-9]
    print("",ac,miss,safe_divide(ac,ac+miss))
    print("",num)


print("real")
test_real_model(date_everyday,call_data)
test_real_model(date_everyday_,call_data)
real_info=get_real_info(date_everyday,call_data)
temp=get_real_info(date_everyday_,call_data)
for i in temp:
    real_info[i]=temp[i]
    
print("LR")
test_classifier_model(LR_prob,label_sheet_list,date_everyday,call_data)
test_classifier_model(LR_prob_,label_sheet_list_,date_everyday_,call_data)
LR_info=get_classifier_info(LR_prob,label_sheet_list,date_everyday,call_data)
temp=get_classifier_info(LR_prob_,label_sheet_list_,date_everyday_,call_data)
for i in temp:
    LR_info[i]=temp[i]
    
print("DNN")
test_classifier_model(DNN_prob,label_sheet_list,date_everyday,call_data)
test_classifier_model(DNN_prob_,label_sheet_list_,date_everyday_,call_data)
DNN_info=get_classifier_info(DNN_prob,label_sheet_list,date_everyday,call_data)
temp=get_classifier_info(DNN_prob_,label_sheet_list_,date_everyday_,call_data)
for i in temp:
    DNN_info[i]=temp[i]
    
print("XGB")
test_classifier_model(XGB_prob,label_sheet_list,date_everyday,call_data)
test_classifier_model(XGB_prob_,label_sheet_list_,date_everyday_,call_data)
XGB_info=get_classifier_info(XGB_prob,label_sheet_list,date_everyday,call_data)
temp=get_classifier_info(XGB_prob_,label_sheet_list_,date_everyday_,call_data)
for i in temp:
    XGB_info[i]=temp[i]


# In[7]:


class GameEnvironment:
    def __init__(self,call_data,use_balance):
        self.call_dict=get_call_dict(call_data)
        self.real_ac,self.real_miss=dict(),dict()
        for (user_name,date,clock) in self.call_dict:
            self.real_ac[date]=0
            self.real_miss[date]=0
        for (user_name,date,clock) in self.call_dict:
            if self.call_dict[(user_name,date,clock)]==1:
                self.real_ac[date]+=1
            else:
                self.real_miss[date]+=1
        self.use_balance=use_balance
    def get_state(self):
        sum_call_num=max(sum(self.call_num),1)
        call_p=[i/sum_call_num for i in self.call_num]
        call_p.append((self.num-self.row)/5000)
        if self.row>=self.user_embedding.shape[0]:
            user=np.zeros((self.user_embedding.shape[1]))
        else:
            user=self.user_embedding[self.row]
        self.row+=1
        return np.hstack((user,call_p))
    def reset(self,user_embedding,label_sheet,today):
        self.user_embedding=user_embedding
        self.user_list=[i for i in label_sheet["EM_USERNAME"]]
        self.today=today
        self.row=0
        self.call_num=[0]*13
        self.ac=0
        self.miss=0
        self.num=self.user_embedding.shape[0]
        return self.get_state()
    def step(self,action):
        state=self.get_state()
        self.call_num[action]+=1
        reward=0
        clock=action+9
        user_name=self.user_list[self.row-2]
        if (user_name,self.today,clock) not in self.call_dict:
            reward=0
        elif self.call_dict[(user_name,self.today,clock)]==1:
            reward=1
            self.ac+=1
        else:
            reward=-1
            self.miss+=1
        if self.use_balance:
            reward+=1/13-(2*self.call_num[action]-1)/self.num
        done=(self.row>self.user_embedding.shape[0])
        return state,reward,done,{}
    def info(self):
        RL_info={"ac":self.ac,"miss":self.miss,"rate":safe_divide(self.ac,self.ac+self.miss)}
        return RL_info

env = GameEnvironment(call_data,use_balance)


# In[8]:


def create_classifier_unit():
    classifier = keras.Sequential(
        [
            keras.layers.Dense(979),
            keras.layers.Dense(feature_num, activation=tf.nn.sigmoid, kernel_regularizer=keras.regularizers.l2(regularize_alpha)),
            keras.layers.Dense(13, activation=tf.nn.sigmoid),
        ]
    )
    return classifier

if use_feature_extractor:
    print("training classifier unit")
    classifier_unit=create_classifier_unit()
    train_x,train_y=get_classifier_data(user_embedding_list,label_sheet_list,call_data)
    classifier_unit.compile(optimizer='Adam',loss=mask_binary_crossentropy,metrics=[])
    classifier_unit.fit(train_x,train_y,batch_size=32,epochs=30,verbose=2)
    feature_extractor=K.function(classifier_unit.layers[0].input,classifier_unit.layers[1].output)
    user_feature=[feature_extractor(user_embedding.values) for user_embedding in user_embedding_list]
    user_feature_=[feature_extractor(user_embedding.values) for user_embedding in user_embedding_list_]
else:
    user_feature=[user_embedding.values for user_embedding in user_embedding_list]
    user_feature_=[user_embedding.values for user_embedding in user_embedding_list_]

for i in range(len(user_feature)):
    user_feature[i]=np.concatenate((user_feature[i],LR_prob[i],DNN_prob[i],XGB_prob[i]),axis=1)
for i in range(len(user_feature_)):
    user_feature_[i]=np.concatenate((user_feature_[i],LR_prob_[i],DNN_prob_[i],XGB_prob_[i]),axis=1)


# In[25]:


def save_to_binary_file(data,name):
    data_path="/home/zhongjie/RL_test/data"
    with open("%s/%s.pkl" % (data_path,name),"wb") as f:
        pickle.dump(data,f)

def load_from_binary_file(name):
    data_path="/home/zhongjie/RL_test/data"
    with open("%s/%s.pkl" % (data_path,name),"rb") as f:
        data=pickle.load(f)
    return data

temp=[i[:,0:128] for i in user_feature]
save_to_binary_file(temp,"user_feature")
temp=[i[:,0:128] for i in user_feature_]
save_to_binary_file(temp,"user_feature_")
save_to_binary_file(label_sheet_list,"label_sheet_list")
save_to_binary_file(label_sheet_list_,"label_sheet_list_")
save_to_binary_file(date_everyday,"date_everyday")
save_to_binary_file(date_everyday_,"date_everyday_")
save_to_binary_file(call_data,"call_data")
save_to_binary_file(LR_prob,"LR_prob")
save_to_binary_file(LR_prob_,"LR_prob_")
save_to_binary_file(DNN_prob,"DNN_prob")
save_to_binary_file(DNN_prob_,"DNN_prob_")
save_to_binary_file(XGB_prob,"XGB_prob")
save_to_binary_file(XGB_prob_,"XGB_prob_")


# In[ ]:




