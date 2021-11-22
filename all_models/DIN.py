#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

data_days=100
random_seed=666
model_name="DIN"
train_epoch=30
reg_c=0.04


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


from deepctr.models import DIN, BST, DIEN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names

def get_list_with_length(now,ls,length):
    res=[]
    for i in range(len(ls)):
        if ls[i]>=now:
            for j in range(max(0,i-length),i):
                res.append(ls[j][1]-8)
            break
    seq_len=len(res)
    while len(res)<length:
        res.append(0)
    return res,seq_len

def get_call_list(call_dict,length):
    behavior_list=dict()
    for user_name,date,clock in call_dict:
        if user_name not in behavior_list:
            behavior_list[user_name]=[]
        if call_dict[user_name,date,clock]==1:
            behavior_list[user_name].append((date,clock))
    for user_name in behavior_list:
        behavior_list[user_name].sort()
    seq_val,seq_len=dict(),dict()
    for user_name,date,clock in call_dict:
        seq_val[(user_name,date,clock)],seq_len[(user_name,date,clock)]=get_list_with_length((date,clock),behavior_list[user_name],length)
    return seq_val,seq_len

def get_DIN_DIEN_data(call_data,user_embedding_list,label_sheet_list,date_everyday):
    call_dict=get_call_dict(call_data)
    seq_val,seq_len=get_call_list(call_dict,5)
    feature_name_dict={i:("x%04d" % j) for i,j in zip(user_embedding_list[0].columns.tolist(),range(len(user_embedding_list[0].columns.tolist())))}
    x={feature_name_dict[i]:[] for i in user_embedding_list[0].columns.tolist()}
    x["item_id"]=[]
    x["seq_length"]=[]
    x["hist_item_id"]=[]
    y=[]
    feature_columns=[DenseFeat(feature_name_dict[i], 1) for i in user_embedding_list[0].columns.tolist()]
    feature_columns.append(SparseFeat('item_id', 13 + 1, embedding_dim=8))
    feature_columns.append(VarLenSparseFeat(SparseFeat(
        'hist_item_id', vocabulary_size=13 + 1,embedding_dim=8, embedding_name='item_id'),maxlen=5, length_name="seq_length"))
    for user_embedding,label_sheet,today in tqdm(zip(user_embedding_list,label_sheet_list,date_everyday)):
        for clock in range(9,22):
            select=[((user_name,today,clock) in call_dict) for user_name in label_sheet["EM_USERNAME"]]
            for feature_name in user_embedding.columns.tolist():
                x[feature_name_dict[feature_name]]+=[i for i,j in zip(user_embedding[feature_name],select) if j]
            x["item_id"]+=[clock-8 for i in select if i]
            x["seq_length"]+=[seq_len[(user_name,today,clock)] for user_name,j in zip(label_sheet["EM_USERNAME"],select) if j]
            x["hist_item_id"]+=[seq_val[(user_name,today,clock)] for user_name,j in zip(label_sheet["EM_USERNAME"],select) if j]
            y+=[call_dict[(user_name,today,clock)] for user_name,j in zip(label_sheet["EM_USERNAME"],select) if j]
    behavior_feature_list=["item_id"]
    for i in x:
        x[i]=np.array(x[i])
    y=np.array(y)
    return x,y,feature_columns,behavior_feature_list

def get_call_list_predict(call_dict,length):
    behavior_list=dict()
    for user_name,date,clock in call_dict:
        if user_name not in behavior_list:
            behavior_list[user_name]=[]
        if call_dict[user_name,date,clock]==1:
            behavior_list[user_name].append((date,clock))
    for user_name in behavior_list:
        behavior_list[user_name].sort()
    seq_val,seq_len=dict(),dict()
    for user_name,date,c in call_dict:
        for clock in range(9,22):
            seq_val[(user_name,date,clock)],seq_len[(user_name,date,clock)]=get_list_with_length((date,clock),behavior_list[user_name],length)
    return seq_val,seq_len

def get_DIN_DIEN_data_predict(call_data,user_embedding_list,label_sheet_list,date_everyday):
    call_dict=get_call_dict(call_data)
    seq_val,seq_len=get_call_list_predict(call_dict,5)
    feature_name_dict={i:("x%04d" % j) for i,j in zip(user_embedding_list[0].columns.tolist(),range(len(user_embedding_list[0].columns.tolist())))}
    x={feature_name_dict[i]:[] for i in user_embedding_list[0].columns.tolist()}
    x["item_id"]=[]
    x["seq_length"]=[]
    x["hist_item_id"]=[]
    feature_columns=[DenseFeat(i, 1) for i in user_embedding_list[0].columns.tolist()]
    feature_columns.append(SparseFeat('item_id', 13 + 1, embedding_dim=8))
    feature_columns.append(VarLenSparseFeat(SparseFeat(
        'hist_item_id', vocabulary_size=13 + 1,embedding_dim=8, embedding_name='item_id'),maxlen=5, length_name="seq_length"))
    label_dict=[]
    for user_embedding,label_sheet,today in tqdm(zip(user_embedding_list,label_sheet_list,date_everyday)):
        for clock in range(9,22):
            select=[True for user_name in label_sheet["EM_USERNAME"]]
            for feature_name in user_embedding.columns.tolist():
                x[feature_name_dict[feature_name]]+=[i for i,j in zip(user_embedding[feature_name],select) if j]
            x["item_id"]+=[clock-8 for i in select if i]
            x["seq_length"]+=[seq_len[(user_name,today,clock)] for user_name,j in zip(label_sheet["EM_USERNAME"],select) if j]
            x["hist_item_id"]+=[seq_val[(user_name,today,clock)] for user_name,j in zip(label_sheet["EM_USERNAME"],select) if j]
            label_dict+=[(user_name,today,clock) for user_name,j in zip(label_sheet["EM_USERNAME"],select) if j]
    behavior_feature_list=["item_id"]
    for i in x:
        x[i]=np.array(x[i])
    return x,feature_columns,behavior_feature_list,label_dict


# In[4]:


def save_to_binary_file(data,name):
    data_path="/home/zhongjie/RL_test/DIN_DIEN_data"
    with open("%s/%s.pkl" % (data_path,name),"wb") as f:
        pickle.dump(data,f)

def load_from_binary_file(name):
    data_path="/home/zhongjie/RL_test/DIN_DIEN_data"
    with open("%s/%s.pkl" % (data_path,name),"rb") as f:
        data=pickle.load(f)
    return data

# x,y,feature_columns,behavior_feature_list=get_DIN_DIEN_data(call_data,user_embedding_list,label_sheet_list,date_everyday)
# save_to_binary_file(label_sheet_list,"label_sheet_list")
# save_to_binary_file(label_sheet_list_,"label_sheet_list_")
# save_to_binary_file(date_everyday,"date_everyday")
# save_to_binary_file(date_everyday_,"date_everyday_")
# save_to_binary_file(user_embedding_list,"user_embedding_list")
# save_to_binary_file(user_embedding_list_,"user_embedding_list_")
# save_to_binary_file(call_data,"call_data")
# save_to_binary_file(x,"x")
# save_to_binary_file(y,"y")
# save_to_binary_file(feature_columns,"feature_columns")
# save_to_binary_file(behavior_feature_list,"behavior_feature_list")

label_sheet_list=load_from_binary_file("label_sheet_list")
label_sheet_list_=load_from_binary_file("label_sheet_list_")
date_everyday=load_from_binary_file("date_everyday")
date_everyday_=load_from_binary_file("date_everyday_")
user_embedding_list=load_from_binary_file("user_embedding_list")
user_embedding_list_=load_from_binary_file("user_embedding_list_")
call_data=load_from_binary_file("call_data")
x=load_from_binary_file("x")
y=load_from_binary_file("y")
feature_columns=load_from_binary_file("feature_columns")
behavior_feature_list=load_from_binary_file("behavior_feature_list")


# In[5]:


#x,y,feature_columns,behavior_feature_list=get_DIN_DIEN_data(call_data,user_embedding_list,label_sheet_list,date_everyday)
if model_name=="DIN":
    model = DIN(feature_columns, behavior_feature_list, l2_reg_dnn=reg_c,dnn_hidden_units=[10], att_hidden_size=[10], att_weight_normalization=True)
    model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy'])
    model.fit(x, y, verbose=2, epochs=train_epoch)
elif model_name=="DIEN":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    model = DIEN(feature_columns, behavior_feature_list, gru_type="AUGRU")
    model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy'])
    model.fit(x, y, verbose=2, epochs=train_epoch)


# In[ ]:


def test_DIN_DIEN_model(model,call_data,user_embedding_list,label_sheet_list,date_everyday):
    call_dict=get_call_dict(call_data)
    x,feature_columns,behavior_feature_list,label_dict=get_DIN_DIEN_data_predict(
        call_data,user_embedding_list,label_sheet_list,date_everyday)
    y_pred=model.predict(x)
    prob={}
    for pred,tp in zip(y_pred,label_dict):
        user_name,date,clock=tp
        if (user_name,date) not in prob:
            prob[(user_name,date)]=[0]*13
        prob[(user_name,date)][clock-9]=pred[0]
    call_num,call_ac,num=0,0,[0]*13
    for user_name,date in prob:
        clock=prob[(user_name,date)].index(max(prob[(user_name,date)]))+9
        num[clock-9]+=1
        if (user_name,date,clock) in call_dict:
            call_num+=1
            if call_dict[(user_name,date,clock)]==1:
                call_ac+=1
    print("result:")
    print(call_ac,call_num,call_ac/call_num)
    print(num)

test_DIN_DIEN_model(model,call_data,user_embedding_list_,label_sheet_list_,date_everyday_)

