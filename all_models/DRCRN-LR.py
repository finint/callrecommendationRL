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

# 是否使用诱导奖励，诱导模型均匀分配
use_balance=False
# 随机种子 便于复现
random_seed=666
# 诱导奖励系数
balance_alpha=800
# 若使用诱导奖励，对奖励归一化
normalize_reward=use_balance
# 分类器模块
classifier_module_list=["nothing","LR","DNN","XGB","boost"]
classifier_module=classifier_module_list[1]


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

def get_call_num(calL_data):
    num=[0]*13
    for clock in call_data["呼叫时间"]:
        clock=int(clock.split(":")[0])-9
        if clock <0 or clock>=13:
            continue
        num[clock]+=1
    return num
    
def save_to_binary_file(data,name):
    data_path="/home/zhongjie/RL_test/data"
    with open("%s/%s.pkl" % (data_path,name),"wb") as f:
        pickle.dump(data,f)

def load_from_binary_file(name):
    data_path="/home/zhongjie/RL_test/data"
    with open("%s/%s.pkl" % (data_path,name),"rb") as f:
        data=pickle.load(f)
    return data

user_feature=load_from_binary_file("user_feature")
user_feature_=load_from_binary_file("user_feature_")
label_sheet_list=load_from_binary_file("label_sheet_list")
label_sheet_list_=load_from_binary_file("label_sheet_list_")
date_everyday=load_from_binary_file("date_everyday")
date_everyday_=load_from_binary_file("date_everyday_")
call_data=load_from_binary_file("call_data")
LR_prob=load_from_binary_file("LR_prob")
LR_prob_=load_from_binary_file("LR_prob_")
DNN_prob=load_from_binary_file("DNN_prob")
DNN_prob_=load_from_binary_file("DNN_prob_")
XGB_prob=load_from_binary_file("XGB_prob")
XGB_prob_=load_from_binary_file("XGB_prob_")

if classifier_module=="LR":
    for i in range(len(user_feature)):
        user_feature[i]=np.concatenate((user_feature[i],LR_prob[i]),axis=1)
    for i in range(len(user_feature_)):
        user_feature_[i]=np.concatenate((user_feature_[i],LR_prob_[i]),axis=1)
if classifier_module=="DNN":
    for i in range(len(user_feature)):
        user_feature[i]=np.concatenate((user_feature[i],DNN_prob[i]),axis=1)
    for i in range(len(user_feature_)):
        user_feature_[i]=np.concatenate((user_feature_[i],DNN_prob_[i]),axis=1)
if classifier_module=="XGB":
    for i in range(len(user_feature)):
        user_feature[i]=np.concatenate((user_feature[i],XGB_prob[i]),axis=1)
    for i in range(len(user_feature_)):
        user_feature_[i]=np.concatenate((user_feature_[i],XGB_prob_[i]),axis=1)
if classifier_module=="boost":
    for i in range(len(user_feature)):
        user_feature[i]=np.concatenate((user_feature[i],LR_prob[i],DNN_prob[i],XGB_prob[i]),axis=1)
    for i in range(len(user_feature_)):
        user_feature_[i]=np.concatenate((user_feature_[i],LR_prob_[i],DNN_prob_[i],XGB_prob_[i]),axis=1)


# In[3]:


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


# In[4]:


class GameEnvironment:
    def __init__(self,call_data,use_balance):
        self.call_dict=get_call_dict(call_data)
        self.p=get_call_num(call_data)
        self.use_balance=use_balance
    def get_state(self):
        sum_call_num=max(sum(self.call_num),1)
        call_p=[i/sum_call_num for i in self.call_num]
        call_p+=[i/5000 for i in self.call_num]
        call_p.append((self.num-self.row)/self.num)
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
            reward=miss_reward
        elif self.call_dict[(user_name,self.today,clock)]==1:
            reward=2
            self.ac+=1
        else:
            reward=0
            self.miss+=1
        if self.use_balance:
            reward+=-balance_alpha*(2*self.call_num[action]-1)/(self.num*self.p[action])
        done=(self.row>self.user_embedding.shape[0])
        return state,reward,done,{}
    def info(self):
        RL_info={"ac":self.ac,"miss":self.miss,"rate":safe_divide(self.ac,self.ac+self.miss)}
        return RL_info
    
class GameEnvironment_Autoreward:
    def __init__(self,call_data,use_balance):
        self.call_dict=get_call_dict(call_data)
        self.p=get_call_num(call_data)
        self.use_balance=use_balance
        self.ac=0
        self.miss=0
        self.sum_ac=0
        self.sum_miss=0
        self.miss_reward=1
    def get_state(self):
        sum_call_num=max(sum(self.call_num),1)
        call_p=[i/sum_call_num for i in self.call_num]
        call_p+=[i/5000 for i in self.call_num]
        call_p.append((self.num-self.row)/self.num)
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
        self.sum_ac+=self.ac
        self.sum_miss+=self.miss
        if self.sum_ac+self.sum_miss!=0:
            self.miss_reward=2*self.sum_ac/(self.sum_ac+self.sum_miss)
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
            reward=self.miss_reward
        elif self.call_dict[(user_name,self.today,clock)]==1:
            reward=2
            self.ac+=1
        else:
            reward=0
            self.miss+=1
        if self.use_balance:
            reward+=-balance_alpha*(2*self.call_num[action]-1)/(self.num*self.p[action])
        done=(self.row>self.user_embedding.shape[0])
        return state,reward,done,{}
    def info(self):
        RL_info={"ac":self.ac,"miss":self.miss,"rate":safe_divide(self.ac,self.ac+self.miss)}
        return RL_info
    
# class GameEnvironment_addmax:
#     def __init__(self,call_data,use_balance):
#         self.call_dict=get_call_dict(call_data)
#         self.p=get_call_num(call_data)
#         self.use_balance=use_balance
#     def get_state(self):
#         sum_call_num=max(sum(self.call_num),1)
#         call_p=[i/sum_call_num for i in self.call_num]
#         call_p+=[i/j for i,j in zip(self.call_num,self.p)]
#         call_p.append((self.num-self.row)/self.num)
#         call_p.append((self.num-self.row)/5000)
#         if self.row>=self.user_embedding.shape[0]:
#             user=np.zeros((self.user_embedding.shape[1]))
#         else:
#             user=self.user_embedding[self.row]
#         self.row+=1
#         return np.hstack((user,call_p))
#     def reset(self,user_embedding,label_sheet,today):
#         self.user_embedding=user_embedding
#         self.user_list=[i for i in label_sheet["EM_USERNAME"]]
#         self.today=today
#         self.row=0
#         self.call_num=[0]*13
#         self.ac=0
#         self.miss=0
#         self.num=self.user_embedding.shape[0]
#         return self.get_state()
#     def step(self,action):
#         previous_max=max([i/j for i,j in zip(self.call_num,self.p)])
#         state=self.get_state()
#         self.call_num[action]+=1
#         reward=0
#         clock=action+9
#         user_name=self.user_list[self.row-2]
#         if (user_name,self.today,clock) not in self.call_dict:
#             reward=1
#         elif self.call_dict[(user_name,self.today,clock)]==1:
#             reward=2
#             self.ac+=1
#         else:
#             reward=0
#             self.miss+=1
#         if self.use_balance:
#             now_max=max([(i+1)/(j+1) for i,j in zip(self.call_num,self.p)])
#             reward+=-balance_alpha*(now_max-previous_max)
#         done=(self.row>self.user_embedding.shape[0])
#         return state,reward,done,{}
#     def info(self):
#         RL_info={"ac":self.ac,"miss":self.miss,"rate":safe_divide(self.ac,self.ac+self.miss)}
#         return RL_info
    
# class GameEnvironment_KL:
#     def __init__(self,call_data,use_balance):
#         self.call_dict=get_call_dict(call_data)
#         self.p=get_call_num(call_data)
#         self.use_balance=use_balance
#     def get_state(self):
#         sum_call_num=max(sum(self.call_num),1)
#         call_p=[i/sum_call_num for i in self.call_num]
#         call_p+=[i/j for i,j in zip(self.call_num,self.p)]
#         call_p.append((self.num-self.row)/self.num)
#         call_p.append((self.num-self.row)/5000)
#         if self.row>=self.user_embedding.shape[0]:
#             user=np.zeros((self.user_embedding.shape[1]))
#         else:
#             user=self.user_embedding[self.row]
#         self.row+=1
#         return np.hstack((user,call_p))
#     def reset(self,user_embedding,label_sheet,today):
#         self.user_embedding=user_embedding
#         self.user_list=[i for i in label_sheet["EM_USERNAME"]]
#         self.today=today
#         self.row=0
#         self.call_num=[0]*13
#         self.ac=0
#         self.miss=0
#         self.num=self.user_embedding.shape[0]
#         return self.get_state()
#     def step(self,action):
#         state=self.get_state()
#         self.call_num[action]+=1
#         reward=0
#         clock=action+9
#         user_name=self.user_list[self.row-2]
#         if (user_name,self.today,clock) not in self.call_dict:
#             reward=1
#         elif self.call_dict[(user_name,self.today,clock)]==1:
#             reward=2
#             self.ac+=1
#         else:
#             reward=0
#             self.miss+=1
#         if self.use_balance and self.row>100:
#             sum_p=sum(self.p)
#             p=[i/sum_p for i in self.p]
#             sum_q=sum(self.call_num)
#             q=[i/sum_q for i in self.call_num]
#             KL=-sum([j*math.log((i+1)/(j+1)) for i,j in zip(q,p)])
#             reward+=balance_alpha*KL
#         done=(self.row>self.user_embedding.shape[0])
#         return state,reward,done,{}
#     def info(self):
#         RL_info={"ac":self.ac,"miss":self.miss,"rate":safe_divide(self.ac,self.ac+self.miss)}
#         return RL_info

env = GameEnvironment_Autoreward(call_data,use_balance)


# In[5]:


class RecurrentQueue:
    def __init__(self):
        self.data=[]
        self.p=0
        self.max_length=10**5
    def size(self):
        return len(self.data)
    def append(self,x):
        if len(self.data)<self.max_length:
            self.data.append(x)
        else:
            self.data[self.p]=x
        self.p=(self.p+1)%self.max_length
    def choose(self,index_list):
        return [self.data[(i+self.p)%len(self.data)] for i in index_list]

def random_choice_index(n,batch_size):
    res=dict()
    while len(res)<batch_size:
        x=random.randint(0,n-1)
        res[x]=1
    return [i for i in res]

class History:
    def __init__(self):
        self.action_history=RecurrentQueue()
        self.state_history=RecurrentQueue()
        self.state_next_history=RecurrentQueue()
        self.rewards_history=RecurrentQueue()
    def size(self):
        return self.action_history.size()
    def append(self,action,state,state_next,rewards):
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.rewards_history.append(reward)
    def random_sample(self,batch_size):
        index_list = random_choice_index(self.size(),batch_size)
        return np.array(self.action_history.choose(index_list)),                np.array(self.state_history.choose(index_list)),                np.array(self.state_next_history.choose(index_list)),                np.array(self.rewards_history.choose(index_list))


# In[6]:


gamma = 0.5
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 512
max_steps_per_episode = 30000
num_actions=13

def create_RL_model(input_shape):
    inputs = layers.Input(shape=(input_shape+28,))
    hidden = layers.Dense(128, activation="relu")(inputs)
    action = layers.Dense(13, activation="softmax")(hidden)
    state = layers.Dense(1, activation="sigmoid")(hidden)
    outputs = action + state
    return keras.Model(inputs=inputs, outputs=outputs)

reset_seed()
model = create_RL_model(user_feature[0].shape[1])
model_target = create_RL_model(user_feature[0].shape[1])
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

history=History()
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
update_after_actions = 100
update_target_network = 10000
loss_function = keras.losses.MSE

def test_model(model,env,user_feature,label_sheet_list,date_everyday):
    for user_embedding,label_sheet,date in zip(
        user_feature,label_sheet_list,date_everyday
    ):
        state = env.reset(user_embedding,label_sheet,date)
        for timestep in range(max_steps_per_episode):
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            state = state_next
            if done:
                break
        print("    testing model")
        info=env.info()
        print("     date:",date)
        print("      real_info",real_info[date])
        print("      LR_info",LR_info[date])
        print("      DNN_info",DNN_info[date])
        print("      XGB_info",XGB_info[date])
        print("      RL_info",env.info())
        print("     ",env.call_num)
        

while True:
    day_index=random.randint(0,len(user_feature)-1)
    state = env.reset(
        user_feature[day_index],
        label_sheet_list[day_index],
        date_everyday[day_index]
    )
    episode_reward = 0

    while True:
        frame_count += 1
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward
        history.append(action,state,state_next,reward)
        state = state_next

        if frame_count % update_after_actions == 0 and history.size() > batch_size:
            action_sample,state_sample,state_next_sample,rewards_sample=history.random_sample(batch_size)
            future_rewards = model_target.predict(state_next_sample)
            updated_q_values = (1-gamma) * np.array(rewards_sample)                             + gamma * tf.reduce_max(future_rewards, axis=1).numpy()
            if normalize_reward:
                updated_q_values=(updated_q_values-updated_q_values.min())*2/(updated_q_values.max()-updated_q_values.min()+(1e-6))
            masks = tf.one_hot(action_sample, num_actions)
            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
#             debug(rewards_sample,"rewards_sample")
#             debug(updated_q_values,"updated_q_values")
#             debug(q_action,"q_action")
#             debug(loss,"loss")
#             for i in grads:
#                 debug(i.shape)
#             for i in range(10):
#                 debug(grads[2][i],"grads[2][0]")
#             fuck

        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())

        if done:
            break

    episode_count += 1
    info=env.info()
    print("episode:",episode_count,"date:",env.today,"reward:",episode_reward)
    print("  real_info",real_info[date_everyday[day_index]])
    print("  LR_info",LR_info[date_everyday[day_index]])
    print("  DNN_info",DNN_info[date_everyday[day_index]])
    print("  XGB_info",XGB_info[date_everyday[day_index]])
    print("  RL_info",env.info())
    print(" ",env.call_num)
    if (episode_count-1)%10==0:
        test_model(model,env,user_feature_,label_sheet_list_,date_everyday_)


# In[ ]:




