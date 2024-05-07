

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from human_lifelog_mil_pytorch.FocalLoss import FocalLoss
from human_lifelog_mil_pytorch.human_lifelog_predictor import LifeLogNet
from human_lifelog_mil_pytorch.human_lifelog_predictor import HumanDataset #, HumanDataset_wr
from human_lifelog_mil_pytorch.utils import getF1Score

import torch
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings( 'ignore' )

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

device = torch.device("cuda:0")

# user list and file name 
group0_users = list(map(lambda f: f"tmp_merged_{f}.csv", ['101','102','103','104','105','106','107','108','109', '110','111','112','113','114','115', '116','117','118', '119', '120']))


best_loss_path = '/workspace/ML_instance/models/max_group0_users.p' # 모델 저장 경로 
# 1: ['112', '117'],
df = pd.concat([
    pd.read_csv('!(path for data )/'+str(i)) 
    for i in group0_users
], 0)
#df.sort_values(by=['datetime'], inplace=True)

print(df.shape)
#import sys 
#sys.exit()

col_means = df.mean()

# fill missing values with the mean of the corresponding column

# 기본 전처리 

df = df.fillna(col_means)
df['e4Acc'] = df.apply(lambda x: [x['e4Acc__x'], x['e4Acc__y'], x['e4Acc__z']], axis=1)
df['mAcc'] = df.apply(lambda x: [x['mAcc__x'], x['mAcc__y'], x['mAcc__z']], axis=1)

df['mGyr'] = df.apply(lambda x: [x['mGyr__x'], x['mGyr__y'], x['mGyr__z']], axis=1)
df['mGps'] = df.apply(lambda x: [x['mGps__lat'], x['mGps__lon']], axis=1) #j, x['mGps__accuracy']
df['mMag'] = df.apply(lambda x: [x['mMag__x'], x['mMag__y'], x['mMag__z']], axis=1)
df['mAcc'] = df.apply(lambda x: [x['mAcc__x'], x['mAcc__y'], x['mAcc__z']], axis=1)
#라벨 전처리 
df['emotionTension'] = df['emotionTension'].apply(lambda x: 0 if x in [0, 1] else x if 1 in [2, 3, 4] else 2)



action_encoder = LabelEncoder()
df['action_label'] = action_encoder.fit_transform(df['action'])
encoder = LabelEncoder()
df['positive_label'] = encoder.fit_transform(df['emotionPositive'])
encoder = LabelEncoder()
df['tension_label'] = encoder.fit_transform(df['emotionTension'])

train_df, test_df = train_test_split(df, test_size=0.2, shuffle = True, random_state=32)
train_df, valid_df = train_test_split(train_df, test_size=0.125, shuffle = True, random_state=32)

print(len(train_df), len(valid_df), len(test_df))

trainset = HumanDataset(train_df)
validset = HumanDataset(valid_df)
testset = HumanDataset(test_df)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=52400, num_workers=4, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=20480, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=20480, num_workers=4)

actionNet = LifeLogNet(class_num = len(df['tension_label'].unique())).to(device)
optimizer = torch.optim.AdamW(actionNet.parameters(), 1e-2)
# loss_fn = torch.nn.NLLLoss()
loss_fn = FocalLoss()



best_f1 = 0

train_f1, train_acc, train_auc = [], [], []
valid_f1, valid_acc, valid_auc = [], [], []

for e in range(int(1000)):
    train_output = []
    train_label = []

    valid_output = []
    valid_label = []
    
    actionNet.train()
    for batch_id, (e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag,emotionPositive ,emotionTension ,action) in enumerate(train_dataloader):

        e4Acc = e4Acc.to(device)
        e4Bvp = e4Bvp.to(device)
        e4Eda = e4Eda.to(device)
        e4Hr = e4Hr.float().to(device)
        e4Temp = e4Temp.to(device)
        mAcc = mAcc.to(device)
        mGps = mGps.float().to(device)
        mGyr = mGyr.to(device)
        mMag = mMag.to(device)
        emotionPositive = emotionPositive.to(device)
        label = emotionTension.long().to(device)
        
        optimizer.zero_grad()

        out = actionNet( e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag)
        #print(out)
        #print(label)
        #print(out.shape)
        #print(label.shape)
        loss = loss_fn(out, label)
        print(f'======\t\tloss: {loss}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actionNet.parameters(), 0.9)
        optimizer.step()

        temp_out = out.detach().cpu().numpy()
        temp_label = label.detach().cpu().numpy()
        train_output += list(temp_out)
        train_label += list(temp_label)
            

    if (e+1) % 50 == 0 or e==0:
        print(f'-----------------------------------------------{e+1} Train END--------------------------------------------')
        print(f'F1 score: {getF1Score(np.array(train_label), np.argmax(train_output, axis=1))}')
        print(f'ACC score: {accuracy_score(np.array(train_label), np.argmax(train_output, axis=1))}')
        #print(f'======\t\tloss: {loss}\t\t======')
    train_f1.append(getF1Score(np.array(train_label), np.argmax(train_output, axis=1)))
    train_acc.append(accuracy_score(np.array(train_label), np.argmax(train_output, axis=1)))
            
    actionNet.eval()
    for batch_id, (e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag,emotionPositive ,emotionTension ,action
    ) in enumerate(valid_dataloader):
        e4Acc = e4Acc.to(device)
        e4Bvp = e4Bvp.to(device)
        e4Eda = e4Eda.to(device)
        e4Hr = e4Hr.float().to(device)
        e4Temp = e4Temp.to(device)
        mAcc = mAcc.to(device)
        mGps = mGps.float().to(device)
        mGyr = mGyr.to(device)
        mMag = mMag.to(device)
        emotionPositive = emotionPositive.to(device)
        label = emotionTension.to(device)

        out = actionNet( e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag)


        temp_out = out.detach().cpu().numpy()
        temp_label = label.detach().cpu().numpy()
        valid_output += list(temp_out)
        valid_label += list(temp_label)
            
    if (e+1) % 50 == 0 or e==0:
        print(f'-----------------------------------------------{e} Validation--------------------------------------------')
        print(f'F1 score: {getF1Score(np.array(valid_label), np.argmax(valid_output, axis=1))}')
        print(f'ACC score: {accuracy_score(np.array(valid_label), np.argmax(valid_output, axis=1))}')

    valid_f1.append(getF1Score(np.array(valid_label), np.argmax(valid_output, axis=1)))
    valid_acc.append(accuracy_score(np.array(valid_label), np.argmax(valid_output, axis=1)))

    if best_f1 <= getF1Score(np.array(valid_label), np.argmax(valid_output, axis=1)):
        best_f1 = getF1Score(np.array(valid_label), np.argmax(valid_output, axis=1))
        torch.save(actionNet.state_dict(), best_loss_path)

actionNet.load_state_dict(torch.load(best_loss_path))