import human_lifelog_mil_pytorch.modules as modules
import torch

class HumanDataset(torch.utils.data.Dataset):
    def __init__(self, DatasetDf):
        self.e4Acc = DatasetDf['e4Acc'].to_list()
        self.e4Bvp = DatasetDf['e4Bvp'].to_list()
        self.e4Eda = DatasetDf['e4Eda'].to_list()
        self.e4Hr = DatasetDf['e4Hr'].to_list()
        self.e4Temp = DatasetDf['e4Temp'].to_list()
        
        self.mAcc = DatasetDf['mAcc'].to_list()
        self.mGps = DatasetDf['mGps'].to_list()
        self.mGyr = DatasetDf['mGyr'].to_list()
        self.mMag = DatasetDf['mMag'].to_list()

        self.emotionPositive = DatasetDf['positive_label'].to_list()
        self.emotionTension = DatasetDf['tension_label'].to_list()
        self.action = DatasetDf['action_label'].to_list()

    def __getitem__(self, i):
        return (
            torch.tensor(self.e4Acc[i]),
            torch.tensor(self.e4Bvp[i]),
            torch.tensor(self.e4Eda[i]),
            torch.tensor(self.e4Hr[i]),
            torch.tensor(self.e4Temp[i]),
            torch.tensor(self.mAcc[i]),
            torch.tensor(self.mGps[i]),
            torch.tensor(self.mGyr[i]),
            torch.tensor(self.mMag[i]),
            torch.tensor(self.emotionPositive[i]),
            torch.tensor(self.emotionTension[i]),
            torch.tensor(self.action[i]),
        )

    def __len__(self):
        return (len(self.e4Acc))

# 인스턴스 분류 및 attention을 위한 feature 생성 모델
class ILNet(torch.nn.Module):
    def __init__(self, in_channel_num, class_num):
        super(ILNet, self).__init__()
        self.check = True
        if in_channel_num == 0:
            self.check=False
            self.Encoder = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.BatchNorm1d(32),
#                 torch.nn.LeakyReLU(),
            )
        else:
            self.Encoder = modules.CausalCNNEncoder(
                in_channels = in_channel_num, 
                channels = 8, 
                depth = 2, 
                reduced_size = 16, 
                out_channels = 32,
                kernel_size = 3
            )
        
        self.Vl = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, class_num),
            torch.nn.LogSoftmax(1)
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            
    def forward(self, x):
        if self.check:
            x = x.transpose(1, 2)
        else:
            x = x.reshape(-1, 1)
        x = self.Encoder(x)
        v = self.Vl(x)

        return v, x
    
class MILNet(torch.nn.Module):
    def __init__(self, class_num):
        super(MILNet, self).__init__()
        # e4Acc Instance classifier
        self.e4AccILNet = ILNet(in_channel_num = 3, class_num = class_num)
        # e4Bvp Instance classifier
        self.e4BvpILNet = ILNet(in_channel_num = 1, class_num = class_num)
        # e4Eda Instance classifier
        self.e4EdaILNet = ILNet(in_channel_num = 1, class_num = class_num)
        # e4Hr Instance classifier
        self.e4HrILNet = ILNet(in_channel_num = 0, class_num = class_num)
        # e4Temp Instance classifier
        self.e4TempILNet = ILNet(in_channel_num = 1, class_num = class_num)
        # mAcc Instance classifier
        self.mAccILNet = ILNet(in_channel_num = 3, class_num = class_num)
        # mGps Instance classifier
        self.mGpsILNet = ILNet(in_channel_num = 0, class_num = class_num)
        # mGyr Instance classifier
        self.mGyrILNet = ILNet(in_channel_num = 3, class_num = class_num)
        # mMag Instance classifier
        self.mMagILNet = ILNet(in_channel_num = 3, class_num = class_num)
        
        self.Kl = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        
        self.Ql = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 9),
            torch.nn.Sigmoid()
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag):
        # e4Acc Instance classification
        e4AccValue, e4AccKey = self.e4AccILNet(e4Acc)
        e4AccValue, e4AccKey = (
            e4AccValue.reshape(e4Acc.shape[0], 1, -1), 
            e4AccKey.reshape(e4Acc.shape[0], 1, -1)
        )
        # e4Bvp Instance classification
        e4BvpValue, e4BvpKey = self.e4BvpILNet(e4Bvp)
        e4BvpValue, e4BvpKey = (
            e4BvpValue.reshape(e4Acc.shape[0], 1, -1), 
            e4BvpKey.reshape(e4Acc.shape[0], 1, -1)
        )
        # e4Eda Instance classification
        e4EdaValue, e4EdaKey = self.e4EdaILNet(e4Eda)
        e4EdaValue, e4EdaKey = (
            e4EdaValue.reshape(e4Eda.shape[0], 1, -1), 
            e4EdaKey.reshape(e4Eda.shape[0], 1, -1)
        )
        # e4Hr Instance classification
        e4HrValue, e4HrKey = self.e4HrILNet(e4Hr)
        e4HrValue, e4HrKey = (
            e4HrValue.reshape(e4Eda.shape[0], 1, -1), 
            e4HrKey.reshape(e4Eda.shape[0], 1, -1)
        )
        # e4Temp Instance classification
        e4TempValue, e4TempKey = self.e4TempILNet(e4Temp)
        e4TempValue, e4TempKey = (
            e4TempValue.reshape(e4Temp.shape[0], 1, -1), 
            e4TempKey.reshape(e4Temp.shape[0], 1, -1)
        )
        # mAcc Instance classification
        mAccValue, mAccKey = self.mAccILNet(mAcc)
        mAccValue, mAccKey = (
            mAccValue.reshape(mAcc.shape[0], 1, -1), 
            mAccKey.reshape(mAcc.shape[0], 1, -1)
        )
        # mGps Instance classification
        mGpsValue, mGpsKey = self.mGpsILNet(mGps)
        mGpsValue, mGpsKey = (
            mGpsValue.reshape(mGps.shape[0], 1, -1), 
            mGpsKey.reshape(mGps.shape[0], 1, -1)
        )
        # mGyr Instance classification
        mGyrValue, mGyrKey = self.mGyrILNet(mGyr)
        mGyrValue, mGyrKey = (
            mGyrValue.reshape(mGyr.shape[0], 1, -1), 
            mGyrKey.reshape(mGyr.shape[0], 1, -1)
        )
        # mMag Instance classification
        mMagValue, mMagKey = self.mMagILNet(mMag)
        mMagValue, mMagKey = (
            mMagValue.reshape(mMag.shape[0], 1, -1), 
            mMagKey.reshape(mMag.shape[0], 1, -1)
        )
        
        # Self-Attention based Aggregation
        self_value = torch.cat(
            [
                e4AccValue, e4BvpValue, e4EdaValue, e4HrValue, e4TempValue,
                mAccValue, mGpsValue, mGyrValue, mMagValue
            ],
             1
        )
        self_key = torch.cat(
            [
                e4AccKey, e4BvpKey, e4EdaKey, e4HrKey, e4TempKey,
                mAccKey, mGpsKey, mGyrKey, mMagKey
            ],
             1
        )
        
        
        self_query = self.Ql(self_key)
        self_key = self.Kl(self_key)
        att = torch.softmax((self_query.transpose(1, 2)@self_key / torch.tensor(9)), 1)
        b = torch.mul(self_value, att)
        
        # return 전체에 대한 예측, 개별 센서별 예측, 개별 센서 value, 모달 주요도(어텐션 가중치)
        return torch.log_softmax(torch.sum(b, 1), 1), b, self_value, att
