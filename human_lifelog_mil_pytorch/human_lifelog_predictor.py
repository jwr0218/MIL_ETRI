import human_lifelog_mil_pytorch.modules as modules
import torch

class HumanDataset(torch.utils.data.Dataset):
    def __init__(self, DatasetDf):
        self.e4Acc = DatasetDf['e4Acc'].to_list()
        #self.e4Bvp = DatasetDf['e4Bvp__value'].to_list()
        #self.e4Eda = DatasetDf['e4Eda__eda'].to_list()
        #self.e4Hr = DatasetDf['e4Hr__hr'].to_list()
        #self.e4Temp = DatasetDf['e4Temp__temp'].to_list()
        self.e4Bvp = DatasetDf['e4Bvp__x'].to_list()
        self.e4Eda = DatasetDf['e4Eda__x'].to_list()
        self.e4Hr = DatasetDf['e4Hr__x'].to_list()
        self.e4Temp = DatasetDf['e4Temp__x'].to_list()
        
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
    
# 각 모달에서 feature을 추출하는 모듈
class ILNet(torch.nn.Module):
    def __init__(self, in_channel_num = 1):
        super(ILNet, self).__init__()
        self.check = True

        self.in_channel_num = in_channel_num
        if in_channel_num == 0:
            self.check=False
            self.Encoder = torch.nn.Sequential(
                torch.nn.Linear(1, 32),
                torch.nn.BatchNorm1d(32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.BatchNorm1d(32),            )
        else:
            self.Encoder = modules.CausalCNNEncoder(
                in_channels = in_channel_num, 
                channels = 8, 
                depth = 2, 
                reduced_size = 16, 
                out_channels = 32,
                kernel_size = 3
            )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
    def forward(self, x):
        #print('Tensor : ',x.shape)
        x = x.reshape(-1, self.in_channel_num, 1)
        x = self.Encoder(x)

        return x
    
# 각 모달에서 추출된 feature을 조합해서 어떠한 cluster에 속해있는지 예측하는 모델
class LifeLogNet(torch.nn.Module):
    def __init__(self, class_num):
        super(LifeLogNet, self).__init__()
        # e4Acc Instance classifier
        self.e4AccILNet = ILNet(in_channel_num = 3)
        # e4Bvp Instance classifier
        self.e4BvpILNet = ILNet(in_channel_num = 1)
        # e4Eda Instance classifier
        self.e4EdaILNet = ILNet(in_channel_num = 1)
        # e4Hr Instance classifier
        self.e4HrILNet = ILNet(in_channel_num = 1)
        # e4Temp Instance classifier
        self.e4TempILNet = ILNet(in_channel_num = 1)
        # mAcc Instance classifier
        self.mAccILNet = ILNet(in_channel_num = 3)
        # mGps Instance classifier
        self.mGpsILNet = ILNet(in_channel_num = 2)
        # mGyr Instance classifier
        self.mGyrILNet = ILNet(in_channel_num = 3)
        # mMag Instance classifier
        self.mMagILNet = ILNet(in_channel_num = 3)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(288, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, class_num),
            torch.nn.LogSoftmax(1)
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag):
        # e4Acc Instance classification
        e4Acc = self.e4AccILNet(e4Acc)
        e4Acc = e4Acc.reshape(e4Acc.shape[0], 1 , -1)
        # e4Bvp Instance classification
        e4Bvp = self.e4BvpILNet(e4Bvp)
        e4Bvp = e4Bvp.reshape(e4Bvp.shape[0], 1, -1)
        # e4Eda Instance classification
        e4Eda = self.e4EdaILNet(e4Eda)
        e4Eda = e4Eda.reshape(e4Eda.shape[0], 1, -1)
        # e4Hr Instance classification
        e4Hr = self.e4HrILNet(e4Hr)
        e4Hr = e4Hr.reshape(e4Hr.shape[0], 1, -1)
        # e4Temp Instance classification
        e4Temp = self.e4TempILNet(e4Temp)
        e4Temp = e4Temp.reshape(e4Temp.shape[0], 1, -1)
        # mAcc Instance classification
        mAcc = self.mAccILNet(mAcc)
        mAcc = mAcc.reshape(mAcc.shape[0], 1, -1)
        # mGps Instance classification
        mGps = self.mGpsILNet(mGps)
        mGps = mGps.reshape(mGps.shape[0], 1, -1)
        # mGyr Instance classification
        mGyr = self.mGyrILNet(mGyr)
        mGyr = mGyr.reshape(mGyr.shape[0], 1, -1)
        # mMag Instance classification
        mMag = self.mMagILNet(mMag)
        mMag = mMag.reshape(mMag.shape[0], 1, -1)
        
        self_value = torch.cat(
            [
                e4Acc, e4Bvp, e4Eda, e4Hr, e4Temp, mAcc, mGps, mGyr, mMag
            ],
             1
        ).reshape(-1, 288)
        
        return self.classifier(self_value)