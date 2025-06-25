import numpy as np
import pandas as pd

class Future_data:
    def __init__(self, path):
        self.data = pd.read_csv(path, header=0, index_col=0)
    
#     data的描述性统计部分
    def date_exp(self):
        maxdate = max(self.data['TradingDate'])
        mindate = min(self.data['TradingDate'])
        print('{} to {}'.format(mindate, maxdate))
    
    def data_shape(self):
        print('data shape is {}'.format(self.data.shape))
        
    def data_columns(self):
        print(self.data.columns)
        
#     数据是否缺失
    def data_isnull(self):
        print(self.data.isnull().sum())
        
#     因子构造部分
    def spread(self):
        '''
        spread factor, 5阶因子
        '''
        for i in range(5):
            self.data['spread0'+str(i+1)] = self.data['SellPrice0'+str(i+1)] - self.data['BuyPrice0'+str(i+1)]
        
    def mid_price(self):
        '''
        mid_price factor, 5阶因子
        '''
        for i in range(5):
            self.data['mid_price0'+str(i+1)] = (self.data['SellPrice0'+str(i+1)] + self.data['BuyPrice0'+str(i+1)])/2
        
    def micro_price(self):
        '''
        micro_price factor, 5阶因子
        '''
        for i in range(5):
            self.data['micro_price0'+str(i+1)] = (self.data['SellPrice0'+str(i+1)]*self.data['BuyVolume0'+str(i+1)] + self.data['BuyPrice0'+str(i+1)]*self.data['SellVolume0'+str(i+1)])/(self.data['BuyVolume0'+str(i+1)] + self.data['SellVolume0'+str(i+1)])
            
    def diff_factor(self):
        '''
        some diff factor, 5阶因子
        '''
        for i in range(5):
            self.data['dPa0'+str(i+1)] = self.data['SellPrice0'+str(i+1)].shift(1) - self.data['SellPrice0'+str(i+1)]
            self.data['dPb0'+str(i+1)] = self.data['BuyPrice0'+str(i+1)].shift(1) - self.data['BuyPrice0'+str(i+1)]
            self.data['dPmicro'+str(i+1)] = self.data['micro_price0'+str(i+1)].shift(1) - self.data['micro_price0'+str(i+1)]    
    
    def label_factor(self):
        '''
        target label, 5, 10, 20, 60, 120
        '''
        def next_label(self):
            '''
            midprice next label, 5, 10, 20, 60, 120
            '''
            label_list = [5, 10, 20, 60, 120]
            for step in label_list:
                self.data['mid_next'+str(step)] = self.data['mid_price01'].rolling(step).mean().shift(1-step)
                self.data['label_next'+str(step)] = np.where(self.data['mid_next'+str(step)]-self.data['mid_price01']>=0.6, 2, np.where(self.data['mid_next'+str(step)]-self.data['mid_price01']<=-0.6, 0, 1))
        try:
            self.data['midprice01'] == True
        except:
            self.mid_price()
        next_label(self)

    def OFI(self):
        '''
        OFI, 5阶因子
        '''
        for i in range(5):
            self.data['delta_vB'+str(i+1)] = np.where(self.data['BuyPrice0'+str(i+1)] < self.data['BuyPrice0'+str(i+1)].shift(1), -self.data['BuyVolume0'+str(i+1)].shift(1), 
                                      np.where(self.data['BuyPrice0'+str(i+1)] == self.data['BuyPrice0'+str(i+1)].shift(1), self.data['BuyVolume0'+str(i+1)]-self.data['BuyVolume0'+str(i+1)].shift(1),
                                      np.where(self.data['BuyPrice0'+str(i+1)] > self.data['BuyPrice0'+str(i+1)].shift(1), self.data['BuyVolume0'+str(i+1)],np.nan)))
            self.data['delta_vA'+str(i+1)] = np.where(self.data['SellPrice0'+str(i+1)] < self.data['SellPrice0'+str(i+1)].shift(1), -self.data['SellVolume0'+str(i+1)].shift(1), 
                                      np.where(self.data['SellPrice0'+str(i+1)] == self.data['SellPrice0'+str(i+1)].shift(1), self.data['SellVolume0'+str(i+1)]-self.data['SellVolume0'+str(i+1)].shift(1),
                                      np.where(self.data['SellPrice0'+str(i+1)] > self.data['SellPrice0'+str(i+1)].shift(1), self.data['SellVolume0'+str(i+1)],np.nan)))

            self.data['OFI0'+str(i+1)] = self.data['delta_vB'+str(i+1)] - self.data['delta_vA'+str(i+1)]
            self.data = self.data.drop(['delta_vB'+str(i+1),'delta_vA'+str(i+1)], axis=1)
            
    def LogquoteSlope(self):
        for i in range(5):
            self.data['LogquoteSlope0'+str(i+1)] = (np.log(self.data['SellPrice0'+str(i+1)]) - np.log(self.data['BuyPrice0'+str(i+1)]))/(np.log(self.data['BuyVolume0'+str(i+1)]+1) + np.log(self.data['SellVolume0'+str(i+1)]+1))
            
    def SOIR(self):
        for i in range(5):
            self.data['SOIR0'+str(i+1)] = ((self.data['BuyVolume0'+str(i+1)] - self.data['SellVolume0'+str(i+1)])/(self.data['BuyVolume0'+str(i+1)] + self.data['SellVolume0'+str(i+1)]))
            self.data = self.data.replace(np.inf, np.nan)
            self.data = self.data.replace(-np.inf, np.nan)
            
    def MPC(self):
        for k in [1, 5]:
            self.data['MPC'+str(k)] = (((self.data['SellPrice01'] + self.data['BuyVolume01'])/2) - ((self.data['SellPrice01'].shift(k) + self.data['BuyVolume01'].shift(k))/2))/((self.data['SellPrice01'].shift(k) + self.data['BuyVolume01'].shift(k))/2)
            
    def factor_all(self):
        self.spread()
        self.mid_price()
        self.micro_price()
        self.diff_factor()
        self.label_factor()
        self.OFI()
        self.LogquoteSlope()
        self.SOIR()
        self.MPC()
        
    def data_preprocess(self):
        self.data = self.data.drop(['Symbol','SettleGrouplD', 'SettlelD','Market','UNIX','ContinueSign', 'ContinueSignName', 'SecuritylD', 'ShortName','BuyOrSell','OpenClose'], axis=1)
#         self.data = pd.get_dummies(self.data)
#         BuyOrSell_mapping = {"S": -1, "B": 1, 'N': 0}
#         self.data['BuyOrSell'] = self.data['BuyOrSell'].map(sex_mapping)
#         OpenClose_mapping = {'双开仓':0, '双平仓':1,'多头换手':2,'空头换手':3,'多头开仓':4,'空头开仓':5,'多头平仓':6,'空头平仓':7,'多空相持':8,'N':9}
#         self.data['OpenClose'] = self.data['OpenClose'].map(OpenClose_mapping)


# 读取数据
IC = Future_data('D:/Lecture/FinalPaper/实习相关材料/IFIC/IFIC/ic.csv')
IC.factor_all()
IC.data_preprocess()

## 数据保存，输出为新的数据
IC.data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/ic_new.csv')


## 将数据分为训练集，验证集，测试集

train_data = IC.data[IC.data['TradingDate'].between(20220801, 20231031)]
val_data = IC.data[IC.data['TradingDate'].between(20221101, 20221131)]
test_data = IC.data[IC.data['TradingDate'].between(20221201, 20221231)]

train_data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/ic_train.csv')
val_data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/ic_val.csv')
test_data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/ic_test.csv')

## lightGBM 最后所使用数据集

lg_train_data = IC.data[IC.data['TradingDate'].between(20220801, 20231231)]
lg_val_data = IC.data[IC.data['TradingDate'].between(20230101, 20230131)]
lg_test_data = IC.data[IC.data['TradingDate'].between(20230201, 20230231)]

lg_train_data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lg_train.csv')
lg_val_data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lg_val.csv')
lg_test_data.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lg_test.csv')