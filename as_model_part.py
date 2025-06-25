import numpy as np
import pandas as pd
import time

# XGB
import xgboost as xgb

# SVM                  
from sklearn import svm            
from sklearn.svm import SVC

# softmax
from sklearn.linear_model import LogisticRegression

# knn
from sklearn.neighbors import KNeighborsClassifier

# randomforest
from sklearn.ensemble import RandomForestRegressor

# lgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier

# ##  将as_data_part中导入Future_data类
# from as_data_part import Future_data 


class Datamodel:
    def __init__(self, data_name, model_name):
        self.train_data = None
        self.test_data = None
        self.label = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # self.train_TradingTime = None
        # self.test_TradingTime = None
        # self.train_TradingDate = None
        # self.test_TradingDate = None
        
        self.y_pred = None
        self.data_name = data_name
        

        self.pcrc = None
        self.modelname = model_name
        self.acc = None

        self.train_time = None
        self.test_time = None
        
    def data_read(self, train_data, test_data, label):
        self.label = label
        self.train_data = train_data.dropna()
        self.test_data = test_data.dropna()
        
              
    # def sample_balance(self):
    #     sample_balance_index = self.train_data[self.train_data[self.label]==1].index
    #     # 对序列随机排序
    #     #  固定随机种子，确保数据相同
    #     np.random.seed(20303)
    #     indices_random = np.random.permutation(sample_balance_index)
                
    #     # 随机抽取与label_next5==+-1的总个数0.5倍作为样本平衡
    #     pnum = self.train_data[self.train_data[self.label]==2].shape[0]
    #     nnum = self.train_data[self.train_data[self.label]==0].shape[0]
        
    #     zero_index = pd.Index(indices_random[:int(np.floor(0.5*(pnum+nnum)))])
    #     p_index = self.train_data[self.train_data[self.label]==2].index
    #     n_index = self.train_data[self.train_data[self.label]==0].index

    #     balance_index=zero_index.append(n_index.append(p_index))
    #     balance_index = [i for i in balance_index if i < self.train_data.shape[0]]
    #     self.train_data = self.train_data.iloc[balance_index[:],:] 
    #     self.train_data = self.train_data.dropna()
        
    def sample_balance(self):
        value_counts = self.train_data[self.label].value_counts()
        print(value_counts)
        min_count = value_counts.min()
        self.train_data = pd.concat([self.train_data[self.train_data[self.label] == value].sample(min_count) for value in value_counts.index])
        self.train_data = self.train_data.dropna()
        value_counts = self.train_data[self.label].value_counts()
        print(value_counts)
        
    
    def data_split(self):
        # split data into features and target
        
        self.X_train = self.train_data.drop(['TradingTime', 'TradingDate', 'label_next5', 'label_next10', 'label_next20', 'label_next60', 'label_next120'], axis=1)
        self.X_test = self.test_data.drop(['TradingTime', 'TradingDate', 'label_next5', 'label_next10', 'label_next20', 'label_next60', 'label_next120'], axis=1)
        self.y_train = self.train_data[self.label]
        self.y_test = self.test_data[self.label]

        self.train_time = self.train_data[['TradingTime', 'TradingDate']]
        self.test_time = self.test_data[['TradingTime', 'TradingDate']]
        
    
                
    def XGBoost(self, booster):
        '''
        booster: 'gblinear','gbtree'
        '''
        model = xgb.XGBClassifier(booster=booster, n_estimators=1000, learning_rate=0.05, objective='multi:softmax', num_class=3)
        model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)
        
    def SVM(self, C, k):
        '''
        kernel == 'linear','poly','rbf','sigmoid'
        '''
        model = svm.SVC(C=C,                         #误差项惩罚系数,默认值是1
                        kernel=k,               #线性核 kenrel="rbf":高斯核
                        class_weight='balanced',
                        decision_function_shape='ovr') #决策函数
        model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)
    
    def softmax(self):
        classifier = LogisticRegression()
        classifier.fit(self.X_train, self.y_train)
        self.y_pred = classifier.predict(self.X_test)

    def knn(self,n_neighbors=5, metric='euclidean'):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)    #实例化KNN模型
        knn.fit(self.X_train, self.y_train) 
        self.y_pred = knn.predict(self.X_test)
                
    def randomforest(self, n_estimator):
        regressor = RandomForestRegressor(n_estimators=n_estimator, random_state=0, bootstrap=True)
        # 因样本量太小，所以尝试使用bootstrap
        regressor.fit(self.X_train, self.y_train)
        self.y_pred = regressor.predict(self.X_test)
        
    def lgbm(self, max_depth=4, learning_rate=0.1, num_leaves=10, n_estimators=100):
        model = LGBMClassifier(
                max_depth=max_depth,  ## 3-5 树模型的最大深度，防止过拟合的重要参数
                learning_rate=learning_rate,
                num_leaves = num_leaves,  ##（0， 2^max_depth-1)  对性能影响
                n_estimators=n_estimators, # 使用多少个弱分类器
                objective='multiclass',
                # boosting = boosting,
                # min_data_in_leaf = min_data_in_leaf,
                # num_class=num_class,
                # min_child_weight=2,
                # subsample=0.8,
                # colsample_bytree=0.8,
                # reg_alpha=0,
                # reg_lambda=1,
                # seed=0, # 随机数种子
                n_jobs=-1 # 设置为多进程
                )
        model.fit(self.X_train,self.y_train, eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)])
        # 对测试集进行预测
        self.y_pred = model.predict(self.X_test)

    def model_fit(self, *args, **kwargs):
        if self.modelname == 'XGBoost':
            self.XGBoost(*args, **kwargs)
        elif self.modelname == 'SVM':
            self.SVM(*args, **kwargs)
        elif self.modelname == 'softmax':
            self.softmax(*args, **kwargs)
        elif self.modelname == 'knn':
            self.knn(*args, **kwargs)
        elif self.modelname == 'randomforest':
            self.randomforest(*args, **kwargs)
        elif self.modelname == 'lightgbm':
            self.lgbm(*args, **kwargs)
        else:
            print('wrong model name, plz check')


# 一些补充函数，用以评判模型
# 将y与y_pred保存出来，并用于以下评判函数

def percent_change(ratio):
    '''
    将数据转为保留二位数的百分比数据，并返回
    '''
    ratio = format(ratio*100, '.2f')
    ratio = f"{float(ratio)}%"
    return ratio

def confusion_matrix_and_accuracy(y_real, y_pred):    # 初始化混淆矩阵为3x3的零矩阵  
    '''
    计算混淆矩阵
    input: real y, pred y
    output: 混淆矩阵cm, 准确率accuracy
    230612 更新版
    （1） 增加输出每个变量的准确率和召回率
    '''
    cm = np.zeros((3, 3), dtype=int)    # 遍历两个序列的元素    
    for i in range(len(y_pred)):        # 根据预测值和真实值的组合，更新混淆矩阵的对应位置        
        if y_pred[i] == 0 and y_real[i] == 0:            
            cm[0][0] += 1        
        elif y_pred[i] == 0 and y_real[i] == 1:            
            cm[0][1] += 1        
        elif y_pred[i] == 0 and y_real[i] == 2:           
            cm[0][2] += 1        
        elif y_pred[i] == 1 and y_real[i] == 0:            
            cm[1][0] += 1        
        elif y_pred[i] == 1 and y_real[i] == 1:            
            cm[1][1] += 1        
        elif y_pred[i] == 1 and y_real[i] == 2:            
            cm[1][2] += 1        
        elif y_pred[i] == 2 and y_real[i] == 0:            
            cm[2][0] += 1        
        elif y_pred[i] == 2 and y_real[i] == 1:            
            cm[2][1] += 1        
        elif y_pred[i] == 2 and y_real[i] == 2:            
            cm[2][2] += 1    # 计算准确率为混淆矩阵对角线元素之和除以总元素个数    
    accuracy = np.trace(cm) / np.sum(cm)    # 返回混淆矩阵和准确率

#     计算每个变量的准确率和召回率
    acc_n1 = cm[0][0]/np.sum(cm[0])
    acc_0 = cm[1][1]/np.sum(cm[1])
    acc_p1 = cm[2][2]/np.sum(cm[2])
    
    rec_n1 = cm[0][0]/(cm[0][0]+cm[1][0]+cm[2][0])
    rec_0 = cm[1][1]/(cm[0][1]+cm[1][1]+cm[2][1])
    rec_p1 = cm[2][2]/(cm[0][2]+cm[1][2]+cm[2][2])
    
    acc_n1 = percent_change(acc_n1)
    rec_n1 = percent_change(rec_n1)
    acc_0 = percent_change(acc_0)
    rec_0 = percent_change(rec_0)
    acc_p1 = percent_change(acc_p1)
    rec_p1 = percent_change(rec_p1)
    
    print('-1, acc:{}, recall:{}'.format(acc_n1, rec_n1))
    print('0, acc:{}, recall:{}'.format(acc_0, rec_0))
    print('1, acc:{}, recall:{}'.format(acc_p1, rec_p1))  
    return cm, accuracy# 调用函数，打印结果cm, 


## 用于跑模型的集成函数，方便调用
def pred_test(train_data, test_data, model_name, label=5,**kwargs):    
    # start = time.time()
    label5 = Datamodel(data_name='label',model_name=model_name)
    label5.data_read(train_data, test_data, 'label_next'+str(label))
    
    # label5.sample_balance()  # 这怎么效果还变差了呢？
    label5.data_split()   # 分成feature和其他

    label5.model_fit(**kwargs)

    cm, accuracy =  confusion_matrix_and_accuracy(list(label5.y_test), label5.y_pred)
    print(f'model: {model_name}, label{label}')
    print(f"混淆矩阵为：\n{cm}, accuracy is {accuracy}")

    ## 整合交易时间
    concat_time = pd.concat([label5.X_test, label5.test_time], axis=1)

    return label5.y_pred, label5.y_test, concat_time


def model_result(model_name, train_data, val_data, label, *args, **kwargs):
    print(f'------------- {model_name} model -------------')
    _,_,_=pred_test(train_data, val_data, model_name=model_name, label=label, *args, **kwargs)
    # _,_,_=pred_test(train_data, val_data, model_name=model_name, label=10, *args, **kwargs)
    # _,_,_=pred_test(train_data, val_data, model_name=model_name, label=20, *args, **kwargs)
    # _,_,_=pred_test(train_data, val_data, model_name=model_name, label=60, *args, **kwargs)
    # _,_,_=pred_test(train_data, val_data, model_name=model_name, label=120, *args, **kwargs)



#### 对于各个label_next Time涨跌因子，挑选模型，比较各种模型，分别选出其最优的模型
if __name__ == '__main__':

    # 读取train，test data
    train_data = pd.read_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lg_train.csv', header=0, index_col=0)
    val_data = pd.read_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lg_val.csv', header=0, index_col=0)
    test_data = pd.read_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lg_test.csv', header=0, index_col=0)


    ## ------------- label_next5 部分 ------------- ##
    ## ------------- lightGBM model ------------- ##
    ## 超参数部分（好好研究一下lightGBM的各种超参数，然后再调参吧）
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=5, max_depth=3, num_leaves=10, n_estimators=50)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=5, max_depth=3, num_leaves=10, n_estimators=100)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=5, max_depth=5, num_leaves=10, n_estimators=50)
    # # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=5, max_depth=3, num_leaves=20, n_estimators=50)
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
    y_pred5, y_test5, X_test = pred_test(model_name='lightgbm',train_data=train_data, test_data=test_data, label=5, max_depth=3, num_leaves=10, n_estimators=50)

    ## ------------- label_next10 部分 ------------- ##
    ## ------------- lightGBM model ------------- ##
    ## 超参数部分（好好研究一下lightGBM的各种超参数，然后再调参吧）
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=10, max_depth=3, num_leaves=10, n_estimators=50)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=10, max_depth=3, num_leaves=10, n_estimators=100)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=10, max_depth=5, num_leaves=10, n_estimators=50)
    # # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=10, max_depth=3, num_leaves=20, n_estimators=50)
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
    y_pred10, y_test10, X_test = pred_test(model_name='lightgbm',train_data=train_data, test_data=test_data, label=10, max_depth=3, num_leaves=10, n_estimators=100)

    ## ------------- label_next20 部分 ------------- ##
    ## ------------- lightGBM model ------------- ##
    ## 超参数部分（好好研究一下lightGBM的各种超参数，然后再调参吧）
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=20, max_depth=3, num_leaves=10, n_estimators=50)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=20, max_depth=3, num_leaves=10, n_estimators=100)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=20, max_depth=5, num_leaves=10, n_estimators=50)
    # # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=20, max_depth=3, num_leaves=20, n_estimators=50)
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
    y_pred20, y_test20, X_test = pred_test(model_name='lightgbm',train_data=train_data, test_data=test_data, label=20, max_depth=3, num_leaves=10, n_estimators=50)

    ## ------------- label_next60 部分 ------------- ##
    ## ------------- lightGBM model ------------- ##
    ## 超参数部分（好好研究一下lightGBM的各种超参数，然后再调参吧）
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=60, max_depth=3, num_leaves=10, n_estimators=50)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=60, max_depth=3, num_leaves=10, n_estimators=100)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=60, max_depth=5, num_leaves=10, n_estimators=50)
    # # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=60, max_depth=3, num_leaves=20, n_estimators=50)
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
    ## 不选择把label_next60纳入投票篮子
    # # y_pred60, y_test60, X_test = pred_test(model_name='lightgbm',train_data=train_data, test_data=test_data, label=60, max_depth=3, num_leaves=10, n_estimators=50)

    ## ------------- label_next120 部分 ------------- ##
    ## ------------- lightGBM model ------------- ##
    ## 超参数部分（好好研究一下lightGBM的各种超参数，然后再调参吧）
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=120, max_depth=3, num_leaves=10, n_estimators=50)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=120, max_depth=3, num_leaves=10, n_estimators=100)
    # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=120, max_depth=5, num_leaves=10, n_estimators=50)
    # # model_result(model_name='lightgbm',train_data=train_data, val_data=val_data, label=120, max_depth=3, num_leaves=20, n_estimators=50)
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
    ## 不选择把label_next60纳入投票篮子
    # # y_pred120, y_test120, X_test = pred_test(model_name='lightgbm',train_data=train_data, test_data=test_data, label=120, max_depth=3, num_leaves=10, n_estimators=50)


    ## ------------- KNN model ------------- ##
    ### 太离谱了，数据太多KNN（N=1都）根本跑不出结果
    ## 超参数部分
    # model_result(model_name='knn',train_data=train_data, val_data=val_data, label=5, n_neighbors=1, metric='euclidean')
    # model_result(model_name='knn',train_data=train_data, val_data=val_data, label=5, n_neighbors=5, metric='euclidean')
    # model_result(model_name='knn',train_data=train_data, val_data=val_data, label=5, n_neighbors=7, metric='euclidean')

    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果

    ## ------------- SVM model ------------- ##
    ### 1w个数据都跑不出来。。。。。
    ## 超参数部分
    # model_result(model_name='SVM', train_data=train_data.iloc[0:10000,:], val_data=val_data.iloc[0:10000,:], label=5, C=1, k='linear')
    # model_result(model_name='SVM', train_data=train_data, val_data=val_data, label=5, C=1, k='rbf')
    # model_result(model_name='SVM', train_data=train_data, val_data=val_data, label=5, C=1, k='poly')
    # model_result(model_name='SVM', train_data=train_data, val_data=val_data, label=5, C=1, k='sigmoid')
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
    # model_result(model_name='SVM', train_data=train_data, val_data=test_data, label=5, C=1, k='rbf')

###     这两个模型与LightGBM几乎同出一家，所以暂时不跑了    ###
    ## ------------- RF model ------------- ##
    ## 超参数部分
    # model_result(model_name='randomforest', train_data=train_data, val_data=val_data, n_estimator=100)
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果

    ## ------------- XGB model ------------- ##
    ## 超参数部分
    # model_result(model_name='XGBoost', train_data=train_data, val_data=val_data, booster='gbtree')
    ## 最终选定超参数后，将重新跑一下结果看一下test_data的结果
###     这两个模型与LightGBM几乎同出一家，所以暂时不跑了    ###



    ## 对预测结果，及需要的数据进行保存，作为回测数据使用
    X_test['y_pred5'] = y_pred5
    X_test['y_test5'] = y_test5
    X_test['y_pred10'] = y_pred10
    X_test['y_test10'] = y_test10
    X_test['y_pred20'] = y_pred20
    X_test['y_test20'] = y_test20
    X_test.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/lgbm.csv')

    ## 数据处理部分
    # 只保留需要的因子

    df = X_test[['TradingTime', 'TradingDate','y_pred5', 'y_test5', 'y_pred10', 'y_test10', 'y_pred20', 'y_test20', 
            'BuyPrice01','BuyPrice02', 'BuyPrice03', 'BuyPrice04', 'BuyPrice05', 
            'SellPrice01','SellPrice02', 'SellPrice03', 'SellPrice04', 'SellPrice05', 
            'BuyVolume01', 'BuyVolume02', 'BuyVolume03', 'BuyVolume04', 'BuyVolume05', 
            'SellVolume01', 'SellVolume02', 'SellVolume03', 'SellVolume04', 'SellVolume05']]
    
    df.to_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/market.csv')
