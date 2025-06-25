####  从上一步得到预测序列，用预测序列的数据，首先做
####  投票篮子首先整合出，最终的预测因子，然后用预测因子来决定挂单

import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

import time

class As_model:
    def __init__(self):
        self.midprice_list = None
        self.spread_list = None
        self.IRA = 0
        self.gamma = 0
        self.kappa = 0
        self.sigma_square = 0

    def read_list(self, midprice_list, spread_list):
        self.midprice_list = midprice_list[0:300]
        self.spread_list = spread_list[0:300]

    def compute_sigma(self):
        self.sigma_square = np.power(np.std(self.midprice_list), 2)

    def compute_gamma(self, IRA=0.1, q=0, T=1):
        '''
        这里需要进一步处理，要将生成相应参数后的采样样本从数据中删去吧？
        '''
        if q == 0:
            self.gamma = IRA*(max(self.spread_list)-min(self.spread_list))/(2*np.abs(0.1)*self.sigma_square*T)
        else:
            self.gamma = IRA*(max(self.spread_list)-min(self.spread_list))/(2*np.abs(q)*self.sigma_square*T)

    def compute_kappa(self, T=1):
        self.kappa = self.gamma/(np.exp((np.average(self.spread_list)-self.sigma_square*self.gamma*T)*self.gamma/2)-1)
    
    def param_prepare(self, IRA=0.2):
        '''
        在此步之前，要先read_list，将spread与midprice的list读取
        '''
        self.compute_sigma()
        self.compute_gamma(IRA=IRA, q=-0.1)
        self.compute_kappa(T=1)
    
    def get_price_set(self, midprice, q=0.1, T=1):
        '''
        此步之前，要param_prepare，将相关参数计算好
        '''

        reservation_price = midprice - q*self.gamma*self.sigma_square*T
        spread = self.gamma*self.sigma_square*T + (2/self.gamma)*np.log(1+(self.gamma/self.kappa))

        ask_price = reservation_price - spread/2
        bid_price = reservation_price + spread/2

        return ask_price, bid_price

    def get_price_set_fixed(self, midprice, q=0.1, T=1, factor_fixed=1):
        '''
        传入修正因子，挂单价位，然后依据修正因子修正reservation price
        '''

        if factor_fixed == 2:
            midprice +=10
        elif factor_fixed == 0:
            midprice -=10

        ask_price, bid_price = self.get_price_set(midprice, q=q, T=T)

        return ask_price, bid_price
        
    def vol_fix(self, vol, q, factor_fixed):
        '''
        因为数据原因，挂单量太小，导致gr5论文的库存风险修正数值小于1，所以暂时简单化处理
        '''
        if q > 0:
            ask_vol = vol 
            bid_vol = vol + 5
        elif q < 0:
            ask_vol = vol + 5
            bid_vol = vol 

        if factor_fixed == 2:
            ask_vol = vol + 5
            bid_vol = vol - 5 
        elif factor_fixed == 0:
            ask_vol = vol - 5
            bid_vol = vol + 5                      
        
        return ask_vol, bid_vol


class Agent:
    def __init__(self):
        '''
        主要用market dataframe作为主体，来记录各个时刻的数据，例如挂单量，
        挂单价格，现金量，期权量，最后可以将整个column画图，作为时序变化图来看
        然后挂单序列出来后，根据撤单时间，去遍历后面tick的买卖价量，判断是否成交
        然后根据是否成交，就有了撤单量column，以及现金量，期权量的变动与否。

        将这个dataframe尽量简化，剩买一买五价量，时间，factor_fixed，
        【spread】和【midprice】
        然后生成的新columns，cash, option, cancel, 再来个asset吧，就是option用市场中间价结算的，加上cash
        
        用行的序列号输入，作为遍历index，从而得到所需的对应上下行的，所需变量的数据

        '''
        self.market_data = None
        self.as_model = None

    def read_data(self, market_data):
        self.market_data = copy.copy(market_data)

    def add_info(self, init_cash=0.0, init_option=0):
        self.market_data.loc[:, 'cash'] = init_cash
        self.market_data.loc[:, 'q'] = init_option
        # self.market_data['asset'] = self.market_data['cash'] + self.market_data['midprice']*self.market_data['q']
        
        self.market_data.loc[:, 'ask_price'] = 0
        self.market_data.loc[:, 'bid_price'] = 0
        self.market_data.loc[:, 'ask_vol'] = 0
        self.market_data.loc[:, 'bid_vol'] = 0
        self.market_data.loc[:, 'ask_vol_backup'] = 0
        self.market_data.loc[:, 'bid_vol_backup'] = 0
        self.market_data.loc[:, 'cancel'] = 0
        
        self.market_data.loc[:,'ask_tradedvol'] = 0  # ask 该步的成交量
        self.market_data.loc[:,'bid_tradedvol'] = 0  # bid 该步的成交量
        
        self.market_data.loc[:, 'cancel'] = 0 
    
    def compute_order(self, row_index, vol=2, fix=True):
        '''
        input: 将midprice，上一步的q，IRA输入as_model的get_price_set,得到ask/bid price，
        再将原本的挂单vol，通过vol_fix【上一步的q】，进行修正，得到ask/bid vol
        output: 输出ask/bid price & vol
        
        '''
        if row_index == 0:
            pass
        else:
            if fix:
                self.market_data.loc[row_index,'ask_price'], self.market_data.loc[row_index,'bid_price'] = self.as_model.get_price_set_fixed(self.market_data.loc[row_index,'midprice'], q=self.market_data.loc[row_index-1,'q'],T=1,factor_fixed=self.market_data.loc[row_index,'factor_fixed'])
                self.market_data.loc[row_index,'ask_vol'], self.market_data.loc[row_index,'bid_vol'] = self.as_model.vol_fix(vol, self.market_data.loc[row_index-1,'q'], self.market_data.loc[row_index, 'factor_fixed'])
            else:
                self.market_data.loc[row_index,'ask_price'], self.market_data.loc[row_index,'bid_price'] = self.as_model.get_price_set(self.market_data.loc[row_index,'midprice'], q=self.market_data.loc[row_index-1,'q'],T=1)
                self.market_data.loc[row_index,'ask_vol'], self.market_data.loc[row_index,'bid_vol'] = vol, vol
        
        self.market_data.loc[row_index, 'ask_vol_backup'] = self.market_data.loc[row_index, 'ask_vol'].copy()
        self.market_data.loc[row_index, 'bid_vol_backup'] = self.market_data.loc[row_index, 'bid_vol'].copy()


    def compute_cancel(self, row_index, withdrawal_duration=3):
        '''
        withdrawal_duation 持续多长时间后撤单，即挂单总时长，需要遍历的长度，暂时默认10tick【肯定要改的地方】
        input: 上一步的cancel,下几步的bid/ask price01-vol，以及该步的ask/bid order price，vol
        ouput：根据，下几步的bid/ask price01，来判断是否成交，以及成交量，然后看是否还有剩，
        如果有剩的话，就是cancel vol，加上之前累积的cancel，输出两者之和
        
        同时，每个挂单，进行判断后，逐步累计更新到下面的行中，使得全部更新完相关行的之后，就是那一时刻的真正成交完后的cash/option，
        以及cancel的值
        '''
        ### 用于判断是否成交
        
        ## 有个前提是，挂买单的时候，挂单量*挂单价应该小于cash，卖单同理，需要由足够的inventory，所以这里直接假定做市商有足够的cash与inventory
        for i in range(withdrawal_duration):
            while self.market_data.loc[row_index, 'ask_vol_backup'] != 0:
                try:
                    if self.market_data.loc[row_index, 'ask_price'] >= self.market_data.loc[row_index+i, 'BuyPrice01']:
                        if self.market_data.loc[row_index, 'ask_vol_backup'] > self.market_data.loc[row_index+i, 'BuyVolume01']:
                            self.market_data.loc[row_index, 'ask_vol_backup'] = self.market_data.loc[row_index, 'ask_vol_backup'] - self.market_data.loc[row_index+i, 'BuyVolume01']
                            
                            ## 对cash, q, 进行更新，该步的成交量，可以计算完后，将vol与备份的相减即可得到
                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i-1, 'cash'] - self.market_data.loc[row_index+i, 'BuyVolume01']*self.market_data.loc[row_index+i, 'BuyPrice01']
                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i-1, 'q'] + self.market_data.loc[row_index+i, 'BuyVolume01']
                            
                            self.market_data.loc[row_index+i, 'BuyVolume01'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化

                            
                            if self.market_data.loc[row_index, 'ask_price'] >= self.market_data.loc[row_index+i, 'BuyPrice02']:
                                if self.market_data.loc[row_index, 'ask_vol_backup'] > self.market_data.loc[row_index+i, 'BuyVolume02']:
                                    self.market_data.loc[row_index, 'ask_vol_backup'] = self.market_data.loc[row_index, 'ask_vol_backup'] - self.market_data.loc[row_index+i, 'BuyVolume02']
                                    
                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index+i, 'BuyVolume02']*self.market_data.loc[row_index+i, 'BuyPrice02']
                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index+i, 'BuyVolume02']
                                    
                                    self.market_data.loc[row_index+i, 'BuyVolume02'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化

                                    if self.market_data.loc[row_index, 'ask_price'] >= self.market_data.loc[row_index+i, 'BuyPrice03']:
                                        if self.market_data.loc[row_index, 'ask_vol_backup'] > self.market_data.loc[row_index+i, 'BuyVolume03']:
                                            self.market_data.loc[row_index, 'ask_vol_backup'] = self.market_data.loc[row_index, 'ask_vol_backup'] - self.market_data.loc[row_index+i, 'BuyVolume03']
                                            
                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index+i, 'BuyVolume03']*self.market_data.loc[row_index+i, 'BuyPrice03']
                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index+i, 'BuyVolume03']
                                            self.market_data.loc[row_index+i, 'BuyVolume03'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化
                                            
                                            if self.market_data.loc[row_index, 'ask_price'] >= self.market_data.loc[row_index+i, 'BuyPrice04']:
                                                if self.market_data.loc[row_index, 'ask_vol_backup'] > self.market_data.loc[row_index+i, 'BuyVolume04']:
                                                    self.market_data.loc[row_index, 'ask_vol_backup'] = self.market_data.loc[row_index, 'ask_vol_backup'] - self.market_data.loc[row_index+i, 'BuyVolume04']
                                                    
                                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index+i, 'BuyVolume04']*self.market_data.loc[row_index+i, 'BuyPrice04']
                                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index+i, 'BuyVolume04']
                                                    self.market_data.loc[row_index+i, 'BuyVolume04'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化
                                                    
                                                    if self.market_data.loc[row_index, 'ask_price'] >= self.market_data.loc[row_index+i, 'BuyPrice05']:
                                                        if self.market_data.loc[row_index, 'ask_vol_backup'] > self.market_data.loc[row_index+i, 'BuyVolume05']:
                                                            self.market_data.loc[row_index, 'ask_vol_backup'] = self.market_data.loc[row_index, 'ask_vol_backup'] - self.market_data.loc[row_index+i, 'BuyVolume05']
                                                            
                                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index+i, 'BuyVolume05']*self.market_data.loc[row_index+i, 'BuyPrice05']
                                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index+i, 'BuyVolume05']
                                                            self.market_data.loc[row_index+i, 'BuyVolume05'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化

                                                        else:
                                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index, 'ask_vol_backup']*self.market_data.loc[row_index+i, 'BuyPrice05']
                                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index, 'ask_vol_backup']
                                                            self.market_data.loc[row_index+i, 'BuyVolume05'] = self.market_data.loc[row_index+i, 'BuyVolume05'] - self.market_data.loc[row_index, 'ask_vol_backup']
                                                            self.market_data.loc[row_index, 'ask_vol_backup'] = 0
                                                    else:
                                                        break
                                                    
                                                else:
                                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index, 'ask_vol_backup']*self.market_data.loc[row_index+i, 'BuyPrice04']
                                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index, 'ask_vol_backup']
                                                    self.market_data.loc[row_index+i, 'BuyVolume04'] = self.market_data.loc[row_index+i, 'BuyVolume04'] - self.market_data.loc[row_index, 'ask_vol_backup']
                                                    self.market_data.loc[row_index, 'ask_vol_backup'] = 0
                                            else:
                                                break
                                    
                                        else:
                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index, 'ask_vol_backup']*self.market_data.loc[row_index+i, 'BuyPrice03']
                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index, 'ask_vol_backup']
                                            self.market_data.loc[row_index+i, 'BuyVolume03'] = self.market_data.loc[row_index+i, 'BuyVolume03'] - self.market_data.loc[row_index, 'ask_vol_backup']
                                            self.market_data.loc[row_index, 'ask_vol_backup'] = 0
                                    else:
                                        break
                                
                                else:
                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] - self.market_data.loc[row_index, 'ask_vol_backup']*self.market_data.loc[row_index+i, 'BuyPrice02']
                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] + self.market_data.loc[row_index, 'ask_vol_backup']
                                    self.market_data.loc[row_index+i, 'BuyVolume02'] = self.market_data.loc[row_index+i, 'BuyVolume02'] - self.market_data.loc[row_index, 'ask_vol_backup']
                                    self.market_data.loc[row_index, 'ask_vol_backup'] = 0
                            else:
                                break
                            
                        else:
                            ### 这里需要对不足量的进行cash操作，不需要，初始cash和inventory均为0，但可以无限负值
                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i-1, 'cash'] - self.market_data.loc[row_index, 'ask_vol_backup']*self.market_data.loc[row_index+i, 'BuyPrice01']
                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i-1, 'q'] + self.market_data.loc[row_index, 'ask_vol_backup']
                            self.market_data.loc[row_index+i, 'BuyVolume01'] = self.market_data.loc[row_index+i-1, 'BuyVolume01'] - self.market_data.loc[row_index, 'ask_vol_backup']
                            self.market_data.loc[row_index, 'ask_vol_backup'] = 0
                            
                    else:
                        break
                
                except:
                    break
        
        
            while self.market_data.loc[row_index, 'bid_vol_backup'] != 0:
                try:
                    if self.market_data.loc[row_index, 'bid_price'] <= self.market_data.loc[row_index+i, 'SellPrice01']:
                        if self.market_data.loc[row_index, 'bid_vol_backup'] > self.market_data.loc[row_index+i, 'SellVolume01']:
                            self.market_data.loc[row_index, 'bid_vol_backup'] = self.market_data.loc[row_index, 'bid_vol_backup'] - self.market_data.loc[row_index+i, 'SellVolume01']
                            
                            ## 对cash, q, 进行更新，该步的成交量，可以计算完后，将vol与备份的相减即可得到
                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i-1, 'cash'] + self.market_data.loc[row_index+i, 'SellVolume01']*self.market_data.loc[row_index+i, 'SellPrice01']
                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i-1, 'q'] - self.market_data.loc[row_index+i, 'SellVolume01']
                            
                            self.market_data.loc[row_index+i, 'SellVolume01'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化

                            
                            if self.market_data.loc[row_index, 'bid_price'] <= self.market_data.loc[row_index+i, 'SellPrice02']:
                                if self.market_data.loc[row_index, 'bid_vol_backup'] > self.market_data.loc[row_index+i, 'SellVolume02']:
                                    self.market_data.loc[row_index, 'bid_vol_backup'] = self.market_data.loc[row_index, 'bid_vol_backup'] - self.market_data.loc[row_index+i, 'SellVolume02']
                                    
                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index+i, 'SellVolume02']*self.market_data.loc[row_index+i, 'SellPrice02']
                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index+i, 'SellVolume02']
                                    
                                    self.market_data.loc[row_index+i, 'SellVolume02'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化

                                    if self.market_data.loc[row_index, 'bid_price'] <= self.market_data.loc[row_index+i, 'SellPrice03']:
                                        if self.market_data.loc[row_index, 'bid_vol_backup'] > self.market_data.loc[row_index+i, 'SellVolume03']:
                                            self.market_data.loc[row_index, 'bid_vol_backup'] = self.market_data.loc[row_index, 'bid_vol_backup'] - self.market_data.loc[row_index+i, 'SellVolume03']
                                            
                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index+i, 'SellVolume03']*self.market_data.loc[row_index+i, 'SellPrice03']
                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index+i, 'SellVolume03']
                                            self.market_data.loc[row_index+i, 'SellVolume03'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化
                                            
                                            if self.market_data.loc[row_index, 'bid_price'] <= self.market_data.loc[row_index+i, 'SellPrice04']:
                                                if self.market_data.loc[row_index, 'bid_vol_backup'] > self.market_data.loc[row_index+i, 'SellVolume04']:
                                                    self.market_data.loc[row_index, 'bid_vol_backup'] = self.market_data.loc[row_index, 'bid_vol_backup'] - self.market_data.loc[row_index+i, 'SellVolume04']
                                                    
                                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index+i, 'SellVolume04']*self.market_data.loc[row_index+i, 'SellPrice04']
                                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index+i, 'SellVolume04']
                                                    self.market_data.loc[row_index+i, 'SellVolume04'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化
                                                    
                                                    if self.market_data.loc[row_index, 'bid_price'] <= self.market_data.loc[row_index+i, 'SellPrice05']:
                                                        if self.market_data.loc[row_index, 'bid_vol_backup'] > self.market_data.loc[row_index+i, 'SellVolume05']:
                                                            self.market_data.loc[row_index, 'bid_vol_backup'] = self.market_data.loc[row_index, 'bid_vol_backup'] - self.market_data.loc[row_index+i, 'SellVolume05']
                                                            
                                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index+i, 'SellVolume05']*self.market_data.loc[row_index+i, 'SellPrice05']
                                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index+i, 'SellVolume05']
                                                            self.market_data.loc[row_index+i, 'SellVolume05'] = 0  ## 将对应位置的删去，但不需要if判断，因为如果是0上面三步更新没有变化

                                                        else:
                                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index, 'bid_vol_backup']*self.market_data.loc[row_index+i, 'SellPrice05']
                                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                                            self.market_data.loc[row_index+i, 'SellVolume05'] = self.market_data.loc[row_index+i, 'SellVolume05'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                                            self.market_data.loc[row_index, 'bid_vol_backup'] = 0
                                                    else:
                                                        break
                                                    
                                                else:
                                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index, 'bid_vol_backup']*self.market_data.loc[row_index+i, 'SellPrice04']
                                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                                    self.market_data.loc[row_index+i, 'SellVolume04'] = self.market_data.loc[row_index+i, 'SellVolume04'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                                    self.market_data.loc[row_index, 'bid_vol_backup'] = 0
                                            else:
                                                break
                                    
                                        else:
                                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index, 'bid_vol_backup']*self.market_data.loc[row_index+i, 'SellPrice03']
                                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                            self.market_data.loc[row_index+i, 'SellVolume03'] = self.market_data.loc[row_index+i, 'SellVolume03'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                            self.market_data.loc[row_index, 'bid_vol_backup'] = 0
                                    else:
                                        break
                                
                                else:
                                    self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i, 'cash'] + self.market_data.loc[row_index, 'bid_vol_backup']*self.market_data.loc[row_index+i, 'SellPrice02']
                                    self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i, 'q'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                    self.market_data.loc[row_index+i, 'SellVolume02'] = self.market_data.loc[row_index+i, 'SellVolume02'] - self.market_data.loc[row_index, 'bid_vol_backup']
                                    self.market_data.loc[row_index, 'bid_vol_backup'] = 0
                            else:
                                break
                            
                        else:
                            ### 这里需要对不足量的进行cash操作，不需要，初始cash和inventory均为0，但可以无限负值
                            self.market_data.loc[row_index+i, 'cash'] = self.market_data.loc[row_index+i-1, 'cash'] + self.market_data.loc[row_index, 'bid_vol_backup']*self.market_data.loc[row_index+i, 'SellPrice01']
                            self.market_data.loc[row_index+i, 'q'] = self.market_data.loc[row_index+i-1, 'q'] - self.market_data.loc[row_index, 'bid_vol_backup']
                            self.market_data.loc[row_index+i, 'SellVolume01'] = self.market_data.loc[row_index+i-1, 'SellVolume01'] - self.market_data.loc[row_index, 'bid_vol_backup']
                            self.market_data.loc[row_index, 'bid_vol_backup'] = 0
                            
                    else:
                        break
                
                except:
                    break
    
    def update(self, IRA=0.2, withdrawal_duration=3, fix=True):
        '''
        因为只能一行数据一行数据的更新，不能一列一列数据的更新【下一行的计算依赖上一行的很多变量】，所以使用apply
        '''
        self.as_model = As_model()
        self.as_model.read_list(list(self.market_data['midprice']), list(self.market_data['spread']))
        self.as_model.param_prepare(IRA=IRA)
        
        for row_index in tqdm(range(self.market_data.shape[0]), desc='Running', unit='iteration', ncols=50):
            self.compute_order(row_index, vol=20, fix=fix)  ## 这里是否要对fix=False，就是原本的as，默认是as_fixed
            self.compute_cancel(row_index, withdrawal_duration=withdrawal_duration)
            if row_index % 100==0:
                print(row_index)

######## 第三部分 读取每日未挂单数据，并得到挂单等其他变量后，整合一个数据集 ########

###  依次读取文件夹里的数据，分别进行处理
import os
import glob
import pandas as pd



# 设置要读取的文件夹路径
folder_path = 'D:/Lecture/FinalPaper/AS_codes_copy/data/data_by_date/'

# 使用 glob 模块获取文件夹中所有的 CSV 文件名
file_pattern = os.path.join(folder_path, '*.csv')
file_list = glob.glob(file_pattern)


cash_temp = 0
q_temp = -0.1
df = pd.read_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/data_by_date/data_20230210.csv', header=0, index_col=0)
df.reset_index(drop=True, inplace=True)

agent = Agent()
# starttime = time.time()
agent.read_data(df)
agent.add_info(init_cash=cash_temp, init_option=q_temp)
agent.update(IRA=0.2, withdrawal_duration=10, fix=False)   #### 核心函数
# print(agent.market_data[['cash','q','ask_vol_backup','bid_vol_backup','ask_vol','bid_vol']].tail(10))
# print(f"time:{time.time()-starttime}, file_date:{file}]")
cash_temp = agent.market_data['cash'].iloc[-1]
q_temp = agent.market_data['q'].iloc[-1]
print(f'cash:{cash_temp}')
print(f'q:{q_temp}')


new_file_name = f"D:/Lecture/FinalPaper/AS_codes_copy/data/unfixed_{20230210}.csv"
                
# 将处理后的 DataFrame 写入新文件
agent.market_data.to_csv(new_file_name, index=False)
