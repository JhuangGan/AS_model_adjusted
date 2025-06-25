
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import time        

import os
import glob
import re

import matplotlib.pyplot as plt

class Order():
    def __init__(self, row, TradingTime, ask_price, ask_vol, bid_price, bid_vol) -> None:
        self.row = row
        self.TradingTime = TradingTime

        self.ask_price = ask_price
        self.ask_vol = ask_vol
        self.bid_price = bid_price
        self.bid_vol = bid_vol

class TimeTrade():
    def __init__(self, row, TradingTime,q,cash, BuyPrice01,BuyPrice02,BuyPrice03,BuyPrice04,BuyPrice05,BuyVolume01,BuyVolume02,BuyVolume03,BuyVolume04,BuyVolume05,
                 SellPrice01,SellPrice02,SellPrice03,SellPrice04,SellPrice05,SellVolume01,SellVolume02,SellVolume03,SellVolume04,SellVolume05,midprice,spread,T,factor_fixed) -> None:
        ### 这里的q和cash为单次时间的变化，需要最后再来一次累加，得到单个时刻的数值

        self.row = row
        self.q = q  ## 由于之前计算AS时为避免q=0设置的0.1这里，由于计算单次q变化，所以改回0
        self.cash = cash
        self.TradingTime = TradingTime
        self.BuyPrice01 = BuyPrice01
        self.BuyPrice02 = BuyPrice02
        self.BuyPrice03 = BuyPrice03
        self.BuyPrice04 = BuyPrice04
        self.BuyPrice05 = BuyPrice05
        self.BuyVolume01 = BuyVolume01
        self.BuyVolume02 = BuyVolume02
        self.BuyVolume03 = BuyVolume03
        self.BuyVolume04 = BuyVolume04
        self.BuyVolume05 = BuyVolume05
        self.SellPrice01 = SellPrice01
        self.SellPrice02 = SellPrice02
        self.SellPrice03 = SellPrice03
        self.SellPrice04 = SellPrice04
        self.SellPrice05 = SellPrice05
        self.SellVolume01 = SellVolume01
        self.SellVolume02 = SellVolume02
        self.SellVolume03 = SellVolume03
        self.SellVolume04 = SellVolume04
        self.SellVolume05 = SellVolume05

        self.midprice = midprice
        self.spread = spread
        self.T = T
        self.factor_fixed = factor_fixed


class AS_model():
    ## 用于存储当日参数，与计算ask_price, bid_price, ask_vol, bid_vol
    def __init__(self, midprice_list, spread_list, IRA=0.1, q=-0.1, T=7, fix=False) -> None:
        self.fix = fix  ## 是否修正

        self.midprice_list = midprice_list
        self.spread_list = spread_list
        self.IRA = IRA
        self.sigma_square = np.power(np.std(self.midprice_list), 2)
        if q == 0:
            self.gamma = IRA*(max(self.spread_list)-min(self.spread_list))/(2*np.abs(0.1)*self.sigma_square*T)
        else:
            self.gamma = IRA*(max(self.spread_list)-min(self.spread_list))/(2*np.abs(q)*self.sigma_square*T)

        self.kappa = self.gamma/(np.exp((np.average(self.spread_list)-self.sigma_square*self.gamma*T)*self.gamma/2)-1)
    
    def get_price_set(self, midprice, q=0.1, T=1, factor_fixed=1, u=10):
        '''
        传入修正因子，挂单价位，然后依据是否修正，才判断修正因子修正reservation price
        '''
        if self.fix:
            if factor_fixed == 2:
                midprice += u
            elif factor_fixed == 0:
                midprice -= u
        
        if q == 0:
            self.gamma = self.IRA*(max(self.spread_list)-min(self.spread_list))/(2*np.abs(0.1)*self.sigma_square*T)
        else:
            self.gamma = self.IRA*(max(self.spread_list)-min(self.spread_list))/(2*np.abs(q)*self.sigma_square*T)

        # self.kappa = self.gamma/(np.exp((np.average(self.spread_list)-self.sigma_square*self.gamma*T)*self.gamma/2)-1)


        reservation_price = midprice - q*self.gamma*self.sigma_square*T
        spread = self.gamma*self.sigma_square*T + (2/self.gamma)*np.log(1+(self.gamma/self.kappa))

        ask_price = reservation_price - spread/2
        bid_price = reservation_price + spread/2

        return ask_price, bid_price
    
    def get_vol(self, vol, q, factor_fixed, q_x):
        '''
        因为数据原因，挂单量太小，导致gr5论文的库存风险修正数值小于1，所以暂时简单化处理
        '''
        ask_vol = vol
        bid_vol = vol
        if self.fix:
            if q > 0:
                bid_vol = bid_vol + q_x
            elif q < 0:
                ask_vol = ask_vol + q_x

            if factor_fixed == 2:
                ask_vol = ask_vol + 20
                bid_vol = bid_vol - 20 
            elif factor_fixed == 0:
                ask_vol = ask_vol - 20
                bid_vol = bid_vol + 20               
        
        return ask_vol, bid_vol


def date_file_as(withdrawal_duration=10, IRA=0.2, vol=20, u=10, fix=True, q_x=0):
    folder_path = 'D:/Lecture/FinalPaper/AS_codes_copy/data/data_by_date/'

    # 使用 glob 模块获取文件夹中所有的 CSV 文件名
    file_pattern = os.path.join(folder_path, '*.csv')
    file_list = glob.glob(file_pattern)

    for file in tqdm(file_list, desc='Running', unit='iteration', ncols=50):
        df = pd.read_csv(file, header=0, index_col=0)
        df.reset_index(drop=True, inplace=True)

        ## 生成ask_price, ask_vol, bid_price, bid_vol, q, cash初始化为0
        df.loc[:, 'cash'] = 0
        df.loc[:, 'q'] = 0
        df.loc[:, 'ask_price'] = 0
        df.loc[:, 'bid_price'] = 0
        df.loc[:, 'ask_vol'] = 0
        df.loc[:, 'bid_vol'] = 0

        Order_list = []
        TimeTrade_list = []
        # 生成每个order的class,并存在Order_list里面
        # 生成每个TimeTrade的class，并存在TimeTrade_list里面
        for index, row in df.iterrows():
            Order_class = Order(index, row['TradingTime'], row['ask_price'], row['ask_vol'], row['bid_price'], row['bid_vol'])
            Order_list.append(Order_class)
            TimeTrade_class = TimeTrade(index,row['TradingTime'], row['q'], row['cash'], row['BuyPrice01'],row['BuyPrice02'],row['BuyPrice03'],row['BuyPrice04'],row['BuyPrice05'],row['BuyVolume01'],row['BuyVolume02'],row['BuyVolume03'],row['BuyVolume04'],row['BuyVolume05'],
                        row['SellPrice01'],row['SellPrice02'],row['SellPrice03'],row['SellPrice04'],row['SellPrice05'],row['SellVolume01'],row['SellVolume02'],row['SellVolume03'],row['SellVolume04'],row['SellVolume05'],row['midprice'],row['spread'],row['T-t'],row['factor_fixed'])
            TimeTrade_list.append(TimeTrade_class)

        # from collections import Counter
        # print(Counter([i.factor_fixed for i in TimeTrade_list]))




        ## 取前300个做当日参数计算
        temp_day_parm = TimeTrade_list[:300].copy()

        ## 生成当日as model
        as_day = AS_model([midprice.midprice for midprice in temp_day_parm], [spread.spread for spread in temp_day_parm], IRA=IRA, q=temp_day_parm[0].q, T=temp_day_parm[0].T, fix=fix)

        # ## 先看买单这边：

        # ## 对于每一个TimeTrade，生成一个对应回测时长长度的temp_order_list，用来存储参与计算的挂单
        # ## 每次TimeTrade的循环，可以将第一个list的第一个元素去掉，然后append下一个，哦不，直接去掉时间最短的就行了
        # ## del list1[0]  list1.append(2)

        # ## 每次判断是否成交，先对temp_order_list进行价格排序，然后判断是否成交，然后去掉时间最短的一个
        temp_order_list = []
        for i in range(len(TimeTrade_list)):
            # 在每次TimeTrade的遍历的同时也会遍历每次的order list，此时更新order
            ## midprice, T, factor_fixed 是i时刻的，但q和cash是读取的是上一时刻的(用于as计算此时的price和vol)，然后在该时刻结束时将其更新
            if i == 0:
                ## 第一个时刻q，cash定为0
                TimeTrade_list[i].cash = 0
                TimeTrade_list[i].q = 0
            else:
                TimeTrade_list[i].cash = TimeTrade_list[i-1].cash
                TimeTrade_list[i].q = TimeTrade_list[i-1].q
            
            Order_list[i].ask_price, Order_list[i].bid_price = as_day.get_price_set(TimeTrade_list[i].midprice, q=TimeTrade_list[i].q, T=TimeTrade_list[i].T, factor_fixed=TimeTrade_list[i].factor_fixed, u=u)
            Order_list[i].ask_vol, Order_list[i].bid_vol = as_day.get_vol(vol=vol, q=TimeTrade_list[i].q, factor_fixed=TimeTrade_list[i].factor_fixed, q_x=q_x)
            
            if i < withdrawal_duration:
                temp_order_list.append(Order_list[i])
            else:
                # 找到index最小的元素并删除
                min_index_item = min(temp_order_list, key=lambda x: x.row)
                temp_order_list.remove(min_index_item)
                # append
                temp_order_list.append(Order_list[i])
        #         # 按照ask_price属性进行排序
                temp_order_list.sort(key=lambda x: x.ask_price, reverse=True)
                
                ## 从temp_order_list开始遍历，从最大ask_price开始，每个都得遍历SellPrice0-5,退出条件是要么价格不如，要么ask_vol=0
                for order in temp_order_list:
                    if order.ask_price >= TimeTrade_list[i].SellPrice01:
                        if order.ask_vol < TimeTrade_list[i].SellVolume01:
                            TimeTrade_list[i].SellVolume01 = TimeTrade_list[i].SellVolume01 - order.ask_vol
                            TimeTrade_list[i].q = TimeTrade_list[i].q + order.ask_vol
                            TimeTrade_list[i].cash = TimeTrade_list[i].cash - order.ask_vol*TimeTrade_list[i].SellPrice01
                            order.ask_vol = 0
                            continue
                        else:
                            order.ask_vol = order.ask_vol - TimeTrade_list[i].SellVolume01
                            TimeTrade_list[i].q = TimeTrade_list[i].q + TimeTrade_list[i].SellVolume01
                            TimeTrade_list[i].cash = TimeTrade_list[i].cash - TimeTrade_list[i].SellVolume01*TimeTrade_list[i].SellPrice01
                            TimeTrade_list[i].SellVolume01 = 0
                            if order.ask_price >= TimeTrade_list[i].SellPrice02:
                                if order.ask_vol < TimeTrade_list[i].SellVolume02:
                                    TimeTrade_list[i].SellVolume02 = TimeTrade_list[i].SellVolume02 - order.ask_vol
                                    TimeTrade_list[i].q = TimeTrade_list[i].q + order.ask_vol
                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash - order.ask_vol*TimeTrade_list[i].SellPrice02
                                    order.ask_vol = 0
                                    continue
                                else:
                                    order.ask_vol = order.ask_vol - TimeTrade_list[i].SellVolume02
                                    TimeTrade_list[i].q = TimeTrade_list[i].q + TimeTrade_list[i].SellVolume02
                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash - TimeTrade_list[i].SellVolume02*TimeTrade_list[i].SellPrice02
                                    TimeTrade_list[i].SellVolume02 = 0
                                    if order.ask_price >= TimeTrade_list[i].SellPrice03:
                                        if order.ask_vol < TimeTrade_list[i].SellVolume03:
                                            TimeTrade_list[i].SellVolume03 = TimeTrade_list[i].SellVolume03 - order.ask_vol
                                            TimeTrade_list[i].q = TimeTrade_list[i].q + order.ask_vol
                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash - order.ask_vol*TimeTrade_list[i].SellPrice03
                                            order.ask_vol = 0
                                            continue
                                        else:
                                            order.ask_vol = order.ask_vol - TimeTrade_list[i].SellVolume03
                                            TimeTrade_list[i].q = TimeTrade_list[i].q + TimeTrade_list[i].SellVolume03
                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash - TimeTrade_list[i].SellVolume03*TimeTrade_list[i].SellPrice03
                                            TimeTrade_list[i].SellVolume03 = 0
                                            if order.ask_price >= TimeTrade_list[i].SellPrice04:
                                                if order.ask_vol < TimeTrade_list[i].SellVolume04:
                                                    TimeTrade_list[i].SellVolume04 = TimeTrade_list[i].SellVolume04 - order.ask_vol
                                                    TimeTrade_list[i].q = TimeTrade_list[i].q + order.ask_vol
                                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash - order.ask_vol*TimeTrade_list[i].SellPrice04
                                                    order.ask_vol = 0
                                                    continue
                                                else:
                                                    order.ask_vol = order.ask_vol - TimeTrade_list[i].SellVolume04
                                                    TimeTrade_list[i].q = TimeTrade_list[i].q + TimeTrade_list[i].SellVolume04
                                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash - TimeTrade_list[i].SellVolume04*TimeTrade_list[i].SellPrice04
                                                    TimeTrade_list[i].SellVolume04 = 0
                                                    if order.ask_price >= TimeTrade_list[i].SellPrice05:
                                                        if order.ask_vol < TimeTrade_list[i].SellVolume05:
                                                            TimeTrade_list[i].SellVolume05 = TimeTrade_list[i].SellVolume05 - order.ask_vol
                                                            TimeTrade_list[i].q = TimeTrade_list[i].q + order.ask_vol
                                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash - order.ask_vol*TimeTrade_list[i].SellPrice05
                                                            order.ask_vol = 0
                                                            continue
                                                        else:
                                                            order.ask_vol = order.ask_vol - TimeTrade_list[i].SellVolume05
                                                            TimeTrade_list[i].q = TimeTrade_list[i].q + TimeTrade_list[i].SellVolume05
                                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash - TimeTrade_list[i].SellVolume05*TimeTrade_list[i].SellPrice05
                                                            TimeTrade_list[i].SellVolume05 = 0
                                                            break
                                                    else:
                                                        break
                                            else:
                                                break
                                    else:
                                        break
                            else:
                                break
                    else:
                        break
                

                # 卖单部分
                # 按照bid_price属性进行降序排序
                temp_order_list.sort(key=lambda x: x.bid_price, reverse=False)
                
                ## 从temp_order_list开始遍历，从最小bid_price开始，每个都得遍历BuyPrice0-5,退出条件是要么价格不如，要么bid_vol=0
                for order in temp_order_list:
                    if order.bid_price <= TimeTrade_list[i].BuyPrice01:
                        if order.bid_vol < TimeTrade_list[i].BuyVolume01:
                            TimeTrade_list[i].BuyVolume01 = TimeTrade_list[i].BuyVolume01 - order.bid_vol
                            TimeTrade_list[i].q = TimeTrade_list[i].q - order.bid_vol
                            TimeTrade_list[i].cash = TimeTrade_list[i].cash + order.bid_vol*TimeTrade_list[i].BuyPrice01
                            order.bid_vol = 0
                            continue
                        else:
                            order.bid_vol = order.bid_vol - TimeTrade_list[i].BuyVolume01
                            TimeTrade_list[i].q = TimeTrade_list[i].q - TimeTrade_list[i].BuyVolume01
                            TimeTrade_list[i].cash = TimeTrade_list[i].cash + TimeTrade_list[i].BuyPrice01*TimeTrade_list[i].BuyVolume01
                            TimeTrade_list[i].BuyVolume01 = 0
                            if order.bid_price <= TimeTrade_list[i].BuyPrice02:
                                if order.bid_vol < TimeTrade_list[i].BuyVolume02:
                                    TimeTrade_list[i].BuyVolume02 = TimeTrade_list[i].BuyVolume02 - order.bid_vol
                                    TimeTrade_list[i].q = TimeTrade_list[i].q - order.bid_vol
                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash + order.bid_vol*TimeTrade_list[i].BuyPrice02
                                    order.bid_vol = 0
                                    continue
                                else:
                                    order.bid_vol = order.bid_vol - TimeTrade_list[i].BuyVolume02
                                    TimeTrade_list[i].q = TimeTrade_list[i].q - TimeTrade_list[i].BuyVolume02
                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash + TimeTrade_list[i].BuyPrice02*TimeTrade_list[i].BuyVolume02
                                    TimeTrade_list[i].BuyVolume02 = 0
                                    if order.bid_price <= TimeTrade_list[i].BuyPrice03:
                                        if order.bid_vol < TimeTrade_list[i].BuyVolume03:
                                            TimeTrade_list[i].BuyVolume03 = TimeTrade_list[i].BuyVolume03 - order.bid_vol
                                            TimeTrade_list[i].q = TimeTrade_list[i].q - order.bid_vol
                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash + order.bid_vol*TimeTrade_list[i].BuyPrice03
                                            order.bid_vol = 0
                                            continue
                                        else:
                                            order.bid_vol = order.bid_vol - TimeTrade_list[i].BuyVolume03
                                            TimeTrade_list[i].q = TimeTrade_list[i].q - TimeTrade_list[i].BuyVolume03
                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash + TimeTrade_list[i].BuyPrice03*TimeTrade_list[i].BuyVolume03
                                            TimeTrade_list[i].BuyVolume03 = 0
                                            if order.bid_price <= TimeTrade_list[i].BuyPrice04:
                                                if order.bid_vol < TimeTrade_list[i].BuyVolume04:
                                                    TimeTrade_list[i].BuyVolume04 = TimeTrade_list[i].BuyVolume04 - order.bid_vol
                                                    TimeTrade_list[i].q = TimeTrade_list[i].q - order.bid_vol
                                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash + order.bid_vol*TimeTrade_list[i].BuyPrice04
                                                    order.bid_vol = 0
                                                    continue
                                                else:
                                                    order.bid_vol = order.bid_vol - TimeTrade_list[i].BuyVolume04
                                                    TimeTrade_list[i].q = TimeTrade_list[i].q - TimeTrade_list[i].BuyVolume04
                                                    TimeTrade_list[i].cash = TimeTrade_list[i].cash + TimeTrade_list[i].BuyPrice04*TimeTrade_list[i].BuyVolume04
                                                    TimeTrade_list[i].BuyVolume04 = 0
                                                    if order.bid_price <= TimeTrade_list[i].BuyPrice05:
                                                        if order.bid_vol < TimeTrade_list[i].BuyVolume05:
                                                            TimeTrade_list[i].BuyVolume05 = TimeTrade_list[i].BuyVolume05 - order.bid_vol
                                                            TimeTrade_list[i].q = TimeTrade_list[i].q - order.bid_vol
                                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash + order.bid_vol*TimeTrade_list[i].BuyPrice05
                                                            order.bid_vol = 0
                                                            continue
                                                        else:
                                                            order.bid_vol = order.bid_vol - TimeTrade_list[i].BuyVolume05
                                                            TimeTrade_list[i].q = TimeTrade_list[i].q - TimeTrade_list[i].BuyVolume05
                                                            TimeTrade_list[i].cash = TimeTrade_list[i].cash + TimeTrade_list[i].BuyPrice05*TimeTrade_list[i].BuyVolume05
                                                            TimeTrade_list[i].BuyVolume05 = 0
                                                            break
                                                    else:
                                                        break
                                            else:
                                                break
                                    else:
                                        break
                            else:
                                break
                    else:
                        break




        # Order_list.sort(key=lambda x: x.row, reverse=False)
        # TimeTrade_list.sort(key=lambda x: x.row, reverse=False)

        df_order = pd.DataFrame([(order.TradingTime, order.ask_price, order.ask_vol, order.bid_price, order.bid_vol) for order in Order_list], columns=['TradingTime', 'ask_price', 'ask_vol_after', 'bid_price', 'bid_vol_after'])

        df_time = pd.DataFrame([(t.TradingTime,t.q, t.cash, t.BuyVolume01,t.BuyVolume02,t.BuyVolume03,t.BuyVolume04,t.BuyVolume05,
                                t.SellVolume01,t.SellVolume02,t.SellVolume03,t.SellVolume04,t.SellVolume05) for t in TimeTrade_list], columns=['TradingTime','q','cash','BuyVolume01_after','BuyVolume02_after','BuyVolume03_after','BuyVolume04_after','BuyVolume05_after',
                                'SellVolume01_after','SellVolume02_after','SellVolume03_after','SellVolume04_after','SellVolume05_after'])

        print(df['factor_fixed'].value_counts())

        df = df[['TradingTime', 'TradingDate', 'BuyVolume01', 'BuyVolume02','BuyVolume03', 'BuyVolume04', 'BuyVolume05', 
                    'SellVolume01','SellVolume02', 'SellVolume03', 'SellVolume04', 'SellVolume05',
                    'BuyPrice01', 'BuyPrice02','BuyPrice03', 'BuyPrice04', 'BuyPrice05', 
                    'SellPrice01','SellPrice02', 'SellPrice03', 'SellPrice04', 'SellPrice05',
                    'midprice','spread']]

        merged = pd.concat([df.set_index('TradingTime'),df_time.set_index('TradingTime'), df_order.set_index('TradingTime')], axis=1).reset_index()
        
        merged['total_asset'] = merged['cash'] + merged['q']*merged['midprice']

                # 从文件名中提取日期信息
        date = re.search(r'(\d{8})', file).group(0)

        if fix:
            new_file_name = f"D:/Lecture/FinalPaper/AS_codes_copy/data/fixed_date_final/{date}.csv"
        else:
            new_file_name = f"D:/Lecture/FinalPaper/AS_codes_copy/data/unfixed_date_final/{date}.csv"
            # 将处理后的 DataFrame 写入新文件
        merged.to_csv(new_file_name, index=False)

        # print(f"{date}: min q{min(merged['q'])},max q{max(merged['q'])}, min cash{min(merged['cash'])},max q{max(merged['cash'])}")
        # if date == 20230201:
        print(f"{date}:final asset{merged['total_asset'].tail(1)}")
        print(f"{date} finished")
        
        # print(min(df_time['q']), max(df_time['q']), min(df_time['cash']),max(df_time['cash']))


date_file_as(withdrawal_duration=20,IRA=0.000001,vol=5,u=0, fix=False, q_x=0)
date_file_as(withdrawal_duration=20,IRA=0.000001,vol=5,u=10, fix=True, q_x=0)


