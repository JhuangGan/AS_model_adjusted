import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator


# 设置绘图风格
def sixplot(df_day, col, title, y_label, save,x_label='Time'):
    plt.style.use('seaborn-darkgrid')

    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制 midprice 折线
    ax.plot(df_day['TradingTime'], df_day[col],alpha=0.6, color='blue', label=col)


    # 添加图例
    # ax.legend()

    # 设置标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # # 调整日期坐标轴的显示间隔为一周
    # ax.xaxis.set_major_locator(DayLocator(interval=21))
    ax.xaxis.set_major_formatter(DateFormatter('%H-%M'))  # 设置日期格式

    # 自动调整日期标签，避免重叠
    # fig.autofmt_xdate()


    plt.savefig('D:/Lecture/FinalPaper/RUCThesis-master/RUCThesis-master/figures/'+save+'.png')
    # # 显示图形
    plt.close()


df_day = pd.read_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/unfixed_date_final/20230216.csv')
df_day['TradingTime'] = pd.to_datetime(df_day['TradingTime'])
# print(df_day)


## 每个时刻的成交量

df_day['bid_trade_volume'] = df_day['BuyVolume01'] + df_day['BuyVolume02'] + df_day['BuyVolume03'] + df_day['BuyVolume04'] + df_day['BuyVolume05'] - (df_day['BuyVolume01_after'] + df_day['BuyVolume02_after'] + df_day['BuyVolume03_after'] + df_day['BuyVolume04_after'] + df_day['BuyVolume05_after'] )
df_day['ask_trade_volume'] = df_day['SellVolume01'] + df_day['SellVolume02'] + df_day['SellVolume03'] + df_day['SellVolume04'] + df_day['SellVolume05'] - (df_day['SellVolume01_after'] + df_day['SellVolume02_after'] + df_day['SellVolume03_after'] + df_day['SellVolume04_after'] + df_day['SellVolume05_after'] )

df_day['bid_cancel_volume'] = df_day['BuyVolume01_after'] + df_day['BuyVolume02_after'] + df_day['BuyVolume03_after'] + df_day['BuyVolume04_after'] + df_day['BuyVolume05_after']
df_day['ask_cancel_volume'] = df_day['SellVolume01_after'] + df_day['SellVolume02_after'] + df_day['SellVolume03_after'] + df_day['SellVolume04_after'] + df_day['SellVolume05_after']

df_day['bid_volume'] = df_day['BuyVolume01'] + df_day['BuyVolume02'] + df_day['BuyVolume03'] + df_day['BuyVolume04'] + df_day['BuyVolume05']
df_day['ask_volume'] = df_day['SellVolume01'] + df_day['SellVolume02'] + df_day['SellVolume03'] + df_day['SellVolume04'] + df_day['SellVolume05']



df_day['cum_ask_cancel_day'] = df_day['ask_cancel_volume'].cumsum()  ## 单日ask累积撤单量
df_day['cum_bid_cancel_day'] = df_day['bid_cancel_volume'].cumsum()  ## 单日ask累积撤单量


# ## 计算撤单比率 = 累积撤单量/累积下单量
# # 计算累积下单量

df_day['cum_ask_volume_day'] = df_day['ask_volume'].cumsum()  ## 单日ask累积挂单量
df_day['cum_bid_volume_day'] = df_day['bid_volume'].cumsum()  ## 单日bid累积挂单量

df_day['ask_cancel_rate_day'] = df_day['cum_ask_cancel_day']/df_day['cum_ask_volume_day']
df_day['bid_cancel_rate_day'] = df_day['cum_bid_cancel_day']/df_day['cum_bid_volume_day']




sixplot(df_day, col='midprice', title='20230215 Price Trend',y_label='MidPrice', save='61midpricetrend',x_label='Time')
sixplot(df_day, col='ask_trade_volume', title='20230215 Executed Ask Volume',y_label='Executed Ask Volume', save='61ask_trade_Volume')
sixplot(df_day, col='bid_trade_volume', title='20230215 Executed Bid Volume',y_label='Executed Bid Volume', save='61bid_trade_Volume')
sixplot(df_day, col='ask_cancel_volume', title='20230215 Ask Cancel Volume',y_label='Cancel Ask Volume', save='61ask_cancel_Volume')
sixplot(df_day, col='bid_cancel_volume', title='20230215 Bid Cancel Volume',y_label='Cancel Bid Volume', save='61bid_cancel_Volume')
sixplot(df_day, col='bid_volume', title='20230215 Bid Volume',y_label='Bid Volume', save='61bid_Volume')
sixplot(df_day, col='ask_volume', title='20230215 Ask Volume',y_label='Ask Volume', save='61ask_Volume')
# sixplot(df_day, col='ask_cancel_rate_day', title='20230215 Ask Cancel Rate',y_label='Ask Cancel Rate', save='61ask_cancel_rate')
# sixplot(df_day, col='bid_cancel_rate_day', title='20230215 Bid Cancel Rate',y_label='Bid Cancel Rate', save='61bid_cancel_rate')
sixplot(df_day, col='q', title='20230215 inventory',y_label='Inventory Volume', save='61q')
sixplot(df_day, col='cash', title='20230215 Cash',y_label='Cash', save='61cash')
sixplot(df_day, col='total_asset', title='20230215 Total Asset',y_label='Total Asset', save='61asset')

df_day['ask-bid-ex']=df_day['ask_trade_volume'] - df_day['bid_trade_volume']
df_day['ask-bid-cancel']=df_day['ask_cancel_volume'] - df_day['bid_cancel_volume']

sixplot(df_day, col='ask-bid-ex', title='Difference Between Executed Ask and Bid Volume ',y_label='Volume', save='61ask-bid-ex')










df_day = pd.read_csv('D:/Lecture/FinalPaper/AS_codes_copy/data/fixed_date_final/20230215.csv')
df_day['TradingTime'] = pd.to_datetime(df_day['TradingTime'])
# print(df_day)


## 每个时刻的成交量

df_day['bid_trade_volume'] = df_day['BuyVolume01'] + df_day['BuyVolume02'] + df_day['BuyVolume03'] + df_day['BuyVolume04'] + df_day['BuyVolume05'] - (df_day['BuyVolume01_after'] + df_day['BuyVolume02_after'] + df_day['BuyVolume03_after'] + df_day['BuyVolume04_after'] + df_day['BuyVolume05_after'] )
df_day['ask_trade_volume'] = df_day['SellVolume01'] + df_day['SellVolume02'] + df_day['SellVolume03'] + df_day['SellVolume04'] + df_day['SellVolume05'] - (df_day['SellVolume01_after'] + df_day['SellVolume02_after'] + df_day['SellVolume03_after'] + df_day['SellVolume04_after'] + df_day['SellVolume05_after'] )

df_day['bid_cancel_volume'] = df_day['BuyVolume01_after'] + df_day['BuyVolume02_after'] + df_day['BuyVolume03_after'] + df_day['BuyVolume04_after'] + df_day['BuyVolume05_after']
df_day['ask_cancel_volume'] = df_day['SellVolume01_after'] + df_day['SellVolume02_after'] + df_day['SellVolume03_after'] + df_day['SellVolume04_after'] + df_day['SellVolume05_after']

df_day['bid_volume'] = df_day['BuyVolume01'] + df_day['BuyVolume02'] + df_day['BuyVolume03'] + df_day['BuyVolume04'] + df_day['BuyVolume05']
df_day['ask_volume'] = df_day['SellVolume01'] + df_day['SellVolume02'] + df_day['SellVolume03'] + df_day['SellVolume04'] + df_day['SellVolume05']

df_day['cum_ask_cancel_day'] = df_day['ask_cancel_volume'].cumsum()  ## 单日ask累积撤单量
df_day['cum_bid_cancel_day'] = df_day['bid_cancel_volume'].cumsum()  ## 单日ask累积撤单量

# ## 计算撤单比率 = 累积撤单量/累积下单量
# # 计算累积下单量

df_day['cum_ask_volume_day'] = df_day['ask_volume'].cumsum()  ## 单日ask累积挂单量
df_day['cum_bid_volume_day'] = df_day['bid_volume'].cumsum()  ## 单日bid累积挂单量

df_day['ask_cancel_rate_day'] = df_day['cum_ask_cancel_day']/df_day['cum_ask_volume_day']
df_day['bid_cancel_rate_day'] = df_day['cum_bid_cancel_day']/df_day['cum_bid_volume_day']


sixplot(df_day, col='midprice', title='20230215 Price Trend',y_label='MidPrice', save='61midpricetrend_adjusted',x_label='Time')
sixplot(df_day, col='ask_trade_volume', title='20230215 Executed Ask Volume',y_label='Executed Ask Volume', save='61ask_trade_vol_adjusted')
sixplot(df_day, col='bid_trade_volume', title='20230215 Executed Bid Volume',y_label='Executed Bid Volume', save='61bid_trade_vol_adjusted')
sixplot(df_day, col='ask_cancel_volume', title='20230215 Ask Cancel Volume',y_label='Cancel Ask Volume', save='61ask_cancel_vol_adjusted')
sixplot(df_day, col='bid_cancel_volume', title='20230215 Bid Cancel Volume',y_label='Cancel Bid Volume', save='61bid_cancel_vol_adjusted')
sixplot(df_day, col='bid_volume', title='20230215 Bid Volume',y_label='Bid Volume', save='61bid_vol_adjusted')
sixplot(df_day, col='ask_volume', title='20230215 Ask Volume',y_label='Ask Volume', save='61ask_vol_adjusted')
# sixplot(df_day, col='ask_cancel_rate_day', title='20230215 Ask Cancel Rate',y_label='Ask Cancel Rate', save='61ask_cancel_rate_adjusted')
# sixplot(df_day, col='bid_cancel_rate_day', title='20230215 Bid Cancel Rate',y_label='Bid Cancel Rate', save='61bid_cancel_rate_adjusted')
sixplot(df_day, col='q', title='20230215 inventory',y_label='Inventory Volume', save='61q_adjusted')
sixplot(df_day, col='cash', title='20230215 Cash',y_label='Cash', save='61cash_adjusted')
sixplot(df_day, col='total_asset', title='20230215 Total Asset',y_label='Total Asset', save='61asset_adjusted')
