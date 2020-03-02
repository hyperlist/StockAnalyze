import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

def BI_Simple_func(row):
    pos = row[row == 1].count()
    neg = row[row == 0].count()
    return (pos-neg)/(pos+neg)

def BI_func(row):
    pos = row[row == 1].count()
    neg = row[row == 0].count()
    bi = np.log(1.0 * (1+pos) / (1+neg))
    return bi
    
#date,open,high,low,close,volume
#日期,股票代码,名称,收盘价,最高价,最低价,开盘价,前收盘,涨跌额,涨跌幅,成交量,成交金额
id = '000001'
df = pd.read_csv('results/000001.csv', parse_dates=['created_time'])
grouped = df['polarity'].groupby(df.created_time.dt.date)

BI_Simple_index = grouped.apply(BI_Simple_func)
BI_index = grouped.apply(BI_func)

sentiment_idx = pd.concat([BI_index.rename('BI'), BI_Simple_index.rename('BI_Simple')], axis=1)

quotes = pd.read_csv("data/000001.csv", encoding = "gbk", parse_dates=['日期'])
quotes.set_index('日期', inplace=True)
quotes.drop(['股票代码','名称','开盘价','前收盘','涨跌额','涨跌幅','成交量','成交金额'], axis=1, inplace=True)  

sentiment_idx.index = pd.to_datetime(sentiment_idx.index)
merged = pd.merge(sentiment_idx, quotes, how='inner', left_index=True, right_index=True)

merged. (method='ffill', inplace=True)
merged['BI_MA'] = merged['BI'].rolling(window=10, center=False).mean()
merged['BI_Simple_MA'] = merged['BI_Simple'].rolling(window=10, center=False).mean()

merged.to_csv(os.path.join('./results/',id+'-idx.csv'))
print(merged)
'''
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df.index, df['BI_MA'], color='#1F77B4', linestyle=':')
ax2.plot(df.index, df['收盘价'], color='#4B73B1')
ax1.set_xlabel('日期')
ax1.set_ylabel('BI指标')
ax2.set_ylabel('上证指数')

plt.show()
'''