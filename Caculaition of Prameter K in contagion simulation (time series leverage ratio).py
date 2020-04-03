# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:02:38 2020

@author: yingl WANG
"""

import codecs
import networkx as nx
import networkx.algorithms.isomorphism as iso
import os.path
import numpy as np
import collections as cl
import matplotlib
from os.path import join
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
import math
#import platform

leverage_ratio_mean= [0.6076797117065776, 0.6021026207702114, 0.6038172553855191, 0.5983193300070662, 0.6086742695286927, 0.598158305875809, 0.6103454538328954, 0.6331384438595948, 0.6404109844607623, 0.666117873995153, 0.6770589334943751, 0.6986644641375865, 0.6599040679051866, 0.6379325292481253, 0.6303155440906371, 0.6068271538507617, 0.6298368401846184, 0.6453198614787364, 0.6208987222384853, 0.6314235053400848, 0.7081809611198399, 0.6615836515721076, 0.6206839118227941, 0.6153751293214306, 0.6163253482099861, 0.6046501627041009, 0.6043030523874803, 0.6032696779557967, 0.5897780753158531, 0.5977963566725081, 0.6030568844457008, 0.5904695214557505, 0.581916748803288, 0.6124483053826234, 0.5748372856430684, 0.5737915112398996, 0.5741145145600816, 0.5719725328484, 0.5716617187925458, 0.5665687673549136, 0.5678490352346388, 0.5684169416277204, 0.5683872751001415, 0.5688140355776847, 0.5650072618353632, 0.5656566796881977, 0.5621742628207272, 0.5591675972367488, 0.557013155430675, 0.5571124212520364, 0.5560858221650133, 0.5599700717826255, 0.558969868886333, 0.5610119432552954, 0.560435837659287, 0.5567066585224107, 0.5563791549283111, 0.5549837006083131, 0.5550266459184884, 0.5552154034326474, 0.5536671984248862, 0.5555148831279054, 0.5504813364983077]

tags = []
for y in range(7, 12):
    for m in range(1, 13):
        tags.append('%02d%02d'%(y, m))
tags.append('1201')
tags.append('1202')
tags.append('1203')

#df= pd.DataFrame()
frames = []
i=0
for tag in tags:
    df1=pd.read_table('C:/network/data/nodes/network%s.txt' %tag)
    df1['leverage_ratio_mean']=leverage_ratio_mean[i]
#    df1['leverage_ratio_mean']=0.5956155107938541
    i=i+1
    frames.append(df1)
    
df1 = pd.concat(frames)    

print(df1.shape)
print(df1.head())
print (df1.columns)
print(df1.info())


# get rid of just created columns
df1.drop(['total_bad_amount', 'total_expire_amount','confer', 'loan', 'region', 'market', 'bad_rate',
       'expire_rate'], axis=1, inplace=True)
    
###删去空值#####
df1.dropna(axis=0, how='any', inplace=True)
df1=df1[~df1['asset'].isin([0])]
df1=df1[~df1['liability'].isin([0])]
###新增一列
df1['liability_ratio']=df1['liability']/df1['asset']






df1.describe()


def func(x1, x2, p): 
    """ 数据拟合所用的函数: A*sin(2*pi*k*x + theta) """
    k = p
    return  1/(1+np.exp(-k*(x1-x2)))

#    return  1/(-k*x)

def residuals(p, y, x1, x2): 
    """ 实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数 """
    #print (y - func(x, p))
    return y - func(x1,x2, p)

# 生成输入数据X和输出输出Y
x1=df1['liability_ratio'].values
x2=df1['leverage_ratio_mean'].values
#print (x)
#x = np.linspace(-2*np.pi, 0, 100)       # x
#k= 1.5         # 真实数据的函数参数
#y0 = func(x, k)             # 真实数据
y0 =df1['risk'].values            # 真实数据

y1 = y0 + 0.026 * np.random.randn(len(x1))   # 加入噪声之后的实验数据

# 需要求解的参数的初值，注意参数的顺序，要和func保持一致
p0 = 3

# 调用leastsq进行数据拟合, residuals为计算误差的函数
# p0为拟合参数的初始值
# args为需要拟合的实验数据，也就是，residuals误差函数
# 除了P之外的其他参数都打包到args中
args=(y1, x1, x2)
print (args)
print (residuals)
plsq = leastsq(residuals, p0, args=(y1, x1, x2)) 
print (plsq)

# 除了初始值之外，还调用了args参数，用于指定residuals中使用到的其他参数
# （直线拟合时直接使用了X,Y的全局变量）,同样也返回一个元组，第一个元素为拟合后的参数数组；

# 这里将 (y1, x)传递给args参数。Leastsq()会将这两个额外的参数传递给residuals()。
# 因此residuals()有三个参数，p是正弦函数的参数，y和x是表示实验数据的数组。

# 拟合的参数
print("拟合参数", plsq[0])
# 拟合参数 [ 10.6359371    0.3397994    0.50520845]
# 发现跟实际值有差距，但是从下面的拟合图形来看还不错，说明结果还是 堪用 的。

# 顺便画张图看看

