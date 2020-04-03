# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:26:14 2017

@author: xc
"""

# in order to revise p ratio, need to change p in line 195, 233, 263 and 291
# -*- coding: utf-8 -*-
import networkx as nx
import networkx.algorithms.isomorphism as iso
import os.path
import numpy as np
import collections as cl
import matplotlib
from os.path import join
import random
import math
from math import log, exp, sqrt
from scipy import stats, array, linalg, dot
from collections import deque
import matplotlib.pyplot as plt
#import platform

import os
import time
from multiprocessing import Pool
n_threads = 8

p=0.01
k=1.18
N=10000
leverage_ratio_mean= [0.6076797117065776, 0.6021026207702114, 0.6038172553855191, 0.5983193300070662, 0.6086742695286927, 0.598158305875809, 0.6103454538328954, 0.6331384438595948, 0.6404109844607623, 0.666117873995153, 0.6770589334943751, 0.6986644641375865, 0.6599040679051866, 0.6379325292481253, 0.6303155440906371, 0.6068271538507617, 0.6298368401846184, 0.6453198614787364, 0.6208987222384853, 0.6314235053400848, 0.7081809611198399, 0.6615836515721076, 0.6206839118227941, 0.6153751293214306, 0.6163253482099861, 0.6046501627041009, 0.6043030523874803, 0.6032696779557967, 0.5897780753158531, 0.5977963566725081, 0.6030568844457008, 0.5904695214557505, 0.581916748803288, 0.6124483053826234, 0.5748372856430684, 0.5737915112398996, 0.5741145145600816, 0.5719725328484, 0.5716617187925458, 0.5665687673549136, 0.5678490352346388, 0.5684169416277204, 0.5683872751001415, 0.5688140355776847, 0.5650072618353632, 0.5656566796881977, 0.5621742628207272, 0.5591675972367488, 0.557013155430675, 0.5571124212520364, 0.5560858221650133, 0.5599700717826255, 0.558969868886333, 0.5610119432552954, 0.560435837659287, 0.5567066585224107, 0.5563791549283111, 0.5549837006083131, 0.5550266459184884, 0.5552154034326474, 0.5536671984248862, 0.5555148831279054, 0.5504813364983077]

dir_data = './data'

var = ['bad_amount', 'expire_amount', 'asset',
       'liability', 'confer', 'loan', 'region', 'risk',
       'market', 'bad_rate', 'expire_rate']
numeric_var = ['asset', 'liability', 'confer', 'loan']
categorical_var = ['region', 'market']
other_var = ['risk', 'bad_rate', 'expire_rate']

tags = []
for y in range(7, 12):
    for m in range(1, 13):
        tags.append('%02d%02d'%(y, m))
tags.append('1201')
tags.append('1202')
tags.append('1203')

full_tags = []
sigma_dict={}
_i=0
for tag in tags:
    sigma_dict[tag]=leverage_ratio_mean[_i]
    _i=_i+1
    full_tags.append('20' + tag)

regions=[11,12,13,14,15,21,22,23,31,32,33,34,35,36,37,41,42,
         43,44,45,46,50,51,52,53,54,61,62,63,64,65,66,71,81,82]
    
LEHMAN_IDX =20 #闆锋浖鍏勫紵鐮翠骇鏈堜唤
FOURTRI_IDX = 22 #鍥涗竾浜块�鍑烘湀浠�
FOURTRI_IDX_END = 47


def get_guarantee_network(tag):
    sigma=sigma_dict[tag]
    
    #print(tag)
    g = nx.DiGraph()
    file1 = os.path.join(dir_data, 'edges', 'edge%s.txt'%tag)
    fin1 = open(file1, 'r')
    fin1.readline()
    for line in fin1:
        tmp_list = line.split()
        if len(tmp_list) == 4:
            (customer, guarantor, balance_, num_) = line.split()
            g.add_edge(guarantor, customer, balance = float(balance_),
                                number = int(num_))
    fin1.close()      
    #Add informations of nodes
    file2 = os.path.join(dir_data, 'nodes', 'network%s.txt'%tag)
    fin2 = open(file2, 'r')
    nodes = set(g.nodes())
    nodes_has_info = set()
    var_name = ['', 'bad_amount', 'expire_amount', 'asset',
                'liability', 'confer', 'loan', 'region', 'risk',
                'market', 'bad_rate', 'expire_rate']
    fin2.readline()
    for line in fin2:
        x = line.split()
        if x[0] in nodes:
            nodes_has_info.add(x[0])
            for i in range(1, 12):
                g.nodes[x[0]][var_name[i]] = float(x[i])
    fin2.close()
   #填充缺失值
    node_list=[]
    lnasset_list=[]
    lnliability_list=[]
    lnloan_list=[]
    
    for node in nodes_has_info:
        if g.nodes[node]['asset']>0:
           node_list.append(node)
           lnasset_list.append(np.log(g.nodes[node]['asset']+1))
           lnliability_list.append(np.log(g.nodes[node]['liability']+1))
           lnloan_list.append(np.log(g.nodes[node]['loan']+1))
   #compute the coefficiences of liner regressions
    x = np.vstack([np.array(lnasset_list), np.ones(len(lnasset_list))]).T
    b1, a1 = np.linalg.lstsq(x, np.array(lnliability_list),rcond=None)[0]  
    b2, a2 = np.linalg.lstsq(x, np.array(lnloan_list),rcond=None)[0]   
    #compute the mean and sd of log(asset)
    narray=np.array(lnasset_list)
    sum1=narray.sum()
    narray2=narray*narray
    sum2=narray2.sum()
    mean_log_asset=sum1/len(lnasset_list)
    sd_log_asset=sqrt(float(sum2)/len(lnasset_list)-mean_log_asset**2)   
    sd = 0.8972
    tmp = np.random.randn(500000)
    i = 0    
    #fill the missing value 
    for node in g.nodes():
        if node not in nodes_has_info or g.nodes[node]['asset'] < 1e-6:
           log_asset = mean_log_asset + sd_log_asset * tmp[i]
           i += 1
           g.nodes[node]['asset'] = np.exp(log_asset)
           g.nodes[node]['liability'] = np.exp(a1 + b1 * log_asset)
           g.nodes[node]['loan'] = np.exp(a2 + b2 * log_asset + sd * tmp[i])
           i += 1
    del tmp
    return (g, nodes_has_info, sigma)


def prob_contagion(asset, liab, bad_amount, k, sigma):  #感染概率
    ratio = (liab + bad_amount) / asset
    x = ratio - sigma
    x *= k
    return 1 / (1 + np.exp(-x))


def process(g, p, k, sigma):
    #初始化状态
    num_bad = 0
    queue = deque()
    node_list=[]
    node_loan=[]
    #loansum=0  #to denote the sum of loan in the guarantee network
   #sorted nodes by indegree from low to high
    #(g, nodes_has_info)=get_guarantee_network(tag)
    for node in g.nodes():
        node_list.append(node)
        node_loan.append(g.nodes[node]['loan'])
    node_loan_dic=dict(zip(node_list,node_loan))     
    sortednodes=sorted(node_loan_dic)
    for node in sortednodes:
        g.nodes[node]['bad_danbao'] = 0
        #loansum+=g.nodes[node]['loan']
    #choose the top p ratio nodes with higher in degree   
    j=0
    lown=(1-p)*g.number_of_nodes() #lown denotes the top (1-p) tatio nodes with low degree
    for node in sortednodes:
        j=j+1
        if j <= lown:
            g.nodes[node]['state'] = 0
        else:
            g.nodes[node]['state'] = 1
            num_bad += 1
            for nd in g.predecessors(node):
                queue.append(nd)
                g.nodes[nd]['bad_danbao'] += g[nd][node]['balance']
    p0 = num_bad / float(g.number_of_nodes())
    n0 = num_bad
    

    totalLoan = 0 #to denote the badloan
    while(len(queue)):
        node = queue.popleft()
        if g.nodes[node]['state'] == 0:
            bad_amount = g.nodes [node]['bad_danbao']
            liab = g.nodes [node]['liability']
            asset = g.nodes [node]['asset']
            if random.random() < prob_contagion(asset, liab, bad_amount, k, sigma):
                totalLoan += bad_amount
                g.nodes [node]['state'] = 1
                num_bad += 1
                for nd in g.predecessors(node):
                    queue.append(nd)
                    g.nodes [nd]['bad_danbao'] += g[nd][node]['balance']
                    
    p1 = num_bad / float(g.number_of_nodes())
    n1 = num_bad
    #totalLoan_ratio=totalLoan/float(loansum)
    #fout.write('\n' + fc.l2s([p, k, p0, p1, n0, n1, totalLoan]))
    #print (p0, n0, p1, n1, totalLoan)
    return (p0, n0, p1, n1, totalLoan)
   # return (p0, n0, p1, n1, totalLoan,totalLoan_ratio)


def onemonth_simulation (g, tag, p, k, sigma, N):
    
    p0=0
    n0=0
    p1=0
    n1=0
    totalLoan=0
   # totalLoan_ratio=0
    
    #(g, nodes_has_info)=get_guarantee_network(tag)
    i = 0
    for num in range(N):
        print(tag+' No. '+str(i))
        i += 1
        (pp0, nn0, pp1, nn1, totalLoan0)=process(g, p, k, sigma)
        p0+=pp0
        n0+=nn0
        p1+=pp1
        n1+=nn1
        totalLoan+=totalLoan0
        #totalLoan_ratio+=totalLoan_ratio0
     
    mean_p0= p0/float(N)
    mean_n0=n0/float(N) 
    mean_p1=p1/float(N) 
    mean_n1=n1/float(N) 
    mean_totalLoan=totalLoan/float(N)
   # mean_totalLoan_ratio=totalLoan_ratio/float(N)
    
    
    #print (mean_p0,mean_n0,mean_p1,mean_n1,mean_totalLoan)
    return (mean_p0,mean_n0,mean_p1,mean_n1,mean_totalLoan)
    #return (mean_p0,mean_n0,mean_p1,mean_n1,mean_totalLoan, mean_totalLoan_ratio) 
    

mean_p0_list=[]
mean_n0_list=[]
mean_p1_list=[]
mean_n1_list=[]
mean_totalLoan_list=[]
mean_totalLoan_ratio_list=[]
      
# for tag in tags:
#     (g, nodes_has_info)=get_guarantee_network(tag) 
#     (mean_p0,mean_n0,mean_p1,mean_n1,mean_totalLoan) = onemonth_simulation (g, tag, p, k, sigma, N)
#     mean_p0_list.append(mean_p0)
#     mean_n0_list.append(mean_n0)
#     mean_p1_list.append(mean_p1)
#     mean_n1_list.append(mean_n1)
#     mean_totalLoan_list.append(mean_totalLoan)


def oper_tags(tag):
    (g,nodes_has_info,sigma)=get_guarantee_network(tag)
    #g1=g.subgraph(nodes_has_info)
    return onemonth_simulation (g, tag, p, k, sigma, N)    

def f(x):
    return x*x   


if __name__ == "__main__":
    #startTime = time.time()
    #print('start')
    #pool = Pool(processes=4)
    with Pool(n_threads) as pool:
        print(pool.map(f,range(10)))
    print('start_pool')
    #pool.close()
    #pool.close()
    pool = Pool(processes=n_threads)
    result_list = pool.map(oper_tags,tags)
    pool.close()
    pool.join()


for item in result_list:
    mean_p0_list.append(item[0])
    mean_n0_list.append(item[1])
    mean_p1_list.append(item[2])
    mean_n1_list.append(item[3])  
    mean_totalLoan_list.append(item[4])
# =============================================================================
# if __name__ == "__main__":
#     #startTime = time.time()
#     pool = Pool(n_threads)
#     result_list = pool.map(oper_tags,tags)
#     pool.close()
#     pool.join()
# 
#     for item in result_list:
#         mean_p0_list.append(item[0])
#         mean_n0_list.append(item[1])
#         mean_p1_list.append(item[2])
#         mean_n1_list.append(item[3])
#         mean_totalLoan_list.append(item[4])
#         #mean_totalLoan_ratio_list.append(item[5])
# =============================================================================


   # mean_totalLoan_ratio_list.append(mean_totalLoan_ratio)
mean_n0_list[57]=float(mean_n0_list[56]+mean_n0_list[58])/2
mean_p1_list[57]=float( mean_p1_list[56]+ mean_p1_list[58])/2
mean_n1_list[57]=float(mean_n1_list[56]+mean_n1_list[58])/2
mean_totalLoan_list[57]=float(mean_totalLoan_list[56]+mean_totalLoan_list[58])/2
print (mean_p0_list)
print (mean_n0_list)
print (mean_p1_list)
print (mean_n1_list)
print (mean_totalLoan_list)
#print (mean_totalLoan_ratio_list)

fw=open('loan_attack_%s.csv'%p,'a')
fw.write(str('mean_p0_list')+',')
for i in mean_p0_list[:-1]:
    fw.write(str(i)+',')
fw.write(str(mean_p0_list[-1])+'\n') 

fw.write(str('mean_n0_list')+',')
for i in mean_n0_list[:-1]:
    fw.write(str(i)+',')
fw.write(str(mean_n0_list[-1])+'\n') 

fw.write(str('mean_p1_list')+',')
for i in mean_p1_list[:-1]:
    fw.write(str(i)+',')
fw.write(str(mean_p1_list[-1])+'\n') 

fw.write(str('mean_n1_list')+',')
for i in mean_n1_list[:-1]:
    fw.write(str(i)+',')
fw.write(str(mean_n1_list[-1])+'\n')  

fw.write(str('mean_totalLoan_list')+',')
for i in mean_totalLoan_list[:-1]:
    fw.write(str(i)+',')
fw.write(str(mean_totalLoan_list[-1])+'\n')
  
fw.close()






N = len(tags)
x = range(N)

    
L1=plt.plot(x, mean_p1_list , 'bo')
L2=plt.axvline(x = 20, linewidth=1, color='k')
L3=plt.axvline(x = 22, linewidth=1, color='r')
L4=plt.axvline(x = 47, linewidth=1, color='r')
#L7=plt.axvline(x = 35, linewidth=1.5, color='r')
#L9=plt.plot(x, baseline2,linewidth=1,color='g')
#L8=plt.plot(x, baseline1,linewidth=1.5,color='k')

plt.legend( ('final ratio of contagion (initial ratio=0.05)', 
             'bankruptcy of Lehman Brothers',
             'start of stimulus package',
             'end of stimulus package'), loc = 'upper left',fontsize=7)

##x ticks
xtick = np.arange(0, N, 6)
xtickLabel = []
for i in xtick:
    xtickLabel.append('20' + tags[i])    
plt.xticks(xtick, xtickLabel, rotation = 45)
plt.xlabel('time')# make axis labels
plt.ylabel('')
plt.title('Final ratio of contagion in guarantee network(attack attack larger loan)') 
plt.savefig('loan_attack_%s.svg'%p,format='svg')
plt.close()




N = len(tags)
x = range(N)

    
L1=plt.plot(x, mean_n1_list , 'bo')
L2=plt.axvline(x = 20, linewidth=1, color='k')
L3=plt.axvline(x = 22, linewidth=1, color='r')
L4=plt.axvline(x = 47, linewidth=1, color='r')
#L7=plt.axvline(x = 35, linewidth=1.5, color='r')
#L9=plt.plot(x, baseline2,linewidth=1,color='g')
#L8=plt.plot(x, baseline1,linewidth=1.5,color='k')

plt.legend( ('final number of contagion (initial ratio=0.05)', 
             'bankruptcy of Lehman Brothers',
             'start of stimulus package',
             'end of stimulus package'), loc = 'upper left',fontsize=7)

##x ticks
xtick = np.arange(0, N, 6)
xtickLabel = []
for i in xtick:
    xtickLabel.append('20' + tags[i])    
plt.xticks(xtick, xtickLabel, rotation = 45)
plt.xlabel('time')# make axis labels
plt.ylabel('')
plt.title('Final number of contagion in guarantee network(attack attack larger loan)') 
plt.show()

N = len(tags)
x = range(N)

    
L1=plt.plot(x, mean_totalLoan_list , 'bo')
L2=plt.axvline(x = 20, linewidth=1, color='k')
L3=plt.axvline(x = 22, linewidth=1, color='r')
L4=plt.axvline(x = 47, linewidth=1, color='r')
#L7=plt.axvline(x = 35, linewidth=1.5, color='r')
#L9=plt.plot(x, baseline2,linewidth=1,color='g')
#L8=plt.plot(x, baseline1,linewidth=1.5,color='k')

plt.legend( ('final total loan of contagion (initial ratio=0.05)', 
             'bankruptcy of Lehman Brothers',
             'start of stimulus package',
             'end of stimulus package'), loc = 'upper left',fontsize=7)

##x ticks
xtick = np.arange(0, N, 6)
xtickLabel = []
for i in xtick:
    xtickLabel.append('20' + tags[i])    
plt.xticks(xtick, xtickLabel, rotation = 45)
plt.xlabel('time')# make axis labels
plt.ylabel('')
plt.title('Final total loan of contagion in guarantee network(attack larger loan)') 
plt.show()


