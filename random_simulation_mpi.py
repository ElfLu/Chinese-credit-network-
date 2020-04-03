# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:24:29 2018

@author: yingl WANG
"""

# -*- coding: utf-8 -*-
#import functional as fc
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
from operator import itemgetter
from random import choice


import os
import time
from multiprocessing import Pool
#import platform
n_threads = 8



#print(n_threads)
p=0.05
k=1.18

N=10000


#dir_data = './data'

dir_data = './data'

leverage_ratio_mean= [0.6076797117065776, 0.6021026207702114, 0.6038172553855191, 0.5983193300070662, 0.6086742695286927, 0.598158305875809, 0.6103454538328954, 0.6331384438595948, 0.6404109844607623, 0.666117873995153, 0.6770589334943751, 0.6986644641375865, 0.6599040679051866, 0.6379325292481253, 0.6303155440906371, 0.6068271538507617, 0.6298368401846184, 0.6453198614787364, 0.6208987222384853, 0.6314235053400848, 0.7081809611198399, 0.6615836515721076, 0.6206839118227941, 0.6153751293214306, 0.6163253482099861, 0.6046501627041009, 0.6043030523874803, 0.6032696779557967, 0.5897780753158531, 0.5977963566725081, 0.6030568844457008, 0.5904695214557505, 0.581916748803288, 0.6124483053826234, 0.5748372856430684, 0.5737915112398996, 0.5741145145600816, 0.5719725328484, 0.5716617187925458, 0.5665687673549136, 0.5678490352346388, 0.5684169416277204, 0.5683872751001415, 0.5688140355776847, 0.5650072618353632, 0.5656566796881977, 0.5621742628207272, 0.5591675972367488, 0.557013155430675, 0.5571124212520364, 0.5560858221650133, 0.5599700717826255, 0.558969868886333, 0.5610119432552954, 0.560435837659287, 0.5567066585224107, 0.5563791549283111, 0.5549837006083131, 0.5550266459184884, 0.5552154034326474, 0.5536671984248862, 0.5555148831279054, 0.5504813364983077]

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
    
LEHMAN_IDX =20
FOURTRI_IDX = 22 
FOURTRI_IDX_END = 47
'''
def get_guarantee_network(tag):
    g = nx.DiGraph()

    fin1 = open(r'C:\network\data\edges\edge%s.txt'%tag )
###### Readlines()[1:] should exculde the first line in the original txt, which is the description of variables###### 
    for line in fin1.readlines()[1:]:
        (customer, guarantor, balance_, num_) = line.split()
        g.add_edge(guarantor, customer, balance = float(balance_), number = int(num_))
    fin1.close()
    nodes =set(g.nodes())    
####### Add informations of nodes #######
    fin2 = open(r'C:\network\data\nodes\network%s.txt'%tag)
    nodes_has_info =[]
    var_name = ['', 'bad_amount', 'expire_amount', 'asset',
                'liability', 'confer', 'loan', 'region', 'risk',
                'market', 'bad_rate', 'expire_rate']

    for line in fin2.readlines()[1:]:
        x = line.split( )     
        if x[0] in nodes:
            nodes_has_info.append(x[0])
            for i in range(1, 12):
                g.node[x[0]][var_name[i]] = float(x[i])

    fin2.close()
'''
def get_guarantee_network(tag):
    #sigma=sigma_dict[tag]
    
    g = nx.DiGraph()
    file1 = os.path.join(dir_data, 'edges', 'edge%s.txt'%tag)
    fin1 = open(file1, 'r')
    fin1.readline()
    nodes_list=[]  ###### null liste of all nodes ######
    customer_list=[]
    for line in fin1:
        tmp_list = line.split()
        if len(tmp_list) == 4:
            customer, guarantor, balance, num = line.split( )
            customer_list.append(customer)
            nodes_list.append(customer)
            nodes_list.append(guarantor)
            #g.add_edge(guarantor, customer, balance = float(balance), number = int(num))
    fin1.close()
####### Add new edges to netowrk #######
#    for node in customer_list:
#        g.add_edge(node, choice(nodes_list))
    file1 = os.path.join(dir_data, 'edges', 'edge%s.txt'%tag)
    fin1 = open(file1, 'r')
    fin1.readline()

    for line in fin1:
        tmp_list = line.split()
        if len(tmp_list) == 4:
            customer, guarantor, balance, num = line.split( )

            g.add_edge(guarantor, customer, balance = float(balance), number = int(num))
            if random.random()<0.1: #####10% node rewire ######
                g.add_edge(choice(nodes_list),customer, balance = g[guarantor][customer]['balance'], number = int(1))
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
           lnasset_list.append(np.log(g.nodes[node]['asset']))
           lnliability_list.append(np.log(g.nodes[node]['liability']))
           lnloan_list.append(np.log(g.nodes[node]['loan']))
            ### lnP ~ lnC - alpha lnk
    x = np.vstack([np.array(lnasset_list), np.ones(len(lnasset_list))]).T
    (b1, a1) = np.linalg.lstsq(x, np.array(lnliability_list),rcond=None)[0]  
    (b2, a2) = np.linalg.lstsq(x, np.array(lnloan_list),rcond=None)[0]   
    #compute the mean and sd of log(asset)
    narray=np.array(lnasset_list)
    sum1=narray.sum()
    narray2=narray*narray
    sum2=narray2.sum()
    mean_log_asset=sum1/len(lnasset_list)
    sd_log_asset=sqrt(sum2/len(lnasset_list)-mean_log_asset**2)   
    sd = 0.8972
    tmp = np.random.randn(500000)
    i = 0
  
    #fill the missing value 
    for node in g.nodes():
        if node in nodes_has_info and g.nodes[node]['asset']==0:
           log_asset = mean_log_asset + sd_log_asset * tmp[i]
           i += 1
           g.nodes[node]['asset'] = np.exp(log_asset)
           #g.node[node]['liability'] = np.exp(a1 + b1 * log_asset)
           #g.node[node]['loan'] = np.exp(a2 + b2 * log_asset + sd * tmp[i])
           i += 1
      
        if node in nodes_has_info and g.nodes[node]['asset']>0 and g.nodes[node]['liability']==0:
           g.nodes[node]['liability'] = np.exp(a1 + b1 * log(g.nodes[node]['asset'])) 
        if node in nodes_has_info and g.nodes[node]['asset']>0 and g.nodes[node]['liability']>0 and g.nodes[node]['loan']==0:
           g.nodes[node]['loan'] = np.exp(a2 + b2 * log(g.nodes[node]['asset']) + sd * tmp[i])
    del tmp 
    return (g,nodes_has_info) 


def prob_contagion(asset, liab, bad_amount, k, sigma):  #感染概率
#    sigma=3.275
    ratio = (liab + bad_amount) / asset
    x = ratio - sigma
    x *= k
    return 1 / (1 + np.exp(-x))

a = 0

def process(g, p, k, sigma):

    #初始化状态
    num_bad = 0
    queue = cl.deque()
    loansum=0
    
    for node in g.nodes():
        
        g.nodes[node]['bad_danbao'] = 0
        x=g.nodes[node]['loan']
        #print (x)  
        if not np.isnan(x):
           loansum+=x
        #print(loansum)

    for node in g.nodes():
        g.nodes[node]['state'] = 0 
        if random.random() < p:    
            g.nodes[node]['state'] = 1
            num_bad += 1
            #balance_set=set() ###### Initialize the loan guarantee of default node ######
            balance=dict()
            asset=[]
            for nd in g.predecessors(node):   ###### Caculate the sum loan guarantee of one node #######
                #balance_set.add(g[nd][node]['balance'])
                if g[nd][node]['balance'] not in balance.keys():
                    balance[g[nd][node]['balance']] = []
                balance[g[nd][node]['balance']].append(nd)
                queue.append(nd)
#            for item in balence         asset.append(g.node[nd]['asset'])
#                
#            total_balance=sum(balance_set)
#            total_set=sum(asset)
            for nd in g.predecessors(node):  ###### Assign average loan guarantee to every node  #######
                #g.node[nd]['bad_danbao'] +=  total_balance/len(list(g.predecessors(node)))
                total_asset = 0
                for item in balance[g[nd][node]['balance']]:
                    total_asset += g.nodes[item]['asset']
                g.nodes[nd]['bad_danbao'] +=  g[nd][node]['balance']*g.nodes[nd]['asset']/total_asset
                #g.node[nd]['bad_danbao'] += g.node[node]['balance']/len(list(g.predecessors(node)))
    if g.number_of_nodes()==0:
       p0=0
    else: 
       p0 = num_bad / float(g.number_of_nodes())
    n0 = num_bad
    #print (n0)
    #print (loansum)

    totalLoan = 0
   # print ('a')
    #print (len(queue))
    while True:
        tmp_queue = []
        
        flag = True
        for node in queue:
        #while(len(queue) >0):
            #print(queue)
            #node = queue.popleft()
             #所有的点都是坏的
            if g.nodes[node]['state'] == 0:
                flag = False #发现有一个好的点，因此循环要继续
                bad_amount = g.nodes[node]['bad_danbao']
                liab = g.nodes[node]['liability']
                asset = g.nodes[node]['asset']
                #if random.random() < prob_contagion(asset, liab, bad_amount, k, sigma):
                result=prob_contagion(asset, liab, bad_amount, k, sigma)
                #print (result)
                if random.random() < result:
                    
                    totalLoan += bad_amount
                    g.nodes[node]['state'] = 1
                    #print ('bad')
                    num_bad += 1
                    #print(num_bad)
                    balance=set()
                    for nd in g.predecessors(node):
                        tmp_queue.append(nd)
                        balance.add( g[nd][node]['balance'])
                    total=sum(list(balance))
                    for nd in g.predecessors(node):                  
                        g.nodes[nd]['bad_danbao'] += total/len(list(g.predecessors(node)))
        queue = tmp_queue
        if flag == True:#所有的点都是坏的，所以跳出
            break

    p1 = num_bad / g.number_of_nodes()
    n1 = num_bad
    #print (num_bad)
    totalLoan_ratio=totalLoan/loansum      
    return (p0, n0, p1, n1, totalLoan,totalLoan_ratio)


def onemonth_simulation (g, tag, p, k, N):
    p0=0
    n0=0
    p1=0
    n1=0
    totalLoan=0
    totalLoan_ratio=0
    # 所以你在这里加sigma够了！
    sigma=sigma_dict[tag]
    
    #(g, nodes_has_info)=get_guarantee_network(tag)
    i = 0
    for num in range(N):
        print(i)
        i = i + 1
        print(tag)
        print (sigma)
        (pp0, nn0, pp1, nn1, totalLoan0,totalLoan_ratio0)=process(g, p, k, sigma)
        p0+=pp0
        n0+=nn0
        p1+=pp1
        n1+=nn1
        totalLoan+=totalLoan0
        totalLoan_ratio+=totalLoan_ratio0
     
    mean_p0= p0/float(N)
    mean_n0=n0/float(N) 
    mean_p1=p1/float(N) 
    mean_n1=n1/float(N) 
    mean_totalLoan=totalLoan/float(N)
    mean_totalLoan_ratio=totalLoan_ratio/float(N)
    
    return (mean_p0,mean_n0,mean_p1,mean_n1,mean_totalLoan, mean_totalLoan_ratio) 
    


mean_p0_list=[]
mean_n0_list=[]
mean_p1_list=[]
mean_n1_list=[]
mean_totalLoan_list=[]
mean_totalLoan_ratio_list=[]

month_count=0     

def oper_tags(tag):
    #print(tag)
#    sigma=leverage_ratio_mean[month_count]
#    month_count=month_count+1
   
    (g,nodes_has_info)=get_guarantee_network(tag)
    #print(sigma)
    g1=g.subgraph(nodes_has_info)
    return onemonth_simulation (g1, tag, p, k,  N)

def f(x):
    return x*x    

#pool = Pool(processes=4)    
#print(pool.map(f,range(10)))


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
#with Pool(n_threads) as pool:
 #   print("pool start")
  #  result_list = pool.map(oper_tags,tags) 
  
for item in result_list:
    mean_p0_list.append(item[0])
    mean_n0_list.append(item[1])
    mean_p1_list.append(item[2])
    mean_n1_list.append(item[3])  
    mean_totalLoan_list.append(item[4])
    mean_totalLoan_ratio_list.append(item[5])
     
# for tag in tags:

#     (g,nodes_has_info)=get_guarantee_network(tag)
#     g1=g.subgraph(nodes_has_info)
#     (mean_p0,mean_n0,mean_p1,mean_n1,mean_totalLoan, mean_totalLoan_ratio) = onemonth_simulation (g1, tag, p, k,  N)
#     mean_p0_list.append(mean_p0)
#     mean_n0_list.append(mean_n0)
#     mean_p1_list.append(mean_p1)
#     mean_n1_list.append(mean_n1)
#     mean_totalLoan_list.append(mean_totalLoan)
#     mean_totalLoan_ratio_list.append(mean_totalLoan_ratio)
  
mean_n0_list[57]=float(mean_n0_list[56]+mean_n0_list[58])/2
mean_p1_list[57]=float( mean_p1_list[56]+ mean_p1_list[58])/2
mean_n1_list[57]=float(mean_n1_list[56]+mean_n1_list[58])/2
mean_totalLoan_list[57]=float(mean_totalLoan_list[56]+mean_totalLoan_list[58])/2

print (mean_p0_list)
print (mean_n0_list)
print (mean_p1_list)
print (mean_n1_list)
print (mean_totalLoan_list)
print (mean_totalLoan_ratio_list)

data = [mean_p0_list,mean_n0_list,mean_p1_list,mean_n1_list,mean_totalLoan_list,mean_totalLoan_ratio_list]
fp = open('simulation_GN_%s.csv'%p,'a')   
fp.write(str(data))
fp.write('\n')
fp.close()


#####plot of simulation#######
tags = []
for y in range(7, 12):
    for m in range(1, 13):
        tags.append('%02d%02d'%(y, m)) 
tags.append('1201')
tags.append('1202')
tags.append('1203')

def region_GN_005 (xfont,yfont,titlefont,legendfont):
    plt.figure(figsize=(14,7))
    N = len(tags)
    x = range(N)
    L5=plt.plot(x,  mean_p1_list , 'bo',linewidth=1) 
    L1=plt.axvline(x = 3, linewidth=2.2, color='k')
    L2=plt.axvline(x = 20, linewidth=2.5, color='k',linestyle='dashed')
    L3=plt.axvline(x = 22, linewidth=2.2,  color='r')
    L4=plt.axvline(x = 47, linewidth=2.5, color= 'r', linestyle='dashed')
       
    #plt.legend(loc='upper center', bbox_to_anchor=(0.6,0.95),ncol=3,fancybox=True,shadow=True)
  
    plt.legend( ('initial ratio is %s %%'%p),loc='upper left',fontsize=legendfont,
               fancybox=True,shadow=True)
  
    xtick = np.arange(0, N, 6)
    xtickLabel = []
    for i in xtick:
        xtickLabel.append('20' +tags[i])     
    plt.xticks(xtick, xtickLabel, rotation = 45,fontsize=xfont)
    plt.yticks(fontsize=yfont)
    plt.ylabel('final contagion ratio',fontsize=16)
    #plt.title('Shandong1',fontsize=titlefont) 
    plt.savefig('simulation_GN_%s.svg'%p,format='svg')
    plt.show()
    plt.close()
    
region_GN_005 (14,14,14,13)

'''
tags = []
for y in range(7, 12):
    for m in range(1, 13):
        tags.append('%02d%02d'%(y, m))
tags.append('1201')
tags.append('1202')
tags.append('1203')
full_tags = []
for tag in tags:
    full_tags.append('20' + tag)
N = len(tags)
x = range(N)
    
LEHMAN_IDX =20 
FOURTRI_IDX = 22 
FOURTRI_IDX_END = 47

N = len(tags)
x = range(N)

plt.figure(figsize=(15,7))


L1=plt.plot(x, random_005, 'bo' ,linewidth=5)
#L2=plt.plot(x, power_law_out, 'ko',linewidth=5)
L3=plt.axvline(x = 20, linewidth=2.2, color='k')
L4=plt.axvline(x = 22, linewidth=2.2, color='r')
L5=plt.axvline(x = 47, linewidth=2.5, color='r',linestyle='dashed')
xtick = np.arange(0, N, 6)
xtickLabel = []
for i in xtick:
    xtickLabel.append('20' +tags[i])     
plt.xticks(xtick, xtickLabel, rotation = 45,fontsize=14)
plt.yticks(fontsize=14)
plt.legend( ('final failed ratio (initial ratio=0.05)','bankruptcy of Lehman Brothers','start of the Chinese economic stimulus package',
             'end of the Chinese economic stimulus package'), fontsize=14, loc='upper left',
           shadow=True,edgecolor='w',facecolor='w')
plt.ylabel('Average ratio of failed enterprises',fontsize=16)
'''
