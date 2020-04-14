# -*- coding: utf-8 -*-
##### to generate the Dircted Configuration Model #######
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from os.path import join
import networkx as nx
import collections as cl
import os.path
import math
from math import log, exp, sqrt
import scipy.stats as stats  
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#from random import shuffle
import pandas as pd

import os
import time
from multiprocessing import Pool
import random
n_threads = 12
import sys


#
#random_powerlaw_tree_sequence(n, gamma=3, seed=None, tries=100)
#Returns a degree sequence for a tree with a power law distribution.
#Parameters
#• n (int,) – The number of nodes.
#• gamma (float) – Exponent of the power law.
#• seed (int, optional) – Seed for random number generator (default=None).
#• tries (int) – Number of attempts to adjust the sequence to make it a tree
########     step 1: Genarate real network
######### golobal prameters  #########
generation_times = 10000
tag = sys.argv[1][:4]
#print(tag)
trid_name=['003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', 
           '201', '120D', '120U', '120C', '210', '300']




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
#tags=['0701','0702']

full_tags = []
#for tag in tags:
#    full_tags.append('20' + tag)

regions=[11,12,13,14,15,21,22,23,31,32,33,34,35,36,37,41,42,
         43,44,45,46,50,51,52,53,54,61,62,63,64,65,66,71,81,82]
    
LEHMAN_IDX =20 
FOURTRI_IDX = 22 
FOURTRI_IDX_END = 47

#get GN and fill in the missing values of the nodes information
def get_guarantee_network(tag):
    
    print(tag)
    g = nx.DiGraph()
    file1 = os.path.join(dir_data, 'edges', 'edge%s.txt'%tag)
    fin1 = open(file1, 'r')
    fin1.readline()
    for line in fin1:
        if len(line.split())==4:
            (customer, guarantor, balance_, num_) = line.split()
            g.add_edge(guarantor, customer, balance = float(balance_), number = int(num_))
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

#    print (in_degree_sequence)
#    print ( out_degree_sequence)
#   #填充缺失值
#    node_list=[]
#    lnasset_list=[]
#    lnliability_list=[]
#    lnloan_list=[]
#    
#    for node in nodes_has_info:
#        if g.nodes [node]['asset']>0:
#           node_list.append(node)
#           lnasset_list.append(np.log(g.nodes[node]['asset']+1))
#           lnliability_list.append(np.log(g.nodes[node]['liability']+1))
#           lnloan_list.append(np.log(g.nodes[node]['loan']+1))
#   #compute the coefficiences of liner regressions
#    x = np.vstack([np.array(lnasset_list), np.ones(len(lnasset_list))]).T
#    b1, a1 = np.linalg.lstsq(x, np.array(lnliability_list))[0]  
#    b2, a2 = np.linalg.lstsq(x, np.array(lnloan_list))[0]   
#    #compute the mean and sd of log(asset)
#    narray=np.array(lnasset_list)
#    sum1=narray.sum()
#    narray2=narray*narray
#    sum2=narray2.sum()
#    mean_log_asset=sum1/len(lnasset_list)
#    sd_log_asset=sqrt(float(sum2)/len(lnasset_list)-mean_log_asset**2)   
#    sd = 0.8972
#    tmp = np.random.randn(500000)
#    i = 0    
#    #fill the missing value 
#    for node in g.nodes():
#        if node not in nodes_has_info or g.node[node]['asset'] < 1e-6:
#           log_asset = mean_log_asset + sd_log_asset * tmp[i]
#           i += 1
#           g.node[node]['asset'] = np.exp(log_asset)
#           g.node[node]['liability'] = np.exp(a1 + b1 * log_asset)
#           g.node[node]['loan'] = np.exp(a2 + b2 * log_asset + sd * tmp[i])
#           i += 1
#    del tmp    
    return (g, nodes_has_info)

#(g, nodes_has_info)=get_guarantee_network('0701')

########     step 2：  Genarate null model

""" # old
#def generation_null_models():
for tag in tags:
    (g, nodes_has_info)=get_guarantee_network(tag)
    in_degree_sequence = list(d for n, d in g.in_degree())
    out_degree_sequence = list(d for n, d in g.out_degree())
######  generate 10000 directed configuration model  
    glb_dic={ '003': [], '012': [], '102': [], 
             '021D':[], '021U': [], '021C': [],  '111D': [], '111U': [], '030T':[], 
             '030C': [], '201': [], '120D': [], '120U': [], '120C':[], '210': [], 
             '300': [], 'null_dyad':[],'sigle_dyad':[], 'mutual_dyad':[] }
    null_dyad_list=[]  
    sigle_dyad_list=[]
    mutual_dyad_list=[]
    
    for i in range(generation_times):
        print (i)
        g1=nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
        tmp_dic= nx.triadic_census(g1)
        for key,value in tmp_dic.items():
            glb_dic[key].append(value)
        
        recip_ratio=nx.overall_reciprocity(g1)
        total_nodes= g1.number_of_nodes()
        total_edges= g1.number_of_edges()
        double_link=recip_ratio*total_edges
        sigle_dyad=total_edges-double_link
        glb_dic['sigle_dyad'].append(sigle_dyad)        
#        sigle_dyad_list.append(sigle_dyad)
        mutual_dyad=double_link/2
        glb_dic['mutual_dyad'].append(mutual_dyad)        
#        mutual_dyad_list.append (mutual_dyad)
        null_dyad=total_nodes*(total_nodes-1)/2-sigle_dyad-mutual_dyad
#        null_dyad_list.append (null_dyad)
        glb_dic['null_dyad'].append(null_dyad)


#    fw=open('2_3_nodes_motif_occurance.csv','a')
    df = pd.DataFrame.from_dict(glb_dic)
    df['time']='20%s'%tag
    df.to_csv("DCM_null_model_%s.csv"%tag)
""" # old    
#tag = '0701'
#tag = sys.argv[1]
g, nodes_has_info=get_guarantee_network(tag)
in_degree_sequence = list(d for n, d in g.in_degree())
out_degree_sequence = list(d for n, d in g.out_degree())
#zipped_list = zipped(in_degree_sequence,out_degree_sequence )

#list(zip(*input_list))[0]

def oper_num(num):
    mp_dic= {}
    g1=nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
    tmp_dic = nx.triadic_census(g1)
    
    recip_ratio=nx.overall_reciprocity(g1)
    total_nodes= g1.number_of_nodes()
    total_edges= g1.number_of_edges()
    double_link=recip_ratio*total_edges
    sigle_dyad=total_edges-double_link
    mutual_dyad=double_link/2
    null_dyad=total_nodes*(total_nodes-1)/2-sigle_dyad-mutual_dyad
    
    tmp_dic['null_dyad'] = null_dyad
    tmp_dic['sigle_dyad'] = sigle_dyad
    tmp_dic['mutual_dyad'] = mutual_dyad
    return tmp_dic
    
    
result_list = []
if __name__ == "__main__":
    startTime = time.perf_counter()
    print(startTime)
    #pool = Pool(processes=4)
    with Pool(n_threads) as pool:
        result_list = pool.map(oper_num,range(generation_times))
        print(result_list)
    stopTime = time.perf_counter()
    print(stopTime)
    print(stopTime - startTime)    
# 结果整合起来
glb_dic={ '003': [], '012': [], '102': [], 
         '021D':[], '021U': [], '021C': [],  '111D': [], '111U': [], '030T':[], 
         '030C': [], '201': [], '120D': [], '120U': [], '120C':[], '210': [], 
         '300': [], 'null_dyad':[],'sigle_dyad':[], 'mutual_dyad':[] }    

for tmp_dic in result_list:
    for key,value in tmp_dic.items():
        glb_dic[key].append(value)

df = pd.DataFrame.from_dict(glb_dic)
df['time']='20%s'%tag
df.to_csv("DCM_null_model_%s.csv"%tag)  

    
#    fw.write(str('null_dyad_list')+',')
#    for i in null_dyad_list[:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(null_dyad_list[-1])+'\n') 
#
#    fw.write(str('sigle_dyad_list')+',')
#    for i in sigle_dyad_list[:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(sigle_dyad_list[-1])+'\n') 
#    
#    fw.write(str('mutual_dyad_list')+',')
#    for i in mutual_dyad_list[:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(mutual_dyad_list[-1])+'\n') 
#    
#    
#    fw.write(str('003')+',')
#    for i in glb_dic['003'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['003']+'\n') 
#
#    fw.write(str('012')+',')
#    for i in glb_dic['012'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['012']+'\n') 
#    
#    fw.write(str('102')+',')
#    for i in glb_dic['102'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['102']+'\n') 
#
#    fw.write(str('021D')+',')
#    for i in glb_dic['021D'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['021D']+'\n') 
#    
#    fw.write(str('021U')+',')
#    for i in glb_dic['021U'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['021U']+'\n') 
#
#    fw.write(str('021C')+',')
#    for i in glb_dic['021C'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['021C']+'\n') 
# 
#    fw.write(str('111D')+',')
#    for i in glb_dic['111D'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['111D']+'\n')    
#    
#    fw.write(str('111U')+',')
#    for i in glb_dic['111U'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['111U']+'\n')    
#     
#    fw.write(str('030T')+',')
#    for i in glb_dic['030T'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['030T']+'\n')         
#
#    fw.write(str('030C')+',')
#    for i in glb_dic['030C'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['030C']+'\n')         
#    
#    fw.write(str('201')+',')
#    for i in glb_dic['201'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['201']+'\n')         
#
#    fw.write(str('120D')+',')
#    for i in glb_dic['120D'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['120D']+'\n')        
#
#    fw.write(str('120U')+',')
#    for i in glb_dic['120U'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['120U']+'\n')         
#
#    fw.write(str('120C')+',')
#    for i in glb_dic['120C'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['120C']+'\n')        
#
#    fw.write(str('210')+',')
#    for i in glb_dic['210'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['210']+'\n')         
#
#    fw.write(str('300')+',')
#    for i in glb_dic['300'][:-1]:
#        fw.write(str(i)+',')
#    fw.write(str(glb_dic['300']+'\n')     

#    fw.close()
#    print (null_dyad_list)
#    print (sigle_dyad_list)
#    print (mutual_dyad_list)
#    print (glb_dic)
#    

    
    


#g2=nx.directed_configuration_model(in_degree_sequence, out_degree_sequence)
#r2= nx.overall_reciprocity(g2)
#print ("g2+=")
#print (r2)



#for i in range(20):
#    print (i)
#    zipped = list (zip(in_degree_sequence, out_degree_sequence ))
#    print (zipped)
#    shuffle(zipped)
#    a=list(zip(*zipped))
##    print (a)
#    g1=nx.directed_configuration_model(a[0], a[1], seed=None)
##    G.append(g1)
#    r1=nx.overall_reciprocity(g1)
#    reciporcity.append (r1)
##    nx.is_isomorphic(g_test, g1)
#print (reciporcity)
