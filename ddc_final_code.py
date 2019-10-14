# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:37:14 2019

@author: dell
"""
####一，导入相应的包
import pandas as pd
import numpy as np
from numpy import log
import sklearn 
import matplotlib.pyplot as plt
import datetime
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as 

plt.rcParams['font.sans-serif']=['SimHei']#显示中文字体
plt.rcParams['axes.unicode_minus'] =False #减号unicode编码
##二，读入数据
data=r"C:/Users/dell/Desktop/全部A股.xlsx"
data=pd.read_excel(data,sheet_name="房地产业",header=None)
#更换数据列名
columns=["stock_code","stock_name","risk_waring_board","date_of_establish","reg_capital","ent_property","tot_staff","province","latest_subject_rat","executive_edu",
         "nat_of_chairman","chairman_compt","salary_of_mana","salary_of_direc","salary_exec","salary_all_exec","EVA","result_z","score_z","X1",
         "X2","X3","X4","X5","return_on_equity","equity_mul","TURNTA","oper_cycle","working_capital_tr","current_ratio",
         "quick_ratio","cash_flow_icr","cash_flow_dr","LEV","cur_tot_rate","ROA","ROE","ROIC","ROCE","money_funds",
         "unearned_revenue","st_loans","lt_loans","main_bus_incom","main_bus_cost","tot_pro","cash_sp","Cash_recborrow"]
data.columns=columns
data.set_index(["stock_code"],inplace=True)
##删除不必要的列
data.to_csv("data.csv")
data.drop(labels=["stock_name","latest_subject_rat","nat_of_chairman","X1","X2","X3","X4","X5","st_loans","lt_loans"],axis=1,inplace=True)
#检查缺失值
data.shape
data.isnull().sum()
#删除缺失值过多的列
data.drop(labels=["chairman_compt","cash_flow_icr","Cash_recborrow"],axis=1,inplace=True)
data.isnull().sum()
data=data.dropna()
##一些变量的处理
data["risk_waring_board"][data["risk_waring_board"]=="否"]=0
data["risk_waring_board"][data["risk_waring_board"]=="是"]=1   
#将成立日期即"date_of_establish"处理为建立年数，并进行分段
type(data['date_of_establish'])
data['date_of_establish']=pd.to_datetime(data['date_of_establish'],format='%Y-%m-%d')
data["year"]=2019-data['date_of_establish'].apply(lambda x:x.year)
#查看每个变量的直方图
#将企业性质，高管学历处理为哑变量
##只取最高学历，华夏幸福样本
data["ent_property"].value_counts()
data_pro=pd.get_dummies(data["ent_property"],prefix="企业性质")
data_pro
data["executive_edu"][data["executive_edu"]=="中专,硕士"]="硕士"
data["executive_edu"].value_counts()
data_edu=pd.get_dummies(data["executive_edu"],prefix="高管学历")
data.shape
data=pd.concat([data,data_edu,data_pro],axis=1)
data.drop(labels="date_of_establish",axis=1,inplace=True)
data.drop(labels="ent_property",axis=1,inplace=True)
data.drop(labels="province",axis=1,inplace=True)
data.drop(labels="executive_edu",axis=1,inplace=True)
data.drop(labels="EVA",axis=1,inplace=True)
data.drop(labels="score_z",axis=1,inplace=True)


data["result_z"][data["result_z"]=="堪忧"]=0
data["result_z"][data["result_z"]=="不稳定"]=1
data["result_z"][data["result_z"]=="良好"]=2
#对数据中量级较大的数取对数
data=data.astype(float)
data["tot_staff"]=data["tot_staff"].apply(lambda x:log(x+1))
data["money_funds"]=data["money_funds"].apply(lambda x:log(x+1))
data["main_bus_incom"]=data["main_bus_incom"].apply(lambda x:log(x+1))
data["main_bus_cost"]=data["main_bus_cost"].apply(lambda x:log(x+1))
data["tot_pro"]=data["tot_pro"].apply(lambda x:log(x+1))
data["cash_sp"]=data["cash_sp"].apply(lambda x:log(x+1))
data["unearned_revenue"]=data["unearned_revenue"].apply(lambda x:log(x+1))

data.isnull().sum()
data=data.drop(labels="tot_pro",axis=1)
data.isnull().sum()
data=data.dropna()
data.to_csv("data_handle.csv")


###模型一：利用余下的变量进行建模
df=data
X=df.drop(labels="result_z",axis=1)
y=df["result_z"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)

##利用特征得分筛选特征
##根据变量得分
from sklearn.linear_model import RandomizedLogisticRegression as RLR
rlr=RLR()
##训练模型
rlr.fit(X_train_std,y_train)
##筛选特征
rlr.get_support()
print(u'通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s' % ','.join(X.columns[rlr.get_support()]))
##获取各特征的分数
rlr.scores_
##分数最高的七个变量
##"LEV":资产负债率
##main_bus_cost  主营业务成本
##cash_sp销售商品、提供劳务收到的现金
##quick_ratio  速动比率
##unearned_revenue  预售账款
##money_funds    货币资金
##equity_mul  权益乘数


####模型5 财务指标和非财务指标
X5=pd.concat([data[["LEV","main_bus_cost","cash_sp","quick_ratio","unearned_revenue"]],data.iloc[:,27:40]],axis=1)
X5.shape
y5=data["result_z"]
X5_train,X5_test,y5_train,y5_test = train_test_split(X5,y5,test_size=0.3,stratify=y5,random_state=0)
print(X5.isnull().sum())

sc=StandardScaler()
sc.fit(X5_train)#计算样本的均值和标准差

X5_train_std=sc.transform(X5_train)
X5_test_std=sc.transform(X5_test)

lr=LogisticRegression(C=100.0,random_state=0)
lr.fit(X5_train_std,y5_train)


#模型预测
y5_pred=lr.predict(X5_test_std)
from sklearn.metrics import accuracy_score
acc5=accuracy_score(y5_test, y5_pred)
print(acc5)
print(classification_report(y5_test,y5_pred,target_names=["0","1","2"]))

y5_pred=pd.DataFrame(y5_pred,index=X5_test.index)
result5=pd.concat([X5_test,y5_pred],axis=1)
result5.to_csv("result5.csv")


###选择一个模型，对摸个企业进行展示

##结果展示
##企业内容展示