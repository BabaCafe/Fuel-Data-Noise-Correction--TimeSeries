# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 23:00:06 2018

@author: Vashi NSIT
"""

import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize']=8,6

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    #print(var,l,t)
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels
def outliers(df,var):
    q1=np.percentile(df[var],25)#.quantile([0.25])
    q3=np.percentile(df[var],75)
    iqr=q3-q1
    lower_outlier=q1-(iqr*3)
    upper_outlier=q3+(iqr*3)
    return q1,q3,lower_outlier,upper_outlier



data= pd.read_excel('Problemstatement.xlsx')
data.describe()

data=data.drop_duplicates()
data=data.reset_index(drop=True)

pd.value_counts(data['Fuel Level (mV)'])
data.plot(x='Cumulative Distance',y='Fuel Level (mV)')

outliers(data,'Fuel Level (mV)')
data[data['Fuel Level (mV)'].isna()]
len(data)

#data.plot( kind='scatter', x='Speed',y='Fuel Level (mV)')
#fig,ax=plt.subplots()
#sns.distplot(data['Fuel Level (mV)'])


d=0
maxdist=0
t=list()
for i in data['Cumulative Distance']:
    k=i-d
    d=i
    t.append(k)
    if k > maxdist:
        maxdist=k

maxdist
actualdistance=np.array(t)
data.insert(4,'Actual Distance',value=actualdistance)
#data.loc[np.where(actualdistance==maxdist)[0]]
#data.head()
#data.corr()
#data.describe()

m=data['Fuel Level (mV)'][0]
s=list()
for i in data['Fuel Level (mV)']:
    k=i-m
    m=i
    s.append(k)


fuellevelchange=np.array(s)
data.insert(5,'Fuel Level Change',value=fuellevelchange)
#data.head()
#data.corr()

#fig,ax=plt.subplots()
#ax.plot(data.loc[3500:3600]['Timestamp'],data.loc[3500:3600]['Fuel Level Change'])
#ax.set_ylim(-100,100)

#np.where(np.logical_and(abs(fuellevelchange)>55,abs(fuellevelchange)<110))[0]

data['Fuel Level Change(abs)']=abs(data['Fuel Level Change'])
data.head()

#fig,ax=plt.subplots()
#sns.distplot(data['Fuel Level Change(abs)'], ax=ax)

def estimate_gaussian(X):  
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return mu, sigma
    
estimate_gaussian(data)
mu,sigma=estimate_gaussian(data)

type(sigma)
sigma.head()
FLCsigma=sigma['Fuel Level Change(abs)']
FLCsigma= (int(FLCsigma)+1 if (FLCsigma-int(FLCsigma))>0.5  else int(FLCsigma))
X=pd.DataFrame({'level':data['Fuel Level (mV)'],'lc':data['Fuel Level Change'],'lcabs':data['Fuel Level Change(abs)'],'cd':data['Cumulative Distance']})
X.head()

#X.plot(kind='scatter', x='cd',y='level')
#checkpoints=np.where(X.lcabs>FLCsigma)[0]
#type(checkpoints)    
#len(checkpoints)
#X.level[0]=651
#X.level[0:3]
#np.sum(X.level[[0,4]])/3

def update1_value(X,ind,pos,checkpoints):
    if X.cd[ind]==X.cd[ind-1]:
        X.level[ind]=X.level[ind-1]
    else:
        X.level[ind]=np.sum(X.level[ind-3:ind])/3
    
    X.lc[ind]=X.level[ind]-X.level[ind-1]
    X.lc[ind+1]=X.level[ind+1]-X.level[ind]
    X.lcabs[ind]=abs(X.lc[ind])        
    X.lcabs[ind+1]=abs(X.lc[ind+1])
    val=ind+1
    if X.lcabs[val]>FLCsigma and val not in checkpoints :
        checkpoints=np.insert(checkpoints,pos+1,val)
    return X,checkpoints

def update2_value(X,ind,pos,checkpoints):
    if X.cd[ind]==X.cd[ind-1]:
        X.level[ind]=X.level[ind-1]
    else:
        X.level[ind]=np.sum(X.level[[ind-1,ind+1]])/2
    X.lc[ind]=X.level[ind]-X.level[ind-1]
    X.lc[ind+1]=X.level[ind+1]-X.level[ind]
    X.lcabs[ind]=abs(X.lc[ind])        
    X.lcabs[ind+1]=abs(X.lc[ind+1])
    val=ind+1
    if X.lcabs[val]>FLCsigma and val not in checkpoints :
        checkpoints=np.insert(checkpoints,pos+1,val)
    return X,checkpoints


##1a NOT USED --as of now but can be used for further iterations instead of update1 after update1 has been used in first iteration.
def update1a_value(X,ind,pos,checkpoints):
    if X.cd[ind]==X.cd[ind-1]:
        X.level[ind]=X.level[ind-1]
    else:
        X.level[ind]=((np.sum(X.level[ind-3:ind])/3)+(np.sum(X.level[ind+1:ind+4])/3))/2
    X.lc[ind]=X.level[ind]-X.level[ind-1]
    X.lc[ind+1]=X.level[ind+1]-X.level[ind]
    X.lcabs[ind]=abs(X.lc[ind])        
    X.lcabs[ind+1]=abs(X.lc[ind+1])
    val=ind+1
    if X.lcabs[val]>FLCsigma and val not in checkpoints :
        checkpoints=np.insert(checkpoints,pos+1,val)
    return X,checkpoints


def update3_value(X,ind,pos,checkpoints):
    if X.cd[ind]==X.cd[ind-1]:
        X.level[ind]=X.level[ind-1]
    else:
        X.level[ind]=((np.sum(X.level[ind-3:ind])/3)+(np.sum(X.level[ind+3:ind+6])/3))/2
    X.lc[ind]=X.level[ind]-X.level[ind-1]
    X.lc[ind+1]=X.level[ind+1]-X.level[ind]
    X.lcabs[ind]=abs(X.lc[ind])        
    X.lcabs[ind+1]=abs(X.lc[ind+1])
    val=ind+1
    if X.lcabs[val]>FLCsigma and val not in checkpoints :
        checkpoints=np.insert(checkpoints,pos+1,val)
    return X,checkpoints


#val=595        
#if X.lcabs[val]<FLCsigma and val not in checkpoints :#if val not in checkpoints:
#    k=np.insert(checkpoints,5,val)
#len(k)
#X.level[checkpoints]
#pod=position of drop

def noise(X,pod):
    j=0
    checkpoints=np.where(X.lcabs>pod)[0]
    while j < len(checkpoints):
        i=checkpoints[j]
        z=np.sum(X.level[i+3:i+6])/3
        d=z-X.level[i]
        r=X.lcabs[i]
        
        if X.level[i]<330 or X.level[i]>780:
            X,checkpoints=update1_value(X,i,j,checkpoints)
            
        
        elif X.lcabs[i+1]>FLCsigma and (X.lc[i]*X.lc[i+1])<0:
            X,checkpoints=update2_value(X,i,j,checkpoints)
            
        
        elif abs(d/r)>=0.5 and (X.lc[i]*d)<0:
            X,checkpoints=update3_value(X,i,j,checkpoints)
            
        j=j+1
    return X


        
pod= FLCsigma
#X=noise(X,pod)
#X.level[17040:17060]
def no_of_iter(data,corr_type,y,d,iters,pod):
    i=0
    while i<iters:
        i=i+1
        checkpoints1=np.where(data.lcabs>pod)[0]
        countbefore=len(checkpoints1)
        
        if corr_type==noise:
            data=corr_type(data,pod)
        else:
            data=corr_type(data,pod,y,d)
        
        checkpoints2=np.where(data.lcabs>pod)[0]
        countafter=len(checkpoints2)
        
        if (countbefore==countafter) and (checkpoints1.all()==checkpoints2.all()):
            print('yeah',i)
            return data
    return data

def update5_value(X,m,n,pos,checkpoints):
    s1=np.sum(X.level[n:n+30])
    s2=np.sum(X.level[m-30:m])
    prev_val_mean=s1/30
    next_val_mean=s2/30
    s3=np.sum(X.level[n:n+50])
    s4=np.sum(X.level[m-50:m])
    prev_val_mean1=s3/50
    next_val_mean1=s4/50
    if abs((prev_val_mean)-(next_val_mean))<25:
        #print('yop')
        X.level[m:n]=int((prev_val_mean + next_val_mean)/2)
        X.lc[m:n+1]=X.level[m:n+1].reset_index(drop=True)-X.level[m-1:n].reset_index(drop=True)
        X.lcabs[m:n+1]=abs(X.lc[m:n+1])
        #print('oh',X[m-1:n+1])
    elif abs((prev_val_mean1)-(next_val_mean1))<25:
        #print('yoop')
        X.level[m:n]=int(((prev_val_mean1) + (next_val_mean1))/2)
        X.lc[m:n+1]=X.level[m:n+1].reset_index(drop=True)-X.level[m-1:n].reset_index(drop=True)
        X.lcabs[m:n+1]=abs(X.lc[m:n+1])
        #print('odh',X[m-1:n+1])
    val=n
    if X.lcabs[val]>FLCsigma and val not in checkpoints :
        checkpoints=np.insert(checkpoints,pos+1,val)
    return X,checkpoints

#levchng=X.level[m:n+1].reset_index(drop=True)-X.level[m-1:n].reset_index(drop=True)    
#X.lc[m:n+1]=levchng- NEVER USE THIS TYPE OF ASSIGNMENT THROUGH VARIABLE IT FAILS

#update5_value(X,17050,17060)

def device_incorrection(X,pod,y,d):
    j=0
    checkpoints=np.where(X.lcabs>pod)[0]
    l=list()
    while j < len(checkpoints):
        i=checkpoints[j]
        if X.lcabs[i]>pod:
            r=X.lc[i]
            #print(i)
            s=0
            while s<3:
                sum=0
                p=0
                t=i+1
                while p<y:
                    k=X.lc[t]
                    sum=sum+k
                    #print('s',sum)
                    if s==0 and abs(sum+r)<d :
                        #print('yo',i,t,p)
                        X,checkpoints=update5_value(X,i,t,j,checkpoints)
                        l.append([i,t])
                        p=20
                        s=3
                    elif s==1 and abs(sum+r)< d+10 :
                        #print('yup',i,t,p)
                        X,checkpoints=update5_value(X,i,t,j,checkpoints)
                        l.append([i,t])
                        p=20
                        s=3
                    elif s==2 and ((sum+r)*r<0 or(abs(np.sum(X.level[i-10:i])/10)-(np.sum(X.level[t:t+10])/10))<d+10):
                        #print('yes',i,t,p)
                        X,checkpoints=update5_value(X,i,t,j,checkpoints)
                        l.append([i,t])
                        p=20
                        s=3
                    p=p+1
                    t=t+1
                s=s+1
        j=j+1
    return X

#X,l=device_incorrection(X,pod,20,15)

# iters=40
X= no_of_iter(X,noise,20,15,40,pod)
X= no_of_iter(X,noise,20,15,40,int(pod/2))

#y=20, d=15, iters=40- Use this one first 
X= no_of_iter(X,device_incorrection,20,15,40,pod)

#Repeat the last three codes one more time then for further improvement use this one.
#y=55 Use this one if further improvement required
X= no_of_iter(X,device_incorrection,55,15,40,pod)


X=pd.DataFrame({'level':data['Fuel Level (mV)'],'lc':data['Fuel Level Change'],'lcabs':data['Fuel Level Change(abs)'],'cd':data['Cumulative Distance']})
#X.head()

#checkpoints=np.where(np.logical_and(X.lcabs>pod, X.level<500 ))

checkpoints=np.where(X.lcabs>pod)[0] 
len(checkpoints)

checkpoints_drops=np.where(X.lcabs>pod)[0]

checkpoints_corr=np.where(X.lcabs>pod)[0]

X.head()
#X.level.plot()
#X['Time']=data['Timestamp']
#X['Actdist']=data['Actual Distance']
#X['Vel']=data['Speed']

def drop(pod):
    checkdrops=checkpoints_drops#np.where(X.lcabs>pod)[0]
    drop=pd.DataFrame()
    j=-35
    for i in checkdrops:
        if i-j>35:
            drop=drop.append(X[i-25:i+50])
        j=i
    return drop

drops=drop(pod)
drops.head()

#X.plot(kind='scatter', x='cd',y='level')

#drops.to_csv('Suspiciouspoints.csv')

#X.to_csv('Solution.csv', index=False)

def normalize(data):
    data=(data-data.mean())/data.std()
    return data

Normalized_X=pd.DataFrame(normalize(X.level))
Normalized_X.head()

Normalized_X.insert(0,'Time',value=data['Timestamp'])
Normalized_X.head()

Normalized_X=Normalized_X.set_index(Normalized_X.Time)

#Normalized_X.set_index('Time').plot()
#plt.plot(time = Normalized_X['Time'], data = Normalized_X['level'])
X.plot(x='cd',y='level')

sns.tsplot([Normalized_X.level])
#describe_more(Normalized_X)
#Normalized_X.info()
#len(Normalized_X)

import matplotlib.dates as mdates
Normalized_X['Timestamp']=Normalized_X.Time.map(lambda t: mdates.date2num(t))
#Normalized_X.Timestamp.head()

fig, ax = plt.subplots()
sns.tsplot(data=Normalized_X.level,time=Normalized_X.Timestamp,ax=ax)
#sns.tsplot(Normalized_X[0:1000], time='Timestamp', value='level', ax=ax)
# assign locator and formatter for the xaxis ticks.
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

# put the labels at 45deg since they tend to be too long
fig.autofmt_xdate()
plt.show()
