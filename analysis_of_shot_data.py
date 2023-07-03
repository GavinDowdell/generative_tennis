# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:51:33 2023

@author: gavin
"""

import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

#%%

# reading the variable width file is a bit of a challenge
# 1. provide a header text for file for the width you want to read in.
#    Having this wide field up front allows pandas to read it
# 2. set usecols options appropriately
 
points = []
with open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced.txt','r') as f:
    for line in f:
        #print(f)
        points.append(len(line.strip().split(',')))


max(points)

header = 'serve'

for i in range(max(points)):
    header = header + ',shot' + str(i)

file = open(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\header.txt','w')
file.write(header)
file.close()


#%%

# test
#df = pd.read_csv(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced_for_analysis.txt'
#                 ,usecols = [0,1,2,3])

df = pd.read_csv(r'C:\gavin\software\python\pytorch\karpathy\makemore\makemore\tennis_shots_new_all_final_reduced_for_analysis.txt'
                 ,usecols = list(range(85)),low_memory=False)

# how many points per length
for i in df.columns:
    print(sum(~df[i].isna()))

#%%

# some intial analysis to look at the sequential non-iid nature
df
df['serve'].value_counts()
# histogram
# value counts already in histogram from since sums per group
# hence a density plot is a value count and then a bar chart
df['serve'].value_counts().plot(kind='bar')
# should not really have to do this
# should be able to operate on the raw data
# like a standard histogram calculation
sns.countplot(df['serve'], color='gray')

# look at the overall shot1 distribution
df['shot1'].value_counts(normalize=True)
df['shot1'].value_counts(normalize=True).plot(kind='bar')
df['shot1'].value_counts(normalize=True)[df['shot1'].value_counts()>1000].plot(kind='bar')

#%%

# look at shot1 for different serve types
# lots for fh returns # forced error
# consider for example how p(shot1='f29') varies over serve type
# second serve to forehand much higher and clearly no chance for a214
df.loc[(df['serve'] == 'a114'),'shot1'].value_counts(normalize=True)
# second serve f/h side to the f/h side
df.loc[(df['serve'] == 'a124'),'shot1'].value_counts(normalize=True)
# lots for bh returns
df.loc[(df['serve'] == 'a214'),'shot1'].value_counts(normalize=True)

#%%

# now some more subtle patterns
# the 2nd shot in both cases f18 but look at the different winners concentration
# attacking - some 
df.loc[(df['serve'] == 'a114') & (df['shot1'] == 'f18'),'shot2'].value_counts(normalize=True)
# defensive
df.loc[(df['serve'] == 'a124') & (df['shot1'] == 'f18'),'shot2'].value_counts(normalize=True)

# 4th shot in rally depends upon first shot
# attacking
df.loc[(df['serve'] == 'a114') & (df['shot1'] == 'f18') & (df['shot2'] == 'f3'),'shot3'].value_counts(normalize=True)
# defensive
df.loc[(df['serve'] == 'a124') & (df['shot1'] == 'f18') & (df['shot2'] == 'f3'),'shot3'].value_counts(normalize=True)

#%%

# lots of interesting relationships
# consider for f1 in 2 scenarios
# a114,f1*,f1,NEXT
# vs during a rally
# f1,f1,f1, NEXT

#before we get going compare the result after the first shot only
# lots of forced errprs
df.loc[(df['serve'] == 'a114'),'shot1'].value_counts(normalize=True)
# now into a rally - less forced errors
df.loc[(df['shot2'] == 'f1'),'shot3'].value_counts(normalize=True)
# HOWEVER ULTIMATELY DIFFERENT SHOTS SO GETTING A DIFFERENT REPRESENATION
# NOT BIG DEAL

# HOW ABOUT f1
# what representation do we need of f1 to predict the next shot


# serve out wide, fh and return to our fh, out fh x-court, NEXT
idx1 = ((df['serve'] == 'a114') & (df['shot1'].isin(['f17','f18','f19']))
                                  & (df['shot2'] == 'f1'))
sum(idx1)


idx2 = ((df['shot2'] == 'f1') & (df['shot3'] == 'f1') & (df['shot4'] == 'f1'))
sum(idx2)

idx3 = ((df['shot2'] == 'f3') & (df['shot3'] == 'b1') & (df['shot4'] == 'f1'))
sum(idx3)


# hence need a different representation of f1, based upon in left to right
# context to capture this difference
df.loc[idx1,'shot3'].value_counts(normalize=True)
df.loc[idx2,'shot5'].value_counts(normalize=True)
# much higher f1 as we have spread the court here
# so the opposition wants to come back x-court
df.loc[idx3,'shot5'].value_counts(normalize=True)






