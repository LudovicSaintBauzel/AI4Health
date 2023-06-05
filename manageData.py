import pandas as pd
import os
import glob
import math
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


os.chdir('/Users/ludo/Documents/src/AI4Health')
# s=glob.glob("*/follow_traj.txt"),glob.glob("*/*/follow_traj.txt"),,glob.glob("*/*/*/follow_traj.txt")
# ,glob.glob("*/*/*/follow_traj.txt"),glob.glob("*/*/follow_traj.txt"), ]
s = [[file for file in glob.glob("*/follow_traj.txt")]]
# s=[[file for file in glob.glob("*/*/*/follow_traj.txt")]]
filtering_data = ['records/test1/traj/follow_traj.txt',
                  'records/BO2504/test/follow_traj.txt',
                  'records/fuzzy/tt/follow_traj.txt',
                  'records/test1/fx/follow_traj.txt',
                  # Above Problematic data : Below test data not relevant
                  'records/test1/av/follow_traj.txt',
                  'records/test2/traj/follow_traj.txt',
                  'records/test2/av/follow_traj.txt',
                  'records/traj/test/follow_traj.txt',
                  'records/move/follow_traj.txt',
                  'NO_march1-3/follow_traj.txt',
                  'NO_march6-10/follow_traj.txt',
                  'GU_marche1/follow_traj.txt',
                  'GU_marche2/follow_traj.txt'
                  ]
flats = []
for sub_s in s:
    for item in sub_s:
        flats.append(item)
print(flats)

df = pd.read_csv(flats[0], sep='\s+', header=None)
df['file'] = pd.Series([flats[0] for i in df[0]])

for ff in flats[1:]:
    dftmp = []
    if ff in filtering_data:
        print("File filtered not loaded : "+str(ff)+"\n")
        continue
    print("File opened and loaded : " + str(ff) + "\n")
    dftmp = pd.read_csv(ff, sep='\s+', header=None)
    if len(dftmp) > 0:
        print("Appending len(dftmp):" + str(len(dftmp))+"\n")
        dftmp['file'] = pd.Series([ff for i in dftmp[0]])
        df = df.append(dftmp)

print('len(df)=' + str(len(df)))

# Machine LEARNING Part

lDatas = len(df)
lowLDatas = math.floor(lDatas/10)
df_d = df.drop('file', axis=1)

dfTrainIN = df_d.drop([0, 18, 19, 20], axis=1)[1:10000]
dfTrainOUT = df_d[19][1:10000]
dfTestIN = df_d.drop([0, 18, 19, 20], axis=1)[10000:20000]
dfTestOUT = df_d[19][10000:20000]

scaler = StandardScaler()

scaler.fit(dfTrainIN)
dfTrainIN = scaler.transform(dfTrainIN)
dfTestIN = scaler.transform(dfTestIN)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(19,2), random_state=1)
clf = MLPRegressor()
clf.fit(X=dfTrainIN, y=dfTrainOUT.values)

predRes = clf.predict(dfTestIN)
res = dfTestOUT-predRes
print(res)
