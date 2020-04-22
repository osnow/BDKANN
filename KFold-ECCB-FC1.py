import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from keras.optimizers import adam
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import sklearn.preprocessing as sk
import random
from tensorflow.python.keras import backend as K
from sklearn.preprocessing import Imputer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

GDSCE = pd.read_csv("/home/hnoghabi/KDNN/GDSC.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
GDSCE.drop_duplicates(keep='last')
GDSCE = pd.DataFrame.transpose(GDSCE)
GDSCE = GDSCE.loc[:,~GDSCE.columns.duplicated()]
GDSCE.index = GDSCE.index.astype('int64')

CCLEE = pd.read_csv("/home/hnoghabi/KDNN/CCLE.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
CCLEE.drop_duplicates(keep='last')
CCLEE = pd.DataFrame.transpose(CCLEE)
CCLEE = CCLEE.loc[:,~CCLEE.columns.duplicated()]

ls = GDSCE.columns.intersection(CCLEE.columns)
land_ls = pd.read_csv("landmark_genes.tsv", 
                    sep = "\t", index_col=0, decimal = ".")
ls4 = ls.intersection(land_ls.index)
GDSCEv2 = GDSCE.loc[:,ls4]
CCLEEv2 = CCLEE.loc[:,ls4]

drugs = set()
celllines = set()
for line in open('/home/hnoghabi/KDNN/CCLE_CTRPv2.shared_with_GDSC.PharmacoDB.responses.tsv').readlines()[1:]:
    line = line.rstrip().split('\t')
    celllines.add(line[1])
    drugs.add(line[4])

data = {cellline : {drug : [0,0] for drug in drugs} for cellline in celllines}

for line in open('/home/hnoghabi/KDNN/CCLE_CTRPv2.shared_with_GDSC.PharmacoDB.responses.tsv').readlines()[1:]:
    line = line.rstrip().split('\t')
    cellline = line[1]
    drug = line[4]
    if not '' == line[-5]:
        IC50 = float(line[-5])
        data[cellline][drug][0] += IC50
        data[cellline][drug][1] += 1

for cellline,drugs in data.items():
    for drug in drugs:
        IC50_sum = data[cellline][drug][0]
        IC50_count = data[cellline][drug][1]
        if IC50_count > 0:
            data[cellline][drug] = IC50_sum/IC50_count
        else:
            data[cellline][drug] = 'NaN'

CCLER = pd.DataFrame.from_dict(data).transpose()
CCLER.to_csv('CCLER.csv', sep=',', decimal = ".")

drugs = set()
celllines = set()
for line in open('/home/hnoghabi/KDNN/GDSC.shared_with_CCLE_CTRPv2.PharmacoDB.responses.tsv').readlines()[1:]:
    line = line.rstrip().split('\t')
    celllines.add(line[1])
    drugs.add(line[4])

data = {cellline : {drug : [0,0] for drug in drugs} for cellline in celllines}

for line in open('/home/hnoghabi/KDNN/GDSC.shared_with_CCLE_CTRPv2.PharmacoDB.responses.tsv').readlines()[1:]:
    line = line.rstrip().split('\t')
    cellline = line[1]
    drug = line[4]
    if not '' == line[-5]:
        IC50 = float(line[-5])
        data[cellline][drug][0] += IC50
        data[cellline][drug][1] += 1

for cellline,drugs in data.items():
    for drug in drugs:
        IC50_sum = data[cellline][drug][0]
        IC50_count = data[cellline][drug][1]
        if IC50_count > 0:
            data[cellline][drug] = IC50_sum/IC50_count
        else:
            data[cellline][drug] = 'NaN'

GDSCR = pd.DataFrame.from_dict(data).transpose()
GDSCR.to_csv('./home/hnoghabi/KDNN/GDSCR.csv', sep=',', decimal = ".")

GDSCR.index = GDSCR.index.astype('int64')
ls2 = GDSCEv2.index.intersection(GDSCR.index)
ls3 = CCLEEv2.index.intersection(CCLER.index)

GDSCEv3 = GDSCEv2.loc[ls2,:]
GDSCRv2 = GDSCR.loc[ls2,:]
CCLEEv3 = CCLEEv2.loc[ls3,:]
CCLERv2 = CCLER.loc[ls3,:]

Mask1 = pd.read_csv("/home/hnoghabi/KDNN/M1.csv", 
                    sep = ",", index_col=0, decimal = ".")
Mask1.drop_duplicates(keep='last')
Mask1 = Mask1.loc[~Mask1.index.duplicated(),:]
Mask1 = Mask1.loc[:,~Mask1.columns.duplicated()]


Mask2 = pd.read_csv("/home/hnoghabi/KDNN/M2.csv", 
                    sep = ",", index_col=0, decimal = ".")
Mask2 = Mask2.loc[~Mask2.index.duplicated(),:]
Mask2 = Mask2.loc[:,~Mask2.columns.duplicated()]

lsC = GDSCEv3.columns.intersection(Mask1.index)
GDSCEv4 = GDSCEv3.loc[:,lsC]
CCLEEv4 = CCLEEv3.loc[:,lsC]
Mask1 = Mask1.loc[lsC,:]

#drugs = ['Parthenolide','ATRA','Doxorubicin','Tamoxifen','lapatinib','Gefitinib',
#         'BIBW2992','masitinib','17-AAG','GDC-0941','MK-2206','NVP-BEZ235','PLX4720', 'Erlotinib']
#GDSCRv4 = GDSCRv2[drugs]
#CCLERv4 = CCLERv2[drugs]
#
#GDSCRv4 = GDSCRv4.replace(to_replace ='NaN', value = GDSCRv4[GDSCRv4!='NaN'].mean())
#CCLERv4 = CCLERv4.replace(to_replace ='NaN', value = CCLERv4[CCLERv4!='NaN'].mean())

drugs = ['ATRA', 'Doxorubicin', 'Tamoxifen', 'Gefitinib', 'BIBW2992','masitinib', '17-AAG', 'GDC-0941', 'MK-2206', 'AZD6244', 'PLX4720']
drugIDs = ['15367', '28748', '41774', '49668', '61390', '63450', '64153', '65326', '67271','90227', '90295']
GDSCRv3 = GDSCRv2[drugs]
CCLERv3 = CCLERv2[drugs]

imputer1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer1 = imputer1.fit(GDSCRv3[drugs].values)
GDSCRv4 = imputer1.transform(GDSCRv3[drugs].values)

imputer2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer2 = imputer2.fit(CCLERv3[drugs].values)
CCLERv4 = imputer2.transform(CCLERv3[drugs].values)

lsP = Mask1.columns.intersection(Mask2.index)

Mask1 = Mask1.loc[:,lsP]
Mask2 = Mask2.loc[lsP,:]

Mask3 = pd.read_csv("/home/hnoghabi/KDNN/M3New.csv", 
                    sep = ",", index_col=0, decimal = ".")
lsD = Mask2.columns.intersection(Mask3.index)
Mask2 = Mask2.loc[:,lsD]
Mask3 = Mask3.loc[lsD,drugIDs]

Mask1v2 = 1- Mask1.values
Mask2v2 = 1-Mask2.values
Mask3v2 = 1-Mask3.values

ls_mb_size = [16, 32, 64]
ls_epoch = [20, 50, 100, 150, 200]
ls_rate = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3]
ls_h_dim1 = [1024, 512, 256, 128]
ls_h_dim2 = [128, 64, 32, 16]

max_iter = 100

save_results_to2 = '/home/hnoghabi/KDNN/keras-Results/KFold-ECCB-FC1V1/'

kf = KFold(n_splits=10, random_state=42, shuffle=True)
for iters in range(max_iter):
    
    mbs = random.choice(ls_mb_size)
    hdm1 = random.choice(ls_h_dim1)
    hdm2 = random.choice(ls_h_dim2)
    epoch = random.choice(ls_epoch)
    rate1 = random.choice(ls_rate)
    rate2 = random.choice(ls_rate)  

    k = 0

    for tr_ind, ts_ind in kf.split(GDSCEv4.values):
        k = k + 1
        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(GDSCEv4.values[tr_ind,:])
        X_trainE = scalerGDSC.transform(GDSCEv4.values[tr_ind,:])
        X_testE = scalerGDSC.transform(GDSCEv4.values[ts_ind,:])
    
        model = Sequential()
        model.add(Dense(hdm1, input_dim=X_trainE.shape[1], kernel_initializer='normal', activation='relu'))
        model.add(Dropout(rate1))
        model.add(Dense(hdm2, activation='relu'))
        model.add(Dropout(rate2))
        model.add(Dense(len(drugs), activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        history = model.fit(X_trainE, GDSCRv4[tr_ind,:], epochs=epoch, batch_size=mbs,  verbose=0, validation_data=(X_testE,GDSCRv4[ts_ind,:]))
        
        plt.plot(history.history['loss'][3:])
        plt.plot(history.history['val_loss'][3:])
        title2 = 'MSE Train iter = {}, fold = {}, mb_size = {},  epoch = {}, rate = ({},{}), dim = ({},{})'.\
                          format(iters, k, mbs, epoch, rate1, rate2, hdm1, hdm2)
        plt.title(title2)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(save_results_to2 + title2 + '.png', dpi = 150, bbox_inches = 'tight')
        plt.close()
    K.clear_session()