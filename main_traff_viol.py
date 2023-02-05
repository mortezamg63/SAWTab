from VIME import vime_semi, vime_self, mlp, perf_metric
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
# tf.executing_eagerly()
from tensorflow.keras.utils import to_categorical
# from MyModel import vime_semi, vime_self, mlp, perf_metric
from datetime import date
import os
import copy
import pandas as pd
import numpy as np
from dirty_cat import TargetEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import random as python_random

def process_criteo_dataset():
    df = pd.read_csv('~/Documents/modular_code/dataset/short_criteo.csv')#'./dataset/short_criteo.csv')
    df=df.drop([df.columns[0], df.columns[1]],axis=1)
    target = df.pop(df.columns[0])
    cont_idx=range(13)
    cont_cols = []
    cat_idx = range(13,39)
    for i in cont_idx:
        cont_cols = cont_cols +[df.columns[i]]

    scaler = StandardScaler()
    cont_features = scaler.fit_transform(df[cont_cols])
    for i in cont_idx:
        df.iloc[:,i] = cont_features[:,i]
    
    for i in cat_idx:
        df.iloc[:,i] = df.iloc[:,i].astype(int)
    
    dftr, dftst, Ytr, Ytst = train_test_split(df,target,test_size=0.2,shuffle=False)
    return dftr, Ytr, dftst, Ytst, cat_idx, cont_idx
       
    
def process_Viol_traffics_dataset(ds_tr):
    train = pd.read_csv(ds_tr)
    target = 'Label'

    ytr = train.pop('Label')
    Xtr = train
       
    Xtr.pop('article')
    
    cols_nan = []
    search_cols = []
    for i in range(len(Xtr.columns)):
        if Xtr[Xtr.columns[i]].isna().sum()>0:
            if 'search' not in Xtr.columns[i]:
                cols_nan.append(Xtr.columns[i])
            else:
                search_cols.append(Xtr.columns[i])

    # drop search columnss
    for c in search_cols: 
        Xtr.pop(c)

    Xtr['violation_type'] = ytr
    Xtr.dropna(subset=cols_nan, inplace = True)
    Ytr = Xtr.pop('violation_type')
    Xtr.pop('agency')

    colss=np.array(Xtr.columns)
    
    colss = ['subagency', 'description', 'location', 'accident', 'belts',
        'personal_injury', 'property_damage', 'fatal', 'commercial_license',
        'hazmat', 'commercial_vehicle', 'alcohol', 'work_zone', 'state',
        'vehicletype', 'make', 'model', 'color', 'charge',
        'contributed_to_accident', 'race', 'gender', 'driver_city',
        'driver_state', 'dl_state', 'arrest_type']#colss[idx]
    dropCols = set(Xtr.columns)-set(colss)
    Xtr = Xtr.drop(dropCols,axis=1)
    df = pd.DataFrame()
    le = LabelEncoder()
    
    for c in colss:
        df[c] = le.fit_transform(Xtr[c])

    le = LabelEncoder()
    Ytr = le.fit_transform(Ytr)

    Ytr = pd.DataFrame(Ytr)
    df = df.reset_index()
    df.pop('index')

    dftr, dftst, Ytr, Ytst = train_test_split(df,Ytr,test_size=0.2,shuffle=False)
    return dftr, Ytr, dftst, Ytst, np.arange(len(dftr.columns)), []
    

def  calculate_representation_statistics(df_l,targets, df_u, pseudolabels, cat_idxs, class_weights):
    df_l=df_l.reset_index() 
    targets = targets.reset_index()
    del df_l['index']
    del targets['index']
    print("[INFO] - Generate representation ...")
    df_l['Label']=targets
    num_classes = int(targets.max()+1)
    if df_u is not None:
        df_u=df_u.reset_index()
        del df_u['index']
        df_u['Label'] = pseudolabels
    
    total_rep = []
    cols_process = df_l.columns[:-1]
    for idx,col in enumerate(cols_process):
        if idx in cat_idxs:
            # pridfnt("Processing", col)
            df_l[col]=df_l[col].fillna('ffffffff')
            Nax = df_l.groupby(col).size()
            Naxc = df_l.groupby([col,'Label']).size()
            if df_u is not None:
                df_u[col]=df_u[col].fillna('ffffffff')
                Nax_u = df_u.groupby(col).size()
                Naxc_u = df_u.groupby([col,'Label']).size()
            temp=[]
            t_rep_nominator = {}
            t_rep_denominator = {}
            t_rep = {}
            
            def calc_prob(Nx,Nxc,islabeled=True):
                for i in range(len(Nx)):
                    try:
                        cat = Nx.index.values[i]
                    except:
                        import pdb;pdb.set_trace()
                    v=Nxc[cat]
                    repVec_nominator = np.zeros(num_classes, dtype=float)
                    repVec_denominator = np.zeros(num_classes, dtype=float)
                    for num,lbl in zip(v,v.index.values):
                        repVec_nominator[int(lbl)] = num if islabeled else class_weights[int(lbl)]*num
                        repVec_denominator[int(lbl)] = Nx.values[i] if islabeled else class_weights[int(lbl)]*Nx.values[i]

                    # Nominator
                    if islabeled:
                        t_rep_nominator[cat] = 2.0*repVec_nominator#4.5*repVec_nominator # labeled weight
                    else:
                        try:
                            t_rep_nominator[cat] += repVec_nominator
                        except:
                            t_rep_nominator[cat] = repVec_nominator
                    
                    # Denominator
                    if islabeled:
                        t_rep_denominator[cat] = 2.0*repVec_denominator # labeled weight
                    else:
                        try:
                            t_rep_denominator[cat] += repVec_denominator
                        except:
                            t_rep_denominator[cat] = repVec_denominator
                    
            
            # Labeled data
            calc_prob(Nax,Naxc,True)
            # Unlabeled data
            if df_u is not None:
                calc_prob(Nax_u,Naxc_u,False)
            # final representation
            for cat in t_rep_nominator.keys():
                t_rep[cat] = np.nan_to_num(t_rep_nominator[cat]/t_rep_denominator[cat])#t_rep_nominator[cat]/t_rep_denominator[cat]
            total_rep.append(t_rep)
        else:
            total_rep.append('cont')

    return total_rep, num_classes

labeled_ratio=0.1
random_seed = 123
num_classes = 4
data_tr, target_tr, data_tst, target_tst, cat_idx, cont_idx = process_Viol_traffics_dataset_TE('../traffic_violation/dataset_shuffled_trfViol.csv')#Documents/traffic_violation/
data_tr, target_tr, data_tst, target_tst, cat_idx, cont_idx = process_Viol_traffics_dataset('../dataset_shuffled_trfViol.csv')

data, targets = data_tr, target_tr
idx = np.arange(len(targets))
labeled_idx = idx
unlabeled_idx = idx
idx = idx

if (labeled_ratio > 0):
    if random_seed is not None:
        idx = np.random.RandomState(seed=random_seed).permutation(len(targets)) # random shuffling 
    else:
        idx = np.random.permutation(len(targets))

if labeled_ratio <= 1.0:
    ns = labeled_ratio * len(idx)
else:
    ns = labeled_ratio
    
ns = int(ns)
labeled_idx = idx[:ns]
valid_ns = int(len(labeled_idx)*0.1)

unlabeled_idx = idx[ns:]
class_weights = np.ones(targets.nunique())


cond_probs, num_classes = calculate_representation_statistics(data.iloc[labeled_idx], targets.iloc[labeled_idx], None, None, cat_idx, class_weights)
class_weight = np.ones(4)
sample_weight = np.ones(len(idx))
data_tr = data.to_numpy()
target_tr = targets.to_numpy()
data_tst = data_tst.to_numpy()
target_tst = target_tst.to_numpy()
pseudo_labels = list()
_pseudo_labels_weights = list()
refined_unlabeled_idx = list()
agree=list()
probs_lp = list()

def create_representation(x):
    record = np.zeros(len(cat_idx)*num_classes+len(cont_idx),dtype=float)
    tmp_idx = 0
    for j in range(len(cat_idx)+len(cont_idx)):
        if cond_probs[j] == 'cont':
            record[tmp_idx] = x[j]
            tmp_idx+=1
        else:
            try:
                record[tmp_idx:tmp_idx+num_classes] = cond_probs[j][x[j]]# 2D output
            except:
                pass # leave zero
            tmp_idx+=num_classes

    return record.astype(np.float32)

def update_representation():
    cpr_train = list()
    for i in range(len(data_tr)):
        cpr_train.append(create_representation(data_tr[i]))

    cpr_train = np.stack(cpr_train)

    cpr_test = list()
    for i in range(len(data_tst)):
        cpr_test.append(create_representation(data_tst[i]))
    cpr_test = np.stack(cpr_test)
    return cpr_train, cpr_test


def VIME_semi(x_train,y_train, x_unlab,x_test,y_test, epoch,total_train,sample_weight, class_weight, pseudoLabels):
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # MLP
    mlp_parameters = dict()
    mlp_parameters['hidden_dim'] = 1100
    mlp_parameters['epochs'] = 100
    mlp_parameters['activation'] = 'tanh'
    mlp_parameters['batch_size'] = 100
    
    if epoch==-1:
        y_test_hat = mlp(x_train, y_train, x_test, mlp_parameters)
        results = perf_metric('acc', y_test, y_test_hat)

        # Report performance
        print('Supervised Performance, Model Name: ' , 'MLP' , 
                ', Performance: ',results)
    #parameters 
    p_m = 0.3 
    alpha = 1.0 
    K = 2
    beta = 1.25 
    label_data_rate = 0.1 # DO NOT CHANGE THIS PARAMETER
    file_name = './save_model/encoder_model.h5'
    vime_self_parameters = dict()
    vime_self_parameters['hidden_dim'] = 104 #65
    vime_self_parameters['batch_size'] = 256 #256
    vime_self_parameters['epochs'] = 15
    vime_self_encoder = vime_self(x_unlab, p_m, alpha, vime_self_parameters)

    # Save encoder
    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    file_name = './save_model/encoder_model_2.h5'
    vime_self_encoder.save(file_name) 

    # Test VIME-Self
    x_train_hat = vime_self_encoder.predict(x_train)
    x_test_hat = vime_self_encoder.predict(x_test)
    
    y_test_hat = mlp(x_train_hat, y_train, x_test_hat, mlp_parameters,class_weight) # add class wieght in loss function in both semi and supervised
    results = perf_metric('acc', y_test, y_test_hat)

    print('VIME-Self Performance: ' + str(results)) 

    vime_semi_parameters = dict()
    vime_semi_parameters['hidden_dim'] = 104 # Traffic Violations #65 Criteo
    vime_semi_parameters['batch_size'] = 256
    vime_semi_parameters['iterations'] = 2000
    vime_semi_parameteers['activation_fn'] = 'tanh'

    y_test_hat,pseudoLabels = vime_semi(x_train, y_train, x_unlab, x_test, vime_semi_parameters, p_m, K, beta, file_name,total_train,class_weight, epoch, pseudoLabels)

    results = perf_metric('acc', y_test, y_test_hat)
    pl_acc =  perf_metric('acc', pseudoLabels, to_categorical(target_tr))
    

    print('VIME Performance: '+ str(results))
    print('PseudoLabel Performance: '+ str(pl_acc))
    return pseudoLabels

cpr_train, cpr_test = update_representation()
pseudoLabels = None

for epoch in range(7):
    if epoch==0:
        pseudoLabels = VIME_semi(cpr_train[labeled_idx],target_tr[labeled_idx],cpr_train[unlabeled_idx], cpr_test, target_tst, epoch, cpr_train,sample_weight, class_weight, pseudoLabels)
    else:
        pseudoLabels = VIME_semi(cpr_train,pseudoLabels,cpr_train, cpr_test, target_tst, epoch,cpr_train, sample_weight, class_weight,  pseudoLabels)

    print("[INFO] - Refinement and Update Policy")
    print("[INFO] - Refinement and Update Policy")
    idx=np.argmax(pseudoLabels,1)
    pseudoLabels[labeled_idx,idx[labeled_idx]]=1.0
    confidence = np.max(pseudoLabels,1)
    orig_pseudoLabels = np.argmax(pseudoLabels,axis=1)

    def change_range(distb, newmin, newmax):
        oldmax,oldmin=distb.max(),distb.min()
        distb=((distb-oldmin)*(newmax-newmin))/(oldmax-oldmin)+newmin
        return distb
    
    def getClassWeights(distb):
        N = np.sum(distb)/len(distb)
        beta = (N-1)/N
        print("beta advised =", beta)
        ones = np.ones(distb.shape)
        distb = np.where(distb==0, ones,distb)
        class_weight = (1-beta)/(1-(beta**distb))
        class_weight = len(class_weight)*class_weight/np.sum(class_weight)
        return class_weight
    
    
    #==============================================================
    def refine_pseudo_labels(orig_pseudoLabels, confidence):
        n_k=[]
        index_range = np.arange(len(orig_pseudoLabels))
        for i in range(num_classes):
            nk = lambda x: orig_pseudoLabels[x]==i and confidence[x]>0.8
            final_output=list(map(nk, index_range))
            n_k.append(np.sum(final_output)+1)
        # Normalize n_k between 1 and 100
        n_k = np.array(n_k)
        n_k = change_range(n_k, 1, 1000)
        # beta = 0.5
        S=getClassWeights(n_k)
        return S 
        
    class_weights2=refine_pseudo_labels(orig_pseudoLabels, confidence)#,beta=0.02)
    pseudoLabels[labeled_idx,idx[labeled_idx]]=1.0
    pseudoLabels = np.argmax(pseudoLabels, axis=1)
    pseudoLabels[labeled_idx] = target_tr[labeled_idx].squeeze()
    
    cond_probs, num_classes = calculate_representation_statistics(data.iloc[labeled_idx], pd.DataFrame(pseudoLabels[labeled_idx]), data.iloc[unlabeled_idx], pd.DataFrame(pseudoLabels[unlabeled_idx]), cat_idx, class_weights2)
    cpr_train, cpr_test = update_representation()
    

