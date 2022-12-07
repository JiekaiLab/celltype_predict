import scanpy as sc
import pandas as pd
import numpy as np
import lightgbm as lgb
import tqdm
import joblib
import nmslib
import scipy.sparse
from scipy.sparse import issparse
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import argparse as arg
import sys
import warnings
warnings.filterwarnings("ignore")

class DESCRIPTION:
    Program = "Usage: python CJKLab_model command"
    Version = "V1.0"
    Contact = "CJKLab"
    Description = "%s\n%s\n%s" % (Program, Version, Contact)

level1_model = './model/Level.model'
retrain_model = './model/Retrain.model'
label_coder = preprocessing.LabelEncoder()
label_coder.classes_ = np.load('./model/cell_type_encode.npy') 

def read_data(datapath):
    print("Loading your data......")
    data = sc.read_h5ad(datapath)
    print("Data loaded")
    return data

def preprocessing_data(data,n_comps = 30,n_neighbors = 10,scale=True,res=1.5):
    adata = data.copy()
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    if scale:
        sc.pp.scale(adata, max_value=10)
    sc.tl.pca(data=adata, svd_solver='arpack', n_comps=n_comps)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(adata)
    sc.tl.leiden(adata,resolution=res)
    return adata

def load_model(path):
    model = lgb.Booster(model_file=path)
    return model

def load_retrain_model(path):
    model = joblib.load(path)
    return model

def load_ct_model(path):
    ct_model = {}
    ct = ['Endothelial cell','Fibroblast','Lymphoid cell','Myeloid cell']
    for i in ct:
        ct_model[i] = joblib.load(path + i + ".pkl")
    return ct_model

def predict_preprocess(raw_data,model):
    use_model = model
    use_feature = use_model.feature_name()
    data = raw_data.copy()
    pre_data = data[:,data.var_names.isin(use_feature)]
    if issparse(pre_data.X):
        df = pd.DataFrame(pre_data.X.A)
    else:
        df = pd.DataFrame(pre_data.X)
    df.index = pre_data.obs_names.tolist()
    df.columns = pre_data.var_names.tolist()
    difference = list(set(use_feature).difference(set(pre_data.var_names)))
    for i in difference:
        df[i] = 0
    return df[use_feature]

def label_decode(raw_predict,label_coder,proba):
    test_l=[]
    test_pro=[]
    ind=[]
    for i in tqdm.tqdm(range(len(raw_predict))):
        test_l.append(raw_predict[i].argmax())
        test_pro.append(raw_predict[i].max())
        if raw_predict[i].max()<proba:
            ind.append(i)
    test_re = label_coder.inverse_transform(test_l).astype(str)
    test_re[ind]='Unknown'
    return test_re

def fastKnn(X1, 
            X2=None, 
            n_neighbors=20, 
            metric='euclidean', 
            M=40, 
            post=0, 
            efConstruction=100,
            efSearch=200):
    if metric == 'euclidean':
        metric = 'l2'
    if metric == 'cosine':
        metric = 'cosinesimil'
    if metric == 'jaccard':
        metric = 'bit_jaccard'
    if metric == 'hamming':
        metric = 'bit_hamming'
    index_time_params = {'M': M,
                         'efConstruction': efConstruction, 
                         'post' : post} 
    efSearch = max(n_neighbors, efSearch)
    query_time_params = {'efSearch':efSearch}
    
    if issparse(X1):
        if '_sparse' not in metric:
            metric = f'{metric}_sparse'
        index = nmslib.init(method='hnsw', space=metric, data_type=nmslib.DataType.SPARSE_VECTOR)
    else:
        index = nmslib.init(method='hnsw', space=metric, data_type=nmslib.DataType.DENSE_VECTOR)
    index.addDataPointBatch(X1)
    index.createIndex(index_time_params, print_progress=False)
    index.setQueryTimeParams(query_time_params)
    if X2 is None:
        neighbours = index.knnQueryBatch(X1, k=n_neighbors)
    else:
        neighbours = index.knnQueryBatch(X2, k=n_neighbors)
    
    distances = []
    indices = []
    for i in neighbours:
        if len(i[0]) != n_neighbors:
            vec_inds = np.zeros(n_neighbors)
            vec_dist = np.zeros(n_neighbors)
            vec_inds[:len(i[0])] = i[0]
            vec_dist[:len(i[1])] = i[1]
            indices.append(vec_inds)
            distances.append(vec_dist)        
        else:
            indices.append(i[0])
            distances.append(i[1])
    distances = np.vstack(distances)
    indices = np.vstack(indices)
    indices = indices.astype(np.int)
    if metric == 'l2':
        distances = np.sqrt(distances)
    
    return(distances, indices)

def Find_kNN(adata,n):
    if n == 'kNN':
        nb = fastKnn(X1=adata.obsm['X_pca'],n_neighbors=10)
    if n == 'outlier_kNN':
        nb = fastKnn(X2=adata[adata.obs['outlier']==-1].obsm['X_umap'],X1=adata[adata.obs['outlier']==1].obsm['X_umap'],n_neighbors=10)
    return nb 

def NN_vote(ref_data,nb,use_column):
    from tqdm import tqdm
    from collections import Counter
    result=[]
    nb = nb[1]
    for i in tqdm(range(len(nb))):
        neighbor = nb[i]
        temp=[]
        temp.extend(ref_data.obs.iloc[neighbor,][use_column].values) 
        max_counts = Counter(temp)
        top_one = max_counts.most_common(1)[0][0]
        result.append(top_one)
    return result

def detect_outlier(adata):
    index = []
    label_out = []
    score = []
    for i in adata.obs['leiden'].unique():
        clf = IsolationForest(n_estimators=500, contamination=0.1)
        temp = adata[adata.obs['leiden']==i]
        X = temp.obsm['X_umap']
        index.extend(temp.obs.index)
        clf.fit(X)
        y_pred = clf.predict(X)
        score_i = clf.score_samples(X)
        label_out.extend(y_pred)
        score.extend(score_i)
    a= []
    b = []
    c=[]
    for i in range(len(index)):
        a.append(index[i])
        b.append(label_out[i])
        c.append(score[i])
    ct_outlier_det = pd.DataFrame(list(zip(b, c)), columns =['label_out', 'score'],index=a)
    ct_outlier_det = ct_outlier_det.reindex(adata.obs.index)
    adata.obs['outlier'] = ct_outlier_det['label_out'].values
    return adata

def Predict_level1(data_path,model_path=level1_model,retrain_model=retrain_model,label_coder=label_coder,proba=0.5,level='first'):
    result = []
    data = read_data(data_path)
    model = load_model(model_path)
    retr_model = load_retrain_model(retrain_model)
    if 'SMART-seq' in data.obs.seq_tech.unique():
        data_smart = data[data.obs.seq_tech=='SMART-seq']
        data_10x = data[data.obs.seq_tech!='SMART-seq']
    else:
        data_10x = data
    for i in data_10x.obs.donor_id.unique():
        data_tmp = data_10x[data_10x.obs.donor_id==i]
        print("Begin Preprocessing Donor " + i + "......")
        data_cp = preprocessing_data(data_tmp,n_comps = 30,n_neighbors = 10,scale=True)
        predict_data = predict_preprocess(data_tmp,model)
        print("Begin Predict......")
        raw_predict= model.predict(predict_data)
        raw_predict_label = label_decode(raw_predict,label_coder,proba)
        data_cp.obs['raw_predict'] = raw_predict_label
        print("Begin Fix Prediction......")
        data_cp.obs['raw_predict_NN'] = NN_vote(data_cp,Find_kNN(data_cp,n='kNN'),use_column='raw_predict')
        data_cp = detect_outlier(data_cp)
        data_cp.obs['outlier_predict_NN'] = data_cp.obs['raw_predict_NN'].values
        cell_anno = data_cp.obs
        cell_anno.loc[cell_anno['outlier']==-1,'outlier_predict_NN'] = NN_vote(data_cp,Find_kNN(data_cp,n='outlier_kNN'),use_column='raw_predict_NN')
        print("Finish "+i+" Lecel1 Prediction!")
        result.append(cell_anno)
    if 'SMART-seq' in data.obs.seq_tech.unique():
        print("Begin Preprocessing the SMART-seq data......")
        data_cp = preprocessing_data(data_smart,n_comps = 30,n_neighbors = 10,scale=True,res=0.5)
        predict_data = predict_preprocess(data_smart,retr_model.booster_)
        print("Begin Predict......")
        raw_predict_label= retr_model.predict(predict_data)
        data_cp.obs['raw_predict'] = raw_predict_label
        print("Begin Fix Prediction......")
        data_cp.obs['outlier_predict_NN'] = NN_vote(data_cp,Find_kNN(data_cp,n='kNN'),use_column='raw_predict')
        cell_anno = data_cp.obs
        print("Finish SMART-seq data Level1 Prediction!")
        result.append(cell_anno)
    merge_result = pd.concat(result)
    merge_result = merge_result.reindex(data.obs.index)
    data.obs['level1'] = merge_result['outlier_predict_NN'].values
    if level == 'first':
        return data.obs
    if level == 'all':
        return data

def remain_label(data):
    anno = data.obs
    remain_ct = ['Cardiomyocyte cell', 'Pericyte', 'Smooth muscle cell', 'Unknown', 'Adipocyte']
    anno_remain = anno[anno.level1.isin(remain_ct)]
    anno_remain['level2'] = anno_remain['level1'].values
    anno_remain['level3'] = anno_remain['level1'].values
    anno_remain['level4'] = anno_remain['level1'].values
    return anno_remain

def Predict_all(data,anno_remain,model_dir_path):
    anno_list = []
    raw_anno = data.obs
    ct = ['Endothelial cell','Fibroblast','Lymphoid cell','Myeloid cell','Mesothelial cell']
    ct_model = load_ct_model(model_dir_path)
    for i in ct:
        print("Begin Predict " + i)
        data_tmp = data[data.obs.level1==i]
        data_cp = preprocessing_data(data_tmp,n_comps = min(50,data_tmp.shape[0]-1),n_neighbors = 5,scale=True,res=0.5)
        predict_data = predict_preprocess(data_tmp,ct_model[i].booster_)
        raw_predict_label= ct_model[i].predict(predict_data)
        data_cp.obs['level2'] = raw_predict_label
        print("Begin Fix Prediction......")
        if i == 'Endothelial cell':
            for ll in data_cp.obs.leiden.unique():
                data_l = data_cp[data_cp.obs.leiden==ll]
                core_result = pd.value_counts(data_l.obs.level2).index[0]
                data_l.obs['level2'] = core_result
                data_l.obs['level2'] = data_l.obs['level2'].str.split(' ', expand=True)[0]
                data_l.obs['level2'] = data_l.obs['level2'] + ' endothelial cell'
                data_l.obs['level3'] = data_l.obs['level2'].values
                data_l.obs['level4'] = data_l.obs['level2'].values
                anno_list.append(data_l.obs)
        elif i == 'Lymphoid cell':
            for ll in data_cp.obs.leiden.unique():
                data_l = data_cp[data_cp.obs.leiden==ll]
                core_result = pd.value_counts(data_l.obs.level2).index[0]                        
                data_l.obs['level2'] = core_result
                data_l.obs['level2'] = data_l.obs['level2'].str.split('_', expand=True)[0]
                data_l.obs['level3'] = data_l.obs['level2'].values
                data_l.obs['level4'] = data_l.obs['level2'].values
                if len(data_l.obs['level2'].unique()) > 1:
                    data_l.obs['level2'] = 'T cell'
                anno_list.append(data_l.obs)
        else:
            for ll in data_cp.obs.leiden.unique():
                data_l = data_cp[data_cp.obs.leiden==ll]
                core_result = pd.value_counts(data_l.obs.level2).index[0]
                data_l.obs['level2'] = core_result
                data_l.obs['level2'] = data_l.obs['level2'].str.split('_', expand=True)[0]
                data_l.obs['level3'] = data_l.obs['level2'].values
                data_l.obs['level4'] = data_l.obs['level2'].values
                anno_list.append(data_l.obs)
    anno_list.append(anno_remain)
    merge_result = pd.concat(anno_list)
    merge_result = merge_result.reindex(raw_anno.index)
    final_result = pd.DataFrame()
    final_result['cell_id'] = merge_result.index
    final_result['level1'] = merge_result['level1'].values
    final_result['level2'] = merge_result['level2'].values
    final_result['level3'] = merge_result['level3'].values
    final_result['level4'] = merge_result['level4'].values
    return final_result



if __name__ == '__main__':
    parser = arg.ArgumentParser(description=DESCRIPTION.Description)
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    parser.add_argument('--input',required = True, type=str, default ='./data/data_input.h5ad', help='input the h5ad data') 
    parser.add_argument('--levels', required = True, default=['first','all']  , help='Predicted level choose')    
    parser.add_argument('--output', required = True, type=str, default ='./result/result_save.csv', help='path to save the result')    
    args = parser.parse_args()
    label_coder = preprocessing.LabelEncoder()
    label_coder.classes_ = np.load('./model/cell_type_encode.npy') 
    level1_model = './model/Level.model'
    retrain_model = './model/Retrain.model'
    if args.levels == 'first':
        result = Predict_level1(args.input,level1_model,retrain_model,level='first')
        result.to_csv(args.output)
    if args.levels == 'all':
        first_data = Predict_level1(args.input,level1_model,retrain_model,level='all')
        anno_remain = remain_label(first_data)
        final = Predict_all(first_data,anno_remain,'../model/model/')
        final.to_csv(args.output)


