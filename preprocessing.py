
import argparse
import os
import warnings

import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print('modules import completed')

def create_unique_index(data):
#     idx_series = data.index.to_series()
#     idx_gb = idx_series.groupby([data.HBO_UUID, data.PERIOD_RANK])
#     idx_tf = idx_gb.transform("first")
#     data["UNIQUE_ID"]=idx_tf
    data["UNIQUE_ID"]=data["HBO_UUID"]+":"+data["PERIOD_RANK"]
    print(f"data head: {data[['UNIQUE_ID', 'HBO_UUID', 'PERIOD_RANK']].head()}")
#     assert data.UNIQUE_ID.nunique()==data.HBO_UUID.nunique()*data.PERIOD_RANK.nunique()
    df_step_1=data.set_index(['UNIQUE_ID'])
    return df_step_1

def fill_missing(data):
    data.FIRST_WATCHED_ASSET_CLASS_SUB_ADJ = data.FIRST_WATCHED_ASSET_CLASS_SUB_ADJ.str.replace("unknown","missing")          
    return data

def transform(data):
    print(f"input data shape: {data.shape}")
    print(f"input data size: {data.memory_usage(index=True).sum()}")
    print(f"original dataframe head: {data.head()}")
    print(f"original dataframe index name: {data.index.name}")
    print(f"original dataframe index 5 values: {data.index[:5]}")
    cols_to_one_hot_encode=["PROVIDER"
                        , "FIRST_WATCHED_ASSET_CLASS_SUB_ADJ"
                        , "FT_SEGMENT", "FT_SUB_SEGMENT"]

    id_list=["UNIQUE_ID", "HBO_UUID"]
    cols_to_exclude_from_num_conversion=cols_to_one_hot_encode+id_list

    obj_to_num_col_list=[col for col in data.select_dtypes(include=['object']).columns if col not in cols_to_exclude_from_num_conversion]

    counter=0
    for col in obj_to_num_col_list:
        counter+=1
        print(f"will process column {col}")
        data[col]=data[col].astype(np.float16)
        print(f"processed column {col}")
        print(f"No. of columns processed: {counter}")
     
    print("size of data after converting object to numeric columns")
    print(f"input data size: {data.memory_usage(index=True).sum()}")
            
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.uint8)
    print(f"Columns to onehot encode: {cols_to_one_hot_encode}")
    encode_df=data[cols_to_one_hot_encode]
    print(f"shape of dataframe to be onehot encoded: {encode_df.shape}")
    encoded_mtx=encoder.fit_transform(encode_df).toarray()
    encoded_df=pd.DataFrame(encoded_mtx, index=data.index, columns=encoder.get_feature_names())
    print("size of categorical column data after one hot encoding")
    print(f"encoded_df data size: {encoded_df.memory_usage(index=True).sum()}")
    print(f"shape of categorical dataframe after onehot encoding: {encoded_df.shape}")
    print(f"encoded dataframe head: {encoded_df.head()}")
    print(f"encoded dataframe index name: {encoded_df.index.name}")
    print(f"encoded dataframe index 5 values: {encoded_df.index[:5]}")
    
    del encode_df
    del encoded_mtx

    df_step_1=pd.concat([data, encoded_df], axis=1)
    
    print("size of combined data set with onehot encoded columns")    
    print(f"shape of full dataframe after merging onehot encoded dataframe: {df_step_1.shape}")
    print(f"df_step_1 data size: {df_step_1.memory_usage(index=True).sum()}")
    
    del data
    del encoded_df
    
    # Keeping column "FT_SEGMENT" in the final to join it back while presenting results
    cols_to_drop=[col for col in cols_to_one_hot_encode if col not in "FT_SEGMENT"]
    # If we do not need columns in the final dataframe, we should drop cols_to_one_hot_encode
    df_step_2=df_step_1.drop(cols_to_drop, axis=1)
    del df_step_1
    df_step_2["UNIQUE_ID"]=df_step_2.index
    return df_step_2

# def add_weight_col(data, weight):
#     stream_mask = (data["NUM_STREAMS_ADJ"]>0)
#     dormant_mask = (data["NUM_STREAMS_ADJ"]==0)
#     data.loc[stream_mask, "WEIGHT"] = 1-weight
#     data.loc[dormant_mask, "WEIGHT"] = weight
#     cols=list(data.columns)
#     cols.insert(1, cols.pop(cols.index("WEIGHT")))
#     data = data.loc[:, cols]
#     return data

def split_dataframe(df, strat_column, target, test_size=0.1):
    df_train,df_test = train_test_split (df, test_size = test_size, stratify=df[[strat_column, target]], random_state=101)
    df_train,df_val = train_test_split (df_train, test_size = test_size, stratify=df_train[[strat_column, target]], random_state=101)
    
    print(f"Distribution by PERIOD_RANK and FLG_TARGET")
    
    print(f"train: {df_train.groupby(by=[strat_column, target]).size().reset_index().rename(columns={0:'COUNT_UUID'})}")
    print(f"test: {df_test.groupby(by=[strat_column, target]).size().reset_index().rename(columns={0:'COUNT_UUID'})}")
    print(f"val: {df_val.groupby(by=[strat_column, target]).size().reset_index().rename(columns={0:'COUNT_UUID'})}")
    
    df_train_X,df_train_y = df_train.drop([target],axis=1), df_train[[target]]
    df_test_X,df_test_y = df_test.drop([target],axis=1), df_test[[target]]
    df_val_X,df_val_y = df_val.drop([target],axis=1), df_val[[target]]
    return df_train_X, df_train_y , df_test_X, df_test_y, df_val_X, df_val_y

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.1)
    parser.add_argument('--csv-weight', type=float, default=0.7)
    args, _ = parser.parse_known_args()
    
    # how train-test-split-ratio will pick up value from args.train_test_split_ratio
    split_ratio = args.train_test_split_ratio
    csv_weight = args.csv_weight
    
    input_data_path = os.path.join('/opt/ml/processing', 'input')
    
    print('reading csv files from s3')
    all_files = glob.glob(os.path.join(input_data_path, "*.csv"))
    print(f"all files: {all_files}")
    df = pd.concat((pd.read_csv(f, header=0, encoding="utf-8") for f in all_files))
    
    print(len(df.columns),'\n')
    print(df.columns)
    
    print('read file completed ')
    
    print('transorming input data')
    
    clean_df_step_1=create_unique_index(df)
    clean_df_step_2=fill_missing(clean_df_step_1)
    clean_df_step_3=transform(clean_df_step_2)
#     clean_df_step_4=add_weight_col(clean_df_step_3, csv_weight)
    
    print('transformation completed','\n')
     
    df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y=split_dataframe(clean_df_step_3, "PERIOD_RANK", "FLG_TARGET", split_ratio)
    
    print(f"train X shape: {df_train_X.shape}")
    print(f"train y shape: {df_train_y.shape}")
    print(f"test X shape: {df_test_X.shape}")
    print(f"test y shape: {df_test_y.shape}")
    print(f"val X shape: {df_val_X.shape}")
    print(f"val y shape: {df_val_y.shape}")
    
    print(f"data split according to period rank")
    
    print(f"train data: {df_train_X.PERIOD_RANK.value_counts(dropna=False)}")
    print(f"test data: {df_test_X.PERIOD_RANK.value_counts(dropna=False)}")
    print(f"val data: {df_val_X.PERIOD_RANK.value_counts(dropna=False)}")
    
    print(f"Normalized data split according to period rank")
    
    print(f"Normalized train data: {df_train_X.PERIOD_RANK.value_counts(dropna=False, normalize=True)}")
    print(f"Normalized test data: {df_test_X.PERIOD_RANK.value_counts(dropna=False, normalize=True)}")
    print(f"Normalized val data: {df_val_X.PERIOD_RANK.value_counts(dropna=False, normalize=True)}")
    
    print('Extracting side info')
    
    side_info_cols=["UNIQUE_ID", "HBO_UUID", "PERIOD_RANK", "FT_SEGMENT"]
    df_train_side_info=df_train_X[side_info_cols]
    df_test_side_info=df_test_X[side_info_cols]
    df_val_side_info=df_val_X[side_info_cols]
    
    # keep PERIOD_RANK in the train, test and val sets
    cols_to_drop=[col for col in side_info_cols if col!="PERIOD_RANK"]
    df_train_X = df_train_X.drop(cols_to_drop, axis=1)
    df_test_X = df_test_X.drop(cols_to_drop, axis=1)
    df_val_X = df_val_X.drop(cols_to_drop, axis=1)
    
    assert "PERIOD_RANK" in df_train_X.columns
    assert "PERIOD_RANK" in df_test_X.columns
    assert "PERIOD_RANK" in df_val_X.columns
    
    print(f"Side info train X shape: {df_train_side_info.shape}")
    print(f"Side info test X shape: {df_test_side_info.shape}")
    print(f"Side info val X shape: {df_val_side_info.shape}")
    
    print(f"train X shape: {df_train_X.shape}")
    print(f"train y shape: {df_train_y.shape}")
    print(f"test X shape: {df_test_X.shape}")
    print(f"test y shape: {df_test_y.shape}")
    print(f"val X shape: {df_val_X.shape}")
    print(f"val y shape: {df_val_y.shape}")
    
    print("Extracted side info")
    
    print("Extracting column names")
    
    print(f"Number of train columns: {df_train_X.shape[1]}")
    print(f"Number of test columns: {df_test_X.shape[1]}")
    print(f"Number of val columns: {df_val_X.shape[1]}")

    df_train_X.columns.to_series().to_csv("/opt/ml/processing/train/train_columns.csv", header = False, index = False)
    df_test_X.columns.to_series().to_csv("/opt/ml/processing/test/test_columns.csv", header = False, index = False)
    df_val_X.columns.to_series().to_csv("/opt/ml/processing/val/val_columns.csv", header = False, index = False)
        
    print("Extracted column names")
        
    print("writing files back to s3")
    
    # change headers back to False after validation
    pd.concat([df_train_y,df_train_X],axis=1).to_csv("/opt/ml/processing/train/train.csv", header = 0, index = False)
    pd.concat([df_val_y,df_val_X],axis=1).to_csv("/opt/ml/processing/val/val.csv", header = 0, index = False)
    pd.concat([df_test_y,df_test_X],axis=1).to_csv("/opt/ml/processing/test/test.csv", header = 0, index = False)
    
    df_train_side_info.to_csv("/opt/ml/processing/train/train_side_info.csv", header = 0, index = False)
    df_test_side_info.to_csv("/opt/ml/processing/val/test_side_info.csv", header = 0, index = False)
    df_val_side_info.to_csv("/opt/ml/processing/test/val_side_info.csv", header = 0, index = False)
    
    print("processing write to s3")
    
    
