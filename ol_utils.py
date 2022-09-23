from google.cloud import bigquery_storage
from google.cloud.bigquery_storage import types
import pandas
import os
from sklearn import metrics
from typing import Counter
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function
def get_dataset(table_name):

    your_project_id = 'obvious-labs'
    project_id = "obvious-labs"
    dataset_id = "coinscale"#"DAMR"
    table_limit = None
    parent = "projects/{}".format(your_project_id)

    table = f"projects/{project_id}/datasets/{dataset_id}/tables/{table_name}"

    bqstorageclient = bigquery_storage.BigQueryReadClient()

    read_options = types.ReadSession.TableReadOptions(
        #selected_fields=["species_common_name", "fall_color"]
    )

    requested_session = types.ReadSession(
        table=table,
        data_format=types.DataFormat.ARROW,
        read_options=read_options,
    )
    read_session = bqstorageclient.create_read_session(
        parent=parent,
        read_session=requested_session,
        max_stream_count=1,
    )

    stream = read_session.streams[0]
    reader = bqstorageclient.read_rows(stream.name)

    frames = []
    for message in reader.rows().pages:
        frames.append(message.to_dataframe())
    df = pandas.concat(frames)

    if table_limit is not None:
        df = df.head(table_limit)

    return df

# Function
def run_model(modelToRun, X_train, X_test, y_train, y_test):
    global models_score  

    time_start = pandas.Timestamp.now()
    print("Model, starting FIT: ",time_start)

    #sample_weight = [x for x in range(0, len(X_train))]
    modelToRun.fit(X_train, y_train, epochs=10)#, sample_weight = sample_weight
    time_total = pandas.Timestamp.now()-time_start

    print("Model, ending FIT: ",pandas.Timestamp.now())
  
    y_pred = modelToRun.predict(X_test)
    y_pred = tf.round(y_pred)
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred), "Recall:", metrics.recall_score(y_test,y_pred))

    data = {'Creation time' : int(round(time_total.total_seconds()/60, 0)),
            'Model':modelToRun,
            #'Score': modelToRun.score(X_test, y_test),
            'Accuracy':metrics.accuracy_score(y_test,y_pred),
            'Recall':metrics.recall_score(y_test,y_pred),
            'PR AUC':metrics.average_precision_score(y_test, y_pred),
            'f1_weighted':metrics.f1_score(y_test, y_pred,average='weighted'),
            'Precision p/class':metrics.precision_score(y_test, y_pred, average=None),
            'Recall p/class':metrics.recall_score(y_test, y_pred, average=None)
          }
    data

    #models_score = models_score.append(data,ignore_index=True)
    return modelToRun

# Function
def get_metrics(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred = tf.round(y_pred)

    data = {'Model':model,
            #'Score': modelToRun.score(X_test, y_test),
            'Accuracy':metrics.accuracy_score(y_test,y_pred),
            'Recall':metrics.recall_score(y_test,y_pred),
            'PR AUC':metrics.average_precision_score(y_test, y_pred),
            'f1_weighted':metrics.f1_score(y_test, y_pred,average='weighted'),
            'Precision p/class':metrics.precision_score(y_test, y_pred, average=None),
            'Recall p/class':metrics.recall_score(y_test, y_pred, average=None)
          }

    print(data)

    return data


# Function
def clean_dataset(df):
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]

# Function
def balance(X, y):
    rus = RandomUnderSampler()
    print('Original dataset shape %s' % Counter(y))

    return rus.fit_resample(X, y)

# Function
def local_train_test_split(X, y, sta, end):
    
    if(len(X) != len(y)) :
        raise Exception("X and y not have different lenghts.")

    length = len(X)

    pos_sta = int(length*sta)
    pos_end = int(length*end)

    X_train = pandas.concat([X.loc[:pos_sta], X.loc[pos_end:length]])
    X_test = X.loc[pos_sta:pos_end]

    y_train = pandas.concat([y.loc[:pos_sta], y.loc[pos_end:length]])
    y_test = y.loc[pos_sta:pos_end]
    
    return X_train, X_test, y_train, y_test

# Function
def local_train_test_split_regular(X, y):
  return train_test_split(X, y, random_state=0, test_size=0.15, shuffle=True)

# Function
def prepare_dataset(df):
    
    print("Before cleaninig:",len(df))
    clean_dataset(df)
    print("After cleaninig:",len(df))
    
    X = df.drop(["target"], axis=1)
    y = df["target"].copy()

    X_resampled, y_resampled = balance(X, y)

    return local_train_test_split(X_resampled, y_resampled)

# Function
def get_data(table_name):
    df = get_dataset(table_name)
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    Counter(y_test)
    return X_train, X_test, y_train, y_test

def set_credential_path(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

#get_data()