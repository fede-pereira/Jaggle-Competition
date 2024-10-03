import pandas as pd
import numpy as np
import xgboost as xgb
import gc
from sklearn.metrics import roc_auc_score

#test y submit
test = False
submit = True

#set seed pandas
np.random.seed(12)


#combinacion de hiperparametros
hipers = []
performance = {}
for max_depth in [ 5, 7, 9]:
    for learning_rate in [0.001,0.005,0.007,0.01, 0.05]:
        for min_child_weight in [5, 7,9,10]:
            hipers.append((max_depth, learning_rate, min_child_weight))
            performance[(max_depth, learning_rate, min_child_weight)] = 0

first = True
df = pd.DataFrame(performance.items(), columns=['hiper', 'auc'])

df.to_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\hiper_results.csv", index=False)
for max_depth, learning_rate, min_child_weight in hipers:
    # Parameters for XGBoost
    params = { 
        'max_depth': max_depth,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': learning_rate,
        'seed': 12,
        'min_child_weight': min_child_weight,
    } 
    # Initialize an empty model to update incrementally
    bst = None

    # Define the chunk size (number of rows per chunk)
    chunk_size = 1000000000  # Adjust this value based on your available memory
    eval_data = pd.read_parquet(
            rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data21_{test}_{submit}.parquet"
        )
    eval_data = eval_data.sample(frac=1/10)
    non_numeric_columns = eval_data.select_dtypes(exclude=['number']).columns
    eval_data = eval_data.drop(columns=non_numeric_columns)
    # Read data in chunks and process incrementally
    for n in range(15, 21):
        print(f"Processing dataset {n}...")
        
        train = pd.read_parquet(
            rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data{n}_{test}_{submit}.parquet"
        )
        train= train.sample(frac=1/20)
        print('traing' )

        # Separate label and features
        y_train = train["Label"]
        X_train = train.drop(columns=["Label"])
        del train
        gc.collect()
        # Select numeric columns only
        X_train = X_train.select_dtypes(include='number')

        if n == 15:
            columns = X_train.columns
            
        df = pd.DataFrame(columns=columns)
        X_train = pd.concat([X_train, df], axis=0)
        if n == 15 and first:
            first = False
            eval_data = pd.concat([eval_data, df], axis=0)
            y_val = eval_data["Label"]

            X_val = eval_data.drop(columns=["Label"])
            X_val = X_val.drop(columns=[col for col in X_val.columns if col not in columns])
            X_val = X_val.select_dtypes(include='number')
            deval = xgb.DMatrix(X_val, label=y_val)

        #elimino columnas que no estaban en el primer dataset
        X_train = X_train.drop(columns=[col for col in X_train.columns if col not in columns])

        X_train = X_train.select_dtypes(include='number')

        # Create DMatrix for this chunk
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Train the model incrementally
        if n == 15:
            
            bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=10,evals=[(dtrain, 'train'), (deval, 'eval')], early_stopping_rounds=10)
        else:
            bst = xgb.train(params, dtrain, num_boost_round=100, xgb_model=bst, verbose_eval=10,evals=[(dtrain, 'train'), (deval, 'eval')], early_stopping_rounds=10)

        # Free up memory after processing the chunk
        del X_train, y_train, dtrain
        gc.collect()



    print('processing eval data')

    # save performance

    # Evaluate the model on the evaluation data
    y_pred = bst.predict(deval)
    
    auc = roc_auc_score(y_val, y_pred)
    print(f"AUC: {auc}")
    performance[(max_depth, learning_rate, min_child_weight)] = auc
    # Free up memory


    #guardo resultados
    df = pd.DataFrame(performance.items(), columns=['hiper', 'auc'])

    df.to_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\hiper_results.csv", index=False)
