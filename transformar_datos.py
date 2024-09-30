import pandas as pd
import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import hashlib
from joblib import Parallel, delayed
import re
def create_list_columns(dataset, label, value):
        value = str(value)  # Ensure value is a string
        return pd.DataFrame({label + '_' + value: dataset[label].apply(lambda x: columna_lista_contiene_valor(x, value))})

def hash_value(value, label):
    combined_value = f"{label}_{value}"
    return int(hashlib.md5(combined_value.encode()).hexdigest(), 16)

def columna_lista_contiene_valor(lista, valor):
    # Si la lista es nula, devolver 0 inmediatamente
    if pd.isna(lista):
        return 0
    
    # Usar expresión regular para eliminar corchetes, comillas y espacios
    lista = re.sub(r'[\[\]" ]', '', lista)
    
    # Dividir la lista por comas y verificar si el valor está presente
    return int(valor in lista.split(','))
    
# Function to extract unique list values
def extract_list_values(dataset, list_labels):
    dic_values = {}
    for label in list_labels:
        temp = dataset[label].str.replace(r'[\[\]"]', '', regex=True).str.replace(' ', '')
        values = set()
        for val in temp.dropna():
            values.update(val.split(','))
        dic_values[label] = values
    return dic_values
def transformo_dataset(dataset,t,category_labels,boolean_labels,list_labels,dic,dic_values):
    
    print(f"Processing dataset {t}...")
    
    # Convert auction_time to datetime and adjust for timezone
    dataset['fecha'] = pd.to_datetime(dataset['auction_time'], unit='s') + pd.to_timedelta(dataset['timezone_offset'], unit='h')

    # Extract time components
    dataset['dia_semana'] = dataset['fecha'].dt.dayofweek
    dataset['hora'] = dataset['fecha'].dt.hour
    dataset['minuto'] = dataset['fecha'].dt.minute

    # One hot encoding for boolean columns
    # boolean_labels = [x for x in dataset.columns if 'boolean' in x]

    # # List of category labels
    # category_labels = [x for x in dataset.columns if 'categorical' in x] + ['gender', 'has_video', 'auction_age', 'creative_width', 'creative_height', 'device_id_type'] + boolean_labels

    new_columns_list = []
    
    # Calculate unique categories in the first iteration (train_data)
    # if t == t:
    #     dic = {label: dataset[label].nunique() for label in category_labels}
    #     list_labels = [x for x in dataset.columns if 'ction_list_' in x]
    #     dic_values = extract_list_values(dataset, list_labels)

    # Generate new columns for categorical variables
    for label in category_labels:
        num_categories = dic[label]
        if num_categories <= 15:
            # Use pandas get_dummies for one-hot encoding
            new_columns_list.append(pd.get_dummies(dataset[label], prefix=label))

    # Create new columns for list-type values
    

    for label in list_labels:
        # Use parallel processing to speed up column creation
        new_columns = Parallel(n_jobs=-1)(
            delayed(create_list_columns)(dataset, label, value) for value in dic_values[label]
        )
        new_columns_list.extend(new_columns)

    # Concatenate new columns to the original dataset
    dataset = pd.concat([dataset] + new_columns_list, axis=1)
    
    # Save the updated dataset to CSV
    

    #dataset.to_csv(rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data{t}.csv", sep=",", index=False)
    dataset.to_parquet(rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data{t}_{test}_{submit}.parquet", index=False)


    return dataset


test = False
submit = True

# Load part of the train data only the first 10000 rows
if test:
    train_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_20.csv")
    train_data = train_data.sample(frac=1/10)
    print(train_data.head())
    eval_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_21.csv")
    eval_data = eval_data.sample(frac=1/10)
elif submit:

    # Parameters for XGBoost
    params = { 
        'max_depth': 5,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 12345
    } 
    # Initialize an empty model to update incrementally
    bst = None

    # Define the chunk size (number of rows per chunk)
    chunk_size = 1000000000  # Adjust this value based on your available memory

    # Read data in chunks and process incrementally
    for n in range(15, 22):
        print(f"Processing dataset {n}...")
        
        train = pd.read_parquet(
            rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data{n}_{test}_{submit}.parquet"
        )
        
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

        #elimino columnas que no estaban en el primer dataset
        X_train = X_train.drop(columns=[col for col in X_train.columns if col not in columns])

        X_train = X_train.select_dtypes(include='number')

        # Create DMatrix for this chunk
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Train the model incrementally
        if bst is None:
            bst = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=10)
        else:
            bst = xgb.train(params, dtrain, num_boost_round=10, xgb_model=bst, verbose_eval=10)

        # Free up memory after processing the chunk
        del X_train, y_train, dtrain
        gc.collect()

    # Save the final model
    bst.save_model(rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\xgb_final_model.json")
    print('processing eval data')
    eval_data = pd.read_parquet(rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data22_{test}_{submit}.parquet")

    non_numeric_columns = eval_data.select_dtypes(exclude=['number']).columns
    eval_data = eval_data.drop(columns=non_numeric_columns)
    df = pd.DataFrame(columns=columns)
    eval_data = pd.concat([eval_data, df], axis=0)
    eval_data = eval_data.drop(columns=[col for col in eval_data.columns if col not in columns])
    eval_data = eval_data.select_dtypes(include='number')
    
    # eval_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_test.csv")
    # # eval_data = eval_data.sample(frac=1/10)
    # eval_data = transformo_dataset(eval_data, 22, category_labels, boolean_labels, list_labels, dic, dic_values)

else:
    n = 1000
    train_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_20.csv", nrows=n)
    
    print(train_data.head())
    eval_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_21.csv", nrows=n)

datasets = [train_data, eval_data]
t = 0




# Precompute the unique categorical values and list values only for train_data (first dataset)
dic = {}
dic_values = {}



    




print("Finished processing datasets")

if test:
    #entreno modelo de xgboost


    y_train = train_data["Label"]
    X_train = train_data.drop(columns=["Label"])
    X_train = X_train.select_dtypes(include='number')
    del train_data
    gc.collect()
    y_val = eval_data["Label"]
    X_val = eval_data.drop(columns=["Label"])
    X_val = X_val.select_dtypes(include='number')

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = { 
        'max_depth': 10,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 12345
    }

    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=10, verbose_eval=10)

    # Predict on the evaluation set
    y_preds = bst.predict(dval)

    # Make the submission file
    # submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
    # submission_df["id"] = submission_df["id"].astype(int)
    # submission_df.to_csv("basic_model.csv", sep=",", index=False)
    # print(submission_df.head())

    # Print the AUC score
    print(roc_auc_score(y_val, y_preds))

    # print the feature importance in a txt file and performance
    print(bst.get_fscore())
    with open(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\feature_importance.txt", "a") as file:
        file.write(f"Performance: {roc_auc_score(y_val, y_preds)}")
        file.write(str(bst.get_fscore())+'\n\n\n')
        
if submit:
    

    # y_train = train_data["Label"]
    # X_train = train_data.drop(columns=["Label"])
    # X_train = X_train.select_dtypes(include='number')
    # del train_data
    # gc.collect()
  
    # dtrain = xgb.DMatrix(X_train, label=y_train)

    # params = { 
    #     'max_depth': 13,
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    #     'seed': 12345,
    # }

    # bst = xgb.train(params, dtrain, num_boost_round=1000, verbose_eval=10)

    # del X_train
    # del y_train
    # del dtrain
    # gc.collect()
    

    X_val = eval_data.select_dtypes(include='number')

    X_val = X_val.drop(columns=["id"])
    dval = xgb.DMatrix(X_val)

    # Predict on the evaluation set
    y_preds = bst.predict(dval)
    


    # make the submission file
    submission_df = pd.DataFrame({"id": eval_data["id"], "label": y_preds})
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df.to_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\basic_model.csv", sep=",", index=False)
    print(submission_df.head())

    # Print the AUC score
 

    # print the feature importance in a txt file and performance
    print(bst.get_fscore())
    with open(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\feature_importance.txt", "a") as file:
        # file.write(f"Performance: {roc_auc_score(y_val, y_preds)}")
        file.write(str(bst.get_fscore())+'\n\n\n')





#alarma de finalización
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

winsound.Beep(frequency, duration)

