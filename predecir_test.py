import pandas as pd 
import xgboost as xgb
import pyarrow as pa
import pyarrow.parquet as pq

import gc

# Load only the column names from the training data from 15 to 21
columnas = set()
for i in range(15, 22):
    colums = pq.read_table(rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data{i}_False_True.parquet").column_names
    columnas.update(colums)
columnas = list(columnas)

print(columnas)

#columns = pd.read_parquet(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data15_False_True.parquet", columns=None).select_dtypes(include='number').columns

path_model = r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\xgb_final_model_2.json"
#load model
model = xgb.Booster()
model.load_model(path_model)

# Load the test data
eval_data = pd.read_parquet(rf"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data22_False_True.parquet")
eval_data = eval_data.apply(pd.to_numeric, errors='coerce')
non_numeric_columns = eval_data.select_dtypes(exclude=['number']).columns
train_data = eval_data.drop(columns=non_numeric_columns)
df = pd.DataFrame(columns=columnas)
train_data = pd.concat([train_data, df], axis=0)
train_data = train_data.apply(pd.to_numeric, errors='coerce')
train_data = train_data.drop(columns=[col for col in train_data.columns if col not in columnas])
train_data = train_data.select_dtypes(include='number')


# Predict on the evaluation set
y_preds = model.predict(xgb.DMatrix(train_data))

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("xgb_model.csv", sep=",", index=False)
