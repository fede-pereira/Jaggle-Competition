import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt


# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

dataset=pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_21.csv")
# dataset= dataset.sample(frac=1/10)

dataset['fecha'] = pd.to_datetime(dataset['auction_time'], unit='s') + pd.to_timedelta(dataset['timezone_offset'], unit='h')

# Extract time components
dataset['dia_semana'] = dataset['fecha'].dt.dayofweek
dataset['hora'] = dataset['fecha'].dt.hour
dataset['minuto'] = dataset['fecha'].dt.minute

dataset.to_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_21_con_hora.csv")
jdneij

# train= train.sample(frac=1/25)
print(dataset.shape)
# visulizar los datos
# balance de clase Label en barras
# train['Label'].value_counts().plot(kind='bar')
# plt.show()

#histograma por hora de a 10 minutos con legenda de Label

train['hora_minuto'] = train['hora'].astype(str) + train['minuto'].astype(str)

# junto los minutos en steps de a 10
train['hora_minuto'] = train['hora_minuto'].str[:3] + '0'

#plot histogram of hora_minuto by Label = 1 ordenados por hora_minuto
train[train['Label']==1]['hora_minuto'].value_counts().plot(kind='bar')



#plot histogram of hora_minuto by Label = 0
train[train['Label']==0]['hora_minuto'].value_counts().plot(kind='bar')


plt.show()




#print(train.head())

jihdeuihde
train.to_excel(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\train_data16_False_True.xlsx")

# Load part of the train data
train_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_21.csv")

# Load the test data
eval_data = pd.read_csv(r"C:\Users\fpereira\OneDrive - BYMA\Documentos\GitHub\Jaggle-Competition\ctr_test.csv")

# Train a tree on the train data
train_data = train_data.sample(frac=1/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')
del train_data
gc.collect()

cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
cls.fit(X_train, y_train)

# Predict on the evaluation set
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["id"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)
