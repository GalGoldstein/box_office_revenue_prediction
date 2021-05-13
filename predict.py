import argparse
import numpy as np
import pandas as pd
from pickle import load
from preprocess_data import prepare_df_for_ml
from pycaret.regression import predict_model, load_model
from train_eval import rmsle
import os
import sys

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

# load model and transformation to pre-process data
model = load_model(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'catboost_model'))
trans = load(open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'trans.pkl'), 'rb'))
scaler = load(open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'scaler.pkl'), 'rb'))

X, y = prepare_df_for_ml(df=data, zerotonan=True)
X['revenue'] = y

# scale popularity for 2019 data
for i, row in X.iterrows():
    if row['release_date'] == 2019:
        new_popularity = np.float64(scaler.transform(np.array([X.at[i, 'popularity']]).reshape(-1, 1)))
        X.at[i, 'popularity'] = new_popularity

# transform df2numpy
X, y = trans.transform(X)

# transform back to df
df = pd.DataFrame(X)
df.columns = df.columns.astype(str)

preds = predict_model(model, data=df)['Label']
preds = np.maximum(np.expm1(preds), np.zeros(len(preds)))  # because we predicted the log(1+x) values

# build the predictions dataframe
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = preds

# save results to .csv
prediction_df.to_csv("prediction.csv", index=False, header=False)

res = rmsle(data['revenue'], prediction_df['revenue'])
print("RMSLE is: {:.6f}".format(res))
