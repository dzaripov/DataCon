from catboost import CatBoostRegressor
import pandas as pd
import sys
import argparse


# Параметры, полученные при оптимизации гиперпараметров программой Optuna
params = {'loss_function': 'RMSE',
          'l2_leaf_reg': 0.10316964527875554,
          'colsample_bylevel': 0.09541076028546017,
          'depth': 10,
          'boosting_type': 'Plain',
          'bootstrap_type': 'MVS',
          'min_data_in_leaf': 18,
          'one_hot_max_size': 2}


parser=argparse.ArgumentParser(
    description='''Prediction for cell viability by nanoparticle data. Please specify a dataset in script arguments''')
parser.add_argument('dataset', nargs='*', default=[1], help='')
args=parser.parse_args()

path_to_file = sys.argv[1]


data = pd.read_csv('db.csv').drop(columns=['is_human_cell', 'is_cancer_cell', 'cell_age'])
X = data.drop(columns=['viability'], axis=1)
y = data['viability']
cat_features = ['material', 'material_type', 
                'surf_charge_cat', 'cell_type', 
                'cell_origin',  'cell_line']

print('Model is learning to predict viability')
model = CatBoostRegressor(**params, learning_rate=0.003, iterations=50000, task_type='CPU')
model.fit(X, y, cat_features=cat_features, verbose=0)


if path_to_file.endswith('.xslx') or path_to_file.endswith('.xsl'):
    df = pd.read_excel(path_to_file)
if path_to_file.endswith('.csv'):
    df = pd.read_csv(path_to_file)
else:
    print('Please, provide nanoparticle data in .csv or excel format with the same columns as db.csv')
    exit()

df = df.drop(columns=['is_human_cell', 'is_cancer_cell', 'cell_age'])

if "viability" in df.columns:
    df = df.drop(columns=['viability'], axis=1)

preds = model.predict(df)
df['viability'] = preds
df.to_csv('prediction.csv', index=False)
print('prediction.csv is out!')
