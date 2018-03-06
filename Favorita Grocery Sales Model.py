import pandas as pd
import numpy as np
import time

print("Loading data...")

# read datasets
dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32', 'onpromotion':'bool' }

train = pd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=["date"], skiprows=range(1, 106672217)) # Cargamos los datos desde 8 de Agosto de 2016 en adelante
test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date'])
item = pd.read_csv('../input/items.csv')
stores = pd.read_csv('../input/stores.csv')
transactions = pd.read_csv('../input/transactions.csv', parse_dates=['date'])
holidays = pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date'])
oil = pd.read_csv('../input/oil.csv', parse_dates=['date'])

print("Loading complete")

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def transformations(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dow'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    if 'unit_sales' in df.columns.values:
       df.loc[(df.unit_sales < 0),'unit_sales'] = 0 # Eliminar Valores Negativos
       df["unit_sales"] = df["unit_sales"].apply(np.log1p)
    df = df.drop(['id','date'], axis=1)
    df = df.fillna(0)
    return df

def NWRMSLE(y, pred, w):
    return mean_squared_error(y, pred, sample_weight=w)**0.5

test_id = test["id"]

print("Appling transformations...")

train.loc[(train.unit_sales < 0),'unit_sales'] = 0 # Eliminar Valores Negativos
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #Calcular Logaritmo de Unidades
train['dow'] = train['date'].dt.dayofweek

print("Calculating mean of daily sales...")

#Promedio de Ventas Diarias
ma_dw = train.groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw').reset_index()

print("Calculating mean of weekly sales...")

#Promedio de Ventas Semanales
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(
       ['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk').reset_index()

# Eliminar columnas de id, promociones y Dia de la Semana
train = train.drop(['id','onpromotion','dow'],axis=1)

# Se crean registros para todos los items, todas las Tiendas en todas las Fechas
# para el correcto calculo del promedio de ventas diarias

# Busca valores únicos de Items, Tiendas, Fechas
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()

# Hace index por Fecha, Tienda, Item
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)

print("Sorting sales...")

train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
).reset_index()

# Rellenar con NaN
train['unit_sales'].fillna(0, inplace=True)

# Toma la Última Fecha
lastdate = train.iloc[train.shape[0]-1].date

# Calcula el promedio de ventas por item y tienda
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(
        ['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais')

from datetime import timedelta

#Selecciona fechas (16, 8, 4, 2, 1, 0.5 Semanas anteriores a la última fecha disponible)
for i in [112,56,28,14,7,3,1]:
    print("Date: ",lastdate-timedelta(int(i)))

#Calcula el promedio de ventas de unidades por item y tienda para las semanas seleccionadas
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')
    
print("Meaning: ", time.strftime("%M:%S", time.gmtime(time.time() - start)) )

# Calcula la mediana de los promedios calculados
ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)
ma_is.drop(list(ma_is.columns.values)[3:],1,inplace=True)
ma_is.head()

train = pd.merge(train, ma_is, how='left', on=['item_nbr','store_nbr'])
train["unit_sales"] = train.mais
train.drop("mais",axis=1,inplace=True)

train.drop("date",axis=1,inplace=True)
test.drop(["id","date"],axis=1,inplace=True)

train
test.drop("onpromotion",axis=1,inplace=True)
test
#test = pd.merge(test, item, how='left', on='item_nbr').drop(['family'], axis=1)


#train = transformations(train)
#test = transformations(test)

print("Training predictors...")

predictores = ["DTR","RFR"]

for predictor in predictores:

    X = train.drop(['unit_sales'], axis=1)
    y = train['unit_sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    if (predictor == 'DTR'):
        estimator = DecisionTreeRegressor()
    if (predictor == 'RFR'):
        estimator = RandomForestRegressor(n_jobs=-1)
    
    estimator.fit(X_train, y_train)
    y_predict = estimator.predict(X_test)
    print("Training "+predictor, time.strftime("%M:%S", time.gmtime(time.time() - start)) )
    #print("NWRMSLE: ", NWRMSLE(y_test , y_predict , X_test["perishable"] ))
    
    y_predict = estimator.predict(test)
    
    solution = pd.DataFrame(np.column_stack([test_id,y_predict]), columns=["id","unit_sales"])
    solution.loc[(solution.unit_sales < 0),'unit_sales'] = 0
    pd.concat([solution["id"].astype(int),solution["unit_sales"]], axis=1).to_csv("predictions"+predictor+".csv.gz", float_format='%.4f', index=None, compression='gzip')
    print("Data Saved")