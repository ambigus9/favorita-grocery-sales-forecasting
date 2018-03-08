from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# load or create your dataset
print('Loading data...')

#Si tienes un PC con mucha RAM
#converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0}
dtypes = {'id':'uint32', 'item_nbr':'int32', 'store_nbr':'int8', 'unit_sales':'float32', 'onpromotion':'bool' }

df_train = pd.read_csv("data/train.csv", dtype=dtypes, parse_dates=["date"], low_memory=True, usecols=[1, 2, 3, 4, 5] ) #,converters=converters)
df_test = pd.read_csv("data/test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool}, parse_dates=["date"] ).set_index(['store_nbr', 'item_nbr', 'date'])
items = pd.read_csv("data/items.csv").set_index("item_nbr")
print('Load complete...')

df_train = df_train.loc[df_train.date>=pd.datetime(2017,1,1)] # Buscamos registros para Fecha deseada
df_train.loc[(df_train.unit_sales < 0),'unit_sales'] = 0 # Eliminar Valores Negativos
df_train["unit_sales"] = df_train["unit_sales"].apply(np.log1p) # Aplicar Logaritmo

df_date = df_train
del df_train

# Apilar en columnas las filas de Date para Train enfocandose en las Promociones
promo_date_train = df_date.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(False)
# Asignar el nombre de las fechas en las columnas de Train
promo_date_train.columns = promo_date_train.columns.get_level_values(1)

# Apilar en columnas las filas de Date para Test enfocandose en las Promociones
promo_date_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)

# Asignar el nombre de las fechas en las columnas de Test
promo_date_test.columns = promo_date_test.columns.get_level_values(1)

# Ajustamos los indices de Test en base a los de train
promo_date_test = promo_date_test.reindex(promo_date_train.index).fillna(False)

promo_date = pd.concat([promo_date_train, promo_date_test], axis=1)
del promo_date_test, promo_date_train

# Apilar en columnas las filas de Date para Train enfocandose en las Unidades
df_date = df_date.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
# Asignar el nombre de las fechas en las columnas de Train
df_date.columns = df_date.columns.get_level_values(1)

#Ventas diarias (Columnas) de Unidades para cada Tienda y cada Item de tienda (Filas)
df_date

# Reindexo items en base a los Items disponibles en Train
items = items.reindex(df_date.index.get_level_values(1))
items.head()

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(tdate, is_train=True):
    X = pd.DataFrame({
        "day_1_date": get_timespan(df_date, tdate, 1, 1).values.ravel(),
        # Calcula el promedio de los 3 días anteriores a la fecha deseada tdate
        "mean_3_date": get_timespan(df_date, tdate, 3, 3).mean(axis=1).values,
        "mean_7_date": get_timespan(df_date, tdate, 7, 7).mean(axis=1).values,
        "mean_14_date": get_timespan(df_date, tdate, 14, 14).mean(axis=1).values,
        "mean_30_date": get_timespan(df_date, tdate, 30, 30).mean(axis=1).values,
        "mean_60_date": get_timespan(df_date, tdate, 60, 60).mean(axis=1).values,
        #"mean_140_date": get_timespan(df_date, tdate, 140, 140).mean(axis=1).values,
        "promo_14_date": get_timespan(promo_date, tdate, 14, 14).sum(axis=1).values,
        "promo_60_date": get_timespan(promo_date, tdate, 60, 60).sum(axis=1).values,
        #"promo_140_date": get_timespan(promo_date, tdate, 140, 140).sum(axis=1).values
    })
    for i in range(7):
        #Promedio de Ventas en base a Dia de la Semana
        X['mean_4_dow{}_date'.format(i)] = get_timespan(df_date, tdate, 28-i, 4, freq='7D').mean(axis=1).values
        #X['mean_20_dow{}_date'.format(i)] = get_timespan(df_date, tdate, 140-i, 20, freq='7D').mean(axis=1).values
    for i in range(16):
        X["promo_{}".format(i)] = promo_date[
            tdate + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        # Unidades Vendidas de Items por tienda para los 16 días posteriores a la fecha deseada tdate
        y = df_date[
            pd.date_range(tdate, periods=16)
        ].values
        return X, y
    return X

print("Preparing dataset...")
tdate = date(2017, 5, 31)
X_l, y_l = [], []

# Elige una ventana temporal de 6
for i in range(6):
    delta = timedelta(days=7 * i)
    print("Calculando promedios deseados para la fecha: ",tdate + delta)
    X_tmp, y_tmp = prepare_dataset(
        # Calcula promedios deseados para fechas cada 7 días (7,14,21,...,42)
        tdate + delta
    )
    # Unir a lista los valores de X
    X_l.append(X_tmp)
    # Unir a lista los valores de y
    y_l.append(y_tmp)

# Concatenamos todos los valores de X
X_train = pd.concat(X_l, axis=0)
# Concatenamos todos los valores de y
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

# Calcula un X y y en base a una fecha deseada (Aún no comprendo porqué)
X_val, y_val = prepare_dataset(date(2017, 7, 26))
# Calculamos el X_test en base al último día + 1 disponible de registros
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

print("Training and predicting models...")
params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 4
}

MAX_ROUNDS = 50
val_pred = []
test_pred = []
cate_vars = []
# Son 16 vueltas porque se van a predecir 16 días! empezando desde 2017-08-16 hasta 2017-08-31
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        # Concatenamos 6 veces items porque se eligió una ventana temporal de 6 y se concatenaron 6 X_train anteriormente.
        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=20, verbose_eval=100
    )
    #Imprime los promedios deseados de "Prepare_dataset"
    #print("\n".join(("%s: %.2f" % x) for x in sorted(
        #zip(X_train.columns, bst.feature_importance("gain")),
        #key=lambda x: x[1], reverse=True
    #)))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    #Guarda la predicción de cada día
    test_pred.append(bst.predict(
        X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    
    print("Val_Pred: ",val_pred)
    print("Test_pred: ",test_pred)

#Evalua el rendimiento en un fragmento comparando las predicciones para uno de los 16 días
print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

print("Making submission...")
y_test = np.array(test_pred).transpose()

print("Making submission...")
y_test = np.array(test_pred).transpose()

df_preds = pd.DataFrame(
    y_test, index=df_date.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

# Une las predicciones con el test (basandose en tienda, item y fecha) y los que no estén entonces lo llena con 0
submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('lgb.csv', float_format='%.4f', index=None)

