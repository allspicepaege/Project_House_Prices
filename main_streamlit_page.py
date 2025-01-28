import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import chi2_contingency
import shap

import sklearn
sklearn.set_config(transform_output="pandas")

from sklearn.model_selection import train_test_split, cross_val_score

#Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

#metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error


st.title('Valuation of real estate objects')
st.write('Загрузите необходимый cvs-файл для дальнейшей обработки...')

uploaded_file = st.file_uploader('Ожидаю CSV файл...', type='csv')
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write('Небольшое представление загруженного файла:')
    st.write(test_df.head(10))

    # обработаем загруженный датасет
    train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    X, y = train.drop('SalePrice', axis=1), train['SalePrice']

    # добавим по строке к каждому датасету для определения NaN корректно
    train_na = X.columns[X.isna().any()]
    test_na = test_df.columns[test_df.isna().any()]

    new_row = {}
    for col in X.columns:
        if col in train_na:
            new_row[col] = np.nan
        elif col in test_na and col not in train_na:
            new_row[col] = np.nan
        else:
            if X[col].dtype == 'object':
                new_row[col] = X[col].mode()[0]
            else:
                new_row[col] = X[col].mean()

    X = pd.concat([X, pd.DataFrame([new_row])], ignore_index=True)
    test_df = pd.concat([test_df, pd.DataFrame([new_row])], ignore_index=True)

    # приравняем по длине y
    y = pd.concat([y, pd.Series([y.mean()])], ignore_index=True)

    # делим датасет для обучения
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # поработаем с NaN
    train_nan = train.columns[train.isna().any()]
    test_nan = test_df.columns[test_df.isna().any()]
    nan_cols = train_nan.union(test_nan)

    #разделим категориальные признаки от числовых, добавим список с пропущенными значениями
    num_features = train.select_dtypes(exclude='object')
    cat_features = train.select_dtypes(include='object')

    nan_dict = {'cat':[col for col in cat_features if col in nan_cols], 'num':[col for col in num_features if col in nan_cols]}

    # разберемся теперь с категориальными признаками
    # сделаем тест хи-квадрат для оценки независимости между признаком и таргетом. Низкое p - есть связь
    p_values = []
    for cat in cat_features.columns:
        cont_table = pd.crosstab(train[cat], train['SalePrice'])
        if (cont_table.values < 5).any():
            chi2, p, dof, ex = chi2_contingency(cont_table, correction=True)
        else:
            chi2, p, dof, ex = chi2_contingency(cont_table)
        p_values.append(p)
    results = pd.DataFrame({'feature':cat_features.columns, 'p_value':p_values})
    results = results.sort_values(by='p_value')

    #отбросим те, у которых p_value слишком большой
    dropping = results.loc[results['p_value'] > 0.001, 'feature'].reset_index()

    # общие фичи, которые выкинем
    drop_features = ['Id', 'YrSold', 'MoSold', 'MiscVal', 'SaleType', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars']
    drop_features += dropping['feature'].tolist()

    # соберем воедино my_imputer и my_dropper
    my_imputer = ColumnTransformer(
        transformers = [
            ('num_imputer', SimpleImputer(strategy='constant', fill_value=0), nan_dict['num']),
            ('cat_imputer_zeros', SimpleImputer(strategy='constant', fill_value='zero'), nan_dict['cat']),
        ],
        verbose_feature_names_out = False,
        remainder = 'passthrough'
    )

    my_dropper = ColumnTransformer(
        transformers = [
            ('drop_features', 'drop', drop_features)
        ],
        verbose_feature_names_out = False,
        remainder = 'passthrough' 
    )

    # поехали учить
    processed_data = my_imputer.fit_transform(train)
    proc_df = pd.DataFrame(processed_data, columns=X_train.columns)
    fitted_data = my_dropper.fit_transform(proc_df)
    fitted_df = pd.DataFrame(fitted_data)

    #fitted_df - почищенный датасет без лишних колонок. предварительный результат
    #дальше идем к нормировке и кодированию
    num_encode_col = fitted_df.select_dtypes(exclude='object').columns.tolist()
    ordinal_encode_col = fitted_df.select_dtypes(include='object').columns.tolist()

    my_encoder = ColumnTransformer(
        transformers = [
            ('ordinal_encoding', OrdinalEncoder(), ordinal_encode_col),
            ('scaling num features', StandardScaler(), num_encode_col)
        ],
        verbose_feature_names_out = False,
        remainder = 'passthrough'
    )

    #соберем препроцессор
    preprocessor = Pipeline(
        [
            ('imputer', my_imputer),
            ('drop features', my_dropper),
            ('encode and scale', my_encoder)
        ]
    )

     #и в Pipeline его с лучшей моделью (после тестов)
    lgbm_ml_pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', lgb.LGBMRegressor(objective='regression', metric='rmse', num_leaves=31, learning_rate=0.05, n_estimators=1000))
        ]
    )
    button = st.button('Предскажи стоимость!')
    if button:
        y_train_log = np.log1p(y_train)
        y_valid_log = np.log1p(y_valid)

        lgbm_ml_pipeline.fit(X_train, y_train_log)

        y_pred_log = lgbm_ml_pipeline.predict(test_df)
        y_pred_original = np.expm1(y_pred_log)

        test_df['SalePrice'] = y_pred_original


        st.write('Провели необходимые рассчеты... В файл добавлена стоимость.')
        st.write(test_df)
        csv = test_df.to_csv(index=False).encode('utf-8')

        st.download_button('Загрузить обработанный файл', csv, file_name='predicted_price.csv', mime='text/csv', key='download-csv')
        
else:
    st.stop()