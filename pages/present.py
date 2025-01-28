import streamlit as st

st.title('Представляем вашему вниманию... Презентацию того, через что мы прошли.')

st.write('Расскажем немного о том, как отбирали данные, что использовали для оценки.')

st.write('Первое, что мы сделали: построили матрицу корреляции числовых признаков.')
st.image('pictures/corr_m_num.png')
st.write('Выглядит не очень. Зато дала нам информацию про коллинеарные признаки. В последствие мы убрали их из нашего Датасета.')

st.write('Сразу рассмотрим матрицу корреляции категориальных признаков - ее мы получили методом chi2.')
st.image('pictures/corr_martix.png')
st.write('Здесь признаков намного больше, и информации тоже больше.')

st.write('Так же мы провели различные тесты для понимания, какие строки действительно важны, а какие стоит убрать.')
col1, col2 = st.columns(2)
with col1:
    st.image('pictures/F_test_num.jpg', caption='F-test num')
with col2:
    st.image('pictures/F-test_cat.jpg', caption='F_test categ')

col3, col4 = st.columns(2)
with col3:
    st.image('pictures/Lasso_num.jpg', caption='lasso num')
with col4:
    st.image('pictures/lasso_cat.jpg', caption='lasso categ')

col5, col6 = st.columns(2)
with col5:
    st.image('pictures/feat_imp_num.jpg', caption='feature imp num')
with col6:
    st.image('pictures/feat_imp_cat.jpg', caption='feature imp categ')

col7, col8 = st.columns(2)
with col7:
    st.image('pictures/perm_imp_num.jpg', caption='permutation imp num')
with col8:
    st.image('pictures/perm_imp_cat.jpg', caption='permutation imp categ')

col9, col10 = st.columns(2)
with col9:
    st.image('pictures/chi2_num.jpg', caption='chi2 num')
with col10:
    st.image('pictures/chi2_cat.jpg', caption='chi2 categ')

col11, col12 = st.columns(2)
with col11:
    st.image('pictures/PFE_num.jpg', caption='PFE num')
with col12:
    st.image('pictures/pfe_cat.jpg', caption='PFE categ')

st.write('Что еще мы смогли построить? Конечно же SHAP value')
st.image('pictures/snap_num.png', caption='SHAP value num')
st.image('pictures/snap_cat.png', caption='SHAP value categ')


st.title('Проговорим немного про модели')
st.write('Мы использовали 4 основные модели - LinearRegression, RandomForest, CatBoost, LightLGBM.')
st.write('Так же была попытка сделать предсказания на основании GXBoost, с чем мы справились, но мы не включили показатели в общую таблицу.')
st.write('Основные метрики по каждой модели:')
st.image('pictures/df_metric.png')
st.write('Лучшая модель на наш скромный взгляд - CatBoost. Использовали ее для обучения в дальнейшем.')
st.write('Pipeline состоял из preprocessor и model - CatBoost.')
st.write('Preprocessor в свою очередь - из my_imputer, my_dropper, my_encoder.')
st.write('Первая часть заполняла пропуски, вторая - выбрасывала ненужные столбцы данных, третья - кодировала признаки и нормировала при необходимости.')

st.title('Что там с Kaggle?')
st.write('Вот какие места мы смогли занять на Kaggle')
st.image('pictures/kaggle_place_Alina.png', caption='Алина')
st.image('pictures/kaggle_place_Misha.jpg', caption='Миша')

st.title('Что еще хотелось бы добавить, но не успели?')
st.write('Миша нашел более лучшие настройки для XGBoost - и его место в Kaggle доказывает это.')
st.write('Так же в планах докрутить SHAP для определенного признака - чтобы понимать, кто и как на него влияет.')
st.write('Спасибо за внимание!')