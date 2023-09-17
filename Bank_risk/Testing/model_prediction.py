import pandas as pd
import pickle

# Загрузим датасет для обучения
application_test = pd.read_csv('application_test.csv')


def predict(df):
    """Функция принимает на вход тестовый датасет, обученную модель.
    Выполняет предсказание и возвращает датасет с новым столбцом
    с предсказанием"""    
    
    def data_transform(df):
        """ Функция для преобразования признаков. 
        На вход подается тестовый датасет, на выходе
        датасет с необходимыми признаками"""
        
        # признаки отобранные feature importance
        col = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 
               'DAYS_EMPLOYED', 'DAYS_BIRTH', 
               'DAYS_LAST_PHONE_CHANGE', 'AMT_CREDIT', 
               'DAYS_REGISTRATION', 'SK_ID_CURR', 
               'DAYS_ID_PUBLISH', 'AMT_ANNUITY', 
               'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 
               'AMT_INCOME_TOTAL', 'OCCUPATION_TYPE', 
               'HOUR_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 
               'AMT_REQ_CREDIT_BUREAU_YEAR', 'CODE_GENDER', 
               'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
               'NAME_TYPE_SUITE', 'OBS_30_CNT_SOCIAL_CIRCLE', 
               'WEEKDAY_APPR_PROCESS_START', 'AMT_REQ_CREDIT_BUREAU_QRT', 
               'OBS_60_CNT_SOCIAL_CIRCLE', 'FLAG_PHONE', 
               'CNT_FAM_MEMBERS', 'NAME_INCOME_TYPE', 
               'FLAG_OWN_CAR', 'FLAG_WORK_PHONE', 
               'CNT_CHILDREN', 'FLAG_OWN_REALTY', 
               'DEF_30_CNT_SOCIAL_CIRCLE', 'REGION_RATING_CLIENT_W_CITY', 
               'NAME_HOUSING_TYPE', 'FLAG_DOCUMENT_3', 
               'AMT_REQ_CREDIT_BUREAU_MON', 'REGION_RATING_CLIENT', 
               'NAME_CONTRACT_TYPE', 'LIVE_CITY_NOT_WORK_CITY', 
               'FLAG_EMAIL', 'REG_CITY_NOT_WORK_CITY', 
               'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_5', 
               'AMT_REQ_CREDIT_BUREAU_WEEK', 'REG_CITY_NOT_LIVE_CITY', 
               'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_16', 
               'REG_REGION_NOT_LIVE_REGION', 'FLAG_DOCUMENT_18']
        
        x = df[col]
        x = x.dropna()
        
        return x
    
    # отбор признаков из исходного датасета
    x = data_transform(df)

    #загрузка модели
    with open('model_risk.pcl', 'rb') as fid:
        model = pickle.load(fid)
    
    #вызываем предсказание
    x['TARGET_predicted'] = model.predict(x).tolist()
              
    return x

display(predict(application_test))
