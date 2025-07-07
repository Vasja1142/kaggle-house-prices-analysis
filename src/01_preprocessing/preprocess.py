# -*- coding: utf-8 -*-
"""
Скрипт для предобработки данных для соревнования Kaggle "House Prices".

Этот скрипт выполняет следующие шаги:
1. Загружает исходные данные (train.csv, test.csv).
2. Удаляет выбросы из обучающего набора данных.
3. Применяет логарифмическое преобразование к целевой переменной (SalePrice).
4. Объединяет обучающий и тестовый наборы для одновременной обработки.
5. Заполняет пропущенные значения, используя лучшие практики (медиана по группе, константы).
6. Создает новые признаки (Feature Engineering) для улучшения модели.
7. Преобразует категориальные признаки в числовые с помощью one-hot encoding.
8. Разделяет данные обратно на обучающий и тестовый наборы.
9. Сохраняет обработанные данные в папку 'data/processed'.
"""

import os
import pandas as pd
import numpy as np

# --- 1. Конфигурация и загрузка данных ---

# Определение путей к папкам
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Создание папки для обработанных данных, если она не существует
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Загрузка данных
try:
    train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'))
except FileNotFoundError:
    print("Ошибка: Убедитесь, что файлы train.csv и test.csv находятся в папке data/raw")
    exit()

print("Данные успешно загружены.")
print(f"Размер обучающего набора: {train_df.shape}")
print(f"Размер тестового набора: {test_df.shape}")

# Сохраняем ID для тестового набора и исходные размеры
test_id = test_df['Id']
ntrain = train_df.shape[0]
ntest = test_df.shape[0]


# --- 2. Обработка выбросов и целевой переменной ---

# Удаление выбросов, как рекомендовано в исходном ноутбуке
# (дома с жилой площадью более 4000 кв. футов)
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
print(f"Размер обучающего набора после удаления выбросов: {train_df.shape}")

# Обновляем ntrain после удаления выбросов для корректного разделения
ntrain = train_df.shape[0]

# Логарифмирование целевой переменной для нормализации распределения
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
y_train = train_df['SalePrice'].values
train_df = train_df.drop('SalePrice', axis=1)

print("Целевая переменная обработана.")


# --- 3. Объединение данных для предобработки ---

# Объединяем train и test для удобства обработки признаков
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Id'], axis=1, inplace=True)
print(f"Размер объединенного набора данных: {all_data.shape}")


# --- 4. Заполнение пропущенных значений ---

print("Начало заполнения пропущенных значений...")

# Категори��льные признаки, где NA означает отсутствие чего-либо
for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
            'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']:
    all_data[col] = all_data[col].fillna('None')

# Численные признаки, где NA означает 0
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']:
    all_data[col] = all_data[col].fillna(0)

# LotFrontage: заполняем медианным значением по району (Neighborhood)
# Это более точный подход, чем общая медиана
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# Заполнение модой для категориальных признаков с редкими пропусками
for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Functional: NA означает 'typical'
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Utilities: почти все значения одинаковы, столбец не несет информации
all_data = all_data.drop(['Utilities'], axis=1)

print("Заполнение пропущенных значений завершено.")
# Проверка, остались ли пропуски
if all_data.isnull().sum().max() > 0:
    print("Внимание! В данных остались пропущенные значения.")
    print(all_data.isnull().sum().sort_values(ascending=False))
else:
    print("Пропусков в данных не осталось.")


# --- 5. Продвинутый Feature Engineering (из ноутбука 02) ---

print("Создание новых признаков (продвинутый уровень)...")

# 5.1. Создание суммарных признаков
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + 0.5 * all_data['HalfBath'] +
                       all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'])
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                          all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                          all_data['WoodDeckSF'])

# 5.2. Создание признаков на основе времени
# Используем 2010 как год отсчета, как в оригинальном ноутбуке
all_data['YearBuildAgo'] = 2010 - all_data['YearBuilt']
all_data['YearRemodAddAgo'] = 2010 - all_data['YearRemodAdd']
all_data['GarageYrBltAgo'] = 2010 - all_data['GarageYrBlt']
all_data['MoSoldAgo'] = 12 - all_data['MoSold'] + 12 * (2010 - all_data['YrSold'])

# 5.3. Преобразование качественных признаков в числ��вые
qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
bsmt_fin_map = {'GLQ': 5, 'ALQ': 4, 'BLQ': 3, 'Rec': 3.5, 'LwQ': 2, 'Unf': 1, 'None': 0}
bsmt_exp_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
garage_fin_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
functional_map = {'Typ': 0, 'Min1': 2, 'Min2': 1, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'None': 0}
paved_drive_map = {'Y': 0, 'P': 1, 'N': 2}

qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in qual_cols:
    all_data[col] = all_data[col].map(qual_map)

all_data['BsmtFinQuality1'] = all_data['BsmtFinType1'].map(bsmt_fin_map)
all_data['BsmtExposure'] = all_data['BsmtExposure'].map(bsmt_exp_map)
all_data['GarageFinish'] = all_data['GarageFinish'].map(garage_fin_map)
all_data['Functional'] = all_data['Functional'].map(functional_map)
all_data['PavedDrive'] = all_data['PavedDrive'].map(paved_drive_map)

# 5.4. Создание взаимодействий признаков
all_data['OverallQual_x_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']

print("Продвинутый feature engineering завершен.")



# --- 6. Кодирование категориальных признаков ---

print("Кодирование категориальных признаков...")

# Применение one-hot encoding
all_data = pd.get_dummies(all_data)

print(f"Размер набора данных после one-hot encoding: {all_data.shape}")


# --- 7. Фильтрация признаков ---

# Загрузка списка признаков, отобранных для XGBoost
FEATURES_PATH = os.path.join(BASE_DIR, 'notebooks', 'features_xgboost.txt')
try:
    with open(FEATURES_PATH, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Ошибка: Файл с признаками не найден по пути {FEATURES_PATH}")
    exit()

# Проверка, все ли выбранные признаки присутствуют в данных
missing_features = [f for f in selected_features if f not in all_data.columns]
if missing_features:
    print(f"Внимание! Следующие признаки из списка не найдены в данных: {missing_features}")
    # Оставляем только те признаки, которые есть в all_data
    selected_features = [f for f in selected_features if f in all_data.columns]

all_data = all_data[selected_features]

print(f"Данные отфильтрованы. Количество признаков: {all_data.shape[1]}")


# --- 8. Разделение и сохранение данных ---

# Разделение обратно на обучающий и тестовый наборы
# Важно: используем обновленный ntrain
train_processed = all_data[:ntrain]
test_processed = all_data[ntrain:]

# Добавляем обратно целевую переменную в обучающий набор
train_processed['SalePrice'] = y_train

# Сохранение обработанных файлов
train_path = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')
test_path = os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv')

train_processed.to_csv(train_path, index=False)
test_processed.to_csv(test_path, index=False)

print(f"Обработанные данные сохранены:")
print(f" - Обучающий набор: {train_path}")
print(f" - Тестовый набор: {test_path}")
print("Предобработка успешно завершена!")
