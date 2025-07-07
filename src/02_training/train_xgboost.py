# -*- coding: utf-8 -*-
"""
Скрипт для оптимизации гиперпараметров XGBoost с помощью Optuna,
логирования в MLflow и сохранения лучшей модели.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Конфигурация ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_DIR = os.path.join(BASE_DIR, 'db')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Настройки эксперимента
EXPERIMENT_NAME = "XGBoost_Optimization"
STUDY_NAME = "xgboost-tuning2" # Новое, более чистое имя
N_TRIALS = 200
STORAGE_NAME = f"sqlite:///{os.path.join(DB_DIR, 'optuna_study_refactored.db')}"


# --- 2. Загрузка данных ---
print("Загрузка обработанных данных...")
try:
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv'))
except FileNotFoundError:
    print("Ошибка: Файл train_processed.csv не найден. Запустите скрипт предобработки.")
    exit()

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']

# Разделение на обучающую и валидационную выборки для Optuna
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Objective-функция для Optuna ---
def objective(trial):
    """
    Функция, которую Optuna будет минимизировать.
    Она обучает модель с предложенными параметрами и возвращает RMSE.
    """
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'booster': 'gbtree',
        'random_state': 42,
        'lambda': trial.suggest_float('lambda', 0.05, 0.3, step=0.01),
        'alpha': trial.suggest_float('alpha', 0.001, 0.03, step=0.0005),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.06, step=0.0025),
        'max_depth': trial.suggest_int('max_depth', 3, 3),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8, step = 0.02),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.5, step=0.02),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
    }

    model = xgb.XGBRegressor(n_estimators=1000, **params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

# --- 4. Запуск оптимизации ---
print(f"Запуск исследования Optuna для XGBoost ({N_TRIALS} попыток)...")
study = optuna.create_study(
    direction='minimize',
    study_name=STUDY_NAME,
    storage=STORAGE_NAME,
    load_if_exists=True
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# --- 5. Логирование и сохранение лучшей модели ---
best_trial = study.best_trial
best_params = best_trial.params
best_score = best_trial.value

print(f"\nОптимизация завершена.")
print(f"Лучший RMSE: {best_score:.5f}")
print("Лучшие гиперпараметры:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Логирование в MLflow
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="XGBoost_Best_Run"):
    print("\nЛогирование результатов в MLflow...")
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", best_score)
    
    # Переобучение модели на всех данных с лучшими параметрами
    print("Переобучение финальной модели на всех данных...")
    final_model = xgb.XGBRegressor(n_estimators=1000, **best_params)
    # Используем весь набор (X, y) для финального обучения
    final_model.fit(X, y) 
    
    # Сохранение модели
    model_filename = f"xgboost_rmse_{best_score:.4f}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(final_model, model_path)
    
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Лучшая модель сохранена в: {model_path}")
    print("Эксперимент успешно залогирован в MLflow.")

print("\nСкрипт успешно завершен.")
