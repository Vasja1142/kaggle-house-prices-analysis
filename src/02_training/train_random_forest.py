# -*- coding: utf-8 -*-
"""
Скрипт для оптимизации гиперпараметров RandomForestRegressor с помощью Optuna,
логирования в MLflow и сохранения лучшей модели.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
EXPERIMENT_NAME = "RandomForest_Optimization"
STUDY_NAME = "randomforest-tuning6"
N_TRIALS = 1000
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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Objective-функция для Optuna ---
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 15),
        'max_depth': trial.suggest_int('max_depth', 15, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 3),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1),
        'max_features': trial.suggest_float('max_features', 0.26, 0.33, step=0.002),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

# --- 4. Запуск оптимизации ---
print(f"Запуск исследования Optuna для RandomForest ({N_TRIALS} попыток)...")
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

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="RandomForest_Best_Run"):
    print("\nЛогирование результатов в MLflow...")
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", best_score)
    
    print("Переобучение финальной модели на всех данных...")
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X, y)
    
    model_filename = f"randomforest_rmse_{best_score:.4f}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(final_model, model_path)
    
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Лучшая модель сохранена в: {model_path}")
    print("Эксперимент успешно залогирован в MLflow.")

print("\nСкрипт успешно завершен.")
