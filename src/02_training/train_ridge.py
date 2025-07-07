# -*- coding: utf-8 -*-
"""
Скрипт для оптимизации гиперпараметра alpha для Ridge регрессии
с помощью Optuna, логирования в MLflow и сохранения лучшей модели.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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
EXPERIMENT_NAME = "Ridge_Optimization"
STUDY_NAME = "ridge-tuning"
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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Objective-функция для Optuna ---
def objective(trial):
    # Для Ridge мы подбираем только один ключевой параметр - alpha
    alpha = trial.suggest_float('alpha', 1e-2, 100.0, log=True)

    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

# --- 4. Запуск оптимизации ---
print(f"Запуск исследования Optuna для Ridge ({N_TRIALS} попыток)...")
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
with mlflow.start_run(run_name="Ridge_Best_Run"):
    print("\nЛогирование результатов в MLflow...")
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", best_score)
    
    print("Переобучение финальной модели на всех данных...")
    final_model = Ridge(**best_params)
    final_model.fit(X, y)
    
    model_filename = f"ridge_rmse_{best_score:.4f}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(final_model, model_path)
    
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Лучшая модель сохранена в: {model_path}")
    print("Эксперимент успешно залогирован в MLflow.")

print("\nСкрипт успешно завершен.")