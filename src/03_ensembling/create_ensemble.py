# -*- coding: utf-8 -*-
"""
Скрипт для создания ансамбля из лучших обученных моделей,
получения предсказаний и формирования финального submission-файла.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import VotingRegressor


# --- 1. Конфигурация ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# --- 2. Загрузка данных ---
print("Загрузка обработанных и исходных данных...")
try:
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv'))
    original_test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'))
except FileNotFoundError as e:
    print(f"Ошибка: Не найдены файлы данных. Убедитесь, что скрипт preprocess.py был запущен.")
    print(e)
    exit()

# --- 3. Загрузка обученных моделей ---
print("Загрузка обученных моделей...")
models_to_load = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
loaded_models = []

if not models_to_load:
    print("Ошибка: В папке 'models' не найдено ни одной модели (.pkl).")
    print("Запустите скрипты из папки 'src/02_training' для обучения моделей.")
    exit()

for model_file in models_to_load:
    try:
        model_path = os.path.join(MODELS_DIR, model_file)
        model = joblib.load(model_path)
        # Извлекаем имя модели из имени файла
        model_name = model_file.split('_')[0]
        loaded_models.append((model_name, model))
        print(f" - Модель '{model_name}' успешно загружена из {model_file}")
    except Exception as e:
        print(f"Ошибка при загрузке модели {model_file}: {e}")

# --- 4. Создание и использование ансамбля ---
print("\nСоздание ансамбля VotingRegressor...")

# VotingRegressor просто усредняет предсказания всех моделей
ensemble = VotingRegressor(estimators=loaded_models, n_jobs=-1)

# Для VotingRegressor не нужен fit, так как модели уже обучены.
# Мы используем его для получения предсказаний.
# Однако, для удобства API, мы можем вызвать fit на пустых данных,
# чтобы он "запомнил" эстиматоры.
# В новых версиях sklearn это не обязательно, но для совместимости полезно.
# ensemble.fit(np.array([]).reshape(0, test_df.shape[1]), np.array([]))

print("Получение предсказаний от ансамбля...")

# Убедимся, что колонки в тестовом наборе соответствуют тем, на которых обучались модели
# (на всякий случай, хотя preprocess.py уже должен был это сделать)
# Загрузим одну из моделей, чтобы получить порядок колонок
first_model = loaded_models[0][1]
if hasattr(first_model, 'feature_names_in_'):
    X_test = test_df[first_model.feature_names_in_]
else:
    # Для старых версий sklearn, где нет feature_names_in_
    # Мы предполагаем, что порядок колонок в test_processed.csv верный
    X_test = test_df

# Предсказания будут усреднены автоматически
log_predictions = np.mean([model.predict(X_test) for name, model in loaded_models], axis=0)


# Преобразуем предсказания обратно в исходный масштаб цен
final_predictions = np.expm1(log_predictions)

# --- 5. Формирование и сохранение файла для отправки ---
submission_df = pd.DataFrame({
    'Id': original_test_df['Id'],
    'SalePrice': final_predictions
})

submission_path = os.path.join(SUBMISSIONS_DIR, 'submission_ensemble.csv')
submission_df.to_csv(submission_path, index=False)

print(f"\nФайл для отправки успешно создан и сохранен по пути: {submission_path}")
print("Первые 5 строк submission файла:")
print(submission_df.head())
print("\nСкрипт успешно завершен.")