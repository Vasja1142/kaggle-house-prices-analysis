# -*- coding: utf-8 -*-
"""
Скрипт для генерации и сохранения SHAP-графиков для анализа
важности признаков лучшей модели XGBoost.
"""

import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. Конфигурация ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- 2. Загрузка данных и модели ---
print("Загрузка данных и лучшей модели XGBoost...")

try:
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv'))
    
    # Ищем лучшую модель XGBoost в папке models
    xgboost_models = [f for f in os.listdir(MODELS_DIR) if f.startswith('xgboost') and f.endswith('.pkl')]
    if not xgboost_models:
        print("Ошибка: Не найдено ни одной модели XGBoost в папке 'models'.")
        exit()
    
    # Предполагаем, что лучший скор - это минимальный RMSE
    xgboost_models.sort() 
    best_model_filename = xgboost_models[0]
    best_model_path = os.path.join(MODELS_DIR, best_model_filename)
    
    model = joblib.load(best_model_path)
    print(f"Лучшая модель XGBoost загружена: {best_model_filename}")

except FileNotFoundError:
    print("Ошибка: Файлы не найдены. Убедитесь, что скрипты предобработки и обучения были запущены.")
    exit()

X = train_df.drop('SalePrice', axis=1)

# --- 3. Расчет SHAP значений ---
print("Расчет SHAP значений... Это может занять некоторое время.")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# --- 4. Генерация и сохранение графиков ---

# Расчет средней абсолютной величины SHAP для каждого признака
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['feature', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=True)

# --- 5. Генерация и сохранение графиков ---

# График 1: Bar plot (общая важность)
print("Создание и сохранение графика Bar plot...")
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
bar_plot_path = os.path.join(REPORTS_DIR, 'shap_summary_bar.png')
# Увеличиваем dpi и используем bbox_inches='tight' для лучшего качества
plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"График сохранен в: {bar_plot_path}")

# График 2: Beeswarm plot (детальное распределение)
print("Создание и сохранение графика Beeswarm plot...")
plt.figure()
shap.summary_plot(shap_values, X, show=False)
beeswarm_plot_path = os.path.join(REPORTS_DIR, 'shap_summary_beeswarm.png')
plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"График сохранен в: {beeswarm_plot_path}")

# График 3: Bar plot (самые слабые признаки)
print("Создание и сохранение графика Bar plot для самых слабых признаков...")
num_weakest_features = 20 
weakest_features = importance_df.head(num_weakest_features)
weakest_feature_indices = [X.columns.get_loc(f) for f in weakest_features['feature']]

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[:, weakest_feature_indices], X.iloc[:, weakest_feature_indices], plot_type="bar", show=False)
weakest_bar_plot_path = os.path.join(REPORTS_DIR, 'shap_summary_bar_weakest.png')
plt.title(f'Top {num_weakest_features} Weakest Features')
plt.savefig(weakest_bar_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"График самых слабых признаков сохранен в: {weakest_bar_plot_path}")


print("\nАнализ важности признаков успешно завершен.")
