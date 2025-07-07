# 1. Используем базовый образ, который мы знаем, что работает на вашей системе
FROM nvcr.io/nvidia/pytorch:25.05-py3

# 2. Обновляем pip до последней версии
RUN python -m pip install --no-cache-dir --upgrade pip

# 3. Копируем файл с зависимостями
WORKDIR /app
COPY requirements.txt .

# 4. КЛЮЧЕВОЙ ШАГ: ПРИНУДИТЕЛЬНАЯ ОЧИСТКА И ПЕРЕУСТАНОВКА
# Сначала полностью удаляем потенциально проблемные пакеты, которые могли
# быть в базовом образе, чтобы избежать любых конфликтов.
RUN python -m pip uninstall -y optuna tqdm optuna-dashboard

# Теперь ставим наши версии с чистого листа.
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. Устанавливаем системные утилиты
RUN apt-get update && apt-get install -y bash procps && rm -rf /var/lib/apt/lists/*

# 6. Копируем остальной код проекта
COPY . .

# 7. Открываем порты
EXPOSE 8888 5000 8081

# 8. Команда по умолчанию для запуска JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]