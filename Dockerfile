# 1. Базовый образ NVIDIA PyTorch
FROM nvcr.io/nvidia/pytorch:25.05-py3

# 2. Установка рабочей директории в контейнере
WORKDIR /app

# 3. Копирование файла с зависимостями
COPY requirements.txt requirements.txt

# 4. Установка дополнительных пакетов с помощью pip из окружения образа
# python и pip должны быть доступны из PATH и принадлежать базовому conda-окружению.
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. Установка bash и procps для полноценной работы терминала в JupyterLab
# Базовый образ, скорее всего, на основе Ubuntu.
RUN apt-get update && apt-get install -y bash procps && rm -rf /var/lib/apt/lists/*

# 6. Копирование всего остального кода проекта
COPY . .

# 7. Делаем порт JupyterLab доступным
EXPOSE 8888

# 8. Команда для запуска JupyterLab
# jupyter должен быть доступен, так как мы его установили через pip.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]