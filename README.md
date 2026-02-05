# РАБОТА В ПРОЦЕССЕ, МБ ЗАВТРА БУДЕТ ВСЕ ПО КРАСОТЕ, А ПОКА ТУТ ПРОСТО РАБОЧИЙ КОД
## diagram2table" 
Преобразование изображений диаграмм в markdown таблицу 

## Перед запуском
uv venv diagram2table-env
diagram2table-env\Scripts\activate   
uv pip install -r requirements.txt
uv pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130

Скачайте модель при помощи скрипта {SCPRIPT}
## Запуск проекта



    python -m src.main both --deployment-mode vlm_quantized

## Полезные исследования в данной области
- https://arxiv.org/abs/2511.22448
- https://ieeexplore.ieee.org/document/9980425