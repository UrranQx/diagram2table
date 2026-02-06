# diagram2table 
Преобразование изображений диаграмм в markdown таблицу.
![UI example](image.png)

## Перед запуском
### Используйте виртуальное окружение в `.venv`
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# При наличии видеокарты с CUDA ядрами (для проверки nvidia-smi) запустить также:
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
Примечание: для поддержки SVG требуется ImageMagick и соответствующие делегаты (должен быть в PATH). Для локальной установки:
- Windows: скачайте ImageMagick с https://imagemagick.org и включите опцию "Install development headers"
- Linux (apt): sudo apt install imagemagick libmagickwand-dev librsvg2-bin


### Рекомендукю заранее скачать  модель при помощи скрипта `scripts\download_models.py`
## Запуск проекта
### Для запуска API + UI используйте параметр "both", а чтобы загрузить квантизированную модель (INT4) используйте флаг `--deployment-mode vlm_quantized`
Вот например
```cmd
python -m src.main both --deployment-mode vlm_quantized
```

## Docker:

Сборка образа:
```cmd
docker build -t diagram2table:local -f docker/Dockerfile .
```

Или 

```cmd
docker compose -f docker/docker-compose.yml build --progress=plain --no-cache diagram2table-gpu
```


Поднять изображение:

```cmd
docker compose up diagram2table-gpu 
```



Примечание: Если при сборке появится ошибка `MagickWand shared library not found` — убедитесь, что пакеты `libmagickwand-dev` и `imagemagick` установлены. Если SVG всё ещё не рендерится, добавьте пакет `librsvg2-bin` (в Dockerfile уже включён) и проверьте `/etc/ImageMagick-6/policy.xml` — в образе мы уже удаляем политики, блокирующие SVG.


## Ссылки на полезные исследования в данной области, а также похожие проекты
- https://arxiv.org/abs/2511.22448
- https://ieeexplore.ieee.org/document/9980425
- https://huggingface.co/jtlicardo/bpmn-information-extraction-v2
- https://github.com/PROSLab/BPMN-Redrawer