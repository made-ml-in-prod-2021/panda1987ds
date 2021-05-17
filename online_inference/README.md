###**Загрузка образа**:

    docker pull panda1987ds/online_inference

###**Запуск**:

    docker run -p 8080:8080 panda1987ds/online_inference

### **Тесты:**
    pytest tests/

### Структура:
    |- online_inference
        |- configs                  <- файлы конфигурации параметров YAML
        |- data                     <- файл данных
        |- logs                     <- логи работы программы
        |- models                   <- модель
        |- src                      <- исходный код проекта
            |- entities             <- классы: запроса, ответа и модели
            |- app.py               <- основной код приложения
            |- request_example.py   <- скрипт с примером запроса
        |- tests                    <- тесты
        |- README.md                <- описание проекта
        |- Dockerfile               <- файл для сборки образа 



