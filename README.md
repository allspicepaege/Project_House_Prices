# **Project_House_Prices**

## **Описание проекта**

Этот проект использует методы машинного обучения для предсказания стоимости участков с домами на основе их характеристик. В процессе решения задачи были выполнены следующие шаги:

- Очистка данных (удаление пропусков, ненужных столбцов и т.д.)
- Замена категориальных признаков на числовые
- Нормализация признаков
- Построение различных моделей для оценки эффективности через метрики
- Выбор лучшей модели
- Обучение модели на новых данных, которые еще не были представлены
- Создание и сохранение файла с предсказаниями в формате submission для Kaggle

Проект обернут в Streamlit-приложение, позволяющее пользователю загружать собственные датасеты и получить предсказания стоимости жилья.

## **Особенности**

- Поддержка загрузки пользовательского датасета в Streamlit
- Обработка и подготовка данных с использованием sklearn.Pipeline
- Возможность получения предсказания на основе новых данных

## **Установка**

Для установки всех зависимостей проекта, выполните следующие шаги:
1. Клонируйте репозиторий:
  ```
  git clone https://github.com/yourusername/yourrepository.git
  ```
2. Перейдите в каталог проекта:
  ```
  cd yourrepository
```
3. Установите необходимые зависимости:
  ```
  pip install -r requirements.txt
```
4. Запустите приложение Streamlit:
```
streamlit run main_streamlit_page.py
```
## **Contributors**
Бутин Михаил
Зарницына Алина



  
