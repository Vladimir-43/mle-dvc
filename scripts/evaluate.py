# scripts/evaluate.py

# 1 — импорты
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os
import json

# 2 — вспомогательные функции - их нет

# 3 — главная функция -  оценка качества модели
def evaluate_model():
    # 3.1 — загрузка гиперпараметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # 3.2 — загрузки предыдущих результатов
    data = pd.read_csv('data/initial_data.csv')    
    with open('models/fitted_model.pkl', 'rb') as fd:
        model = joblib.load(fd) 
        
    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    # Проверка качества на кросс-валидации
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        model,
        data,
        data['target'],
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics']
    )

    # сохраните результата кросс-валидации в cv_res.json
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3) 
    os.makedirs('cv_results', exist_ok=True) # создание директории, если её ещё нет
    with open('cv_results/cv_res.json', 'w', encoding="utf-8") as fd:
        json.dump(cv_res, fd)

if __name__ == '__main__':
    evaluate_model()
