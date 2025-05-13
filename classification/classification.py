import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_train():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Выбираем полезные признаки
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[features + ['Survived']]
    
    # Заполняем пропуски
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Преобразуем категориальные данные
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    
    # Разделяем на обучающую и тестовую выборки
    X = df[features]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Точность модели: {accuracy:.2f}")

    # Сохраняем модель
    joblib.dump(model, 'titanic_model.pkl')
    print("Модель обучена и сохранена!")

def predict_survival():
    model = joblib.load('titanic_model.pkl')
    
    # Ввод данных
    pclass = int(input("Класс (1, 2, 3): "))
    sex = input("Пол (male/female): ")
    age = float(input("Возраст: "))
    sibsp = int(input("Братья/сестры/супруги на борту: "))
    parch = int(input("Родители/дети на борту: "))
    fare = float(input("Стоимость билета: "))
    embarked = input("Порт посадки (C/Q/S): ")
    
    # Кодируем категориальные данные
    sex = 1 if sex.lower() == 'male' else 0
    embarked = {'C': 0, 'Q': 1, 'S': 2}.get(embarked.upper(), 2)
    
    # Делаем предсказание
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    probability = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]
    
    print(f"Вероятность выживания: {probability:.2f}")
    print("Выжил!" if prediction == 1 else "Не выжил")

if __name__ == "__main__":
    load_and_train()
    while True:
        predict_survival()
        if input("Хотите ввести ещё одного пассажира? (y/n): ").lower() != 'y':
            break