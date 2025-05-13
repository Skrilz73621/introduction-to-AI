import numpy as np
import joblib

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
    
    print(f"Вероятность выживания: {probability:.2f}%")
    print("Выжил!" if prediction == 1 else "Не выжил")

if __name__ == "__main__":
    while True:
        predict_survival()
        if input("Хотите ввести ещё одного пассажира? (y/n): ").lower() != 'y':
            break
