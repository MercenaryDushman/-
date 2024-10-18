import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Функция для очистки консоли (работает как для Windows, так и для Unix систем)
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Функция для отображения логотипа
def show_logo():
    logo = """
             _.-'''''-._
          .-'  _     _  '-.
        .'    (_)   (_)    '.
       /  _               _  \\
      ;  (_)   _     _   (_)  ;
      |       (_)   (_)       |
      ;                     ;  
       \\    _           _    /
        '.  (_)       (_)  .'
          '-.__________.-'
              
            WELCOME
    """
    print(logo)

# Функция для отображения меню
def show_menu():
    print("===================================")
    print("|         MENU OF COMMANDS        |")
    print("===================================")
    print("| 1. Generate Data                |")
    print("| 2. Train Isolation Forest Model |")
    print("| 3. Predict Anomalies            |")
    print("| 4. Exit                         |")
    print("===================================")

# Функция для генерации данных
def generate_data():
    print("Generating data...")

    # Генерация нормальных данных
    normal_data = {
        'source_ip': np.random.choice(['192.168.1.1', '192.168.1.2', '192.168.1.3'], 1000),
        'destination_ip': np.random.choice(['8.8.8.8', '8.8.4.4', '1.1.1.1'], 1000),
        'source_port': np.random.randint(1024, 65535, 1000),
        'destination_port': np.random.randint(80, 443, 1000),
        'packet_size': np.random.randint(100, 1500, 1000)
    }

    # Генерация аномальных данных
    anomalous_data = {
        'source_ip': np.random.choice(['192.168.100.1', '192.168.200.2', '192.168.300.3'], 100),
        'destination_ip': np.random.choice(['10.0.0.1', '172.16.0.1', '192.168.0.1'], 100),
        'source_port': np.random.randint(1024, 65535, 100),
        'destination_port': np.random.randint(80, 443, 100),
        'packet_size': np.random.randint(2000, 5000, 100)  # Аномальные пакеты
    }

    # Преобразование в DataFrame
    df_normal = pd.DataFrame(normal_data)
    df_anomalous = pd.DataFrame(anomalous_data)

    # Метки
    df_normal['Label'] = 0
    df_anomalous['Label'] = 1

    # Объединение
    df = pd.concat([df_normal, df_anomalous], ignore_index=True)

    # Преобразование IP-адресов
    le = LabelEncoder()
    df['source_ip'] = le.fit_transform(df['source_ip'])
    df['destination_ip'] = le.fit_transform(df['destination_ip'])

    print("Data generated successfully!")
    return df

# Функция для обучения модели
def train_model(df):
    print("Training model...")
    X = df.drop('Label', axis=1)  # Признаки
    y = df['Label']  # Метки

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_train)

    print("Model trained successfully!")
    return model, X_test, y_test

# Функция для предсказания аномалий
def predict_anomalies(model, X_test, y_test):
    print("Predicting anomalies...")
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)  # Преобразование предсказаний

    # Вывод отчета
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

# Основной цикл программы
def main():
    while True:
        clear_console()  # Очистка консоли перед каждым выводом
        show_logo()
        
        df = None
        model = None
        
        show_menu()
        choice = input("Enter your choice (1-4): ")

        clear_console()  # Очистка консоли после ввода команды
        
        if choice == '1':
            df = generate_data()
            input("\nPress Enter to continue...")  # Пауза перед возвратом в меню
        elif choice == '2':
            if df is not None:
                model, X_test, y_test = train_model(df)
            else:
                print("Please generate data first!")
            input("\nPress Enter to continue...")
        elif choice == '3':
            if model is not None:
                predict_anomalies(model, X_test, y_test)
            else:
                print("Please train the model first!")
            input("\nPress Enter to continue...")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

input()
