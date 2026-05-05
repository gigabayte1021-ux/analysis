import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Загрузка данных (имитация датасета стартапов РК)
# В реальном проекте здесь будет: df = pd.read_csv('../data/startups_kz.csv')
data = {
    'investment_round': [1, 2, 1, 3, 2, 1, 4, 1, 2, 3],
    'team_size': [3, 5, 2, 10, 4, 2, 15, 3, 6, 8],
    'is_it_sector': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    'has_mentor': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    'success': [1, 1, 0, 1, 0, 0, 1, 0, 1, 1] # Целевая переменная
}
df = pd.DataFrame(data)

# 2. Системный анализ: Описательная статистика
print("Статистика по факторам успеха:")
print(df.describe())

# 3. Подготовка данных для моделирования
X = df.drop('success', axis=1)
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Построение модели логистической регрессии (как в твоем отчете)
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Оценка результатов
predictions = model.predict(X_test)
print("\nОтчет о классификации:")
print(classification_report(y_test, predictions))

# 6. Определение значимости факторов (коэффициенты)
importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
print("\nВлияние факторов на успех стартапа:")
print(importance.sort_values(by='Importance', ascending=False))
