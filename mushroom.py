import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 載入資料
data = pd.read_csv('mushrooms.csv')

# 編碼類別型特徵
label_encoder = LabelEncoder()
for colums in data.columns:
    data[colums] = label_encoder.fit_transform(data[colums])
    print(data)
# 拆分特徵與標籤
X = data.drop('class', axis=1)
y = data['class']

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 建立隨機森林分類器
clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=3)
clf.fit(X_train, y_train)

# 預測測試集
y_pred = clf.predict(X_test)

# 評估模型
print(f"模型準確率: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("分類報告:")
print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))

# 混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.title("confusion matrix")
plt.xlabel("predict")
plt.ylabel("true")
plt.show()

# 特徵重要性
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("feature important")
plt.show()
