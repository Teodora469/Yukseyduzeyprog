import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Veri Yükleme
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values
X_test = test_data.values

# 2. Eğitim ve Doğrulama Seti Bölme
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Eğitimi
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Doğrulama Setinde Değerlendirme
y_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# 5. Test Setinde Tahmin
test_predictions = clf.predict(X_test)

# 6. Submission Dosyası Oluşturma
submission = pd.read_csv("data/sample_submission.csv")
submission["Label"] = test_predictions
submission.to_csv("submission.csv", index=False)



# sample_submission.csv'yi yükle
submission = pd.read_csv("data/sample_submission.csv")

# Test tahminleri
submission["Label"] = test_predictions

# submission.csv olarak kaydet
submission.to_csv("submission.csv", index=False)

print("Tahminler submission.csv dosyasına kaydedildi!")



# Oluşturulan submission.csv'yi yükle ve kontrol et
submission_check = pd.read_csv("submission.csv")
print(submission_check.head())


from sklearn.metrics import classification_report, confusion_matrix

# Validation setinde tahmin yap
y_val_pred = clf.predict(X_val)

# Sınıflandırma raporunu görüntüle
print("Sınıflandırma Raporu:")
print(classification_report(y_val, y_val_pred))

# Karmaşıklık matrisi (confusion matrix)
print("Karmaşıklık Matrisi:")
print(confusion_matrix(y_val, y_val_pred))



