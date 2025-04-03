import json
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

class ModelTraining:
    def __init__(self, train_data_path, test_data_path):
        # Test ve train dosyalarının varlığını kontrol et
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            raise FileNotFoundError("Test or train data path is wrong or does not exist.")

        self.train_df = pd.read_csv(train_data_path)
        self.test_df = pd.read_csv(test_data_path)

    def train_test_target_feature_split(self):
        X_train, y_train = self.train_df.drop(columns=["Target"]), self.train_df["Target"]
        X_test, y_test = self.test_df.drop(columns=["Target"]), self.test_df["Target"]
        return X_train, X_test, y_train, y_test

    def model_train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = confusion_matrix(y_test, preds)

        return {
            "accuracy": acc,
            "confusion_matrix": f1.tolist()  # JSON uyumlu olması için confusion matrix'i liste olarak kaydediyoruz
        }

    def dump_model_to_pkl(self, model, file_name, export_path, X_test, y_test):
        """
        Modeli, sadece modelin kendisini pickle dosyasına kaydeder.
        Parametreler ve değerlendirme metrikleri JSON dosyasına kaydedilir.
        """
        file_name = f"{file_name}.pkl"
        os.makedirs(export_path, exist_ok=True)  # Export path yoksa oluştur

        file_path = os.path.join(export_path, file_name)

        # Modeli pickle ile kaydet
        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model kaydedildi: {file_path}")

        # Değerlendirme metriklerini JSON dosyasına kaydet
        model_info = {
            "evaluation": self.evaluate_model(model, X_test, y_test)  # Değerlendirme metriklerini kaydet
        }
        json_file_path = os.path.join(export_path, f"{file_name.replace('.pkl', '.json')}")
        with open(json_file_path, "w") as json_file:
            json.dump(model_info, json_file, indent=4)

        print(f"Model bilgileri JSON dosyasına kaydedildi: {json_file_path}")

    def load_model_from_pkl(self, file_path):
        """
        Kaydedilmiş modeli yükler
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {file_path}")

        with open(file_path, "rb") as f:
            model = pickle.load(f)

        return model
