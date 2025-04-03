from xgboost import XGBClassifier
from src.components.model_training_and_evaluation import ModelTraining

def model_training_main():
    model_trainer = ModelTraining(
        train_data_path="src/dataset/preprocessed_train_data.csv",
        test_data_path="src/dataset/preprocessed_test_data.csv"
    )

    X_train, X_test, y_train, y_test = model_trainer.train_test_target_feature_split()

    # Modeli oluştur
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, objective='binary:logistic')

    # Modeli eğit
    trained_model = model_trainer.model_train(model, X_train, y_train)

    # Modeli, parametreleri ve skorları kaydet
    model_trainer.dump_model_to_pkl(trained_model, "xgb_clf", "src/models", X_test, y_test)

    # Modeli yükle ve bilgileri yazdır
    print("Yüklenen Model Accuracy models/ Klasörünün json dosyasına kaydedildi.")

