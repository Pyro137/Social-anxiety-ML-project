from src.components.data_preprocessing import DataPreprocessing
from pathlib import Path
import pandas as pd
def preprocessing_main():
    classification = True  # Eğer regresyon istiyorsan False yap
    
    data_preprocessing = DataPreprocessing(
        raw_data_path=Path("dataset/raw_df.csv"), 
        target_column="Anxiety Level (1-10)"
    )
    
    # Veriyi ayır
    X_train, X_test, y_train, y_test = data_preprocessing.split_data_test_train()
    
    # Özellikleri ön işle
    num_train = data_preprocessing.numerical_preprocessing(X_train)
    cat_train = data_preprocessing.categorical_preprocessing(X_train)
    train_preprocessed = data_preprocessing.concat_cat_and_numeric_data(num_train, cat_train)
    
    num_test = data_preprocessing.numerical_preprocessing(X_test)
    cat_test = data_preprocessing.categorical_preprocessing(X_test)
    test_preprocessed = data_preprocessing.concat_cat_and_numeric_data(num_test, cat_test)
    
    # Hedef değişkeni işleyelim
    y_train_processed, y_test_processed = data_preprocessing.preprocess_target(y_train, y_test, classification=classification)
    
    # Verileri kaydet (train ve test birlikte target ile)
    data_preprocessing.export_data(train_preprocessed, pd.Series(y_train_processed, name="Target"), "preprocessed_train_data.csv")
    data_preprocessing.export_data(test_preprocessed, pd.Series(y_test_processed, name="Target"), "preprocessed_test_data.csv")

