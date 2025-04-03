import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataPreprocessing:
    def __init__(self, raw_data_path: Path, target_column: str):
        self.data_path = raw_data_path
        self.target_column = target_column
        self.raw_data = pd.read_csv(raw_data_path)

    def split_data_test_train(self):
        train_df, test_df = train_test_split(self.raw_data, test_size=0.2, random_state=42)
        
        # Hedef değişkeni ayır
        y_train = train_df[self.target_column]
        y_test = test_df[self.target_column]
        
        # Özelliklerden hedef değişkeni kaldır
        X_train = train_df.drop(columns=[self.target_column])
        X_test = test_df.drop(columns=[self.target_column])
        
        return X_train, X_test, y_train, y_test
    
    def numerical_preprocessing(self, data_will_process):
        numerical_cols = data_will_process.select_dtypes(include=['int64', 'float64']).columns
        std_scaler = StandardScaler()
        scaled_data = std_scaler.fit_transform(data_will_process[numerical_cols])
        numerical_preprocessed_data = pd.DataFrame(data=scaled_data, columns=numerical_cols)
        
        # Save the scaler to disk
        with open('src/models/scaler.pkl', 'wb') as f:
            pickle.dump(std_scaler, f)
        
        return numerical_preprocessed_data
        
    def categorical_preprocessing(self, data_will_process):
        categorical_cols = [col for col in data_will_process.columns if data_will_process[col].dtype == "object"]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
        encoded_categorical_data = encoder.fit_transform(data_will_process[categorical_cols])
        
        # Save the encoder to disk
        with open('src/models/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        
        return encoded_categorical_data
    
    def concat_cat_and_numeric_data(self, numerical_preprocessed_data, encoded_categorical_data):
        preprocessed_data = pd.concat([numerical_preprocessed_data.reset_index(drop=True), 
                                       encoded_categorical_data.reset_index(drop=True)], axis=1)
        return preprocessed_data
    
    def export_data(self, preprocessed_data, y_data, filename):
        output_path = Path(f"src/dataset/{filename}")
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Pathi oluştur,
        full_data = pd.concat([preprocessed_data, y_data.reset_index(drop=True)], axis=1)
        full_data.to_csv(output_path, index=False)

    def preprocess_target(self, y_train, y_test, classification=True):
        if classification:
            # Subtract 1 from the target labels to shift them to the range [0, 1, 2, ..., 9]
            y_train_adjusted = y_train - 1
            y_test_adjusted = y_test - 1
            return y_train_adjusted, y_test_adjusted
        else:
            return y_train, y_test  # For regression, but not used here

    
    def categorize_target(self, y):
        # Removed this method, as we now handle the target as 1-10 class labels directly
        return y
