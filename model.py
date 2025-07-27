import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def classify_wholesale_customers(file_path):
    """
    Classifies wholesale customers based on their annual spending on diverse product categories
    to predict their 'Channel' (Horeca or Retail) using a RandomForestClassifier.

    Args:
        file_path (str): The path to the CSV file containing the wholesale customer data.
    """
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("Initial data head:")
        print(df.head())
        print("\nData Info:")
        df.info()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return
    target_variable = 'Channel'
    spending_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    required_columns = [target_variable] + spending_features
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\nError: Missing required columns in the dataset: {missing_columns}")
        print("Please ensure the CSV contains all the expected columns.")
        return
    X = df[spending_features] 
    y = df[target_variable]  

    print(f"\nTarget variable selected: '{target_variable}'")
    print(f"Features selected for classification: {spending_features}")
    print("Sample of features (X) data:")
    print(X.head())
    print("\nSample of target (y) data:")
    print(y.head())
    print(f"\nTarget variable unique values and counts:\n{y.value_counts()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nData split into training (70%) and testing (30%) sets.")
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=spending_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=spending_features)
    print("\nFeatures scaled (StandardScaler applied). Sample of scaled training data:")
    print(X_train_scaled_df.head())
    print("\nInitializing and training RandomForestClassifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_train)
    print("Model training complete.")
    y_pred = classifier.predict(X_test_scaled)
    print("\nPredictions made on the test set.")
    print("\n--- Model Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=y.unique(), yticklabels=y.unique())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    print("\nConfusion Matrix plot displayed.")
    print("\nClassification process complete.")
    return df
file_path = "Wholesale customers data.csv"
classified_df = classify_wholesale_customers(file_path)

if classified_df is not None:
    print("\nFirst 10 rows of the original DataFrame (no new 'Cluster' column added for classification):")
    print(classified_df.head(10))
