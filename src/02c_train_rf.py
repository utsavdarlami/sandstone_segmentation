import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

print("Preparing to Train the Model")
BASE_DIR = os.getcwd()
dataset_path = BASE_DIR + "/data/processed/final_dataset.csv"
# dataset_path = "./data/processed/final_dataset.csv"
final_dataframe = pd.read_csv(dataset_path)

# Splitting final_dataframe into features and target
X = final_dataframe.drop('Mask_label', axis=1)
y = final_dataframe['Mask_label']

# Dataset Splitting
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=101)

print("Our Random Forest Model")
rf_model = RandomForestClassifier(max_depth=20, random_state=42)

print("Training the model")
rf_model.fit(X_train, y_train)

print(rf_model)

print('Evaluating the model')
predictions = rf_model.predict(X_test)

print('Classification Report')
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


# Confusion Matrix
cf_matrix = confusion_matrix(y_test, predictions)
categories = ["29", "76", "150", "226"]
print('Confusion Matrix')
print(cf_matrix)

# normalized Confusion Matrix
n_cf_matrix = cf_matrix / cf_matrix.astype(np.float64).sum(axis=1)
print('Normalized Confusion Matrix')
print(np.round(n_cf_matrix, 3))

# Visualizing normalized confusion matrix
ncfplot1 = sns.heatmap(np.round(n_cf_matrix, 3),
                       annot=True,
                       xticklabels=categories,
                       yticklabels=categories)
ncfplot1.figure.savefig('results/rf_results/normalized_cf_matrix.png')
plt.show()

# Percentage Confusion Matrix
pcfplot2 = sns.heatmap(cf_matrix/np.sum(cf_matrix),
                       annot=True,
                       fmt='.2%',
                       cmap='Blues',
                       xticklabels=categories,
                       yticklabels=categories)
pcfplot2.figure.savefig('results/rf_results/percentage_cf_matrix.png')
plt.show()

# Pickling the model
# Saving model to disk
pickle.dump(rf_model, open('./models/rf_dummy.pkl', 'wb'))
