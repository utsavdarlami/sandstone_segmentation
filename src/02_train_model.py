import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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


# Plotting number of mask value present in out dataset
sns.set_style("whitegrid")
ax = sns.countplot(x=final_dataframe['Mask_label'], linewidth=1)
plt.title('Number of mask value present in our dataset')
plt.xlabel('Mask_value')
plt.ylabel('Count')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(final_dataframe)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    i = p.get_x() + p.get_width() / 2
    j = p.get_height()
    ax.annotate(percentage, (i, j), ha='center', va='center')
plt.tight_layout()
plt.savefig('results/mask_count_histplot.png')


# Splitting final_dataframe into features and target
X = final_dataframe.drop('Mask_label', axis=1)
y = final_dataframe['Mask_label']

# Dataset Splitting
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=101)

dtree = DecisionTreeClassifier()
print("Our Decision Tree Model")
print(dtree)

print("Training the model")
dtree.fit(X_train, y_train)

print('Evaluating the model')
predictions = dtree.predict(X_test)

print('Classification Report')
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")


# Confusion Matrix
cf_matrix = confusion_matrix(y_test, predictions)
categories = ["29", "76", "150", "179"]
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
ncfplot1.figure.savefig('results/normalized_cf_matrix.png')

# Percentage Confusion Matrix
pcfplot2 = sns.heatmap(cf_matrix/np.sum(cf_matrix),
                       annot=True,
                       fmt='.2%',
                       cmap='Blues',
                       xticklabels=categories,
                       yticklabels=categories)
pcfplot2.figure.savefig('results/percentage_cf_matrix.png')

# Pickling the model
# Saving model to disk
pickle.dump(dtree, open('./models/dtree.pkl', 'wb'))
