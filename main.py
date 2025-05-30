import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load and filter training dataset for selected classes
train_data = pd.read_csv('sign_mnist_13bal_train.csv')
selected_train_labels = [2, 4, 7]
filtered_train_data = train_data[train_data['class'].isin(selected_train_labels)]

# Load and filter testing dataset for selected classes
test_data = pd.read_csv('sign_mnist_13bal_test.csv')
selected_test_labels = [3, 7, 5]
filtered_test_data = test_data[test_data['class'].isin(selected_test_labels)]

# Normalize and split features and labels (training)
X_train = filtered_train_data.drop('class', axis=1) / 255.0
y_train = filtered_train_data['class']

# Normalize and split features and labels (validation/testing)
X_validate = filtered_test_data.drop('class', axis=1) / 255.0
y_validate = filtered_test_data['class']

# Optional: create a validation subset from training data if needed
# X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=30, random_state=0)

# Initialize and train the neural network
neural_net_model = MLPClassifier(hidden_layer_sizes=(20,), random_state=42, tol=0.005)
neural_net_model.fit(X_train, y_train)

# Print model architecture
layer_sizes = [neural_net_model.coefs_[0].shape[0]] + [coef.shape[1] for coef in neural_net_model.coefs_]
print(f"Training set size: {len(y_train)}")
print(f"Layer sizes: {' x '.join(map(str, layer_sizes))}")

# Predict on training and validation sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_validate)

# Evaluate performance on validation set
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

for true, pred in zip(y_validate, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# Evaluate training accuracy for overfitting check
correct_counts_training = sum(1 for true, pred in zip(y_train, y_pred_train) if true == pred)
total_counts_training = len(y_train)

# Print per-class and overall validation accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:.0f}%")
print("----------")
overall_accuracy = overall_correct / len(y_validate) * 100
training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Validation Accuracy: {overall_accuracy:.1f}%")
print(f"Overall Training Accuracy: {training_accuracy:.1f}%")

# Print confusion matrix
class_ids = sorted(set(y_validate) | set(y_pred))
conf_matrix = confusion_matrix(y_validate, y_pred, labels=class_ids)

print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()
for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))

#manually note misclassified classes after reviewing confusion matrix
print("Note: Look at the matrix to see which classes were most frequently misclassified.")
