from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
nn.fit(X_train_scaled, y_train)

print("Training Accuracy:", accuracy_score(y_train, nn.predict(X_train_scaled)))
print("Test Accuracy:", accuracy_score(y_test, nn.predict(X_test_scaled)))

# Feature scaling is necessary because neural networks are sensitive
# to input magnitude and may not converge otherwise.

# An epoch is one full pass through the training dataset.