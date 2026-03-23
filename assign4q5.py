from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
nn.fit(X_train_scaled, y_train)

print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt.predict(X_test)))

print("\nNeural Network Confusion Matrix:")
print(confusion_matrix(y_test, nn.predict(X_test_scaled)))

# Decision Trees are more interpretable but can overfit.
# Neural Networks capture complex patterns but are harder to interpret.