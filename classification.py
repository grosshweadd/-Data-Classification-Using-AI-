from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Input features and target labels
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree model
classifier = DecisionTreeClassifier(random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Display project information
print("=" * 50)
print("      DATA CLASSIFICATION USING AI")
print("=" * 50)

print(f"\nDataset Size: {len(X)} samples")
print(f"Training Data: {len(X_train)} samples")
print(f"Testing Data: {len(X_test)} samples")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Display sample predictions
print("\nSample Predictions")
print("-" * 50)

for i in range(5):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[predictions[i]]

    print(
        f"Sample {i+1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )

print("\nProject Completed Successfully!")
