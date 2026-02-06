import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'diabetes_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Set the style of the visualization
sns.set(style="whitegrid")

# Create histograms for the numeric features
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Histograms of Numeric Features')

# Age
sns.histplot(data['age'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age')

# BMI
sns.histplot(data['bmi'], bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('BMI')

# HbA1c level
sns.histplot(data['HbA1c_level'], bins=30, kde=True, ax=axes[0, 2])
axes[0, 2].set_title('HbA1c Level')

# Blood glucose level
sns.histplot(data['blood_glucose_level'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Blood Glucose Level')

# Hypertension
sns.histplot(data['hypertension'], bins=2, kde=False, ax=axes[1, 1])
axes[1, 1].set_title('Hypertension')

# Heart disease
sns.histplot(data['heart_disease'], bins=2, kde=False, ax=axes[1, 2])
axes[1, 2].set_title('Heart Disease')

# Diabetes
sns.histplot(data['diabetes'], bins=2, kde=False, ax=axes[2, 0])
axes[2, 0].set_title('Diabetes')

# Gender (converted to numeric for plotting)
sns.histplot(data['gender'].apply(lambda x: 1 if x == 'Male' else 0), bins=2, kde=False, ax=axes[2, 1])
axes[2, 1].set_title('Gender (0: Female, 1: Male)')

# Smoking history (count plot)
sns.countplot(x='smoking_history', data=data, ax=axes[2, 2])
axes[2, 2].set_title('Smoking History')
plt.xticks(rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Encode categorical features
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

# Separate features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

# Dictionary to store the evaluation results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Store the results
    results[model_name] = {
        "classification_report": report,
        "confusion_matrix": matrix
    }

# Print the results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print("Classification Report:")
    print(pd.DataFrame(result["classification_report"]).transpose())
    print("Confusion Matrix:")
    print(result["confusion_matrix"])
    print("\n")
