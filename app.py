Step 1: Environment Setup 

Install necessary Python libraries: 
!pip install pandas scikit-learn matplotlib seaborn

Step 2: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

Step 3: Data Collection

def generate_data(n_students=400):
np.random.seed(42)
df = pd.DataFrame({
'student_id': range(1, n_students + 1),
'name': [f"Student_{i+1}" for i in range(n_students)],
'gender': np.random.choice(['M', 'F'], size=n_students),
'age': np.random.randint(15, 22, size=n_students),
'study_time': np.random.randint(1, 5, size=n_students),
'G1': np.random.randint(0, 21, size=n_students),
'G2': np.random.randint(0, 21, size=n_students),
'attendance': np.random.uniform(75, 100, size=n_students).round(2),
'prev_grade': np.random.randint(0, 21, size=n_students)
})
noise = np.random.normal(0, 3, size=n_students)
df['G3'] = (0.3 * df['G1'] + 0.5 * df['G2'] + 0.1 * df['attendance'] + noise).round().clip(0, 20).astype(int)
return df
Step 4: Data Preprocessing

def preprocess_data(df, imputer_choice):
df_encoded = df.drop(columns=['student_id', 'name']).copy()
label_encoders = {}
for column in df_encoded.select_dtypes(include=['object']).columns:
le = LabelEncoder()
df_encoded[column] = le.fit_transform(df_encoded[column])
label_encoders[column] = le
X = df_encoded.drop(columns=['G3'])
y = df_encoded['G3']
# Introduce some missing values randomly
mask = np.random.rand(*X.shape) < 0.05
X_missing = X.mask(mask)
if imputer_choice == "mice":
imputer = IterativeImputer(random_state=0)
else:
imputer = KNNImputer(n_neighbors=5)

X_imputed = imputer.fit_transform(X_missing)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_train_bin = (y_train >= 10).astype(int)
y_test_bin = (y_test >= 10).astype(int)
return X_train, X_test, y_train_bin, y_test_bin, y_test, d

Step 5: Train-Model

def train_model(model_choice, X_train, y_train_bin):
if model_choice == "svm":
model = SVC()
else:
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train_bin)
return model

Step 6:Evaluate model

def evaluate_model(model, X_test, y_test_bin):
preds = model.predict(X_test)
acc = accuracy_score(y_test_bin, preds)
cm = confusion_matrix(y_test_bin, preds)
return preds, acc, cm

Step 7:Visualize results
def visualize_results(y_test, y_test_bin, preds, acc, cm, model_choice, df):
print("\n===== Sample Student Data =====")
print(df.head(10).to_string(index=False))
print("\n===== Top 5 Students by Final Grade =====")
print(df.sort_values(by='G3', ascending=False).head(5).to_string(index=False))

print("\n===== Students Who Passed (G3 >= 10) =====")
print(df[df['G3'] >= 10].to_string(index=False))
print("\n===== Students Who Failed (G3 < 10) =====")
print(df[df['G3'] < 10].to_string(index=False))
df['rank'] = df['G3'].rank(ascending=False, method='min').astype(int)
print("\n===== Student Rankings =====")
print(df[['student_id', 'name', 'G3', 'rank']].sort_values(by='rank').to_string(index=False))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(y_test, bins=20, kde=True)
plt.title("Distribution of Final Grades (G3)")

plt.subplot(1, 3, 2)
labels = ['Fail', 'Pass']
sizes = [np.sum(preds == 0), np.sum(preds == 1)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Predicted Pass/Fail Ratio")

plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix ({model_choice.upper()})")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
    plt.show()
  print(f"\nModel Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)

Step 8:Interative Menu
def main_menu():
imputer_choice = "knn"
model_choice = "knn"
df = generate_data()
while True:
print("\n========= Student Performance Prediction Menu =========")
print(f"1. Change Imputation Method (Current: {imputer_choice.upper()})")
print(f"2. Change Model Type (Current: {model_choice.upper()})")
print("3. Train Model, Evaluate & Show Results")
print("4. Exit")
choice = input("Enter your choice: ").strip()
if choice == '1':
method = input("Enter imputation method (knn/mice): ").strip().lower()
if method in ["knn", "mice"]:
imputer_choice = method
print(f" Imputation method changed to {imputer_choice.upper()}")
else:
print(" Invalid imputation method.")
elif choice == '2':
model = input("Enter model type (knn/svm): ").strip().lower()
if model in ["knn", "svm"]:
model_choice = model
print(f" Model type changed to {model_choice.upper()}")
else:
print(" Invalid model type.")

elif choice == '3':
print(f"\nTraining model with {imputer_choice.upper()} imputation and {model_choice.upper()} model...")
X_train, X_test, y_train_bin, y_test_bin, y_test, df_updated = preprocess_data(df, imputer_choice)

model = train_model(model_choice, X_train, y_train_bin)
preds, acc, cm = evaluate_model(model, X_test, y_test_bin)
visualize_results(y_test, y_test_bin, preds, acc, cm, model_choice, df_updated)
elif choice == '4':
print(" Exiting program. Goodbye!")
break
else:
print(" Invalid choice. Please try again.")

# Run the interactive menu
if __name__ == "__main__":
main_menu()










Step 9: Interactive Menu System

View student details
Compare model accurucy
Show top 5 students
Display Pass/failed students
Display ranking
Confusion matrix
