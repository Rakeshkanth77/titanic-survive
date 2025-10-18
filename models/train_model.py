import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib



df = pd.read_csv('../data/train.csv')
print("Data loaded!, shape:", df.shape)
print(df.info())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[features + ['Survived']].dropna()

print("Data after dropping NA, shape:", df.shape)
print(df.info())

X = df[features]
y = df['Survived']

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

le_sex = LabelEncoder()
X_train['Sex'] = le_sex.fit_transform(X_train['Sex'])
X_test['Sex'] = le_sex.transform(X_test['Sex'])
le_embarked = LabelEncoder()
X_train['Embarked'] = le_embarked.fit_transform(X_train['Embarked'])
X_test['Embarked'] = le_embarked.transform(X_test['Embarked'])
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# save the model
joblib.dump(best_model, '../models/decision_tree_model.pkl')
joblib.dump({'sex': le_sex, 'embarked': le_embarked}, '../models/label_encoders.pkl')
print("Model and encoders saved!")

prediction_data = pd.DataFrame(
    [[3, 1, 22.0, 1, 0, 7.25, 2]],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
)

print(best_model.predict(prediction_data))




# Load the saved model and encoders
best_model = joblib.load('../models/decision_tree_model.pkl')
encoders = joblib.load('../models/label_encoders.pkl')

le_sex = encoders['sex']
le_embarked = encoders['embarked']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Load test data
test_df = pd.read_csv('../data/test.csv')

# Apply same preprocessing using loaded encoders
test_df['Sex'] = le_sex.transform(test_df['Sex'])
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Embarked'] = le_embarked.transform(test_df['Embarked'])

# Make predictions
predictions = best_model.predict(test_df[features])

# Prepare submission file
results = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
results.to_csv('submission.csv', index=False)
print("Submission saved!")