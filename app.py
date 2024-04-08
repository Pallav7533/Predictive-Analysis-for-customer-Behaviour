from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Correct import
import joblib

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load the trained model
model = joblib.load('rf_classifier.pkl')

# Load the dataset
data = pd.read_csv('Customer_Behaviour.csv').dropna()

# Feature Engineering
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Purchased', 'User ID'])
y = data['Purchased']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input data from the form
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        salary = int(request.form['salary'])

        # Make prediction using the loaded model
        prediction = model.predict([[gender, age, salary]])

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
