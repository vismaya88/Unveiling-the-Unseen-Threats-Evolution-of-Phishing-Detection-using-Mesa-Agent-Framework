import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from flask import Flask, render_template, request, jsonify
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Read data from Excel file
dataset = "F:\\3rd sem\\end sem projects\\foa project\\dataset\\dataset_phishing.csv"
df = pd.read_csv(dataset, encoding='latin1')

# Extract phishing and legitimate URLs
phishing_urls = df[df['label'] == 'phishing']['url'].tolist()
legitimate_urls = df[df['label'] == 'legitimate']['url'].tolist()

# Function to extract features from URLs
def extract_features(url):
    features = []
    features.append(len(url))
    features.append(1 if 'https' in url else 0)
    features.append(1 if '@' in url else 0)
    features.append(url.count('.'))
    features.append(1 if re.search('http://.*/', url) else 0)
    return features

# Agent class for URLs
class URLAgent(Agent):
    def __init__(self, unique_id, model, url, label):
        super().__init__(unique_id, model)
        self.url = url
        self.label = label
        self.features = extract_features(url)

    def step(self):
        
        pass

# Model class
class URLModel(Model):
    def __init__(self, phishing_urls, legitimate_urls):
        self.num_agents = len(phishing_urls) + len(legitimate_urls)
        self.schedule = RandomActivation(self)

        # Create agents for phishing URLs
        for idx, url in enumerate(phishing_urls):
            agent = URLAgent(idx, self, url, 1)  # 1 indicates phishing
            self.schedule.add(agent)

        # Create agents for legitimate URLs
        for idx, url in enumerate(legitimate_urls):
            agent = URLAgent(idx + len(phishing_urls), self, url, 0)  # 0 indicates legitimate
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            agent_reporters={"Label": "label", "Features": "features"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Create a Mesa model
model = URLModel(phishing_urls, legitimate_urls)

# Run the model for a certain number of steps (you can adjust this)
for i in range(30):
    model.step()

# Extract data for analysis
data = model.datacollector.get_agent_vars_dataframe().reset_index()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    np.vstack(data['Features']), data['Label'], test_size=0.2, random_state=42
)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Flask routes
foa_app = Flask(__name__)

@foa_app.route('/')
def index():
    return render_template('index.html')

@foa_app.route('/classify', methods=['POST'])
def classify():
    url = request.form['url']
    new_features = np.array(extract_features(url)).reshape(1, -1)
    prediction = classifier.predict(new_features)
    result = 'phishing' if prediction[0] == 1 else 'legitimate'
    return jsonify({'result': result})

if __name__ == '__main__':
    foa_app.run(debug=True, use_reloader=False)
