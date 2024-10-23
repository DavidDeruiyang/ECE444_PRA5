import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the base URL of your Flask API
BASE_URL = 'http://ECE444pra5v3-env.eba-jgm6c28t.us-east-1.elasticbeanstalk.com/predict'

test_cases = [
    {"text": "This is fake news"},  # Fake news
    {"text": "Aliens are living among us!"},  # Fake news
    {"text": "The stock market is down today."},  # Real news
    {"text": "The weather is sunny and pleasant."}  # Real news
]

# Prepare to collect latency data
latency_data = []

# Function for testing predictions
def test_prediction(text):
    response = requests.post(BASE_URL, json={"text": text})
    if response.status_code == 200:
        return response.json()['prediction']
    else:
        return None

# 1. Functional/Unit Tests
print("Running Functional/Unit tests: ")
for case in test_cases:
    prediction = test_prediction(case['text'])
    print(f"Input: {case['text']} => Prediction: {prediction}")

# 2. Latency/Performance Testing
print("\nRunning Latency/Performance tests:")
for case in test_cases:
    text = case['text']
    for _ in range(100):  # 100 API calls
        start_time = time.time()
        prediction = test_prediction(text)
        latency = time.time() - start_time
        latency_data.append({"text": text, "latency": latency})

# Convert latency data to DataFrame and save to CSV
df_latency = pd.DataFrame(latency_data)
csv_file = 'latency_results.csv'
df_latency.to_csv(csv_file, index=False)
print(f"\nLatency results saved to {csv_file}")

# 3. Generate Boxplot
plt.figure(figsize=(10, 6))
df_latency.boxplot(column='latency', by='text')
plt.title('Latency Performance Boxplot')
plt.suptitle('')
plt.xlabel('Test Case')
plt.ylabel('Latency (seconds)')
plt.grid()
plt.savefig('latency_boxplot.png')
plt.show()

# Calculate average performance
average_latency = df_latency.groupby('text')['latency'].mean()
print("\nAverage Latency (seconds):")
print(average_latency)
