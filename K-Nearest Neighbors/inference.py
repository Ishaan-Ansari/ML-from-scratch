import pandas as pd
from model import KNN

# Load the dataset
data = pd.read_csv('./K-Nearest Neighbors/TShirt_size.csv')

# Split the data into features (X) and labels (y)
X = data[['Height (in cms)', 'Weight (in kgs)']].values
y = data['T Shirt Size'].values


# Train the KNN model
knn = KNN(k=3)
knn.fit(X, y)

# Make predictions on new data
new_data = pd.DataFrame({'Height (in cms)': [160, 170], 'Weight (in kgs)': [60, 70]})
new_data = new_data.values

predictions = knn.predict(new_data)

print(predictions)

