from sklearn.model_selection import train_test_split

# Assuming 'datarows' is a list of data rows
datarows = [
  {'text':'a cat'},
  {'text':'a boy'},
  {'text':'a girl'},
  {'text':'a man'},
  {'text':'a woman'},
  {'text':'a robot'},
  {'text':'a dog'},
]  # Replace with the actual list of data rows

# Define the ratio for splitting (e.g., 80% for training and 20% for validation)
train_ratio = 0.8
validation_ratio = 0.2

# Split the data rows into training and validation sets
training_datarows, validation_datarows = train_test_split(datarows, train_size=train_ratio, test_size=validation_ratio)

# Print the sizes of the training and validation sets
print(f"Training datarows count: {len(training_datarows)}")
print(training_datarows)
print(f"Validation datarows count: {len(validation_datarows)}")
print(validation_datarows)