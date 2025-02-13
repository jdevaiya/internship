import pandas as pd
import numpy as np

# Load the data
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Display the first few rows of the data
print("Original Data:")
print(data.head())

# 1. Data Cleaning
# Remove duplicate rows
data = data.drop_duplicates()

# Handle missing values: Fill missing numeric values with the mean
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Fill missing categorical values with a placeholder
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

# 2. Data Transformation
# Add a new column: Calculate total price (e.g., for travel bookings)
if 'price_per_unit' in data.columns and 'quantity' in data.columns:
    data['total_price'] = data['price_per_unit'] * data['quantity']

# Normalize a numeric column (e.g., total_price)
if 'total_price' in data.columns:
    data['normalized_price'] = (data['total_price'] - data['total_price'].min()) / \
                                (data['total_price'].max() - data['total_price'].min())

# 3. Basic Analysis
# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Group by a category and calculate average total price
if 'category' in data.columns and 'total_price' in data.columns:
    avg_price_by_category = data.groupby('category')['total_price'].mean()
    print("\nAverage Total Price by Category:")
    print(avg_price_by_category)

# Save the cleaned and transformed data to a new file
output_file = 'cleaned_data.csv'
data.to_csv(output_file, index=False)
print(f"\nCleaned data saved to {output_file}.")
