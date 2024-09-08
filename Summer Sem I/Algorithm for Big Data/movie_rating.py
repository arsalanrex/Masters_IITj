import pandas as pd

roll_number = 'M23AID062'

# Load data from CSV into a DataFrame
df = pd.read_csv('/Users/arsalan/Downloads/Frac 3/Algorithms for Big Data/movies_datasets/M23AID059_data.csv', sep=',')

# Compute average rating for each movieId
avg_ratings = df.groupby('movieId')['rating'].mean().reset_index()

# Round avg rating to 2 decimal places
avg_ratings['rating'] = avg_ratings['rating'].round(2)

# Rename columns for output format
avg_ratings.columns = ['movieId', 'avg_rating']

# Write results to CSV
avg_ratings.to_csv('M23AID059_output.csv', index=False)

print("Average ratings saved to M23AID0xx_output.csv")