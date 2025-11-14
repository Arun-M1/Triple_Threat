import pandas as pd

try:
    df = pd.read_csv('test_raw_data.csv')
    rows, cols = df.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")
except FileNotFoundError:
    print("Error: The specified CSV file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")