import pandas as pd

# Load the CSV (you can also load from a file with pd.read_csv("your_file.csv"))
data = pd.read_csv("unlearning_search_results_3.csv")

# Filter where p == 0.01
filtered_data = data[data['p'] == 0.01]

# Print or save
print(filtered_data)
# Optionally, save to a new CSV
filtered_data = filtered_data.drop(columns=['p'])
filtered_data.to_csv("filtered_results.csv", index=False)
