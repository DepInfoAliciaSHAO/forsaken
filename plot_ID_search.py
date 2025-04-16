import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os
import re

# Load your CSV file
df = pd.read_csv("filtered_results.csv")  # Replace with your actual CSV file path

# Function to extract tensor values if necessary
def extract_value(val):
    try:
        # If the value is a string and looks like 'tensor(value)', extract the number
        if isinstance(val, str):
            match = re.match(r'tensor\((.*)\)', val)
            if match:
                return float(match.group(1))  # Extract the number inside the parentheses and convert to float
        return val
    except Exception as e:
        print(f"Error extracting value: {e}")
        return val

# Apply the function to 'verification_err' column
df['verification_err'] = df['verification_err'].apply(extract_value)

# Set Seaborn style for better readability
sns.set(style="whitegrid")

# Create a directory to save plots if it doesn't exist
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Forget Accuracy vs T for each xi, varying lambda_ and eta_mu
xi_vals = df['xi'].unique()  # Get all unique xi values

for xi in xi_vals:
    df_filtered = df[df['xi'] == xi]
    
    # Plot Forget Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="T", y="forget_acc", hue="lambda_", style="eta_mu", markers=True, dashes=False, palette="tab10")
    plt.title(f"Forget Accuracy vs T for xi = {xi}", fontsize=14)
    plt.xlabel("T", fontsize=12)
    plt.ylabel("Forget Accuracy", fontsize=12)
    plt.legend(title="Lambda & Eta_mu", loc="upper left", fontsize=10, bbox_to_anchor=(1, 1))  # Moved legend outside
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forget_accuracy_vs_T_xi_{xi}.png")  # Save the plot
    plt.close()  # Close the plot to free memory
    
    # Plot Verification Error with log scale on y-axis
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="T", y="verification_err", hue="lambda_", style="eta_mu", markers=True, dashes=False, palette="tab10")
    plt.title(f"Verification Error vs T for xi = {xi}", fontsize=14)
    plt.xlabel("T", fontsize=12)
    plt.ylabel("Verification Error", fontsize=12)
    plt.yscale("log")  # Set y-axis to log scale
    plt.legend(title="Lambda & Eta_mu", loc="upper left", fontsize=10, bbox_to_anchor=(1, 1))  # Moved legend outside
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification_error_vs_T_xi_{xi}_log_y.png")  # Save the plot with log scale y
    plt.close()  # Close the plot to free memory

# 2. Forget Accuracy vs lambda_ for each xi, varying T and eta_mu (log scale for x-axis)
for xi in xi_vals:
    df_filtered = df[df['xi'] == xi]
    
    # Plot Forget Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="lambda_", y="forget_acc", hue="T", style="eta_mu", markers=True, dashes=False, palette="tab10")
    plt.title(f"Forget Accuracy vs Lambda for xi = {xi}", fontsize=14)
    plt.xlabel("Lambda", fontsize=12)
    plt.ylabel("Forget Accuracy", fontsize=12)
    plt.xscale("log")  # Set x-axis to log scale
    plt.legend(title="T & Eta_mu", loc="upper left", fontsize=10, bbox_to_anchor=(1, 1))  # Moved legend outside
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forget_accuracy_vs_lambda_xi_{xi}_log_x.png")  # Save the plot with log scale x
    plt.close()  # Close the plot to free memory
    
    # Plot Verification Error with log scale on y-axis
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="lambda_", y="verification_err", hue="T", style="eta_mu", markers=True, dashes=False, palette="tab10")
    plt.title(f"Verification Error vs Lambda for xi = {xi}", fontsize=14)
    plt.xlabel("Lambda", fontsize=12)
    plt.ylabel("Verification Error", fontsize=12)
    plt.xscale("log")  # Set x-axis to log scale
    plt.yscale("log")  # Set y-axis to log scale
    plt.legend(title="T & Eta_mu", loc="upper left", fontsize=10, bbox_to_anchor=(1, 1))  # Moved legend outside
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification_error_vs_lambda_xi_{xi}_log_x_log_y.png")  # Save the plot with log scale x and y
    plt.close()  # Close the plot to free memory

# 3. Forget Accuracy vs eta_mu for each xi, varying lambda_ and T (log scale for x-axis)
for xi in xi_vals:
    df_filtered = df[df['xi'] == xi]
    
    # Plot Forget Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="eta_mu", y="forget_acc", hue="lambda_", style="T", markers=True, dashes=False, palette="tab10")
    plt.title(f"Forget Accuracy vs Eta_mu for xi = {xi}", fontsize=14)
    plt.xlabel("Eta_mu", fontsize=12)
    plt.ylabel("Forget Accuracy", fontsize=12)
    plt.xscale("log")  # Set x-axis to log scale
    plt.legend(title="Lambda & T", loc="upper left", fontsize=10, bbox_to_anchor=(1, 1))  # Moved legend outside
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forget_accuracy_vs_eta_mu_xi_{xi}_log_x.png")  # Save the plot with log scale x
    plt.close()  # Close the plot to free memory
    
    # Plot Verification Error with log scale on y-axis
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x="eta_mu", y="verification_err", hue="lambda_", style="T", markers=True, dashes=False, palette="tab10")
    plt.title(f"Verification Error vs Eta_mu for xi = {xi}", fontsize=14)
    plt.xlabel("Eta_mu", fontsize=12)
    plt.ylabel("Verification Error", fontsize=12)
    plt.xscale("log")  # Set x-axis to log scale
    plt.yscale("log")  # Set y-axis to log scale
    plt.legend(title="Lambda & T", loc="upper left", fontsize=10, bbox_to_anchor=(1, 1))  # Moved legend outside
    plt.tight_layout()
    plt.savefig(f"{output_dir}/verification_error_vs_eta_mu_xi_{xi}_log_x_log_y.png")  # Save the plot with log scale x and y
    plt.close()  # Close the plot to free memory
