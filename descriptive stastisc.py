# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv('C:/Users/samsi/Documents/pdal/output1.csv')
df = pd.DataFrame(data)
print("Dataset:")
print(df)

pd.set_option('display.max_columns', None)  # Showing all the columns
pd.set_option('display.width', None)       # Expanding the width
pd.set_option('display.max_colwidth', None)  ## Prevent truncation of column values

print("\nDescriptive Statistics (Pandas):")
print(df.describe(include ='all'))

# Mean
print("\nMean of Intensity:", df["Intensity"].mean())
print("Mean of Y:", df["Y"].mean())

# Median
print("\nMedian of Intensity :", df["Intensity"].median())
print("Median of Y:", df["Y"].median())

# Mode
print("\nMode of Intensity:", stats.mode(df["Intensity"], keepdims=True).mode[0])
print("Mode of Y:", stats.mode(df["Y"], keepdims=True).mode[0])

import pandas as pd
import matplotlib.pyplot as plt

#variance
variance = df.var()
plt.figure(figsize=(10, 6))
plt.bar(variance.index, variance.values, color='skyblue')
plt.title("Variance of Each Column", fontsize=16)
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Variance", fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()



# Standard Deviation
print("\nStandard Deviation of Intensity:", df["Intensity"].std())
print("Standard Deviation of Y:", df["Y"].std())

# Skewness
print("\nSkewness of Classification:", stats.skew(df["Classification"]))
print("Skewness of Y:", stats.skew(df["Y"]))

# Kurtosis
print("\nKurtosis of Intensity:", stats.kurtosis(df["Intensity"]))
print("Kurtosis of Y:", stats.kurtosis(df["Y"]))


