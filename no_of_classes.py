import pandas as pd

file_path = "C:/Users/samsi/Documents/pdal/output1.csv"  
lidar_data = pd.read_csv(file_path)

class_labels = lidar_data['Classification']  

unique_classes = class_labels.unique()
total_classes = len(unique_classes)

print(f"Total number of classes: {total_classes}")
print(f"Classification: {unique_classes}")

