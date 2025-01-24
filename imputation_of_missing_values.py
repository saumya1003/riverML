import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/samsi/Dropbox/PC/Downloads/sample_radiosonde.csv")

missing_percentage = 0.05  # 5%
num_missing = int(len(data) * missing_percentage)
missing_indices = np.random.choice(data.index, num_missing, replace=False)
data.loc[missing_indices, 'Temp'] = np.nan

data.to_csv('C:/Users/samsi/Dropbox/PC/Downloads/missing_values12.csv', index=False)

# Prepare dataset
def preprocess_data(data, target_column='Temp'):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Fill missing values in features with mean
    X.fillna(X.mean(), inplace=True)

    # Split into training and testing sets
    y_missing = y.isna()
    X_train, X_test, y_train, y_test, missing_train, missing_test = train_test_split(
        X, y, y_missing, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, missing_train, missing_test

X_train, X_test, y_train, y_test, missing_train, missing_test = preprocess_data(data)

# CGAN components
latent_dim = 100

# Generator model
def build_generator(latent_dim, input_dim):
    noise = Input(shape=(latent_dim,))
    missing_mask = Input(shape=(1,))
    inputs = Concatenate()([noise, missing_mask])
    
    x = Dense(128)(inputs)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    output = Dense(input_dim, activation='linear')(x)
    
    return Model([noise, missing_mask], output, name="Generator")

# Discriminator model
def build_discriminator(input_dim):
    real_input = Input(shape=(input_dim,))
    missing_mask = Input(shape=(1,))
    inputs = Concatenate()([real_input, missing_mask])
    
    x = Dense(256)(inputs)
    x = LeakyReLU(0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model([real_input, missing_mask], output, name="Discriminator")

# Build CGAN
generator = build_generator(latent_dim, X_train.shape[1])
discriminator = build_discriminator(X_train.shape[1])
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                      loss='binary_crossentropy', metrics=['accuracy'])

# Combined model
noise = Input(shape=(latent_dim,))
missing_mask = Input(shape=(1,))
generated_data = generator([noise, missing_mask])
discriminator.trainable = False
validity = discriminator([generated_data, missing_mask])
combined_model = Model([noise, missing_mask], validity)
combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
                       loss='binary_crossentropy')

# Training the CGAN
epochs = 5000
batch_size = 32

real_label = np.ones((batch_size, 1))
fake_label = np.zeros((batch_size, 1))

'''import time

start_time = time.time()
target_duration = 10 * 60  # 10 minutes (in seconds)

while (time.time() - start_time) < target_duration:'''
for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_data = X_train.iloc[idx]
    missing_mask_batch = missing_train.iloc[idx].values.reshape(-1, 1)
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_data = generator.predict([noise, missing_mask_batch])
    
    d_loss_real = discriminator.train_on_batch([real_data, missing_mask_batch], real_label)
    d_loss_fake = discriminator.train_on_batch([fake_data, missing_mask_batch], fake_label)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train Generator
    g_loss = combined_model.train_on_batch([noise, missing_mask_batch], real_label)
    
    # Print progress
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

# Impute missing values
def impute_missing(generator, data, missing_mask, latent_dim):
    noise = np.random.normal(0, 1, (data.shape[0], latent_dim))
    imputed_values = generator.predict([noise, missing_mask])
    return imputed_values

#missing_mask_resized = np.resize(missing_mask, (X_train.shape[0], 1))
missing_mask = data['Temp'].isna().astype(int).values.reshape(-1, 1)

# Align `missing_mask` with `X_train` indices
'''missing_mask = missing_mask[data.index.isin(X_train.index)].reshape(-1, 1)
missing_mask = data['Temp'].isna().astype(int).values.reshape(-1, 1)
imputed_values = impute_missing(generator, X_train, missing_mask, latent_dim)
data.loc[data['Temp'].isna(), 'Temp'] = imputed_values;'''

'''print(X_train.shape)  
print(y_train.shape)
print(missing_mask.shape) 
'''

# Step 1: Count missing values in 'Temp'
missing_count = data['Temp'].isna().sum()
print(f"Number of missing values in 'Temp': {missing_count}")

# Step 2: Check the length of imputed_values
imputed_length = len(imputed_values)
print(f"Length of imputed_values: {imputed_length}")

# Step 3: Match lengths
if missing_count != imputed_length:
    print(f"Mismatch detected! Adjusting imputed_values...")
    if imputed_length > missing_count:
        # Truncate to match
        imputed_values = imputed_values[:missing_count]
    else:
        # Regenerate or pad imputed_values
        raise ValueError(f"Insufficient imputed values. Expected {missing_count}, got {imputed_length}.")

# Step 4: Assign imputed values to the missing rows
missing_indices = data['Temp'][data['Temp'].isna()].index
data.loc[missing_indices, 'Temp'] = imputed_values

# Save imputed dataset
data.to_csv('C:/Users/samsi/Dropbox/PC/Downloads/imputed_values25.csv', index=False)

#the original datasetand imputed value dataset
original_data = pd.read_csv("C:/Users/samsi/Dropbox/PC/Downloads/sample_radiosonde.csv")
imputed_data = pd.read_csv("C:/Users/samsi/Dropbox/PC/Downloads/imputed_values25.csv")

#Verify input shapes and columns
print("Original data shape:", original_data.shape)
print("Imputed data shape:", imputed_data.shape)

target_column ='Temp'
if target_column not in original_data.columns:
    raise ValueError(f"Target column '{target_column}' not found in original_data.")
if target_column not in imputed_data.columns:
    raise ValueError(f"Target column '{target_column}' not found in imputed_data.")

#Align data indices
aligned_original_data = original_data.loc[original_data.index.isin(imputed_data.index)]
aligned_imputed_data = imputed_data.loc[imputed_data.index.isin(original_data.index)]

original_values = aligned_original_data[target_column].dropna().values
imputed_values = aligned_imputed_data[target_column].dropna().values

if original_values.size == 0:
    raise ValueError("No valid values in aligned original_data for the target column.")
if imputed_values.size == 0:
    raise ValueError("No valid values in aligned imputed_data for the target column.")

if len(original_values) != len(imputed_values):
    raise ValueError("Mismatch between lengths of original_values and imputed_values.")

mse = np.mean((original_values - imputed_values) ** 2)
mae = np.mean(np.abs(original_values - imputed_values))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

