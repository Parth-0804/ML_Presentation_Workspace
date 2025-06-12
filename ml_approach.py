import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
# Place this script in the same folder as your two CSV files,
# or update the file paths below.
try:
    arthur_df = pd.read_csv('ARTHUR_Queue_Times_By_Date_Time.csv')
    holidays_df = pd.read_csv('download (1).csv') # This is your Regional_Holidays file
    print("Files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure the CSV files are in the same folder as the script.")
    exit()

# --- 1. Data Merging and Initial Cleaning ---
print("\n--- 1. Starting Data Merging and Initial Cleaning ---")
arthur_df['FullDateTime'] = pd.to_datetime(arthur_df['FullDateTime'])
holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
arthur_df['Date'] = arthur_df['FullDateTime'].dt.date
arthur_df['Date'] = pd.to_datetime(arthur_df['Date'])
df = pd.merge(arthur_df, holidays_df, on='Date', how='left')
holiday_cols = [col for col in df.columns if 'Is_' in col]
for col in holiday_cols:
    df[col] = df[col].fillna(False)
print("Data Merging and Initial Cleaning Complete.")

# --- 2. Data Quality Check ---
print("\n--- 2. Starting Data Quality Check ---")
df['ARTHUR_WaitTime'] = df['ARTHUR_WaitTime'].apply(lambda x: 0 if x < 0 else x)
df['temperature_in_celsius'] = df['temperature_in_celsius'].fillna(method='ffill')
df['precipitation_in_percent'] = df['precipitation_in_percent'].fillna(method='ffill')
df['wind_speed_in_kmh'] = df['wind_speed_in_kmh'].fillna(method='ffill')
print("Data Quality Check Complete.")

# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- 3. Starting Exploratory Data Analysis (EDA) ---")

# Distribution of Wait Times
plt.figure(figsize=(12, 6))
sns.histplot(df['ARTHUR_WaitTime'], bins=50, kde=True)
plt.title('Distribution of ARTHUR Wait Times')
plt.xlabel('Wait Time (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('arthur_wait_time_distribution.png')
plt.show()

# Time Series Plot
plt.figure(figsize=(15, 7))
df.set_index('FullDateTime')['ARTHUR_WaitTime'].plot()
plt.title('ARTHUR Wait Times Over Time')
plt.ylabel('Wait Time (minutes)')
plt.xlabel('Date and Time')
plt.grid(True)
plt.savefig('arthur_wait_time_series.png')
plt.show()

# Boxplot for Holidays
plt.figure(figsize=(10, 6))
sns.boxplot(x='Is_Public_Holiday_DE_BW', y='ARTHUR_WaitTime', data=df)
plt.title('Wait Time on Public Holidays (Baden-Württemberg) vs. Non-Holidays')
plt.xlabel('Is Public Holiday in Baden-Württemberg')
plt.ylabel('Wait Time (minutes)')
plt.grid(True)
plt.savefig('wait_time_vs_holiday.png')
plt.show()

# Scatterplot for Weather
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temperature_in_celsius', y='ARTHUR_WaitTime', data=df, alpha=0.5)
plt.title('Wait Time vs. Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Wait Time (minutes)')
plt.grid(True)
plt.savefig('wait_time_vs_temperature.png')
plt.show()
print("Exploratory Data Analysis Complete. Plots have been saved as PNG files.")

# --- 4. Feature Engineering ---
print("\n--- 4. Starting Feature Engineering ---")
df['Hour'] = df['FullDateTime'].dt.hour
df['DayOfWeek'] = df['FullDateTime'].dt.dayofweek
df['DayOfYear'] = df['FullDateTime'].dt.dayofyear
df['WeekOfYear'] = df['FullDateTime'].dt.isocalendar().week
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6])
df['Month'] = df['FullDateTime'].dt.month
df['Is_Any_German_Holiday'] = df[[col for col in df.columns if 'DE' in col and 'Is_' in col]].any(axis=1)
df['Is_Any_French_Holiday'] = df[[col for col in df.columns if 'FR' in col and 'Is_' in col]].any(axis=1)
df['Is_Any_Swiss_Holiday'] = df[[col for col in df.columns if 'CH' in col and 'Is_' in col]].any(axis=1)
df['Is_Any_Regional_Holiday_Impacting_EP'] = df['Is_Any_German_Holiday'] | df['Is_Any_French_Holiday'] | df['Is_Any_Swiss_Holiday']
df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
df['month_sin'] = np.sin(2 * np.pi * df['Month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['Month']/12)
df['Is_Raining'] = df['precipitation_in_percent'] > 0
df['Is_Hot_Weather'] = df['temperature_in_celsius'] > 25
df = df.sort_values('FullDateTime')
df['ARTHUR_WaitTime_Lag_15min'] = df['ARTHUR_WaitTime'].shift(1)
df['ARTHUR_WaitTime_Lag_60min'] = df['ARTHUR_WaitTime'].shift(4)
df['ARTHUR_WaitTime_MA_30min'] = df['ARTHUR_WaitTime'].rolling(window=2).mean()
df['ARTHUR_WaitTime_MA_60min'] = df['ARTHUR_WaitTime'].rolling(window=4).mean()
print("Feature Engineering Complete.")

# --- 5. Final DataFrame ---
print("\n--- 5. Final DataFrame with Engineered Features ---")
print("Displaying the first 5 rows of the processed data:")
print(df.head())

# Save the new dataframe to a csv file
df.to_csv('ARTHUR_Queue_Times_With_Holidays.csv', index=False)
print("\n'ARTHUR_Queue_Times_With_Holidays.csv' created successfully in your local folder.")